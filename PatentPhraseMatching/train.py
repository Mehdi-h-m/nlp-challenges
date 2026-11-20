"""DistilBERT regression for the U.S. Patent Phrase to Phrase Matching challenge."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification

ROOT = Path(__file__).resolve().parent
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT.parent))

from bert_training import fit, load_tokenizer, setup, training_args


CPC_SECTIONS = {
    "A": "human necessities",
    "B": "performing operations and transporting",
    "C": "chemistry and metallurgy",
    "D": "textiles and paper",
    "E": "fixed constructions",
    "F": "mechanical engineering, lighting, heating, weapons and blasting",
    "G": "physics",
    "H": "electricity",
    "Y": "emerging and cross-sectional technologies",
}


def make_loader(frame, tokenizer, args, *, train=False):
    contexts = frame["context"].astype(str)
    section_names = contexts.str[0].map(CPC_SECTIONS).fillna("")
    prompts = frame["anchor"] + " | patent category " + contexts + " " + section_names
    encoded = tokenizer(
        prompts.tolist(), frame["target"].tolist(), truncation=True, max_length=args.max_length
    )
    features = [
        {key: values[i] for key, values in encoded.items()}
        for i in range(len(frame))
    ]
    if train:
        for row, score in zip(features, frame["score"]):
            row["labels"] = float(score)

    def collate(rows):
        inputs = [{key: value for key, value in row.items() if key != "labels"} for row in rows]
        batch = tokenizer.pad(inputs, return_tensors="pt", pad_to_multiple_of=8)
        if train:
            batch["labels"] = torch.tensor([row["labels"] for row in rows], dtype=torch.float32)
        return batch

    return DataLoader(features, batch_size=args.batch_size, shuffle=train, collate_fn=collate)


@torch.inference_mode()
def predict(model, batches, device):
    model.eval()
    predictions = []
    for batch in batches:
        outputs = model(**{key: value.to(device) for key, value in batch.items()})
        predictions.extend(outputs.logits.flatten().cpu().tolist())
    return np.clip(predictions, 0.0, 1.0)


def main(argv=None):
    args = training_args(ROOT / "data", ROOT / "submissions" / "submission.csv", 80, argv)
    device = setup(args)
    train = pd.read_csv(args.data / "train.csv", dtype={"id": str}, keep_default_na=False)
    test = pd.read_csv(args.data / "test.csv", dtype={"id": str}, keep_default_na=False)
    groups = train["anchor"].str.strip().str.casefold()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=args.seed)
    train_idx, val_idx = next(splitter.split(train, groups=groups))
    train_rows, val_rows = train.iloc[train_idx], train.iloc[val_idx]
    targets = val_rows["score"].to_numpy()
    print(f"Rows: {len(train_rows):,} training, {len(val_rows):,} validation, {len(test):,} test", flush=True)

    tokenizer = load_tokenizer(args.model)
    train_batches = make_loader(train_rows, tokenizer, args, train=True)
    val_batches = make_loader(val_rows, tokenizer, args)
    model = DistilBertForSequenceClassification.from_pretrained(
        args.model, num_labels=1, problem_type="regression"
    )

    def evaluate(model):
        predictions = predict(model, val_batches, device)
        if np.ptp(predictions) == 0 or np.ptp(targets) == 0:
            return 0.0
        return float(np.corrcoef(targets, predictions)[0, 1])

    print(f"Best held-out Pearson r: {fit(model, train_batches, evaluate, args, device):.5f}", flush=True)
    predictions = predict(model, make_loader(test, tokenizer, args), device)
    if not np.isfinite(predictions).all():
        raise RuntimeError("The model produced an invalid similarity score.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": test["id"], "score": predictions}).to_csv(args.output, index=False)
    print(f"Wrote {len(test):,} predictions to {args.output}", flush=True)


if __name__ == "__main__":
    main()
