"""DistilBERT span extraction for the Tweet Sentiment Extraction challenge."""

from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import DistilBertForQuestionAnswering

ROOT = Path(__file__).resolve().parent
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT.parent))

from bert_training import fit, load_tokenizer, setup, training_args


def gold_span(text, selected):
    start = text.find(selected)
    if start >= 0:
        return start, start + len(selected)
    # Some annotations collapse whitespace present in the original tweet.
    pattern = r"\s+".join(re.escape(word) for word in selected.split())
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return match.span() if match else None


def make_loader(frame, tokenizer, args, *, train=False):
    texts = frame["text"].tolist()
    encoded = tokenizer(
        frame["sentiment"].tolist(),
        texts,
        truncation="only_second",
        max_length=args.max_length,
        return_offsets_mapping=True,
    )
    offsets = encoded.pop("offset_mapping")
    features = []
    for index, text in enumerate(texts):
        tweet_offsets = [
            (start, end) if sequence == 1 and end > start else (-1, -1)
            for (start, end), sequence in zip(offsets[index], encoded.sequence_ids(index))
        ]
        inputs = {key: values[index] for key, values in encoded.items()}
        if train:
            selected = frame["selected_text"].iat[index]
            span = gold_span(text, selected)
            if span is None:
                continue
            start, end = span
            tokens = [
                idx for idx, (left, right) in enumerate(tweet_offsets)
                if left >= 0 and left < end and right > start
            ]
            if not tokens or tweet_offsets[tokens[-1]][1] < end:
                continue
            features.append((inputs, (tokens[0], tokens[-1])))
        else:
            features.append((inputs, (text, tweet_offsets)))

    def collate(rows):
        inputs, metadata = zip(*rows)
        batch = tokenizer.pad(list(inputs), return_tensors="pt", pad_to_multiple_of=8)
        if train:
            starts, ends = zip(*metadata)
            batch["start_positions"] = torch.tensor(starts)
            batch["end_positions"] = torch.tensor(ends)
            return batch
        return batch, metadata

    return DataLoader(features, batch_size=args.batch_size, shuffle=train, collate_fn=collate)


def decode_span(text, offsets, starts, ends):
    eligible = [i for i, (left, _) in enumerate(offsets) if left >= 0]
    if not eligible:
        return text
    # Track the strongest start at or before each candidate end.
    start = eligible[0]
    best_start = best_end = start
    best_score = float("-inf")
    for end in eligible:
        if starts[end] > starts[start]:
            start = end
        score = starts[start] + ends[end]
        if score > best_score:
            best_start, best_end, best_score = start, end, score
    return text[offsets[best_start][0]:offsets[best_end][1]]


@torch.inference_mode()
def predict(model, frame, tokenizer, args, device):
    model.eval()
    predictions = frame["text"].tolist()
    indices = np.flatnonzero(frame["sentiment"].to_numpy() != "neutral")
    batches = make_loader(frame.iloc[indices], tokenizer, args) if len(indices) else ()
    index_iter = iter(indices)
    for batch, metadata in batches:
        outputs = model(**{key: value.to(device) for key, value in batch.items()})
        for (text, offsets), starts, ends in zip(
            metadata, outputs.start_logits.cpu().numpy(), outputs.end_logits.cpu().numpy()
        ):
            predictions[next(index_iter)] = decode_span(text, offsets, starts, ends)
    return predictions


def jaccard(selected, predicted):
    left = set(selected.lower().split())
    right = set(predicted.lower().split())
    return len(left & right) / len(left | right) if left or right else 1.0


def main(argv=None):
    args = training_args(ROOT / "data", ROOT / "submissions" / "submission.csv", 112, argv)
    device = setup(args)
    train = pd.read_csv(args.data / "train.csv", keep_default_na=False, dtype=str)
    test = pd.read_csv(args.data / "test.csv", keep_default_na=False, dtype=str)
    train = train.assign(selected_text=train["selected_text"].str.strip())
    train = train[train["text"].str.strip().ne("") & train["selected_text"].ne("")]
    train_rows, val_rows = train_test_split(
        train, test_size=0.15, stratify=train["sentiment"], random_state=args.seed
    )
    polar_rows = train_rows[train_rows["sentiment"] != "neutral"]
    tokenizer = load_tokenizer(args.model)
    train_batches = make_loader(polar_rows, tokenizer, args, train=True)
    print(
        f"Rows: {len(train_batches.dataset):,} aligned positive/negative training spans "
        f"from {len(polar_rows):,} rows, {len(val_rows):,} validation, {len(test):,} test",
        flush=True,
    )
    model = DistilBertForQuestionAnswering.from_pretrained(args.model)

    def evaluate(model):
        predicted = predict(model, val_rows, tokenizer, args, device)
        return float(np.mean([jaccard(gold, pred) for gold, pred in zip(val_rows["selected_text"], predicted)]))

    print(f"Best validation Jaccard: {fit(model, train_batches, evaluate, args, device):.5f}", flush=True)
    predicted = predict(model, test, tokenizer, args, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"textID": test["textID"], "selected_text": predicted}).to_csv(args.output, index=False)
    print(f"Wrote {len(test):,} predictions to {args.output}", flush=True)


if __name__ == "__main__":
    main()
