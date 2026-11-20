"""Shared training loop for the two DistilBERT competition baselines."""

import argparse
from pathlib import Path
from time import perf_counter

import torch
from tokenizers import BertWordPieceTokenizer, Tokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerFast, get_linear_schedule_with_warmup, set_seed


def training_args(default_data, default_output, default_length, argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=default_data)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--model", default="distilbert/distilbert-base-uncased")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=default_length)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.max_length) <= 0 or args.learning_rate <= 0:
        parser.error("Epochs, batch size, max length and learning rate must be positive.")
    return args


def setup(args):
    set_seed(args.seed)
    if args.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(args.device)


def load_tokenizer(model):
    path = Path(model)
    vocab = path / "vocab.txt"
    if not vocab.is_file() or (path / "tokenizer.json").is_file():
        return AutoTokenizer.from_pretrained(model, use_fast=True)
    # The offline checkpoint ships only the original WordPiece vocabulary.
    wordpiece = BertWordPieceTokenizer(str(vocab), lowercase=True)
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_str(wordpiece.to_str()),
        unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]",
        cls_token="[CLS]", mask_token="[MASK]",
        model_input_names=["input_ids", "attention_mask"],
    )


def fit(model, batches, evaluate, args, device):
    """Keep the epoch with the best held-out competition metric."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(batches) * args.epochs
    if not total_steps:
        raise ValueError("No training batches.")
    scheduler = get_linear_schedule_with_warmup(optimizer, total_steps // 10, total_steps)
    fp16 = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    best_score = float("-inf")
    best_weights = None
    start = perf_counter()
    step = 0
    print(f"Training on {device}: {len(batches)} batches per epoch, {args.epochs} epochs.", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=fp16):
                loss = model(**{key: value.to(device) for key, value in batch.items()}).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous_scale:
                scheduler.step()
            step += 1
            if step % 100 == 0 or step == total_steps:
                eta = (perf_counter() - start) / step * (total_steps - step) / 60
                print(f"Step {step}/{total_steps} | loss {loss.item():.3f} | training ETA {eta:.1f} min", flush=True)
        score = evaluate(model)
        print(f"Epoch {epoch}: validation score {score:.5f}", flush=True)
        if score > best_score:
            best_score = score
            best_weights = {key: value.detach().to("cpu", copy=True) for key, value in model.state_dict().items()}
    if best_weights is None:
        raise RuntimeError("Validation did not produce a finite score.")
    model.load_state_dict(best_weights)
    return best_score
