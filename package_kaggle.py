"""Build self-contained, internet-off Kaggle notebooks from the training scripts."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHALLENGES = {
    "PatentPhraseMatching": (
        "us-patent-phrase-to-phrase-matching", "Patent Phrase Matching DistilBERT"
    ),
    "TweetSentimentExtraction": (
        "tweet-sentiment-extraction", "Tweet Sentiment Extraction DistilBERT"
    ),
}
WEIGHTS = "abhishek/distilbertbaseuncased"


def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def package(challenge, username, output_dir):
    competition, title = CHALLENGES[challenge]
    slug = title.lower().replace(" ", "-")
    notebook_name = f"{slug}.ipynb"
    folder = output_dir / slug
    folder.mkdir(parents=True, exist_ok=True)
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"# {title}\n", f"Fine-tune and validate DistilBERT for [{competition}](https://www.kaggle.com/competitions/{competition})."],
        },
        code_cell("%%writefile /kaggle/working/bert_training.py\n" + (ROOT / "bert_training.py").read_text(encoding="utf-8")),
        code_cell("%%writefile /kaggle/working/train.py\n" + (ROOT / challenge / "train.py").read_text(encoding="utf-8")),
        code_cell(
            f"""import os
import subprocess
import sys
from pathlib import Path

input_root = Path("/kaggle/input")
competition = {competition!r}
data = next(
    path for path in (input_root / "competitions" / competition, input_root / competition)
    if (path / "train.csv").is_file()
)
weights = next(
    path.parent for path in input_root.rglob("pytorch_model.bin")
    if "distilbertbaseuncased" in path.parts and (path.parent / "vocab.txt").is_file()
)
subprocess.run(
    [sys.executable, "-u", "/kaggle/working/train.py", "--data", str(data),
     "--model", str(weights), "--device", "cuda", "--output", "/kaggle/working/submission.csv"],
    env={{**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}},
    check=True,
)
"""
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"name": "python3", "language": "python", "display_name": "Python 3"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    metadata = {
        "id": f"{username}/{slug}",
        "title": title,
        "code_file": notebook_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "competition_sources": [competition],
        "dataset_sources": [WEIGHTS],
        "kernel_sources": [],
        "model_sources": [],
    }
    for name, content in ((notebook_name, notebook), ("kernel-metadata.json", metadata)):
        (folder / name).write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    print(folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("username")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "kaggle")
    args = parser.parse_args()
    for challenge in CHALLENGES:
        package(challenge, args.username, args.output_dir)
