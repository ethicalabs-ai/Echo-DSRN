"""
benchmarks/run_generative_clf_eval.py
──────────────────────────────────────────────────────────────────────────────
Evaluates the EchoForGenerativeClassification model on the Amazon MASSIVE benchmark.
Instead of evaluating embeddings (like MTEB), this script evaluates the direct
zero-shot generative classification log-probabilities natively.
"""

import argparse
import os
import sys

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Ensure we can import echo_dsrn if running from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import echo_dsrn  # noqa: F401


def load_massive_test(locales=None):
    """Loads the MASSIVE test set."""
    print("Loading AmazonScience/massive test set...")
    ds = load_dataset(
        "AmazonScience/massive",
        split="test",
        revision="refs/convert/parquet",
        trust_remote_code=True,
    )
    if locales:
        locale_set = set(locales)
        ds = ds.filter(lambda x: x["locale"] in locale_set, num_proc=4)
    return ds


def main():
    parser = argparse.ArgumentParser(description="Evaluate Generative Intent Classifier on MASSIVE")
    parser.add_argument(
        "--model",
        type=str,
        default="ethicalabs/Echo-SmolTools-114M-Intent-CLF-Gen",
        help="Path or Hub ID to the Echo classifier model.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation. WARNING: Must be 1 for Echo-DSRN because the architecture does not currently mask out padding tokens from recurrent state updates.",
    )
    parser.add_argument(
        "--langs",
        type=str,
        default="en-US",
        help="Comma-separated list of MASSIVE locales (e.g. 'en-US,fr-FR'). Use 'all' for all.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading tokenizer and model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # We load it using AutoModelForSequenceClassification because we want the generative classifier
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    # The generative classifier requires the tokenizer for label scoring
    model.set_tokenizer(tokenizer)

    locales = None if args.langs == "all" else args.langs.split(",")
    dataset = load_massive_test(locales)

    print(f"Evaluating on {len(dataset)} samples across {args.langs} locales...")

    correct = 0
    total = len(dataset)

    for i in tqdm(range(0, total, args.batch_size), desc="Evaluating"):
        batch = dataset[i : i + args.batch_size]
        texts = batch["utt"]
        labels = torch.tensor(batch["intent"], dtype=torch.long, device=device)

        # classify() handles the chat template and tokenization
        with torch.inference_mode():
            _, probs = model.classify(texts, tokenizer)
            preds = probs.argmax(dim=-1)
            correct += (preds == labels).sum().item()

    accuracy = correct / total
    print("\n" + "=" * 50)
    print("RESULTS:")
    print("=" * 50)
    print(f"Model:    {args.model}")
    print(f"Locales:  {args.langs}")
    print(f"Samples:  {total}")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print("=" * 50)


if __name__ == "__main__":
    main()
