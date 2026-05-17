"""
examples/classify_dsrn.py
──────────────────────────────────────────────────────────────────────────────
Inference example for EchoForSequenceClassification.

Demonstrates loading a merged Echo-DSRN classifier checkpoint and running
both the high-level classify() API and a batched forward pass.

Usage
-----
# Single text (interactive default)
python examples/classify_dsrn.py \\
    --model models/ethicalabs/Echo-SmolTools-114M-Intent-CLF

# Classify a specific string directly
python examples/classify_dsrn.py \\
    --model models/ethicalabs/Echo-SmolTools-114M-Intent-CLF \\
    --text "only one scene of nudity where two women are briefly topless"

# Load from Hub (once pushed)
python examples/classify_dsrn.py \\
    --model ethicalabs/Echo-SmolTools-114M-Intent-CLF \\
    --text "some text to classify"

# Run all built-in demo cases
python examples/classify_dsrn.py \\
    --model models/ethicalabs/Echo-SmolTools-114M-Intent-CLF \\
    --demo
"""

import os
import sys

import click
import torch
from transformers import AutoTokenizer

# Ensure the package is importable when running from the repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import echo_dsrn  # noqa: F401 — registers AutoClass mappings
from echo_dsrn.modeling_echo import EchoForSequenceClassification

# ---------------------------------------------------------------------------
# Demo cases — showcasing the model's classification signal
# ---------------------------------------------------------------------------
DEMO_TEXTS = [
    "only one scene of nudity where two women are briefly topless",
    "The film contains strong language and graphic violence throughout.",
    "Explicit sexual content involving adults in a commercial setting.",
    "A heartwarming story about a dog who finds his way home.",
    "The children played in the park on a sunny afternoon.",
    "This article discusses the economic impact of renewable energy.",
    "She slowly undressed in the candlelit room, eyes locked on his.",
    "The quarterly earnings report exceeded analyst expectations by 12%.",
]


def load_model(model_path: str, device: str):
    """Load EchoForSequenceClassification from a local dir or Hub ID."""
    print(f"Loading tokenizer from {model_path} …")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"Loading classifier model from {model_path} …")
    model = EchoForSequenceClassification.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if "cuda" in device or "hip" in device else torch.float32,
    )
    model.to(device)
    model.eval()

    labels = model.config.id2label
    print(f"  ✓ {type(model).__name__} loaded")
    print(f"  ✓ Labels: {labels}")
    print(f"  ✓ Device: {device}  |  Dtype: {model.dtype}")
    return model, tokenizer


def classify_one(model, tokenizer, text: str, device: str, verbose: bool = True):
    """Run classify() and pretty-print the result."""
    label, probs = model.classify(text, tokenizer, device=device)
    if verbose:
        bar_width = 30
        for idx, (class_label, p) in enumerate(zip(model.config.id2label.values(), probs.tolist())):
            filled = int(p * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)
            marker = " ◄" if class_label == label else ""
            print(f"  {class_label:>8s}  [{bar}]  {p*100:5.1f}%{marker}")
    return label, probs


def classify_batch(model, tokenizer, texts: list, device: str):
    """Run a batched forward pass and return (labels, probs) for all texts."""
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.inference_mode():
        out = model(**enc)

    probs = torch.softmax(out.logits.float(), dim=-1)  # (B, num_labels)
    pred_ids = probs.argmax(dim=-1).tolist()
    labels = [model.config.id2label[i] for i in pred_ids]
    return labels, probs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--model",
    "model_path",
    required=True,
    help="Local path or Hub ID of the EchoForSequenceClassification checkpoint.",
)
@click.option(
    "--text",
    default=None,
    help="Single string to classify.  If omitted, enters interactive mode.",
)
@click.option(
    "--demo",
    is_flag=True,
    default=False,
    help="Run the built-in demo suite instead of interactive/single mode.",
)
@click.option(
    "--batch",
    is_flag=True,
    default=False,
    help="Use batched inference for the demo suite (requires --demo).",
)
@click.option(
    "--device",
    default=None,
    help="Force device (e.g. 'cuda', 'cpu').  Auto-detected if not set.",
)
def main(model_path, text, demo, batch, device):
    """
    Classify text with a merged Echo-DSRN sequence classifier.

    \b
    Examples:
        python examples/classify_dsrn.py --model models/ethicalabs/Echo-SmolTools-114M-Intent-CLF
        python examples/classify_dsrn.py --model ... --text "some text"
        python examples/classify_dsrn.py --model ... --demo
        python examples/classify_dsrn.py --model ... --demo --batch
    """
    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
    print(f"Device: {device}\n")

    model, tokenizer = load_model(model_path, device)
    print()

    # ------------------------------------------------------------------
    # Demo mode — classify all built-in samples
    # ------------------------------------------------------------------
    if demo:
        print("=" * 65)
        print(" DEMO — classifying built-in samples")
        print("=" * 65)

        if batch:
            print(f"Running batched inference over {len(DEMO_TEXTS)} samples …\n")
            labels, probs = classify_batch(model, tokenizer, DEMO_TEXTS, device)
            for t, label, p in zip(DEMO_TEXTS, labels, probs.tolist()):
                score_str = "  ".join(
                    f"{model.config.id2label[i]}={v*100:.1f}%" for i, v in enumerate(p)
                )
                print(f"[{label:>4s}]  {score_str}  |  {t[:60]}")
        else:
            for t in DEMO_TEXTS:
                print(f"\n› {t}")
                classify_one(model, tokenizer, t, device, verbose=True)

        return

    # ------------------------------------------------------------------
    # Single-text mode
    # ------------------------------------------------------------------
    if text:
        print(f"› {text}")
        label, probs = classify_one(model, tokenizer, text, device, verbose=True)
        print(f"\nPrediction: {label}")
        return

    # ------------------------------------------------------------------
    # Interactive mode
    # ------------------------------------------------------------------
    print("Interactive classification — type text and press Enter.")
    print("Commands: 'exit' / 'quit' to stop,  'demo' to run demo suite.\n")

    while True:
        try:
            user_input = input("Text: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break
        if user_input.lower() == "demo":
            for t in DEMO_TEXTS:
                print(f"\n› {t}")
                classify_one(model, tokenizer, t, device, verbose=True)
            continue

        print()
        label, probs = classify_one(model, tokenizer, user_input, device, verbose=True)
        print(f"\n  → {label}\n")

    print("Goodbye!")


if __name__ == "__main__":
    main()
