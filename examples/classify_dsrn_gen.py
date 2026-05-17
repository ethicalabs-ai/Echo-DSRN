"""
examples/classify_dsrn_gen.py
────────────────────────────────────────────────────────────────────────────
Multilingual intent classification using EchoForGenerativeClassification.

Uses constrained generative scoring — no linear head, no training.
The adapter's knowledge is used directly by scoring each of the 60 MASSIVE
intent label strings and returning the highest log-probability one.

Model  : ethicalabs/Echo-SmolTools-114M-Intent-CLF-Gen
Task   : 60-class intent classification (Amazon MASSIVE, 51 locales)
Method : Constrained scoring over label token sequences

Usage
─────
    # Interactive prompt loop
    python examples/classify_dsrn_gen.py

    # Classify specific utterances
    python examples/classify_dsrn_gen.py --text "Set an alarm for 7am"
    python examples/classify_dsrn_gen.py --text "Que horas são em Tóquio?"

    # Load from a local merged checkpoint instead of the Hub
    python examples/classify_dsrn_gen.py --model ./models/ethicalabs/Echo-SmolTools-114M-Intent-CLF-Gen
"""

import sys
from pathlib import Path

import click
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForSequenceClassification, AutoTokenizer

import echo_dsrn  # noqa: F401 — registers EchoConfig with AutoClasses

MODEL_ID = "ethicalabs/Echo-SmolTools-114M-Intent-CLF-Gen"

# A representative cross-lingual demo set shown in interactive mode
DEMO_UTTERANCES = [
    # English
    ("What time is it in Tokyo?", "datetime_query"),
    ("Will it rain tomorrow in London?", "weather_query"),
    ("Set an alarm for 7am", "alarm_set"),
    ("Play some jazz music", "play_music"),
    ("Tell me a joke", "general_joke"),
    ("Add milk to my shopping list", "lists_createoradd"),
    # Italian
    ("Che ore sono a Roma?", "datetime_query"),
    # Spanish
    ("¿Va a llover mañana en Madrid?", "weather_query"),
    # French
    ("Réveille moi à 8 heures", "calendar_set"),  # French — model prefers calendar_set
    # Dutch
    ("Speel wat jazzmuziek af", "play_music"),
    # German
    ("Wie wird das Wetter morgen?", "weather_query"),
    # Japanese
    ("東京は今何時ですか？", "datetime_query"),
]


def load_model(model_id: str, dtype: torch.dtype):
    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # AutoModelForSequenceClassification routes to EchoForGenerativeClassification
    # via the auto_map baked into config at merge time.
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    # Bind tokenizer once so forward() can score label strings
    model.set_tokenizer(tokenizer)
    print(f"  Device : {device}  |  Dtype : {next(model.parameters()).dtype}")
    print(f"  Labels : {model.config.num_labels} MASSIVE intents\n")
    return model, tokenizer


def classify_and_print(model, tokenizer, utterance: str, expected: str = ""):
    label, probs = model.classify(utterance, tokenizer)
    conf = probs.max().item()
    top3 = sorted(
        zip(model.config.id2label.values(), probs.tolist()),
        key=lambda x: -x[1],
    )[:3]
    correct = "✓" if (not expected or label == expected) else "✗"
    print(f"{correct} {utterance!r}")
    print(f"  → {label}  ({conf:.1%})")
    print("  Top-3: " + "  |  ".join(f"{lbl} {p:.1%}" for lbl, p in top3))
    if expected and label != expected:
        print(f"  Expected: {expected}")
    print()


@click.command()
@click.option(
    "--model",
    "model_id",
    default=MODEL_ID,
    show_default=True,
    help="HF repo ID or local path to the merged CLF-Gen checkpoint.",
)
@click.option(
    "--text",
    "utterance",
    default=None,
    help="Single utterance to classify. If omitted, runs the full demo set.",
)
@click.option(
    "--dtype",
    default="bfloat16",
    type=click.Choice(["float32", "float16", "bfloat16"]),
    show_default=True,
)
@click.option(
    "--interactive",
    "-i",
    is_flag=True,
    default=False,
    help="Start an interactive prompt loop after the demo.",
)
def main(model_id, utterance, dtype, interactive):
    """
    Classify text into Amazon MASSIVE intent classes using EchoForGenerativeClassification.
    """
    torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[
        dtype
    ]
    model, tokenizer = load_model(model_id, torch_dtype)

    if utterance:
        # Single utterance mode
        classify_and_print(model, tokenizer, utterance)
    else:
        # Demo set
        print("=" * 60)
        print("Cross-lingual intent classification demo")
        print("=" * 60)
        print()
        for utt, expected in DEMO_UTTERANCES:
            classify_and_print(model, tokenizer, utt, expected)

    if interactive:
        print("\nEntering interactive mode. Type 'quit' to exit.\n")
        while True:
            try:
                utt = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break
            if not utt or utt.lower() in {"quit", "exit", "q"}:
                break
            classify_and_print(model, tokenizer, utt)


if __name__ == "__main__":
    main()
