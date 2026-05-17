"""
scratch/merge_intent_gen_clf.py
────────────────────────────────────────────────────────────────────────────
Merge Echo-SmolTools-114M-Intent-PEFT into EchoForGenerativeClassification.

No training required — the adapter's generative knowledge is used directly
via constrained scoring over the 60 MASSIVE intent label strings.

Usage
─────
    python scratch/merge_intent_gen_clf.py
    python scratch/merge_intent_gen_clf.py --output models/ethicalabs/Echo-SmolTools-114M-Intent-CLF-Gen
    python scratch/merge_intent_gen_clf.py --dtype float16
"""

import shutil
import sys
from pathlib import Path

import click
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

import echo_dsrn  # noqa: F401
from echo_dsrn.modeling_generative_clf import EchoForGenerativeClassification

# ---------------------------------------------------------------------------
# Canonical MASSIVE intent ordering
# Source: _INTENTS list in AmazonScience/massive/raw/main/massive.py
# ---------------------------------------------------------------------------
MASSIVE_INTENTS = [
    "datetime_query",  # 0
    "iot_hue_lightchange",  # 1
    "transport_ticket",  # 2
    "takeaway_query",  # 3
    "qa_stock",  # 4
    "general_greet",  # 5
    "recommendation_events",  # 6
    "music_dislikeness",  # 7
    "iot_wemo_off",  # 8
    "cooking_recipe",  # 9
    "qa_currency",  # 10
    "transport_traffic",  # 11
    "general_quirky",  # 12
    "weather_query",  # 13
    "audio_volume_up",  # 14
    "email_addcontact",  # 15
    "takeaway_order",  # 16
    "email_querycontact",  # 17
    "iot_hue_lightup",  # 18
    "recommendation_locations",  # 19
    "play_audiobook",  # 20
    "lists_createoradd",  # 21
    "news_query",  # 22
    "alarm_query",  # 23
    "iot_wemo_on",  # 24
    "general_joke",  # 25
    "qa_definition",  # 26
    "social_query",  # 27
    "music_settings",  # 28
    "audio_volume_other",  # 29
    "calendar_remove",  # 30
    "iot_hue_lightdim",  # 31
    "calendar_query",  # 32
    "email_sendemail",  # 33
    "iot_cleaning",  # 34
    "audio_volume_down",  # 35
    "play_radio",  # 36
    "cooking_query",  # 37
    "datetime_convert",  # 38
    "qa_maths",  # 39
    "iot_hue_lightoff",  # 40
    "iot_hue_lighton",  # 41
    "transport_query",  # 42
    "music_likeness",  # 43
    "email_query",  # 44
    "play_music",  # 45
    "audio_volume_mute",  # 46
    "social_post",  # 47
    "alarm_set",  # 48
    "qa_factoid",  # 49
    "calendar_set",  # 50
    "play_game",  # 51
    "alarm_remove",  # 52
    "lists_remove",  # 53
    "transport_taxi",  # 54
    "recommendation_movies",  # 55
    "iot_coffee",  # 56
    "music_query",  # 57
    "play_podcasts",  # 58
    "lists_query",  # 59
]
assert len(MASSIVE_INTENTS) == 60

ID2LABEL = {i: label for i, label in enumerate(MASSIVE_INTENTS)}

BASE_MODEL = "ethicalabs/Echo-DSRN-114M-v0.1.2"
PEFT_ADAPTER = "ethicalabs/Echo-SmolTools-114M-Intent-PEFT"
DEFAULT_OUT = str(
    Path(__file__).parent.parent / "models" / "ethicalabs" / "Echo-SmolTools-114M-Intent-CLF-Gen"
)

SYSTEM_PROMPT = "You are a helpful multilingual intent classification assistant."


@click.command()
@click.option(
    "--base", default=BASE_MODEL, show_default=True, help="Base model HF repo or local path."
)
@click.option(
    "--adapter", default=PEFT_ADAPTER, show_default=True, help="PEFT adapter HF repo or local path."
)
@click.option("--output", default=DEFAULT_OUT, show_default=True, help="Output directory.")
@click.option(
    "--dtype",
    default="bfloat16",
    show_default=True,
    type=click.Choice(["float32", "float16", "bfloat16"]),
    help="Weight dtype for the saved checkpoint.",
)
@click.option(
    "--smoke-test", is_flag=True, default=False, help="Run 4 classify() calls before saving."
)
def main(base, adapter, output, dtype, smoke_test):
    """
    Merge Echo-SmolTools-114M-Intent-PEFT into EchoForGenerativeClassification
    and save a ready-to-push HF checkpoint.
    """
    torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[
        dtype
    ]
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load base + adapter
    # ------------------------------------------------------------------
    print(f"[1/4] Loading base model: {base}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base, trust_remote_code=True, torch_dtype=torch_dtype
    )
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)

    print(f"[2/4] Loading LoRA adapter: {adapter}")
    peft_model = PeftModel.from_pretrained(base_model, adapter, trust_remote_code=True)

    print("[3/4] Merging adapter …")
    merged = peft_model.merge_and_unload()

    # ------------------------------------------------------------------
    # 2. Wrap as EchoForGenerativeClassification
    # ------------------------------------------------------------------
    print("[4/4] Building EchoForGenerativeClassification …")
    model = EchoForGenerativeClassification.from_causal_lm(
        merged,
        num_labels=60,
        id2label=ID2LABEL,
        system_prompt=SYSTEM_PROMPT,
        user_template="Classify the intent of the following request: {utt}",
    )

    # ------------------------------------------------------------------
    # 3. Optional smoke test
    # ------------------------------------------------------------------
    if smoke_test:
        print("\nSmoke test:")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device).eval()

        tests = [
            "What time is it in Rome?",
            "Will it rain tomorrow in Paris?",
            "Set an alarm for 7am",
            "Play some jazz music",
            "Che ore sono a Tokyo?",  # Italian — multilingual
            "¿Va a llover mañana en Madrid?",  # Spanish — multilingual
        ]
        for utt in tests:
            label, probs = model.classify(
                utt,
                tokenizer,
                system_prompt=SYSTEM_PROMPT,
            )
            conf = probs.max().item()
            print(f"  {utt!r:50s} → {label}  ({conf:.1%})")

    # ------------------------------------------------------------------
    # 4. Save
    # ------------------------------------------------------------------
    print(f"\nSaving to {output_path.resolve()} …")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # Copy all echo_dsrn package source files so trust_remote_code works.
    # triton_scan.py is required at import time by modeling_echo.py — it must
    # be present alongside the other .py files in the checkpoint directory.
    pkg = Path(__file__).parent.parent / "echo_dsrn"
    required = [
        "triton_scan.py",
        "utils.py",
        "configuration_echo.py",
        "modeling_echo.py",
        "modeling_generative_clf.py",
        "__init__.py",
    ]
    for name in required:
        src = pkg / name
        if not src.exists():
            raise FileNotFoundError(f"Required source file missing: {src}")
        shutil.copy(src, output_path / name)
        print(f"  copied {name}")

    # Generate Model Card
    readme_content = f"""---
library_name: transformers
tags:
- text-classification
- zero-shot-classification
- echo-dsrn
base_model:
- {base}
---

# {output_path.name}

This is a **generative** sequence classification model based on the **Echo-DSRN** architecture.
It was merged from the base model [`{base}`](https://huggingface.co/{base})
and the PEFT adapter [`{adapter}`](https://huggingface.co/{adapter}).

No additional linear head is trained — the adapter's generative knowledge is used directly via
**constrained next-token scoring**: for each candidate label the model sums the log-probability
of each of its tokens, then picks the highest-scoring one.

## Model Details

- **Architecture:** `EchoForGenerativeClassification`
- **Base model:** `{base}`
- **Adapter:** `{adapter}`
- **Labels:** 60 Amazon MASSIVE intents (51 languages)
- **Dtype:** `{dtype}`
- **Constraint Method:** Next-token generative scoring

## Usage

This model requires `trust_remote_code=True` to load the custom architecture.

```python
import torch
from transformers import AutoTokenizer
from echo_dsrn.modeling_generative_clf import EchoForGenerativeClassification

model_id = "{output_path.name}" # or your hub path

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = EchoForGenerativeClassification.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.{dtype},
    device_map="auto",
)

# Single utterance
label, probs = model.classify("Enter your text here", tokenizer)
print(f"Prediction: {{label}}")
```
"""
    with open(output_path / "README.md", "w") as f:
        f.write(readme_content)

    print()
    print("=" * 65)
    print("✓ Done.")
    print(f"  Output : {output_path.resolve()}")
    print(f"  Dtype  : {dtype}")
    print(f"  Labels : {len(ID2LABEL)} MASSIVE intents")
    print()
    print("To push:")
    print("  huggingface-cli upload ethicalabs/Echo-SmolTools-114M-Intent-CLF-Gen \\")
    print(f"      {output_path.resolve()}")


if __name__ == "__main__":
    main()
