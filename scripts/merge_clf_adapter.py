"""
scripts/merge_clf_adapter.py
──────────────────────────────────────────────────────────────────────────────
Merge a LoRA classification adapter into a base Echo-DSRN model and export
the result as a standalone EchoForSequenceClassification checkpoint.

The produced directory is immediately push-able to the Hugging Face Hub and
supports AutoModelForSequenceClassification.from_pretrained() with
trust_remote_code=True.

Usage
-----
    python scripts/merge_clf_adapter.py \\
        --base   ethicalabs/Echo-DSRN-114M-v0.1.2 \\
        --adapter ethicalabs/Echo-SmolTools-114M-NSFW-CLF-PEFT \\
        --output  ./merged-nsfw-clf \\
        --num-labels 2 \\
        --id2label "0:Safe,1:NSFW"

    # Then push to Hub
    huggingface-cli upload ethicalabs/Echo-SmolTools-114M-NSFW-CLF ./merged-nsfw-clf
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_id2label(raw: str) -> dict:
    """Parse ``'0:Safe,1:NSFW'`` into ``{0: 'Safe', 1: 'NSFW'}``."""
    mapping = {}
    for pair in raw.split(","):
        k, v = pair.strip().split(":")
        mapping[int(k.strip())] = v.strip()
    return mapping


def main():
    parser = argparse.ArgumentParser(
        description="Merge a LoRA CLF adapter into an Echo-DSRN base and export as "
        "EchoForSequenceClassification."
    )
    parser.add_argument(
        "--base",
        required=True,
        help="Hub ID or local path of the base EchoForCausalLM model.",
    )
    parser.add_argument(
        "--adapter",
        required=True,
        help="Hub ID or local path of the PEFT LoRA adapter.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory to save the merged EchoForSequenceClassification model.",
    )
    parser.add_argument(
        "--num-labels",
        type=int,
        default=2,
        help="Number of output classes (default: 2).",
    )
    parser.add_argument(
        "--id2label",
        default=None,
        help="Label mapping in the form '0:Safe,1:NSFW'. Defaults to '0:0,1:1,...'.",
    )
    parser.add_argument(
        "--classifier-dropout",
        type=float,
        default=0.0,
        help="Dropout probability before the classification head (default: 0.0).",
    )
    parser.add_argument(
        "--label-token-ids",
        default=None,
        help="Comma-separated token IDs to seed the classifier head from lm_head rows. "
        "Order must match --id2label class order. "
        "Example: '29900,29896' for tokens '0' and '1'.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
        help="Target dtype for the merged weights.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="System prompt to bake into the model config for the classify() method.",
    )
    parser.add_argument(
        "--user-template",
        type=str,
        default=None,
        help="User template to bake into the config (e.g. 'Classify: {text}').",
    )
    args = parser.parse_args()

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.dtype]

    # --- Resolve id2label / label2id ---
    if args.id2label:
        id2label = parse_id2label(args.id2label)
    else:
        id2label = {i: str(i) for i in range(args.num_labels)}
    label2id = {v: k for k, v in id2label.items()}

    # --- Resolve label_token_ids ---
    label_token_ids = None
    if args.label_token_ids:
        label_token_ids = [int(t.strip()) for t in args.label_token_ids.split(",")]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    step = 1
    total_steps = 5

    print(f"[{step}/{total_steps}] Loading base model: {args.base}")
    step += 1
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )

    print(f"[{step}/{total_steps}] Loading LoRA adapter: {args.adapter}")
    step += 1
    peft_model = PeftModel.from_pretrained(base, args.adapter, trust_remote_code=True)

    print(f"[{step}/{total_steps}] Merging adapter weights into backbone …")
    step += 1
    merged_causal = peft_model.merge_and_unload()

    # Import here so the script is runnable from the repo root without install
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from echo_dsrn.modeling_echo import EchoForSequenceClassification  # noqa: E402

    print(f"[{step}/{total_steps}] Converting to EchoForSequenceClassification …")
    step += 1
    clf = EchoForSequenceClassification.from_causal_lm(
        merged_causal,
        num_labels=args.num_labels,
        id2label=id2label,
        label2id=label2id,
        classifier_dropout=args.classifier_dropout,
        label_token_ids=label_token_ids,
        system_prompt=args.system_prompt,
        user_template=args.user_template,
    )
    clf.eval()

    print(f"[{step}/{total_steps}] Saving to {output_dir} …")
    clf.save_pretrained(output_dir)

    # Save tokenizer alongside so the directory is self-contained
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    # Copy modeling code
    src_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "echo_dsrn")
    for fname in os.listdir(src_dir):
        if fname.endswith(".py"):
            shutil.copy(os.path.join(src_dir, fname), output_dir)

    # Generate Model Card
    readme_content = f"""---
library_name: transformers
tags:
- text-classification
- echo-dsrn
base_model:
- {args.base}
---

# {output_dir.name}

Binary sequence classification model based on the **Echo-DSRN** architecture.
Merged from the base model [`{args.base}`](https://huggingface.co/{args.base})
and the PEFT adapter [`{args.adapter}`](https://huggingface.co/{args.adapter}).

The classification head is seeded from the `lm_head` token rows for the label tokens.
The chat template used during training is baked into `config.json` and applied automatically by `classify()`.

## Model Details

- **Architecture:** `EchoForSequenceClassification`
- **Base model:** `{args.base}`
- **Adapter:** `{args.adapter}`
- **Labels:** {id2label}
- **Dtype:** `{args.dtype}`

## Usage

This model requires `trust_remote_code=True` to load the custom architecture.

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_id = "{output_dir.name}"  # or your hub path

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.{args.dtype},
    device_map="auto",
)

label, probs = model.classify("Enter your text here", tokenizer)
print(f"Prediction: {{label}}")
```
"""
    with open(output_dir / "README.md", "w") as f:
        f.write(readme_content)

    print()
    print("✓ Done.")
    print(f"  Labels  : {id2label}")
    print(f"  Dtype   : {args.dtype}")
    print(f"  Saved to: {output_dir.resolve()}")
    print()
    print("To push to the Hub:")
    print(f"  huggingface-cli upload <your-org/repo-name> {output_dir.resolve()}")


if __name__ == "__main__":
    main()
