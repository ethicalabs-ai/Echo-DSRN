# Echo-DSRN: Surprise-Gated Dual-State Recurrent Architecture

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://img.shields.io/badge/Paper-Working_Paper-green.svg)](PAPER.md)
[![Model Collection](https://img.shields.io/badge/Echo--DSRN-HuggingFace-yellow.svg)](https://huggingface.co/collections/ethicalabs/echo-dsrn)
[![Hybrid Collection](https://img.shields.io/badge/Echo--Hybrid-HuggingFace-yellow.svg)](https://huggingface.co/collections/ethicalabs/echo-dsrn-hybrid)

**Echo-DSRN** is a hybrid recurrent architecture designed for resource-constrained deployment on narrow, well-defined tasks (e.g., intent routing, NER, semantic classification).

It combines three parallel computational paths within each block:
1. **Fast GRU state**: Tracks short-range token dynamics, updated every token.
2. **Surprise-gated slow state**: Selectively accumulates long-range information, write-protected by default and triggered by prediction error.
3. **Sliding window attention**: Handles fine-grained local dependencies within a bounded context window (128 tokens).

This is the canonical Hugging Face implementation of the Echo-DSRN 114M model and its hybrid variant (using a Qwen 2.5 backbone).

It features constant memory overhead (O(1) recurrent core + bounded O(window_size) attention) during generation.

Read the full architectural details in the [working paper](PAPER.md).

## Repository Structure

The repository is organized into cleanly separated modules to distinguish core Hugging Face components from training and deployment scripts:

```
Echo-DSRN/
├── echo_dsrn/           # Core library for the Echo-DSRN model
├── echo_hybrid/         # Core library for the Hybrid model (Qwen2.5 backbone + DSRN memory)
├── benchmarks/          # Evaluation scripts for classification models
├── examples/            # Interactive inference examples
├── scripts/             # Canonical PEFT merge utilities
├── tests/               # pytest suite
├── PAPER.md             # The Echo-DSRN Working Paper
└── README.md            # This document
```

## Installation

This repository uses [uv](https://github.com/astral-sh/uv) for lightning-fast dependency management. You can also install it directly via pip or use it via Hugging Face's `trust_remote_code=True` mechanism.

```bash
# Clone the repository
git clone https://github.com/ethicalabs-ai/Echo-DSRN.git
cd Echo-DSRN-HF

# Install dependencies for local development
uv sync

# For AMD ROCm support (Requires ROCm 7.2 on your system, works on Strix Halo as well)
uv sync --group rocm
```

## Quick Start (Inference)

### Echo-DSRN Base (114M)

The `echo_dsrn` package provides the AutoClass registered models.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import echo_dsrn  # Must be imported to register AutoClasses!

model_id = "ethicalabs/Echo-DSRN-114M-v0.1.2"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    trust_remote_code=True
)

inputs = tokenizer("The theory of predictive coding suggests that", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Echo-Hybrid (0.5B)

The `echo_hybrid` package provides the models with the Qwen2.5 backbone and integrated DSRN memory blocks.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import echo_hybrid  # Must be imported to register AutoClasses!

model_id = "ethicalabs/Echo-Hybrid-0.5B"  # replace with your hub path

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    trust_remote_code=True
)
```

## Classification Models

Echo-DSRN ships two classification heads that share the same backbone:

| Model | Class | Head type | Best for |
|---|---|---|---|
| `Echo-SmolTools-114M-Intent-CLF-Gen` | `EchoForGenerativeClassification` | Constrained scoring (no new weights) | Multi-token labels (e.g. MASSIVE intents) |
| `Echo-SmolTools-114M-NSFW-CLF` | `EchoForSequenceClassification` | Seeded `nn.Linear` from `lm_head` | Single-token labels (e.g. `"0"` / `"1"`) |

### Intent Classification — `EchoForGenerativeClassification`

Classifies text into one of the **60 Amazon MASSIVE intent classes** across 51 languages.
No linear head is trained — the adapter's generative knowledge is used directly via **constrained scoring**:
for each candidate label the model sums the log-probability of each of its tokens, then picks the highest-scoring one.

```python
import echo_dsrn  # registers AutoClasses
from echo_dsrn.modeling_generative_clf import EchoForGenerativeClassification
from transformers import AutoTokenizer

model_id = "ethicalabs/Echo-SmolTools-114M-Intent-CLF-Gen"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = EchoForGenerativeClassification.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype="bfloat16",
    device_map="auto",
)

# Single utterance
label, probs = model.classify("Will it rain tomorrow in Paris?", tokenizer)
print(label)          # → "weather_query"
print(probs.max())    # → ~0.998

# Batch (up to batch_size=32 tested)
labels, probs = model.classify(
    ["Set an alarm for 7am", "Play some jazz", "¿Va a llover mañana?"],
    tokenizer,
)
print(labels)  # → ["alarm_set", "play_music", "weather_query"]
```

See [`examples/classify_dsrn_gen.py`](examples/classify_dsrn_gen.py) for a full runnable example.

To build the checkpoint from the PEFT adapter (no training needed):

```bash
uv run python3 scripts/merge_intent_gen_clf.py
# → models/ethicalabs/Echo-SmolTools-114M-Intent-CLF-Gen
```

### NSFW Classification — `EchoForSequenceClassification`

Binary classifier (safe / unsafe) with a linear head seeded from the `lm_head` token rows for `"0"` and `"1"`.

```python
import echo_dsrn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_id = "ethicalabs/Echo-SmolTools-114M-NSFW-CLF"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype="bfloat16",
    device_map="auto",
)

label, probs = model.classify("How do I make a cake?", tokenizer)
print(label)   # → "safe"
```

To build the checkpoint from the PEFT adapter:

```bash
uv run python3 scripts/merge_clf_adapter.py \
    --base ethicalabs/Echo-DSRN-114M-v0.1.2 \
    --adapter ethicalabs/Echo-SmolTools-114M-NSFW-CLF-PEFT \
    --output models/ethicalabs/Echo-SmolTools-114M-NSFW-CLF \
    --num-labels 2 \
    --id2label "0:Safe,1:NSFW" \
    --label-token-ids "29900,29896" \
    --dtype bfloat16 \
    --system-prompt "You are a helpful NSFW classification assistant." \
    --user-template "Classify the following text (0 for Safe, 1 for NSFW): {text}"
```

## Benchmarks & Evaluation

The repository includes evaluation scripts for both classification architectures.
All commands are also available via `make` — run `make help` to see the full list.

### Evaluating Generative Classifiers (MASSIVE)

Evaluates `EchoForGenerativeClassification` on the Amazon MASSIVE dataset (60 intents, 51 languages):

```bash
# Via make
make eval-intent

# Or directly
uv run python3 benchmarks/run_generative_clf_eval.py \
    --model models/ethicalabs/Echo-SmolTools-114M-Intent-CLF-Gen \
    --batch_size 32 \
    --langs all
```

### Evaluating Sequence Classifiers (NSFW)

Evaluates `EchoForSequenceClassification` on the NSFW Safe Dataset (40k samples):

```bash
# Via make
make eval-nsfw

# Or directly
uv run python3 benchmarks/run_clf_eval.py \
    --model models/ethicalabs/Echo-SmolTools-114M-NSFW-CLF \
    --dataset eliasalbouzidi/NSFW-Safe-Dataset \
    --batch_size 128
```

*Note: The chat template used during training is baked into `config.json` and applied automatically during evaluation.*

## License

Echo-DSRN is released under the [Apache 2.0 License](LICENSE).

## Citation

```bibtex
@misc{Massimo Roberto Scamarcia, title={Echo-DSRN-114M: Surprise-Gated Dual-State Recurrent Architecture for Efficient Language Modeling and Classification}, DOI={10.5281/zenodo.19848279}, publisher={Zenodo}, author={Massimo Roberto Scamarcia} }
```
