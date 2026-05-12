# Echo-DSRN: Surprise-Gated Dual-State Recurrent Architecture

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://img.shields.io/badge/Paper-Working_Paper-green.svg)](PAPER.md)
[![Model Collection](https://img.shields.io/badge/Models-HuggingFace-yellow.svg)](https://huggingface.co/collections/ethicalabs/echo-dsrn)

**Echo-DSRN** is a hybrid recurrent architecture designed for resource-constrained deployment on narrow, well-defined tasks (e.g., intent routing, NER, semantic classification).

It combines three parallel computational paths within each block:
1. **Fast GRU state**: Tracks short-range token dynamics, updated every token.
2. **Surprise-gated slow state**: Selectively accumulates long-range information, write-protected by default and triggered by prediction error.
3. **Sliding window attention**: Handles fine-grained local dependencies within a bounded context window (128 tokens).

This is the canonical Hugging Face implementation of the Echo-DSRN model and its hybrid variant (using a Qwen 2.5 backbone).

It features constant memory overhead (O(1) recurrent core + bounded O(window_size) attention) during generation.

Read the full architectural details in the [working paper](PAPER.md).

## Repository Structure

The repository is organized into cleanly separated modules to distinguish core Hugging Face components from training and deployment scripts:

```
Echo-DSRN-HF/
├── echo_dsrn/           # Core library for the base 114M model (Config, Model, Cache)
├── echo_hybrid/         # Core library for the 0.5B hybrid model (Qwen2.5 backbone + DSRN memory)
├── training/            # SFT, DPO, and Hybrid training scripts
├── examples/            # Interactive scripts (talk_dsrn.py, gradio_hybrid_app.py)
├── deploy/              # Inference endpoint handlers
├── scripts/             # Utilities (PEFT merging, weight conversion, upscaling)
├── tests/               # Model and memory unit tests
├── PAPER.md             # The Echo-DSRN Working Paper
└── README.md            # This document
```

## Installation

This repository uses [uv](https://github.com/astral-sh/uv) for lightning-fast dependency management. You can also install it directly via pip or use it via Hugging Face's `trust_remote_code=True` mechanism.

```bash
# Clone the repository
git clone https://github.com/ethicalabs-ai/Echo-DSRN-HF.git
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
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import echo_hybrid  # Must be imported to register AutoClasses!

model_id = "ethicalabs/Echo-Hybrid-0.5B-Base"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    trust_remote_code=True
)

# You can run interactive sessions using the examples:
# python examples/talk_hybrid.py
```

## Training

All training scripts have been extracted to the `training/` directory. They support Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO).

Example of starting SFT on the base architecture:
```bash
python training/train_dsrn_sft.py \
    --model_name_or_path ethicalabs/Echo-DSRN-114M \
    --dataset_name naufalso/smoltalk2_non_thinking \
    --new_model_name Echo-DSRN-114M-Instruct
```

For advanced training, the codebase implements **Targeted Exact-Match Binding** and granular layer freezing for structural grounding.

## License

Echo-DSRN is released under the [Apache 2.0 License](LICENSE).
