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
├── examples/            # Interactive scripts
├── scripts/             # Utilities (PEFT merging, weight conversion, upscaling)
├── scratch/             # Scratch code
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
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import echo_hybrid  # Must be imported to register AutoClasses!

model_id = "mrs83/Kurtis-EON1-Hybrid-0.7B-v0.1.0"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    trust_remote_code=True
)
```

## License

Echo-DSRN is released under the [Apache 2.0 License](LICENSE).

## Citation

```bibtex
@misc{Massimo Roberto Scamarcia, title={Echo-DSRN-114M: Surprise-Gated Dual-State Recurrent Architecture for Efficient Language Modeling and Classification}, DOI={10.5281/zenodo.19848279}, publisher={Zenodo}, author={Massimo Roberto Scamarcia} }
```
