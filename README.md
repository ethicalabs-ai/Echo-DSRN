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
├── echo_embedding/      # Embedding model + conversion utilities
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

# ROCm (local development — AMD GPU, ROCm 7.2+)
uv sync --extra rocm

# CPU-only (CI, inference without GPU, or non-ROCm machines)
uv sync --extra cpu
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

## Serving with vLLM (OpenAI-compatible API)

Echo-DSRN and Echo-Hybrid serve **natively in vLLM** (≥ 0.27) on ROCm via
`trust_remote_code` — no custom engine, no plugins. The surprise-gated
recurrent state is preserved through vLLM's decode loop (use `--enforce-eager`;
CUDA-graph capture would freeze the recurrent state and wipe memory each step).

The full serving stack runs from `docker-compose.yml` (vLLM ROCm base image):

| Port | Service | Default model | API |
|---|---|---|---|
| 8001 | chat / tool-calling | `mrs83/Kurtis-EON1-Hybrid-2B-v0.1.2` | `/v1/chat/completions` |
| 8002 | intent classification | `ethicalabs/Echo-DSRN-v0.1.4-Embed-Intent-CLF` | `/classify` |
| 8003 | sentence embeddings | `ethicalabs/Echo-DSRN-v0.1.3-Embed-Exp` | `/v1/embeddings` |

```bash
docker compose up -d            # builds the ROCm images, serves all three
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"echo-hybrid","messages":[{"role":"user","content":"Hello!"}]}'
```

Model choices are overridable via environment variables (`HYBRID_MODEL`,
`INTENT_MODEL`, `EMBED_MODEL`) or the repo-local `.env` file (gitignored).
All three services run on vLLM (generation runner for chat, pooling runner
for classification and embeddings). The Echo embedding checkpoints are
sentence-transformers-format repos; vLLM serves them through the pooling
runner (`--convert embed`) with the same `trust_remote_code` modules —
embeddings match the SentenceTransformer path to ~0.997 cosine.

## Surprise-Gate Temperature Modulation (`α`)

Echo-DSRN exposes a novel generation parameter `surprise_temperature_alpha` that couples
the output token distribution to the model's own internal surprise gate λ_t:

```
logits = logits / (1 + α · λ_t)
```

When the model is internally confident (λ_t ≈ 0), the distribution stays sharp.
When the model detects structural surprise (λ_t ≈ 1), the distribution flattens —
the model becomes self-aware of its uncertainty without external calibration.

**This is a pure inference parameter** — no fine-tuning or weight changes needed.

```python
from echo_dsrn import EchoForCausalLM, EchoConfig

config = EchoConfig.from_pretrained("ethicalabs/Echo-DSRN-114M-v0.1.2", trust_remote_code=True)
config.surprise_temperature_alpha = 1.0  # moderate modulation

model = EchoForCausalLM.from_pretrained(
    "ethicalabs/Echo-DSRN-114M-v0.1.2",
    config=config,
    trust_remote_code=True,
)
```

| Model | Recommended α | Effect |
|-------|--------------|--------|
| Echo-DSRN-114M | 1.0–2.0 | Breaks repetition loops, adds +3–4 nats entropy |
| Echo-Hybrid-2B | 0.3–0.5 | Creative writing boost without hallucination |

Higher α values produce more diverse, exploratory output. Lower values stay closer to
the base model's distribution. Set to 0.0 to disable.

### Speculative Decoding with DSpark

Echo-DSRN's surprise gate doubles as a confidence signal for speculative decoding.
The `DSparkEchoScheduler` in `echo_dsrn/dspark_scheduler.py` uses α-modulated draft
tokens with gate-driven dynamic cutoff — no external verification head needed.

```python
from echo_dsrn.dspark_scheduler import DSparkEchoScheduler, DSparkEchoConfig

scheduler = DSparkEchoScheduler(draft_model, DSparkEchoConfig(
    max_draft_len=8, tau_load=0.05, surprise_temperature_alpha=1.0,
))
draft_ids, confidence = scheduler.draft(input_ids)
accepted = scheduler.verify(target_model, input_ids, draft_ids,
                            confidence["cutoff_lens"])
```

At τ_load=0.05, Echo-DSRN-114M achieves 64% token efficiency when drafting against
Phi-3-mini-4k-instruct (3.8B target) — the confidence signal correctly identifies
reliable draft positions.

## Embedding Models

Echo-DSRN can be converted to a dense sentence embedding model via
`EchoModelForSentenceEmbedding`. It pools the recurrent slow state `c_all` across
tokens (`mean_c_all`, 2048-dim) and is compatible with the `sentence-transformers`
library.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "ethicalabs/Echo-DSRN-v0.1.3-Embed-Intent", trust_remote_code=True
)
embeddings = model.encode(["What is the weather?", "Will it rain today?"])
# → (2, 2048) float32 tensor

# Or via the HuggingFace pipeline:
from transformers import pipeline

pipe = pipeline("feature-extraction", model="ethicalabs/Echo-DSRN-v0.1.3-Embed-Intent", trust_remote_code=True)
embeddings = pipe("What is the weather today?")
```

CPU-only inference (loads instantly, ~220 sent/sec):

```python
from echo_embedding.modeling_embedding import EchoModelForSentenceEmbedding
from transformers import AutoTokenizer
import torch

model = EchoModelForSentenceEmbedding.from_pretrained(
    "ethicalabs/Echo-DSRN-v0.1.3-Embed-Intent", trust_remote_code=True
).eval()
tok = AutoTokenizer.from_pretrained(
    "ethicalabs/Echo-DSRN-v0.1.3-Embed-Intent", trust_remote_code=True
)

enc = tok(["What is the weather?", "Will it rain today?"],
          return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    out = model(**enc, output_all_states=True)
    embeddings = out.all_c_all[-1].mean(dim=1)  # mean_c_all pooling
# → (2, 2048) float32 tensor
```

### MTEB Benchmark

| Model | Task | Score |
|-------|------|-------|
| [Echo-DSRN-v0.1.3-Embed-Exp](https://huggingface.co/ethicalabs/Echo-DSRN-v0.1.3-Embed-Exp) | STS (7 tasks) | **0.753** avg Spearman |
| [Echo-DSRN-v0.1.3-Embed-Intent](https://huggingface.co/ethicalabs/Echo-DSRN-v0.1.3-Embed-Intent) | MassiveIntentClassification (51 langs) | **72.42%** accuracy |
| [Echo-DSRN-v0.1.3-Embed-Intent](https://huggingface.co/ethicalabs/Echo-DSRN-v0.1.3-Embed-Intent) | MassiveScenarioClassification (51 langs) | **79.00%** accuracy |

Both models are available on the Hub. Training is reproducible from the pipeline
documented in `echo_embedding/`.

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

### Three paths to build an Echo classifier

`EchoForSequenceClassification` supports two construction paths, plus a generative
classifier for multi-token labels:

#### Path 1: Causal LM → Classifier (`from_causal_lm()`) — `EchoForSequenceClassification`

Builds on a generative backbone. Used by `Echo-DSRN-v0.1.3-Intent-CLF` (60-class MASSIVE).

- **Pooling:** Last-token hidden state (768-dim fast state)
- **Inference:** Chat template required — `system_prompt` + `user_template` baked into config
- **Training:** Frozen backbone → sklearn LogisticRegression → copy weights to `nn.Linear` head
- **Strength:** Exploits LM-trained surface-form features; ~83% en-US

```python
from echo_dsrn.modeling_echo import EchoForSequenceClassification
from transformers import AutoTokenizer

model = EchoForSequenceClassification.from_pretrained(
    "ethicalabs/Echo-DSRN-v0.1.3-Intent-CLF", trust_remote_code=True
)
tok = AutoTokenizer.from_pretrained(
    "ethicalabs/Echo-DSRN-v0.1.3-Intent-CLF", trust_remote_code=True
)
# classify() wraps text in chat template automatically
label, probs = model.classify("turn off the lights", tokenizer=tok)
# → iot_hue_lightoff
```

#### Path 2: Embedding → Classifier (`from_embedding()`)

Builds on an MNRL-trained embedding model. Used by `Echo-DSRN-v0.1.4-Embed-Intent-CLF` (60-class MASSIVE).

- **Pooling:** Mean of recurrent slow states (`mean_c_all`, 2048-dim)
- **Inference:** Raw text — no chat template (`classification_use_chat_template: false`)
- **Training:** Sklearn SGDClassifier init (86.49% train acc) + cross-entropy fine-tuning
- **Strength:** Cross-lingual consistency from MNRL-trained embedding space; ~79% en-US

```python
from echo_dsrn.modeling_echo import EchoForSequenceClassification
from transformers import AutoTokenizer

model = EchoForSequenceClassification.from_pretrained(
    "ethicalabs/Echo-DSRN-v0.1.4-Embed-Intent-CLF", trust_remote_code=True
)
tok = AutoTokenizer.from_pretrained(
    "ethicalabs/Echo-DSRN-v0.1.4-Embed-Intent-CLF", trust_remote_code=True
)
# classify() passes raw text directly
label, probs = model.classify("turn off the lights", tokenizer=tok)
# → iot_hue_lightoff

# Or via pipeline:
from transformers import pipeline
pipe = pipeline("text-classification", model="ethicalabs/Echo-DSRN-v0.1.4-Embed-Intent-CLF", trust_remote_code=True)
pipe("turn off the lights")[0]  # → {'label': 'iot_hue_lightoff', 'score': 0.98}
```

#### Path 3: Generative Classifier — `EchoForGenerativeClassification`

No classification head — the generative adapter's own token distribution is the classifier.

For each candidate label (e.g., `"weather_query"`), the model sums the
log-probabilities of its tokens conditioned on the input utterance. The label
with the highest total log-probability wins. Since the input prefix is identical
for all candidates, the KV cache is computed once and shared across labels.

**Zero added parameters, zero training** — the same adapter that generates text
also scores labels. Used by `Echo-SmolTools-114M-Intent-CLF-Gen` (60-class MASSIVE).

See the [Intent Classification](#intent-classification--echoforgenerativeclassification) section above for full documentation.

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
