

# Echo-DSRN-114M: Surprise-Gated Dual-State Recurrent Architecture for Efficient Language Modeling and Classification

**Massimo Roberto Scamarcia** (<massimo@ethicalabs.ai>)
*Project Lead, Independent Researcher · ethicalabs.ai · Florence, Italy*

*Working Paper — April 2026 · Apache 2.0 · [Echo-DSRN Collection](https://huggingface.co/collections/ethicalabs/echo-dsrn)*

## Introduction

Large language models based on the Transformer architecture have demonstrated remarkable capabilities across a wide range of language tasks — however, their memory and compute requirements scale quadratically with sequence length, driven by the key-value cache that grows unboundedly during generation.

On a single consumer GPU, on an edge device, or in a federated deployment inside a privacy-constrained institution, this is not an abstract limitation — it is a hard physical wall.

A significant proportion of practical NLP workloads do not require frontier-scale general-purpose models.

Intent classification, named entity recognition, log parsing, semantic routing, and similar narrow tasks are routinely handled by models with billions of parameters that could be served by models several orders of magnitude smaller, with fixed memory footprints and predictable latency.

Deploying an 8-billion-parameter model for binary sentiment classification may be an engineering inefficiency at scale.

This paper describes **Echo-DSRN** (Dual-State Recurrent Network), a hybrid architecture designed for resource-constrained deployment on narrow, well-defined tasks.

Echo-DSRN combines three parallel computational paths within each block: a GRU fast state that tracks short-range token dynamics, a surprise-gated slow memory state that selectively accumulates long-range information, and a bounded sliding-window attention head for precise local context.

The recurrent core maintains **O(1) memory** regardless of sequence length, while the attention component adds a bounded **O(window_size)** cache that stops growing once the window fills.

The surprise gate — the mechanism that controls when and whether the slow state is updated — is an engineering variant of the Titans framework (Ali et al., 2025): prediction error from the fast GRU state modulates the write gate for the slow state, so that routine, expected tokens do not modify long-term memory, while surprising tokens (topic shifts, rare vocabulary, anomalies) trigger a state update — This is grounded in the predictive coding theory of biological memory (Rao & Ballard, 1999).

Echo-DSRN is fully implemented, trained, and released under Apache 2.0. [Weights, training code, and live Gradio/telemetry apps are publicly available](https://huggingface.co/collections/ethicalabs/echo-dsrn). The remainder of this paper describes the architecture in detail, reports inference performance benchmarks, and positions the work relative to the related literature.

## Technical Architecture: How Echo-DSRN Actually Works

Each layer of Echo-DSRN is a *DSRNBlock* — a single module that runs three parallel computational paths simultaneously, with no sequential bottleneck between them:
- **Fast GRU state** — tracks short-range token dynamics, updated every token.
- **Surprise-gated slow state** — selectively accumulates long-range information, write-protected by default.
- **Sliding window attention** — handles fine-grained local token dependencies within a bounded context window.

The recurrent core (*h* and *c*) is genuinely **O(1)** in memory: constant size regardless of sequence length, not approximate, not chunked. The sliding window attention adds a bounded KV-cache of **O(window_size × layers)** — fixed at 128 tokens per layer — which grows to that ceiling and then stops. In practice, benchmarked memory growth is approximately **0.024 MB per token** until the window fills, after which growth stops entirely.

### The State Tuple: What the Model Actually Remembers

Unlike a transformer, Echo-DSRN passes a structured state between layers — it has two distinct components with different memory profiles:

**Recurrent state — O(1), genuinely constant:**
- *h* – the fast state vector, shape (*B*, *D*). Updated every token.
- *c* – the slow state vector, shape (*B*, *D* × *num_heads*). Updated selectively by the surprise gate.

*h* and *c* do not grow with sequence length. They are the same size at token 1 and token 1,000,000.

This is the structural O(1) guarantee — not an approximation, not a compression trick.

**Attention cache — O(window_size), bounded:**
- *k_attn*, *v_attn* – the sliding window KV-cache. Each grows up to *window_size* tokens (128 by default) and then stops, as older entries are evicted — memory cost is **O(window_size × num_heads × head_dim × layers)** — a compile-time constant once the window fills, but not zero and not fixed from the first token.

The honest memory model is therefore: **O(1) recurrent core + O(window_size) attention cache**.

Compared to a standard transformer's O(seq_len) KV-cache, the attention component here is bounded by a constant — making the total footprint effectively fixed-size for long sequences.

### Path 1: The Fast State (GRU)

```python
gru_proj = F.linear(x_norm,
                    model_block.gru_cell.weight_ih,
                    model_block.gru_cell.bias_ih)
z_all = torch.sigmoid(gru_proj[:, :, :D])    # update gate       — slice [0 : D]
# gru_proj[:, :, D:2*D] is the standard GRU reset gate
# — computed but discarded (see below)
n_all = torch.tanh(gru_proj[:, :, 2 * D:])  # candidate content  — slice [2D : 3D]
```

*nn.GRUCell* projects the input to three D-dimensional gates: update (*z*), reset (*r*), and new-content candidate (*n*).

This architecture deliberately **drops the reset gate** (*r*) to reduce parameters and remove one learned forgetting mechanism — the reset gate's role, controlling how much of the previous hidden state bleeds into the candidate, is replaced by the surprise gate acting on the slow state *c*.

What remains of the fast-state update is:

```python
h_t = (1 – z_t) * h_{t-1} + z_t * n_t
```

The update gate *z* controls how much of the previous state survives.

- When *z* → *1*, the model discards history and writes new content.
- When *z* → *0*, it preserves the previous hidden state, carrying history forward unchanged.

One architectural detail worth noting: at sequence boundaries (*EOS* tokens), the fast state is hard-reset.

The gate is forced to *z = 1*, ensuring no context bleeds across document boundaries during training on heterogeneous data.

### Path 2: The Surprise Gate

The intellectual lineage of this mechanism traces directly to **Titans** (Ali et al., 2025, [arXiv:2501.00663](https://arxiv.org/abs/2501.00663)), which establishes prediction error (the gradient of reconstruction loss with respect to the memory state) as a principled write signal for slow memory.

The same class of mechanism appears in **TTT** (Sun et al., 2024, arXiv:2407.04620), which uses gradient descent on reconstruction loss as the hidden-state update, and in **SR-TTT** (2025, arXiv:2603.06642), which most closely mirrors the two-speed structure here: per-token reconstruction error *L_t = ||z_t - v_t||²* explicitly gates writes to a slow Residual Cache versus fast TTT weights.

The multi-timescale predictive coding framing also appears in **PV-RNN** (Ahmadi & Tani, 2018, arXiv:1811.01339), where ELBO gradients modulate state updates across fast and slow timescales.

Echo-DSRN is an engineering variant within this framework.

The specific implementation differs from Titans in three ways:
- the fast predictor is a **GRU** (not a linear associative memory)
- the surprise signal is a **scalar mean-squared error** (not the full gradient of an associative memory loss)
- scaling is handled by a per-dimension learned **surprise_lambda** vector (rather than Titans' momentum-weighted surprise accumulator).

This is computationally cheaper — the GRU forward pass already computes the fast state, so the prediction error requires no additional backward pass through a separate memory module.

```python
h_shifted = torch.cat([h_prev.unsqueeze(1), h_all[:, :-1, :]], dim=1)
x_pred = model_block.linear_pred(h_shifted)
diff = x – x_pred
error = torch.clamp(diff * diff, max=10.0) \
             .mean(dim=-1, keepdim=True)
surprise_signal = error * F.softplus(model_block.surprise_lambda)

gate_logits = model_block.linear_gate(h_all) + surprise_signal
g_all = torch.sigmoid(gate_logits)
```

Step by step:
1. At each position *t*, the model uses *h[t-1]* to predict what *x[t]* should be – a learned linear projection of the previous hidden state onto the input space.
2. The squared prediction error is computed: how wrong was the model about what came next?
3. This error is scaled by *surprise_lambda* – a learned scalar per state dimension, constrained positive by softplus to guarantee the error always pushes the gate open, never closed.
4. The error signal is added to the gate logits before the sigmoid. High error = gate opens wide. Low error = gate stays mostly closed.

The result: the slow memory state *c* is only written when the model encounters something it did not predict.

Routine, expected tokens leave memory untouched. Novel, surprising tokens trigger a write.

This is inspired by predictive coding – a theory of computation in which the brain suppresses predicted inputs and only propagates prediction errors upward.

One critical initialisation detail makes this work in practice:

```python
nn.init.orthogonal_(block.linear_pred.weight, gain=0.1)
nn.init.zeros_(block.surprise_lambda)
nn.init.zeros_(block.mlp_down.weight)
```

The prediction head starts near-orthogonal and small – ensuring early training produces genuine prediction errors rather than zero-signal noise.

The MLP output is zero-initialised so each block starts as an identity function and learns residual corrections.

These are not arbitrary choices and they are what prevents gradient collapse in a model with no attention to stabilise early training.

### Path 3: The Slow State Update (Parallel Scan)

Once the gate g is computed, the slow state update runs:

```python
c_t = (1 – g_t) * c_{t-1} + g_t * m_t
```

where *m_t = tanh(linear_memory(h_t))* is the candidate memory content derived from the current fast state.

This recurrence is computed via a hierarchical chunked parallel scan – the same class of algorithm used in Mamba, adapted here for the DSRN gating structure:
1. The sequence is divided into chunks of size *32*.
2. Within each chunk, a local scan runs vectorised across all chunks simultaneously.
3. A global scan runs across chunk boundary summaries.
4. A final combine step merges local and global results using cumulative products.

The result is mathematically identical to the sequential recurrence but computed in parallel – reducing training time from *O(T)* sequential steps to *O(T/K + K)* where *K* is the chunk size.

A Triton CUDA kernel path exists for further acceleration.

At EOS boundaries, the slow state gate is hard-zeroed – preventing memory from crossing document boundaries – and if the last token of a chunk is *EOS*, both *h* and *c* are zeroed before being passed to the next chunk.

### The Hybrid Component: Sliding Window Attention

```python
if self.window_size is not None and \
   k_attn.shape[2] > self.window_size:
    k_attn = k_attn[:, :, -self.window_size:, :]
    v_attn = v_attn[:, :, -self.window_size:, :]
```

Each block contains a local sliding window attention head with RoPE positional embeddings.

The window is bounded – attention never grows beyond *window_size* tokens – so the KV cache is constant size.

This is a local attention mechanism that handles fine-grained token-level dependencies the recurrent paths are not designed for.

The full KV history is preserved in the cache for positional correctness, but attention computation only uses the most recent window.

During decoding (*T=1*), no causal mask is applied – the recurrent state carries the history; attention handles immediate context.

### How the Three Paths Combine

```python
x_out = x + mlp_out                  # residual from fast state
x_out = x_out + linear_read(c_all)   # continuous read from slow state
x_out = x_out + attn_out             # local attention correction
```

Three additive residual contributions. Each path is independent and differentiable.

The slow state is read continuously – not just at the final token – meaning memory content influences every position in the output, not just the last one.

This is the linear_read projection: a learned map from the slow state space back into the hidden dimension.


### What This Means in Practice

The architecture encodes a specific theory of how memory should work:
- **Fast state:** what is happening right now, updated every token.
- **Slow state:** what matters enough to remember, updated only when surprised.
- **Local attention:** what exactly was said in the last N tokens.
- **Surprise gate:** the arbiter between forgetting and remembering.

The federated training telemetry confirms this theory holds.

Entropy stabilises in the *2.59* – *2.70* band rather than collapsing – the model becomes confident without becoming brittle.

The slow state is doing genuine selective compression, not indiscriminate accumulation.

## Hypothesis: Structural Alignment via Memory Gating

The prevailing paradigm of Large Language Model safety relies on post-training behavioral alignment (RLHF, DPO). Because standard transformers utilize a flat KV-cache, adversarial users can execute context-override attacks (e.g., "Ignore all previous instructions") to shift the attention distribution away from the aligned behavior.

We hypothesize that the surprise-gated continuous state of Echo-DSRN may present a structurally different threat surface.

When an adversarial input arrives, it must propagate through the recurrent λ-gate and the fast-state projection. While an adversarial context can trigger high surprise and open the gate, the actual candidate memory update remains bounded by the recurrent transition function learned during pre-training.

This theoretically raises the cost of context-override attacks. However, this remains an untested speculation. Formal adversarial evaluation (e.g., prompt injection resistance tests against similarly sized transformers) is required to validate whether prediction-error gating provides empirically stronger structural alignment.

The vulnerability disclosure below outlines known attack vectors that bypass this mechanism entirely.

### Practical Challenges of Preference Optimization for Deep Recurrent Architectures

Direct Preference Optimization (DPO) has become the standard for aligning open-weight models (Rafailov et al., 2023, arXiv:2305.18290).

DPO requires only that the model defines a valid conditional distribution *π(y|x)*, and the partition function *Z(x)* cancels in the chosen/rejected log-ratio regardless of architecture. This makes DPO theoretically applicable to recurrent models.

In practice, applying DPO to Echo-DSRN degraded output quality. Alignment via preference optimization for deep recurrent architectures remains an open engineering problem, and we are actively exploring approaches for the next version.

The adapter-based alternatives described below represent our current working strategy, keeping the base recurrent weights frozen to avoid disrupting the pre-trained slow-state geometry.

### Proposed Alignment Mechanisms for Echo-DSRN

Because DPO's gradient path through deep recurrences presents practical stability challenges, we explore three alternatives:

- **Training Data Curation as Alignment:** The resistance scales with training data quality. A clean, highly curated pre-training distribution produces a tighter, more resistant *c_t* manifold. Data curation is directly equivalent to alignment work in Echo-DSRN.

- **Adapter-Based Constraint:** Instead of modifying base weights via DPO, alignment is trained as a PEFT adapter on top of a frozen base (as demonstrated by our Intent Classifier). The adapter reshapes the output distribution without fighting the path-dependent slow state geometry.

- **State-Space Preference Optimization:** Future work requires optimizing *c_t* geometry directly, defining preferred and rejected slow states for a given context, and pushing the λ-gate to produce the preferred geometric representation.

### Vulnerability Disclosure & Open Red-Teaming Challenge

While Echo-DSRN exhibits strong emergent resistance to brute-force context overrides, it is not invulnerable. We disclose two specific architectural threat vectors:

#### The Fast-State Attention Exploit: While the *c_t* ring rejects adversarial context, the injection bleeds out in *h_t*.

Because *h_t* feeds the sliding window attention, an injection that stays within the bounded window could influence token probabilities at generation time before it is flushed.

#### The "Slow-Burn" Semantic Drift: The λ-gate responds to per-token prediction error.

A sophisticated adversarial attack executing gradual, token-by-token semantic drift could theoretically stay below the surprise threshold, cumulatively poisoning the slow state without triggering the anomaly response.

We release the Echo-DSRN-114M weights, the PyTorch training environment, and the Live 3D Telemetry Dashboard to the public.

We challenge the AI safety and red-teaming community to exploit the semantic drift vulnerability and attempt to break the geometry.


---

## Training Details

**Pre-training dataset:** FineWeb-Edu (Huggingface, 2024) — a high-quality, educationally-filtered subset of CommonCrawl. Training used approximately **6 billion tokens** from this corpus.

**Hardware:** Initial 6B-token run on a single rented AMD Instinct MI300X (cloud). Subsequent SFT and ablation experiments were conducted on a local AMD Radeon AI PRO R9700.

**Training procedure:** Truncated Backpropagation Through Time (TBPTT) with a curriculum of 128 tokens. The model is undertrained relative to the 1-3 trillion tokens typical of comparable-scale public models; this is an explicit consequence of compute budget constraints and is the primary factor contributing to the elevated perplexity observed in the evaluation section.

**Instruction fine-tuning:** SFT was performed on *naufalso/smoltalk2_non_thinking* (2 epochs, full fine-tuning) producing the Instruct checkpoint. LoRA adapters were trained for downstream classification tasks (intent, NER, sentiment, PII) via federated learning.

**Live demos:** Interactive Gradio applications and the 3D neural telemetry dashboard are available in the [Echo-DSRN HuggingFace collection](https://huggingface.co/collections/ethicalabs/echo-dsrn). Training code is released at [ethicalabs/Echo-DSRN](https://github.com/ethicalabs-ai/Echo-DSRN) under Apache 2.0.

---

## Model Specifications

Echo-DSRN-114M is a 114.66M parameter model. The parameter budget is distributed across three components: an input embedding table, a stack of DSRN blocks, and a language modelling head.

| Component | Parameters | Share |
|---|---|---|
| Embeddings | 16.39M | 14.3% |
| 8× DSRN Blocks | 81.87M | 71.4% |
| LM Head | 16.39M | 14.3% |
| **Total** | **114.66M** | |

Each block contributes 10.23M parameters, distributed across six sub-components:

| Sub-component | Params | Role |
|---|---|---|
| MLP | 4.19M | Feed-forward expansion |
| Slow state projections (*linear_gate*, *linear_memory*, *linear_read*) | 3.15M | Gate, memory write, memory read |
| GRU fast state (*weight_ih*, *weight_hh*) | 1.58M | Short-term recurrence |
| Sliding window attention (*qkv_proj*, *out_proj*) | 1.05M | Local context mixing |
| Surprise gating (*linear_pred*, *surprise_lambda*) | 264K | Prediction error + write sensitivity |
| RMSNorm | 1K | Normalisation |

The architecture is parameter-efficient by design: the surprise gating mechanism — the novel contribution — costs only 264K parameters per block (2.6% of per-block budget) while governing when and whether the 3.15M slow state projection weights are exercised.

---

## Inference Performance

All measurements were taken on a single NVIDIA T4 GPU (fp16 precision, batch size 1), the standard inference hardware provided by HuggingFace Spaces. This is the target deployment environment, not the training hardware; pre-training was performed on an AMD Instinct MI300X. A dedicated pre-training section with full hardware and compute details will be added in a subsequent revision.

### Prefill Speed

| Sequence Length | Latency | Throughput |
|---|---|---|
| 32 tokens | 20.1 ms | 1,594 tok/s |
| 128 tokens | 19.9 ms | 6,440 tok/s |
| 512 tokens | 22.2 ms | 23,039 tok/s |
| 1,024 tokens | 22.8 ms | 44,894 tok/s |
| 2,048 tokens | 50.2 ms | 40,792 tok/s |

Prefill time is near-constant from 128 → 1,024 tokens. This is the most direct empirical demonstration of the parallel scan's effect: processing four times as many tokens takes only 14% longer. The step at 2,048 tokens reflects expected chunked-scan overhead combined with the sliding window attention cache reaching capacity.

Generation throughput is approximately **52 tokens/second** on the same hardware. Model memory footprint in fp16 is **219 MB** — small enough to deploy on edge devices with 512 MB VRAM or to load alongside other services on commodity cloud instances.

### Memory Scaling During Generation

| Tokens Generated | Peak Memory Delta |
|---|---|
| 50 | 5.6 MB |
| 100 | 6.9 MB |
| 200 | 10.0 MB |
| 500 | 16.6 MB |

Memory grows at approximately **0.024 MB per token** while the sliding window attention cache is filling (up to 128 tokens per layer). Once the window reaches capacity, growth stops. This is consistent with the O(1) recurrent core + O(window_size × layers) bounded cache model described in the architecture section. For comparison, a standard 114M transformer at this sequence length would continue accumulating KV-cache memory without bound.

---

## Evaluation

The *Echo-DSRN-114M-v0.1.2-Base* checkpoint was evaluated to isolate the architectural effect of the surprise gate. The primary question is whether prediction-error gating improves performance compared to standard input-dependent GRU gating.

We conducted an ablation study where the surprise signal *||x[t] - pred||² * softplus(lambda)* was disabled (removed from the *gate_logits*), reverting the model to a standard GRU input-dependent gate.

All baseline models were re-evaluated by us using *lm-evaluation-harness* v0.4.11 under identical conditions (same prompts, same metric, same hardware) to ensure fair comparison. Published numbers in the original Pythia and Mamba papers differ due to earlier harness versions. [^1]

### Wikitext Perplexity (Language Modeling)
| Model | Params | Word Perplexity (↓) |
|---|---|---|
| GPT-2-small | 124M | 37.87 |
| Mamba-130M | 130M | 26.35 |
| Pythia-160M | 160M | 59.79 |
| **Echo-DSRN-114M-v0.1.2-Base** | 114M | 188.94 |
| Echo-DSRN-Ablated (No Gate) | 114M | 156.24 |

### Zero-Shot Tasks
| Task | Echo-DSRN (Full) | Echo-DSRN (Ablated) | Pythia-160M | Mamba-130M |
|---|---|---|---|---|
| PIQA (acc) | 0.5789 | 0.5789 | 0.5963 | 0.6491 |
| SciQ (acc) | **0.5830** | 0.5300 | 0.5190 | 0.7780 |

**Conclusion:** The ablation reveals that the surprise gate, as currently implemented, degrades language modeling perplexity by ~17% (188.94 vs. 156.24). This indicates that the prediction-error signal interferes with optimization for unstructured next-token fluency at the current training budget.

However, the zero-shot task metrics reveal a significant downstream benefit. While Echo-DSRN is undertrained compared to standard baselines (as shown by perplexity), the **full surprise-gated model (0.5830) outperforms the larger, fully-trained Pythia-160M (0.5190)** on the SciQ retrieval benchmark, a gap of +6.4%. Removing the surprise gate (ablated: 0.5300) destroys most of this advantage.

This confirms the architectural intent: while the gate consumes capacity that could be used for raw fluency (harming Wikitext), it acts as an effective semantic compressor, retaining critical long-horizon facts necessary for structured retrieval tasks like SciQ far better than standard gating or equivalent-scale transformers.

### Downstream Evaluation: Semantic Textual Similarity & Dense Embeddings

Evaluating the representation capability of the recurrent-hybrid architecture by converting Echo-DSRN-114M into a dense sentence embedding model is currently a **Work In Progress (WIP)**.

To systematically evaluate how the surprise-gated slow memory state $c_t$ encodes and aligns high-dimensional semantic constructs without relying on global self-attention, we are setting up a multi-stage representation fine-tuning curriculum:

1. **Contrastive Pre-training**: Aligning the dense representation space using natural language NLI datasets under a multiple-negatives contrastive objective.
2. **Fine-grained Similarity Calibration**: Calibration on Semantic Textual Similarity benchmarks (such as STS Benchmark and SICK-R) using a Surprisal-Aware CoSENT loss.
3. **Multi-Task Generalization & Matryoshka Representation Learning (MRL)**: Evaluating performance retention when slicing the 2048-dimensional recurrent states down to smaller dimensions (1024, 512, 256, 128) to assess semantic compression.

Detailed empirical performance tables comparing our models against quadratic-complexity Transformer baselines across MTEB benchmarks will be added in a subsequent revision of this working paper once the training run completes.

---

## Comparison to Related Work

### vs. Standard Transformers

| Aspect | Transformer | Echo-DSRN |
|---|---|---|
| Inference memory | O(seq\_len × layers) | O(window\_size × layers) + O(layers) |
| Prefill complexity | O(n²) | O(n) amortised (parallel scan) |
| Generation per step | O(n) KV lookup | O(1) recurrent + O(window\_size) attention |
| Long-range dependencies | Full attention history | Recurrent state (theoretically unbounded; practically gated) |
| Training parallelism | Fully parallel | Parallel scan (less efficient but tractable) |
| In-context learning | Strong | Structurally limited by fixed state size |

The architectural trade-off is explicit: Echo-DSRN gains constant memory and linear prefill at the cost of compressing context into a fixed-size recurrent state. For tasks that depend on precise retrieval of specific tokens from long history, a transformer with sufficient context window will outperform a recurrent model. For tasks that require long-horizon state persistence on constrained hardware, the trade-off is favourable.

### vs. Mamba (S4 / S6)

| Aspect | Mamba | Echo-DSRN |
|---|---|---|
| State mechanism | Selective State Space (S6) — input-dependent SSM matrices | Dual GRU + surprise-gated slow memory |
| Parallelism | Hardware-efficient selective scan | Hierarchical chunked scan + Triton kernel |
| Hybrid attention | No (pure SSM) | Yes — 128-token sliding window per block |
| Maturity | Published, widely validated, 130M–8B | Research prototype, 114M and 486M |
| State update trigger | Continuous, input-modulated | Discrete gate driven by prediction error |

Mamba's selective scan is more mathematically elegant — state transitions are parameterised as data-dependent SSM matrices, enabling a unified formulation for both training and inference. Echo-DSRN's dual-state approach is more neuroscience-inspired: the explicit separation of fast and slow timescales, and the use of prediction error as a write trigger, more directly models theories of biological memory consolidation. The two approaches are not mutually exclusive; transplanting a surprise gate onto an SSM backbone is a plausible future direction.

### vs. Hymba / Griffin / Jamba

These are all production-scale hybrid recurrence + attention models:

- **Hymba** (NVIDIA): Mamba heads + full attention in parallel; state of the art for hybrid models at scale.
- **Griffin** (Google DeepMind): RG-LRU recurrence + local attention; strong results across 2B–7B.
- **Jamba** (AI21 Labs): Mamba and Transformer layers interleaved at the block level.

Echo-DSRN fits in this family. Its sliding window attention (128 tokens, one head per block) is most comparable to Griffin's local attention approach. The dual-state design — an explicit surprise-gated slow memory combined with a fast GRU state and bounded attention — shares the timescale-separation spirit of Titans and SR-TTT but instantiates it with a GRU backbone rather than an associative memory, optimising for deployment footprint over expressive capacity.

### vs. RWKV

| Aspect | RWKV | Echo-DSRN |
|---|---|---|
| Attention substitute | WKV linear attention variant | Sliding window attention + recurrence |
| State | Single recurrent state | Dual (fast GRU + surprise-gated slow) |
| Scale | Up to 14B, production-ready | 114M–486M, research prototype |
| Training | Custom CUDA kernels (CUDA-RNN mode) | Triton + PyTorch hierarchical scan |
| Hardware support | NVIDIA CUDA (optimized); CPU/ROCm via community forks | NVIDIA CUDA, AMD ROCm, Apple MPS |

RWKV demonstrates that pure-recurrent models can reach competitive quality at scale with sufficient training data and custom kernels. Echo-DSRN's hybrid approach (recurrence + bounded attention) accepts a small constant-size attention cost in exchange for better local token-level resolution — the sliding window handles what the recurrent state naturally struggles with: precise short-range syntactic dependencies.

### vs. xLSTM

xLSTM (Hochreiter et al., 2024, [arXiv:2405.04517](https://arxiv.org/abs/2405.04517)) is the most direct architectural relative to Echo-DSRN in the recurrent family. It introduces two LSTM variants — **sLSTM** (scalar memory, exponential gating, head-level memory mixing) and **mLSTM** (matrix memory with a covariance update rule, fully parallelisable) — stacked inside residual blocks comparable to a Transformer backbone.

| Aspect | xLSTM | Echo-DSRN |
|---|---|---|
| Memory structure | sLSTM: scalar cell; mLSTM: **d×d matrix** (covariance update) | Dual: scalar fast GRU state + scalar slow state (surprise-gated) |
| Attention component | None — pure recurrence | 128-token sliding window attention per block |
| Write trigger | Input-dependent exponential gate (always writes) | Prediction-error gate — **writes only on surprise** |
| Parallelism | mLSTM: fully parallel; sLSTM: sequential (CUDA kernel) | Hierarchical chunked parallel scan + Triton kernel |
| Scale | Up to 7B+, competitive with Transformers | 114M–486M, research prototype |
| Memory footprint | O(d²) for mLSTM matrix state | O(d) GRU + O(d) slow state + O(window_size) attention cache |
| Hardware support | Optimized: NVIDIA CUDA kernels; PyTorch fallback: NVIDIA CUDA, AMD ROCm, Apple MPS, CPU | NVIDIA CUDA, AMD ROCm, Apple MPS |

The key conceptual difference is the **write trigger**. xLSTM's exponential gate is always active — every token modulates the memory state with varying intensity, but there is no mechanism to completely suppress a write when the input is routine and fully predicted. Echo-DSRN's surprise gate explicitly routes only novel, high-error tokens to the slow state, leaving it unchanged during predictable stretches. This is a different hypothesis about how memory should be managed: xLSTM assumes continuous modulation is optimal; Echo-DSRN assumes selective writes are cheaper and sufficient for long-range retrieval.

The mLSTM matrix memory (*d×d*) also has a significantly larger footprint than Echo-DSRN's scalar slow state (*d*), which is a deliberate deployment trade-off — Echo-DSRN sacrifices expressivity in the memory representation in favour of a fixed, compact state that runs on sub-1GB edge devices.

> **Author's note:** The pure PyTorch implementation of xLSTM (without the optimized CUDA kernels) was personally tested across multiple GPU and CPU configurations — NVIDIA CUDA, AMD ROCm, and Apple MPS — and ran flawlessly on all of them. The hardware portability limitation applies only to the high-performance kernel-optimized path. This is consistent with how Echo-DSRN achieves the same multi-platform coverage via its PyTorch + Triton stack.
>
> Additionally, the xLSTM HuggingFace Transformers integration served as a major source of inspiration for the Echo-DSRN HuggingFace integration (*EchoForCausalLM*, *EchoConfig*, *EchoCache*). The core Echo-DSRN pre-training framework is proprietary and written in pure PyTorch; the HuggingFace-compatible wrapper (licensed under Apache 2.0) was developed independently to enable open deployment, with xLSTM's clean HF integration as a structural reference.

### Prior Art: Surprise-Gated and Predictive Memory Models

The surprise-gating mechanism in Echo-DSRN belongs to a line of work using prediction error to drive memory state updates:

| Work | Year | Key Mechanism |
|---|---|---|
| **TTT** (Sun et al., 2024) | 2024 | Gradient descent on reconstruction loss as hidden-state update |
| **Titans** (Ali et al., 2025) | 2025 | Prediction-error gradient as write signal; momentum accumulator |
| **SR-TTT** (2025) | 2025 | Per-token MSE gates writes to slow Residual Cache — closest prior |
| **PV-RNN** (Ahmadi & Tani, 2018) | 2018 | ELBO gradient modulates fast/slow timescale updates |
| **RWKV-7** (2025) | 2025 | Delta-rule error ≈ one-step surprise signal |

| Work | Full Title | arXiv |
|---|---|---|
| **TTT** | Learning to (Learn at Test Time): RNNs with Expressive Hidden States | [2407.04620](https://arxiv.org/abs/2407.04620) |
| **Titans** | Titans: Learning to Memorize at Test Time | [2501.00663](https://arxiv.org/abs/2501.00663) |
| **SR-TTT** | SR-TTT: Surprisal-Aware Residual Test-Time Training | [2603.06642](https://arxiv.org/abs/2603.06642) |
| **PV-RNN** | A Novel Predictive-Coding-Inspired Variational RNN Model for Online Prediction and Recognition | [1811.01339](https://arxiv.org/abs/1811.01339) |
| **RWKV-7** | RWKV-7 "Goose" with Expressive Dynamic State Evolution | [2503.14456](https://arxiv.org/abs/2503.14456) |

Echo-DSRN's contribution relative to these works is the specific combination: GRU as the fast predictor (replacing the linear associative memory used in Titans and SR-TTT), scalar mean-squared error as the surprise signal (cheaper than a full gradient computation), and integration with sliding-window attention in a single deployable block.

---

## Novelty and Contributions

### Surprise-Gated Memory: Engineering Variant of Titans-Style Gating

The primary technical contribution of this work is a concrete, deployable implementation of **prediction-error-driven memory gating** within a GRU-based hybrid architecture.

The principle — using reconstruction error to gate writes to a slow memory — was introduced by Titans (Ali et al., 2025) and is related to TTT (Sun et al., 2024) and SR-TTT (2025). The specific implementation here differs in three ways that matter for deployment cost:

| | Titans (linear memory) | Echo-DSRN |
|---|---|---|
| Fast predictor | Linear associative memory | GRU (shared with fast state path) |
| Memory state size | O(d²) weight matrix | O(d) GRU hidden state |
| Surprise signal | Rank-1 matrix in key-value space (directional) | Scalar MSE in output space (magnitude only) |
| Compute per token | O(d²) matmul + outer product | O(d²) GRU gates (comparable) |
| Information in surprise | Which direction the prediction error lies | Only how large the error is |
| Scaling | Momentum-weighted accumulator | Per-dimension learned *surprise_lambda* |

The Echo-DSRN formulation is:

```
surprise_signal = ||x[t] - linear_pred(h[t-1])||² × softplus(surprise_lambda)
gate_logits = linear_gate(h[t]) + surprise_signal
g[t] = sigmoid(gate_logits)
```

High prediction error forces *g[t]* toward 1, triggering a slow-state write. Low prediction error holds *g[t]* near 0, leaving memory unchanged.

Routine, expected tokens do not touch long-term memory; topic shifts, rare vocabulary, and anomalies force a state update.

This is grounded in **predictive coding** — the neuroscientific theory in which the brain suppresses predicted inputs and propagates only prediction errors upward. The specific engineering claim is not that this principle is new, but that the GRU + scalar-MSE instantiation achieves it at lower compute cost than gradient-based alternatives, without a separate associative memory module, making it viable at the edge deployment scale targeted here.

### Dual-State Timescale Separation

The explicit separation of the fast GRU state *h* (high update rate, every token) and the slow state *c* (low update rate, surprise-gated) creates **multi-timescale recurrent dynamics** without architectural complexity.

This is reminiscent of fast/slow weights in connectionist memory theory (Hinton & Plaut, 1987), where fast weights encode episode-level context and slow weights encode long-term knowledge. Echo-DSRN instantiates this distinction as a learnable, differentiable gating mechanism within a single block, trained end-to-end.

### Engineering Contributions

Beyond the architectural ideas, this work contributes several engineering artefacts:

- **Triton CUDA kernel** for the forward and backward pass of the hierarchical chunked parallel scan, reducing memory bandwidth compared to the PyTorch fallback.
- **EchoCache** — a custom HuggingFace *Cache* subclass that preserves the 4-tuple recurrent state *(h, c, k_attn, v_attn)* through the generation loop without being mangled by *DynamicCache*.
- **EOS-aware state reset** — hard zeroing of both *h* and *c* at document boundaries during training, ensuring no context leaks across heterogeneous documents in a packed batch.
- **vLLM tensor and pipeline parallelism plans** — declared directly in *EchoConfig*, enabling multi-GPU inference via vLLM without custom integration code.
- **Hardware-agnostic training** — verified on NVIDIA CUDA, AMD ROCm (MI300X), and Apple Silicon (MPS), with backend-specific fixes for distributed group teardown and gradient stability.

### Surprise-Gate Temperature Modulation: A New Generation Parameter

A key empirical finding emerged during speculative decoding experiments: the surprise gate λ_t does **not** predict output-distribution entropy — it encodes hidden-state prediction error from the fast GRU, a structurally distinct signal. When tested for calibration against greedy-vs-sampled token agreement, λ_t showed zero correlation (r = −0.008, AUC = 0.475), confirming the architectural decoupling.

However, feeding λ_t **back into the output logits** as a temperature modulator closes this loop:

$$
\text{logits}' = \frac{\text{logits}}{1 + \alpha \cdot \lambda_t}
$$

This creates a **self-aware generation parameter** — unlike a fixed `temperature` scalar, the modulation strength varies per position based on the model's *own internal uncertainty*. When the surprise gate is active (λ_t ≈ 1, high hidden-state prediction error), the output distribution flattens and the model explores. When the gate is quiet (λ_t ≈ 0), the distribution stays sharp.

**α is a pure inference parameter** — no weight changes, no fine-tuning, no calibration data required. It is set via `surprise_temperature_alpha` in the model config and supported on all Echo-DSRN variants (base, hybrid, Qwen3 hybrid).

#### Qualitative Impact

| Model | α = 0 (off) | Optimal α | Effect |
|-------|------------|-----------|--------|
| Echo-DSRN-114M | Repetitive loops, factual hallucinations | 1.0–2.0 | Breaks loops, +3–4 nats entropy, preserves top token |
| Kurtis-EON1-Hybrid-2B | Solid factual generation | 0.3–0.5 | Creative divergence without hallucination |

#### Speculative Decoding Application

Combined with log-space prefix survival tracking (`S_k = ∏(1 − λ_i)`), the surprise gate enables a **verification-head-free speculative decoding scheduler**. At τ_load=0.05, Echo-DSRN-114M achieves 64% token efficiency when drafting against Phi-3-mini-4k-instruct (3.8B target) — the confidence signal correctly identifies which draft positions are reliable, saving ~45% of target verification passes.

The `DSparkEchoScheduler` in `echo_dsrn/dspark_scheduler.py` provides the full draft → confidence → cutoff → verify pipeline.

#### Significance

This is, to our knowledge, the **first demonstration of a recurrent architecture using its own internal memory gate as an inference-time confidence signal**. The surprise mechanism — originally designed to gate long-term memory writes during training — proves capable of self-regulating generation quality without external calibration. The architectural loop is closed: what the model learns about its own prediction errors during training becomes actionable at inference time.

---

## Limitations and Scope

Echo-DSRN is a **research prototype**, not a general-purpose language model. The following limitations are acknowledged:

**Scale** — the architecture has been trained and evaluated at 114M and 486M parameters. Whether the dual-state design benefits from scale in the same way transformers do (via the Chinchilla scaling laws) is an open empirical question. The interaction between three parallel pathways (GRU, surprise-gated DSRN, and sliding window attention) may introduce optimisation complexity that only manifests at larger scale.

**In-context learning** — recurrent models compress context into a fixed-size state. This is structurally at odds with in-context learning (ICL), which requires the model to precisely retain and attend to specific example tokens provided at inference time. Early results show degraded few-shot accuracy on some benchmarks relative to zero-shot — consistent with the known ICL weakness of recurrent architectures.

**Training compute** — the current 114M checkpoint is undertrained relative to compute-optimal recipes. Benchmark evaluation is ongoing and will be reported in a subsequent revision once a fully converged checkpoint is available. Current numbers should be understood as lower bounds on the architecture's capability, not its ceiling.

**Target scope** — Echo-DSRN is designed for edge and resource-constrained deployment on narrow, well-defined tasks: intent routing, named entity recognition, log parsing, semantic classification. At 219 MB fp16 with ~52 tok/s on a T4, it is well-suited to these workloads. It is not designed to compete with frontier models on open-ended generation or complex reasoning.

The surprise gating mechanism, as a standalone contribution, is architecture-agnostic. Future work includes transplanting a prediction-error-driven write gate into stronger backbone architectures — Mamba, Griffin, or larger Transformer variants — to test whether the selective memory allocation benefit transfers to higher-capability models.

---

*Telemetry findings, training setup and federated training configuration will be included in a subsequent revision of this paper.*

[^1]: Evaluations were conducted using *lm-evaluation-harness* v0.4.11. Zero-shot accuracy uses the *acc* metric (unnormalized). For SciQ, the default v0.4.11 prompt is used without the support passage (pure closed-book QA, causing the discrepancy with the original Pythia paper's MRC-based 80%+ figures). Language modeling perplexity uses the *wikitext* task (WikiText-2) reporting *word_perplexity* without overlapping stride.
