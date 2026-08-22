# Changelog

All notable changes to Echo-DSRN-HF are documented here.

---

## [0.1.10] — 2026-08-22

### Added

- **vLLM pooling-runner serving for the embedding models.** New
  `echo-embed` compose service (port 8003) serves `Echo-DSRN-v0.1.3-Embed-Exp`
  through vLLM 0.27.1's pooling runner (`--convert embed`), replacing the
  SentenceTransformer-based service.
- **`EchoModelForPooling`.** A pooling adapter whose module tree is identical
  to `EchoModel` — keeping `model.blocks.*` checkpoint keys resolvable through
  vLLM's weight mapper — while its forward returns per-sequence pooled
  embeddings broadcast to every token. `from_embedding()` now routes
  `auto_map["AutoModel"]` to it, unblocking embed-backbone classifiers
  (`Echo-DSRN-v0.1.4-Embed-Intent-CLF`) on vLLM.
- **vLLM backend registration.** The embedding/classification classes expose
  `_supports_attention_backend`, clearing vLLM's architecture validation.

### Fixed

- **Batched embeddings collided under vLLM.** The Transformers backend runs
  each step as one flattened `[1, N]` forward with `position_ids` restarting
  per sequence and no `attention_mask`, so the DSRN pooled `c_all` over all
  sequences — every item in a batch received the same vector. Segment
  boundaries are now detected from the `position_ids` resets and each segment
  runs as its own forward with fresh state.
- **Padding polluted the non-causal attention on the plain/ST path.** The
  bidirectional (`non_causal_window`) attention now blocks pad key-positions
  from the input `attention_mask` (a no-op under vLLM, which never passes
  one), the embedding adapters forward `attention_mask` into the base model,
  and the embed-family tokenizers right-pad (`padding_side: right`) so the
  recurrence no longer processes pad tokens before the text. Padded ST
  batches now match single requests exactly.
- **DSRN attention adapted to vLLM 0.27 module-first dispatch** (`_attn_call`),
  and the default hybrid service pinned to GPU 0.

### Changed

- **Dependency floors raised for the vLLM 0.27.1 serving stack:**
  `transformers>=5.15.0` (was 5.14.1) and `torch>=2.13.0` (was 2.10.0, both
  CPU and ROCm extras).
- **Port 8002 default intent classifier is now the embed-backbone
  `Echo-DSRN-v0.1.4-Embed-Intent-CLF`** (compose default + README), served via
  the pooling runner; the LM-based `Echo-DSRN-v0.1.3-Intent-CLF` remains
  available as a fallback.
- Registry sync keeps sentence-transformers repos out (their modules are
  pushed manually).

---

## [0.1.9] — 2026-08-21

### Fixed

- **`surprise_lambda_init` is now honored by the HF wrapper.** Previously the
  config key was ignored and `surprise_lambda` was hardcoded to zero at
  initialization; the model now initializes it to `config.surprise_lambda_init`
  (default `0.0`, preserving prior behavior). This makes the trainer CLI and the
  HF wrapper agree on one source of truth, which is load-bearing for pre-training
  convergence (see the gate-bias/surprise-lambda init sweep in the Echo-DSRN
  pre-training protocol).
- **`eos_mask` now threads through the pure-DSRN forward path** (previously only
  the hybrid path wired it). The parallel-scan kernels wipe the fast state and
  suppress slow-state writes at document boundaries, and zero the inter-chunk
  carry when a chunk ends on EOS — preventing recurrent state from leaking across
  documents in TBPTT pre-training (`--mask_eos` now behaves as documented in
  `docs/pretrain_echo_tiny.md`).

---

## [0.1.3a] — 2026-05-17

### Fixed

- **Backward compatibility for v0.1.2 checkpoints (`mlp_bias=False` default)**
  - Added `mlp_bias: bool = False` to `EchoConfig`. The default `False` ensures all
    existing v0.1.2 checkpoints load cleanly without any `mlp_up.bias` /
    `mlp_down.bias` tensors being randomly initialized (which caused `NaN`/`+Inf`
    overflows in `bfloat16`).
  - `DSRNBlock` now respects `config.mlp_bias` when constructing `mlp_up` and
    `mlp_down`, using a `getattr(..., False)` guard for configs loaded from old JSON
    pre-dating the field.
  - `EchoForCausalLM` gains `_keys_to_ignore_on_load_missing` to suppress HuggingFace
    load warnings for missing bias keys when `mlp_bias=False`.
  - `EchoForCausalLM.from_pretrained` adds a defense-in-depth zero-out pass: if any
    MLP bias tensors were accidentally initialized despite `mlp_bias=False`, they are
    zeroed and a `UserWarning` is emitted.
  - Fixed unconditional `nn.init.zeros_(block.mlp_down.bias)` in `EchoModel.__init__`
    that raised `AttributeError` when `mlp_bias=False`.

### Migration

| Checkpoint | Package | `mlp_bias` config | Result |
|---|---|---|---|
| v0.1.2 | 0.1.3+ | `False` (default) | ✅ Loads cleanly, no bias |
| v0.1.2 | 0.1.3+ | `True` | ⚠️ Biases missing → zeroed with warning |
| v0.1.3+ | 0.1.3+ | `True` | ✅ Full bias as trained |
| v0.1.3+ | 0.1.2 | N/A | ❌ Package too old, upgrade required |

No action required for v0.1.2 → v0.1.3 migration. The default `mlp_bias=False`
handles it automatically.
