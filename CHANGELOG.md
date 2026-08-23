# Changelog

All notable changes to Echo-DSRN-HF are documented here.

---

## [Unreleased]

---

## [0.1.11] — 2026-08-23

### Added

- **Cross-vocabulary (TLI) speculative decoding.** `DSparkEchoScheduler` can
  now draft against a target model with a *different* tokenizer (e.g. Echo-DSRN
  32k → Qwen 152k/248k). New module `echo_dsrn/speculative/vocab_mapper.py`
  builds the string-normalized token-level intersection `I`, restricts draft
  logits to `I`, and translates draft↔target ids. The scheduler records
  per-step draft states and `rollback(n_accepted)` restores the exact cache at
  the accepted prefix; `step()` maintains draft and target KV-cache invariants
  for a lossless generation loop (the target's own token is emitted at the
  first rejection). Losslessness verified end-to-end against real Qwen
  targets.
- **`scripts/benchmark_cross_speculative.py`.** CLI benchmark of tokens/sec
  and leading-run acceptance for Echo-DSRN → Qwen, including a cached-greedy
  vanilla reference for speedup comparison. New `--target-device-map auto`
  (split large targets across GPUs via accelerate) and `--target-quant 4bit`
  (bitsandbytes NF4) flags for targets that do not fit in VRAM unquantized.

### Known limitations (cross-vocabulary TLI)

- **Leading-run acceptance is the binding constraint, not raw agreement.**
  Stateless per-position agreement measures 25–50% (Qwen2.5-0.5B 25%, Phi-4
  50%), but loop leading-run acceptance is ~1.5% for every cross-vocabulary
  target measured (Qwen2.5-0.5B, Phi-4 14B, Qwen3.8-27B). Token-level
  speculation only pays off when the draft's per-position agreement is
  high enough that *runs* survive; 25–50% raw converts to <2% leading.
- **Speedup is < 1 on all measured targets** (0.13×–0.34×): cross-vocabulary
  speculation is currently a net loss versus greedy decoding, including the
  27B scenario (0.17×) that was hoped to flip the economics.
- **Phi-4 does not share the draft tokenizer.** Despite the assumption in
  early planning, `microsoft/Phi-4` uses a 100,352-vocab tokenizer (the
  draft is 32,017); only `Phi-3.5-mini-instruct` / `Phi-3-mini-4k-instruct`
  are genuinely same-vocabulary (verified by identical encoding). Phi-4 must
  be treated as cross-vocabulary.
- **Hybrid targets lose target-side cache reuse.** Targets with
  linear-attention layers (Qwen3.8's gated delta net) cannot have their KV
  cache truncated to a shorter prefix; the scheduler falls back to a full
  re-prefix every round, which is lossless but adds per-round target cost.

### Fixed

- **Cross-vocab exact-string verification keys were compared across
  unrelated per-tokenizer numberings.** Exact-string keys were assigned by
  first-appearance order within each tokenizer, so the same token string
  (e.g. " th") received different key values in the draft and target tables.
  `matches_draft_to_target()` therefore rejected nearly every proposal —
  measured acceptance collapsed to 0% for all cross-vocabulary targets
  (14,245 of 14,248 shared tokens mismatched) even where raw-ID agreement
  was 50%. Keys are now assigned from a single shared string table spanning
  both vocabularies. `mask_logits()` also raises a clear error when the
  mapper was sized from the tokenizer rather than the LM head.
- **Target cache truncation crashed on hybrid (linear-attention) targets.**
  `_truncate_kv_cache()` unpacked every cache layer as `(k, v)`, which
  fails on Qwen3.8's gated-delta-net layers (`LinearAttentionLayer`).
  Non-truncatable caches (layers carrying conv/recurrent state) are now
  detected via `_cache_is_truncatable()` and fall back to a full re-prefix
  next round — always lossless, verified against the real 64-layer Qwen3.8
  cache (48 linear + 16 full-attention layers).

- **Cached continuation dropped the recurrent state for pure-DSRN models.**
  `EchoModel` treated any cache whose `get_seq_length() == 0` as empty, which
  discarded the `(h, c)` state of 2-tuple (attention-less) caches on every
  cached forward. Non-empty `EchoCache` objects are now never treated as
  empty.
- **Cached XLSTM continuation dropped its state (pending — ships with the
  upcoming `echo_xlstm` module release).** `XLSTMCache`'s 3-tuple `(h, C, n)`
  states report `get_seq_length() == 0`, so `XLSTMForCausalLM` re-initialized
  the recurrent + matrix-memory state to zeros on every cached forward — any
  stepwise/streaming generation of an attention-less XLSTM silently lost all
  context after the prefill. Non-empty caches are now never treated as empty.
  Affects `use_hybrid_attention=False` XLSTM configs only (e.g.
  Echo-XLSTM-0.7B); one-shot inference and hybrid XLSTMs are unaffected.

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
  one) and the embedding adapters forward `attention_mask` into the base
  model. The embed-family tokenizers stay **left-padded** to reproduce the
  published MTEB benchmarks (card parity); a right-padded training version is
  planned, which keeps padded batches consistent with single-request
  embeddings.
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
