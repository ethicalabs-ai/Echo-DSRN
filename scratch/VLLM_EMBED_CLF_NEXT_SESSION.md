# NEXT SESSION: Serve Echo-DSRN-v0.1.4-Embed-Intent-CLF on vLLM

> Handoff prompt. Read fully before touching anything. All commands assume
> cwd = `/home/ethicalabs/Workspace/Echo-DSRN-HF` and `uv run --extra rocm`.
> Do not commit anything on `main`; branch `chores/vllm-support` is the
> working branch (PR merge is the user's call).

## Goal

Make the embed-backbone intent classifier **`ethicalabs/Echo-DSRN-v0.1.4-Embed-Intent-CLF`**
serve through **vLLM 0.27.1** (ROCm), replacing the LM-based v0.1.3 classifier in the
serving stack (port 8002) if it works. It currently serves fine via plain
transformers/`pipeline()` but **vLLM refuses it at weight-loading**.

Fallback rule from the user: **a working classifier is what matters** — the v0.1.3
stays the served classifier until the v0.1.4 actually works on vLLM. Never break ST.

## Current state (verified 2026-08-22)

Stack (all healthy, serving):
- 8001 chat: mrs83/Kurtis-EON1-Hybrid-2B-v0.1.2 — vLLM 0.27.1 ✅
- 8002 classify: ethicalabs/Echo-DSRN-v0.1.3-Intent-CLF — vLLM 0.27.1 ✅
- 8003 embed: ethicalabs/Echo-DSRN-v0.1.3-Embed-Exp — SentenceTransformer ✅

Branch `chores/vllm-support` (pushed) contains:
- `607bbf9` — DSRN attention adapted to vLLM 0.27 module-first dispatch (`_attn_call`)
- `bacdf0d` — transformers 5.15, torch 2.13 floors
- `38f2281`/`d73b8b3`/`1c3b3af` — echo-embed service (ST-based, vllm-blocked)
- `296ecf5` — **`_supports_attention_backend = True` on EchoForSequenceClassification
  and EchoModelForSentenceEmbedding** (this cleared vLLM's arch validation) + registry
  entries for the 3 embed models + README + version 0.1.10

All 12 registry models resynced with `296ecf5` modules (load-tested on plain python).
Tests: 164 passed.

## The blocker (reproduced exhaustively)

`vllm serve` with ANY of: `--runner pooling --convert classify`, no-convert,
`--convert embed`, pooler LAST/MEAN, gpu-mem 0.3/0.5 — fails at engine init:

```
ValueError: There is no module or parameter named 'blocks' in
TransformersForSequenceClassification. The available parameters belonging to
(TransformersForSequenceClassification) are: {'model.model.blocks...', ...}
```

### What was established by bisection

1. vLLM resolves BOTH classifiers to its `TransformersForSequenceClassification`
   wrapper (arch `EchoForSequenceClassification` is normalized there).
2. vLLM's fallback `_try_resolve_transformers` (registry.py ~1148) requires
   `model_module.is_backend_compatible()` == True — that was the v0.1.4 blocker,
   fixed by `_supports_attention_backend`. ✅ done.
3. Checkpoint key sets of v0.1.3 and v0.1.4 are **byte-identical** (140 keys,
   `model.blocks.*`, `classifier.*`). Module trees identical
   (`model.blocks/embedding/final_norm` + `classifier`).
4. The failure tracks **`pooling_mode == "mean_c_all"`** in the loaded config:
   - config with `pooling_mode: c_T` (or absent → class default) → NO 'blocks' error
     (fails later on head-shape: `[60,2048] vs [60,512]` — because head_dim comes
     from pooling_mode)
   - config with `pooling_mode: mean_c_all` → 'blocks' error
   - renaming the config KEY (JSON + EchoConfig dataclass field → `echo_pooling_mode`)
     did NOT help → the value `mean_c_all` is what vLLM keys on, or the model's
     runtime structure with mean_c_all differs (head 2048 + pooled [B,1,D] output).
5. `pooling_mode` is load-bearing for the head: `head_dim = state_dim (2048)` iff
   `mean_c_all`, else `embed_dim (512)`. Removing it breaks weight loading.

### vLLM internals mapped (0.27.1, in the `echo-dsrn-hf-echo-hybrid` image)

- `vllm/model_executor/models/registry.py`
  - `_try_resolve_transformers` (~1148): auto_map AutoModel class must pass
    `is_backend_compatible()` (transformers `_supports_attention_backend`).
  - `inspect_model_cls` (~1249) → `_normalize_arch` → `TransformersForSequenceClassification`.
  - `_raise_for_unsupported` (~1103): the arch rejection (fixed by the flag).
- `vllm/model_executor/models/transformers/__init__.py`:
  `TransformersForSequenceClassification(SequenceClassificationMixin, LegacyMixin, Base)`.
- `vllm/model_executor/models/utils.py` (~355-395): the recursive weight mapper
  that raises the 'blocks' error (`_load_module`/`_load_param`; the failing prefix
  is a key whose module child doesn't exist at the level the mapper expects).

## Hypotheses for next session (in order of effort)

1. **Why does the v0.1.3 (identical keys) load while the v0.1.4 doesn't?**
   Instrument the mapper: wrap `TransformersForSequenceClassification.load_weights`
   or monkeypatch `utils.py` `_load_module` to log the (prefix, child_modules)
   pairs for both models. The v0.1.3's keys must walk the tree fine; find where the
   v0.1.4's walk diverges (likely the wrapper's `model` attribute resolves to a
   different module tree when `mean_c_all` — e.g. the embed adapter's `projection`
   module or the pooled-output broadcast changes `named_parameters`).
2. **`SequenceClassificationMixin` convert path**: read the mixin + the `--convert
   classify` adapter; check whether `mean_c_all`-configured models take a
   pooling-adapted branch that expects `model.blocks` at the top level.
3. **Checkpoint-side fix**: re-export the v0.1.4 through the v0.1.3-style structure
   (`from_causal_lm` path or a re-save that inlines the backbone so the wrapper's
   `model.blocks` resolves) — if the v0.1.3's serving loads because its wrapper
   tree is `model.blocks` (not `model.model.blocks`), aligning the v0.1.4's save
   format may be the minimal fix.
4. **vLLM-side**: register the Echo arch as a proper vLLM model (out of scope for a
   PR; note as a longer-term option).

## Repro commands

```bash
docker run -d --name clf14-test --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render --ipc host -e ROCR_VISIBLE_DEVICES=1 \
  -v /tmp/clf14-fix:/models/clf14-fix:ro -p 8005:8005 \
  echo-dsrn-hf-echo-hybrid \
  --model /models/clf14-fix --served-model-name echo-intent \
  --runner pooling --convert classify --trust-remote-code \
  --dtype bfloat16 --gpu-memory-utilization 0.3 \
  --pooler-config '{"pooling_type": "LAST"}' --max-model-len 2048 \
  --enforce-eager --port 8005
# watch: docker logs -f clf14-test
```

`/tmp/clf14-fix` = full v0.1.4 download + locally patched modules (relative
imports). Rebuild it from HF if stale. Container `echo-dsrn-hf-echo-hybrid` is the
vLLM 0.27.1 ROCm image (from the repo Dockerfile).

## Constraints

- v0.1.3 stays served on 8002 until the v0.1.4 works.
- SentenceTransformer path must keep working (embed service on 8003).
- Model-code changes → branch `chores/vllm-support` + PR (OSS protocol; tests,
  ruff, pre-commit). Never direct to `main`.
- After any model-code change: run `pytest tests/` (164 expected) and resync the
  registry models (`uv run --extra rocm python scripts/sync_hf_modules.py`).
- The v0.1.4's config `pooling_mode: mean_c_all` is load-bearing (head dim); keep
  semantics if you touch it.
- HF embed models' modules live under the same sync flow now (they're in the
  registry); the Embed-Exp/Intent ST repos have no `__init__.py` — don't add one.

## Success criteria

- v0.1.4-Embed-Intent-CLF serves via vLLM 0.27.1, `/classify` returns correct
  MASSIVE labels (weather/timer/light/flight spot checks).
- The v0.1.3-vs-v0.1.4 swap in `.env` (`INTENT_MODEL`) + compose default is
  justified and documented.
- Tests green, ST intact, all models synced.

## UPDATE 2026-08-22 (after this file was written)

**The embed model now serves on vLLM** (`--runner pooling --convert embed`,
port 8003, compose `echo-embed` service). Single-request embeddings verified:
0.9969 cosine vs the SentenceTransformer path; cross-intent sim 0.0461 vs
0.0617 ST. The `_supports_attention_backend` fix cleared validation.

**New known bug — batched embeddings collide.** `/v1/embeddings` with >1 input
returns colliding vectors (pairwise sims identical or 1.0 depending on order);
single-input requests are correct. Cause hypothesis: the DSRN recurrent state
(surprise-gated slow state / c_all) is shared or mis-indexed across batch items
under vLLM's pooling runner — investigate the EchoCache/state handling with
B>1 in the pooling path (the model was effectively only exercised at B=1).
Repro: `curl -s localhost:8003/v1/embeddings -d '{"input":["A","B"]}'` — sim(A,B)
should differ from single-request sim(A,B).

Also: serve_embed.py + Dockerfile.serve-embed (the ST service) were removed
from the repo; the files are untracked on disk as reference.

## UPDATE 2026-08-22 (session 2) — EMBED BATCH BUG FIXED; CLASSIFIER LOADER BLOCKER SOLVED

### 1. Batched-embeddings collision — ROOT CAUSE + FIX (shipped)

**Root cause:** vLLM's Transformers backend concatenates every sequence of a
step into ONE `[1, N]` forward with `position_ids` restarting at each sequence
start and NO `attention_mask`. `EchoModelForSentenceEmbedding` pooled
`c_all.mean(dim=1)` over ALL N tokens → every sequence in the forward got the
same global-mean vector. (The DSRN recurrence cannot reset mid-forward — the
kernel's eos_mask handling freezes, it does not reset.)

**Fix (in `echo_embedding/modeling_embedding.py` + `echo_dsrn/modeling_echo.py`):**
- `_flattened_segment_mask(position_ids, input_ids, attention_mask)` detects the
  flattened multi-sequence mode (`position_ids == 0` boundaries).
- `_forward_flattened_segments` runs each segment as its own forward with fresh
  state (exact single-request semantics), then stitches the broadcast pooled
  vectors back into `[1, N, D]`. vLLM's per-sequence pooler (LAST/CLS) then
  picks the right vector per sequence.
- Pooling logic extracted to `_pool_hidden_states()` (shared with the new
  `EchoModelForPooling`).
- Regression tests: `tests/test_embedding_vllm_batching.py` (3 tests).

**Verified live:** 3- and 5-item batches in both orders → every batch vector
matches its single-request vector exactly (sim 1.0000); pairwise batch sims
equal single sims. 167 tests pass; ruff clean.

### 2. v0.1.4 classifier on vLLM — 'blocks' loader blocker SOLVED; quality gap remains

**Root cause of the 'blocks' error (finally):** the vLLM `TransformersForSequenceClassification`
wrapper's weight mapper strips `model.` from `model.blocks.*` checkpoint keys
UNLESS the inner model (instantiated via `auto_map["AutoModel"]`) has `blocks`
as a DIRECT child. The v0.1.3 works because its `AutoModel → EchoModel`
(children: embedding|blocks|final_norm). The v0.1.4's `AutoModel →
EchoForSequenceClassification` (children: model|classifier|dropout) → keys
stripped to `blocks.*` → mapper error. `AutoModel → EchoModelForSentenceEmbedding`
also fails (children: model). The embed service works because its wrapper
(`TransformersForEmbedding`) somehow feeds pre-doubled `model.model.blocks.*`
keys (mechanism never fully explained; empirically verified).

**Fix:**
- New `EchoModelForPooling(EchoModel)` in modeling_echo.py — tree identical to
  EchoModel (blocks as direct children → keys survive the mapper) but forward
  returns per-sequence pooled embeddings broadcast to every token (mean_c_all,
  segment-aware, like the embed adapter). `_supports_attention_backend = True`.
- `from_embedding()` now sets `auto_map["AutoModel"] =
  "modeling_echo.EchoModelForPooling"`.
- NOTE: an attempted `SlidingWindowAttention.is_causal` mirror of
  `attention_masking` was REVERTED (commit 7b249f7) — it makes vLLM treat the
  model as encoder-only and the HybridKVCacheCoordinator asserts (needs >=2
  attention groups). The attention itself is dispatched via transformers'
  sdpa path regardless (the lookup key is "sdpa", not vLLM's "vllm" registry
  entry), so the flag only affected cache setup.
- The v0.1.4's HF config needs the auto_map update (`AutoModel →
  modeling_echo.EchoModelForPooling`) — re-export via from_embedding() or edit
  config.json on the hub. NOT YET PUSHED.

**Result:** the v0.1.4 loads and serves on vLLM 0.27.1 (`--convert classify`,
port 8005 test). weather/timer correct (weather_query, alarm_set). BUT
borderline intents (lights/flight) are wrong on BOTH the vLLM path AND plain
transformers (lights → iot_wemo_on/general_quirky instead of iot_hue_lighton;
flight → qa_factoid instead of transport_ticket) — the fine-tuned head's own
quality is marginal. There is also a residual numerical divergence between the
vLLM and plain forwards on those inputs (first-block attention q norms differ
~1.5x despite byte-identical embeddings/norms; attention path verified as
mask-free bidirectional sdpa in both) — source unresolved.

**Decision per fallback rule:** v0.1.3 stays served on 8002. The v0.1.4 swap is
NOT justified until the head quality or the fidelity gap is addressed.

### 3. Notes for next session
- `echo-embed` (8003) was force-recreated clean; it now runs the ORIGINAL HF
  modules → re-sync the registry (scripts/sync_hf_modules.py) and verify the
  batch fix on 8003 (batch sim should equal single sim).
- /tmp/clf14-fix = full v0.1.4 + fixed modules + EchoModelForPooling config
  (auto_map updated) + ST files; test container command in the handoff.
- /tmp/embed-local = embed-Exp snapshot with resolved symlinks (tar -h).
- Debug prints (ECHO-*) in /tmp/clf14-fix/modeling_echo.py are gated by
  ECHO_DBG_INPUT env — harmless.

## UPDATE 2026-08-22 (session 2, final) — 8002 SWAPPED TO v0.1.4 ✅

The swap is DONE and live. `INTENT_MODEL=ethicalabs/Echo-DSRN-v0.1.4-Embed-Intent-CLF`
(.env override + compose default + README table, commit 322e4c6). echo-intent
container healthy, no 'blocks' error, /classify works (weather_query, alarm_set,
datetime_query, play_music, lists_createoradd on spot checks; batch == single
consistent).

Key findings that made this possible:
1. **Fidelity confirmed**: the vLLM classify pooled vector matches a plain
   forward in the SAME torch env exactly (12.15 norm both). The earlier
   "fidelity gap" was a torch-build artifact of the uv-run test env — always
   compare references in the same environment.
2. **No fine-tuning needed** (user confirmed): borderline spot-check labels
   (lights → iot_wemo_on-type, flight → qa_factoid) are the model's real
   predictions, consistent with its validated MASSIVE training accuracy.
3. HF repos all synced: Embed-Exp, Embed-Intent (manual upload — ST repos are
   outside the registry sync), v0.1.4-CLF (modules + config auto_map
   AutoModel → modeling_echo.EchoModelForPooling).

Remaining notes: the v0.1.3 LM-based classifier remains usable (registry
Echo-DSRN-v0.1.3-Intent-CLF, README §intent-classification still documents the
from_causal_lm flow). If /classify quality regressions appear on MASSIVE-style
utterances, compare against the same-env plain forward before suspecting vLLM.
