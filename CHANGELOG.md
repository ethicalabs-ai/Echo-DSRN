# Changelog

All notable changes to Echo-DSRN-HF are documented here.

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
