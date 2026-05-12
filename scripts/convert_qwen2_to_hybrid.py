"""
echo_hybrid/convert_from_qwen2.py
────────────────────────────────────────────────────────────────────────────
Initialise Echo-Hybrid weights from a Qwen2 checkpoint.

What this script does
─────────────────────
1. Loads the Qwen2 config and wraps it in HybridEchoConfig.
2. Creates a HybridEchoForCausalLM from the hybrid config (random init).
3. Loads Qwen2 pre-trained weights into the model with strict=False so that
   DSRN injector params (absent from Qwen2) are left with their default inits.
4. Applies the critical zero / orthogonal / bias inits to every injector.
5. Saves the result to output_dir alongside the tokenizer.

After running this script, the model at Step 0 is numerically identical to
vanilla Qwen2-0.5B because linear_read is all-zeros (identity injection).

Validation: run test_hybrid_inference.py and confirm outputs match Qwen2.

CLI usage
─────────
    uv run python echo_hybrid/convert_from_qwen2.py \
        --base_model_id Qwen/Qwen2.5-0.5B \
        --output_dir models/Echo-Hybrid-0.5B-Base \
        --dsrn_state_dim 512 \
        --dsrn_injection_stride 4 \
        --gate_bias_init 1.0

Or import and call convert_qwen2_to_hybrid() from another script.

CRITICAL NOTE
─────────────
Do NOT pass dtype=bfloat16 to from_pretrained() for the Qwen2 source — we
load in fp32 and let the user choose precision at training time.  This avoids
orthogonal init failing on bfloat16 tensors.
"""

import argparse

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Ensure echo_hybrid classes are registered before any AutoClass usage.
# We import by path since this script may be run from the repo root.
import echo_hybrid  # noqa: F401 — triggers AutoConfig.register
from echo_hybrid.configuration_hybrid import HybridEchoConfig
from echo_hybrid.modeling_hybrid import HybridEchoForCausalLM


def convert_qwen2_to_hybrid(
    base_model_id: str = "Qwen/Qwen2.5-0.5B",
    output_dir: str = "models/Echo-Hybrid-0.5B-Base",
    dsrn_state_dim: int = 512,
    dsrn_injection_stride: int = 4,
    gate_bias_init: float = 1.0,
    dtype: torch.dtype = torch.float32,
) -> HybridEchoForCausalLM:
    """
    Build Echo-Hybrid-0.5B-Base from a Qwen2 checkpoint.

    Parameters
    ──────────
    base_model_id : str
        HuggingFace model ID (or local path) of the Qwen2 base checkpoint.
    output_dir : str
        Directory to save the converted model.
    dsrn_state_dim : int
        Slow-state dimension for every DSRNMemoryInjector.
    dsrn_injection_stride : int
        Insert one DSRN injector after every N transformer layers.
    gate_bias_init : float
        Initial bias for linear_gate in every injector.  Use 2.0 if c_t
        norms fail to grow during Phase-1 warm-up.
    dtype : torch.dtype
        Dtype for the saved checkpoint (fp32 recommended for the base).

    Returns
    ───────
    HybridEchoForCausalLM
        The freshly initialised hybrid model (also saved to output_dir).
    """
    print(f"Loading Qwen2 config from: {base_model_id}")
    qwen_config = AutoConfig.from_pretrained(base_model_id, trust_remote_code=False)

    # ── Build HybridEchoConfig ────────────────────────────────────────────
    # to_dict() gives us all Qwen2 fields.  We inject the DSRN extras on top.
    qwen_dict = qwen_config.to_dict()
    # Remove model_type so Qwen2Config doesn't override our type
    qwen_dict.pop("model_type", None)
    qwen_dict.pop("auto_map", None)

    hybrid_config = HybridEchoConfig(
        dsrn_state_dim=dsrn_state_dim,
        dsrn_injection_stride=dsrn_injection_stride,
        gate_bias_init=gate_bias_init,
        **qwen_dict,
    )
    print(
        f"HybridEchoConfig: hidden_size={hybrid_config.hidden_size}, "
        f"num_layers={hybrid_config.num_hidden_layers}, "
        f"dsrn_state_dim={hybrid_config.dsrn_state_dim}, "
        f"stride={hybrid_config.dsrn_injection_stride}, "
        f"num_injectors={hybrid_config.num_hidden_layers // hybrid_config.dsrn_injection_stride}"
    )

    # ── Create model from hybrid config ──────────────────────────────────
    print("Initialising HybridEchoForCausalLM from config (random DSRN weights)...")
    model = HybridEchoForCausalLM(hybrid_config)

    # ── Load Qwen2 weights ────────────────────────────────────────────────
    print(f"Loading Qwen2 weights from: {base_model_id}")
    qwen_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=dtype,
        trust_remote_code=False,
    )
    qwen_state = qwen_model.state_dict()

    # Key remapping: Qwen2ForCausalLM uses "model.*" for backbone weights and
    # "lm_head.*" for the head.  Our hybrid nests the backbone under
    # "model.backbone.*", so we must rewrite every "model." prefix to
    # "model.backbone." before calling load_state_dict.
    #
    # Example mappings:
    #   "model.embed_tokens.weight"          → "model.backbone.embed_tokens.weight"
    #   "model.layers.0.self_attn.q_proj.weight" → "model.backbone.layers.0.self_attn.q_proj.weight"
    #   "lm_head.weight"                     → "lm_head.weight"  (unchanged)
    remapped_state = {}
    for k, v in qwen_state.items():
        if k.startswith("model."):
            new_k = "model.backbone." + k[len("model.") :]
        else:
            new_k = k  # lm_head.* stays as-is
        remapped_state[new_k] = v

    # strict=False: remapped state has no memory_injectors.* keys → those
    # stay at their default inits, which we overwrite below.
    missing, unexpected = model.load_state_dict(remapped_state, strict=False)

    # Missing keys should only be DSRN injector params
    dsrn_missing = [k for k in missing if "memory_injectors" in k]
    other_missing = [k for k in missing if "memory_injectors" not in k]

    print(
        f"  Weight transfer complete.\n"
        f"  DSRN injector params (expected missing): {len(dsrn_missing)}\n"
        f"  Other missing keys (should be 0):        {len(other_missing)}\n"
        f"  Unexpected keys    (should be 0):        {len(unexpected)}"
    )
    if other_missing:
        print(f"  ⚠️  Unexpected missing: {other_missing[:5]}")
    if unexpected:
        print(f"  ⚠️  Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    # Free Qwen2 model memory
    del qwen_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Apply critical DSRN inits ─────────────────────────────────────────
    # Zero-injection at Step 0 is achieved via linear_pred = 0, NOT linear_read.
    #
    # OLD (broken): linear_read = 0 → r_t = 0 → c_t = 0 → all gradients = 0
    #               (gradient desert: W_pred, W_mem, W_gate get exactly zero grad)
    #
    # NEW (fixed):  linear_pred = 0 → x_out = x + 0 @ c_t = x (still invisible ✅)
    #               linear_read = tiny random → r_t ≠ 0 → c_t ≠ 0 → all grads flow ✅
    #               The state learns internally before influencing x — better curriculum.
    print("Applying critical DSRN injector initializations...")
    for idx, injector in enumerate(model.model.memory_injectors):
        # linear_read → small random (std=0.001): r_t ≠ 0 so gradients flow through
        # W_mem, W_gate, W_pred from step 1. Tiny std keeps initial activations near-zero.
        nn.init.normal_(injector.linear_read.weight, mean=0.0, std=0.001)
        print(f"  Injector {idx}: linear_read → normal(0, 0.001) — gradients unblocked")

        # linear_pred → ZEROS: x_out = x + W_pred @ c_t = x + 0 = x at Step 0.
        # This is the new zero-injection anchor. Replaces the broken linear_read=0 approach.
        nn.init.zeros_(injector.linear_pred.weight)
        if hasattr(injector.linear_pred, "bias") and injector.linear_pred.bias is not None:
            nn.init.zeros_(injector.linear_pred.bias)
        print(f"  Injector {idx}: linear_pred → zeros (injector invisible at Step 0)")

        # surprise_lambda → zeros
        nn.init.zeros_(injector.surprise_lambda)

        # linear_gate.bias → gate_bias_init (open gates, promotes state retention)
        nn.init.constant_(injector.linear_gate.bias, gate_bias_init)
        print(f"  Injector {idx}: linear_gate.bias → {gate_bias_init}")

    # ── Validate zero-injection invariant ────────────────────────────────
    # New invariant: linear_pred.weight must be 0.0 (not linear_read).
    print("\nValidating zero-injection invariant (all linear_pred norms should be 0.0)...")
    for idx, injector in enumerate(model.model.memory_injectors):
        norm = injector.linear_pred.weight.norm().item()
        status = "✅" if norm == 0.0 else "❌"
        print(f"  {status} Injector {idx}: linear_pred.weight norm = {norm:.6f}")

    # ── Save ──────────────────────────────────────────────────────────────
    # Cast to bfloat16 before saving: all DSRN inits (orthogonal, zeros, bias)
    # are already complete so the cast is safe.  This halves peak RAM during
    # safetensors serialisation (16.5 GB fp32 → 8.2 GB bf16 for 4B model).
    print("\nCasting to bfloat16 before save...")
    model = model.to(torch.bfloat16)

    print(f"Saving Echo-Hybrid model to: {output_dir}")
    model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="2GB")

    print(f"Saving tokenizer from: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=False)
    tokenizer.save_pretrained(output_dir)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    dsrn_params = (
        sum(p.numel() for inj in model.model.memory_injectors for p in inj.parameters()) / 1e6
    )
    print(
        f"\n✅ Echo-Hybrid-0.5B-Base saved to {output_dir}\n"
        f"   Total parameters : {total_params:.1f}M\n"
        f"   DSRN injectors   : {dsrn_params:.1f}M\n"
        f"   Qwen2 backbone   : {total_params - dsrn_params:.1f}M\n"
        f"\nNext step: run scripts/test_hybrid_inference.py to confirm\n"
        f"           outputs match vanilla Qwen2-0.5B (DSRN invisible at Step 0)."
    )

    return model


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Convert Qwen2-0.5B to Echo-Hybrid-0.5B-Base")
    parser.add_argument(
        "--base_model_id",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="HuggingFace model ID or local path of the Qwen2 base checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/Echo-Hybrid-0.5B-Base",
        help="Directory to save the converted Echo-Hybrid checkpoint.",
    )
    parser.add_argument(
        "--dsrn_state_dim",
        type=int,
        default=512,
        help="Dimension of the c_t slow-state (per injector).",
    )
    parser.add_argument(
        "--dsrn_injection_stride",
        type=int,
        default=4,
        help="Insert a DSRN injector after every N transformer layers.",
    )
    parser.add_argument(
        "--gate_bias_init",
        type=float,
        default=1.0,
        help="Initial linear_gate bias (increase to 2.0 if c_t norms stay ~0 after warm-up).",
    )
    args = parser.parse_args()

    convert_qwen2_to_hybrid(
        base_model_id=args.base_model_id,
        output_dir=args.output_dir,
        dsrn_state_dim=args.dsrn_state_dim,
        dsrn_injection_stride=args.dsrn_injection_stride,
        gate_bias_init=args.gate_bias_init,
    )


if __name__ == "__main__":
    main()
