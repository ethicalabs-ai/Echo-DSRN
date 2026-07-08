#!/usr/bin/env python3
"""
scripts/convert_to_dspark.py
────────────────────────────────────────────────────────────────────────
Convert a base Echo-DSRN checkpoint into the modern, evaluation-ready variant.

Sets surprise_temperature_alpha, output_surprise_gate_logits, and marks
embeddings as tieable (without post-hoc overwrite — the model keeps its
trained lm_head/embedding weights).

Usage:
  uv run --extra rocm python scripts/convert_to_dspark.py \
      ethicalabs/Echo-DSRN-114M-v0.1.2 \
      ./models/Echo-DSRN-v0.1.4

Output model is ready for lm_eval:
  uv run lm_eval --model hf \
      --model_args pretrained=./models/Echo-DSRN-v0.1.4,trust_remote_code=True \
      --tasks arc_easy,arc_challenge,hellaswag,piqa \
      --batch_size 16
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import torch

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))


def main():
    p = argparse.ArgumentParser(
        description="Convert Echo-DSRN checkpoint to evaluation-ready variant"
    )
    p.add_argument("input", help="Input model path or HF Hub ID")
    p.add_argument("output", help="Output directory")
    p.add_argument(
        "--alpha", type=float, default=1.0, help="surprise_temperature_alpha (default: 1.0)"
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # ── Resolve model class from architectures ───────────────────────────
    from transformers import AutoConfig

    raw_config = AutoConfig.from_pretrained(args.input, trust_remote_code=True)
    archs = getattr(raw_config, "architectures", []) or []
    arch_map = {
        "EchoForCausalLM": ("echo_dsrn", "EchoForCausalLM"),
        "HybridEchoForCausalLM": ("echo_hybrid", "HybridEchoForCausalLM"),
        "Qwen3HybridEchoForCausalLM": ("echo_hybrid", "Qwen3HybridEchoForCausalLM"),
    }
    cls = None
    for arch_name, (mod_name, cls_name) in arch_map.items():
        if arch_name in archs:
            mod = __import__(mod_name, fromlist=[cls_name])
            cls = getattr(mod, cls_name)
            break
    if cls is None:
        raise ValueError(f"Cannot resolve model class from architectures={archs}")
    print(f"Model class: {cls.__name__}")

    # ── Set modern config flags ──────────────────────────────────────────
    raw_config.surprise_temperature_alpha = args.alpha
    raw_config.output_surprise_gate_logits = True
    raw_config.tie_word_embeddings = True
    print(f"surprise_temperature_alpha = {args.alpha}")
    print("output_surprise_gate_logits = True")
    print("tie_word_embeddings = True (preserving trained weights)")

    # ── Load and save ────────────────────────────────────────────────────
    print(f"Loading {args.input}...")
    model = cls.from_pretrained(
        args.input,
        config=raw_config,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")

    model.save_pretrained(args.output)
    raw_config.save_pretrained(args.output)

    # ── Copy tokenizer ───────────────────────────────────────────────────
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.input, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.save_pretrained(args.output)

    # ── Copy Python sources for trust_remote_code ────────────────────────
    _root = Path(__file__).resolve().parent.parent
    echo_files = ["configuration_echo.py", "modeling_echo.py", "triton_scan.py"]
    for fname in echo_files:
        src = _root / "echo_dsrn" / fname
        if src.exists():
            shutil.copy2(src, os.path.join(args.output, fname))

    # Hybrid models also need echo_hybrid sources
    if "HybridEcho" in str(archs):
        hybrid_files = ["configuration_hybrid.py", "modeling_hybrid.py", "dsrn_memory_block.py"]
        for fname in hybrid_files:
            src = _root / "echo_hybrid" / fname
            if src.exists():
                shutil.copy2(src, os.path.join(args.output, fname))

    # Pre-cache triton_scan for HF dynamic loader
    try:
        from transformers.dynamic_module_utils import get_cached_module_file

        get_cached_module_file(args.output, "triton_scan.py")
    except Exception:
        pass

    print(f"\n✓ Exported to {args.output}")
    print("  Ready for lm_eval:")
    print("  uv run lm_eval --model hf \\")
    print(f"      --model_args pretrained={args.output},trust_remote_code=True \\")
    print("      --tasks arc_easy,arc_challenge,hellaswag,piqa \\")
    print("      --batch_size 16")


if __name__ == "__main__":
    main()
