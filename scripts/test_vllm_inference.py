"""
test_vllm_inference.py
──────────────────────────────────────────────────────────────────────────────
Test Echo-DSRN / Echo-Hybrid inference via vLLM.

Supports:
  • Named model shortcuts     (dsrn, hybrid)
  • HuggingFace Hub model IDs (full Hub path)
  • Local checkpoint directories

Usage examples:
  # Named shortcut — Echo-DSRN 114M v0.1.2 (default)
  python test_vllm_inference.py --model dsrn

  # Named shortcut — Echo-Hybrid 0.5B
  python test_vllm_inference.py --model hybrid

  # Explicit Hub ID
  python test_vllm_inference.py --model ethicalabs/Echo-DSRN-114M-v0.1.2

  # Local checkpoint
  python test_vllm_inference.py --model ./checkpoints/echo-dsrn-114m

  # Full options
  python test_vllm_inference.py \\
      --model dsrn \\
      --prompt "The theory of predictive coding suggests that" \\
      --max-tokens 80 \\
      --tensor-parallel-size 1

Requirements:
  pip install vllm  (ROCm: see pyproject.toml [dependency-groups] rocm)
"""

import argparse
import os
import sys

# ── Ensure the local packages are importable when running from the repo root ─
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ── Register both local architectures with HF AutoClass ──────────────────────
# Required so vLLM can resolve model_type="echo" / "echo_hybrid" via
# trust_remote_code=True without needing the classes baked into vLLM itself.

try:
    import echo_hybrid  # noqa: F401  # registers HybridEchoConfig + HybridEchoForCausalLM
except ImportError:
    pass  # optional; only needed when --model hybrid is used

# ── Known model shortcuts ─────────────────────────────────────────────────────
# Avoids typing full Hub IDs on the command line.
KNOWN_MODELS: dict[str, str] = {
    "dsrn": "ethicalabs/Echo-DSRN-114M-v0.1.2",
    "hybrid": "ethicalabs/Echo-Hybrid-0.5B",
}


def parse_args() -> argparse.Namespace:
    shortcut_help = ", ".join(f"{k!r} → {v}" for k, v in KNOWN_MODELS.items())
    parser = argparse.ArgumentParser(
        description="Test Echo-DSRN / Echo-Hybrid inference with vLLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="dsrn",
        help=(
            "Model to load. Accepts a named shortcut, a HuggingFace Hub ID, or a "
            "local checkpoint path (must contain config.json). "
            f"Named shortcuts: {shortcut_help}."
        ),
    )
    parser.add_argument(
        "--prompt",
        default="The theory of predictive coding suggests that",
        help="Prompt string to run generation on.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=80,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0 = greedy).",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Weight dtype passed to vLLM.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="Fraction of GPU memory vLLM may use (0.0–1.0).",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        default=False,
        help="Disable trust_remote_code (enabled by default for Echo models).",
    )
    return parser.parse_args()


def resolve_model_path(model: str) -> str:
    """
    Resolve a model identifier in priority order:
      1. Named shortcut  (e.g. "dsrn", "hybrid")
      2. Local path      (contains os.sep or starts with ".")
      3. Hub ID          (everything else)
    """
    # 1. Named shortcut
    if model in KNOWN_MODELS:
        resolved = KNOWN_MODELS[model]
        print(f"[INFO]  Shortcut {model!r} → {resolved}")
        return resolved

    # 2. Local path
    if os.sep in model or model.startswith("."):
        abs_path = os.path.abspath(model)
        if not os.path.isdir(abs_path):
            sys.exit(f"[ERROR] Local model path does not exist: {abs_path}")
        if not os.path.isfile(os.path.join(abs_path, "config.json")):
            sys.exit(
                f"[ERROR] config.json not found in {abs_path}. "
                "Is this a valid HuggingFace checkpoint directory?"
            )
        print(f"[INFO]  Using local checkpoint: {abs_path}")
        return abs_path

    # 3. Hub ID
    print(f"[INFO]  Using Hub model: {model}")
    return model


def main() -> None:
    args = parse_args()

    # ── Import vLLM (optional dep — give a helpful message if missing) ──────
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        sys.exit(
            "[ERROR] vLLM is not installed.\n"
            "Install it with:\n"
            "  pip install vllm\n"
            "For ROCm:\n"
            "  pip install vllm --extra-index-url https://download.pytorch.org/whl/rocm7.1"
        )

    trust_remote_code = not args.no_trust_remote_code
    model_id = resolve_model_path(args.model)

    print(f"[INFO]  trust_remote_code = {trust_remote_code}")
    print(f"[INFO]  dtype             = {args.dtype}")
    print(f"[INFO]  tensor-parallel   = {args.tensor_parallel_size}")
    print("[INFO]  Loading model …")

    llm = LLM(
        model=model_id,
        trust_remote_code=trust_remote_code,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    print(f"\n[PROMPT] {args.prompt}\n")
    outputs = llm.generate([args.prompt], sampling_params)

    for output in outputs:
        for completion in output.outputs:
            print(f"[OUTPUT] {output.prompt}{completion.text}")
            print(
                f"[STATS]  finish_reason={completion.finish_reason}  "
                f"tokens={len(completion.token_ids)}"
            )


if __name__ == "__main__":
    main()
