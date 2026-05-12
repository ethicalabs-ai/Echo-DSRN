"""
echo_hybrid/test_memory.py
─────────────────────────────────────────────────────────────────────────────
Needle-in-haystack scaling test for Echo-Hybrid.

Tests two modes:
  Standard  (use_kv_cache=True)  — chunk-by-chunk with backbone KV cache.
  Ablation  (use_kv_cache=False) — full-context re-feed mirroring talk.py.

A --max_vram_mb boundary prevents OOM: the test aborts gracefully and reports
current VRAM if the limit is reached.
"""

import argparse
import gc

import torch
from transformers import AutoConfig, AutoTokenizer

from echo_hybrid.configuration_hybrid import HybridEchoConfig
from echo_hybrid.modeling_hybrid import HybridEchoCache, HybridEchoForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MAX_VRAM_MB = 6000  # override with --max_vram_mb


def _vram_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0


def _vram_summary():
    if torch.cuda.is_available():
        cur = torch.cuda.memory_allocated() / 1024**2
        peak = torch.cuda.max_memory_allocated() / 1024**2
        res = torch.cuda.memory_reserved() / 1024**2
        print(
            f"\n── VRAM Summary ────────────────────────────────────────────────\n"
            f"  Current   : {cur:>8.1f} MB\n"
            f"  Peak      : {peak:>8.1f} MB   (inference only, model weights excluded)\n"
            f"  Reserved  : {res:>8.1f} MB   (total cache held by allocator)\n"
            f"─────────────────────────────────────────────────────────────"
        )


def _over_limit(label: str, step: str, max_vram_mb: float) -> bool:
    cur = _vram_mb()
    if cur > max_vram_mb:
        print(
            f"  ⚠  VRAM limit reached at {step}  "
            f"({cur:.1f} MB > {max_vram_mb:.0f} MB). Aborting."
        )
        return True
    return False


def test_memory_scaling(
    label: str,
    model_path: str,
    use_kv_cache: bool = True,
    target_len: int = 500,
    max_vram_mb: float = DEFAULT_MAX_VRAM_MB,
):
    print(f"\n{'=' * 64}")
    print(f"SCALING TEST: {label}  (use_kv_cache={use_kv_cache})")
    print(f"{'=' * 64}")

    AutoConfig.register("echo_hybrid", HybridEchoConfig)
    config = HybridEchoConfig.from_pretrained(model_path)
    config.use_kv_cache = use_kv_cache

    model = HybridEchoForCausalLM.from_pretrained(
        model_path,
        config=config,
        device_map=DEVICE,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Reset peak counter after load so the summary reflects only inference.
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    needle = "The secret code is 'MAGIC-VOODOO'."
    noise = "The ROCm software stack is optimized for AMD GPUs. "
    query = "\nQuestion: What was the secret code?\nAnswer:"

    needle_ids = tokenizer(needle, return_tensors="pt").input_ids.to(DEVICE)
    noise_ids = tokenizer(noise, return_tensors="pt").input_ids.to(DEVICE)
    query_ids = tokenizer(query, return_tensors="pt").input_ids.to(DEVICE)
    num_steps = max(1, target_len // noise_ids.shape[1])

    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print(f"  Initial VRAM : {_vram_mb():.1f} MB  (limit: {max_vram_mb:.0f} MB)")

    generated_tokens = []
    aborted = False

    with torch.no_grad():
        if use_kv_cache:
            # ── STANDARD MODE ────────────────────────────────────────────────
            # Chunk-by-chunk with backbone KV cache (identical to test_memory v1).
            pkv = None

            out = model(needle_ids, use_cache=True)
            pkv = out.past_key_values

            for i in range(1, num_steps + 1):
                if _over_limit(label, f"haystack step {i}", max_vram_mb):
                    aborted = True
                    break
                out = model(noise_ids, past_key_values=pkv, use_cache=True)
                pkv = out.past_key_values
                if i % max(1, num_steps // 4) == 0:
                    print(f"  Haystack step {i}/{num_steps}: {_vram_mb():.1f} MB")

            if not aborted:
                print("\n  Generating Answer (cached — single-token autoregressive)...")
                curr_ids = query_ids
                for step in range(15):
                    if _over_limit(label, f"gen step {step}", max_vram_mb):
                        aborted = True
                        break
                    out = model(curr_ids, past_key_values=pkv, use_cache=True)
                    pkv = out.past_key_values
                    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated_tokens.append(next_token.item())
                    curr_ids = next_token

        else:
            # ── ABLATION MODE ────────────────────────────────────────────────
            # Mirrors talk.py exactly: full-context re-feed every generation step.
            # 1. Build full context once (needle + haystack + query).
            print(f"  Building full context ({num_steps} noise chunks)...")
            full_ctx = torch.cat([needle_ids] + [noise_ids] * num_steps + [query_ids], dim=1)
            print(f"  Full context : {full_ctx.shape[1]} tokens")

            if _over_limit(label, "context build", max_vram_mb):
                aborted = True

            if not aborted:
                # 2. Single forward pass over the full prompt to seed DSRN state.
                print("\n  Generating Answer (ablation — full-context re-feed)...")
                ctx_ids = full_ctx
                dsrn_states = []

                for step in range(15):
                    if _over_limit(label, f"gen step {step}", max_vram_mb):
                        aborted = True
                        break

                    if dsrn_states:
                        carry = HybridEchoCache.from_legacy_cache(dsrn_states)
                        carry.seen_tokens = 0  # full re-feed; position resets
                        out = model(ctx_ids, past_key_values=carry, use_cache=True)
                    else:
                        out = model(ctx_ids, past_key_values=None, use_cache=True)

                    pkv = out.past_key_values
                    dsrn_states = pkv.dsrn_states if pkv else []
                    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    del out
                    del pkv

                    generated_tokens.append(next_token.item())
                    ctx_ids = torch.cat([ctx_ids, next_token], dim=1)

    if not aborted:
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f"  FINAL RESPONSE: {repr(response)}")
        if "MAGIC-VOODOO" in response:
            print(f"✅ {label} PASSED!")
        else:
            print(f"❌ {label} FAILED!")
    else:
        print(f"⚠  {label} ABORTED (VRAM limit — no OOM)")

    _vram_summary()

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Needle-in-haystack VRAM scaling test for Echo-Hybrid."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/Echo-Hybrid-0.5B-Kurtis-EON1",
        help="Path to the HybridEcho model directory.",
    )
    parser.add_argument(
        "--target_len",
        type=int,
        default=500,
        help="Target haystack length in tokens.",
    )
    parser.add_argument(
        "--max_vram_mb",
        type=float,
        default=DEFAULT_MAX_VRAM_MB,
        help="VRAM ceiling in MB.  Test aborts gracefully if exceeded.",
    )
    args = parser.parse_args()

    test_memory_scaling(
        "Echo-Hybrid-Standard",
        args.model_path,
        use_kv_cache=True,
        target_len=args.target_len,
        max_vram_mb=args.max_vram_mb,
    )
    test_memory_scaling(
        "Echo-Hybrid-Ablation",
        args.model_path,
        use_kv_cache=False,
        target_len=args.target_len,
        max_vram_mb=args.max_vram_mb,
    )
