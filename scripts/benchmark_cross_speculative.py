#!/usr/bin/env python3
"""
scripts/benchmark_cross_speculative.py
────────────────────────────────────────────────────────────────────────────
Benchmark cross-vocabulary (TLI) speculative decoding: Echo-DSRN draft →
Qwen target, using two different tokenizers.

Measures, per prompt and in aggregate:

  * Speculative tokens/sec — the full draft→verify→rollback loop with
    maintained draft and target KV caches.
  * Acceptance rate — accepted draft tokens / attempted draft tokens
    (leading-run acceptance, the metric that drives speculative speedup).
  * Vanilla target tokens/sec (greedy, cached) for comparison.

Notes on cross-vocabulary acceptance
────────────────────────────────────
The acceptance ceiling is bounded by how often the target's greedy token
falls inside the shared intersection $I$ (typically 40-75% for Qwen targets).
On top of that ceiling, the draft only proposes in-$I$ tokens and conditions
on a token-by-token translated context (out-of-$I$ tokens map to the draft
UNK token).  Additionally, on bf16 the incremental stepwise draft state
drifts slightly from a one-shot prefill of the same context (the hybrid
attention's masked vs unmasked path), which further lowers acceptance versus
a full re-prefix each round.  Losslessness is unaffected: the generated
stream always matches the target's own greedy decoding.

Usage
─────
    uv run --extra rocm python scripts/benchmark_cross_speculative.py \
        --draft ethicalabs/Echo-DSRN-114M-v0.1.2 \
        --target Qwen/Qwen3.8-27B \
        --max-draft 8 --tau-load 0.05 --max-new-tokens 64

Smaller targets (e.g. Qwen/Qwen2.5-0.5B or Qwen/Qwen3-0.6B) work identically
and are useful for CPU or quick smoke runs.
"""

import argparse
import sys
import time
from pathlib import Path

import torch

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from echo_dsrn import EchoConfig, EchoForCausalLM  # noqa: E402
from echo_dsrn.dspark_scheduler import (  # noqa: E402
    DSparkEchoConfig,
    DSparkEchoScheduler,
)
from echo_dsrn.speculative.vocab_mapper import build_vocab_intersection  # noqa: E402

PROMPTS = [
    "The capital of France is Paris, a city known for",
    "Machine learning is a field of artificial intelligence that",
    "Quantum mechanics describes the behavior of",
    "Python is a programming language widely used for",
    "The theory of evolution by natural selection was proposed by",
    "In mathematics, the Pythagorean theorem states that",
    "The speed of light in a vacuum is approximately",
    "Photosynthesis is the process by which plants convert",
    "DNA replication occurs during the",
    "Gradient descent is an optimization algorithm that",
]


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark cross-vocabulary TLI speculative decoding")
    p.add_argument("--draft", default="ethicalabs/Echo-DSRN-114M-v0.1.2")
    p.add_argument("--target", default="Qwen/Qwen3.8-27B")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--max-draft", type=int, default=8)
    p.add_argument("--tau-load", type=float, default=0.05)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--num-prompts", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_models(args):
    device = args.device
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    print(f"Loading draft: {args.draft} (α={args.alpha})")
    cfg = EchoConfig.from_pretrained(args.draft, trust_remote_code=True)
    cfg.surprise_temperature_alpha = args.alpha
    draft = (
        EchoForCausalLM.from_pretrained(
            args.draft, config=cfg, trust_remote_code=True, torch_dtype=dtype
        )
        .to(device)
        .eval()
    )

    print(f"Loading target: {args.target}")
    target = AutoModelForCausalLM.from_pretrained(args.target, torch_dtype=dtype).to(device).eval()

    draft_tok = AutoTokenizer.from_pretrained(args.draft, trust_remote_code=True)
    target_tok = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)
    return draft, target, draft_tok, target_tok


def spec_generate(scheduler, target, prompt_ids, max_new, eos_id=None, device="cuda"):
    """Lossless speculative generation loop (batch 1) using step()'s caches."""
    target_ids = prompt_ids
    draft_state, target_cache = None, None
    generated, attempted, accepted_total = [], 0, 0
    while len(generated) < max_new:
        r = scheduler.step(
            target_ids,
            target,
            past_key_values=draft_state,
            target_past_key_values=target_cache,
            return_cache=True,
        )
        chunk = r["accepted_tokens"]
        target_ids = torch.cat([target_ids, chunk], dim=1)
        draft_state = r["past_key_values"]
        target_cache = r["target_cache"]
        generated.extend(chunk[0].tolist())
        attempted += scheduler.config.max_draft_len
        accepted_total += int(r["n_accepted"][0])
        if eos_id is not None and chunk[0].tolist().count(eos_id):
            break
    return generated, accepted_total, attempted


def vanilla_generate(target, prompt_ids, max_new, eos_id=None):
    """Reference greedy decode with KV cache (the standard non-speculative path)."""
    out = target.generate(
        input_ids=prompt_ids,
        max_new_tokens=max_new,
        do_sample=False,
        use_cache=True,
        eos_token_id=eos_id,
        pad_token_id=eos_id,
    )
    return out[0, prompt_ids.shape[1] :].tolist()


def sync(device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = args.device

    draft, target, draft_tok, target_tok = load_models(args)

    print("Building token-level intersection table...")
    mapper = build_vocab_intersection(
        draft_tok,
        target_tok,
        draft_vocab_size=draft.config.vocab_size,
    )
    print(
        f"  draft vocab {mapper.draft_vocab_size} ∩ target vocab "
        f"{mapper.target_vocab_size} = {mapper.intersection_size} shared tokens"
    )

    scheduler = DSparkEchoScheduler(
        draft,
        DSparkEchoConfig(
            max_draft_len=args.max_draft,
            tau_load=args.tau_load,
            surprise_temperature_alpha=args.alpha,
            vocab_mapper=mapper,
        ),
    )

    eos_id = target_tok.eos_token_id
    prompts = PROMPTS[: args.num_prompts]

    print(
        f"\n{'prompt':>6s}  {'spec tok/s':>10s}  {'vanilla tok/s':>13s}  {'speedup':>7s}  "
        f"{'accept%':>7s}"
    )
    print("-" * 60)

    totals = {"spec": 0.0, "vanilla": 0.0, "accepted": 0, "attempted": 0}

    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            prompt_ids = target_tok(prompt, return_tensors="pt").input_ids.to(device)

            # Speculative decode.
            sync(device)
            t0 = time.perf_counter()
            generated, accepted, attempted = spec_generate(
                scheduler, target, prompt_ids, args.max_new_tokens, eos_id=eos_id, device=device
            )
            sync(device)
            spec_time = time.perf_counter() - t0
            spec_tps = len(generated) / spec_time

            # Vanilla greedy reference (cached).
            sync(device)
            t0 = time.perf_counter()
            ref_tokens = vanilla_generate(target, prompt_ids, args.max_new_tokens, eos_id=eos_id)
            sync(device)
            vanilla_tps = len(ref_tokens) / (time.perf_counter() - t0)

            accept = 100.0 * accepted / max(attempted, 1)
            speedup = spec_tps / max(vanilla_tps, 1e-9)
            totals["spec"] += spec_tps
            totals["vanilla"] += vanilla_tps
            totals["accepted"] += accepted
            totals["attempted"] += attempted
            print(
                f"{i + 1:6d}  {spec_tps:10.2f}  {vanilla_tps:13.2f}  {speedup:7.2f}x  "
                f"{accept:6.1f}%"
            )

    n = len(prompts)
    total_accept = 100.0 * totals["accepted"] / max(totals["attempted"], 1)
    total_speedup = totals["spec"] / max(totals["vanilla"], 1e-9)
    print("-" * 60)
    print(
        f"  avg    {totals['spec'] / n:10.2f}  {totals['vanilla'] / n:13.2f}  "
        f"{total_speedup:7.2f}x  {total_accept:6.1f}%"
    )
    print(
        f"\n  tokens/sec speedup over greedy: {total_speedup:.2f}x  "
        f"(acceptance {total_accept:.1f}%)"
    )


if __name__ == "__main__":
    main()
