#!/usr/bin/env python3
"""
scripts/eval_alpha_sweep.py
────────────────────────────────────────────────────────────────────────
Sweep surprise_temperature_alpha values across reasoning benchmarks
using lm-evaluation-harness.

Tests the hypothesis: does self-aware temperature modulation improve
zero-shot and few-shot reasoning performance?

Benchmarks:
  - arc_easy, arc_challenge  (science reasoning)
  - hellaswag                 (commonsense reasoning)
  - piqa                      (physical commonsense)

Alpha values tested:
  [0.0, 0.3, 0.5, 1.0, 1.5, 2.0]  (default, tune with --alphas)

Outputs:
  - results_sweep.json          — aggregated metrics per α
  - alpha_performance_curve.png — accuracy vs α curves

Usage:
  uv run --extra rocm python scripts/eval_alpha_sweep.py \
      --model ethicalabs/Echo-DSRN-114M-v0.1.2 \
      --tasks arc_easy,arc_challenge,hellaswag,piqa \
      --alphas 0.0,0.3,0.5,1.0,1.5,2.0 \
      --limit 100 \
      --output-dir results/alpha_sweep
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import lm_eval  # noqa: E402

# Register the custom model wrapper with lm_eval
import echo_dsrn.lm_eval_wrapper  # noqa: E402, F401 — registers 'echo_dsrn_alpha'


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep surprise_temperature_alpha across reasoning benchmarks"
    )
    p.add_argument(
        "--model",
        default="ethicalabs/Echo-DSRN-114M-v0.1.2",
        help="Model path or HuggingFace Hub ID",
    )
    p.add_argument(
        "--alphas",
        type=str,
        default="0.0,0.3,0.5,1.0,1.5,2.0",
        help="Comma-separated alpha values to sweep",
    )
    p.add_argument(
        "--tasks",
        type=str,
        default="arc_easy,arc_challenge,hellaswag,piqa",
        help="Comma-separated lm_eval task names",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit samples per task (None = full dataset)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation",
    )
    p.add_argument(
        "--output-dir",
        default="results/alpha_sweep",
        help="Directory for output files",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--fewshot",
        type=int,
        default=0,
        help="Number of few-shot examples (default: 0 = zero-shot)",
    )
    return p.parse_args()


def parse_alphas(alphas_str: str) -> List[float]:
    return [float(x.strip()) for x in alphas_str.split(",")]


def parse_tasks(tasks_str: str) -> List[str]:
    return [t.strip() for t in tasks_str.split(",")]


# ── Main sweep ───────────────────────────────────────────────────────────────


def run_sweep(
    model_path: str,
    alphas: Sequence[float],
    tasks: Sequence[str],
    limit: int | None,
    batch_size: int,
    output_dir: str,
    fewshot: int,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)

    all_results: Dict[str, Any] = {
        "model": model_path,
        "alphas": list(alphas),
        "tasks": list(tasks),
        "limit": limit,
        "fewshot": fewshot,
        "results": {},
    }

    for alpha in alphas:
        label = f"α={alpha}"
        print(f"\n{'═' * 60}")
        print(f"Evaluating {model_path} with {label}")
        print(f"{'═' * 60}")

        try:
            result = lm_eval.simple_evaluate(
                model="echo_dsrn_alpha",
                model_args=f"pretrained={model_path},alpha={alpha}",
                tasks=list(tasks),
                num_fewshot=fewshot,
                limit=limit,
                batch_size=batch_size,
                device=None,  # let wrapper handle device
            )
        except Exception as exc:
            print(f"  ✗ FAILED: {exc}")
            all_results["results"][label] = {"error": str(exc)}
            continue

        # Extract metrics
        metrics: Dict[str, float] = {}
        task_results = result.get("results", {})
        for task_name, task_metrics in task_results.items():
            # lm_eval keys are like "acc,none", "acc_norm,none" — take the first acc
            for key, value in task_metrics.items():
                if "acc" in key.lower() and value is not None:
                    metrics[task_name] = value
                    print(f"  {task_name}: {value:.4f}")
                    break  # take first accuracy metric

        all_results["results"][label] = metrics

        # Clear GPU cache between runs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # Save results
    results_path = os.path.join(output_dir, "results_sweep.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    return all_results


# ── Visualization ────────────────────────────────────────────────────────────


def plot_results(all_results: Dict[str, Any], output_dir: str) -> None:
    """Generate accuracy vs α curves for each benchmark."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot generation.")
        print("  Install with: uv add matplotlib")
        return

    alphas = all_results["alphas"]
    tasks = all_results["tasks"]
    results = all_results["results"]

    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 4), squeeze=False)
    axes = axes[0]

    for idx, task in enumerate(tasks):
        ax = axes[idx]
        x_vals: List[float] = []
        y_vals: List[float] = []

        for alpha in alphas:
            label = f"α={alpha}"
            if label in results and task in results[label]:
                x_vals.append(alpha)
                y_vals.append(results[label][task])

        if x_vals:
            ax.plot(x_vals, y_vals, "o-", color="steelblue", markersize=8, linewidth=2)
            ax.axhline(
                y=y_vals[0],
                color="gray",
                linestyle="--",
                alpha=0.5,
                label=f"α=0 baseline ({y_vals[0]:.3f})",
            )
            # Highlight max
            max_idx = y_vals.index(max(y_vals))
            ax.plot(
                x_vals[max_idx],
                y_vals[max_idx],
                "r*",
                markersize=14,
                label=f"Best α={x_vals[max_idx]} ({y_vals[max_idx]:.3f})",
            )

        ax.set_title(task.replace("_", " ").title(), fontsize=11)
        ax.set_xlabel("surprise_temperature_alpha")
        ax.set_ylabel("Accuracy")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.3)

    model_name = all_results.get("model", "unknown").split("/")[-1]
    fig.suptitle(
        f"Surprise-Temperature α Sweep — {model_name}",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "alpha_performance_curve.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {plot_path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    alphas = parse_alphas(args.alphas)
    tasks = parse_tasks(args.tasks)

    print(f"Model:  {args.model}")
    print(f"Alphas: {alphas}")
    print(f"Tasks:  {tasks}")
    print(f"Limit:  {args.limit or 'full dataset'}")
    print(f"Fewshot: {args.fewshot}")

    results = run_sweep(
        model_path=args.model,
        alphas=alphas,
        tasks=tasks,
        limit=args.limit,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        fewshot=args.fewshot,
    )

    plot_results(results, args.output_dir)


if __name__ == "__main__":
    main()
