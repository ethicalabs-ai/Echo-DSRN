"""
benchmarks/run_mteb_embedding.py
──────────────────────────────────────────────────────────────────────────────
Official MTEB validation suite for Echo-DSRN embedding model.
Loads the model natively using SentenceTransformer and evaluates it using
the modern mteb API.
"""

import argparse
import json
import os
import sys

import mteb
import torch
from sentence_transformers import SentenceTransformer

# Ensure we can import local packages if running from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    parser = argparse.ArgumentParser(description="Run MTEB Evaluation on Echo-DSRN Embedding Model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/Echo-DSRN-v0.1.3-Embed",
        help="Path to the Echo embedding model to evaluate.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="STS12",
        help="Comma-separated list of MTEB task names to evaluate on (default: 'STS12').",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save MTEB results.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on.",
    )
    args = parser.parse_args()

    print(f"📦 Loading SentenceTransformer from: {args.model_path}...")
    # Load model with trust_remote_code=True since it uses our custom modeling classes
    model = SentenceTransformer(args.model_path, trust_remote_code=True, device=args.device)
    print("✅ Model loaded successfully!")

    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    print(f"🔍 Fetching MTEB task objects for: {task_list}...")
    tasks = mteb.get_tasks(tasks=task_list)

    print("\n🚀 Running MTEB Evaluation...")
    print(f"  - Model: {args.model_path}")
    print(f"  - Tasks: {task_list}")
    print(f"  - Output directory: {args.output_dir}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    from mteb.models.sentence_transformer_wrapper import (
        SentenceTransformerEncoderWrapper,
    )

    # Wrap model and set unique name based on the path
    wrapped_model = SentenceTransformerEncoderWrapper(model)
    model_name = os.path.basename(os.path.normpath(args.model_path))
    full_name = f"ethicalabs/{model_name}"

    # Update metadata to ensure unique cache keys and result filenames
    wrapped_model.mteb_model_meta = wrapped_model.mteb_model_meta.model_copy(
        update={"name": full_name}
    )

    print(f"🏷️ Assigned unique MTEB model name: {full_name}")

    # Run evaluation forcing a live run (overwrite_strategy="always")
    results = mteb.evaluate(model=wrapped_model, tasks=tasks, overwrite_strategy="always")

    # Save results dictionary to output directory
    output_file = os.path.join(args.output_dir, "mteb_evaluation_results.json")
    print(f"💾 Saving evaluation results to {output_file}...")

    # Use model_dump_json to handle datetimes and other custom objects, then load to format
    try:
        results_json = results.model_dump_json()
    except AttributeError:
        results_json = results.json()

    results_dict = json.loads(results_json)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)

    print("\n📊 Summary of Scores:")
    for task_res in results_dict.get("task_results", []):
        task_name = task_res.get("task_name", "Unknown")
        print(f"  👉 Task: {task_name}")
        scores = task_res.get("scores", {})
        # Print some representative scores (e.g. test split)
        test_scores = scores.get("test", [])
        if test_scores and isinstance(test_scores, list):
            # Print the first score dictionary in test list
            print(f"     Test Scores: {test_scores[0]}")
        elif isinstance(scores, dict):
            print(f"     Scores: {scores}")

    print(f"\n🎉 Evaluation complete! Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
