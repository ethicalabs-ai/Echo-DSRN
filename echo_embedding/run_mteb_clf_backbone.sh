#!/usr/bin/env bash
set -euo pipefail
# Post-CE MTEB: evaluate classifier's embedding backbone quality.
# Extracts mean_c_all embeddings and benchmarks via MTEB.
# Compare against baseline: 72.42% (original Embed-Intent model).
cd "$(dirname "$0")/.."
PYTHONPATH=. uv run --extra rocm python echo_embedding/run_mteb_embedding_clf.py \
  --model_path ethicalabs/Echo-DSRN-v0.1.4-Embed-Intent-CLF \
  --tasks MassiveIntentClassification,MassiveScenarioClassification \
  --output_dir results \
  --device cuda
