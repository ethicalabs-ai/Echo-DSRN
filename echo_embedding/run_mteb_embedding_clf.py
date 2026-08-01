#!/usr/bin/env python3
"""
echo_embedding/run_mteb_embedding_clf.py
──────────────────────────────────────────
MTEB evaluation of a classifier model's embedding backbone.

Extracts mean_c_all embeddings from EchoForSequenceClassification (the
classifier discards its linear head and pools the recurrent slow state)
and evaluates via the standard MTEB protocol.

Usage:
    PYTHONPATH=. uv run python echo_embedding/run_mteb_embedding_clf.py \\
        --model_path models/Echo-DSRN-v0.1.3-Embed-Intent-CLF \\
        --tasks MassiveIntentClassification,MassiveScenarioClassification
"""

import argparse
import json
import os
import sys
from typing import List, Optional

import mteb
import numpy as np
import torch
from mteb.models.model_meta import ModelMeta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from echo_dsrn.modeling_echo import EchoForSequenceClassification
from transformers import AutoTokenizer


class ClassifierEmbeddingWrapper:
    """Extract embeddings from an EchoForSequenceClassification backbone.

    Runs the model with mean_c_all pooling (2048-dim recurrent slow states),
    producing SentenceTransformer-compatible embeddings for MTEB evaluation.
    """

    mteb_model_meta: ModelMeta

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        batch_size: int = 64,
    ):
        self.device = device
        self.batch_size = batch_size
        self.model = EchoForSequenceClassification.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model.eval()
        self.model.to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_name = os.path.basename(os.path.normpath(model_path))
        self.mteb_model_meta = ModelMeta(
            name=f"ethicalabs/{model_name}-embed-backbone",
            revision="no_revision_available",
            languages=["multilingual"],
            release_date=None,
        )

    def encode(
        self,
        sentences,
        batch_size: Optional[int] = None,
        show_progress_bar: bool = True,
        convert_to_numpy: bool = True,
        **kwargs,
    ) -> np.ndarray:
        # MTEB passes a DataLoader[BatchedInput], not a plain list
        if not isinstance(sentences, list):
            sentences = [text for batch in sentences for text in batch["text"]]

        batch_size = batch_size or self.batch_size
        all_embeddings = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            with torch.no_grad():
                out = self.model.model(**enc, output_all_states=True)
            c_all = out.all_c_all[-1]  # (B, T, hidden_size * num_heads)
            mask = enc["attention_mask"].unsqueeze(-1).to(c_all.dtype)
            pooled = (c_all * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            all_embeddings.append(pooled.cpu().to(torch.float32).numpy())

        return np.concatenate(all_embeddings, axis=0)

    # ── MTEB protocol stubs ──────────────────────────────────────
    def similarity(self, e1, e2):
        from mteb.similarity_functions import cos_sim
        return cos_sim(e1, e2)

    def similarity_pairwise(self, e1, e2):
        from mteb.similarity_functions import pairwise_cos_sim
        return pairwise_cos_sim(e1, e2)


def main():
    parser = argparse.ArgumentParser(
        description="MTEB evaluation on classifier embedding backbone"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/Echo-DSRN-v0.1.3-Embed-Intent-CLF",
        help="Path to EchoForSequenceClassification model",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="MassiveIntentClassification,MassiveScenarioClassification",
        help="Comma-separated MTEB task names",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )
    args = parser.parse_args()

    print(f"⚡ Loading classifier from: {args.model_path}")
    wrapped = ClassifierEmbeddingWrapper(
        args.model_path, device=args.device, batch_size=args.batch_size
    )
    print(f"   Model name: {wrapped.mteb_model_meta.name}")
    print(f"   Pooling: mean_c_all (2048-dim)")

    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    print(f"🔍 MTEB tasks: {task_list}")
    tasks = mteb.get_tasks(tasks=task_list)

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n🚀 Running MTEB evaluation...")
    results = mteb.evaluate(model=wrapped, tasks=tasks)

    output_file = os.path.join(args.output_dir, "mteb_clf_backbone_results.json")
    print(f"💾 Saving to {output_file}")
    with open(output_file, "w") as f:
        json.dump(results.to_dict(), f, indent=2)

    # Print summary
    print("\n📊 Summary:")
    for task_res in results.task_results:
        name = task_res.task_name
        scores = task_res.scores.get("test", task_res.scores.get("validation", []))
        if scores and isinstance(scores, list):
            accuracies = [s.get("main_score", 0) for s in scores]
            if accuracies:
                print(f"  {name}: {np.mean(accuracies):.4f}")

    print(f"\n✅ Done. Results in {output_file}")


if __name__ == "__main__":
    main()
