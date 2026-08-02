#!/usr/bin/env python3
"""Re-evaluate MASSIVE tasks one at a time, saving after each."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mteb
import numpy as np
import torch
from mteb.models.model_meta import ModelMeta
from echo_dsrn.modeling_echo import EchoForSequenceClassification
from transformers import AutoTokenizer


class Wrapper:
    mteb_model_meta = ModelMeta.create_empty(overwrites={"name": "ethicalabs/Echo-DSRN-v0.1.4-Embed-Intent-CLF-post-ce"})

    def __init__(self, model_path="ethicalabs/Echo-DSRN-v0.1.4-Embed-Intent-CLF", device="cuda"):
        self.device = device
        self.model = EchoForSequenceClassification.from_pretrained(model_path, trust_remote_code=True).eval().to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def encode(self, sentences, batch_size=64, **kwargs):
        if not isinstance(sentences, list):
            sentences = [text for batch in sentences for text in batch["text"]]
        all_embeds = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            enc = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
            with torch.no_grad():
                out = self.model.model(**enc, output_all_states=True)
            c_all = out.all_c_all[-1]
            mask = enc["attention_mask"].unsqueeze(-1).to(c_all.dtype)
            pooled = (c_all * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            all_embeds.append(pooled.cpu().to(torch.float32).numpy())
        return np.concatenate(all_embeds, axis=0)

    def similarity(self, e1, e2): from mteb.similarity_functions import cos_sim; return cos_sim(e1, e2)
    def similarity_pairwise(self, e1, e2): from mteb.similarity_functions import pairwise_cos_sim; return pairwise_cos_sim(e1, e2)


w = Wrapper()
outdir = "results"

for task_name in ["MassiveIntentClassification", "MassiveScenarioClassification"]:
    outfile = os.path.join(outdir, f"mteb_clf_{task_name}.json")
    if os.path.exists(outfile):
        print(f"⏭ Skipping {task_name} (already done)")
        continue
    print(f"🔍 {task_name}...")
    tasks = mteb.get_tasks(tasks=[task_name])
    result = mteb.evaluate(model=w, tasks=tasks)
    with open(outfile, "w") as f:
        f.write(result.model_dump_json(indent=2))
    print(f"   Saved {outfile}")
    # Quick stats
    for tr in result.task_results:
        for split, scores in tr.scores.items():
            accs = [s["main_score"] for s in scores]
            print(f"   {tr.task_name}/{split}: {np.mean(accs):.4f}")

# Merge
merged = {"task_results": []}
for task_name in ["MassiveIntentClassification", "MassiveScenarioClassification"]:
    outfile = os.path.join(outdir, f"mteb_clf_{task_name}.json")
    if os.path.exists(outfile):
        with open(outfile) as f:
            data = json.load(f)
        merged["task_results"].append(data.get("task_results", data))

with open(os.path.join(outdir, "mteb_clf_backbone_results.json"), "w") as f:
    json.dump(merged, f, indent=2)
print("\n✅ Merged results saved")
