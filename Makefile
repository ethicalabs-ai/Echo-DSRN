.PHONY: help merge-nsfw merge-intent merge-all eval-nsfw eval-intent eval-all

help:
	@echo "Echo-DSRN Classification Utilities"
	@echo "=================================="
	@echo "make merge-nsfw    - Build the NSFW Sequence Classifier checkpoint"
	@echo "make merge-intent  - Build the MASSIVE Intent Generative Classifier checkpoint"
	@echo "make merge-all     - Build both checkpoints"
	@echo ""
	@echo "make eval-nsfw     - Run evaluation on the NSFW Safe Dataset"
	@echo "make eval-intent   - Run evaluation on the Amazon MASSIVE dataset"
	@echo "make eval-all      - Run both evaluations"

# --- Merging Commands ---

merge-nsfw:
	uv run python3 scripts/merge_clf_adapter.py \
		--base ethicalabs/Echo-DSRN-114M-v0.1.2 \
		--adapter ethicalabs/Echo-SmolTools-114M-NSFW-CLF-PEFT \
		--output models/ethicalabs/Echo-SmolTools-114M-NSFW-CLF \
		--num-labels 2 \
		--id2label "0:Safe,1:NSFW" \
		--label-token-ids "29900,29896" \
		--dtype bfloat16 \
		--system-prompt "You are a helpful NSFW classification assistant." \
		--user-template "Classify the following text (0 for Safe, 1 for NSFW): {text}"

merge-intent:
	uv run python3 scripts/merge_intent_gen_clf.py

merge-all: merge-nsfw merge-intent


# --- Evaluation Commands ---

eval-nsfw:
	uv run python3 benchmarks/run_clf_eval.py \
		--model models/ethicalabs/Echo-SmolTools-114M-NSFW-CLF \
		--dataset eliasalbouzidi/NSFW-Safe-Dataset \
		--batch_size 128

eval-intent:
	uv run python3 benchmarks/run_generative_clf_eval.py \
		--model models/ethicalabs/Echo-SmolTools-114M-Intent-CLF-Gen \
		--batch_size 32 \
		--langs all

eval-all: eval-nsfw eval-intent
