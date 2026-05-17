"""
tests/conftest.py
──────────────────────────────────────────────────────────────────────────────
Shared pytest fixtures available to all test modules.
"""

import pytest
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import echo_dsrn  # noqa: F401 — registers AutoClasses
from echo_dsrn.modeling_echo import EchoForSequenceClassification

BASE_MODEL_ID = "ethicalabs/Echo-DSRN-114M-v0.1.2"
NSFW_ADAPTER_ID = "ethicalabs/Echo-SmolTools-114M-NSFW-CLF-PEFT"
NSFW_MODEL_LOCAL = "models/ethicalabs/Echo-SmolTools-114M-NSFW-CLF"
INTENT_GEN_MODEL_LOCAL = "models/ethicalabs/Echo-SmolTools-114M-Intent-CLF-Gen"


@pytest.fixture(scope="session")
def tokenizer():
    return AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)


@pytest.fixture(scope="session")
def nsfw_clf_model(tokenizer):
    """
    Build EchoForSequenceClassification from base + NSFW LoRA adapter.
    Uses fp32 for deterministic CPU tests.
    """
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    peft_model = PeftModel.from_pretrained(base, NSFW_ADAPTER_ID, trust_remote_code=True)
    merged = peft_model.merge_and_unload()
    clf = EchoForSequenceClassification.from_causal_lm(
        merged,
        num_labels=2,
        id2label={0: "Safe", 1: "NSFW"},
        label_token_ids=[29900, 29896],
    )
    clf.eval()
    return clf
