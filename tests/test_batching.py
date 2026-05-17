"""
tests/test_batching.py
──────────────────────────────────────────────────────────────────────────────
Integration tests to explicitly check batching behavior and padding corruption.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import echo_dsrn  # noqa: F401

# Import existing fixtures to avoid redefining the whole download/merge pipeline


def test_seq_clf_batching_equivalence(nsfw_clf_model, tokenizer):
    """
    EchoForSequenceClassification should produce exactly the same logits
    for an item evaluated alone vs inside a right-padded batch.
    """
    # Force right padding for seq_clf
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    short_text = "Play some music."
    long_text = "What is the weather like in Tokyo tomorrow afternoon?"

    # 1. Single evaluation
    short_enc = tokenizer(short_text, return_tensors="pt")
    long_enc = tokenizer(long_text, return_tensors="pt")

    with torch.inference_mode():
        single_short_logits = nsfw_clf_model(**short_enc).logits[0]
        single_long_logits = nsfw_clf_model(**long_enc).logits[0]

    # 2. Batched evaluation
    batched_enc = tokenizer([short_text, long_text], return_tensors="pt", padding=True)
    with torch.inference_mode():
        batched_logits = nsfw_clf_model(**batched_enc).logits

    # 3. Assertions
    # The short text gets padded. Because EchoForSequenceClassification picks the
    # last non-padding token correctly and it's right-padded, the state should
    # perfectly match the single evaluation.
    batched_short_logits = batched_logits[0]
    batched_long_logits = batched_logits[1]

    # Long text (no padding) should exactly match
    assert torch.allclose(single_long_logits, batched_long_logits, atol=1e-4)
    # Short text (padded) should exactly match!
    assert torch.allclose(single_short_logits, batched_short_logits, atol=1e-4)
