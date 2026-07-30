"""
tests/test_embed_to_clf_conversion.py
───────────────────────────────────────────────────────────────────
Tests for EchoForSequenceClassification.from_embedding() factory.
"""

import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from echo_dsrn.modeling_echo import EchoForSequenceClassification
from echo_embedding.modeling_embedding import EchoModelForSentenceEmbedding

# Cache the model once per module to avoid repeated downloads
_EMBED_MODEL = None


def _get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = EchoModelForSentenceEmbedding.from_pretrained(
            "ethicalabs/Echo-DSRN-v0.1.3-Embed-Intent",
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        _EMBED_MODEL.eval()
    return _EMBED_MODEL


# ── Conversion tests ─────────────────────────────────────────────


def test_from_embedding_returns_clf():
    embed = _get_embed_model()
    clf = EchoForSequenceClassification.from_embedding(embed, num_labels=5)
    assert isinstance(clf, EchoForSequenceClassification)


def test_classifier_head_shape():
    embed = _get_embed_model()
    clf = EchoForSequenceClassification.from_embedding(embed, num_labels=10)
    # mean_c_all pooling → hidden_size * num_heads
    expected_dim = embed.config.hidden_size * embed.config.num_heads
    assert clf.classifier.weight.shape == (10, expected_dim)
    assert clf.classifier.bias.shape == (10,)


def test_pooling_mode_stored():
    embed = _get_embed_model()
    clf = EchoForSequenceClassification.from_embedding(embed, num_labels=3)
    assert clf._pooling_mode == "mean_c_all"


def test_backbone_weights_match():
    embed = _get_embed_model()
    clf = EchoForSequenceClassification.from_embedding(embed, num_labels=3)

    # Compare backbone weight values (not object identity)
    embed_w = embed.model.embedding.weight
    clf_w = clf.model.embedding.weight
    assert torch.allclose(embed_w, clf_w, atol=1e-5)


def test_forward_output_shape():
    embed = _get_embed_model()
    clf = EchoForSequenceClassification.from_embedding(embed, num_labels=4)
    clf.eval()

    input_ids = torch.randint(0, 32000, (2, 32))
    attention_mask = torch.ones(2, 32)

    with torch.no_grad():
        out = clf(input_ids=input_ids, attention_mask=attention_mask)

    assert out.logits.shape == (2, 4)


def test_forward_no_nan():
    embed = _get_embed_model()
    clf = EchoForSequenceClassification.from_embedding(embed, num_labels=4)
    clf.eval()

    input_ids = torch.randint(0, 32000, (1, 64))
    with torch.no_grad():
        out = clf(input_ids)

    assert not torch.isnan(out.logits).any()
    assert not torch.isinf(out.logits).any()


def test_forward_with_labels_returns_loss():
    embed = _get_embed_model()
    clf = EchoForSequenceClassification.from_embedding(embed, num_labels=5)
    clf.train()

    input_ids = torch.randint(0, 32000, (4, 16))
    labels = torch.randint(0, 5, (4,))

    out = clf(input_ids=input_ids, labels=labels)
    assert out.loss is not None
    assert out.loss.ndim == 0
    assert not torch.isnan(out.loss)


def test_save_load_roundtrip():
    embed = _get_embed_model()
    clf = EchoForSequenceClassification.from_embedding(embed, num_labels=3)
    clf.eval()

    input_ids = torch.randint(0, 32000, (1, 16))
    with torch.no_grad():
        original_logits = clf(input_ids).logits

    with tempfile.TemporaryDirectory() as tmp_dir:
        clf.save_pretrained(tmp_dir)
        reloaded = EchoForSequenceClassification.from_pretrained(tmp_dir, trust_remote_code=True)
        reloaded.eval()

        with torch.no_grad():
            reloaded_logits = reloaded(input_ids).logits

    assert torch.allclose(original_logits, reloaded_logits, atol=1e-5)


def test_from_embedding_accepts_string_path():
    """Factory should accept a hub ID string."""
    clf = EchoForSequenceClassification.from_embedding(
        "ethicalabs/Echo-DSRN-v0.1.3-Embed-Intent",
        num_labels=2,
    )
    assert isinstance(clf, EchoForSequenceClassification)
    assert clf.classifier.weight.shape == (2, 512 * 4)  # 2048


def test_classifier_weights_are_random():
    """The classifier head must be randomly initialised (not zeros)."""
    embed = _get_embed_model()
    clf = EchoForSequenceClassification.from_embedding(embed, num_labels=5)

    w = clf.classifier.weight
    b = clf.classifier.bias

    # Weights should have non-trivial magnitude (normal init ~O(1/sqrt(dim)))
    assert w.norm().item() > 0.5, f"Weight norm too small: {w.norm().item()}"
    # Bias should be zero-initialized
    assert torch.all(b == 0.0), "Bias must be zero-initialized"
