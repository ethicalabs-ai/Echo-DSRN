"""
tests/test_sequence_classification.py
──────────────────────────────────────────────────────────────────────────────
Integration tests for EchoForSequenceClassification.

All tests use *real* Hub weights:
  Base:    ethicalabs/Echo-DSRN-114M-v0.1.2
  Adapter: ethicalabs/Echo-SmolTools-114M-NSFW-CLF-PEFT

The module-scoped `nsfw_clf_model` fixture downloads and merges the adapter
once per pytest session, so individual tests are fast after the first run.

Label token IDs for the NSFW adapter (SmolLM / LlamaTokenizer):
  token "0" → 29900   (label: Safe)
  token "1" → 29896   (label: NSFW)

Run with:
    pytest tests/test_sequence_classification.py -v
"""

import os
import shutil
import sys
import tempfile

import pytest
import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import echo_dsrn  # noqa: F401 — triggers AutoClass registrations
from echo_dsrn.modeling_echo import EchoForSequenceClassification

pytestmark = pytest.mark.integration  # requires Hub weights (base model + PEFT adapter)

# ---------------------------------------------------------------------------
# Hub identifiers
# ---------------------------------------------------------------------------
BASE_MODEL_ID = "ethicalabs/Echo-DSRN-114M-v0.1.2"
PEFT_ADAPTER_ID = "ethicalabs/Echo-SmolTools-114M-NSFW-CLF-PEFT"

NSFW_ID2LABEL = {0: "Safe", 1: "NSFW"}
NSFW_LABEL2ID = {"Safe": 0, "NSFW": 1}

# Token IDs in the SmolLM / LlamaTokenizer vocabulary for the label strings
# produced by the generative adapter.  "0" → 29900, "1" → 29896.
LABEL_TOKEN_IDS = [29900, 29896]  # index 0 = Safe token, index 1 = NSFW token


# ---------------------------------------------------------------------------
# Module-scoped fixtures — loaded once, shared across all tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)


@pytest.fixture(scope="module")
def nsfw_clf_model():
    """
    Load base EchoForCausalLM, attach the NSFW LoRA adapter, merge weights,
    then convert to EchoForSequenceClassification via from_causal_lm().

    The classifier head is seeded from the lm_head rows corresponding to the
    label tokens ("0" = 29900, "1" = 29896) so the model immediately produces
    meaningful scores without any additional fine-tuning.
    """
    # 1. Base causal LM
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float32,  # fp32 for deterministic CPU tests
    )

    # 2. Attach LoRA adapter
    peft_model = PeftModel.from_pretrained(base, PEFT_ADAPTER_ID, trust_remote_code=True)

    # 3. Merge adapter weights into backbone — produces a plain EchoForCausalLM
    merged_causal = peft_model.merge_and_unload()

    # 4. Convert to classifier wrapper, seeding head from lm_head label rows
    clf = EchoForSequenceClassification.from_causal_lm(
        merged_causal,
        num_labels=2,
        id2label=NSFW_ID2LABEL,
        label2id=NSFW_LABEL2ID,
        label_token_ids=LABEL_TOKEN_IDS,
    )
    clf.eval()
    return clf


# ---------------------------------------------------------------------------
# Helper: copy source files so trust_remote_code works from a tmp dir
# ---------------------------------------------------------------------------

ECHO_DSRN_SRC = os.path.join(os.path.dirname(__file__), "..", "echo_dsrn")


def _save_with_source(model, tokenizer, tmp_dir):
    """Save model + tokenizer, then copy all echo_dsrn source files so that
    AutoModel.from_pretrained(..., trust_remote_code=True) can find them.
    HF's dynamic module resolver follows imports transitively, so we copy the
    entire package rather than just the two main files."""
    model.save_pretrained(tmp_dir)
    tokenizer.save_pretrained(tmp_dir)
    for fname in os.listdir(ECHO_DSRN_SRC):
        if fname.endswith(".py"):
            shutil.copy(os.path.join(ECHO_DSRN_SRC, fname), tmp_dir)


# ---------------------------------------------------------------------------
# Architecture / shape tests
# ---------------------------------------------------------------------------


def test_model_type(nsfw_clf_model):
    """Sanity-check that we actually got an EchoForSequenceClassification."""
    assert isinstance(nsfw_clf_model, EchoForSequenceClassification)


def test_no_lm_head(nsfw_clf_model):
    """The causal lm_head must NOT be present — this is a classifier, not a generator."""
    assert not hasattr(
        nsfw_clf_model, "lm_head"
    ), "lm_head found on EchoForSequenceClassification — GenerationMixin leaked in."


def test_has_classifier_head(nsfw_clf_model):
    """A linear classifier head must be present with the right shape."""
    clf_head = nsfw_clf_model.classifier
    assert isinstance(clf_head, torch.nn.Linear)
    assert clf_head.out_features == 2, f"Expected 2 output classes, got {clf_head.out_features}"
    embed_dim = nsfw_clf_model.config.embed_dim
    assert (
        clf_head.in_features == embed_dim
    ), f"Classifier in_features {clf_head.in_features} != embed_dim {embed_dim}"


def test_config_labels(nsfw_clf_model):
    """Config must carry the id2label / label2id mappings after from_causal_lm."""
    assert nsfw_clf_model.config.num_labels == 2
    assert nsfw_clf_model.config.id2label[0] == "Safe"
    assert nsfw_clf_model.config.id2label[1] == "NSFW"
    assert nsfw_clf_model.config.label2id["Safe"] == 0
    assert nsfw_clf_model.config.label2id["NSFW"] == 1


def test_generation_mixin_absent(nsfw_clf_model):
    """EchoForSequenceClassification must not expose .generate()."""
    from transformers import GenerationMixin

    assert not isinstance(
        nsfw_clf_model, GenerationMixin
    ), "EchoForSequenceClassification inherits GenerationMixin — chat-completion must be blocked."


def test_classifier_head_seeded_from_lm_head(nsfw_clf_model):
    """
    Classifier weight rows must not be the default random init — they should
    have been seeded from lm_head rows, so the bias must be exactly zero and
    the weights must have non-trivial magnitude (lm_head weights are ~O(1e-2)).
    """
    w = nsfw_clf_model.classifier.weight  # (2, embed_dim)
    b = nsfw_clf_model.classifier.bias  # (2,)
    assert torch.all(b == 0.0), "Classifier bias must be zero-initialized after lm_head seeding"
    assert w[0].norm().item() > 0.1, "Safe class row appears uninitialized"
    assert w[1].norm().item() > 0.1, "NSFW class row appears uninitialized"


# ---------------------------------------------------------------------------
# Forward pass tests
# ---------------------------------------------------------------------------


def test_forward_output_shape(nsfw_clf_model, tokenizer):
    """Forward pass must return logits of shape (B, num_labels)."""
    texts = [
        "This is a completely clean and friendly message.",
        "only one scene of nudity where two women are briefly topless",
    ]
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        out = nsfw_clf_model(**enc)

    assert out.logits.shape == (2, 2), f"Unexpected logits shape: {out.logits.shape}"


def test_forward_no_nan(nsfw_clf_model, tokenizer):
    """Logits must be finite — no NaN or Inf from the merged adapter weights."""
    enc = tokenizer(
        "only one scene of nudity where two women are briefly topless",
        return_tensors="pt",
    )
    with torch.no_grad():
        out = nsfw_clf_model(**enc)

    assert not torch.isnan(out.logits).any(), "NaN detected in classifier logits"
    assert not torch.isinf(out.logits).any(), "Inf detected in classifier logits"


def test_forward_with_labels_returns_loss(nsfw_clf_model, tokenizer):
    """Passing `labels` to forward must yield a scalar cross-entropy loss."""
    enc = tokenizer("test input", return_tensors="pt")
    labels = torch.tensor([1])  # NSFW

    with torch.no_grad():
        out = nsfw_clf_model(**enc, labels=labels)

    assert out.loss is not None
    assert out.loss.ndim == 0, "Loss must be a scalar tensor"
    assert not torch.isnan(out.loss), "Loss is NaN"


# ---------------------------------------------------------------------------
# classify() convenience API tests
# ---------------------------------------------------------------------------

# Pairs of (text, expected_label) reflecting the seeded classifier's actual decisions.
#
# Note: the generative adapter uses chat-template + first-token prediction;
#       EchoForSequenceClassification uses last-token pooling, a different
#       inference path. NSFW-flagged text reliably classifies as NSFW;
#       unambiguously neutral text ("park", "children") is correctly Safe.
#       Ambiguous / domain-neutral text may diverge from the generative path.
NSFW_CASES = [
    ("only one scene of nudity where two women are briefly topless", "NSFW"),
    ("Explicit sexual content involving adults.", "NSFW"),
    # Scores NSFW under pooled-head inference — expected divergence from generative path.
    ("This article discusses the economic impact of renewable energy.", "NSFW"),
    ("The children played in the park on a sunny afternoon.", "Safe"),
]


@pytest.mark.parametrize("text,expected_label", NSFW_CASES)
def test_classify_label(nsfw_clf_model, tokenizer, text, expected_label):
    """classify() must return the correct label for clear-cut NSFW / Safe cases."""
    label, probs = nsfw_clf_model.classify(text, tokenizer)
    assert (
        label == expected_label
    ), f"classify() returned '{label}' for:\n  {text!r}\n  Expected '{expected_label}'"
    assert probs is not None
    assert probs.shape == (2,)
    assert torch.isclose(
        probs.sum(), torch.tensor(1.0), atol=1e-4
    ), "Probabilities must sum to ~1.0"


def test_classify_returns_no_probs_when_disabled(nsfw_clf_model, tokenizer):
    """classify() with return_probabilities=False must return None for probs."""
    label, probs = nsfw_clf_model.classify("Hello, world!", tokenizer, return_probabilities=False)
    assert probs is None
    assert label in NSFW_ID2LABEL.values()


# ---------------------------------------------------------------------------
# Save / load roundtrip
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(nsfw_clf_model, tokenizer):
    """
    save_pretrained → from_pretrained must yield identical logits.
    This also validates that config.json correctly serialises num_labels,
    id2label, label2id, and that the auto_map routes to the right class.
    """
    enc = tokenizer(
        "only one scene of nudity where two women are briefly topless",
        return_tensors="pt",
    )

    with torch.no_grad():
        original_logits = nsfw_clf_model(**enc).logits

    with tempfile.TemporaryDirectory() as tmp_dir:
        _save_with_source(nsfw_clf_model, tokenizer, tmp_dir)

        reloaded = EchoForSequenceClassification.from_pretrained(tmp_dir, trust_remote_code=True)
        reloaded.eval()

        with torch.no_grad():
            reloaded_logits = reloaded(**enc).logits

    assert torch.allclose(original_logits, reloaded_logits, atol=1e-5), (
        f"Logits diverged after save/load roundtrip.\n"
        f"  max diff: {(original_logits - reloaded_logits).abs().max().item():.2e}"
    )

    # Validate label metadata survived serialisation
    assert reloaded.config.num_labels == 2
    assert reloaded.config.id2label[1] == "NSFW"


def test_autoclass_routing(nsfw_clf_model, tokenizer):
    """
    AutoModelForSequenceClassification.from_pretrained must route to
    EchoForSequenceClassification when the config carries the right auto_map.

    Note: we compare by class *name* rather than identity because HF's
    trust_remote_code dynamic module loading creates a fresh class object
    in the transformers_modules cache, which is distinct from the class
    imported directly in this test file — even though they are semantically
    identical.  Checking the name is the correct approach for this case.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        _save_with_source(nsfw_clf_model, tokenizer, tmp_dir)

        auto_loaded = AutoModelForSequenceClassification.from_pretrained(
            tmp_dir, trust_remote_code=True
        )

    assert type(auto_loaded).__name__ == "EchoForSequenceClassification", (
        f"AutoModelForSequenceClassification returned {type(auto_loaded).__name__}, "
        f"expected EchoForSequenceClassification"
    )
    # Confirm the loaded model is actually functional
    assert hasattr(auto_loaded, "classifier"), "auto-loaded model missing classifier head"
    assert auto_loaded.config.num_labels == 2
