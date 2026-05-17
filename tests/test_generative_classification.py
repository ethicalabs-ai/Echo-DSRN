"""
tests/test_generative_classification.py
────────────────────────────────────────────────────────────────────────────
Integration tests for EchoForGenerativeClassification.

All tests use the real Hub checkpoints:
  • Base model : ethicalabs/Echo-DSRN-114M-v0.1.2
  • Adapter    : ethicalabs/Echo-SmolTools-114M-Intent-PEFT

A module-scoped fixture loads and merges the model once; all tests share it.

Run with:
    pytest tests/test_generative_classification.py -v
"""

import pytest
import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

import echo_dsrn  # noqa: F401  # registers EchoConfig with AutoClass
from echo_dsrn.modeling_generative_clf import EchoForGenerativeClassification

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_MODEL_ID = "ethicalabs/Echo-DSRN-114M-v0.1.2"
PEFT_ADAPTER = "ethicalabs/Echo-SmolTools-114M-Intent-PEFT"
SYSTEM_PROMPT = "You are a helpful multilingual intent classification assistant."
USER_TEMPLATE = "Classify the intent of the following request: {utt}"
NUM_LABELS = 60

MASSIVE_INTENTS = [
    "datetime_query",
    "iot_hue_lightchange",
    "transport_ticket",
    "takeaway_query",
    "qa_stock",
    "general_greet",
    "recommendation_events",
    "music_dislikeness",
    "iot_wemo_off",
    "cooking_recipe",
    "qa_currency",
    "transport_traffic",
    "general_quirky",
    "weather_query",
    "audio_volume_up",
    "email_addcontact",
    "takeaway_order",
    "email_querycontact",
    "iot_hue_lightup",
    "recommendation_locations",
    "play_audiobook",
    "lists_createoradd",
    "news_query",
    "alarm_query",
    "iot_wemo_on",
    "general_joke",
    "qa_definition",
    "social_query",
    "music_settings",
    "audio_volume_other",
    "calendar_remove",
    "iot_hue_lightdim",
    "calendar_query",
    "email_sendemail",
    "iot_cleaning",
    "audio_volume_down",
    "play_radio",
    "cooking_query",
    "datetime_convert",
    "qa_maths",
    "iot_hue_lightoff",
    "iot_hue_lighton",
    "transport_query",
    "music_likeness",
    "email_query",
    "play_music",
    "audio_volume_mute",
    "social_post",
    "alarm_set",
    "qa_factoid",
    "calendar_set",
    "play_game",
    "alarm_remove",
    "lists_remove",
    "transport_taxi",
    "recommendation_movies",
    "iot_coffee",
    "music_query",
    "play_podcasts",
    "lists_query",
]
assert len(MASSIVE_INTENTS) == NUM_LABELS
ID2LABEL = {i: lbl for i, lbl in enumerate(MASSIVE_INTENTS)}


# ---------------------------------------------------------------------------
# Module-scoped fixture — loads once, shared across all tests in this file
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)


@pytest.fixture(scope="module")
def intent_gen_clf_model(tokenizer):
    """Load base + adapter, merge, wrap as EchoForGenerativeClassification."""
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    peft = PeftModel.from_pretrained(base, PEFT_ADAPTER, trust_remote_code=True)
    merged = peft.merge_and_unload()

    model = EchoForGenerativeClassification.from_causal_lm(
        merged,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        system_prompt=SYSTEM_PROMPT,
        user_template=USER_TEMPLATE,
    )
    model.eval()
    model.set_tokenizer(tokenizer)
    return model


# ---------------------------------------------------------------------------
# Structural tests (no classify() call needed)
# ---------------------------------------------------------------------------


def test_model_type(intent_gen_clf_model):
    assert type(intent_gen_clf_model).__name__ == "EchoForGenerativeClassification"


def test_no_new_weights(intent_gen_clf_model):
    """The classifier should have the exact same parameter names as CausalLM — no classifier head added."""
    param_names = {n for n, _ in intent_gen_clf_model.named_parameters()}
    assert (
        "classifier.weight" not in param_names
    ), "Generative classifier must not have a linear head"
    assert "lm_head.weight" in param_names, "lm_head must still be present"


def test_num_labels(intent_gen_clf_model):
    assert intent_gen_clf_model.config.num_labels == NUM_LABELS


def test_all_massive_labels_in_config(intent_gen_clf_model):
    for i, label in enumerate(MASSIVE_INTENTS):
        assert intent_gen_clf_model.config.id2label[i] == label


def test_prompt_baked_in_config(intent_gen_clf_model):
    assert intent_gen_clf_model.config.classification_system_prompt == SYSTEM_PROMPT
    assert "{utt}" in intent_gen_clf_model.config.classification_user_template


def test_weight_dtype(intent_gen_clf_model):
    param = next(intent_gen_clf_model.parameters())
    assert param.dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# Forward pass tests
# ---------------------------------------------------------------------------


def test_forward_output_shape(intent_gen_clf_model, tokenizer):
    texts = ["What time is it?", "Play some music"]
    enc = tokenizer(
        [
            f"<|system|> {SYSTEM_PROMPT}<|end|><|user|> {USER_TEMPLATE.format(utt=t)}<|end|><|assistant|>"
            for t in texts
        ],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )
    with torch.inference_mode():
        out = intent_gen_clf_model.forward(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            _tokenizer=tokenizer,
        )
    assert out.logits.shape == (2, NUM_LABELS)


def test_forward_no_nan(intent_gen_clf_model, tokenizer):
    enc = tokenizer(
        ["Turn off the lights please"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    )
    with torch.inference_mode():
        out = intent_gen_clf_model.forward(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            _tokenizer=tokenizer,
        )
    assert torch.isfinite(out.logits).all(), "Logits contain NaN or Inf"


def test_softmax_sums_to_one(intent_gen_clf_model, tokenizer):
    enc = tokenizer(
        ["Set an alarm for 7am"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    )
    with torch.inference_mode():
        out = intent_gen_clf_model.forward(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            _tokenizer=tokenizer,
        )
    probs = torch.softmax(out.logits.float(), dim=-1)
    assert abs(probs.sum().item() - 1.0) < 1e-3


# ---------------------------------------------------------------------------
# classify() tests — using the full chat template
# ---------------------------------------------------------------------------

INTENT_CASES = [
    ("Set an alarm for 7am", "alarm_set"),
    ("Play some jazz music", "play_music"),
    ("What time is it in Tokyo?", "datetime_query"),
    ("Will it rain tomorrow in Paris?", "weather_query"),
]


@pytest.mark.parametrize("utt,expected", INTENT_CASES)
def test_classify_english(intent_gen_clf_model, tokenizer, utt, expected):
    label, probs = intent_gen_clf_model.classify(utt, tokenizer)
    assert label == expected, (
        f"Got '{label}' for '{utt}', expected '{expected}'. "
        f"Top-3: {sorted(zip(MASSIVE_INTENTS, probs.tolist()), key=lambda x: -x[1])[:3]}"
    )


MULTILINGUAL_CASES = [
    ("Che ore sono a Roma?", "datetime_query"),  # Italian
    ("Piensa en un chiste", "general_joke"),  # Spanish
    ("Mets une alarme à 7 heures", "alarm_set"),  # French
    ("Speel wat jazzmuziek af", "play_music"),  # Dutch
]


@pytest.mark.parametrize("utt,expected", MULTILINGUAL_CASES)
def test_classify_multilingual(intent_gen_clf_model, tokenizer, utt, expected):
    label, probs = intent_gen_clf_model.classify(utt, tokenizer)
    assert label == expected, (
        f"Got '{label}' for '{utt}', expected '{expected}'. "
        f"Top-3: {sorted(zip(MASSIVE_INTENTS, probs.tolist()), key=lambda x: -x[1])[:3]}"
    )


def test_classify_batch(intent_gen_clf_model, tokenizer):
    """Batch inference must return one label per input."""
    utts = ["Set an alarm for 7am", "Play some jazz music", "What is the weather?"]
    labels, probs = intent_gen_clf_model.classify(utts, tokenizer)
    assert len(labels) == 3
    assert probs.shape == (3, NUM_LABELS)
    assert labels[0] == "alarm_set"
    assert labels[1] == "play_music"


# ---------------------------------------------------------------------------
# NSFW classifier regression guard — ensure existing class is unaffected
# ---------------------------------------------------------------------------


def test_nsfw_classifier_unaffected():
    """
    EchoForSequenceClassification must still be importable and registerable.
    This guards against accidentally breaking it while adding the new class.
    """
    from echo_dsrn.modeling_echo import EchoForSequenceClassification

    assert hasattr(EchoForSequenceClassification, "from_causal_lm")
    assert hasattr(EchoForSequenceClassification, "classify")
