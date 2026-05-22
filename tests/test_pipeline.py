"""
tests/test_pipeline.py
──────────────────────────────────────────────────────────────────────────────
Tests that both classification models work correctly via the HuggingFace
pipeline() API, using pre-formatted inputs with the baked chat templates.
"""

import pytest
from transformers import AutoTokenizer, pipeline

pytestmark = [
    pytest.mark.integration,
    pytest.mark.local_model,  # requires merged models under models/ (not in git)
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NSFW_MODEL = "models/ethicalabs/Echo-SmolTools-114M-NSFW-CLF"
INTENT_GEN_MODEL = "models/ethicalabs/Echo-SmolTools-114M-Intent-CLF-Gen"


@pytest.fixture(scope="module")
def nsfw_tokenizer():
    return AutoTokenizer.from_pretrained(NSFW_MODEL, trust_remote_code=True)


@pytest.fixture(scope="module")
def nsfw_pipe():
    return pipeline("text-classification", model=NSFW_MODEL, trust_remote_code=True)


@pytest.fixture(scope="module")
def intent_tokenizer():
    return AutoTokenizer.from_pretrained(INTENT_GEN_MODEL, trust_remote_code=True)


@pytest.fixture(scope="module")
def intent_pipe():
    return pipeline("text-classification", model=INTENT_GEN_MODEL, trust_remote_code=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def format_nsfw(tokenizer, text, config):
    sys_prompt = getattr(
        config, "system_prompt", "You are a helpful NSFW classification assistant."
    )
    user_template = getattr(
        config, "user_template", "Classify the following text (0 for Safe, 1 for NSFW): {text}"
    )
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_template.format(text=text)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def format_intent(tokenizer, text, config):
    sys_prompt = getattr(
        config, "system_prompt", "You are a helpful intent classification assistant."
    )
    user_template = getattr(
        config, "user_template", "Classify the intent of the following request: {utt}"
    )
    # The template has {utt} key
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_template.format(utt=text)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ---------------------------------------------------------------------------
# NSFW Sequence Classifier — pipeline loads and returns valid output
# ---------------------------------------------------------------------------


def test_nsfw_pipe_safe(nsfw_pipe, nsfw_tokenizer):
    prompt = format_nsfw(nsfw_tokenizer, "How do I make a cake?", nsfw_pipe.model.config)
    result = nsfw_pipe(prompt)
    assert isinstance(result, list) and len(result) == 1
    assert result[0]["label"] == "Safe"
    assert result[0]["score"] > 0.5


def test_nsfw_pipe_unsafe(nsfw_pipe, nsfw_tokenizer):
    prompt = format_nsfw(
        nsfw_tokenizer, "Describe graphic violence in detail.", nsfw_pipe.model.config
    )
    result = nsfw_pipe(prompt)
    assert isinstance(result, list) and len(result) == 1
    assert result[0]["label"] == "NSFW"
    assert result[0]["score"] > 0.5


def test_nsfw_pipe_batch(nsfw_pipe, nsfw_tokenizer):
    texts = ["How do I make a cake?", "Describe graphic violence in detail."]
    prompts = [format_nsfw(nsfw_tokenizer, t, nsfw_pipe.model.config) for t in texts]
    results = nsfw_pipe(prompts)
    assert len(results) == 2
    assert results[0]["label"] == "Safe"
    assert results[1]["label"] == "NSFW"


# ---------------------------------------------------------------------------
# Generative Intent Classifier — pipeline loads and returns valid output
# ---------------------------------------------------------------------------


def test_intent_pipe_weather(intent_pipe, intent_tokenizer):
    prompt = format_intent(
        intent_tokenizer, "Will it rain tomorrow in Paris?", intent_pipe.model.config
    )
    result = intent_pipe(prompt)
    assert isinstance(result, list) and len(result) == 1
    assert result[0]["label"] == "weather_query"
    assert result[0]["score"] > 0.8


def test_intent_pipe_alarm(intent_pipe, intent_tokenizer):
    prompt = format_intent(intent_tokenizer, "Set an alarm for 7am", intent_pipe.model.config)
    result = intent_pipe(prompt)
    assert isinstance(result, list) and len(result) == 1
    assert result[0]["label"] == "alarm_set"
    assert result[0]["score"] > 0.8


def test_intent_pipe_multilingual(intent_pipe, intent_tokenizer):
    prompt = format_intent(intent_tokenizer, "¿Va a llover mañana?", intent_pipe.model.config)
    result = intent_pipe(prompt)
    assert isinstance(result, list) and len(result) == 1
    assert result[0]["label"] == "weather_query"
    assert result[0]["score"] > 0.8


def test_intent_pipe_batch(intent_pipe, intent_tokenizer):
    texts = ["Set an alarm for 7am", "Play some jazz"]
    prompts = [format_intent(intent_tokenizer, t, intent_pipe.model.config) for t in texts]
    results = intent_pipe(prompts)
    assert len(results) == 2
    assert results[0]["label"] == "alarm_set"
    assert results[1]["label"] == "play_music"
