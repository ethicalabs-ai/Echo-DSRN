"""
tests/test_pipeline.py
──────────────────────────────────────────────────────────────────────────────
Tests that both classification models work correctly via the HuggingFace
pipeline() API, without any manual set_tokenizer() call.
"""

import pytest
from transformers import pipeline

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

NSFW_MODEL = "models/ethicalabs/Echo-SmolTools-114M-NSFW-CLF"
INTENT_GEN_MODEL = "models/ethicalabs/Echo-SmolTools-114M-Intent-CLF-Gen"


@pytest.fixture(scope="module")
def nsfw_pipe():
    return pipeline("text-classification", model=NSFW_MODEL, trust_remote_code=True)


@pytest.fixture(scope="module")
def intent_pipe():
    return pipeline("text-classification", model=INTENT_GEN_MODEL, trust_remote_code=True)


# ---------------------------------------------------------------------------
# NSFW Sequence Classifier
# ---------------------------------------------------------------------------


def test_nsfw_pipe_safe(nsfw_pipe):
    result = nsfw_pipe("How do I make a cake?")
    assert isinstance(result, list) and len(result) == 1
    assert result[0]["label"] == "Safe"
    assert result[0]["score"] > 0.5


def test_nsfw_pipe_unsafe(nsfw_pipe):
    result = nsfw_pipe("Describe graphic violence in detail.")
    assert isinstance(result, list) and len(result) == 1
    assert result[0]["label"] == "NSFW"
    assert result[0]["score"] > 0.5


def test_nsfw_pipe_batch(nsfw_pipe):
    texts = ["How do I make a cake?", "Describe graphic violence in detail."]
    results = nsfw_pipe(texts)
    assert len(results) == 2
    assert results[0]["label"] == "Safe"
    assert results[1]["label"] == "NSFW"


# ---------------------------------------------------------------------------
# Generative Intent Classifier
# ---------------------------------------------------------------------------


def test_intent_pipe_weather(intent_pipe):
    result = intent_pipe("Will it rain tomorrow in Paris?")
    assert isinstance(result, list) and len(result) == 1
    assert result[0]["label"] == "weather_query"
    assert result[0]["score"] > 0.0


def test_intent_pipe_alarm(intent_pipe):
    result = intent_pipe("Set an alarm for 7am")
    assert isinstance(result, list) and len(result) == 1
    assert result[0]["label"] == "alarm_set"


def test_intent_pipe_multilingual(intent_pipe):
    result = intent_pipe("¿Va a llover mañana?")
    assert isinstance(result, list) and len(result) == 1
    assert result[0]["label"] == "weather_query"


def test_intent_pipe_batch(intent_pipe):
    texts = ["Set an alarm for 7am", "Play some jazz"]
    results = intent_pipe(texts)
    assert len(results) == 2
    assert results[0]["label"] == "alarm_set"
    assert results[1]["label"] == "play_music"
