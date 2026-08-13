"""
tests/test_pipeline.py
──────────────────────
Tests that both classification models work correctly via the echo_dsrn
pipeline() API. Raw strings are rendered with the baked chat templates
(system prompt + user template) automatically by
ChatTextClassificationPipeline, so no pre-formatting is needed.
"""

import pytest
from transformers import AutoTokenizer

from echo_dsrn import pipeline

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
    # device="cpu": ROCm GPU transfer triggers a 128s HIP kernel recompile per load
    return pipeline("text-classification", model=NSFW_MODEL, trust_remote_code=True, device="cpu")


@pytest.fixture(scope="module")
def intent_tokenizer():
    return AutoTokenizer.from_pretrained(INTENT_GEN_MODEL, trust_remote_code=True)


@pytest.fixture(scope="module")
def intent_pipe():
    # device="cpu": ROCm GPU transfer triggers a 128s HIP kernel recompile per load
    return pipeline(
        "text-classification", model=INTENT_GEN_MODEL, trust_remote_code=True, device="cpu"
    )


# ---------------------------------------------------------------------------
# NSFW Sequence Classifier — pipeline loads and returns valid output
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
# Generative Intent Classifier — pipeline loads and returns valid output
# ---------------------------------------------------------------------------


def test_intent_pipe_weather(intent_pipe):
    result = intent_pipe("Will it rain tomorrow in Paris?")
    assert isinstance(result, list) and len(result) == 1
    assert result[0]["label"] == "weather_query"
    assert result[0]["score"] > 0.8


def test_intent_pipe_alarm(intent_pipe):
    result = intent_pipe("Set an alarm for 7am")
    assert isinstance(result, list) and len(result) == 1
    assert result[0]["label"] == "alarm_set"
    assert result[0]["score"] > 0.8


def test_intent_pipe_multilingual(intent_pipe):
    result = intent_pipe("¿Va a llover mañana?")
    assert isinstance(result, list) and len(result) == 1
    assert result[0]["label"] == "weather_query"
    assert result[0]["score"] > 0.8


def test_intent_pipe_batch(intent_pipe):
    texts = ["Set an alarm for 7am", "Play some jazz"]
    results = intent_pipe(texts)
    assert len(results) == 2
    assert results[0]["label"] == "alarm_set"
    assert results[1]["label"] == "play_music"
