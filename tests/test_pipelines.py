"""
tests/test_pipelines.py
────────────────────────
Unit tests for ``echo_dsrn.pipelines`` — chat-template handling for
text-classification pipelines. No model weights are loaded; the model and
tokenizer are fakes.
"""

from types import SimpleNamespace

import pytest
import torch
from transformers import AutoTokenizer

from echo_dsrn.pipelines import ChatTextClassificationPipeline, pipeline

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeTokenizer:
    """Records ``apply_chat_template`` calls and accepts ``__call__``."""

    chat_template = "{{ messages | length }}"

    def __init__(self):
        self.formatted_messages = []
        self.tokenized_inputs = []

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True, **kwargs):
        self.formatted_messages.append((messages, add_generation_prompt))
        return f"FORMATTED:{len(messages)}"

    def __call__(self, inputs, return_tensors="pt", **kwargs):
        self.tokenized_inputs.append(inputs)
        return {"input_ids": inputs}


class BareTokenizer(FakeTokenizer):
    """Tokenizer without a chat template."""

    chat_template = None


class FakeModel:
    """Minimal stand-in satisfying what ``Pipeline.__init__`` touches."""

    def __init__(self, config=None):
        self.config = config or SimpleNamespace()
        self.device = torch.device("cpu")
        self.input_modalities = "text"
        self.output_modalities = "text"

    def can_generate(self):
        return False


def make_pipe(tokenizer=None, model_config=None, **kwargs):
    tokenizer = tokenizer or FakeTokenizer()
    model = FakeModel(SimpleNamespace(**model_config) if model_config else None)
    return ChatTextClassificationPipeline(model=model, tokenizer=tokenizer, device="cpu", **kwargs)


# ---------------------------------------------------------------------------
# Message inputs
# ---------------------------------------------------------------------------


def test_message_dicts_are_chat_formatted():
    tokenizer = FakeTokenizer()
    pipe = make_pipe(tokenizer=tokenizer)

    out = pipe.preprocess([{"role": "user", "content": "hello"}])

    assert tokenizer.formatted_messages == [([{"role": "user", "content": "hello"}], True)]
    assert out["input_ids"] == "FORMATTED:1"


def test_chat_object_input_is_chat_formatted():
    tokenizer = FakeTokenizer()
    pipe = make_pipe(tokenizer=tokenizer)
    inputs = SimpleNamespace(messages=[{"role": "user", "content": "hello"}])

    out = pipe.preprocess(inputs)

    assert tokenizer.formatted_messages == [([{"role": "user", "content": "hello"}], True)]
    assert out["input_ids"] == "FORMATTED:1"


def test_system_prompt_auto_filled_from_config():
    tokenizer = FakeTokenizer()
    pipe = make_pipe(tokenizer=tokenizer, model_config={"system_prompt": "SYS"})

    pipe.preprocess([{"role": "user", "content": "hello"}])

    assert tokenizer.formatted_messages == [
        ([{"role": "system", "content": "SYS"}, {"role": "user", "content": "hello"}], True)
    ]


def test_system_prompt_not_duplicated_when_messages_define_one():
    tokenizer = FakeTokenizer()
    pipe = make_pipe(tokenizer=tokenizer, model_config={"system_prompt": "SYS"})
    messages = [{"role": "system", "content": "USER SYS"}, {"role": "user", "content": "hello"}]

    pipe.preprocess(messages)

    assert tokenizer.formatted_messages[0][0] == messages


def test_generative_clf_config_key_is_supported():
    tokenizer = FakeTokenizer()
    pipe = make_pipe(tokenizer=tokenizer, model_config={"classification_system_prompt": "GEN-SYS"})

    pipe.preprocess([{"role": "user", "content": "hello"}])

    assert tokenizer.formatted_messages[0][0][0] == {"role": "system", "content": "GEN-SYS"}


# ---------------------------------------------------------------------------
# Plain string inputs
# ---------------------------------------------------------------------------


def test_plain_string_wrapped_by_default():
    tokenizer = FakeTokenizer()
    pipe = make_pipe(tokenizer=tokenizer)

    out = pipe.preprocess("hello")

    assert tokenizer.formatted_messages == [([{"role": "user", "content": "hello"}], True)]
    assert out["input_ids"] == "FORMATTED:1"


def test_plain_string_wrapped_with_explicit_true():
    tokenizer = FakeTokenizer()
    pipe = make_pipe(tokenizer=tokenizer, use_chat_template=True)

    pipe.preprocess("hello")

    assert tokenizer.formatted_messages == [([{"role": "user", "content": "hello"}], True)]


def test_plain_string_rendered_with_user_template():
    tokenizer = FakeTokenizer()
    pipe = make_pipe(
        tokenizer=tokenizer,
        model_config={
            "system_prompt": "SYS",
            "user_template": "Classify this: {text}",
        },
    )

    pipe.preprocess("hello")

    assert tokenizer.formatted_messages == [
        (
            [
                {"role": "system", "content": "SYS"},
                {"role": "user", "content": "Classify this: hello"},
            ],
            True,
        )
    ]


def test_plain_string_rendered_with_generative_user_template():
    tokenizer = FakeTokenizer()
    pipe = make_pipe(
        tokenizer=tokenizer,
        model_config={
            "classification_system_prompt": "GEN-SYS",
            "classification_user_template": "Classify the intent: {utt}",
        },
    )

    pipe.preprocess("hello")

    assert tokenizer.formatted_messages == [
        (
            [
                {"role": "system", "content": "GEN-SYS"},
                {"role": "user", "content": "Classify the intent: hello"},
            ],
            True,
        )
    ]


def test_plain_string_falls_back_to_raw_when_template_has_no_placeholder():
    tokenizer = FakeTokenizer()
    pipe = make_pipe(tokenizer=tokenizer, model_config={"user_template": "No placeholder here"})

    pipe.preprocess("hello")

    assert tokenizer.formatted_messages == [
        ([{"role": "user", "content": "No placeholder here"}], True)
    ]


def test_plain_string_passed_through_when_disabled():
    tokenizer = FakeTokenizer()
    pipe = make_pipe(tokenizer=tokenizer, use_chat_template=False)

    out = pipe.preprocess("hello")

    assert tokenizer.formatted_messages == []
    assert out["input_ids"] == "hello"


def test_plain_string_passed_through_when_config_disables_chat():
    tokenizer = FakeTokenizer()
    pipe = make_pipe(tokenizer=tokenizer, model_config={"classification_use_chat_template": False})

    out = pipe.preprocess("hello")

    assert tokenizer.formatted_messages == []
    assert out["input_ids"] == "hello"


def test_per_call_use_chat_template_override():
    tokenizer = FakeTokenizer()
    pipe = make_pipe(tokenizer=tokenizer, use_chat_template=True)

    out = pipe.preprocess("hello", use_chat_template=False)

    assert tokenizer.formatted_messages == []
    assert out["input_ids"] == "hello"


def test_plain_string_passed_through_when_tokenizer_has_no_template():
    tokenizer = BareTokenizer()
    pipe = make_pipe(tokenizer=tokenizer)

    out = pipe.preprocess("hello")

    assert tokenizer.formatted_messages == []
    assert out["input_ids"] == "hello"


def test_message_input_raises_without_chat_template():
    pipe = make_pipe(tokenizer=BareTokenizer())

    with pytest.raises(ValueError, match="chat template"):
        pipe.preprocess([{"role": "user", "content": "hello"}])


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _mock_transformers_pipeline(monkeypatch):
    captured = {}

    def fake_transformers_pipeline(task, *args, **kwargs):
        captured["task"] = task
        captured["kwargs"] = kwargs
        return "SENTINEL"

    monkeypatch.setattr("echo_dsrn.pipelines.transformers_pipeline", fake_transformers_pipeline)
    return captured


def _mock_config_load(monkeypatch, architectures):
    monkeypatch.setattr(
        "echo_dsrn.pipelines.EchoConfig.from_pretrained",
        classmethod(lambda cls, *a, **k: SimpleNamespace(architectures=architectures)),
    )


def test_factory_uses_chat_pipeline_for_text_classification(monkeypatch):
    captured = _mock_transformers_pipeline(monkeypatch)
    monkeypatch.setattr(
        "echo_dsrn.pipelines.EchoConfig.from_pretrained",
        classmethod(lambda cls, *a, **k: (_ for _ in ()).throw(RuntimeError("no config"))),
    )

    result = pipeline("text-classification", model="fake")

    assert result == "SENTINEL"
    assert captured["kwargs"]["pipeline_class"] is ChatTextClassificationPipeline
    assert captured["kwargs"]["model"] == "fake"  # fallback: unchanged string


def test_factory_does_not_intercept_other_tasks(monkeypatch):
    captured = _mock_transformers_pipeline(monkeypatch)

    result = pipeline("text-generation", model="fake")

    assert result == "SENTINEL"
    assert "pipeline_class" not in captured["kwargs"]


def test_factory_returns_working_chat_pipeline():
    tokenizer = FakeTokenizer()
    config = SimpleNamespace(system_prompt="SYS", _commit_hash="dummy")
    model = FakeModel(config)

    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device="cpu")

    assert isinstance(pipe, ChatTextClassificationPipeline)
    out = pipe.preprocess([{"role": "user", "content": "hello"}])
    assert tokenizer.formatted_messages == [
        ([{"role": "system", "content": "SYS"}, {"role": "user", "content": "hello"}], True)
    ]
    assert out["input_ids"] == "FORMATTED:2"


def test_factory_allows_explicit_pipeline_class_override(monkeypatch):
    captured = _mock_transformers_pipeline(monkeypatch)
    monkeypatch.setattr(
        "echo_dsrn.pipelines.EchoConfig.from_pretrained",
        classmethod(lambda cls, *a, **k: (_ for _ in ()).throw(RuntimeError("no config"))),
    )

    class OtherPipeline:
        pass

    pipeline("text-classification", model="fake", pipeline_class=OtherPipeline)

    assert captured["kwargs"]["pipeline_class"] is OtherPipeline


def test_factory_preloads_architectures_class(monkeypatch):
    captured = _mock_transformers_pipeline(monkeypatch)
    _mock_config_load(monkeypatch, ["EchoForGenerativeClassification"])

    sentinel_model = object()
    monkeypatch.setattr(
        "echo_dsrn.pipelines.EchoForGenerativeClassification.from_pretrained",
        classmethod(lambda cls, *a, **k: sentinel_model),
    )
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: "SENTINEL_TOKENIZER")

    result = pipeline("text-classification", model="some/checkpoint")

    assert result == "SENTINEL"
    assert captured["kwargs"]["model"] is sentinel_model
    assert captured["kwargs"]["tokenizer"] == "SENTINEL_TOKENIZER"
    assert captured["kwargs"]["pipeline_class"] is ChatTextClassificationPipeline


def test_factory_preloads_sequence_classification_class(monkeypatch):
    captured = _mock_transformers_pipeline(monkeypatch)
    _mock_config_load(monkeypatch, ["EchoForSequenceClassification"])

    sentinel_model = object()
    monkeypatch.setattr(
        "echo_dsrn.pipelines.EchoForSequenceClassification.from_pretrained",
        classmethod(lambda cls, *a, **k: sentinel_model),
    )
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: "SENTINEL_TOKENIZER")

    pipeline("text-classification", model="some/checkpoint")

    assert captured["kwargs"]["model"] is sentinel_model
    assert captured["kwargs"]["tokenizer"] == "SENTINEL_TOKENIZER"


def test_factory_skips_preload_for_unknown_architecture(monkeypatch):
    captured = _mock_transformers_pipeline(monkeypatch)
    _mock_config_load(monkeypatch, ["SomeOtherForSequenceClassification"])

    pipeline("text-classification", model="some/checkpoint")

    assert captured["kwargs"]["model"] == "some/checkpoint"
    assert "tokenizer" not in captured["kwargs"]


def test_factory_preload_honors_explicit_tokenizer_and_model_kwargs(monkeypatch):
    captured = _mock_transformers_pipeline(monkeypatch)
    _mock_config_load(monkeypatch, ["EchoForSequenceClassification"])

    sentinel_model = object()
    seen_kwargs = {}

    def fake_from_pretrained(cls, *a, **k):
        seen_kwargs.update(k)
        return sentinel_model

    monkeypatch.setattr(
        "echo_dsrn.pipelines.EchoForSequenceClassification.from_pretrained",
        classmethod(fake_from_pretrained),
    )

    pipeline(
        "text-classification",
        model="some/checkpoint",
        tokenizer="MY_TOKENIZER",
        model_kwargs={"torch_dtype": "bfloat16"},
    )

    assert captured["kwargs"]["tokenizer"] == "MY_TOKENIZER"
    assert seen_kwargs["torch_dtype"] == "bfloat16"
