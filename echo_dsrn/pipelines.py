"""
echo_dsrn/pipelines.py
──────────────────────
Chat-template-aware pipeline support.

``transformers.TextClassificationPipeline.preprocess`` feeds raw text straight
to the tokenizer — it never reads the chat prompts baked into the config
(``system_prompt`` / ``user_template``) and never applies the tokenizer's chat
template, so raw ``pipeline("text-classification", ...)`` input is
out-of-distribution for chat-trained classifiers.

:class:`ChatTextClassificationPipeline` closes that gap:

• Message inputs (lists of ``{"role", "content"}`` dicts, and the ``Chat``
  objects the base ``Pipeline.__call__`` builds from them) are rendered with
  ``tokenizer.apply_chat_template(..., add_generation_prompt=True)`` before
  tokenization.
• Plain strings are rendered with the baked user template (like
  ``classify()``) and wrapped as a single user message when
  ``use_chat_template`` is enabled.
• The baked system prompt is prepended when the messages don't define one,
  matching ``EchoForSequenceClassification.classify()``.

Use the module-level :func:`pipeline` factory to opt in:

    from echo_dsrn import pipeline

It injects ``pipeline_class=ChatTextClassificationPipeline`` for the
``"text-classification"`` task and delegates every other task to
``transformers.pipeline`` unchanged.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Type

from transformers import TextClassificationPipeline
from transformers import pipeline as transformers_pipeline

from .configuration_echo import EchoConfig
from .modeling_echo import EchoForSequenceClassification
from .modeling_generative_clf import EchoForGenerativeClassification


class ChatTextClassificationPipeline(TextClassificationPipeline):
    """
    Text-classification pipeline that applies the model's chat template.

    Extends ``TextClassificationPipeline`` so that:

    • message-dict inputs (``[{"role": ..., "content": ...}, ...]``) and the
      ``Chat`` objects the base ``Pipeline.__call__`` builds from them are
      rendered with ``tokenizer.apply_chat_template`` before tokenization;
    • plain strings are rendered with the baked user template (like
      ``classify()``) and wrapped as a single user message when
      ``use_chat_template`` is enabled;
    • the baked ``system_prompt`` is prepended when the messages contain no
      system turn, matching ``EchoForSequenceClassification.classify()``.

    Args:
        use_chat_template (``bool``, *optional*):
            Whether to apply the chat template to plain string inputs.
            ``None`` (the default) defers to
            ``model.config.classification_use_chat_template`` (default
            ``True``). Message inputs are always formatted when the tokenizer
            has a chat template, regardless of this flag.
    """

    def __init__(self, *args, use_chat_template: Optional[bool] = None, **kwargs):
        self.use_chat_template = use_chat_template
        super().__init__(*args, **kwargs)

    def _sanitize_parameters(self, use_chat_template=None, **kwargs):
        preprocess_params, forward_params, postprocess_params = super()._sanitize_parameters(
            **kwargs
        )
        if use_chat_template is not None:
            preprocess_params["use_chat_template"] = use_chat_template
        return preprocess_params, forward_params, postprocess_params

    def preprocess(self, inputs, **tokenizer_kwargs):
        use_chat = tokenizer_kwargs.pop("use_chat_template", None)
        if use_chat is None:
            use_chat = self.use_chat_template
        if use_chat is None:
            use_chat = getattr(self.model.config, "classification_use_chat_template", True)

        messages = self._extract_messages(inputs)
        if messages is not None:
            formatted = self._format_messages(messages)
            if formatted is None:
                raise ValueError(
                    "Message inputs require a tokenizer with a chat template; "
                    f"{type(self.tokenizer).__name__} has none."
                )
            inputs = formatted
        elif use_chat and isinstance(inputs, str):
            formatted = self._format_messages(
                [{"role": "user", "content": self._apply_user_template(inputs)}]
            )
            if formatted is not None:
                inputs = formatted
        return super().preprocess(inputs, **tokenizer_kwargs)

    # ------------------------------------------------------------------
    # Chat-template helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_messages(inputs) -> Optional[List[dict]]:
        """
        Return the message dicts for ``Chat``-like / message-list inputs.

        ``Pipeline.__call__`` already converts a list of message dicts into a
        ``Chat`` object before ``preprocess`` runs; accept both shapes so
        ``preprocess`` can also be driven directly.
        """
        messages = getattr(inputs, "messages", None)
        if (
            messages is None
            and isinstance(inputs, list)
            and inputs
            and all(isinstance(m, dict) and "role" in m for m in inputs)
        ):
            messages = inputs
        return messages

    def _system_prompt(self) -> Optional[str]:
        """Baked system prompt — supports both the LM-CLF and generative-CLF key names."""
        config = self.model.config
        return getattr(config, "system_prompt", None) or getattr(
            config, "classification_system_prompt", None
        )

    def _user_template(self) -> Optional[str]:
        """Baked user-message template — supports both the LM-CLF and generative-CLF key names."""
        config = self.model.config
        return getattr(config, "user_template", None) or getattr(
            config, "classification_user_template", None
        )

    def _apply_user_template(self, text: str) -> str:
        """
        Render a raw string with the baked user template, like ``classify()``.

        The LM-CLF family uses a ``{text}`` placeholder and the generative-CLF
        family a ``{utt}`` one; try both and fall back to the raw string.
        """
        template = self._user_template()
        if not template:
            return text
        try:
            return template.format(text=text)
        except (KeyError, IndexError):
            try:
                return template.format(utt=text)
            except (KeyError, IndexError):
                return text

    def _auto_fill_system_prompt(self, messages: List[dict]) -> List[dict]:
        """Prepend the baked system prompt unless the messages already define one."""
        if any(m.get("role") == "system" for m in messages):
            return messages
        system_prompt = self._system_prompt()
        if system_prompt:
            return [{"role": "system", "content": system_prompt}, *messages]
        return messages

    def _format_messages(self, messages: List[dict]) -> Optional[str]:
        """
        Render messages with the tokenizer's chat template (+ assistant turn).

        Returns ``None`` for tokenizers without a chat template so callers can
        fall back to raw input (matching ``classify()``'s bare-tokenizer path).
        """
        if getattr(self.tokenizer, "chat_template", None) is None:
            return None
        return self.tokenizer.apply_chat_template(
            self._auto_fill_system_prompt(messages),
            add_generation_prompt=True,
            tokenize=False,
        )


# config.architectures → model class. Set at training time and authoritative
# (see the repo convention: never sniff weights to detect the class).
_CLF_ARCHITECTURES: Dict[str, Type] = {
    "EchoForSequenceClassification": EchoForSequenceClassification,
    "EchoForGenerativeClassification": EchoForGenerativeClassification,
}


def _load_echo_classifier(model: str, kwargs: dict):
    """
    Load an echo classifier checkpoint with the class named in its config.

    The transformers factory resolves every echo config to the locally
    registered ``EchoForSequenceClassification`` (via ``_from_pipeline``),
    ignoring ``config.architectures`` — so generative checkpoints load with a
    randomly initialized ``classifier`` head. Pre-loading here with the
    config-named class and passing the instance keeps both families correct.
    """
    trust_remote_code = kwargs.get("trust_remote_code", False)
    try:
        config = EchoConfig.from_pretrained(model, trust_remote_code=trust_remote_code)
    except Exception:
        return model  # not an echo checkpoint — let transformers handle it

    model_class = _CLF_ARCHITECTURES.get((config.architectures or [None])[0])
    if model_class is None:
        return model

    from transformers import AutoTokenizer

    model_kwargs = dict(kwargs.get("model_kwargs") or {})
    torch_dtype = kwargs.get("torch_dtype")
    if torch_dtype is not None:
        if isinstance(torch_dtype, str):
            import torch

            torch_dtype = getattr(torch, torch_dtype)
        model_kwargs.setdefault("dtype", torch_dtype)
        kwargs.pop("torch_dtype", None)
    kwargs.pop("model_kwargs", None)

    kwargs["model"] = model_class.from_pretrained(
        model, trust_remote_code=trust_remote_code, **model_kwargs
    )
    kwargs["tokenizer"] = kwargs.get("tokenizer") or AutoTokenizer.from_pretrained(
        model, trust_remote_code=trust_remote_code
    )
    return kwargs["model"]


def pipeline(task: str, *args, **kwargs):
    """
    ``transformers.pipeline`` wrapper that applies the chat template for
    ``"text-classification"``.

    For that task the pipeline class is swapped to
    :class:`ChatTextClassificationPipeline` and echo checkpoints are loaded
    with the class named in ``config.architectures`` (so both
    ``EchoForSequenceClassification`` and ``EchoForGenerativeClassification``
    work). Every other task is delegated to ``transformers.pipeline``
    unchanged.
    """
    if task == "text-classification":
        kwargs.setdefault("pipeline_class", ChatTextClassificationPipeline)
        model = kwargs.get("model")
        if isinstance(model, str) and kwargs.get("config") is None:
            kwargs["model"] = _load_echo_classifier(model, kwargs)
    return transformers_pipeline(task, *args, **kwargs)
