"""
echo_dsrn/lm_eval_wrapper.py
────────────────────────────────────────────────────────────────────────────
Custom lm-evaluation-harness wrapper that injects surprise_temperature_alpha
into Echo-DSRN model configs before evaluation.

Usage with lm_eval:
    from echo_dsrn.lm_eval_wrapper import EchoDSRNHFLM
    model = EchoDSRNHFLM(
        pretrained="ethicalabs/Echo-DSRN-114M-v0.1.2",
        surprise_temperature_alpha=1.0,
        batch_size=8,
    )

Or via simple_evaluate:
    lm_eval.simple_evaluate(
        model="echo_dsrn_alpha",
        model_args="pretrained=ethicalabs/Echo-DSRN-114M-v0.1.2,alpha=1.0",
        tasks=["arc_easy"],
    )
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM
from transformers import AutoConfig


def _resolve_arch_class(config: AutoConfig) -> type:
    """Resolve the correct Python model class from config.architectures."""
    archs = getattr(config, "architectures", []) or []
    arch_map = {
        "EchoForCausalLM": ("echo_dsrn", "EchoForCausalLM"),
        "EchoForSequenceClassification": ("echo_dsrn", "EchoForSequenceClassification"),
        "HybridEchoForCausalLM": ("echo_hybrid", "HybridEchoForCausalLM"),
        "Qwen3HybridEchoForCausalLM": ("echo_hybrid", "Qwen3HybridEchoForCausalLM"),
    }
    import importlib

    for arch_name, (module_name, class_name) in arch_map.items():
        if arch_name in archs:
            mod = importlib.import_module(module_name)
            return getattr(mod, class_name)

    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM


@register_model("echo_dsrn_alpha")
class EchoDSRNHFLM(HFLM):
    """
    lm-eval wrapper that injects surprise_temperature_alpha into Echo-DSRN configs.

    Extra args (passed via model_args string):
        alpha: float — surprise_temperature_alpha value (default: 0.0)
    """

    def __init__(
        self,
        pretrained: str,
        surprise_temperature_alpha: float = 0.0,
        backend: str = "default",
        batch_size: int = 8,
        max_batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        device: Optional[str] = None,
        dtype: str | torch.dtype | None = "auto",
        trust_remote_code: bool = True,
        **kwargs: Any,
    ):
        # Ensure device is a non-None string for HFLM
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._alpha = surprise_temperature_alpha
        self._pretrained_id = pretrained

        # Accept 'alpha' as alias from model_args string parsing
        if "alpha" in kwargs:
            self._alpha = float(kwargs.pop("alpha"))

        super().__init__(
            pretrained=pretrained,
            backend=backend,
            batch_size=batch_size,
            max_batch_size=max_batch_size,
            max_length=max_length,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

    def _create_model(
        self,
        pretrained: str,
        revision: str | None = "main",
        dtype: str | torch.dtype | None = "auto",
        trust_remote_code: bool = True,
        **kwargs: Any,
    ) -> None:
        """Override to inject surprise_temperature_alpha into config before loading."""
        config = AutoConfig.from_pretrained(
            pretrained, revision=revision, trust_remote_code=trust_remote_code
        )
        config.surprise_temperature_alpha = self._alpha
        if hasattr(config, "output_surprise_gate_logits"):
            config.output_surprise_gate_logits = True

        model_cls = _resolve_arch_class(config)
        dtype_kwargs: dict[str, Any] = {}
        if isinstance(dtype, str) and dtype != "auto":
            dtype_kwargs["torch_dtype"] = getattr(torch, dtype)
        elif isinstance(dtype, torch.dtype):
            dtype_kwargs["torch_dtype"] = dtype

        self._model = model_cls.from_pretrained(
            pretrained,
            revision=revision,
            config=config,
            trust_remote_code=trust_remote_code,
            **dtype_kwargs,
        )
        self._model.eval()

        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained, revision=revision, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self._config = config

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def model_name(self) -> str:
        return f"{self._pretrained_id} (α={self._alpha})"
