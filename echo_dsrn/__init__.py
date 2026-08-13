"""
echo_dsrn/__init__.py
────────────────────────────────────────────────────────────────────────────
Package init: registers echo_dsrn classes with HuggingFace AutoClass so
that AutoConfig.from_pretrained() and AutoModelForCausalLM.from_pretrained()
work transparently without trust_remote_code=True.
"""

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)

from .configuration_echo import EchoConfig
from .modeling_echo import EchoForCausalLM, EchoForSequenceClassification, EchoModel
from .modeling_generative_clf import EchoForGenerativeClassification
from .pipelines import ChatTextClassificationPipeline, pipeline

# Register with HuggingFace so AutoClass routing works
AutoConfig.register("echo", EchoConfig)
AutoModelForCausalLM.register(EchoConfig, EchoForCausalLM)
AutoModelForSequenceClassification.register(EchoConfig, EchoForSequenceClassification)

__all__ = [
    "EchoConfig",
    "EchoModel",
    "EchoForCausalLM",
    "EchoForSequenceClassification",
    "EchoForGenerativeClassification",
    "ChatTextClassificationPipeline",
    "pipeline",
]
