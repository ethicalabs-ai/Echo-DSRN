"""
echo_dsrn/__init__.py
────────────────────────────────────────────────────────────────────────────
Package init: registers echo_dsrn classes with HuggingFace AutoClass so
that AutoConfig.from_pretrained() and AutoModelForCausalLM.from_pretrained()
work transparently without trust_remote_code=True.
"""

from transformers import AutoConfig, AutoModelForCausalLM

from .configuration_echo import EchoConfig
from .modeling_echo import EchoForCausalLM, EchoModel

# Register with HuggingFace so AutoClass routing works
AutoConfig.register("echo", EchoConfig)
AutoModelForCausalLM.register(EchoConfig, EchoForCausalLM)

__all__ = [
    "EchoConfig",
    "EchoModel",
    "EchoForCausalLM",
]
