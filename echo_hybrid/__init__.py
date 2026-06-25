"""
echo_hybrid/__init__.py
────────────────────────────────────────────────────────────────────────────
Package init: registers echo_hybrid classes with HuggingFace AutoClass so
that AutoConfig.from_pretrained() and AutoModelForCausalLM.from_pretrained()
work transparently without trust_remote_code=True.

Usage
─────
    import echo_hybrid  # must be imported before any AutoClass call

    model = AutoModelForCausalLM.from_pretrained(
        "models/Echo-Hybrid-0.5B-Base",
        device_map="auto",
    )
"""

from transformers import AutoConfig, AutoModelForCausalLM

from .configuration_hybrid import HybridEchoConfig, Qwen3HybridEchoConfig
from .dsrn_memory_block import DSRNMemoryInjector
from .modeling_hybrid import (
    HybridEchoCache,
    HybridEchoForCausalLM,
    HybridEchoModel,
    Qwen3HybridEchoForCausalLM,
    Qwen3HybridEchoModel,
)

# Register with HuggingFace so AutoClass routing works
AutoConfig.register("echo_hybrid", HybridEchoConfig)
AutoModelForCausalLM.register(HybridEchoConfig, HybridEchoForCausalLM)
AutoConfig.register("qwen3_echo_hybrid", Qwen3HybridEchoConfig)
AutoModelForCausalLM.register(Qwen3HybridEchoConfig, Qwen3HybridEchoForCausalLM)

__all__ = [
    "HybridEchoConfig",
    "Qwen3HybridEchoConfig",
    "DSRNMemoryInjector",
    "HybridEchoModel",
    "HybridEchoForCausalLM",
    "Qwen3HybridEchoModel",
    "Qwen3HybridEchoForCausalLM",
    "HybridEchoCache",
]
