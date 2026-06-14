"""
echo_embedding/__init__.py
────────────────────────────────────────────────────────────────────────────
Package init: registers EchoModelForSentenceEmbedding with HuggingFace AutoModel
so that AutoModel.from_pretrained() works transparently using the embedding adapter.
"""

from transformers import AutoConfig, AutoModel

from echo_dsrn.configuration_echo import EchoConfig

from .convert_model import convert_model
from .modeling_embedding import EchoModelForSentenceEmbedding

# Dynamic registration of the embedding model class with HuggingFace
AutoConfig.register("echo", EchoConfig)
AutoModel.register(EchoConfig, EchoModelForSentenceEmbedding)

__all__ = ["EchoModelForSentenceEmbedding", "convert_model"]
