"""
echo_dsrn/speculative
────────────────────
Cross-vocabulary speculative decoding support: Token-Level Intersection (TLI)
mapping between the Echo-DSRN draft vocabulary and an arbitrary target
vocabulary.
"""

from .vocab_mapper import (
    VocabMapper,
    build_vocab_intersection,
    is_special_token,
    normalize_token_text,
)

__all__ = [
    "VocabMapper",
    "build_vocab_intersection",
    "normalize_token_text",
    "is_special_token",
]
