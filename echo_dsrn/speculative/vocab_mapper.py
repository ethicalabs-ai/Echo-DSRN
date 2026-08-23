"""
echo_dsrn/speculative/vocab_mapper.py
────────────────────────────────────────────────────────────────────────────
Token-Level Intersection (TLI) mapping for cross-vocabulary speculative
decoding.

When the draft model and the target model use different tokenizers, the draft
can only propose tokens whose *string* exists in both vocabularies.  This
module builds that shared intersection $I$ and provides:

  1. ``mask_logits``            — restrict the draft's output distribution to $I$
  2. ``translate_draft_to_target`` — draft token ids → target token ids (for
     verification against the target)
  3. ``translate_target_to_draft`` — target token ids → draft token ids (for
     building the draft's conditioning context from the target prefix)

The mapping is string-level: both vocabularies are normalized (space markers,
case, Unicode NFC) and tokens whose normalized strings match are considered
equivalent.  This is a heuristic — byte-level tokenizers will only share a
fraction of their vocabularies — but the verification step stays lossless
regardless, because only tokens actually present in $I$ are ever proposed.
"""

import unicodedata
from typing import Dict, Mapping, Optional

import torch

# Space markers used by the two common tokenizer families:
#   \u0120 — byte-level BPE (GPT-2 / Qwen / Llama 3)
#   \u2581 — SentencePiece (Phi / Llama 1-2 style)
_SPACE_MARKERS = ("\u0120", "\u2581")


def normalize_token_text(text: str) -> str:
    """Normalize a token piece for cross-vocabulary string matching."""
    for marker in _SPACE_MARKERS:
        text = text.replace(marker, " ")
    return unicodedata.normalize("NFC", text.strip().lower())


def is_special_token(text: str) -> bool:
    """True for control/special tokens such as ``<s>`` or ``<|endoftext|>``."""
    return len(text) > 1 and text.startswith("<") and text.endswith(">")


class VocabMapper:
    """Bi-directional token-level intersection between draft and target vocabularies.

    Args:
        draft_to_target: mapping of draft vocab ids to target vocab ids for every
            token in the shared intersection $I$.
        draft_vocab_size: size of the draft vocabulary (embedding rows). May be
            larger than the tokenizer's ``vocab_size`` when the model reserves
            extra ids.
        target_vocab_size: size of the target vocabulary.
        draft_unk_id: draft-vocab id used as a fallback when translating target
            tokens that are outside $I$ (e.g. the draft tokenizer's UNK token).
            If ``None``, translating an out-of-intersection target token raises.
    """

    def __init__(
        self,
        draft_to_target: Mapping[int, int],
        draft_vocab_size: int,
        target_vocab_size: int,
        draft_unk_id: Optional[int] = None,
    ):
        self.draft_to_target: Dict[int, int] = dict(draft_to_target)
        self.target_to_draft: Dict[int, int] = {v: k for k, v in self.draft_to_target.items()}
        self.draft_vocab_size = int(draft_vocab_size)
        self.target_vocab_size = int(target_vocab_size)
        self.draft_unk_id = draft_unk_id

        if self.draft_unk_id is not None and not 0 <= self.draft_unk_id < self.draft_vocab_size:
            raise ValueError(
                f"draft_unk_id {self.draft_unk_id} out of range for draft vocab size "
                f"{self.draft_vocab_size}"
            )

        # Lazy per-device tensors.
        self._draft_mask: Optional[torch.Tensor] = None
        self._draft_to_target_tensor: Optional[torch.Tensor] = None
        self._target_to_draft_tensor: Optional[torch.Tensor] = None

    # ── Introspection ─────────────────────────────────────────────────────

    @property
    def intersection_size(self) -> int:
        """Number of shared tokens in $I$."""
        return len(self.draft_to_target)

    # ── Logit masking (draft side) ────────────────────────────────────────

    def mask_logits(self, logits: torch.Tensor, mask_value: float = float("-inf")) -> torch.Tensor:
        """Restrict draft logits to the intersection.

        ``logits`` has shape ``(..., V_draft)``.  Positions outside $I$ are set
        to ``mask_value`` so ``argmax`` can never propose an unmappable token.
        """
        mask = self._get_draft_mask(logits.device)
        return logits.masked_fill(~mask, mask_value)

    # ── Translation ───────────────────────────────────────────────────────

    def translate_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        """Map draft token ids to target token ids (position-preserving).

        ``draft_ids`` must only contain ids in $I$ (masked drafting guarantees
        this); out-of-intersection ids raise :class:`ValueError`.
        """
        lookup = self._get_draft_to_target_tensor(draft_ids.device)
        translated = lookup[draft_ids]
        if bool((translated < 0).any()):
            raise ValueError(
                "draft_ids contains tokens outside the intersection; "
                "mask draft logits with mask_logits() before drafting"
            )
        return translated

    def translate_target_to_draft(
        self,
        target_ids: torch.Tensor,
        unk_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Map target token ids to draft token ids (position-preserving).

        Target tokens outside $I$ map to ``unk_token_id`` (falls back to the
        mapper's ``draft_unk_id``).  If no UNK id is available, out-of-
        intersection ids raise :class:`ValueError`.
        """
        lookup = self._get_target_to_draft_tensor(target_ids.device)
        translated = lookup[target_ids]
        unk = unk_token_id if unk_token_id is not None else self.draft_unk_id
        if unk is not None:
            return torch.where(translated < 0, torch.full_like(translated, unk), translated)
        if bool((translated < 0).any()):
            raise ValueError(
                "target_ids contains tokens outside the intersection and no "
                "unk_token_id was provided"
            )
        return translated

    # ── Lazy tensor helpers ───────────────────────────────────────────────

    def _get_draft_mask(self, device) -> torch.Tensor:
        if self._draft_mask is None or self._draft_mask.device != device:
            mask = torch.zeros(self.draft_vocab_size, dtype=torch.bool)
            mask[list(self.draft_to_target.keys())] = True
            self._draft_mask = mask.to(device)
        return self._draft_mask

    def _get_draft_to_target_tensor(self, device) -> torch.Tensor:
        if self._draft_to_target_tensor is None or self._draft_to_target_tensor.device != device:
            lookup = torch.full((self.draft_vocab_size,), -1, dtype=torch.long)
            for draft_id, target_id in self.draft_to_target.items():
                lookup[draft_id] = target_id
            self._draft_to_target_tensor = lookup.to(device)
        return self._draft_to_target_tensor

    def _get_target_to_draft_tensor(self, device) -> torch.Tensor:
        if self._target_to_draft_tensor is None or self._target_to_draft_tensor.device != device:
            lookup = torch.full((self.target_vocab_size,), -1, dtype=torch.long)
            for draft_id, target_id in self.draft_to_target.items():
                lookup[target_id] = draft_id
            self._target_to_draft_tensor = lookup.to(device)
        return self._target_to_draft_tensor


# ─────────────────────────────────────────────────────────────────────────────
# Builder
# ─────────────────────────────────────────────────────────────────────────────


def _build_text_index(tokenizer, vocab_size: int) -> Dict[str, int]:
    """Map normalized token strings to tokenizer ids (first occurrence wins)."""
    tokens = tokenizer.convert_ids_to_tokens(list(range(vocab_size)))
    index: Dict[str, int] = {}
    for token_id, text in enumerate(tokens):
        if text is None or is_special_token(text):
            continue
        normalized = normalize_token_text(text)
        if not normalized:
            continue
        index.setdefault(normalized, token_id)
    return index


def build_vocab_intersection(
    draft_tokenizer,
    target_tokenizer,
    draft_vocab_size: Optional[int] = None,
    target_vocab_size: Optional[int] = None,
    unk_token_id: Optional[int] = None,
) -> VocabMapper:
    """Build the TLI mapping from two HuggingFace tokenizers.

    String-level matching on normalized token pieces (see
    :func:`normalize_token_text`).  Special tokens and tokens that normalize to
    an empty string are excluded.  ``draft_vocab_size`` defaults to the draft
    tokenizer's ``vocab_size`` — pass the model's ``config.vocab_size`` when the
    embedding is larger than the tokenizer.
    """
    draft_vocab_size = draft_vocab_size or draft_tokenizer.vocab_size
    target_vocab_size = target_vocab_size or target_tokenizer.vocab_size

    draft_index = _build_text_index(draft_tokenizer, draft_tokenizer.vocab_size)
    target_index = _build_text_index(target_tokenizer, target_tokenizer.vocab_size)

    draft_to_target: Dict[int, int] = {}
    for normalized, draft_id in draft_index.items():
        target_id = target_index.get(normalized)
        if target_id is not None:
            draft_to_target[draft_id] = target_id

    if unk_token_id is None:
        unk_token_id = getattr(draft_tokenizer, "unk_token_id", None)

    return VocabMapper(
        draft_to_target,
        draft_vocab_size=draft_vocab_size,
        target_vocab_size=target_vocab_size,
        draft_unk_id=unk_token_id,
    )
