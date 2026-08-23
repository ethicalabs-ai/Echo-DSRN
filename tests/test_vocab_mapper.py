"""
tests/test_vocab_mapper.py
────────────────────────────────────────────────────────────────────────────
Tests for the cross-vocabulary Token-Level Intersection (TLI) mapping.

Covers:
  1. Normalization and special-token detection
  2. Intersection construction (pure dicts, no tokenizers)
  3. Logit masking — proposals restricted to $I$
  4. draft → target translation accuracy (bijective on $I$)
  5. target → draft translation with UNK fallback
  6. Out-of-intersection error handling
  7. Real-tokenizer intersection build (local_model marker)
"""

import pytest
import torch

from echo_dsrn.speculative.vocab_mapper import (
    VocabMapper,
    build_vocab_intersection,
    is_special_token,
    normalize_token_text,
)

# Draft vocab 32 tokens, target vocab 48 tokens.  Intersection: draft 0..9 ↔
# target 20..29.  Draft UNK = 0.  Target ids 40..42 and all ids < 20 or > 29
# are target-only.
DRAFT_TO_TARGET = {i: i + 20 for i in range(10)}


def make_mapper(**kwargs) -> VocabMapper:
    defaults = dict(draft_vocab_size=32, target_vocab_size=48, draft_unk_id=0)
    defaults.update(kwargs)
    return VocabMapper(DRAFT_TO_TARGET, **defaults)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Normalization
# ─────────────────────────────────────────────────────────────────────────────


class TestNormalization:
    def test_space_markers_are_equivalent(self):
        """Ġ (byte BPE) and ▁ (sentencepiece) both mean space."""
        assert normalize_token_text("\u0120The") == normalize_token_text("\u2581the") == "the"

    def test_case_and_whitespace_insensitive(self):
        assert normalize_token_text("  The  ") == "the"

    def test_unicode_nfc(self):
        """Composed and decomposed accents must match."""
        assert normalize_token_text("caf\u00e9") == normalize_token_text("cafe\u0301")

    def test_special_token_detection(self):
        assert is_special_token("<s>")
        assert is_special_token("<|endoftext|>")
        assert not is_special_token("hello")
        assert not is_special_token("a<b")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Intersection construction
# ─────────────────────────────────────────────────────────────────────────────


class TestMapperConstruction:
    def test_intersection_size(self):
        assert make_mapper().intersection_size == 10

    def test_maps_are_inverses(self):
        mapper = make_mapper()
        assert set(mapper.target_to_draft) == set(range(20, 30))
        for draft_id, target_id in mapper.draft_to_target.items():
            assert mapper.target_to_draft[target_id] == draft_id

    def test_rejects_out_of_range_unk(self):
        with pytest.raises(ValueError):
            VocabMapper({0: 5}, draft_vocab_size=32, target_vocab_size=48, draft_unk_id=99)

    def test_accepts_empty_intersection(self):
        mapper = VocabMapper({}, draft_vocab_size=8, target_vocab_size=8)
        assert mapper.intersection_size == 0


# ─────────────────────────────────────────────────────────────────────────────
# 3. Logit masking
# ─────────────────────────────────────────────────────────────────────────────


class TestLogitMasking:
    def test_argmax_never_leaves_intersection(self):
        mapper = make_mapper()
        for _ in range(5):
            logits = torch.randn(2, 3, 32)
            masked = mapper.mask_logits(logits)
            chosen = masked.argmax(dim=-1)
            for draft_id in chosen.flatten().tolist():
                assert draft_id in mapper.draft_to_target

    def test_masked_logits_are_negative_inf_outside_intersection(self):
        mapper = make_mapper()
        logits = torch.zeros(1, 1, 32)
        masked = mapper.mask_logits(logits)
        outside = [i for i in range(32) if i not in mapper.draft_to_target]
        assert torch.isinf(masked[0, 0, outside]).all()
        inside = [i for i in range(32) if i in mapper.draft_to_target]
        assert (masked[0, 0, inside] == 0.0).all()

    def test_mask_works_on_any_device(self):
        mapper = make_mapper()
        logits = torch.randn(1, 32)
        masked = mapper.mask_logits(logits)
        assert masked.shape == logits.shape

    def test_draft_vocab_larger_than_tokenizer(self):
        """Ids beyond the tokenizer (embedding padding rows) must be masked out."""
        mapper = make_mapper(draft_vocab_size=64)  # rows 32..63 have no mapping
        logits = torch.zeros(1, 64)
        masked = mapper.mask_logits(logits)
        assert torch.isinf(masked[0, 32:]).all()


# ─────────────────────────────────────────────────────────────────────────────
# 4. draft → target translation
# ─────────────────────────────────────────────────────────────────────────────


class TestDraftToTargetTranslation:
    def test_position_preserving(self):
        mapper = make_mapper()
        draft_ids = torch.tensor([[3, 7, 0]])
        translated = mapper.translate_draft_to_target(draft_ids)
        assert translated.tolist() == [[23, 27, 20]]
        assert translated.shape == draft_ids.shape

    def test_bijective_round_trip(self):
        mapper = make_mapper()
        # Round trip on every target id in $I$: target → draft → target.
        for target_id in range(20, 30):
            tensor = torch.tensor([target_id])
            back = mapper.translate_draft_to_target(mapper.translate_target_to_draft(tensor))
            assert back.item() == target_id

    def test_translation_is_exact_on_intersection(self):
        """Every draft id in $I$ must map to a distinct, valid target id."""
        mapper = make_mapper()
        target_ids = mapper.translate_draft_to_target(torch.arange(10))
        assert len(set(target_ids.tolist())) == 10
        assert (target_ids >= 0).all() and (target_ids < mapper.target_vocab_size).all()

    def test_out_of_intersection_raises(self):
        mapper = make_mapper()
        with pytest.raises(ValueError):
            mapper.translate_draft_to_target(torch.tensor([11]))  # 11 not in I


# ─────────────────────────────────────────────────────────────────────────────
# 5. target → draft translation (UNK fallback)
# ─────────────────────────────────────────────────────────────────────────────


class TestTargetToDraftTranslation:
    def test_in_intersection_maps_back(self):
        mapper = make_mapper()
        assert mapper.translate_target_to_draft(torch.tensor([23])).item() == 3

    def test_out_of_intersection_maps_to_unk(self):
        mapper = make_mapper()
        target_ids = torch.tensor([[40, 5, 23]])  # 40, 5 target-only; 23 in I
        translated = mapper.translate_target_to_draft(target_ids)
        assert translated.tolist() == [[0, 0, 3]]  # unk id = 0

    def test_explicit_unk_override(self):
        mapper = make_mapper()
        translated = mapper.translate_target_to_draft(torch.tensor([40]), unk_token_id=7)
        assert translated.item() == 7

    def test_raises_without_unk(self):
        mapper = make_mapper(draft_unk_id=None)
        with pytest.raises(ValueError):
            mapper.translate_target_to_draft(torch.tensor([40]))
        # In-intersection ids still fine without unk
        assert mapper.translate_target_to_draft(torch.tensor([23])).item() == 3


# ─────────────────────────────────────────────────────────────────────────────
# 6. Exact-string verification keys
# ─────────────────────────────────────────────────────────────────────────────


class TestExactStringVerification:
    def test_matches_requires_exact_string(self):
        """Case-colliding tokens ('The' vs 'the') must not verify against
        each other, even when the fuzzy intersection conflates them."""
        # draft 0 = "The", draft 1 = "the"; target 20 = "The", target 21 = "the"
        # The fuzzy map conflates them into representatives, but the exact
        # keys must keep them distinct.
        mapper = VocabMapper(
            {0: 20, 1: 20},  # both fuzzy-map to target 20 ("The")
            draft_vocab_size=4,
            target_vocab_size=24,
            draft_unk_id=0,
            draft_exact_keys={0: 0, 1: 1},
            target_exact_keys={20: 0, 21: 1},
        )
        assert bool(mapper.matches_draft_to_target(torch.tensor([0]), torch.tensor([20])))
        assert bool(mapper.matches_draft_to_target(torch.tensor([1]), torch.tensor([21])))
        assert not bool(mapper.matches_draft_to_target(torch.tensor([0]), torch.tensor([21])))
        assert not bool(mapper.matches_draft_to_target(torch.tensor([1]), torch.tensor([20])))

    def test_matches_falls_back_to_translated_ids_without_keys(self):
        mapper = make_mapper()  # no exact keys
        accepted = mapper.matches_draft_to_target(torch.tensor([3]), torch.tensor([23]))
        assert bool(accepted)
        rejected = mapper.matches_draft_to_target(torch.tensor([3]), torch.tensor([22]))
        assert not bool(rejected)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Real-tokenizer build
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.local_model
class TestBuildVocabIntersection:
    def test_build_from_real_tokenizers(self):
        """Draft (Phi-3.5 32k) vs Qwen2.5 (152k) share a healthy intersection."""
        from transformers import AutoTokenizer

        draft_tok = AutoTokenizer.from_pretrained(
            "ethicalabs/Echo-DSRN-114M-v0.1.2", trust_remote_code=True
        )
        target_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)

        mapper = build_vocab_intersection(draft_tok, target_tok)

        assert mapper.intersection_size > 5000
        assert mapper.draft_vocab_size == draft_tok.vocab_size
        assert mapper.target_vocab_size == target_tok.vocab_size
        # Common English words must survive normalization matching.  Mirror
        # the builder's first-wins semantics (setdefault), not last-wins.
        draft_index = {}
        for i in range(draft_tok.vocab_size):
            normalized = normalize_token_text(draft_tok.convert_ids_to_tokens(i))
            if normalized:
                draft_index.setdefault(normalized, i)
        for word in (" the", " is", " of", " and", " to"):
            normalized = normalize_token_text(word)
            if normalized in draft_index:
                draft_id = draft_index[normalized]
                assert draft_id in mapper.draft_to_target, f"'{word}' missing from $I$"
        # All target ids in the mapping must be valid target ids.
        target_ids = list(mapper.draft_to_target.values())
        assert all(0 <= tid < target_tok.vocab_size for tid in target_ids)

    def test_build_respects_draft_vocab_size_override(self):
        """The mapper can mask over embedding rows the tokenizer cannot decode."""
        from transformers import AutoTokenizer

        draft_tok = AutoTokenizer.from_pretrained(
            "ethicalabs/Echo-DSRN-114M-v0.1.2", trust_remote_code=True
        )
        target_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)

        mapper = build_vocab_intersection(
            draft_tok, target_tok, draft_vocab_size=draft_tok.vocab_size + 17
        )
        assert mapper.draft_vocab_size == draft_tok.vocab_size + 17
        # Unmapped tail rows must be masked out of the logits.
        logits = torch.zeros(1, mapper.draft_vocab_size)
        masked = mapper.mask_logits(logits)
        assert torch.isinf(masked[0, draft_tok.vocab_size :]).all()
