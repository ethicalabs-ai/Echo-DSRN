"""
tests/test_dspark_rollback.py
────────────────────────────────────────────────────────────────────────────
Tests for the DSpark scheduler's cross-vocabulary (TLI) support and the
state-rollback logic.

Covers:
  1. VocabMapper wiring — proposals restricted to $I$, translated verification
  2. rollback() exactness — rolled-back state == fresh forward over the prefix
  3. rollback() continuation equivalence — logits match after rollback+continue
  4. step() cache contract — target cache truncation/rebuild lengths
  5. Losslessness — spec loop output equals target-greedy reference stream
  6. Full-acceptance clamp (draft state covers prefix minus last token)
"""

import pytest
import torch
import torch.nn as nn

from echo_dsrn.configuration_echo import EchoConfig
from echo_dsrn.dspark_scheduler import (
    DSparkEchoConfig,
    DSparkEchoScheduler,
    _cache_is_truncatable,
)
from echo_dsrn.modeling_echo import EchoForCausalLM
from echo_dsrn.speculative.vocab_mapper import VocabMapper

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

# Draft vocab 32, target vocab 48.  Intersection: draft 0..9 ↔ target 20..29.
DRAFT_TO_TARGET = {i: i + 20 for i in range(10)}


def make_mapper() -> VocabMapper:
    return VocabMapper(DRAFT_TO_TARGET, draft_vocab_size=32, target_vocab_size=48, draft_unk_id=0)


def make_small_config(**overrides) -> EchoConfig:
    defaults = dict(
        embed_dim=32,
        num_layers=2,
        num_heads=2,
        vocab_size=32,
        mlp_ratio=2.0,
        use_hybrid_attention=False,
        use_rmsnorm=False,
    )
    defaults.update(overrides)
    return EchoConfig(**defaults)


def make_draft(**overrides) -> EchoForCausalLM:
    return EchoForCausalLM(make_small_config(**overrides))


def make_scheduler(draft, **config_overrides) -> DSparkEchoScheduler:
    defaults = dict(max_draft_len=4, tau_load=0.5, enable_confidence=False)
    defaults.update(config_overrides)
    return DSparkEchoScheduler(draft, DSparkEchoConfig(**defaults))


class GreedyMockTarget(nn.Module):
    """Stateless mock target with deterministic, position-dependent greedy.

    Greedy token at input position ``t``: even positions are in the
    intersection (target 20..29), odd positions are target-only (40..42) —
    exercising both the translation and the UNK-context paths.  When
    ``return_cache`` is set the forward simulates a real KV cache that is
    extended (not replaced) across calls, matching HF cache semantics.
    """

    def __init__(self, target_vocab_size: int = 48, return_cache: bool = False):
        super().__init__()
        self.target_vocab_size = target_vocab_size
        self.return_cache = return_cache
        self.last_input_ids = None

    def greedy(self, t: int) -> int:
        if t % 2 == 0:
            return 20 + ((t // 2) % 10)
        return 40 + (t % 3)

    def forward(self, input_ids, past_key_values=None, **kwargs):
        self.last_input_ids = input_ids
        B, T = input_ids.shape
        offset = past_key_values.get_seq_length() if past_key_values is not None else 0
        logits = torch.full((B, T, self.target_vocab_size), float("-inf"))
        for t in range(T):
            logits[:, t, self.greedy(offset + t)] = 10.0
        out = type("Out", (), {"logits": logits})()
        if self.return_cache:
            from transformers import DynamicCache

            cache = DynamicCache()
            cache.update(torch.randn(B, 2, offset + T, 8), torch.randn(B, 2, offset + T, 8), 0)
            out.past_key_values = cache
        return out


class FakeLinearStateLayer:
    """Mimics a transformers 5.x linear-attention cache layer.

    Hybrid targets (e.g. gated delta net) carry recurrent/convolutional state
    that cannot be rewound to a shorter prefix, unlike plain (k, v) layers.
    """

    def __init__(self):
        self.conv_states = [torch.zeros(1, 1, 1, 4)]
        self.recurrent_states = [torch.zeros(1, 1, 4)]


class HybridCacheMockTarget(GreedyMockTarget):
    """GreedyMockTarget whose cache also holds a non-truncatable layer."""

    def forward(self, input_ids, past_key_values=None, **kwargs):
        out = super().forward(input_ids, past_key_values=past_key_values, **kwargs)
        if self.return_cache:
            cache = out.past_key_values
            cache.layers.append(FakeLinearStateLayer())
        return out


def run_spec_loop(scheduler, target, prompt_ids, max_new, return_cache=False):
    """Minimal generation loop using step()'s cache contract (batch 1)."""
    target_ids = prompt_ids
    draft_state, target_cache = None, None
    generated, attempted, accepted_total = [], 0, 0
    while len(generated) < max_new:
        r = scheduler.step(
            target_ids,
            target,
            past_key_values=draft_state,
            target_past_key_values=target_cache,
            return_cache=return_cache,
        )
        chunk = r["accepted_tokens"]
        target_ids = torch.cat([target_ids, chunk], dim=1)
        draft_state = r["past_key_values"]
        target_cache = r["target_cache"]
        generated.extend(chunk[0].tolist())
        attempted += scheduler.config.max_draft_len
        accepted_total += int(r["n_accepted"][0])
    return generated, accepted_total, attempted


# ─────────────────────────────────────────────────────────────────────────────
# 1. TLI wiring: masking + translated verification
# ─────────────────────────────────────────────────────────────────────────────


class TestTLIWiring:
    def test_draft_proposals_restricted_to_intersection(self):
        draft = make_draft()
        scheduler = make_scheduler(draft, vocab_mapper=make_mapper())
        input_ids = torch.randint(0, 32, (1, 6))
        draft_ids, _ = scheduler.draft(input_ids)
        for token in draft_ids.flatten().tolist():
            assert token in DRAFT_TO_TARGET, f"draft proposed out-of-I token {token}"

    def test_verify_compares_translated_tokens(self):
        """An in-I draft token is accepted iff it maps to the target's greedy."""
        draft = make_draft()
        scheduler = make_scheduler(draft, vocab_mapper=make_mapper())
        target = GreedyMockTarget()
        # Prompt of length 3: the prediction at position 2 (predicting the
        # first generated token) is greedy(2) = 21 — in the intersection.
        prompt = torch.tensor([[1, 2, 3]])
        draft_ids = torch.tensor([[3]])  # → target 23
        accepted, details = scheduler.verify(target, prompt, draft_ids, return_details=True)
        assert accepted[0, 0].item() is False
        assert details["target_tokens"][0, 0].item() == 21

        draft_ids = torch.tensor([[1]])  # → target 21
        accepted, details = scheduler.verify(target, prompt, draft_ids, return_details=True)
        assert accepted[0, 0].item() is True
        assert details["target_tokens"][0, 0].item() == 21

    def test_verify_cache_mode_feeds_last_prefix_token_plus_drafts(self):
        """With a target cache, only [last prefix token, drafts] are fed forward."""
        from transformers import DynamicCache

        draft = make_draft()
        scheduler = make_scheduler(draft, vocab_mapper=make_mapper())
        target = GreedyMockTarget(return_cache=True)
        prompt = torch.randint(0, 48, (1, 5))
        draft_ids = torch.tensor([[1, 2]])  # → target 21, 22
        cache = DynamicCache()
        cache.update(torch.zeros(1, 2, 5, 8), torch.zeros(1, 2, 5, 8), 0)
        scheduler.verify(target, prompt, draft_ids, past_key_values=cache, return_details=True)
        fed = target.last_input_ids
        assert fed.shape == (1, 3)  # last prefix token + 2 drafts
        assert fed[0, 0].item() == prompt[0, -1].item()
        assert fed[0, 1:].tolist() == [21, 22]

    def test_rollback_requires_prior_draft(self):
        draft = make_draft()
        scheduler = make_scheduler(draft)
        with pytest.raises(RuntimeError):
            scheduler.rollback(torch.tensor([0]))


# ─────────────────────────────────────────────────────────────────────────────
# 2. rollback() exactness
# ─────────────────────────────────────────────────────────────────────────────


class TestRollbackExactness:
    @pytest.mark.parametrize("hybrid", [False, True])
    def test_rollback_matches_fresh_forward(self, hybrid):
        """Rolled-back state equals a fresh forward over the same prefix."""
        draft = make_draft(
            use_hybrid_attention=hybrid, use_rmsnorm=hybrid, output_surprise_gate_logits=True
        )
        scheduler = make_scheduler(draft)
        prompt = torch.randint(0, 32, (1, 6))
        draft_ids, _ = scheduler.draft(prompt)

        for k in [0, 1, 2, 4]:
            state_k = scheduler.rollback(torch.tensor([k]))
            if k == 0:
                assert state_k is None  # fresh start rolls back to None
                continue
            # rollback clamps to max_draft_len - 1 (the final cache never
            # contains the last drafted token).
            m = min(k, scheduler.config.max_draft_len - 1)
            ref = draft(
                input_ids=torch.cat([prompt, draft_ids[:, :m]], dim=1),
                use_cache=True,
                return_dict=True,
            )
            ref_state = list(ref.past_key_values.states)
            for li in range(len(ref_state)):
                for a, b in zip(state_k[li], ref_state[li]):
                    assert torch.allclose(a, b, atol=1e-4), f"layer {li} mismatch"

    @pytest.mark.parametrize("hybrid", [False, True])
    def test_rollback_continuation_matches_fresh(self, hybrid):
        """Logits after continuing from a rolled-back state match a fresh run."""
        draft = make_draft(
            use_hybrid_attention=hybrid, use_rmsnorm=hybrid, output_surprise_gate_logits=True
        )
        scheduler = make_scheduler(draft)
        prompt = torch.randint(0, 32, (1, 6))
        draft_ids, _ = scheduler.draft(prompt)

        for k in [1, 2, 3]:
            state_k = scheduler.rollback(torch.tensor([k]))
            cont = draft(
                input_ids=draft_ids[:, k : k + 1],
                past_key_values=state_k,
                use_cache=True,
                return_dict=True,
            )
            ref = draft(
                input_ids=torch.cat([prompt, draft_ids[:, : k + 1]], dim=1),
                use_cache=True,
                return_dict=True,
            )
            assert torch.allclose(
                cont.logits[:, -1], ref.logits[:, -1], atol=1e-4
            ), f"next-token logits mismatch after rollback to {k}"

    def test_rollback_full_acceptance_returns_last_state(self):
        """rollback(max_draft_len) returns the final cache, which covers the
        accepted chunk minus its last token (that token is re-fed next round)."""
        draft = make_draft(use_hybrid_attention=True, use_rmsnorm=True)
        scheduler = make_scheduler(draft, max_draft_len=4)
        prompt = torch.randint(0, 32, (1, 5))
        draft_ids, _ = scheduler.draft(prompt)
        state_full = scheduler.rollback(torch.tensor([4]))
        ref = draft(
            input_ids=torch.cat([prompt, draft_ids[:, :-1]], dim=1),
            use_cache=True,
            return_dict=True,
        )
        ref_state = list(ref.past_key_values.states)
        for li in range(len(ref_state)):
            for a, b in zip(state_full[li], ref_state[li]):
                assert torch.allclose(a, b, atol=1e-4)

    def test_rollback_with_cache_prefix(self):
        """Rollback from a mid-stream state keeps the original prefix covered."""
        draft = make_draft(use_hybrid_attention=True, use_rmsnorm=True)
        scheduler = make_scheduler(draft)
        prefix = torch.randint(0, 32, (1, 5))
        first, _ = scheduler.draft(prefix)
        state_1 = scheduler.rollback(torch.tensor([1]))  # covers prefix + 1 token
        # Continue drafting one more round from the rolled-back state.
        second, _ = scheduler.draft(first[:, 1:2], past_key_values=state_1)
        state_2 = scheduler.rollback(torch.tensor([0]))  # back to prefix + 1
        # The state after rolling the second round back to 0 must equal the
        # state after the first round's single accepted token.
        for li in range(len(state_2)):
            assert torch.allclose(state_2[li][0], state_1[li][0], atol=1e-4)
            assert torch.allclose(state_2[li][1], state_1[li][1], atol=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# 3. step() cache contract
# ─────────────────────────────────────────────────────────────────────────────


class TestStepCacheContract:
    def test_target_cache_truncated_on_full_acceptance(self):
        draft = make_draft()
        scheduler = make_scheduler(draft, vocab_mapper=make_mapper())
        target = GreedyMockTarget(return_cache=True)
        prompt = torch.randint(0, 48, (1, 5))
        r = scheduler.step(prompt, target, return_cache=True)
        # The returned cache must cover the new stream minus its last token,
        # regardless of how many tokens were accepted.
        expected = prompt.shape[1] + r["accepted_tokens"].shape[1] - 1
        assert r["target_cache"] is not None
        assert r["target_cache"].get_seq_length() == expected

    def test_target_cache_rebuilt_on_partial_acceptance(self):
        draft = make_draft()
        scheduler = make_scheduler(draft, vocab_mapper=make_mapper())
        target = GreedyMockTarget(return_cache=True)
        prompt = torch.tensor([[1, 2, 3]])  # greedy at pos 3 is out of I → rejection
        # First round: build a target cache over the prompt.
        r1 = scheduler.step(prompt, target, return_cache=True)
        new_prefix = torch.cat([prompt, r1["accepted_tokens"]], dim=1)
        # Second round with the returned cache; positions 4+ have in-I greedy
        # every other position, forcing a mix of acceptance and rejection.
        r2 = scheduler.step(
            new_prefix,
            target,
            past_key_values=r1["past_key_values"],
            target_past_key_values=r1["target_cache"],
            return_cache=True,
        )
        expected = new_prefix.shape[1] + r2["accepted_tokens"].shape[1] - 1
        assert r2["target_cache"] is not None
        assert r2["target_cache"].get_seq_length() == expected

    def test_step_returns_target_vocab_tokens(self):
        draft = make_draft()
        scheduler = make_scheduler(draft, vocab_mapper=make_mapper())
        target = GreedyMockTarget()
        prompt = torch.tensor([[1, 2, 3]])
        r = scheduler.step(prompt, target, return_cache=False)
        for token in r["accepted_tokens"][0].tolist():
            assert 0 <= token < target.target_vocab_size
        assert r["n_accepted"][0].item() <= scheduler.config.max_draft_len


# ─────────────────────────────────────────────────────────────────────────────
# 3b. Cache truncatability — hybrid targets fall back to a full re-prefix
# ─────────────────────────────────────────────────────────────────────────────


class TestCacheTruncatability:
    def test_none_is_truncatable(self):
        assert _cache_is_truncatable(None) is True

    def test_legacy_kv_tuples_are_truncatable(self):
        k, v = torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4)
        assert _cache_is_truncatable([(k, v), (k, v)]) is True

    def test_linear_state_layer_is_not_truncatable(self):
        k, v = torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4)
        cache = [(k, v), FakeLinearStateLayer()]
        assert _cache_is_truncatable(cache) is False

    def test_unknown_layer_is_not_truncatable(self):
        assert _cache_is_truncatable([object()]) is False

    def test_dynamic_cache_with_linear_layer_is_not_truncatable(self):
        from transformers import DynamicCache

        cache = DynamicCache()
        cache.update(torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4), 0)
        cache.layers.append(FakeLinearStateLayer())
        assert _cache_is_truncatable(cache) is False

    def test_plain_dynamic_cache_is_truncatable(self):
        from transformers import DynamicCache

        cache = DynamicCache()
        cache.update(torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4), 0)
        assert _cache_is_truncatable(cache) is True


class TestHybridTargetFallback:
    def test_hybrid_target_cache_stays_none(self):
        draft = make_draft()
        scheduler = make_scheduler(draft, vocab_mapper=make_mapper())
        target = HybridCacheMockTarget(return_cache=True)
        prompt = torch.tensor([[1, 2, 3]])
        r = scheduler.step(prompt, target, return_cache=True)
        assert r["target_cache"] is None

    def test_hybrid_target_spec_stream_equals_greedy_reference(self):
        """Re-prefix mode must stay lossless end to end."""
        draft = make_draft()
        scheduler = make_scheduler(draft, vocab_mapper=make_mapper(), max_draft_len=4)
        target = HybridCacheMockTarget()
        prompt = torch.tensor([[1, 2, 3]])
        max_new = 24

        generated, _, _ = run_spec_loop(scheduler, target, prompt, max_new, return_cache=True)

        reference = [target.greedy(prompt.shape[1] - 1 + i) for i in range(max_new)]
        assert len(generated) == max_new, "spec loop stalled"
        assert generated == reference


# ─────────────────────────────────────────────────────────────────────────────
# 4. Losslessness
# ─────────────────────────────────────────────────────────────────────────────


class TestLosslessness:
    def test_spec_stream_equals_target_greedy_reference(self):
        """The generated stream must match the target's own greedy decoding."""
        draft = make_draft()
        scheduler = make_scheduler(draft, vocab_mapper=make_mapper(), max_draft_len=4)
        target = GreedyMockTarget()
        prompt = torch.tensor([[1, 2, 3]])
        max_new = 24

        generated, accepted_total, attempted = run_spec_loop(scheduler, target, prompt, max_new)

        # Reference: greedy decoding of the mock target, position by position.
        reference = [target.greedy(2 + i) for i in range(max_new)]
        assert len(generated) == max_new, "spec loop stalled"
        assert generated == reference, (
            f"spec stream diverged from target-greedy reference\n"
            f"  spec: {generated}\n  ref:  {reference}"
        )
        assert accepted_total <= attempted

    def test_losslessness_with_confidence_enabled(self):
        """The gate-driven dynamic cutoff must not break losslessness."""
        draft = make_draft(output_surprise_gate_logits=True)
        scheduler = make_scheduler(
            draft, vocab_mapper=make_mapper(), max_draft_len=4, enable_confidence=True
        )
        target = GreedyMockTarget()
        prompt = torch.tensor([[1, 2, 3]])
        max_new = 20

        generated, _, _ = run_spec_loop(scheduler, target, prompt, max_new)
        reference = [target.greedy(2 + i) for i in range(max_new)]
        assert generated == reference

    def test_step_never_stalls(self):
        """Every step must emit at least one (possibly replacement) token."""
        draft = make_draft()
        scheduler = make_scheduler(draft, vocab_mapper=make_mapper())
        target = GreedyMockTarget()
        prompt = torch.randint(0, 48, (1, 4))
        for _ in range(3):
            r = scheduler.step(prompt, target, return_cache=False)
            assert r["accepted_tokens"].shape[1] >= 1
            prompt = torch.cat([prompt, r["accepted_tokens"]], dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Full-acceptance clamp
# ─────────────────────────────────────────────────────────────────────────────


class TestFullAcceptanceClamp:
    def test_draft_state_covers_prefix_minus_last_token(self):
        """A fully-accepted round rolls the draft state back one token so the
        next round can re-feed the last token (cache contract)."""
        draft = make_draft(use_hybrid_attention=True, use_rmsnorm=True)
        scheduler = make_scheduler(draft, vocab_mapper=make_mapper(), max_draft_len=4)

        class CopyTarget(nn.Module):
            """Target that always predicts the next input token (accepts all)."""

            def forward(self, input_ids, **kwargs):
                B, T = input_ids.shape
                logits = torch.full((B, T, 48), float("-inf"))
                for t in range(T - 1):
                    logits[:, t, input_ids[0, t + 1]] = 10.0
                logits[:, T - 1, 21] = 10.0  # fallback at the final position
                return type("Out", (), {"logits": logits})()

        prompt = torch.tensor([[1, 2, 3]])
        r = scheduler.step(prompt, CopyTarget(), return_cache=False)
        assert r["n_accepted"][0].item() == scheduler.config.max_draft_len
        # Draft state must cover prompt + draft_len - 1 positions (k/v length).
        state = r["past_key_values"]
        assert state is not None
        kv_len = state[0][2].shape[2]
        assert kv_len == prompt.shape[1] + scheduler.config.max_draft_len - 1

    def test_continuation_after_full_acceptance_matches_fresh(self):
        """Continuing from the clamped state with the last accepted token
        equals a fresh forward over the full accepted stream."""
        draft = make_draft(use_hybrid_attention=True, use_rmsnorm=True)
        scheduler = make_scheduler(draft, vocab_mapper=make_mapper(), max_draft_len=3)

        class CopyTarget(nn.Module):
            def forward(self, input_ids, **kwargs):
                B, T = input_ids.shape
                logits = torch.full((B, T, 48), float("-inf"))
                for t in range(T - 1):
                    logits[:, t, input_ids[0, t + 1]] = 10.0
                logits[:, T - 1, 21] = 10.0
                return type("Out", (), {"logits": logits})()

        prompt = torch.tensor([[1, 2, 3]])
        r = scheduler.step(prompt, CopyTarget(), return_cache=False)
        state = r["past_key_values"]
        accepted = r["accepted_tokens"]  # draft_len tokens, fully accepted
        # Continue with the last accepted token (translated back to the draft
        # vocabulary) from the returned state.
        last_draft_token = scheduler.config.vocab_mapper.translate_target_to_draft(accepted[:, -1:])
        cont = draft(
            input_ids=last_draft_token, past_key_values=state, use_cache=True, return_dict=True
        )
        # Fresh reference: the full accepted stream, translated to the draft
        # vocabulary (the draft conditions on its own tokenizer's encoding).
        ref_stream = scheduler.config.vocab_mapper.translate_target_to_draft(
            torch.cat([prompt, accepted], dim=1)
        )
        ref = draft(input_ids=ref_stream, use_cache=True, return_dict=True)
        assert torch.allclose(cont.logits[:, -1], ref.logits[:, -1], atol=1e-4)
