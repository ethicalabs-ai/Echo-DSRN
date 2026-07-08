"""
tests/test_dspark_confidence.py
────────────────────────────────────────────────────────────────────────────
Tests for Echo-DSRN DSpark confidence integration.

Covers:
  1. Config flags and backward compatibility (output_surprise_gate_logits=False)
  2. Gate logits exposure from base DSRN and hybrid models
  3. Confidence math: λ_t, c_t, log-space survival, dynamic cutoff
  4. Surprise spike → early cutoff verification
  5. DSparkEchoScheduler draft/verify pipeline
"""

import pytest
import torch
import torch.nn as nn

from echo_dsrn.confidence import (
    aggregate_gate_logits,
    compute_confidence,
    dynamic_cutoff,
    extract_gate_logits,
    log_prefix_survival,
    raw_lambda,
    token_confidence,
)
from echo_dsrn.configuration_echo import EchoConfig
from echo_dsrn.dspark_scheduler import DSparkEchoConfig, DSparkEchoScheduler
from echo_dsrn.modeling_echo import (
    DSRNBlock,
    EchoForCausalLM,
    EchoModel,
    dsrn_parallel_kernel_hybrid,
    dsrn_parallel_kernel_legacy,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def make_small_config(**overrides) -> EchoConfig:
    defaults = dict(
        embed_dim=64,
        num_layers=2,
        num_heads=2,
        vocab_size=1000,
        mlp_ratio=2.0,
        use_hybrid_attention=False,
        use_rmsnorm=False,  # Use LayerNorm for legacy kernel compatibility
    )
    defaults.update(overrides)
    return EchoConfig(**defaults)


def make_dsrn_block(**overrides) -> DSRNBlock:
    config = make_small_config(**overrides)
    return DSRNBlock(config)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Config flags and backward compatibility
# ─────────────────────────────────────────────────────────────────────────────


class TestConfigBackwardCompatibility:
    def test_flag_defaults_to_false(self):
        """output_surprise_gate_logits must default to False."""
        config = EchoConfig()
        assert config.output_surprise_gate_logits is False

    def test_flag_persists_in_save_load(self, tmp_path):
        """Flag should survive config serialization round-trip."""
        config = EchoConfig(output_surprise_gate_logits=True, embed_dim=64)
        config.save_pretrained(tmp_path)
        loaded = EchoConfig.from_pretrained(tmp_path)
        assert loaded.output_surprise_gate_logits is True

        config_off = EchoConfig(output_surprise_gate_logits=False, embed_dim=64)
        config_off.save_pretrained(tmp_path)
        loaded_off = EchoConfig.from_pretrained(tmp_path)
        assert loaded_off.output_surprise_gate_logits is False


# ─────────────────────────────────────────────────────────────────────────────
# 2. Kernel gate_logits exposure
# ─────────────────────────────────────────────────────────────────────────────


class TestKernelGateLogits:
    @pytest.fixture
    def block(self):
        return make_dsrn_block()

    @pytest.fixture
    def inputs(self):
        B, T = 2, 8
        D = 64
        x = torch.randn(B, T, D)
        h_prev = torch.zeros(B, D)
        c_prev = torch.zeros(B, D * 2)  # state_size = D * num_heads
        return x, h_prev, c_prev

    def test_legacy_kernel_returns_gate_logits(self, block, inputs):
        """Legacy kernel must return gate_logits as the 7th element."""
        x, h_prev, c_prev = inputs
        result = dsrn_parallel_kernel_legacy(block, x, h_prev, c_prev)
        assert len(result) == 7, f"Expected 7 outputs, got {len(result)}"
        gate_logits = result[6]
        B, T, D = x.shape
        state_size = D * 2  # num_heads=2
        assert gate_logits.shape == (B, T, state_size)

    def test_hybrid_kernel_returns_gate_logits(self, block, inputs):
        """Hybrid kernel must return gate_logits as the 7th element."""
        block.use_rmsnorm = True
        x, h_prev, c_prev = inputs
        result = dsrn_parallel_kernel_hybrid(block, x, h_prev, c_prev)
        assert len(result) == 7
        gate_logits = result[6]
        B, T, D = x.shape
        state_size = D * 2
        assert gate_logits.shape == (B, T, state_size)

    def test_gate_logits_are_pre_sigmoid(self, block, inputs):
        """gate_logits must be raw pre-activation values (not sigmoided).
        A sigmoid-bounded tensor would be strictly in (0, 1).
        Values > 1.0 prove they are raw logits."""
        x, h_prev, c_prev = inputs
        result = dsrn_parallel_kernel_legacy(block, x, h_prev, c_prev)
        gate_logits = result[6]
        # gate_logits should extend beyond (0, 1) — raw logits, not sigmoid outputs
        assert gate_logits.max() > 1.0, (
            f"gate_logits max={gate_logits.max().item():.4f} ≤ 1.0 — "
            "may already be sigmoid-bounded"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. DSRNBlock gate_logits propagation
# ─────────────────────────────────────────────────────────────────────────────


class TestDSRNBlockGateLogits:
    @pytest.fixture
    def block(self):
        return make_dsrn_block()

    @pytest.fixture
    def state_prev(self):
        D = 64
        h = torch.zeros(2, D)
        c = torch.zeros(2, D * 2)  # state_size
        return (h, c)

    @pytest.fixture
    def x(self):
        return torch.randn(2, 8, 64)

    def test_block_without_flag_returns_3_elements(self, block, x, state_prev):
        """Default: DSRNBlock returns (x_out, state, gate_stats) — 3 elements."""
        out = block(x, state_prev)
        assert len(out) == 3, f"Expected 3 outputs, got {len(out)}"
        assert out[0].shape == x.shape
        assert len(out[1]) == 2  # (h, c)

    def test_block_with_output_all_states_no_flag(self, block, x, state_prev):
        """output_all_states=True, flag=False → 5 elements."""
        out = block(x, state_prev, output_all_states=True)
        assert len(out) == 5, f"Expected 5 outputs, got {len(out)}"

    def test_block_with_flag_returns_4_elements(self, block, x, state_prev):
        """output_gate_logits=True → 4 elements (gate_logits appended)."""
        out = block(x, state_prev, output_gate_logits=True)
        assert len(out) == 4, f"Expected 4 outputs, got {len(out)}"
        gate_logits = out[3]
        assert gate_logits.shape == (2, 8, block.state_size)

    def test_block_with_flag_and_output_all_states(self, block, x, state_prev):
        """Both flags → 6 elements."""
        out = block(x, state_prev, output_all_states=True, output_gate_logits=True)
        assert len(out) == 6, f"Expected 6 outputs, got {len(out)}"
        gate_logits = out[5]
        assert gate_logits.shape == (2, 8, block.state_size)


# ─────────────────────────────────────────────────────────────────────────────
# 4. EchoModel gate_logits propagation
# ─────────────────────────────────────────────────────────────────────────────


class TestEchoModelGateLogits:
    @pytest.fixture
    def model(self):
        config = make_small_config(output_surprise_gate_logits=True)
        return EchoModel(config)

    @pytest.fixture
    def input_ids(self):
        return torch.randint(0, 1000, (2, 8))

    def test_model_output_has_all_gate_logits(self, model, input_ids):
        """With flag=True, EchoModel output must have all_gate_logits."""
        out = model(input_ids, return_dict=True)
        assert hasattr(out, "all_gate_logits"), "output missing all_gate_logits"
        gl = out.all_gate_logits
        assert (
            len(gl) == model.num_layers
        ), f"Expected {model.num_layers} gate_logits, got {len(gl)}"
        for layer_gl in gl:
            assert layer_gl.shape == (2, 8, model.state_dim)

    def test_model_without_flag_no_gate_logits(self, input_ids):
        """With flag=False, output must NOT have all_gate_logits."""
        config = make_small_config(output_surprise_gate_logits=False)
        model = EchoModel(config)
        out = model(input_ids, return_dict=True)
        assert not hasattr(out, "all_gate_logits") or out.all_gate_logits is None


# ─────────────────────────────────────────────────────────────────────────────
# 5. EchoForCausalLM gate_logits propagation
# ─────────────────────────────────────────────────────────────────────────────


class TestEchoForCausalLMGateLogits:
    @pytest.fixture
    def model(self):
        config = make_small_config(output_surprise_gate_logits=True)
        return EchoForCausalLM(config)

    @pytest.fixture
    def input_ids(self):
        return torch.randint(0, 1000, (2, 8))

    def test_causallm_output_has_all_gate_logits(self, model, input_ids):
        """EchoForCausalLM must propagate gate_logits from EchoModel."""
        out = model(input_ids, return_dict=True)
        assert hasattr(out, "all_gate_logits"), "CausalLM output missing all_gate_logits"
        assert out.all_gate_logits is not None
        assert len(out.all_gate_logits) == model.model.num_layers

    def test_causallm_forward_without_labels(self, model, input_ids):
        """Forward without labels should still return gate_logits."""
        out = model(input_ids, labels=None, return_dict=True)
        assert out.loss is None
        assert hasattr(out, "all_gate_logits")

    def test_causallm_lm_head_still_works(self, model, input_ids):
        """LM head logits should still work with gate_logits enabled."""
        out = model(input_ids, return_dict=True)
        assert out.logits.shape == (2, 8, 1000)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Confidence math: stateless functions
# ─────────────────────────────────────────────────────────────────────────────


class TestConfidenceMath:
    def test_raw_lambda_in_range(self):
        """λ_t = σ(z_t) must be in (0, 1]."""
        z = torch.randn(2, 4, 128) * 5.0
        lam = raw_lambda(z)
        assert (lam > 0).all()
        assert (lam <= 1).all()

    def test_token_confidence_identity(self):
        """c_t = 1 − λ_t = 1 − σ(z_t)."""
        z = torch.randn(2, 4, 128)
        c = token_confidence(z)
        expected = 1.0 - torch.sigmoid(z)
        assert torch.allclose(c, expected, atol=1e-6)

    def test_log_prefix_survival_monotonic(self):
        """S_k must be monotonically decreasing."""
        z = torch.randn(2, 8, 128)
        _, S = log_prefix_survival(z)
        diffs = S[:, 1:] - S[:, :-1]
        assert (diffs <= 0).all(), "S_k must be monotonically decreasing"

    def test_log_prefix_survival_starts_at_one_minus_lambda(self):
        """S_1 = 1 − λ_1."""
        z = torch.randn(2, 4, 128)
        _, S = log_prefix_survival(z)
        lam = raw_lambda(z)
        expected_S1 = 1.0 - lam[:, 0, :]
        assert torch.allclose(S[:, 0, :], expected_S1, atol=1e-5)

    def test_dynamic_cutoff_all_survive(self):
        """When all S_k ≥ τ, cutoff should be full sequence length."""
        z = -10.0 * torch.ones(2, 8, 128)
        cutoff = dynamic_cutoff(z, tau_load=0.1)
        assert (cutoff == 8).all(), f"Expected full cutoff (8), got {cutoff}"

    def test_dynamic_cutoff_lambda_spike_forces_early_cutoff(self):
        """A λ spike at position 3 must force cutoff ≤ 4."""
        B, T, D = 2, 8, 128
        z = -3.0 * torch.ones(B, T, D)
        z[:, 3, :] = 10.0  # Spike → λ ≈ 1.0
        cutoff = dynamic_cutoff(z, tau_load=0.5)
        assert (cutoff <= 4).all(), f"Spike should force cutoff ≤ 4, got {cutoff}"

    def test_aggregation_mean(self):
        """Mean aggregation should average across layers."""
        g1 = torch.ones(2, 3, 4)
        g2 = 3.0 * torch.ones(2, 3, 4)
        result = aggregate_gate_logits([g1, g2])
        assert torch.allclose(result, 2.0 * torch.ones(2, 3, 4))

    def test_aggregation_last(self):
        """Last aggregation uses only the final layer."""
        g1 = torch.ones(2, 3, 4)
        g2 = 5.0 * torch.ones(2, 3, 4)
        result = aggregate_gate_logits([g1, g2], aggregation="last")
        assert torch.allclose(result, g2)

    def test_aggregation_first(self):
        """First aggregation uses only the first layer."""
        g1 = 7.0 * torch.ones(2, 3, 4)
        g2 = torch.ones(2, 3, 4)
        result = aggregate_gate_logits([g1, g2], aggregation="first")
        assert torch.allclose(result, g1)

    def test_compute_confidence_pipeline(self):
        """Full pipeline: aggregate → λ → confidence → survival."""
        gl = [torch.randn(2, 4, 8)]
        result = compute_confidence(gl, tau_load=0.3, return_cutoff=True)
        assert "raw_lambda" in result
        assert "token_confidence" in result
        assert "prefix_survival" in result
        assert "cutoff_lens" in result
        assert result["cutoff_lens"].shape == (2,)


# ─────────────────────────────────────────────────────────────────────────────
# 7. extract_gate_logits helper
# ─────────────────────────────────────────────────────────────────────────────


class TestExtractGateLogits:
    def test_extract_from_output_with_attr(self):
        fake_output = type("Fake", (), {"all_gate_logits": ["foo"]})()
        assert extract_gate_logits(fake_output) == ["foo"]

    def test_extract_from_tuple(self):
        fake_tensor = torch.randn(2, 4, 8)
        result = extract_gate_logits((None, None, [fake_tensor]))
        assert result is not None and len(result) == 1

    def test_extract_none_when_missing(self):
        assert extract_gate_logits(type("Fake", (), {})()) is None


# ─────────────────────────────────────────────────────────────────────────────
# 8. DSparkEchoScheduler
# ─────────────────────────────────────────────────────────────────────────────


class TestDSparkScheduler:
    @pytest.fixture
    def draft_model(self):
        config = make_small_config(output_surprise_gate_logits=True)
        return EchoForCausalLM(config)

    @pytest.fixture
    def scheduler(self, draft_model):
        return DSparkEchoScheduler(
            draft_model=draft_model,
            config=DSparkEchoConfig(max_draft_len=4, tau_load=0.5),
        )

    def test_scheduler_creation(self, scheduler, draft_model):
        assert scheduler.draft_model is draft_model
        assert scheduler.config.max_draft_len == 4
        assert draft_model.config.output_surprise_gate_logits is True

    def test_scheduler_draft(self, scheduler):
        input_ids = torch.randint(0, 1000, (2, 4))
        draft_ids, conf = scheduler.draft(input_ids)
        assert draft_ids.shape == (2, 4)
        assert "token_confidence" in conf or not conf

    def test_scheduler_step(self, scheduler):
        """Full draft→verify step with self as target."""
        input_ids = torch.randint(0, 1000, (2, 4))
        result = scheduler.step(input_ids, scheduler.draft_model)
        assert "accepted_tokens" in result
        assert "accepted_mask" in result
        assert "draft_ids" in result

    def test_verify_cutoff_respects_length(self, scheduler):
        input_ids = torch.randint(0, 1000, (2, 4))
        draft_ids = torch.randint(0, 1000, (2, 4))
        cutoff_lens = torch.tensor([1, 3])
        prefix_len = input_ids.shape[1]

        class AcceptAllModel(nn.Module):
            def __call__(self, input_ids, **kwargs):
                B, T = input_ids.shape
                logits = torch.full((B, T, 1000), float("-inf"))
                for b in range(B):
                    for t in range(draft_ids.shape[1]):
                        pos = prefix_len - 1 + t
                        if pos < T:
                            logits[b, pos, draft_ids[b, t]] = 10.0
                return type("Fake", (), {"logits": logits})()

        accepted = scheduler.verify(AcceptAllModel(), input_ids, draft_ids, cutoff_lens)
        assert accepted[0, 0].item() is True
        assert accepted[0, 1].item() is False
        assert accepted[1, 0].item() is True
        assert accepted[1, 2].item() is True
        assert accepted[1, 3].item() is False


# ─────────────────────────────────────────────────────────────────────────────
# 9. End-to-end: surprise spike forces early termination
# ─────────────────────────────────────────────────────────────────────────────


class TestSurpriseSpikeCutoff:
    def test_spike_cutoff(self):
        B, T, D = 2, 8, 4
        z = -2.0 * torch.ones(B, T, D)
        z[:, 3, :] = 5.0
        cutoff = dynamic_cutoff(z, tau_load=0.3)
        assert (cutoff <= 4).all(), f"Spike should force cutoff ≤ 4, got {cutoff}"

    def test_no_spike_full_survival(self):
        z = -3.0 * torch.ones(2, 8, 4)
        cutoff = dynamic_cutoff(z, tau_load=0.1)
        assert (cutoff == 8).all()

    def test_log_space_no_underflow(self):
        z = 100.0 * torch.ones(2, 16, 128)
        log_S, S = log_prefix_survival(z)
        assert torch.isfinite(log_S).all(), "log_S has non-finite values"
        assert S[:, -1, :].max() < 1e-3
