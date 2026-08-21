"""
Tests for pre-training initialization fixes:

1. ``surprise_lambda_init`` is honored by the HF wrapper (previously hardcoded
   to zeros) — one source of truth between trainer CLI and model construction.
2. ``eos_mask`` threads through the pure-DSRN forward path (previously only the
   hybrid path wired it): the kernels wipe the fast state, suppress slow-state
   writes at document boundaries, and zero the inter-chunk carry when the chunk
   ends on EOS.

These run on CPU and exercise the PyTorch scan path (Triton only activates on
GPU); the kernels under test are shared (``dsrn_parallel_kernel_legacy``).
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from echo_dsrn.configuration_echo import EchoConfig
from echo_dsrn.modeling_echo import EchoForCausalLM

EOS_ID = 9


@pytest.fixture
def tiny_echo():
    config = EchoConfig(
        vocab_size=128,
        embed_dim=32,
        num_layers=2,
        num_heads=2,
        mlp_ratio=4,
        max_position_embeddings=128,
    )
    model = EchoForCausalLM(config)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# surprise_lambda init
# ---------------------------------------------------------------------------


def test_surprise_lambda_init_default_is_zero():
    config = EchoConfig(
        vocab_size=128,
        embed_dim=32,
        num_layers=2,
        num_heads=2,
        max_position_embeddings=128,
    )
    model = EchoForCausalLM(config)
    for block in model.model.blocks:
        assert torch.all(block.surprise_lambda == 0.0)


def test_surprise_lambda_init_honored_from_config():
    config = EchoConfig(
        vocab_size=128,
        embed_dim=32,
        num_layers=2,
        num_heads=2,
        max_position_embeddings=128,
        surprise_lambda_init=0.5,
    )
    model = EchoForCausalLM(config)
    for block in model.model.blocks:
        assert torch.all(block.surprise_lambda == 0.5)


# ---------------------------------------------------------------------------
# eos_mask state reset (pure-DSRN forward path)
# ---------------------------------------------------------------------------


def _forward_states(model, input_ids, eos_mask):
    """Run EchoModel with output_all_states, return (h_all, c_all) per layer."""
    x, next_states, c_states, gate_stats, h_all, c_all = model.model(
        input_ids=input_ids,
        eos_mask=eos_mask,
        output_all_states=True,
        output_dsrn_telemetry=True,
        return_dict=False,
    )
    return h_all, c_all, next_states


def test_eos_mask_resets_states_at_document_boundary(tiny_echo):
    model = tiny_echo
    # EOS at position 3; compare mask vs no-mask runs
    input_ids = torch.tensor([[1, 2, 3, EOS_ID, 4, 5, 6, 7]])
    eos_mask = input_ids == EOS_ID

    h_all, c_all, _ = _forward_states(model, input_ids, eos_mask)
    h_nomask, c_nomask, _ = _forward_states(model, input_ids, None)

    for layer, (h, c) in enumerate(zip(h_all, c_all)):
        # Positions before the boundary are untouched by the mask.
        assert torch.allclose(h[0, :4], h_nomask[layer][0, :4])
        assert torch.allclose(c[0, :4], c_nomask[layer][0, :4])
        # Post-EOS position (t=4): slow-state write is suppressed -> c unchanged.
        assert torch.allclose(c[0, 4], c[0, 3])
        # The mask must change both states at the boundary vs the no-mask run.
        assert not torch.allclose(c[0, 4], c_nomask[layer][0, 4])
        assert not torch.allclose(h[0, 4], h_nomask[layer][0, 4])


def test_eos_mask_zeroes_inter_chunk_carry(tiny_echo):
    model = tiny_echo
    input_ids = torch.tensor([[1, 2, 3, 4, 5, EOS_ID]])
    eos_mask = input_ids == EOS_ID

    _, _, next_states = _forward_states(model, input_ids, eos_mask)

    for state in next_states:
        h_new, c_new = state[0], state[1]
        assert torch.all(h_new == 0.0), "fast state must be zeroed after trailing EOS"
        assert torch.all(c_new == 0.0), "slow state must be zeroed after trailing EOS"


def test_no_eos_mask_keeps_evolving_state(tiny_echo):
    model = tiny_echo
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])

    _, c_all, _ = _forward_states(model, input_ids, None)

    # Without a mask, the slow state evolves at every position.
    for c in c_all:
        assert not torch.allclose(c[0, 5], c[0, 4])
