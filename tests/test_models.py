import os

# Ensure our local packages are importable (for testing purposes before pip install)
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# We must import the packages to trigger HF AutoClass registrations
from echo_dsrn.configuration_echo import EchoConfig
from echo_dsrn.modeling_echo import EchoForCausalLM
from echo_hybrid.configuration_hybrid import HybridEchoConfig
from echo_hybrid.modeling_hybrid import HybridEchoForCausalLM


@pytest.fixture
def tiny_echo_dsrn():
    """Create a tiny, randomly initialized Echo-DSRN model for fast testing."""
    config = EchoConfig(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
        window_size=32,
    )
    # Instantiate without from_pretrained to get a fast randomly initialized model
    model = EchoForCausalLM(config)
    model.eval()
    return model


@pytest.fixture
def tiny_echo_hybrid():
    """Create a tiny, randomly initialized Echo-Hybrid model for fast testing."""
    config = HybridEchoConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
        window_size=32,
    )
    model = HybridEchoForCausalLM(config)
    model.eval()
    return model


def test_echo_dsrn_forward_pass(tiny_echo_dsrn):
    """Test that a forward pass through Echo-DSRN works and returns valid logits."""
    model = tiny_echo_dsrn
    batch_size = 2
    seq_len = 10

    # Create dummy input ids
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))

    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    assert outputs.logits.shape == (batch_size, seq_len, 1000)
    assert not torch.isnan(outputs.logits).any()


def test_echo_dsrn_generation(tiny_echo_dsrn):
    """Test that Echo-DSRN can successfully generate new tokens."""
    model = tiny_echo_dsrn
    batch_size = 1
    seq_len = 5

    input_ids = torch.randint(0, 1000, (batch_size, seq_len))

    with torch.no_grad():
        generated = model.generate(input_ids, max_new_tokens=3, do_sample=False, use_cache=True)

    assert generated.shape == (batch_size, seq_len + 3)


def test_echo_hybrid_forward_pass(tiny_echo_hybrid):
    """Test that a forward pass through Echo-Hybrid works and returns valid logits."""
    model = tiny_echo_hybrid
    batch_size = 2
    seq_len = 10

    input_ids = torch.randint(0, 1000, (batch_size, seq_len))

    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    assert outputs.logits.shape == (batch_size, seq_len, 1000)
    assert not torch.isnan(outputs.logits).any()


def test_echo_hybrid_generation(tiny_echo_hybrid):
    """Test that Echo-Hybrid can successfully generate new tokens."""
    model = tiny_echo_hybrid
    batch_size = 1
    seq_len = 5

    input_ids = torch.randint(0, 1000, (batch_size, seq_len))

    with torch.no_grad():
        generated = model.generate(input_ids, max_new_tokens=3, do_sample=False, use_cache=True)

    assert generated.shape == (batch_size, seq_len + 3)


def test_autoclass_registration(tiny_echo_dsrn, tiny_echo_hybrid):
    """Verify that the models are properly registered with HuggingFace's AutoClasses."""
    from transformers import AutoModelForCausalLM

    # Check echo_dsrn registration
    dsrn_model = AutoModelForCausalLM.from_config(tiny_echo_dsrn.config)
    assert isinstance(dsrn_model, type(tiny_echo_dsrn))

    # Check echo_hybrid registration
    hybrid_model = AutoModelForCausalLM.from_config(tiny_echo_hybrid.config)
    assert isinstance(hybrid_model, type(tiny_echo_hybrid))


def test_backward_compat_no_mlp_bias():
    """v0.1.2-style checkpoint: mlp_bias=False (default) → no bias tensors exist,
    and even if somehow created, the from_pretrained guard zeros them."""
    config = EchoConfig(
        vocab_size=1000,
        hidden_size=64,
        num_layers=2,
        num_heads=2,
        mlp_bias=False,
    )
    model = EchoForCausalLM(config)

    # mlp_bias=False — no bias tensors should have been allocated at all.
    for name, param in model.named_parameters():
        if "mlp" in name and "bias" in name:
            # Bias tensors must not exist when mlp_bias=False.
            raise AssertionError(
                f"{name} exists but mlp_bias=False — DSRNBlock is ignoring the config flag."
            )

    # Confirm the model forward pass still works cleanly with no biases.
    input_ids = torch.randint(0, 1000, (1, 8))
    with torch.no_grad():
        out = model(input_ids=input_ids)
    assert not torch.isnan(out.logits).any(), "logits contain NaN with mlp_bias=False"
    assert not torch.isinf(out.logits).any(), "logits contain Inf with mlp_bias=False"


def test_mlp_bias_true_roundtrip():
    """Config with mlp_bias=True: bias tensors must survive a save + from_pretrained roundtrip."""
    import tempfile

    config = EchoConfig(
        vocab_size=1000,
        hidden_size=64,
        num_layers=2,
        num_heads=2,
        mlp_bias=True,
    )
    model = EchoForCausalLM(config)

    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        loaded = EchoForCausalLM.from_pretrained(tmp)

    bias_count = 0
    for name, param in loaded.named_parameters():
        if "mlp" in name and "bias" in name:
            assert param is not None, f"{name} is None after roundtrip"
            bias_count += 1

    # 2 layers × 2 projections (mlp_up + mlp_down) = 4 bias tensors expected.
    assert bias_count == 4, f"Expected 4 MLP bias tensors, found {bias_count}"
