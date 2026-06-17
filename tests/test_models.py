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


def test_echo_return_dict_support(tiny_echo_dsrn):
    """Test that return_dict=True and return_dict=False behave correctly in EchoModel and EchoForCausalLM."""
    from transformers.modeling_outputs import (
        BaseModelOutputWithPast,
        CausalLMOutputWithPast,
    )

    causal_model = tiny_echo_dsrn
    base_model = causal_model.model
    batch_size = 2
    seq_len = 8
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))

    # --- Test 1: EchoModel with return_dict=True ---
    out_dict = base_model(input_ids=input_ids, return_dict=True)
    assert isinstance(out_dict, BaseModelOutputWithPast)
    assert out_dict.last_hidden_state.shape == (batch_size, seq_len, base_model.embed_dim)
    assert out_dict.past_key_values is not None
    assert out_dict.hidden_states is None

    # --- Test 2: EchoModel with return_dict=False ---
    out_tuple = base_model(input_ids=input_ids, return_dict=False)
    assert isinstance(out_tuple, tuple)
    assert len(out_tuple) == 2
    assert out_tuple[0].shape == (batch_size, seq_len, base_model.embed_dim)

    # --- Test 3: EchoModel with output_hidden_states=True ---
    out_hs = base_model(input_ids=input_ids, return_dict=True, output_hidden_states=True)
    assert isinstance(out_hs, BaseModelOutputWithPast)
    assert out_hs.hidden_states is not None
    assert len(out_hs.hidden_states) == 1
    assert out_hs.hidden_states[0].shape == (batch_size, seq_len, base_model.embed_dim)

    # --- Test 4: EchoForCausalLM with return_dict=True ---
    causal_out_dict = causal_model(input_ids=input_ids, return_dict=True)
    assert isinstance(causal_out_dict, CausalLMOutputWithPast)
    assert causal_out_dict.logits.shape == (batch_size, seq_len, causal_model.config.vocab_size)
    assert causal_out_dict.past_key_values is not None
    assert causal_out_dict.hidden_states is None

    # --- Test 5: EchoForCausalLM with return_dict=False ---
    causal_out_tuple = causal_model(input_ids=input_ids, return_dict=False)
    assert isinstance(causal_out_tuple, tuple)
    # Without labels, it should return (logits, past_key_values)
    assert len(causal_out_tuple) == 2
    assert causal_out_tuple[0].shape == (batch_size, seq_len, causal_model.config.vocab_size)

    # --- Test 6: EchoForCausalLM with output_hidden_states=True ---
    causal_out_hs = causal_model(input_ids=input_ids, return_dict=True, output_hidden_states=True)
    assert isinstance(causal_out_hs, CausalLMOutputWithPast)
    assert causal_out_hs.hidden_states is not None
    assert len(causal_out_hs.hidden_states) == 1
    assert causal_out_hs.hidden_states[0].shape == (batch_size, seq_len, base_model.embed_dim)


def test_trl_chunked_nll_compatibility(tiny_echo_dsrn):
    """
    Verify compatibility with SFTTrainer's chunked_nll flow.
    TRL intercepts outputs of the base model or causal LM to get last_hidden_state
    and computes chunked cross-entropy.
    """
    model = tiny_echo_dsrn
    batch_size = 2
    seq_len = 8
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))

    # 1. Base model must return an object with 'last_hidden_state' attribute
    base_outputs = model.model(input_ids=input_ids, return_dict=True)
    assert hasattr(base_outputs, "last_hidden_state")
    assert base_outputs.last_hidden_state is not None
    assert base_outputs.last_hidden_state.shape == (batch_size, seq_len, model.model.embed_dim)

    # 2. Causal LM model must expose 'hidden_states' in its return_dict output
    # (used by some trainers to extract intermediate representations)
    causal_outputs = model(input_ids=input_ids, return_dict=True, output_hidden_states=True)
    assert hasattr(causal_outputs, "hidden_states")
    assert causal_outputs.hidden_states is not None
    assert len(causal_outputs.hidden_states) == 1
    assert causal_outputs.hidden_states[0].shape == base_outputs.last_hidden_state.shape


def test_dsrn_gate_bf16_saturation_no_nan():
    """Gate sigmoid/tanh must not produce NaN gradients under bf16 saturation.

    When gate parameters drift to extreme values during training (common after
    Stage 1 MNRL), sigmoid/tanh saturate to exact 0/1 in bf16.  The backward
    must not produce 0 × inf = NaN.
    """
    if not torch.cuda.is_available():
        pytest.skip("bf16 saturation test requires CUDA")

    from echo_dsrn.modeling_echo import DSRNBlock

    config = EchoConfig(
        hidden_size=64,
        num_attention_heads=2,
        num_key_value_heads=2,
        window_size=16,
        use_hybrid_attention=True,
    )
    block = DSRNBlock(config).cuda().bfloat16().train()

    # Push gate biases to saturation regime
    with torch.no_grad():
        block.linear_gate.bias.fill_(6.0)  # sigmoid(6) ≈ 0.998
        block.gru_cell.bias_ih.fill_(6.0)  # sigmoid(6) ≈ 0.998

    B, T, D = 2, 64, config.hidden_size
    x = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    h_prev = torch.zeros(B, D, device="cuda", dtype=torch.bfloat16)
    c_prev = torch.zeros(B, D * config.num_heads, device="cuda", dtype=torch.bfloat16)

    state = (h_prev, c_prev)
    out = block(x, state)
    loss = out[0].sum()
    loss.backward()

    # Any NaN gradient in the block means the fix is broken
    for name, p in block.named_parameters():
        if p.grad is not None:
            assert not p.grad.isnan().any(), f"NaN grad in {name}"

    # Input gradient must also be clean
    assert not x.grad.isnan().any(), "NaN in input gradient"

    # Forward must be finite
    assert not out[0].isnan().any(), "NaN in forward output"
