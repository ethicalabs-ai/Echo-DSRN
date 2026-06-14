import os
import sys

import pytest
import torch

# Ensure our local packages are importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoConfig, AutoModel

# Import package registry and configuration
from echo_dsrn.configuration_echo import EchoConfig
from echo_embedding import EchoModelForSentenceEmbedding


@pytest.fixture
def tiny_echo_embedding_config():
    """Create a tiny configuration for fast testing."""
    return EchoConfig(
        vocab_size=1000,
        hidden_size=64,
        num_layers=2,
        num_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
        window_size=32,
    )


@pytest.fixture
def tiny_echo_embedding(tiny_echo_embedding_config):
    """Instantiate a tiny randomly initialized embedding model."""
    model = EchoModelForSentenceEmbedding(tiny_echo_embedding_config)
    model.eval()
    return model


def test_embedding_forward_pass_shapes(tiny_echo_embedding):
    """Test that the embedding model returns 3D tensors of the correct shape."""
    model = tiny_echo_embedding
    batch_size = 2
    seq_len = 8

    input_ids = torch.randint(0, 1000, (batch_size, seq_len))

    with torch.no_grad():
        outputs = model(input_ids=input_ids, return_dict=True)

    expected_dim = model.config.hidden_size * model.config.num_heads
    assert outputs.last_hidden_state.shape == (batch_size, seq_len, expected_dim)
    assert not torch.isnan(outputs.last_hidden_state).any()

    # Verify we can access past_key_values
    assert outputs.past_key_values is not None
    # For EchoCache with hybrid attention, we check the length of layer states
    assert len(outputs.past_key_values) == model.config.num_layers


def test_embedding_return_modes(tiny_echo_embedding):
    """Test return_dict=True/False support in the embedding forward pass."""
    model = tiny_echo_embedding
    batch_size = 1
    seq_len = 5
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))

    # 1. return_dict = False (returns tuple)
    with torch.no_grad():
        out_tuple = model(input_ids=input_ids, return_dict=False)

    assert isinstance(out_tuple, tuple)
    assert len(out_tuple) == 2
    expected_dim = model.config.hidden_size * model.config.num_heads
    assert out_tuple[0].shape == (batch_size, seq_len, expected_dim)

    # 2. return_dict = True (returns BaseModelOutputWithPast)
    with torch.no_grad():
        out_dict = model(input_ids=input_ids, return_dict=True)

    from transformers.modeling_outputs import BaseModelOutputWithPast

    assert isinstance(out_dict, BaseModelOutputWithPast)


def test_embedding_projection(tiny_echo_embedding_config):
    """Test embedding model with an active projection layer."""
    config = tiny_echo_embedding_config
    config.project_embeddings = True
    config.embedding_dim = 128

    model = EchoModelForSentenceEmbedding(config)
    model.eval()

    batch_size = 2
    seq_len = 6
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))

    with torch.no_grad():
        outputs = model(input_ids=input_ids, return_dict=True)

    assert outputs.last_hidden_state.shape == (batch_size, seq_len, 128)
    assert model.projection is not None
    assert isinstance(model.projection, torch.nn.Linear)


def test_pooling_mathematical_equivalence(tiny_echo_embedding):
    """Verify that mean pooling of the returned 3D tensor equals the final state 'c'."""
    model = tiny_echo_embedding
    batch_size = 1
    seq_len = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))

    with torch.no_grad():
        outputs = model(input_ids=input_ids, return_dict=True)

    # Extract last layer's state c directly
    last_layer = outputs.past_key_values[-1]
    c_state = last_layer[1]  # shape (batch, state_dim)

    # Simulate Mean Pooling over outputs
    embeddings_3d = outputs.last_hidden_state
    mean_pooled = embeddings_3d.mean(dim=1)  # Average over sequence length

    # Mean pooling should perfectly recover the broadcasted c_state
    assert torch.allclose(mean_pooled, c_state, atol=1e-6)


def test_autoclass_registration_embedding(tiny_echo_embedding_config):
    """Verify that the model registers correctly with HuggingFace's AutoModel."""
    # Registering configuration class with registry
    config = tiny_echo_embedding_config

    # Resolve from AutoConfig
    resolved_config = AutoConfig.for_model("echo")
    assert resolved_config == EchoConfig or isinstance(resolved_config, EchoConfig)

    # Resolve and instantiate from AutoModel
    model = AutoModel.from_config(config)
    assert isinstance(model, EchoModelForSentenceEmbedding)
