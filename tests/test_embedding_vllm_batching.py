import os
import sys

import pytest
import torch

# Ensure our local packages are importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from echo_dsrn.configuration_echo import EchoConfig
from echo_embedding import EchoModelForSentenceEmbedding


@pytest.fixture
def tiny_mean_c_all_embedding():
    """A tiny mean_c_all embedding model for vLLM flattened-batch tests.

    Mirrors the real embed models: non-causal window attention.
    """
    config = EchoConfig(
        vocab_size=1000,
        hidden_size=64,
        num_layers=2,
        num_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
        window_size=32,
        pooling_mode="mean_c_all",
        attention_masking="non_causal_window",
    )
    model = EchoModelForSentenceEmbedding(config)
    model.eval()
    return model


def _single_pooled(model, token_ids):
    ids = torch.tensor([token_ids])
    with torch.no_grad():
        out = model(input_ids=ids)
    return out.last_hidden_state[0, 0]  # broadcast vector


def test_flattened_batch_matches_single_requests(tiny_mean_c_all_embedding):
    """vLLM's flattened [1, N] forward must produce per-sequence vectors that
    match running each sequence on its own."""
    model = tiny_mean_c_all_embedding
    torch.manual_seed(0)
    seqs = [
        [torch.randint(0, 1000, (4,)).tolist()],
        [torch.randint(0, 1000, (7,)).tolist()],
        [torch.randint(0, 1000, (3,)).tolist()],
    ]
    seqs = [s[0] for s in seqs]

    singles = [_single_pooled(model, s) for s in seqs]

    # Emulate the vLLM Transformers backend: concatenate all sequences into
    # one [1, N] forward with position_ids restarting at each sequence start
    # and no attention_mask.
    concat = [t for s in seqs for t in s]
    positions = [i for s in seqs for i in range(len(s))]
    input_ids = torch.tensor([concat])
    position_ids = torch.tensor([positions])
    with torch.no_grad():
        out = model(input_ids=input_ids, position_ids=position_ids)

    # Each sequence's last token carries its own pooled vector.
    offsets = []
    acc = 0
    for s in seqs:
        acc += len(s)
        offsets.append(acc - 1)

    for i, off in enumerate(offsets):
        got = out.last_hidden_state[0, off]
        assert torch.allclose(
            got, singles[i], atol=1e-5
        ), f"segment {i} diverges from its single-request embedding"


def test_flattened_single_segment_unchanged(tiny_mean_c_all_embedding):
    """A single-segment flattened forward (one sequence) must produce the same
    vector as the no-mask path used before the segment fix."""
    model = tiny_mean_c_all_embedding
    torch.manual_seed(1)
    seq = torch.randint(0, 1000, (6,)).tolist()

    ref = _single_pooled(model, seq)

    input_ids = torch.tensor([seq])
    position_ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
    with torch.no_grad():
        out = model(input_ids=input_ids, position_ids=position_ids)
    assert torch.allclose(out.last_hidden_state[0, 0], ref, atol=1e-6)


def test_real_batch_with_mask_unchanged(tiny_mean_c_all_embedding):
    """The plain transformers path (B>1 with attention_mask) must keep working
    and never trigger the flattened-segment branch."""
    model = tiny_mean_c_all_embedding
    torch.manual_seed(2)
    ids = torch.randint(0, 1000, (2, 5))
    mask = torch.ones((2, 5), dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask)
    expected_dim = model.config.hidden_size * model.config.num_heads
    assert out.last_hidden_state.shape == (2, 5, expected_dim)
    assert not torch.isnan(out.last_hidden_state).any()


def test_padded_batch_matches_singles(tiny_mean_c_all_embedding):
    """Padding must not leak into the non-causal attention: a padded batch
    produces the same per-sequence vectors as single requests."""
    model = tiny_mean_c_all_embedding
    torch.manual_seed(3)
    short = torch.randint(0, 1000, (3,)).tolist()
    long = torch.randint(0, 1000, (7,)).tolist()

    single = _single_pooled(model, short)

    padded = short + [0] * (len(long) - len(short))
    ids = torch.tensor([padded, long])
    mask = torch.tensor(
        [
            [1] * len(short) + [0] * (len(long) - len(short)),
            [1] * len(long),
        ]
    )
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask)
    got = out.last_hidden_state[0, 0]
    assert torch.allclose(
        got, single, atol=1e-5
    ), "padded-batch embedding diverges from the single-request embedding"
