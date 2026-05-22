"""
tests/test_utils.py
──────────────────────────────────────────────────────────────────────────────
Unit tests for echo_dsrn.utils — training visualisation utilities.

These tests do not require a GPU or a network connection.  They use a minimal
stub tokenizer and in-memory dataset-style dicts so the whole suite runs in
milliseconds.
"""

from echo_dsrn.utils import visualize_masked_samples

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubTokenizer:
    """Minimal tokenizer stub: decode returns token IDs joined by spaces."""

    def decode(self, token_ids, skip_special_tokens=False):
        return " ".join(str(t) for t in token_ids)


def _make_dataset(*rows):
    """Return a list-of-dicts that mimics a HF Dataset's __getitem__ interface."""

    class _DS:
        def __init__(self, data):
            self._data = data

        def __len__(self):
            return len(self._data)

        def __getitem__(self, idx):
            return self._data[idx]

    return _DS(list(rows))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_empty_labels_row_skipped(capsys):
    """A row with an empty labels list should be skipped without error."""
    ds = _make_dataset({"input_ids": [1, 2, 3], "labels": []})
    visualize_masked_samples(ds, _StubTokenizer(), num_samples=1)
    out = capsys.readouterr().out
    # The header is printed, but no Sample line should appear
    assert "PREVIEWING" in out
    assert "Sample 0" not in out


def test_all_masked(capsys):
    """All tokens masked (-100): single span printed in masked style."""
    ids = [10, 20, 30]
    labels = [-100, -100, -100]
    ds = _make_dataset({"input_ids": ids, "labels": labels})
    visualize_masked_samples(ds, _StubTokenizer(), num_samples=1)
    out = capsys.readouterr().out
    assert "Sample 0" in out
    # All tokens should appear in the output (decoded as "10 20 30")
    assert "10" in out and "20" in out and "30" in out


def test_all_loss(capsys):
    """No tokens masked: single span printed in loss style."""
    ids = [5, 6, 7]
    labels = [5, 6, 7]
    ds = _make_dataset({"input_ids": ids, "labels": labels})
    visualize_masked_samples(ds, _StubTokenizer(), num_samples=1)
    out = capsys.readouterr().out
    assert "Sample 0" in out
    assert "5" in out


def test_span_transition_masked_then_loss(capsys):
    """Span switch from masked → loss triggers a flush + new span."""
    ids = [1, 2, 3, 4]
    labels = [-100, -100, 3, 4]  # first two masked, last two loss
    ds = _make_dataset({"input_ids": ids, "labels": labels})
    visualize_masked_samples(ds, _StubTokenizer(), num_samples=1)
    out = capsys.readouterr().out
    assert "Sample 0" in out
    # Both halves decoded and present
    assert "1" in out
    assert "3" in out


def test_span_transition_loss_then_masked(capsys):
    """Span switch from loss → masked triggers a flush."""
    ids = [10, 20, 30, 40]
    labels = [10, 20, -100, -100]
    ds = _make_dataset({"input_ids": ids, "labels": labels})
    visualize_masked_samples(ds, _StubTokenizer(), num_samples=1)
    out = capsys.readouterr().out
    assert "Sample 0" in out


def test_num_samples_cap(capsys):
    """num_samples is respected — only the first N rows are printed."""
    rows = [{"input_ids": [i], "labels": [i]} for i in range(5)]
    ds = _make_dataset(*rows)
    visualize_masked_samples(ds, _StubTokenizer(), num_samples=2)
    out = capsys.readouterr().out
    assert "Sample 0" in out
    assert "Sample 1" in out
    assert "Sample 2" not in out


def test_multiple_span_transitions(capsys):
    """Alternating mask/loss spans: M M L L M — three distinct spans flushed."""
    ids = [1, 2, 3, 4, 5]
    labels = [-100, -100, 3, 4, -100]
    ds = _make_dataset({"input_ids": ids, "labels": labels})
    visualize_masked_samples(ds, _StubTokenizer(), num_samples=1)
    out = capsys.readouterr().out
    assert "Sample 0" in out
    # All token ids appear in decoded output
    for tok in ["1", "2", "3", "4", "5"]:
        assert tok in out


def test_newline_in_token_replaced(capsys):
    """Newlines in decoded text are replaced so ANSI codes don't break."""

    class _NewlineTokenizer:
        def decode(self, token_ids, skip_special_tokens=False):
            return "hello\nworld"

    ids = [1, 2]
    labels = [1, 2]
    ds = _make_dataset({"input_ids": ids, "labels": labels})
    # Should not raise; newline is handled inside the function
    visualize_masked_samples(ds, _NewlineTokenizer(), num_samples=1)
