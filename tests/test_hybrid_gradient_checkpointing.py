"""
tests/test_hybrid_gradient_checkpointing.py
────────────────────────────────────────────────────────────────────────────
Regression tests for two VRAM bugs in HybridEchoForCausalLM:

  Bug 1 — logits.float() global cast
    A full float32 logit tensor was materialised and retained by autograd
    (+2.5 GiB at batch=2, seq=512 with vocab=151 936).  The fix keeps
    logits in bfloat16 and casts only inside the cross_entropy call.

  Bug 2 — DSRN hooks bypass gradient checkpoint boundaries
    register_forward_hook fires *after* the checkpoint wrapper scope, so
    the injector graph was never discarded (~10.4 GiB at batch=4, seq=512).
    The fix wraps each injector call in torch.utils.checkpoint when
    self.gradient_checkpointing and self.training are both True.

Tests run on CPU with a tiny randomly-initialised model — no GPU required.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from echo_hybrid.configuration_hybrid import HybridEchoConfig
from echo_hybrid.modeling_hybrid import HybridEchoForCausalLM

# ─────────────────────────────────────────────────────────────────────────────
# Shared fixture
# ─────────────────────────────────────────────────────────────────────────────

VOCAB = 256
HIDDEN = 64
LAYERS = 4  # must be >= dsrn_injection_stride (default 4) for at least 1 injector
SEQ = 16
BATCH = 2


@pytest.fixture
def tiny_hybrid() -> HybridEchoForCausalLM:
    """Tiny, randomly initialised Echo-Hybrid model — fast CPU testing."""
    config = HybridEchoConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=HIDDEN * 2,
        num_hidden_layers=LAYERS,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    model = HybridEchoForCausalLM(config)
    # Use float32 for CPU tests (bfloat16 not well supported on all CPU builds).
    model = model.float()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Bug 1 — logits dtype
# ─────────────────────────────────────────────────────────────────────────────


class TestLogitsDtype:
    """logits returned by forward() must stay in the model dtype, not float32."""

    def test_logits_dtype_matches_model_weights(self, tiny_hybrid):
        """Logits must NOT be upcast to float32 unconditionally.

        Before the fix, `logits = logits.float()` always ran, so logits.dtype
        was always torch.float32 regardless of the model dtype.  After the fix,
        logits stay in the model's compute dtype (float32 here because we
        called .float() on the model in the fixture — the key check is that no
        *extra* upcast happens for bfloat16 models).
        """
        model = tiny_hybrid
        model.eval()
        input_ids = torch.randint(0, VOCAB, (BATCH, SEQ))

        with torch.no_grad():
            out = model(input_ids=input_ids)

        # The model weights are float32 (set in fixture).
        # logits must be the same dtype — no silent extra cast to a *different* dtype.
        weight_dtype = model.lm_head.weight.dtype
        assert out.logits.dtype == weight_dtype, (
            f"logits.dtype={out.logits.dtype} does not match lm_head dtype={weight_dtype}. "
            "A global .float() cast is still being applied."
        )

    def test_logits_dtype_bfloat16(self, tiny_hybrid):
        """In bfloat16 mode, logits must remain bfloat16 (not silently promoted to float32)."""
        # Skip if bfloat16 is not supported on this CPU build.
        pytest.importorskip("torch")
        if not torch.cuda.is_available() and not hasattr(torch, "bfloat16"):
            pytest.skip("bfloat16 not available on this platform")

        model = tiny_hybrid.bfloat16()
        model.eval()
        input_ids = torch.randint(0, VOCAB, (1, SEQ))

        with torch.no_grad():
            out = model(input_ids=input_ids)

        assert out.logits.dtype == torch.bfloat16, (
            f"Expected bfloat16 logits, got {out.logits.dtype}. "
            "The global .float() cast has returned."
        )

    def test_loss_computed_correctly_with_labels(self, tiny_hybrid):
        """Training forward with labels must return a finite scalar loss."""
        model = tiny_hybrid
        model.train()
        input_ids = torch.randint(0, VOCAB, (BATCH, SEQ))
        labels = input_ids.clone()

        out = model(input_ids=input_ids, labels=labels)

        assert out.loss is not None, "loss is None when labels are provided"
        assert out.loss.shape == (), f"Expected scalar loss, got shape {out.loss.shape}"
        assert torch.isfinite(out.loss), f"loss is not finite: {out.loss.item()}"


# ─────────────────────────────────────────────────────────────────────────────
# Bug 2 — gradient checkpointing + DSRN hooks
# ─────────────────────────────────────────────────────────────────────────────


class TestGradientCheckpointing:
    """DSRN injectors must participate in gradient checkpointing during training."""

    @staticmethod
    def _enable_gc(model: HybridEchoForCausalLM) -> HybridEchoForCausalLM:
        model.gradient_checkpointing_enable()
        return model

    def test_gc_flag_propagates_to_inner_model(self, tiny_hybrid):
        """gradient_checkpointing_enable() must set the flag on HybridEchoModel."""
        model = self._enable_gc(tiny_hybrid)
        assert model.model.gradient_checkpointing, (
            "HybridEchoModel.gradient_checkpointing is False after gradient_checkpointing_enable(). "
            "_set_gradient_checkpointing() is not propagating the flag correctly."
        )

    def test_training_forward_backward_with_gc(self, tiny_hybrid):
        """A full forward+backward with gradient checkpointing must not raise."""
        model = self._enable_gc(tiny_hybrid)
        model.train()
        input_ids = torch.randint(0, VOCAB, (BATCH, SEQ))
        labels = input_ids.clone()

        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()  # Must not raise

        # Every LoRA / model parameter that requires grad must have a gradient.
        params_with_grad = [
            (n, p) for n, p in model.named_parameters() if p.requires_grad and p.grad is not None
        ]
        assert len(params_with_grad) > 0, (
            "No gradients computed after backward() with gradient checkpointing enabled. "
            "The DSRN hook is likely still exiting early."
        )

    def test_dsrn_injectors_run_during_gc_forward(self, tiny_hybrid):
        """DSRN injectors must fire during a GC-enabled training forward pass.

        With gradient checkpointing on, use_cache is forced to False so DSRN
        states are not surfaced through past_key_values.  We verify the injectors
        ran by wrapping each one with a mock that records calls while still
        executing the real forward.
        """
        from unittest.mock import patch

        model = self._enable_gc(tiny_hybrid)
        model.train()

        num_injectors = model.model.num_injectors
        if num_injectors == 0:
            pytest.skip("No DSRN injectors in this config.")

        input_ids = torch.randint(0, VOCAB, (BATCH, SEQ))
        labels = input_ids.clone()

        # Wrap each injector's __call__ with wraps= so real forward still runs
        # but calls are counted.
        with patch.object(
            model.model.memory_injectors[0],
            "forward",
            wraps=model.model.memory_injectors[0].forward,
        ) as mock_injector:
            out = model(input_ids=input_ids, labels=labels)
            out.loss.backward()

        assert mock_injector.call_count >= 1, (
            "DSRNMemoryInjector[0].forward was never called during the GC training forward. "
            "The hook may be exiting early under gradient checkpointing."
        )

    def test_gc_does_not_break_eval_mode(self, tiny_hybrid):
        """Enabling gradient checkpointing must not affect eval/inference behavior."""
        model = self._enable_gc(tiny_hybrid)
        model.eval()
        input_ids = torch.randint(0, VOCAB, (BATCH, SEQ))

        with torch.no_grad():
            out = model(input_ids=input_ids)

        assert out.logits.shape == (BATCH, SEQ, VOCAB)
        assert not torch.isnan(out.logits).any(), "NaN in logits during eval after GC enable"

    def test_gradients_flow_through_dsrn_injectors(self, tiny_hybrid):
        """Gradients must flow through memory_injectors when GC is enabled.

        This is the critical correctness check: if the hook bypassed the
        checkpoint boundary, injector parameters would receive zero gradient
        (their computation graph is detached).  We verify that at least one
        injector parameter has a non-zero gradient after backward.
        """
        model = self._enable_gc(tiny_hybrid)
        model.train()

        if model.model.num_injectors == 0:
            pytest.skip("No DSRN injectors in this config — skipping gradient flow check.")

        input_ids = torch.randint(0, VOCAB, (BATCH, SEQ))
        labels = input_ids.clone()

        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()

        injector_grads = [
            (n, p.grad)
            for n, p in model.model.memory_injectors.named_parameters()
            if p.requires_grad and p.grad is not None
        ]
        assert len(injector_grads) > 0, (
            "No gradients on any DSRNMemoryInjector parameter after backward(). "
            "The injector computation graph is likely detached (hook bypass bug)."
        )

        # At least one injector gradient must be non-zero.
        nonzero = [(n, g) for n, g in injector_grads if g.abs().max() > 0]
        assert len(nonzero) > 0, (
            "All DSRNMemoryInjector gradients are zero after backward() with GC enabled. "
            "The hook-based injection is likely detached from the autograd graph."
        )
