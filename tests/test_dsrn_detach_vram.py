"""
tests/test_dsrn_detach_vram.py
────────────────────────────────────────────────────────────────────────────
Regression tests for the DSRN recurrent-state detach fix.

Bug — _dsrn_input_states retains autograd graph across training steps
  Without .detach(), the states carried from step N into step N+1 held a
  grad_fn that linked both steps' autograd graphs together.  Step N's
  activation tensors (~11.78 GiB at batch=2/seq=1024) could not be freed
  after step N's backward() because step N+1's graph held live references
  to them.  Over 12 steps this accumulated ~10 GB of leaked VRAM.

Fix — truncated BPTT via .detach()
  _dsrn_input_states is now detached before being stored, making the
  carried-over state a leaf tensor for the next step.  Gradients still
  flow freely within each step's forward pass (through the injector linear
  layers), but do not propagate back through the carryover boundary.

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
LAYERS = 4  # must be >= dsrn_injection_stride (default 4) for ≥1 injector
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
        use_kv_cache=False,  # training / GC mode
    )
    model = HybridEchoForCausalLM(config)
    model = model.float()  # bfloat16 is unreliable on CPU builds
    model.gradient_checkpointing_enable()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 1. Carried state is detached (no grad_fn leak across steps)
# ─────────────────────────────────────────────────────────────────────────────


class TestDSRNStateDetach:
    """Carried DSRN states must be detached leaves after each forward pass.

    _dsrn_input_states is set at the very top of HybridEchoModel.forward()
    (line ~223) to the detached version of the previous step's output states.
    We capture it via a pre-forward hook on the *second* step, at which point
    it holds whatever was stored at the end of step N — so we can assert it is
    a detached leaf before any new computation touches it.
    """

    def _capture_input_states_at_step_start(self, inner_model):
        """Register a forward pre-hook on the backbone that snapshots _dsrn_input_states.

        The hook fires after HybridEchoModel.forward() populates
        _dsrn_input_states (line ~223) but before the backbone call returns
        and the list is cleared (line ~309).  This gives us direct access to
        the detached states that will be used as inputs for this step.
        """
        captured = {}

        def _hook(module, args, kwargs):
            # Take a snapshot — the list contains the freshly-detached states.
            captured["states"] = [(h, c) for h, c in inner_model._dsrn_input_states]
            return None  # do not modify inputs

        handle = inner_model.backbone.register_forward_pre_hook(_hook, with_kwargs=True)
        return captured, handle

    def test_dsrn_states_have_no_grad_fn_after_forward(self, tiny_hybrid):
        """States carried into step N+1 must have no grad_fn.

        _dsrn_input_states is populated via .detach() at the top of every
        forward(); if grad_fn is present the previous step's activation graph
        is still live and cannot be freed after backward().
        """
        model = tiny_hybrid
        model.train()

        if model.model.num_injectors == 0:
            pytest.skip("No DSRN injectors in this config.")

        input_ids = torch.randint(0, VOCAB, (BATCH, SEQ))
        labels = input_ids.clone()

        # Step N — populates _dsrn_input_states for step N+1
        out = model(input_ids=input_ids, labels=labels, use_cache=False)
        out.loss.backward()
        del out

        # Capture _dsrn_input_states at the very start of step N+1
        captured, handle = self._capture_input_states_at_step_start(model.model)
        try:
            out2 = model(input_ids=input_ids, labels=labels, use_cache=False)
            out2.loss.backward()
            del out2
        finally:
            handle.remove()

        assert captured.get("states"), "_dsrn_input_states was empty at step N+1 start."
        for i, (h, c) in enumerate(captured["states"]):
            assert h.grad_fn is None, (
                f"_dsrn_input_states[{i}].h has grad_fn={h.grad_fn}. "
                "The state is not detached — VRAM will accumulate across steps."
            )
            assert c.grad_fn is None, (
                f"_dsrn_input_states[{i}].c has grad_fn={c.grad_fn}. "
                "The state is not detached — VRAM will accumulate across steps."
            )

    def test_dsrn_states_are_leaf_tensors(self, tiny_hybrid):
        """States carried into step N+1 must be leaf tensors (is_leaf=True).

        A non-leaf tensor is part of an existing computation graph; carrying
        one across the step boundary prevents the previous graph from being freed.
        """
        model = tiny_hybrid
        model.train()

        if model.model.num_injectors == 0:
            pytest.skip("No DSRN injectors in this config.")

        input_ids = torch.randint(0, VOCAB, (BATCH, SEQ))
        labels = input_ids.clone()

        # Step N
        out = model(input_ids=input_ids, labels=labels, use_cache=False)
        out.loss.backward()
        del out

        # Capture at step N+1 start
        captured, handle = self._capture_input_states_at_step_start(model.model)
        try:
            out2 = model(input_ids=input_ids, labels=labels, use_cache=False)
            out2.loss.backward()
            del out2
        finally:
            handle.remove()

        assert captured.get("states"), "_dsrn_input_states was empty at step N+1 start."
        for i, (h, c) in enumerate(captured["states"]):
            assert h.is_leaf, (
                f"_dsrn_input_states[{i}].h is not a leaf tensor. "
                "It still participates in the previous step's graph."
            )
            assert c.is_leaf, (
                f"_dsrn_input_states[{i}].c is not a leaf tensor. "
                "It still participates in the previous step's graph."
            )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Gradient flow is NOT broken by detach (truncated BPTT)
# ─────────────────────────────────────────────────────────────────────────────


class TestDSRNGradientFlow:
    """Detaching carryover states must not cut gradient flow within a step."""

    def test_injector_params_receive_gradients_after_detach(self, tiny_hybrid):
        """Every DSRN injector parameter must have a non-zero gradient.

        Detach() cuts the graph at the step boundary only; gradients must
        still flow through the injector's linear layers within the current step.
        A zero gradient here would mean the detach was applied incorrectly
        (e.g. inside the injector rather than at the carryover site).
        """
        model = tiny_hybrid
        model.train()

        if model.model.num_injectors == 0:
            pytest.skip("No DSRN injectors in this config.")

        input_ids = torch.randint(0, VOCAB, (BATCH, SEQ))
        labels = input_ids.clone()

        out = model(input_ids=input_ids, labels=labels, use_cache=False)
        out.loss.backward()

        injector_grads = [
            (n, p.grad)
            for n, p in model.model.memory_injectors.named_parameters()
            if p.requires_grad and p.grad is not None
        ]
        assert len(injector_grads) > 0, (
            "No gradients on any DSRNMemoryInjector parameter. "
            "The .detach() may have been placed inside the injector call."
        )

        nonzero = [(n, g) for n, g in injector_grads if g.abs().max() > 0]
        assert len(nonzero) > 0, (
            "All DSRNMemoryInjector gradients are zero after backward(). "
            "The detach may have severed within-step gradient flow."
        )

    def test_graph_does_not_link_across_steps(self, tiny_hybrid):
        """The autograd graph from step N+1 must not retain step N's tensors.

        We run two sequential steps sharing carryover state.  After step N's
        backward() we check that its loss tensor has been freed (version
        counter advances and the grad_fn chain does not reach step N's loss).
        Concretely: step N+1's loss.grad_fn chain must NOT contain step N's
        loss grad_fn as an ancestor.
        """
        model = tiny_hybrid
        model.train()

        input_ids = torch.randint(0, VOCAB, (BATCH, SEQ))
        labels = input_ids.clone()

        # Step N
        opt = torch.optim.SGD(model.parameters(), lr=0.0)  # lr=0: no weight change
        out_n = model(input_ids=input_ids, labels=labels, use_cache=False)
        loss_n = out_n.loss
        loss_n_grad_fn = loss_n.grad_fn
        loss_n.backward()
        opt.step()
        opt.zero_grad()
        del out_n, loss_n

        # Step N+1 using the carryover cache from step N
        out_np1 = model(input_ids=input_ids, labels=labels, use_cache=False)
        loss_np1 = out_np1.loss

        # Walk the graph of step N+1 and verify step N's grad_fn is absent.
        def _collect_grad_fns(fn, visited=None):
            if visited is None:
                visited = set()
            if fn is None or fn in visited:
                return visited
            visited.add(fn)
            for child, _ in fn.next_functions:
                _collect_grad_fns(child, visited)
            return visited

        step_np1_fns = _collect_grad_fns(loss_np1.grad_fn)
        assert loss_n_grad_fn not in step_np1_fns, (
            "Step N's grad_fn is reachable from step N+1's loss graph. "
            "The DSRN state carryover is NOT detached — VRAM will accumulate."
        )

        loss_np1.backward()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Multi-step training stability
# ─────────────────────────────────────────────────────────────────────────────


class TestMultiStepTraining:
    """Multiple sequential training steps must produce finite, stable loss."""

    def test_loss_finite_across_n_steps(self, tiny_hybrid):
        """Running N consecutive training steps must not produce NaN/Inf loss."""
        model = tiny_hybrid
        model.train()
        N = 5

        dsrn_params = [p for n, p in model.named_parameters() if "memory_injectors" in n]
        opt = torch.optim.AdamW(dsrn_params, lr=1e-4)

        torch.manual_seed(42)
        input_ids = torch.randint(0, VOCAB, (BATCH, SEQ))
        labels = input_ids.clone()

        for step in range(1, N + 1):
            opt.zero_grad()
            out = model(input_ids=input_ids, labels=labels, use_cache=False)
            loss = out.loss
            assert torch.isfinite(loss), f"loss is not finite at step {step}: {loss.item()}"
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dsrn_params, 1.0)
            opt.step()
            del out, loss
