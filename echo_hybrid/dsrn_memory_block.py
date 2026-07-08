"""
echo_hybrid/dsrn_memory_block.py
────────────────────────────────────────────────────────────────────────────
Standalone DSRNMemoryInjector — purely additive residual memory block.

This module is the heart of the Echo-Hybrid architecture.  It is imported by
modeling_hybrid.py and is kept separate to maximise reusability.

Architecture
────────────
The injector receives the transformer's residual stream (B, T, D_transformer)
and performs three operations:

  1. Fast-state update  (h_t) — lightweight linear SSM scan, O(T log T)
  2. Slow-state update  (c_t) — surprise-gated write, O(T log T)
  3. Memory read         (x_out = x + linear_read(c_all)) — purely additive

MLP and self-attention are intentionally absent — those are handled by the
Qwen2 backbone sitting around each injector.

CRITICAL NOTES (from AGENTS.md and design spec)
────────────────────────────────────────────────
• gru_cell.weight_hh / bias_hh are intentionally UNUSED.  The parallel scan
  eliminates the h_{t-1} recurrent dependency.  Do not "fix" the unused
  grad warning.
• linear_read.weight MUST be zero-initialized.  At Step 0 the injector has
  zero effect on the residual stream, so Qwen2's pre-trained weights are
  preserved exactly.
• linear_gate.bias MUST be initialized to gate_bias_init (default 1.0) so
  that memory writes can begin immediately.
• surprise_lambda starts at zero (softplus(0) ≈ 0.693, still small) and is
  learned during training.
• linear_pred.weight uses orthogonal init (gain=0.1) for stable surprise
  signal at initialization.

Imports
───────
We import dsrn_parallel_scan, rms_norm_fn, and HymbaRMSNorm directly from
echo_hf.modeling_echo — we do NOT copy or edit that file.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Canonical imports from the parent Echo-DSRN package.
from echo_dsrn.modeling_echo import HymbaRMSNorm, dsrn_parallel_scan, rms_norm_fn

from .configuration_hybrid import HybridEchoConfig


class DSRNMemoryInjector(nn.Module):
    """
    DSRN memory injector for the Echo-Hybrid architecture.

    Receives the transformer residual stream x (B, T, D_transformer),
    updates the DSRN recurrent states (h_t, c_t), and injects the
    compressed memory back into x as a purely additive residual.

    Parameters
    ──────────
    config : HybridEchoConfig
        D_transformer = config.hidden_size  (896 for Qwen2-0.5B)
        D_state       = config.dsrn_state_dim  (512 by default)
    """

    def __init__(self, config: HybridEchoConfig):
        super().__init__()
        D = config.hidden_size  # transformer residual dim
        D_s = config.dsrn_state_dim  # slow-state dim

        self.use_triton = config.dsrn_use_triton

        # ── Normalisation ─────────────────────────────────────────────────
        # RMSNorm matches the Hybrid kernel path in modeling_echo.py.
        self.norm_fast = HymbaRMSNorm(D)

        # ── Fast-State GRU ────────────────────────────────────────────────
        # weight_ih shape: (3*D, D) — encodes z (gate), unused r_internal, r (input)
        # weight_hh / bias_hh intentionally unused (parallel scan design).
        self.gru_cell = nn.GRUCell(D, D)

        # ── Slow-State (Surprise-Gated) ───────────────────────────────────
        # Prediction head: h_{t-1} → x̂_t  (for computing surprise error)
        self.linear_pred = nn.Linear(D, D, bias=False)

        # Gate: maps fast-state h_t → gate logits over D_s dims
        self.linear_gate = nn.Linear(D, D_s)

        # Memory write: h_t → candidate memory vector in D_s space
        self.linear_memory = nn.Linear(D, D_s)

        # Surprise scalar (learnable, one per slow-state dim)
        self.surprise_lambda = nn.Parameter(torch.zeros(D_s))

        # Memory read: c_t (D_s) → residual contribution (D_transformer)
        # CRITICAL: initialized to ZEROS — injector is invisible at Step 0
        self.linear_read = nn.Linear(D_s, D, bias=False)

        # Apply critical initializations immediately after construction
        self._init_weights()

    # ─────────────────────────────────────────────────────────────────────
    # Initialization
    # ─────────────────────────────────────────────────────────────────────

    def _init_weights(self):
        """Apply the critical zero / orthogonal / bias inits documented in the spec."""
        # linear_read: zero-init so injector is a no-op at Step 0
        nn.init.zeros_(self.linear_read.weight)

        # linear_pred: orthogonal (gain=0.1) for stable surprise signal
        # Must be done in fp32 (orthogonal_ requires float)
        w = self.linear_pred.weight
        if w.dtype in (torch.bfloat16, torch.float16):
            tmp = torch.empty_like(w, dtype=torch.float32, device=w.device)
            nn.init.orthogonal_(tmp, gain=0.1)
            with torch.no_grad():
                w.copy_(tmp.to(device=w.device, dtype=w.dtype))
        else:
            nn.init.orthogonal_(self.linear_pred.weight, gain=0.1)

        # linear_gate: bias = gate_bias_init so gates start open
        # NOTE: gate_bias_init is not passed to __init__ here because the
        # injector is constructed before convert_from_qwen2 applies it.
        # convert_from_qwen2.py re-applies nn.init.constant_ afterward.
        # We default to 1.0 here as a safe fallback.
        nn.init.constant_(self.linear_gate.bias, 1.0)

        # surprise_lambda: start at zero (softplus(0) ≈ 0.693, still small)
        nn.init.zeros_(self.surprise_lambda)

    # ─────────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,  # (B, T, D_transformer)
        h_prev: torch.Tensor,  # (B, D_transformer)  fast state
        c_prev: torch.Tensor,  # (B, D_state)         slow state
        eos_mask: Optional[torch.Tensor] = None,  # (B, T) bool/int
        return_gate_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        ───────
        x_out  : (B, T, D_transformer)  — updated residual stream
        h_new  : (B, D_transformer)     — fast state for next chunk
        c_new  : (B, D_state)           — slow state for next chunk
        """
        B, T, D = x.shape

        # ── 1. Normalise ──────────────────────────────────────────────────
        x_norm = rms_norm_fn(x, self.norm_fast.weight)

        # ── 2. Fast-State (parallel SSM scan) ────────────────────────────
        # GRU projection: (B, T, 3*D)  — only weight_ih / bias_ih used
        gru_proj = F.linear(x_norm, self.gru_cell.weight_ih, self.gru_cell.bias_ih)

        # Slice layout: [z | r_internal | r]   (each of size D)
        # z: forget gate, r: input candidate
        z_all = torch.sigmoid(gru_proj[:, :, :D])  # (B, T, D)
        r_all = torch.tanh(gru_proj[:, :, 2 * D :])  # (B, T, D)

        # EOS reset: if a sequence ends at t, reset z so h is wiped
        if eos_mask is not None:
            reset_mask = torch.roll(eos_mask, shifts=1, dims=1)
            reset_mask[:, 0] = 0
            z_all = torch.where(
                reset_mask.unsqueeze(-1) > 0,
                torch.ones_like(z_all),
                z_all,
            )

        # Parallel scan: h_t = (1 - z_t)*h_{t-1} + z_t*r_t
        h_all = dsrn_parallel_scan(z_all, r_all, h_prev, use_triton=self.use_triton)  # (B, T, D)
        h_new = h_all[:, -1]  # (B, D)

        # ── 3. Surprise-Gated Slow-State ──────────────────────────────────
        # Causal shift: predict x[t] from h[t-1]
        h_shifted = torch.cat([h_prev.unsqueeze(1), h_all[:, :-1]], dim=1)  # (B, T, D)
        x_pred = self.linear_pred(h_shifted)  # (B, T, D)

        # Prediction error → surprise signal
        diff = x - x_pred
        error = torch.clamp(diff * diff, max=10.0).mean(dim=-1, keepdim=True)  # (B, T, 1)
        surprise_signal = error * F.softplus(self.surprise_lambda)  # (B, T, D_s)

        # Gated memory update
        gate_logits = self.linear_gate(h_all) + surprise_signal  # (B, T, D_s)
        g_all = torch.sigmoid(gate_logits)  # (B, T, D_s)
        m_all = torch.tanh(self.linear_memory(h_all))  # (B, T, D_s)

        # EOS reset: wipe gate so no write happens after sequence end
        if eos_mask is not None:
            reset_mask = torch.roll(eos_mask, shifts=1, dims=1)
            reset_mask[:, 0] = 0
            g_all = torch.where(
                reset_mask.unsqueeze(-1) > 0,
                torch.zeros_like(g_all),
                g_all,
            )

        # Parallel scan: c_t = (1 - g_t)*c_{t-1} + g_t*m_t
        c_all = dsrn_parallel_scan(g_all, m_all, c_prev, use_triton=self.use_triton)  # (B, T, D_s)
        c_new = c_all[:, -1]  # (B, D_s)

        # Inter-chunk EOS reset: zero out carry-over state if last token is EOS
        if eos_mask is not None:
            last_is_eos = eos_mask[:, -1].float()  # (B,)
            keep = (1.0 - last_is_eos).unsqueeze(-1)  # (B, 1)
            h_new = h_new * keep
            c_new = c_new * keep

        # ── 4. Memory Read (purely additive residual) ─────────────────────
        # linear_read is zero-initialized → no effect at Step 0
        x_out = x + self.linear_read(c_all)  # (B, T, D)

        if return_gate_logits:
            return x_out, h_new, c_new, gate_logits
        return x_out, h_new, c_new

    # ─────────────────────────────────────────────────────────────────────
    # Utility: c_t state norm (for Phase-1 warm-up diagnostics)
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def state_norm(c: torch.Tensor) -> float:
        """Return the mean L2 norm of c_t across the batch — use for warm-up checks."""
        return c.float().norm(dim=-1).mean().item()
