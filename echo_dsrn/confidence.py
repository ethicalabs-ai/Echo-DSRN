"""
echo_dsrn/confidence.py
────────────────────────────────────────────────────────────────────────────
Echo-DSRN Confidence Utilities for Speculative Decoding.

Pure mathematical functions — no learned parameters.  The confidence signal
comes from the surprise gate λ_t = σ(z_t), optionally modulated by the
model's own surprise_temperature_alpha.

Core mapping
────────────
  1. Surprise gate:       λ_t  = σ(z_t)              ∈ (0, 1)
  2. Token confidence:     c_t  = 1 − λ_t            ∈ (0, 1)
  3. Log-space survival:   log S_k = Σ log(1 − λ_i)
  4. Dynamic cutoff:       ℓ = max{k : S_k ≥ τ_load}

Integration
───────────
  - Base DSRN:    gate_logits in all_gate_logits per layer
  - Hybrid:       gate_logits in injector_gate_logits per injector

Aggregation across layers/injectors is element-wise mean.
"""

from typing import List, Optional, Tuple

import torch

# ─────────────────────────────────────────────────────────────────────────────
# Core: gate logits → confidence
# ─────────────────────────────────────────────────────────────────────────────


def aggregate_gate_logits(
    gate_logits: List[torch.Tensor],
    aggregation: str = "mean",
) -> torch.Tensor:
    """Aggregate gate_logits from multiple layers into (B, T, D)."""
    if not gate_logits:
        raise ValueError("gate_logits list is empty")
    stacked = torch.stack(gate_logits, dim=0)  # (L, B, T, D)
    if aggregation == "mean":
        return stacked.mean(dim=0)
    elif aggregation == "last":
        return stacked[-1]
    elif aggregation == "first":
        return stacked[0]
    raise ValueError(f"Unknown aggregation: {aggregation}")


def raw_lambda(gate_logits: torch.Tensor) -> torch.Tensor:
    """λ_t = σ(z_t) ∈ (0, 1)."""
    return torch.sigmoid(gate_logits)


def token_confidence(gate_logits: torch.Tensor) -> torch.Tensor:
    """c_t = 1 − λ_t = 1 − σ(z_t)."""
    return 1.0 - raw_lambda(gate_logits)


# ─────────────────────────────────────────────────────────────────────────────
# Log-space prefix survival
# ─────────────────────────────────────────────────────────────────────────────


def log_prefix_survival(
    gate_logits: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    log S_k = Σ_{i=1}^k log(1 − λ_i),  S_k = exp(log S_k).

    Returns (log_S, S), both (B, T, D).
    """
    lam = raw_lambda(gate_logits).clamp(max=1.0 - 1e-6)
    log_1m = torch.log(1.0 - lam)
    log_S = torch.cumsum(log_1m, dim=1)
    return log_S, torch.exp(log_S)


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic cutoff
# ─────────────────────────────────────────────────────────────────────────────


def dynamic_cutoff(
    gate_logits: torch.Tensor,
    tau_load: float = 0.5,
) -> torch.Tensor:
    """
    ℓ = max{k : S_k ≥ τ_load} for each batch element.

    Returns (B,) LongTensor of cutoff lengths.
    """
    _, S = log_prefix_survival(gate_logits)
    S_mean = S.mean(dim=-1)  # (B, T)

    above = S_mean >= tau_load
    below = ~above
    T_val = above.shape[1]
    first_below = below.float().argmax(dim=1)
    all_above = ~below.any(dim=1)
    first_below = torch.where(
        all_above,
        torch.tensor(T_val, device=gate_logits.device, dtype=first_below.dtype),
        first_below,
    )
    return first_below


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: full pipeline
# ─────────────────────────────────────────────────────────────────────────────


def compute_confidence(
    gate_logits_per_layer: List[torch.Tensor],
    tau_load: float = 0.5,
    return_cutoff: bool = True,
    aggregation: str = "mean",
) -> dict:
    """Aggregate → λ → survival → cutoff. All stateless."""
    aggregated = aggregate_gate_logits(gate_logits_per_layer, aggregation)
    lam = raw_lambda(aggregated)
    conf = token_confidence(aggregated)
    log_S, S = log_prefix_survival(aggregated)

    result = {
        "aggregated_gate_logits": aggregated,
        "raw_lambda": lam,
        "token_confidence": conf,
        "log_prefix_survival": log_S,
        "prefix_survival": S,
    }
    if return_cutoff:
        result["cutoff_lens"] = dynamic_cutoff(aggregated, tau_load=tau_load)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Utility: extract gate_logits from model outputs
# ─────────────────────────────────────────────────────────────────────────────


def extract_gate_logits(model_output) -> Optional[List[torch.Tensor]]:
    """Extract gate_logits from model forward pass output."""
    if hasattr(model_output, "all_gate_logits"):
        return model_output.all_gate_logits
    if isinstance(model_output, tuple) and len(model_output) > 2:
        candidate = model_output[2]
        if isinstance(candidate, list) and candidate and isinstance(candidate[0], torch.Tensor):
            return candidate
    return None
