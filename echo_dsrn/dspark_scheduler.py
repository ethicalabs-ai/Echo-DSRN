"""
echo_dsrn/dspark_scheduler.py
────────────────────────────────────────────────────────────────────────────
DSpark-Echo Speculative Decoding Scheduler.

Uses Echo-DSRN's surprise-temperature modulation (surprise_temperature_alpha)
as the confidence mechanism.  The draft model runs with α > 0, so its output
distribution is naturally flattened where the surprise gate is active.
Token confidence comes from the max softmax probability of the modulated
logits — no external calibration needed.

Architecture
────────────
  ┌──────────────────────────────────────┐
  │  DSparkEchoScheduler                 │
  │                                      │
  │  ┌──────────────┐  ┌───────────────┐ │
  │  │ Echo-DSRN    │  │ Confidence    │ │
  │  │ Draft Model  │──│ (α-modulated) │ │
  │  │ (α > 0)      │  │               │ │
  │  └──────────────┘  └──────┬────────┘ │
  │                           │          │
  │                    ┌──────▼────────┐ │
  │                    │ Dynamic       │ │
  │                    │ Cutoff ℓ      │ │
  │                    └──────┬────────┘ │
  │                           │          │
  │                    ┌──────▼────────┐ │
  │                    │ Verification  │ │
  │                    │ (Target LLM)  │ │
  │                    └──────────────┘ │
  └──────────────────────────────────────┘
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .confidence import compute_confidence, extract_gate_logits


@dataclass
class DSparkEchoConfig:
    """Configuration for the DSpark-Echo speculative decoding scheduler."""

    max_draft_len: int = 8
    tau_load: float = 0.5
    confidence_aggregation: str = "mean"
    surprise_temperature_alpha: float = 1.0
    enable_confidence: bool = True


class DSparkEchoScheduler:
    """
    DSpark speculative decoding scheduler with surprise-temperature confidence.

    The draft model must have surprise_temperature_alpha > 0 set in its config.
    This couples the output distribution to the model's internal surprise gate,
    producing naturally-calibrated token confidence scores.

    Usage::

        draft = EchoForCausalLM.from_pretrained(..., surprise_temperature_alpha=1.0)
        target = AutoModelForCausalLM.from_pretrained(...)

        scheduler = DSparkEchoScheduler(draft_model=draft)
        draft_ids, conf = scheduler.draft(input_ids)
        accepted = scheduler.verify(target, input_ids, draft_ids, conf["cutoff_lens"])
    """

    def __init__(self, draft_model: nn.Module, config: Optional[DSparkEchoConfig] = None):
        self.draft_model = draft_model
        self.config = config or DSparkEchoConfig()
        self._enable_gate_logits()

    def _enable_gate_logits(self):
        if hasattr(self.draft_model, "config"):
            cfg = self.draft_model.config
            if hasattr(cfg, "output_surprise_gate_logits"):
                cfg.output_surprise_gate_logits = True
            if hasattr(cfg, "surprise_temperature_alpha"):
                cfg.surprise_temperature_alpha = self.config.surprise_temperature_alpha

    # ── Draft ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def draft(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values=None,
        **kwargs,
    ) -> Tuple[torch.LongTensor, dict]:
        """Run draft model autoregressively, collect gate_logits + confidence."""
        max_len = self.config.max_draft_len
        draft_ids_list, gate_logits_steps = [], []
        current_ids, current_pkv = input_ids, past_key_values

        for _ in range(max_len):
            out = self.draft_model(
                input_ids=current_ids,
                attention_mask=attention_mask if current_pkv is None else None,
                past_key_values=current_pkv,
                use_cache=True,
                return_dict=True,
                **kwargs,
            )
            next_token = out.logits[:, -1:, :].argmax(dim=-1)
            draft_ids_list.append(next_token)
            gl = extract_gate_logits(out)
            if gl is not None:
                gate_logits_steps.append([g[:, -1:, :] for g in gl])
            current_ids = next_token
            current_pkv = out.past_key_values
            attention_mask = None

        draft_ids = torch.cat(draft_ids_list, dim=1)  # (B, draft_len)

        # Aggregate gate_logits across steps
        if gate_logits_steps:
            num_layers = len(gate_logits_steps[0])
            gl_combined = [
                torch.cat([step[li] for step in gate_logits_steps], dim=1)
                for li in range(num_layers)
            ]
        else:
            gl_combined = []

        conf = {}
        if self.config.enable_confidence and gl_combined:
            conf = compute_confidence(
                gl_combined,
                tau_load=self.config.tau_load,
                return_cutoff=True,
                aggregation=self.config.confidence_aggregation,
            )

        return draft_ids, conf

    # ── Verify ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def verify(
        self,
        target_model: nn.Module,
        input_ids: torch.LongTensor,
        draft_ids: torch.LongTensor,
        cutoff_lens: Optional[torch.LongTensor] = None,
    ) -> torch.BoolTensor:
        """Verify draft tokens against target model. Returns (B, draft_len) bool mask."""
        prefix_len = input_ids.shape[1]
        full_input = torch.cat([input_ids, draft_ids], dim=1)
        target_out = target_model(input_ids=full_input)
        target_logits = target_out.logits if hasattr(target_out, "logits") else target_out[0]
        target_preds = target_logits[:, prefix_len - 1 : prefix_len - 1 + draft_ids.shape[1], :]
        target_tokens = target_preds.argmax(dim=-1)
        accepted = draft_ids == target_tokens

        if cutoff_lens is not None and self.config.enable_confidence:
            positions = torch.arange(draft_ids.shape[1], device=draft_ids.device).unsqueeze(0)
            accepted = accepted & (positions < cutoff_lens.unsqueeze(1))

        return accepted

    # ── Full step ────────────────────────────────────────────────────────

    @torch.no_grad()
    def step(
        self,
        input_ids: torch.LongTensor,
        target_model: nn.Module,
    ) -> dict:
        """Draft → verify → extract accepted tokens."""
        draft_ids, conf = self.draft(input_ids)
        accepted = self.verify(target_model, input_ids, draft_ids, conf.get("cutoff_lens"))
        # Collect accepted tokens per batch element
        tokens_list = []
        for b in range(accepted.shape[0]):
            row = accepted[b]
            false_pos = (~row).nonzero(as_tuple=True)[0]
            end = false_pos[0].item() if len(false_pos) > 0 else row.shape[0]
            tokens_list.append(draft_ids[b, :end])
        max_len = max(len(t) for t in tokens_list)
        padded = torch.zeros(
            accepted.shape[0], max(max_len, 1), dtype=torch.long, device=draft_ids.device
        )
        for b, t in enumerate(tokens_list):
            if len(t) > 0:
                padded[b, : len(t)] = t
        return {
            "accepted_tokens": padded,
            "accepted_mask": accepted,
            "draft_ids": draft_ids,
            "confidence": conf,
        }
