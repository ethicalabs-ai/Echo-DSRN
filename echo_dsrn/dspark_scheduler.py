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

Cross-vocabulary (TLI) support
──────────────────────────────
When ``DSparkEchoConfig.vocab_mapper`` is set, the draft model may use a
different tokenizer than the target.  Draft proposals are restricted to the
shared token-level intersection $I$ (see ``echo_dsrn.speculative.vocab_mapper``)
and translated to the target vocabulary before verification.  The draft
conditioning context is built by translating the target prefix token-by-token;
out-of-intersection target tokens map to the draft UNK token.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .confidence import compute_confidence, extract_gate_logits
from .speculative.vocab_mapper import VocabMapper


@dataclass
class DSparkEchoConfig:
    """Configuration for the DSpark-Echo speculative decoding scheduler."""

    max_draft_len: int = 8
    tau_load: float = 0.5
    confidence_aggregation: str = "mean"
    surprise_temperature_alpha: float = 1.0
    enable_confidence: bool = True
    vocab_mapper: Optional[VocabMapper] = None


def _cache_is_truncatable(cache) -> bool:
    """Whether ``cache`` can be safely truncated to a shorter prefix.

    Plain attention layers store per-position (k, v) tensors and truncate
    exactly.  Hybrid targets (transformers 5.x linear-attention layers, e.g.
    gated delta net) additionally carry recurrent/convolutional state that
    encodes history beyond the raw positions and cannot be rewound — for those
    the scheduler falls back to a full re-prefix instead.
    """
    if cache is None:
        return True
    for layer in cache.layers if hasattr(cache, "layers") else cache:
        if hasattr(layer, "conv_states") or hasattr(layer, "recurrent_states"):
            return False
        if hasattr(layer, "keys"):  # transformers >= 5.x DynamicLayer
            continue
        try:  # legacy (k, v) tuple
            k, v = layer
            _ = k, v
        except (TypeError, ValueError):
            return False
    return True


def _truncate_kv_cache(cache, length: int):
    """Truncate a target KV cache so it covers only the first ``length`` positions.

    Works with ``transformers`` ``DynamicCache`` (5.x ``DynamicLayer`` objects)
    and legacy list-of-(k, v) tuples.  Returns ``None`` for ``None`` input.
    """
    if cache is None:
        return None
    layers = []
    for layer in cache.layers if hasattr(cache, "layers") else cache:
        if hasattr(layer, "keys"):  # transformers >= 5.x DynamicLayer
            k, v = layer.keys, layer.values
        else:  # legacy (k, v) tuple
            k, v = layer
        if k is None:
            continue
        layers.append((k[:, :, :length, :], v[:, :, :length, :]))
    try:
        from transformers import DynamicCache

        return DynamicCache(layers)
    except (ImportError, TypeError, ValueError):
        return layers


class DSparkEchoScheduler:
    """
    DSpark speculative decoding scheduler with surprise-temperature confidence.

    The draft model must have surprise_temperature_alpha > 0 set in its config.
    This couples the output distribution to the model's internal surprise gate,
    producing naturally-calibrated token confidence scores.

    Usage (same vocabulary)::

        draft = EchoForCausalLM.from_pretrained(..., surprise_temperature_alpha=1.0)
        target = AutoModelForCausalLM.from_pretrained(...)

        scheduler = DSparkEchoScheduler(draft_model=draft)
        draft_ids, conf = scheduler.draft(input_ids)
        accepted = scheduler.verify(target, input_ids, draft_ids, conf["cutoff_lens"])

    Usage (cross-vocabulary, TLI)::

        mapper = build_vocab_intersection(draft_tokenizer, target_tokenizer)
        scheduler = DSparkEchoScheduler(
            draft_model=draft,
            config=DSparkEchoConfig(vocab_mapper=mapper),
        )
        result = scheduler.step(target_ids, target, return_cache=True)
    """

    def __init__(self, draft_model: nn.Module, config: Optional[DSparkEchoConfig] = None):
        self.draft_model = draft_model
        self.config = config or DSparkEchoConfig()
        self._enable_gate_logits()
        # Rollback state, populated by draft().
        self._last_hc_states: Optional[list] = None
        self._last_draft_final_state: Optional[list] = None
        self._last_draft_prefix_len: int = 0

    def _enable_gate_logits(self):
        if hasattr(self.draft_model, "config"):
            cfg = self.draft_model.config
            if hasattr(cfg, "output_surprise_gate_logits"):
                cfg.output_surprise_gate_logits = True
            if hasattr(cfg, "surprise_temperature_alpha"):
                cfg.surprise_temperature_alpha = self.config.surprise_temperature_alpha

    # ── State helpers (rollback) ──────────────────────────────────────────

    @staticmethod
    def _snapshot_state(past_key_values):
        """Copy the draft state so later in-place cache updates cannot corrupt it."""
        if past_key_values is None:
            return None
        if hasattr(past_key_values, "states"):  # EchoCache
            return list(past_key_values.states)
        return list(past_key_values)

    @staticmethod
    def _state_seq_len(state) -> int:
        """Number of positions covered by a draft cache state (0 when unknown)."""
        if state is None:
            return 0
        if hasattr(state, "get_seq_length"):
            return state.get_seq_length()
        if isinstance(state, (list, tuple)) and state:
            layer = state[-1]
            if isinstance(layer, (list, tuple)) and len(layer) == 4:
                return layer[2].shape[2]
        return 0

    # ── Draft ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def draft(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values=None,
        **kwargs,
    ) -> Tuple[torch.LongTensor, dict]:
        """Run draft model autoregressively, collect gate_logits + confidence.

        ``input_ids`` are in the draft vocabulary.  When a ``vocab_mapper`` is
        configured the proposals are restricted to the intersection $I$, so
        every returned ``draft_ids`` token is translatable to the target
        vocabulary.  Per-step state snapshots are recorded for
        :meth:`rollback`.
        """
        max_len = self.config.max_draft_len
        mapper = self.config.vocab_mapper
        draft_ids_list, gate_logits_steps = [], []
        current_ids, current_pkv = input_ids, past_key_values

        # Snapshot the input state before any forward pass can mutate it.
        snapshots = [self._snapshot_state(past_key_values)]
        prefix_len = input_ids.shape[1] + self._state_seq_len(past_key_values)

        for _ in range(max_len):
            out = self.draft_model(
                input_ids=current_ids,
                attention_mask=attention_mask if current_pkv is None else None,
                past_key_values=current_pkv,
                use_cache=True,
                return_dict=True,
                **kwargs,
            )
            logits = out.logits[:, -1:, :]
            if mapper is not None:
                logits = mapper.mask_logits(logits)
            next_token = logits.argmax(dim=-1)
            draft_ids_list.append(next_token)
            gl = extract_gate_logits(out)
            if gl is not None:
                gate_logits_steps.append([g[:, -1:, :] for g in gl])
            current_ids = next_token
            current_pkv = out.past_key_values
            snapshots.append(self._snapshot_state(current_pkv))
            attention_mask = None

        draft_ids = torch.cat(draft_ids_list, dim=1)  # (B, draft_len)

        # Record rollback state: per-step (h, c) plus the final full state
        # (k/v is monotonic, so rollback slices the final k/v to the prefix).
        self._last_hc_states = [self._extract_hc(s) for s in snapshots]
        self._last_draft_final_state = snapshots[-1]
        self._last_draft_prefix_len = prefix_len

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

    @staticmethod
    def _extract_hc(state):
        """Keep only the recurrent (h, c) part of a draft state."""
        if state is None:
            return None
        return [(layer[0], layer[1]) for layer in state]

    # ── Rollback ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def rollback(self, n_accepted: torch.LongTensor):
        """Return the draft cache state at the accepted prefix.

        ``n_accepted`` has shape ``(B,)`` and holds, per batch row, the number
        of draft tokens accepted by the last verification.  The returned state
        covers exactly the first ``prefix_len + min(n, max_draft_len - 1)``
        positions (the last drafted token never enters the cache, so a fully
        accepted round returns the final state, which is one token short of the
        full chunk — exactly the state the generation loop needs to re-feed the
        last token).  Usable as ``past_key_values`` for the next :meth:`draft`
        call.

        The state is exact for pure-DSRN drafts (recurrent (h, c) only) and for
        batch size 1.  For hybrid drafts with batched rows that accepted
        different numbers of tokens, the attention (k, v) tensors are padded to
        the longest row — draft quality may degrade for shorter rows, but
        verification remains lossless.
        """
        hc_steps = self._last_hc_states
        final_state = self._last_draft_final_state
        prefix_len = self._last_draft_prefix_len
        if hc_steps is None:
            raise RuntimeError("rollback() requires a prior draft() call")

        n_per_row = [int(n) for n in n_accepted.tolist()]
        # The final cache covers prefix + (max_draft_len - 1) positions; the
        # last generated token is only fed back on the next round.
        m_per_row = [min(n, max(0, self.config.max_draft_len - 1)) for n in n_per_row]
        # Nothing accepted from a fresh start: the pre-draft state was None.
        if hc_steps[0] is None and all(m == 0 for m in m_per_row):
            return None
        n_layers = (
            len(hc_steps[0])
            if hc_steps[0] is not None
            else (len(final_state) if final_state is not None else 0)
        )
        if n_layers == 0:
            return None
        has_kv = final_state is not None and len(final_state[0]) == 4
        ref_hc = next((s for s in hc_steps if s is not None), None)

        rows = []
        for b, m in enumerate(m_per_row):
            # h/c of the state covering prefix + m positions: snapshot index
            # is m + 1 (m == 0 → the pre-draft state, index 0).
            hc = hc_steps[m + 1] if m > 0 else hc_steps[0]
            layers = []
            for li in range(n_layers):
                if hc is not None:
                    hb, cb = hc[li][0][b : b + 1], hc[li][1][b : b + 1]
                else:
                    # Fresh start (accepted nothing with no prior cache):
                    # the model's initial state is exactly zeros / empty k,v.
                    hb = torch.zeros_like(ref_hc[li][0][b : b + 1])
                    cb = torch.zeros_like(ref_hc[li][1][b : b + 1])
                if has_kv:
                    k_final, v_final = final_state[li][2], final_state[li][3]
                    if hc is not None:
                        kb = k_final[b : b + 1, :, : prefix_len + m, :]
                        vb = v_final[b : b + 1, :, : prefix_len + m, :]
                    else:
                        kb = k_final[b : b + 1, :, :0, :]
                        vb = v_final[b : b + 1, :, :0, :]
                    layers.append((hb, cb, kb, vb))
                else:
                    layers.append((hb, cb))
            rows.append(layers)

        # Pad attention k/v to the longest row so the batch stays rectangular.
        if has_kv and len(rows) > 1:
            max_t = max(rows[b][li][2].shape[2] for li in range(n_layers) for b in range(len(rows)))
            for b in range(len(rows)):
                for li in range(n_layers):
                    hb, cb, kb, vb = rows[b][li]
                    pad = max_t - kb.shape[2]
                    if pad > 0:
                        rows[b][li] = (
                            hb,
                            cb,
                            F.pad(kb, (0, 0, 0, pad)),
                            F.pad(vb, (0, 0, 0, pad)),
                        )

        # Re-batch rows → per-layer tuples with a leading batch dimension.
        layers_out = []
        for li in range(n_layers):
            hs = torch.cat([rows[b][li][0] for b in range(len(rows))], dim=0)
            cs = torch.cat([rows[b][li][1] for b in range(len(rows))], dim=0)
            if has_kv:
                ks = torch.cat([rows[b][li][2] for b in range(len(rows))], dim=0)
                vs = torch.cat([rows[b][li][3] for b in range(len(rows))], dim=0)
                layers_out.append((hs, cs, ks, vs))
            else:
                layers_out.append((hs, cs))
        return layers_out

    # ── Verify ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def verify(
        self,
        target_model: nn.Module,
        input_ids: torch.LongTensor,
        draft_ids: torch.LongTensor,
        cutoff_lens: Optional[torch.LongTensor] = None,
        past_key_values=None,
        return_details: bool = False,
    ):
        """Verify draft tokens against the target model.

        ``input_ids`` are the current prefix in the **target** vocabulary;
        ``draft_ids`` are draft-vocabulary proposals (translated internally when
        a ``vocab_mapper`` is configured).  Returns ``(B, draft_len)`` bool mask
        of accepted positions, or ``(mask, details)`` when ``return_details``
        is set, where ``details`` contains ``target_tokens`` (the target's
        greedy ids at each draft position, in the target vocabulary) and
        ``cache`` (the target KV cache from the verification forward).

        When ``past_key_values`` is provided it must cover ``input_ids[:-1]``
        (the last prefix token is fed together with the drafts).
        """
        mapper = self.config.vocab_mapper
        draft_len = draft_ids.shape[1]
        target_drafts = (
            mapper.translate_draft_to_target(draft_ids) if mapper is not None else draft_ids
        )
        use_cache = return_details
        forward_kwargs = {"use_cache": True} if use_cache else {}

        if past_key_values is not None:
            new_ids = torch.cat([input_ids[:, -1:], target_drafts], dim=1)
            out = target_model(input_ids=new_ids, past_key_values=past_key_values, **forward_kwargs)
            logits = out.logits if hasattr(out, "logits") else out[0]
            target_preds = logits[:, :-1, :]
        else:
            full_input = torch.cat([input_ids, target_drafts], dim=1)
            out = target_model(input_ids=full_input, **forward_kwargs)
            logits = out.logits if hasattr(out, "logits") else out[0]
            prefix_len = input_ids.shape[1]
            target_preds = logits[:, prefix_len - 1 : prefix_len - 1 + draft_len, :]

        target_tokens = target_preds.argmax(dim=-1)
        if mapper is not None:
            # Exact-string comparison (fuzzy representative ids would reject
            # correct proposals that collide under normalization).
            accepted = mapper.matches_draft_to_target(draft_ids, target_tokens)
        else:
            accepted = target_drafts == target_tokens

        if cutoff_lens is not None and self.config.enable_confidence:
            positions = torch.arange(draft_len, device=draft_ids.device).unsqueeze(0)
            accepted = accepted & (positions < cutoff_lens.unsqueeze(1))

        if return_details:
            cache = out.past_key_values if hasattr(out, "past_key_values") else None
            return accepted, {"target_tokens": target_tokens, "cache": cache}
        return accepted

    # ── Full step ────────────────────────────────────────────────────────

    @torch.no_grad()
    def step(
        self,
        input_ids: torch.LongTensor,
        target_model: nn.Module,
        past_key_values=None,
        target_past_key_values=None,
        return_cache: bool = False,
    ) -> dict:
        """Draft → verify → extract accepted tokens → roll back draft state.

        Cross-vocabulary (TLI) contract
        ───────────────────────────────
        When ``config.vocab_mapper`` is set, ``input_ids`` must be the current
        accepted prefix in the **target** vocabulary.  The draft conditioning
        context is derived by translating it token-by-token (out-of-
        intersection tokens map to the draft UNK token).  The returned
        ``accepted_tokens`` are in the target vocabulary.

        Cache contract (stateful generation loop)
        ─────────────────────────────────────────
        ``past_key_values`` is the draft state covering the draft-vocabulary
        translation of ``input_ids[:-1]`` (all but the last token); pass the
        ``past_key_values`` value returned by the previous ``step()``.
        ``target_past_key_values`` is the target cache covering ``input_ids[:-1]``
        (the last token is fed as part of the verification input); pass the
        ``target_cache`` value returned by the previous call.

        The returned ``accepted_tokens`` include the target's own greedy token
        at the first rejected draft position (standard speculative-decoding
        replacement), so the loop is lossless and self-advancing.  The returned
        ``past_key_values`` (draft state) and ``target_cache`` are rolled back /
        advanced to cover the new prefix minus its last token, ready for the
        next call.

        Targets whose KV cache cannot be truncated (hybrid models with
        linear-attention layers carrying recurrent/convolutional state, e.g.
        gated delta net) return ``target_cache=None``: the next call re-prefixes
        the full sequence instead of continuing from a cache.  This is always
        lossless — only the target-side compute differs.
        """
        mapper = self.config.vocab_mapper
        if mapper is not None:
            if past_key_values is None:
                draft_input = mapper.translate_target_to_draft(input_ids)
            else:
                draft_input = mapper.translate_target_to_draft(input_ids[:, -1:])
        else:
            draft_input = input_ids if past_key_values is None else input_ids[:, -1:]

        draft_ids, conf = self.draft(draft_input, past_key_values=past_key_values)
        accepted, details = self.verify(
            target_model,
            input_ids,
            draft_ids,
            cutoff_lens=conf.get("cutoff_lens"),
            past_key_values=target_past_key_values,
            return_details=True,
        )
        target_tokens = details["target_tokens"]

        # Per row: accepted draft tokens (target greedy at accepted positions)
        # plus the target's replacement token at the first rejection.
        tokens_list, n_accepted_list = [], []
        for b in range(accepted.shape[0]):
            row = accepted[b]
            false_pos = (~row).nonzero(as_tuple=True)[0]
            k = false_pos[0].item() if len(false_pos) > 0 else row.shape[0]
            n_accepted_list.append(k)
            if k < row.shape[0]:
                chunk = torch.cat([target_tokens[b, :k], target_tokens[b, k : k + 1]])
            else:
                chunk = target_tokens[b, :k]
            tokens_list.append(chunk)

        max_len = max(len(t) for t in tokens_list)
        padded = torch.zeros(
            accepted.shape[0], max(max_len, 1), dtype=torch.long, device=draft_ids.device
        )
        for b, t in enumerate(tokens_list):
            if len(t) > 0:
                padded[b, : len(t)] = t

        n_accepted = torch.tensor(n_accepted_list, dtype=torch.long, device=draft_ids.device)
        # Loop invariant: the returned draft state covers the new prefix minus
        # its last token.  rollback() clamps to max_draft_len - 1 internally,
        # so a fully-accepted round returns the final cache (one token short of
        # the chunk) and that token is re-fed next round.
        new_draft_state = self.rollback(n_accepted)

        # Target cache for the next round (covers the new prefix minus its
        # last token).  Targets with non-truncatable caches (hybrid models with
        # linear-attention / recurrent-state layers) skip truncation and
        # rebuild entirely: target_cache stays None and the next verify()
        # re-prefixes the full sequence — always lossless, just slower.
        target_cache = None
        if return_cache:
            new_prefix_len = input_ids.shape[1] + padded.shape[1]
            partial = bool((n_accepted < draft_ids.shape[1]).any())
            truncatable = _cache_is_truncatable(details["cache"])
            if not partial and truncatable:
                target_cache = _truncate_kv_cache(details["cache"], new_prefix_len - 1)
            elif partial and accepted.shape[0] == 1 and truncatable:
                # Rebuild cheaply: re-run the target over just the new tokens.
                if target_past_key_values is not None:
                    out = target_model(
                        input_ids=padded[:, : int(n_accepted[0]) + 1],
                        past_key_values=target_past_key_values,
                        use_cache=True,
                    )
                else:
                    out = target_model(
                        input_ids=torch.cat([input_ids, padded], dim=1), use_cache=True
                    )
                cache = out.past_key_values if hasattr(out, "past_key_values") else None
                target_cache = _truncate_kv_cache(cache, new_prefix_len - 1)
            # Heterogeneous multi-row batches fall back to a re-prefix next
            # round (target_cache stays None) — always correct, just slower.

        return {
            "accepted_tokens": padded,
            "accepted_mask": accepted,
            "draft_ids": draft_ids,
            "target_tokens": target_tokens,
            "n_accepted": n_accepted,
            "confidence": conf,
            "past_key_values": new_draft_state,
            "target_cache": target_cache,
        }
