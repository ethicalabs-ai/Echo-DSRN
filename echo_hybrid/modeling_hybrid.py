"""
echo_hybrid/modeling_hybrid.py
────────────────────────────────────────────────────────────────────────────
HybridEchoModel  — Qwen2 backbone with DSRN memory injectors
HybridEchoForCausalLM — wraps the model with an LM head and generation support
HybridEchoCache  — DynamicCache subclass that also carries DSRN recurrent states

FINAL STABLE ARCHITECTURE: Stateless Backbone with Position Alignment.
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as gradient_checkpoint
from transformers import GenerationMixin
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Model,
    Qwen2PreTrainedModel,
)

from .configuration_hybrid import HybridEchoConfig
from .dsrn_memory_block import DSRNMemoryInjector

# ─────────────────────────────────────────────────────────────────────────────
# Cache
# ─────────────────────────────────────────────────────────────────────────────


class HybridEchoCache(DynamicCache):
    def __init__(self, config=None):
        # transformers 5.x: DynamicCache uses self.layers (not self.key_cache).
        # config must be passed so __init__ creates the per-layer cache objects;
        # without it self.layers=[] and update() silently discards KV tensors.
        super().__init__(config=config)
        self.dsrn_states: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.seen_tokens = 0

    @classmethod
    def from_legacy_cache(
        cls,
        dsrn_states: List[Tuple[torch.Tensor, torch.Tensor]],
        config=None,
    ) -> "HybridEchoCache":
        cache = cls(config=config)
        cache.dsrn_states = dsrn_states
        return cache

    def reorder_cache(self, beam_idx: torch.LongTensor):
        super().reorder_cache(beam_idx)
        self.dsrn_states = [
            (
                h.index_select(0, beam_idx.to(h.device)),
                c.index_select(0, beam_idx.to(c.device)),
            )
            for h, c in self.dsrn_states
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Inner model — Qwen2 backbone + hooked DSRN injectors
# ─────────────────────────────────────────────────────────────────────────────


class HybridEchoModel(Qwen2PreTrainedModel):
    config_class = HybridEchoConfig
    _tp_plan = {}  # No tensor parallel plan; required by vLLM TransformersForCausalLM

    def __init__(self, config: HybridEchoConfig):
        super().__init__(config)
        self.backbone = Qwen2Model(config)
        stride = config.dsrn_injection_stride
        self.num_injectors = config.num_hidden_layers // stride
        self.memory_injectors = nn.ModuleList(
            [DSRNMemoryInjector(config) for _ in range(self.num_injectors)]
        )
        self._dsrn_input_states = []
        self._dsrn_output_states = []
        self._eos_mask = None
        self._hook_handles = []
        self._register_dsrn_hooks()
        self.gradient_checkpointing = False

    def _apply_dsrn_critical_inits(self):
        bias_val = getattr(self.config, "gate_bias_init", 1.0)
        for injector in self.memory_injectors:
            nn.init.zeros_(injector.linear_read.weight)
            nn.init.constant_(injector.linear_gate.bias, bias_val)
            nn.init.zeros_(injector.surprise_lambda)
            w = injector.linear_pred.weight
            if w.dtype in (torch.bfloat16, torch.float16):
                tmp = torch.empty_like(w, dtype=torch.float32, device=w.device)
                nn.init.orthogonal_(tmp, gain=0.1)
                with torch.no_grad():
                    w.copy_(tmp.to(device=w.device, dtype=w.dtype))
            else:
                nn.init.orthogonal_(w, gain=0.1)

    def _make_hook(self, injector_idx: int):
        def hook(module: nn.Module, layer_input: tuple, layer_output):
            if not self._dsrn_input_states:
                return layer_output
            if isinstance(layer_output, torch.Tensor):
                hidden_states = layer_output
                is_bare_tensor = True
            else:
                hidden_states = layer_output[0]
                is_bare_tensor = False

            h_prev, c_prev = self._dsrn_input_states[injector_idx]
            injector = self.memory_injectors[injector_idx]
            eos_mask = self._eos_mask

            if self.gradient_checkpointing and self.training:
                # Wrap the injector in a gradient checkpoint so its intermediate
                # activations are discarded during the forward pass and recomputed
                # per-injector during backward.  use_reentrant=False is required
                # for compatibility with torch.compile and nested checkpointing.
                def injector_fn(hs, h, c):
                    return injector(hs, h, c, eos_mask=eos_mask)

                x_out, h_new, c_new = gradient_checkpoint(
                    injector_fn,
                    hidden_states,
                    h_prev,
                    c_prev,
                    use_reentrant=False,
                )
            else:
                x_out, h_new, c_new = injector(hidden_states, h_prev, c_prev, eos_mask=eos_mask)

            self._dsrn_output_states.append((h_new, c_new))
            if is_bare_tensor:
                return x_out
            return (x_out,) + layer_output[1:]

        return hook

    def _register_dsrn_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []
        stride = self.config.dsrn_injection_stride
        layers = self.backbone.layers
        for injector_idx in range(self.num_injectors):
            target_layer_idx = (injector_idx + 1) * stride - 1
            if target_layer_idx < len(layers):
                handle = layers[target_layer_idx].register_forward_hook(
                    self._make_hook(injector_idx)
                )
                self._hook_handles.append(handle)

    def get_input_embeddings(self):
        return self.backbone.embed_tokens

    def set_input_embeddings(self, value):
        self.backbone.embed_tokens = value

    def _init_dsrn_states(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        D = self.config.hidden_size
        D_s = self.config.dsrn_state_dim
        return [
            (
                torch.zeros(batch_size, D, device=device, dtype=dtype),
                torch.zeros(batch_size, D_s, device=device, dtype=dtype),
            )
            for _ in range(self.num_injectors)
        ]

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[HybridEchoCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        eos_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # Two independent cache controls:
        #   use_dsrn_cache — whether to return a HybridEchoCache carrying DSRN states.
        #   use_kv_cache   — config flag controlling backbone KV cache (→ backbone_use_cache below).
        # Ablation mode disables backbone KV but DSRN states must still survive across
        # steps. Default is True so generation always carries recurrent state forward.
        use_dsrn_cache = use_cache if use_cache is not None else True

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            B = input_ids.shape[0]
            device = input_ids.device
            dtype = self.backbone.embed_tokens.weight.dtype
            seq_len = input_ids.shape[1]
        elif inputs_embeds is not None:
            B = inputs_embeds.shape[0]
            device = inputs_embeds.device
            dtype = inputs_embeds.dtype
            seq_len = inputs_embeds.shape[1]
        else:
            raise ValueError("Provide input_ids or inputs_embeds.")

        if past_key_values is not None and isinstance(past_key_values, HybridEchoCache):
            dsrn_states = past_key_values.dsrn_states
            if not dsrn_states:
                dsrn_states = self._init_dsrn_states(B, device, dtype)
        else:
            dsrn_states = self._init_dsrn_states(B, device, dtype)
            if past_key_values is None:
                past_key_values = HybridEchoCache(config=self.config) if use_dsrn_cache else None

        self._dsrn_input_states = dsrn_states  # ALWAYS run injectors regardless of cache mode.
        # use_dsrn_cache only gates whether output states are *returned* in the
        # HybridEchoCache.  Setting this to [] when use_cache=False (training mode)
        # caused the hook to exit early → injectors bypassed → grad_norm=0.
        self._dsrn_output_states = []
        self._eos_mask = eos_mask

        # ── Qwen2 requires 2D position_ids ────────────────────────────────
        final_position_ids = position_ids if position_ids is not None else cache_position
        if final_position_ids is not None:
            if final_position_ids.dim() == 1:
                final_position_ids = final_position_ids.unsqueeze(0)
            if final_position_ids.shape[1] > seq_len:
                final_position_ids = final_position_ids[:, -seq_len:]

        # ── STATELESS BACKBONE WITH POSITION ALIGNMENT ───────────────────
        # backbone_use_cache is driven solely by config.use_kv_cache — it is
        # independent of use_dsrn_cache.  Disabling DSRN state return must
        # never silently disable the backbone KV cache.
        backbone_use_cache = getattr(self.config, "use_kv_cache", True)
        if not backbone_use_cache:
            # Force attention_mask=None to avoid SDPA length mismatches in stateless mode
            attention_mask = None

            # ── CRITICAL: Stateless RoPE Grounding ──────────────────────
            # When the KV-cache is purged every step, the backbone sees an
            # empty KV-cache and defaults position to 0 for every token,
            # causing the "systemsystemsystem" identity collapse.
            # We derive the true absolute position from seen_tokens so RoPE
            # stays coherent even without a physical KV history.
            if final_position_ids is None:
                # Logical position of the first token in this call
                offset = 0
                if past_key_values is not None and hasattr(past_key_values, "seen_tokens"):
                    offset = past_key_values.seen_tokens
                final_position_ids = torch.arange(
                    offset, offset + seq_len, device=device, dtype=torch.long
                ).unsqueeze(
                    0
                )  # (1, seq_len) — broadcast over batch

        backbone_out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=final_position_ids,
            past_key_values=past_key_values if backbone_use_cache else None,
            inputs_embeds=inputs_embeds,
            use_cache=backbone_use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        new_dsrn_states = self._dsrn_output_states
        new_cache = backbone_out.past_key_values

        if use_dsrn_cache and (new_cache is not None or not backbone_use_cache):
            if new_cache is None or not isinstance(new_cache, HybridEchoCache):
                hybrid_cache = HybridEchoCache(config=self.config)
                if new_cache is not None:
                    if hasattr(new_cache, 'layers'):
                        hybrid_cache.layers = new_cache.layers
                    if hasattr(new_cache, 'get_seq_length'):
                        hybrid_cache.seen_tokens = new_cache.get_seq_length()
                elif past_key_values is not None:
                    if hasattr(past_key_values, 'seen_tokens'):
                        hybrid_cache.seen_tokens = past_key_values.seen_tokens
                    elif hasattr(past_key_values, 'get_seq_length'):
                        hybrid_cache.seen_tokens = past_key_values.get_seq_length()
                new_cache = hybrid_cache

            new_cache.dsrn_states = new_dsrn_states
            if not getattr(self.config, "use_kv_cache", True):
                new_cache.seen_tokens += seq_len

        elif past_key_values is not None:
            if isinstance(past_key_values, HybridEchoCache):
                past_key_values.dsrn_states = new_dsrn_states
                if not getattr(self.config, "use_kv_cache", True):
                    past_key_values.seen_tokens += seq_len
                new_cache = past_key_values

        self._dsrn_input_states = []
        self._dsrn_output_states = []
        self._eos_mask = None

        if not return_dict:
            return (backbone_out.last_hidden_state, new_cache)

        backbone_out.past_key_values = new_cache
        return backbone_out


class HybridEchoForCausalLM(Qwen2PreTrainedModel, GenerationMixin):
    config_class = HybridEchoConfig
    _tp_plan = {}  # No tensor parallel plan; required by vLLM TransformersForCausalLM
    _is_causal = True
    supports_gradient_checkpointing = True
    _supports_cache_class = True
    _supports_static_cache = False
    main_input_name = "input_ids"

    def __init__(self, config: HybridEchoConfig):
        super().__init__(config)
        self.model = HybridEchoModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        # Do NOT call _apply_dsrn_critical_inits here - it overwrites checkpoint weights

    def get_input_embeddings(self):
        return self.model.backbone.embed_tokens

    def set_input_embeddings(self, value):
        self.model.backbone.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def _set_gradient_checkpointing(self, enable=True, gradient_checkpointing_func=None):
        self.model.gradient_checkpointing = enable
        self.model.backbone.gradient_checkpointing = enable
        if enable:
            # Gradient checkpointing and the KV cache are mutually exclusive:
            # GC discards activations between layers while the cache would
            # re-materialise them, negating all memory savings.
            self.config.use_cache = False
            self.config.use_kv_cache = False

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[HybridEchoCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        eos_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            eos_mask=eos_mask,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        # logits stays in the model dtype (bfloat16) — do NOT cast the full tensor
        # to float32 here.  A global .float() materialises a second copy of
        # [batch, seq, vocab_size] in fp32, retaining it in the autograd graph
        # (+2.5 GiB at batch=2, seq=512 with vocab=151 936).  F.cross_entropy
        # handles numerical stability internally via per-row log-sum-exp; the
        # inline .float() cast below is fused into the cross-entropy kernel and
        # never stored as a standalone tensor.
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size).float(),  # fused cast
                shift_labels.view(-1),
            )

        if not return_dict:
            out = (logits,) + (outputs.past_key_values,)
            return ((loss,) + out) if loss is not None else out

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        **kwargs,
    ):
        past_len = 0
        if past_key_values is not None:
            # Prefer actual KV tensor count; fall back to seen_tokens in ablation
            # mode where KV tensors are empty (get_seq_length() returns 0).
            kv_len = (
                past_key_values.get_seq_length()
                if hasattr(past_key_values, "get_seq_length")
                else 0
            )
            if kv_len > 0:
                past_len = kv_len
            elif hasattr(past_key_values, "seen_tokens"):
                past_len = past_key_values.seen_tokens

        if cache_position is None:
            cache_position = torch.arange(
                past_len, past_len + input_ids.shape[1], device=input_ids.device
            )

        if past_len > 0:
            if input_ids.shape[1] > 1:
                input_ids = input_ids[:, -1:]

        use_cache = kwargs.get("use_cache")
        if use_cache is None:
            use_cache = getattr(self.config, "use_kv_cache", True)

        # In ablation mode the backbone has no physical KV cache to infer
        # position from, so we must pass position_ids explicitly.
        # cache_position already holds the correct absolute positions, so we
        # promote it to 2D position_ids for the backbone's RoPE computation.
        if not getattr(self.config, "use_kv_cache", True) and position_ids is None:
            position_ids = cache_position.unsqueeze(0)  # (1, seq_len)

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "position_ids": position_ids,
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        if isinstance(past_key_values, HybridEchoCache):
            past_key_values.reorder_cache(beam_idx)
        return past_key_values

    def generate(self, *args, **kwargs):
        """Override generate() to ensure HybridEchoCache is used from the start.

        Without this, generate() begins with past_key_values=None which causes
        HybridEchoModel.forward() to fall into the else-branch and create a plain
        DynamicCache.  On subsequent steps isinstance(pkv, HybridEchoCache) fails,
        dsrn_states resets to zeros every token, and the model collapses
        ('systemsystemsystem').

        We also enforce Standard Mode (CACHED+DSRN) by saving use_kv_cache=True on
        self.config so that HybridEchoModel.forward() passes the backbone KV cache
        through correctly — without it, backbone_use_cache evaluates to False and
        the backbone receives past_key_values=None every step.
        """
        self.config.use_kv_cache = True
        if not isinstance(kwargs.get("past_key_values"), HybridEchoCache):
            kwargs["past_key_values"] = HybridEchoCache(config=self.config)
        return super().generate(*args, **kwargs)
