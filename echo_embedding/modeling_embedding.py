from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast

try:
    # pyrefly: ignore [missing-import]
    from .configuration_echo import EchoConfig

    # pyrefly: ignore [missing-import]
    from .modeling_echo import (
        EchoModel,
        EchoPreTrainedModel,
        _flattened_segment_mask,
        _pool_hidden_states,
    )
except ImportError:
    from echo_dsrn.configuration_echo import EchoConfig
    from echo_dsrn.modeling_echo import (
        EchoModel,
        EchoPreTrainedModel,
        _flattened_segment_mask,
        _pool_hidden_states,
    )


class EchoModelForSentenceEmbedding(EchoPreTrainedModel):
    """
    Sentence embedding adapter for Echo-DSRN.
    Extracts the recurrent state 'c' or sequences from layers and shapes them
    for sentence-transformers compatibility.
    """

    _supports_attention_backend = True

    def __init__(self, config: EchoConfig):
        super().__init__(config)
        self.model = EchoModel(config)
        self.pooling_mode = getattr(config, "pooling_mode", "c_T")

        # Determine target dimension for the projection input
        if self.pooling_mode == "hybrid":
            proj_in_dim = config.hidden_size * (config.num_heads + 1)
        elif self.pooling_mode == "mean_x_out":
            proj_in_dim = config.hidden_size
        else:  # "c_T" or "mean_c_all"
            proj_in_dim = config.hidden_size * config.num_heads

        # Optional projection layer to map back to a specific target embedding dimension.
        self.project_embeddings = getattr(config, "project_embeddings", False)
        self.projection_mlp = getattr(config, "projection_mlp", False)
        if self.projection_mlp:
            target_dim = getattr(config, "embedding_dim", config.hidden_size)
            hidden_dim = getattr(config, "projection_hidden_dim", 1024)
            self.projection = nn.Sequential(
                nn.Linear(proj_in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, target_dim, bias=False),
            )
        elif self.project_embeddings:
            target_dim = getattr(config, "embedding_dim", config.hidden_size)
            self.projection = nn.Linear(proj_in_dim, target_dim, bias=False)
        else:
            self.projection = None

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embedding

    def set_input_embeddings(self, value):
        self.model.embedding = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        pooling_mode = getattr(self.config, "pooling_mode", "c_T")
        # Support explicit override via kwargs, else use pooling_mode
        explicit = kwargs.pop("output_all_states", None)
        output_all_states = (
            explicit if explicit is not None else (pooling_mode in ["mean_c_all", "hybrid"])
        )

        # vLLM's Transformers backend runs every sequence of a step as one
        # flattened [1, N] forward (no attention_mask, position_ids restarting
        # per sequence).  The DSRN recurrence cannot reset mid-forward, so
        # each segment runs as its own forward with fresh state.
        new_seq = _flattened_segment_mask(position_ids, input_ids, attention_mask)
        if new_seq is not None:
            return self._forward_flattened_segments(
                input_ids,
                position_ids,
                new_seq,
                return_dict,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                **kwargs,
            )

        # 1. Base model forward pass
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            output_all_states=output_all_states,
            **kwargs,
        )

        # Determine sequence length for broadcasting
        if input_ids is not None:
            seq_len = input_ids.shape[1]
        elif inputs_embeds is not None:
            seq_len = inputs_embeds.shape[1]
        else:
            seq_len = 1

        # 2. Pool representations according to pooling_mode (shared helper)
        pooled = _pool_hidden_states(outputs, pooling_mode, attention_mask)

        # 3. Apply optional projection
        if self.projection is not None:
            embeddings = self.projection(pooled)
        else:
            embeddings = pooled

        # 4. Broadcast to shape (Batch, Seq_Len, Dim) for pooling safety
        embeddings_3d = embeddings.unsqueeze(1).expand(-1, seq_len, -1)

        if not return_dict:
            return (embeddings_3d, outputs.past_key_values)

        result = BaseModelOutputWithPast(
            last_hidden_state=embeddings_3d,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        # Propagate all_c_all / all_h_all from the raw model output
        if hasattr(outputs, "all_c_all"):
            result.all_c_all = outputs.all_c_all
        if hasattr(outputs, "all_h_all"):
            result.all_h_all = outputs.all_h_all
        return result

    def _forward_flattened_segments(
        self,
        input_ids,
        position_ids,
        new_seq,
        return_dict,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """Run each flattened-batch segment as its own forward.

        vLLM concatenates all sequences scheduled in a step into one
        ``[1, N]`` forward with ``position_ids`` restarting at each sequence
        start.  The DSRN recurrence carries state across the whole forward
        (its boundary handling freezes, it does not reset), so the segments
        cannot share one scan.  Running each segment independently reproduces
        the single-request semantics exactly; the pooled vectors are then
        stitched back into the ``[1, N]`` layout vLLM expects.
        """
        seg_ids = torch.cumsum(new_seq.long(), dim=1) - 1  # (1, N)
        num_segs = int(seg_ids.max().item()) + 1
        seg_outputs = []
        for i in range(num_segs):
            sel = seg_ids[0] == i
            seg_outputs.append(
                self.forward(
                    input_ids=input_ids[:, sel],
                    position_ids=position_ids[:, sel],
                    **kwargs,
                )
            )

        if not return_dict:
            return (
                torch.cat([o[0] for o in seg_outputs], dim=1),
                seg_outputs[-1][1],
            )

        result = BaseModelOutputWithPast(
            last_hidden_state=torch.cat([o.last_hidden_state for o in seg_outputs], dim=1),
            past_key_values=seg_outputs[-1].past_key_values,
            hidden_states=(
                [
                    torch.cat([o.hidden_states[layer_idx] for o in seg_outputs], dim=1)
                    for layer_idx in range(len(seg_outputs[0].hidden_states))
                ]
                if seg_outputs[0].hidden_states is not None
                else None
            ),
            attentions=(
                [
                    torch.cat([o.attentions[layer_idx] for o in seg_outputs], dim=1)
                    for layer_idx in range(len(seg_outputs[0].attentions))
                ]
                if seg_outputs[0].attentions is not None
                else None
            ),
        )
        if hasattr(seg_outputs[0], "all_c_all") and seg_outputs[0].all_c_all is not None:
            result.all_c_all = [
                torch.cat([o.all_c_all[layer_idx] for o in seg_outputs], dim=1)
                for layer_idx in range(len(seg_outputs[0].all_c_all))
            ]
        if hasattr(seg_outputs[0], "all_h_all") and seg_outputs[0].all_h_all is not None:
            result.all_h_all = [
                torch.cat([o.all_h_all[layer_idx] for o in seg_outputs], dim=1)
                for layer_idx in range(len(seg_outputs[0].all_h_all))
            ]
        return result
