from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast

try:
    # pyrefly: ignore [missing-import]
    from .configuration_echo import EchoConfig

    # pyrefly: ignore [missing-import]
    from .modeling_echo import EchoModel, EchoPreTrainedModel
except ImportError:
    from echo_dsrn.configuration_echo import EchoConfig
    from echo_dsrn.modeling_echo import EchoModel, EchoPreTrainedModel


class EchoModelForSentenceEmbedding(EchoPreTrainedModel):
    """
    Sentence embedding adapter for Echo-DSRN.
    Extracts the recurrent state 'c' or sequences from layers and shapes them
    for sentence-transformers compatibility.
    """

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
        output_all_states = pooling_mode in ["mean_c_all", "hybrid"]

        # 1. Base model forward pass
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
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

        def mean_pooling(token_embeddings, mask):
            input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            return sum_embeddings / sum_mask

        # 2. Extract and pool representations according to pooling_mode
        if pooling_mode == "mean_c_all":
            # Extract full sequence of recurrent slow states c_all from last layer
            c_all = outputs.all_c_all[-1]  # shape: (Batch, Seq_Len, State_Dim)
            if attention_mask is not None:
                pooled = mean_pooling(c_all, attention_mask)
            else:
                pooled = c_all.mean(dim=1)
        elif pooling_mode == "mean_x_out":
            # Mean pool the final hidden state
            last_hidden_state = outputs.last_hidden_state  # shape: (Batch, Seq_Len, hidden_size)
            if attention_mask is not None:
                pooled = mean_pooling(last_hidden_state, attention_mask)
            else:
                pooled = last_hidden_state.mean(dim=1)
        elif pooling_mode == "hybrid":
            # Concatenate pooled fast states (h_all) and slow states (c_all) from last layer
            h_all = outputs.all_h_all[-1]  # shape: (Batch, Seq_Len, hidden_size)
            c_all = outputs.all_c_all[-1]  # shape: (Batch, Seq_Len, State_Dim)
            if attention_mask is not None:
                pooled_h = mean_pooling(h_all, attention_mask)
                pooled_c = mean_pooling(c_all, attention_mask)
            else:
                pooled_h = h_all.mean(dim=1)
                pooled_c = c_all.mean(dim=1)
            pooled = torch.cat(
                [pooled_h, pooled_c], dim=-1
            )  # shape: (Batch, hidden_size + State_Dim)
        else:  # "c_T" (default baseline behavior)
            past = outputs.past_key_values
            if hasattr(past, "__getitem__"):
                last_layer_state = past[-1]
            elif hasattr(past, "states"):  # EchoCache support
                last_layer_state = past.states[-1]
            else:
                raise ValueError("Could not extract recurrent state from model cache.")
            pooled = last_layer_state[1]  # shape: (Batch, State_Dim)

        # 3. Apply optional projection
        if self.projection is not None:
            embeddings = self.projection(pooled)
        else:
            embeddings = pooled

        # 4. Broadcast to shape (Batch, Seq_Len, Dim) for pooling safety
        embeddings_3d = embeddings.unsqueeze(1).expand(-1, seq_len, -1)

        if not return_dict:
            return (embeddings_3d, outputs.past_key_values)

        return BaseModelOutputWithPast(
            last_hidden_state=embeddings_3d,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
