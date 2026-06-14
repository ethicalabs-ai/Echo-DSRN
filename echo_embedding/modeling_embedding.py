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
    Extracts the recurrent state 'c' from the final layer and shapes it
    for sentence-transformers compatibility.
    """

    def __init__(self, config: EchoConfig):
        super().__init__(config)
        self.model = EchoModel(config)

        # Optional projection layer to map state_dim (hidden_size * num_heads)
        # back to a specific target embedding dimension.
        self.project_embeddings = getattr(config, "project_embeddings", False)
        if self.project_embeddings:
            target_dim = getattr(config, "embedding_dim", config.hidden_size)
            self.projection = nn.Linear(
                config.hidden_size * config.num_heads, target_dim, bias=False
            )
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

        # 1. Base model forward pass
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )

        # Determine sequence length for broadcasting
        if input_ids is not None:
            seq_len = input_ids.shape[1]
        elif inputs_embeds is not None:
            seq_len = inputs_embeds.shape[1]
        else:
            seq_len = 1

        # 2. Extract final recurrent state 'c' from the last layer
        past = outputs.past_key_values
        if hasattr(past, "__getitem__"):
            last_layer_state = past[-1]
        elif hasattr(past, "states"):  # EchoCache support
            last_layer_state = past.states[-1]
        else:
            raise ValueError("Could not extract recurrent state from model cache.")

        # index 1 is c (the recurrent slow state)
        c_state = last_layer_state[1]  # shape: (Batch, State_Dim)

        # 3. Apply optional projection
        if self.projection is not None:
            embeddings = self.projection(c_state)
        else:
            embeddings = c_state

        # 4. Broadcast to shape (Batch, Seq_Len, Dim) for pooling safety
        # Replicates c_T across all tokens; averages/CLS collapse back to c_T.
        embeddings_3d = embeddings.unsqueeze(1).expand(-1, seq_len, -1)

        if not return_dict:
            return (embeddings_3d, outputs.past_key_values)

        return BaseModelOutputWithPast(
            last_hidden_state=embeddings_3d,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
