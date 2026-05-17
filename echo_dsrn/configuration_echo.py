from typing import Optional

from transformers import PretrainedConfig


class EchoConfig(PretrainedConfig):
    model_type = "echo"

    def __init__(
        self,
        vocab_size=49152,
        embed_dim=None,
        num_layers=4,
        num_heads=4,
        mlp_ratio=4,
        gate_bias_init=0.0,
        use_hybrid_attention=True,
        use_rmsnorm=True,
        mlp_bias: bool = False,
        # --- Classification fields (optional, ignored by CausalLM) ---
        num_labels: int = 2,
        id2label: Optional[dict] = None,
        label2id: Optional[dict] = None,
        classifier_dropout: float = 0.0,
        **kwargs,
    ):
        # Synchronize hidden_size / embed_dim (HF synonym pair).
        # Priority: explicit embed_dim > explicit hidden_size > package default (768).
        hidden_size = kwargs.pop("hidden_size", None)

        if embed_dim is None and hidden_size is None:
            embed_dim = 768  # package default
        elif embed_dim is None:
            embed_dim = hidden_size
        elif hidden_size is None:
            hidden_size = embed_dim
        elif embed_dim != hidden_size:
            raise ValueError(
                f"embed_dim ({embed_dim}) and hidden_size ({hidden_size}) must be equal in "
                "Echo-DSRN — they are the same architectural dimension. Pass only one."
            )

        hidden_size = embed_dim  # keep them in sync

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.gate_bias_init = gate_bias_init
        self.use_hybrid_attention = use_hybrid_attention
        self.use_rmsnorm = use_rmsnorm
        self.mlp_bias = mlp_bias
        self.classifier_dropout = classifier_dropout

        # Standard HF aliases
        self.num_hidden_layers = num_layers
        self.num_attention_heads = num_heads

        # TGI/HF AutoMap support
        self.auto_map = {
            "AutoConfig": "configuration_echo.EchoConfig",
            "AutoModel": "modeling_echo.EchoModel",
            "AutoModelForCausalLM": "modeling_echo.EchoForCausalLM",
            "AutoModelForSequenceClassification": ("modeling_echo.EchoForSequenceClassification"),
        }

        # vLLM Advanced Parallelism Plans
        self.base_model_tp_plan = {
            "model.embedding": "rowwise",
            "lm_head": "colwise",
            "model.blocks.*.attn.qkv_proj": "colwise",
            "model.blocks.*.attn.out_proj": "rowwise",
            "model.blocks.*.mlp_up": "colwise",
            "model.blocks.*.mlp_down": "rowwise",
            "model.blocks.*.linear_gate": "colwise",
            "model.blocks.*.linear_memory": "colwise",
            "model.blocks.*.linear_read": "rowwise",
        }

        self.base_model_pp_plan = {
            "blocks": (["x", "state_prev"], ["x", "h_new_full"])  # Inputs  # Outputs
        }

        # PretrainedConfig manages id2label / label2id / num_labels as
        # properties internally. Pass them through super().__init__ so HF's
        # property setters run in the correct order. We must NOT pop them here.
        if id2label is not None:
            kwargs["id2label"] = {int(k): v for k, v in id2label.items()}
            kwargs["label2id"] = {v: int(k) for k, v in id2label.items()}
        elif "id2label" not in kwargs:
            # Inject defaults so the property chain initialises cleanly
            default_id2label = {i: str(i) for i in range(num_labels)}
            kwargs["id2label"] = default_id2label
            kwargs["label2id"] = {v: k for k, v in default_id2label.items()}

        if label2id is not None and "label2id" not in kwargs:
            kwargs["label2id"] = label2id

        super().__init__(**kwargs)
