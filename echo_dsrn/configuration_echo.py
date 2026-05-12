from transformers import PretrainedConfig


class EchoConfig(PretrainedConfig):
    model_type = "echo"

    def __init__(
        self,
        vocab_size=49152,
        embed_dim=768,
        num_layers=4,
        num_heads=4,
        mlp_ratio=4,
        gate_bias_init=0.0,
        use_hybrid_attention=True,
        use_rmsnorm=True,
        **kwargs,
    ):
        # Synchronize hidden_size and embed_dim
        hidden_size = kwargs.pop("hidden_size", embed_dim)
        if embed_dim != hidden_size:
            # Prefer larger if both are non-standard
            major_dim = max(embed_dim, hidden_size)
            embed_dim = hidden_size = major_dim

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.gate_bias_init = gate_bias_init
        self.use_hybrid_attention = use_hybrid_attention
        self.use_rmsnorm = use_rmsnorm

        # Standard HF aliases
        self.num_hidden_layers = num_layers
        self.num_attention_heads = num_heads

        # TGI/HF AutoMap support
        self.auto_map = {
            "AutoConfig": "configuration_echo.EchoConfig",
            "AutoModel": "modeling_echo.EchoModel",
            "AutoModelForCausalLM": "modeling_echo.EchoForCausalLM",
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

        super().__init__(**kwargs)
