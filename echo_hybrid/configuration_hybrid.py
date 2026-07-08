"""
echo_hybrid/configuration_hybrid.py
─────────────────────────────────────────────────────────────────────────────
HybridEchoConfig: extends Qwen2Config with DSRN memory-injector parameters.

Design rationale
────────────────
Rather than inventing an entirely new config, we subclass Qwen2Config so that
every existing Qwen2 hyper-parameter (hidden_size, num_hidden_layers, etc.) is
available without duplication.  The only additions are the four DSRN-specific
fields documented below.

CRITICAL NOTES (from AGENTS.md)
─────────────────────────────────
• model_type MUST be "echo_hybrid" so AutoConfig routing works after
  AutoConfig.register("echo_hybrid", HybridEchoConfig).
• Do NOT use this config with EchoForCausalLM — that model expects EchoConfig.
"""

from transformers import Qwen2Config, Qwen3Config


class HybridEchoConfig(Qwen2Config):
    """
    Qwen2Config subclass that adds DSRN memory-injector fields.

    New fields
    ──────────
    dsrn_state_dim : int
        Dimension of the c_t slow-state vector maintained by each
        DSRNMemoryInjector.  Defaults to 512.  Can be set equal to
        hidden_size (896 for Qwen2-0.5B) for a richer slow-state, at the
        cost of extra parameters per injector.

    dsrn_injection_stride : int
        Insert one DSRNMemoryInjector after every N transformer layers.
        For Qwen2-0.5B (24 layers) the default of 4 yields 6 injectors.

    dsrn_use_triton : bool
        Route the parallel scan to the custom Triton kernel defined in
        echo_hf/triton_scan.py.  Disabled by default because the Triton
        kernel targets CUDA/ROCm and is not available everywhere.

    gate_bias_init : float
        Initial value of linear_gate.bias in every injector.  A positive
        value (~1.0) keeps memory gates open at init, allowing gradients to
        flow into c_t immediately.  Increase to 2.0 if c_t norms do not
        grow beyond ~0.0 after Phase-1 warm-up.

    use_kv_cache : bool
        Controls the Qwen2 backbone KV-cache.  Independent of use_cache
        (DSRN state return).
        - True  (default / recommended): Standard Hybrid mode — mode 2.
          Backbone KV-cache active; attention handles fast-state, DSRN handles
          slow-state.  Best quality and lowest peak VRAM.
        - False: Ablation / stateless mode — mode 1.
          Backbone KV-cache disabled; every forward re-feeds the full growing
          context so attention stays coherent.  DSRN slow-state is the sole
          cross-step memory.  Useful for ablation studies and "Attention Tax"
          vs "Recurrent Gain" benchmarks.
    """

    model_type = "echo_hybrid"

    def __init__(
        self,
        dsrn_state_dim: int = 512,
        dsrn_injection_stride: int = 4,
        dsrn_use_triton: bool = False,
        gate_bias_init: float = 1.0,
        use_kv_cache: bool = True,  # Kill-switch: False = DSRN-only ablation
        output_surprise_gate_logits: bool = False,
        surprise_temperature_alpha: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dsrn_state_dim = dsrn_state_dim
        self.dsrn_injection_stride = dsrn_injection_stride
        self.dsrn_use_triton = dsrn_use_triton
        self.gate_bias_init = gate_bias_init
        self.use_kv_cache = use_kv_cache
        self.output_surprise_gate_logits = output_surprise_gate_logits
        self.surprise_temperature_alpha = surprise_temperature_alpha
        self.auto_map = {
            "AutoConfig": "configuration_hybrid.HybridEchoConfig",
            "AutoModel": "modeling_hybrid.HybridEchoModel",
            "AutoModelForCausalLM": "modeling_hybrid.HybridEchoForCausalLM",
        }


class Qwen3HybridEchoConfig(Qwen3Config):
    """
    Qwen3Config subclass that adds DSRN memory-injector fields.

    Identical to HybridEchoConfig but inherits from Qwen3Config instead of
    Qwen2Config, supporting the Qwen3 model family (e.g. Qwen/Qwen3-0.6B).

    New fields
    ----------
    dsrn_state_dim : int
        Dimension of the c_t slow-state vector maintained by each
        DSRNMemoryInjector.  Defaults to 512.

    dsrn_injection_stride : int
        Insert one DSRNMemoryInjector after every N transformer layers.
        For Qwen3-0.6B (28 layers) the default of 4 yields 7 injectors.

    dsrn_use_triton : bool
        Route the parallel scan to the custom Triton kernel.

    gate_bias_init : float
        Initial value of linear_gate.bias in every injector.

    use_kv_cache : bool
        Controls the backbone KV-cache.  Independent of use_cache (DSRN state return).
    """

    model_type = "qwen3_echo_hybrid"

    def __init__(
        self,
        dsrn_state_dim: int = 512,
        dsrn_injection_stride: int = 4,
        dsrn_use_triton: bool = False,
        gate_bias_init: float = 1.0,
        use_kv_cache: bool = True,
        output_surprise_gate_logits: bool = False,
        surprise_temperature_alpha: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dsrn_state_dim = dsrn_state_dim
        self.dsrn_injection_stride = dsrn_injection_stride
        self.dsrn_use_triton = dsrn_use_triton
        self.gate_bias_init = gate_bias_init
        self.use_kv_cache = use_kv_cache
        self.output_surprise_gate_logits = output_surprise_gate_logits
        self.surprise_temperature_alpha = surprise_temperature_alpha
        self.auto_map = {
            "AutoConfig": "configuration_hybrid.Qwen3HybridEchoConfig",
            "AutoModel": "modeling_hybrid.Qwen3HybridEchoModel",
            "AutoModelForCausalLM": "modeling_hybrid.Qwen3HybridEchoForCausalLM",
        }
