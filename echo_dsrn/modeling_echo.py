from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)

from .configuration_echo import EchoConfig

if TYPE_CHECKING:
    # Force HF trust_remote_code AST parser to bundle triton_scan.py
    pass

try:
    # pyrefly: ignore [missing-import]
    from vllm.model_executor.models.transformers import ALL_ATTENTION_FUNCTIONS
except ImportError:
    ALL_ATTENTION_FUNCTIONS = {}

try:
    from transformers.cache_utils import Cache
except ImportError:

    class Cache:
        pass


class EchoCache(Cache):
    """
    Custom Cache to prevent Hugging Face's DynamicCache from dropping
    the (k_attn, v_attn) elements from the DSRN 4-tuple state.
    """

    def __init__(self, states=None):
        self.states = states if states is not None else []
        self.layers = self.states  # HF expectation

    @property
    def is_compileable(self):
        return False

    def get_seq_length(self, layer_idx=0):
        if not self.states or len(self.states) <= layer_idx:
            return 0
        state = self.states[layer_idx]
        if len(state) == 4:
            return state[2].shape[2]
        return 0

    def get_max_length(self):
        return None

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # EchoModel handles its own cache updates internally within the blocks.
        # This update method is just a shim to satisfy the Cache protocol.
        # k, v are already updated in the state tuple returned by the block.
        if len(self.states) > layer_idx:
            state = self.states[layer_idx]
            if len(state) == 4:
                return state[2], state[3]
        return key_states, value_states

    def get_usable_length(self, new_seq_length, layer_idx=0):
        return self.get_seq_length(layer_idx)

    def __getitem__(self, idx):
        return self.states[idx]

    def __len__(self):
        return len(self.states)

    def __iter__(self):
        return iter(self.states)

    def reorder_cache(self, beam_idx: torch.LongTensor):
        reordered_states = []
        for layer_state in self.states:
            reordered_layer_state = tuple(
                tensor.index_select(0, beam_idx.to(tensor.device)) for tensor in layer_state
            )
            reordered_states.append(reordered_layer_state)
        self.states = reordered_states


# --- STANDALONE KERNELS (AUTOMAGICALLY INLINED) ---
def _sequential_scan(a, b, h):
    """
    Core sequential scan for a batch of sequences.
    Vectorized across all dimensions except time.
    """
    a.shape[:-1]
    a.shape[-1]
    # a, b: (..., T, D)
    # h: (..., D)
    T = a.shape[-2]

    res = torch.empty_like(b)
    curr_h = h
    for t in range(T):
        curr_h = a[..., t, :] * curr_h + b[..., t, :]
        res[..., t, :] = curr_h
    return res, curr_h


def dsrn_parallel_scan(g_t, m_t, c_0=None, chunk_size=32, use_triton=False):
    """
    Parallel implementation of the DSRN slow-state update:
    c_t = (1 - g_t) * c_{t-1} + g_t * m_t

    Uses a Hierarchical Chunked Scan for O(T/K + K) speed and stability,
    or a custom Triton kernel for dramatically reduced memory bandwidth.
    """
    # Triton kernel for GPU-accelerated parallel scan.
    if use_triton and g_t.is_cuda:
        try:
            from .triton_scan import triton_dsrn_parallel_scan

            return triton_dsrn_parallel_scan(g_t, m_t, c_0)
        except ImportError:
            import warnings

            warnings.warn("Triton scan unavailable. Falling back to PyTorch scan.", UserWarning)

    orig_dtype = g_t.dtype
    a = (1.0 - g_t).float()
    b = (g_t * m_t).float()

    B, T, D = a.shape
    device = a.device

    # Pad T to be multiple of chunk_size
    pad_len = (chunk_size - (T % chunk_size)) % chunk_size
    if pad_len > 0:
        a = F.pad(a, (0, 0, 0, pad_len), value=1.0)
        b = F.pad(b, (0, 0, 0, pad_len), value=0.0)

    new_T = T + pad_len
    num_chunks = new_T // chunk_size

    # 1. Reshape to (B, num_chunks, chunk_size, D)
    a_chunks = a.view(B, num_chunks, chunk_size, D)
    b_chunks = b.view(B, num_chunks, chunk_size, D)

    # 2. Local scan within each chunk (vectorized across B and num_chunks)
    h_init_local = torch.zeros(B, num_chunks, D, device=device, dtype=torch.float32)
    c_res, c_final = _sequential_scan(a_chunks, b_chunks, h_init_local)

    # Summary of a for each chunk (product of a)
    a_final = torch.prod(a_chunks, dim=2)  # (B, num_chunks, D)

    # 3. Global scan across chunk summaries
    h_0 = c_0.float() if c_0 is not None else torch.zeros(B, D, device=device, dtype=torch.float32)

    # h_chunk_outputs[:, j] is the state AFTER chunk j.
    h_chunk_outputs, _ = _sequential_scan(a_final, c_final, h_0)
    # The state BEFORE chunk j is h_chunk_outputs[:, j-1].
    h_starts = torch.cat([h_0.unsqueeze(1), h_chunk_outputs[:, :-1]], dim=1)

    # 4. Final combine: h_{j, i} = a_prefix_{j, i} * h_starts[j] + c_res[j, i]
    a_prefix = torch.cumprod(a_chunks, dim=2)
    final_h = a_prefix * h_starts.unsqueeze(2) + c_res

    # Reshape back and crop, then cast back to original dtype
    return final_h.view(B, -1, D)[:, :T].to(orig_dtype)


def rms_norm_fn(hidden_states, weight, eps=1e-6):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.contiguous().to(torch.float32)
    variance = (hidden_states * hidden_states).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return weight * hidden_states.to(input_dtype)


def dsrn_parallel_kernel_legacy(
    model_block: nn.Module,
    x: torch.Tensor,
    h_prev: torch.Tensor,
    c_prev: torch.Tensor,
    eos_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Legacy DSRN kernel (Fixed LayerNorm, No Surprise Read).
    Identical to the version that passed verification.
    """
    B, T, D = x.shape

    # 1. Norm and Projections
    x_norm = F.layer_norm(
        x,
        (D,),
        weight=model_block.norm_fast.weight,
        bias=model_block.norm_fast.bias,
    )

    # Fast State — float32 sigmoid/tanh to avoid bf16 saturation NaN
    gru_proj = F.linear(x_norm, model_block.gru_cell.weight_ih, model_block.gru_cell.bias_ih)
    z_all = torch.sigmoid(gru_proj[:, :, :D].float()).to(x.dtype)
    r_all = torch.tanh(gru_proj[:, :, 2 * D :].float()).to(
        x.dtype
    )  # Optimization: slice instead of chunk

    # --- EOS RESET LOGIC (Fast State) ---
    if eos_mask is not None:
        reset_mask = torch.roll(eos_mask, shifts=1, dims=1)
        reset_mask[:, 0] = (
            0  # First token reset depends on previous chunk eos, handled by h_prev/c_prev passing 0
        )

        # Apply strict reset to z_all
        z_all = torch.where(reset_mask.unsqueeze(-1) > 0, torch.ones_like(z_all), z_all)

    # h_t = (1 - z_t) * h_{t-1} + z_t * r_t
    h_all = dsrn_parallel_scan(
        z_all, r_all, h_prev, use_triton=getattr(model_block, "use_triton", False)
    )
    h_new = h_all[:, -1]

    # 2. Slow State Path
    # CAUSAL SHIFT: Predict x[t] using h[t-1]
    # h_all is [h_1, ..., h_T]. We need [h_0, ..., h_{T-1}]
    # Prepend h_prev to shift
    h_shifted = torch.cat([h_prev.unsqueeze(1), h_all[:, :-1, :]], dim=1)

    x_pred = model_block.linear_pred(h_shifted)
    diff = x - x_pred
    error = torch.clamp((diff * diff).float(), max=10.0).to(x.dtype).mean(dim=-1, keepdim=True)
    # Constrain surprise_lambda strictly positive to guarantee error opens the memory gate
    surprise_signal = error * torch.nn.functional.softplus(model_block.surprise_lambda.float()).to(
        x.dtype
    )

    # Gates
    gate_logits = model_block.linear_gate(h_all) + surprise_signal
    g_all = torch.sigmoid(gate_logits.float()).to(x.dtype)
    m_all = torch.tanh(model_block.linear_memory(h_all).float()).to(x.dtype)

    # --- EOS RESET LOGIC (Slow State) ---
    if eos_mask is not None:
        reset_mask = torch.roll(eos_mask, shifts=1, dims=1)
        reset_mask[:, 0] = 0

        g_all = torch.where(reset_mask.unsqueeze(-1) > 0, torch.zeros_like(g_all), g_all)

    # c_t
    c_all = dsrn_parallel_scan(
        g_all, m_all, c_prev, use_triton=getattr(model_block, "use_triton", False)
    )
    c_new = c_all[:, -1]

    # --- Inter-Chunk Reset ---
    # If the LAST token is EOS, then h_new/c_new (which are states FOR NEXT CHUNK) must be 0.
    if eos_mask is not None:
        last_is_eos = eos_mask[:, -1].float()  # (B,)
        keep_prob = (1.0 - last_is_eos).unsqueeze(-1)  # (B, 1)
        h_new = h_new * keep_prob
        c_new = c_new * keep_prob
    gate_stats = g_all.mean(dim=-1)

    # 3. Final MLP Path
    h_norm = F.layer_norm(
        h_all, (D,), weight=model_block.norm_ff.weight, bias=model_block.norm_ff.bias
    )
    mlp_out = model_block.mlp_down(model_block.mlp_act(model_block.mlp_up(h_norm)))

    x_out = x + mlp_out

    # Continuous Read (Surprise Gate Fix)
    # Enabled on Legacy to fix Disconnected Slow State bug while keeping LayerNorm
    x_out = x_out + model_block.linear_read(c_all)

    return x_out, h_new, c_new, gate_stats, h_all, c_all


def dsrn_parallel_kernel_hybrid(
    model_block: nn.Module,
    x: torch.Tensor,
    h_prev: torch.Tensor,
    c_prev: torch.Tensor,
    eos_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Hybrid DSRN kernel (RMSNorm + Surprise Read).
    """
    B, T, D = x.shape

    # 1. Norm (RMSNorm hardcoded for Hybrid path)
    x_norm = rms_norm_fn(x, model_block.norm_fast.weight)

    # Fast State — compute sigmoid/tanh in float32 to avoid bf16 saturation
    # producing 0 × inf = NaN in the backward pass.
    gru_proj = F.linear(x_norm, model_block.gru_cell.weight_ih, model_block.gru_cell.bias_ih)
    z_all = torch.sigmoid(gru_proj[:, :, :D].float()).to(x.dtype)
    r_all = torch.tanh(gru_proj[:, :, 2 * D :].float()).to(x.dtype)

    # --- EOS RESET LOGIC (Fast State) ---
    if eos_mask is not None:
        reset_mask = torch.roll(eos_mask, shifts=1, dims=1)
        reset_mask[:, 0] = 0
        z_all = torch.where(reset_mask.unsqueeze(-1) > 0, torch.ones_like(z_all), z_all)

    h_all = dsrn_parallel_scan(
        z_all, r_all, h_prev, use_triton=getattr(model_block, "use_triton", False)
    )
    h_new = h_all[:, -1]

    # 2. Slow State
    # CAUSAL SHIFT: Predict x[t] using h[t-1]
    h_shifted = torch.cat([h_prev.unsqueeze(1), h_all[:, :-1, :]], dim=1)

    x_pred = model_block.linear_pred(h_shifted)
    diff = x - x_pred
    error = torch.clamp((diff * diff).float(), max=10.0).to(x.dtype).mean(dim=-1, keepdim=True)
    # Constrain surprise_lambda strictly positive to guarantee error opens the memory gate
    surprise_signal = error * torch.nn.functional.softplus(model_block.surprise_lambda.float()).to(
        x.dtype
    )

    gate_logits = model_block.linear_gate(h_all) + surprise_signal
    g_all = torch.sigmoid(gate_logits.float()).to(x.dtype)
    m_all = torch.tanh(model_block.linear_memory(h_all).float()).to(x.dtype)

    # --- EOS RESET LOGIC (Slow State) ---
    if eos_mask is not None:
        reset_mask = torch.roll(eos_mask, shifts=1, dims=1)
        reset_mask[:, 0] = 0
        g_all = torch.where(reset_mask.unsqueeze(-1) > 0, torch.zeros_like(g_all), g_all)

    c_all = dsrn_parallel_scan(
        g_all, m_all, c_prev, use_triton=getattr(model_block, "use_triton", False)
    )
    c_new = c_all[:, -1]

    # --- Inter-Chunk Reset ---
    if eos_mask is not None:
        last_is_eos = eos_mask[:, -1].float()
        keep_prob = (1.0 - last_is_eos).unsqueeze(-1)
        h_new = h_new * keep_prob
        c_new = c_new * keep_prob
    gate_stats = g_all.mean(dim=-1)

    # 3. Final MLP
    h_norm = rms_norm_fn(h_all, model_block.norm_ff.weight)
    mlp_out = model_block.mlp_down(model_block.mlp_act(model_block.mlp_up(h_norm)))
    x_out = x + mlp_out

    # Continuous Read (Hybrid Feature)
    if model_block.use_hybrid_attention:
        x_out = x_out + model_block.linear_read(c_all)

    return x_out, h_new, c_new, gate_stats, h_all, c_all


def dsrn_parallel_kernel(
    model_block: nn.Module,
    x: torch.Tensor,
    h_prev: torch.Tensor,
    c_prev: torch.Tensor,
    eos_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Wrapper for backward compatibility. Dispatches based on config.
    """
    if getattr(model_block, "use_rmsnorm", False):
        return dsrn_parallel_kernel_hybrid(model_block, x, h_prev, c_prev, eos_mask=eos_mask)
    return dsrn_parallel_kernel_legacy(model_block, x, h_prev, c_prev, eos_mask=eos_mask)


class HymbaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        HymbaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class EchoRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=4096, base=10000.0, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.device = device

        # We NO LONGER use buffers here because they are being corrupted by
        # Hugging Face's weight loading mechanism for this specific model.
        # We will compute and move them on the first forward pass.
        self._cos_cached = None
        self._sin_cached = None

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        # Compute inv_freq locally
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim)
        )
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self._cos_cached = emb.cos().to(dtype)
        self._sin_cached = emb.sin().to(dtype)

    def forward(self, x, seq_len=None):
        if (
            self._cos_cached is None
            or seq_len > self.max_seq_len_cached
            or self._cos_cached.device != x.device
        ):
            self._set_cos_sin_cache(
                seq_len=max(seq_len, self.max_position_embeddings), device=x.device, dtype=x.dtype
            )

        return (
            self._cos_cached[:seq_len].to(dtype=x.dtype),
            self._sin_cached[:seq_len].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)  # (B, 1, T, D)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)  # (B, 1, T, D)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SlidingWindowAttention(nn.Module):
    def __init__(self, config: EchoConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.window_size = getattr(config, "window_size", 128)
        self.attention_masking = getattr(config, "attention_masking", "causal")

        self.qkv_proj = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=False)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.rotary_emb = EchoRotaryEmbedding(
            self.head_dim,
            base=getattr(config, "rope_theta", 10000.0),
        )

    def forward(
        self,
        x,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # --- RoPE Injection ---
        if position_ids is None:
            # Fallback if position_ids was not passed
            seq_length_with_past = T
            if past_key_values is not None:
                seq_length_with_past += past_key_values[0].shape[2]
            position_ids = (
                torch.arange(
                    seq_length_with_past - T,
                    seq_length_with_past,
                    dtype=torch.long,
                    device=x.device,
                )
                .unsqueeze(0)
                .view(-1, T)
            )

        kv_seq_len = k.shape[2]
        if past_key_values is not None:
            kv_seq_len += past_key_values[0].shape[2]

        cos, sin = self.rotary_emb(v, seq_len=kv_seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        # ----------------------

        if past_key_values is not None:
            k_past, v_past = past_key_values
            k = torch.cat([k_past, k], dim=2)
            v = torch.cat([v_past, v], dim=2)

        # The cache MUST store the full history, do not overwrite it with truncated slices
        current_key_value = (k, v)

        # Create slices for attention computation
        k_attn = k
        v_attn = v

        # Enforce Sliding Window (Truncate oldest tokens for attention ONLY)
        if self.window_size is not None and k_attn.shape[2] > self.window_size:
            k_attn = k_attn[:, :, -self.window_size :, :]
            v_attn = v_attn[:, :, -self.window_size :, :]

        attn_fn = ALL_ATTENTION_FUNCTIONS.get(
            kwargs.get("attn_implementation", "sdpa"), F.scaled_dot_product_attention
        )

        # Determining causality and windowing:
        # 1. Training (T > 1): Use sliding window causal mask.
        # 2. Decoding (T = 1): Use sliding window and NO CAUSAL MASK
        if T > 1:
            # Training/Prefill: Attend to full k, v but apply band-limited causal mask
            # Build sliding window causal mask (T, kv_seq_len)
            kv_all_seq_len = k.shape[2]
            past_seq_len = kv_all_seq_len - T

            mask = torch.zeros((T, kv_all_seq_len), device=x.device, dtype=x.dtype)

            row_idx = torch.arange(T, device=x.device).view(-1, 1)
            col_idx = torch.arange(kv_all_seq_len, device=x.device).view(1, -1)
            abs_pos = row_idx + past_seq_len

            if self.attention_masking == "non_causal_window":
                w_half = self.window_size // 2 if self.window_size is not None else None
                if w_half is not None:
                    # Keep tokens in range [abs_pos - w_half, abs_pos + w_half]
                    mask = torch.where(torch.abs(abs_pos - col_idx) > w_half, float("-inf"), mask)
            else:
                # Causal upper triangle = -inf
                mask = torch.where(col_idx > abs_pos, float("-inf"), mask)

                # Keep tokens in range [abs_pos - self.window_size, abs_pos]
                if self.window_size is not None:
                    mask = torch.where((abs_pos - col_idx) >= self.window_size, float("-inf"), mask)

            # Replace -inf with 0 for the permitted window (float mask expected by sdpa)
            mask = torch.where(mask == float("-inf"), mask, torch.zeros_like(mask))

            y = attn_fn(q, k, v, attn_mask=mask.unsqueeze(0).unsqueeze(0))
        else:
            # Decoding: Recurrent step, attend only to the last window_size tokens
            y = attn_fn(q, k_attn, v_attn, is_causal=False)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y), current_key_value


class DSRNBlock(nn.Module):
    def __init__(self, config: EchoConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.state_size = config.hidden_size * config.num_heads
        self.use_triton = getattr(config, "use_triton", True)
        self.use_hybrid_attention = getattr(config, "use_hybrid_attention", True)
        self.use_rmsnorm = getattr(config, "use_rmsnorm", True)

        # Fast State (GRU)
        if self.use_rmsnorm:
            self.norm_fast = HymbaRMSNorm(config.hidden_size)
        else:
            self.norm_fast = nn.LayerNorm(config.hidden_size)

        self.gru_cell = nn.GRUCell(config.hidden_size, config.hidden_size)

        # Hybrid Attention
        if self.use_hybrid_attention:
            self.attn = SlidingWindowAttention(config)

        # Slow State (DSRN)
        self.linear_read = nn.Linear(self.state_size, config.hidden_size, bias=False)
        self.linear_gate = nn.Linear(config.hidden_size, self.state_size)
        self.linear_memory = nn.Linear(config.hidden_size, self.state_size)

        # -- Surprise Mechanism --
        self.linear_pred = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.surprise_lambda = nn.Parameter(torch.zeros(self.state_size))

        # Feed-Forward
        if self.use_rmsnorm:
            self.norm_ff = HymbaRMSNorm(config.hidden_size)
        else:
            self.norm_ff = nn.LayerNorm(config.hidden_size)

        # Simple MLP: Linear -> GELU -> Linear
        # mlp_up / mlp_act / mlp_down are the ONLY registered submodules.
        # No self.mlp alias — that caused double-registration and spurious "missing keys".
        intermediate_size = getattr(
            config, "intermediate_size", int(config.hidden_size * getattr(config, "mlp_ratio", 4.0))
        )
        # Use getattr guard so configs loaded from old JSON (pre-mlp_bias field) default safely.
        _mlp_bias = getattr(config, "mlp_bias", False)
        self.mlp_up = nn.Linear(config.hidden_size, intermediate_size, bias=_mlp_bias)
        self.mlp_act = nn.GELU()
        self.mlp_down = nn.Linear(intermediate_size, config.hidden_size, bias=_mlp_bias)

    def forward(self, x: torch.Tensor, state_prev: Tuple[torch.Tensor, ...], **kwargs) -> Tuple:

        # Unpack state
        # Supports (h, c) or (h, c, k_attn, v_attn)
        h_prev = state_prev[0]
        c_prev = state_prev[1]

        if self.use_triton and x.is_cuda:
            # Placeholder for Triton
            pass

        # Use Parallel Kernel
        x_out, h_new, c_new, gate_stats, h_all, c_all = dsrn_parallel_kernel(
            self, x, h_prev, c_prev
        )

        if self.use_hybrid_attention:
            # Re-apply norm for attention branch (cleanest for surgical transplant)
            x_norm = self.norm_fast(x)

            # Extract attention state from tuple if present (h, c, k_attn, v_attn)
            # HF state structure is now: (h, c, k_attn, v_attn)
            # But wait, past_key_values in forward loop is just (h,c) from legacy code.
            # We need to expand the state tuple to include attention KV.

            attn_kv = None
            if len(state_prev) == 4:
                attn_kv = (state_prev[2], state_prev[3])

            attn_out, new_attn_kv = self.attn(x_norm, past_key_values=attn_kv, **kwargs)
            x_out = x_out + attn_out

            # Update state with new KV
            if new_attn_kv is not None:
                h_new_full = (h_new, c_new, new_attn_kv[0], new_attn_kv[1])
            else:
                h_new_full = (h_new, c_new)
        else:
            h_new_full = (h_new, c_new)

        if kwargs.get("output_all_states", False):
            return x_out, h_new_full, gate_stats, h_all, c_all
        return x_out, h_new_full, gate_stats


class EchoPreTrainedModel(PreTrainedModel):
    config_class = EchoConfig
    base_model_prefix = "model"
    _no_split_modules = ["DSRNBlock"]

    # Silently drop legacy mlp.0.*/mlp.1.*/mlp.2.* alias keys if they exist in old
    # local training checkpoints from before the self.mlp aliasing was removed.
    # The canonical names are mlp_up.* / mlp_act.* / mlp_down.* which load fine.
    _keys_to_ignore_on_load_unexpected = [
        r".*\.mlp\.0\..*",
        r".*\.mlp\.1\..*",
        r".*\.mlp\.2\..*",
    ]

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)


class EchoModel(EchoPreTrainedModel):
    supports_gradient_checkpointing = True
    _supports_attention_backend = True

    def __init__(self, config: EchoConfig):
        super().__init__(config)
        self.embed_dim = config.embed_dim
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.state_dim = config.embed_dim * config.num_heads

        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.blocks = nn.ModuleList([DSRNBlock(config) for _ in range(config.num_layers)])

        if getattr(config, "use_rmsnorm", False):
            self.final_norm = HymbaRMSNorm(config.hidden_size)
        else:
            self.final_norm = nn.LayerNorm(config.hidden_size)

        self.gradient_checkpointing = False

        self.post_init()

        # --- ZOMBIE GRADIENT PATCH (FIXED) ---
        # Fixed: Now using controlled bias defaults to 1.0 to encourage open gates initially
        bias_val = getattr(config, "gate_bias_init", 1.0)
        for block in self.blocks:
            nn.init.constant_(block.linear_gate.bias, bias_val)
            # Init Surprise
            if (
                block.linear_pred.weight.dtype in (torch.bfloat16, torch.float16)
                and block.linear_pred.weight.is_cuda
            ):
                _device = block.linear_pred.weight.device
                _dtype = block.linear_pred.weight.dtype
                temp_w = torch.empty_like(
                    block.linear_pred.weight, dtype=torch.float32, device="cpu"
                )
                nn.init.orthogonal_(temp_w, gain=0.1)
                with torch.no_grad():
                    block.linear_pred.weight.copy_(temp_w.to(device=_device, dtype=_dtype))
            else:
                nn.init.orthogonal_(block.linear_pred.weight, gain=0.1)

            nn.init.zeros_(block.surprise_lambda)
            # CRITICAL: Zero-Init Residual Output (Identity Start)
            nn.init.zeros_(block.mlp_down.weight)
            if block.mlp_down.bias is not None:
                nn.init.zeros_(block.mlp_down.bias)

    def _set_gradient_checkpointing(self, enable=True, gradient_checkpointing_func=None):
        """Enable/disable gradient checkpointing."""
        self.gradient_checkpointing = enable

    def get_input_embeddings(self):
        return self.embedding

    def set_input_embeddings(self, value):
        self.embedding = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_dsrn_telemetry: Optional[bool] = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_all_states: Optional[bool] = False,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        return_dict = (
            return_dict
            if return_dict is not None
            else getattr(self.config, "use_return_dict", True)
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_len = input_ids.shape
            x = self.embedding(input_ids)
        elif inputs_embeds is not None:
            batch_size, seq_len, _ = inputs_embeds.shape
            x = inputs_embeds
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = x.device

        # Initialize states if not provided or if it's an empty Cache object
        is_empty_cache = (
            hasattr(past_key_values, "get_seq_length") and past_key_values.get_seq_length() == 0
        )
        if past_key_values is None or is_empty_cache:
            past_key_values = []
            for _ in range(self.num_layers):
                h = torch.zeros(batch_size, self.embed_dim, device=device, dtype=x.dtype)
                c = torch.zeros(batch_size, self.state_dim, device=device, dtype=x.dtype)
                past_key_values.append((h, c))

        current_states = past_key_values
        next_states = []

        all_gate_stats = [] if output_dsrn_telemetry else None
        all_c_states = [] if output_dsrn_telemetry else None
        all_h_all = [] if output_all_states else None
        all_c_all = [] if output_all_states else None

        # Layer-Major Execution
        for i, block in enumerate(self.blocks):

            # Handle potential DynamicCache structure or list of tuples
            if hasattr(current_states, "__getitem__"):
                state_i = current_states[i]
            else:
                state_i = current_states[i]

            if len(state_i) == 2:
                # DSRN Only
                pass
            elif len(state_i) == 4:
                # DSRN + Attention State
                pass
            else:
                # Fallback for empty/malformed states
                h_prev = torch.zeros(batch_size, self.embed_dim, device=device)
                c_prev = torch.zeros(batch_size, self.state_dim, device=device)
                state_i = (h_prev, c_prev)

            # Use gradient checkpointing if enabled
            if self.gradient_checkpointing and self.training:
                # Checkpointing complex states is tricky, usually just pass h/c
                out = torch.utils.checkpoint.checkpoint(
                    block,
                    x,
                    state_i,
                    use_reentrant=False,
                    output_all_states=output_all_states,
                    **kwargs,
                )
            else:
                out = block(x, state_i, output_all_states=output_all_states, **kwargs)

            x = out[0]
            next_states.append(out[1])

            if output_dsrn_telemetry:
                all_gate_stats.append(out[2])
                all_c_states.append(out[1][1])

            if output_all_states:
                all_h_all.append(out[3])
                all_c_all.append(out[4])

        x = self.final_norm(x)

        if isinstance(current_states, EchoCache):
            current_states.states = next_states
            next_states = current_states
        elif EchoCache is not None:
            next_states = EchoCache(next_states)

        # Revert to raw tuple outputs if return_dict=False is requested
        if not return_dict:
            if output_dsrn_telemetry:
                if output_all_states:
                    return x, next_states, all_c_states, all_gate_stats, all_h_all, all_c_all
                return x, next_states, all_c_states, all_gate_stats
            if output_all_states:
                return x, next_states, all_h_all, all_c_all
            return x, next_states

        # Standard HF Object wrapper containing last_hidden_state
        output_obj = BaseModelOutputWithPast(
            last_hidden_state=x,
            past_key_values=next_states,
            hidden_states=(x,) if output_hidden_states else None,
            attentions=None,
        )
        if output_dsrn_telemetry:
            output_obj.all_c_states = all_c_states
            output_obj.all_gate_stats = all_gate_stats
        if output_all_states:
            output_obj.all_h_all = all_h_all
            output_obj.all_c_all = all_c_all
        return output_obj


class EchoForCausalLM(EchoPreTrainedModel, GenerationMixin):
    _is_causal = True
    supports_gradient_checkpointing = True
    _supports_cache_class = False
    _supports_static_cache = False
    main_input_name = "input_ids"
    # Required by the modern HF tie_weights() mechanism (transformers ≥ 4.47).
    # Without this dict being non-None, tie_weights() returns early even when
    # tie_word_embeddings=True and get_input/output_embeddings() are both defined.
    _tied_weights_keys = {"lm_head.weight": "model.embedding.weight"}

    @property
    def _keys_to_ignore_on_load_missing(self):
        # When mlp_bias=False (the default, and the setting for all v0.1.2 checkpoints),
        # bias tensors are not present in the checkpoint and should not trigger warnings.
        # When mlp_bias=True, these keys WILL exist in the checkpoint — do not silence them.
        if not getattr(self.config, "mlp_bias", False):
            return [r"model\.blocks\.\d+\.mlp_(up|down)\.bias"]
        return []

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

        # Defense-in-depth: if mlp_bias=False but bias tensors were somehow initialized
        # (e.g. an old code path created them), zero them out to prevent NaN/Inf
        # corruption when running in bfloat16.
        if not getattr(model.config, "mlp_bias", False):
            zeroed = 0
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if "mlp_up.bias" in name or "mlp_down.bias" in name:
                        param.zero_()
                        zeroed += 1
            if zeroed:
                import warnings

                warnings.warn(
                    f"Zeroed {zeroed} MLP bias tensor(s) that were missing from the "
                    f"checkpoint. This indicates a config/checkpoint mismatch. "
                    f"Ensure mlp_bias=False in EchoConfig for v0.1.2 checkpoints.",
                    UserWarning,
                )

        return model

    def __init__(self, config: EchoConfig):
        super().__init__(config)
        self.model = EchoModel(config)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self._latest_c_states = None
        self._latest_gate_stats = None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embedding

    def set_input_embeddings(self, value):
        self.model.embedding = value

    def _set_gradient_checkpointing(self, enable=True, gradient_checkpointing_func=None):
        """Enable/disable gradient checkpointing."""
        self.model._set_gradient_checkpointing(enable, gradient_checkpointing_func)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_dsrn_telemetry: Optional[bool] = False,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else getattr(self.config, "output_attentions", False)
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else getattr(self.config, "output_hidden_states", False)
        )
        use_cache = use_cache if use_cache is not None else getattr(self.config, "use_cache", True)

        return_dict = (
            return_dict
            if return_dict is not None
            else getattr(self.config, "use_return_dict", True)
        )

        '''
        If kwargs is getting overloaded with extra args HF generate passes,
        we safely extract kwargs here.
        '''
        # Pass position_ids explicitly alongside **kwargs
        kwargs["position_ids"] = position_ids

        # Call the base EchoModel
        model_out = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_dsrn_telemetry=output_dsrn_telemetry,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,  # Pass return_dict explicitly
            **kwargs,
        )

        # Handle BaseModelOutputWithPast or raw tuple output gracefully
        if hasattr(model_out, "last_hidden_state"):
            hidden_states = model_out.last_hidden_state
            new_states = model_out.past_key_values
        else:
            hidden_states = model_out[0]
            new_states = model_out[1]

        # Extract telemetry if model returned raw tuple (or via custom properties)
        if hasattr(model_out, "all_c_states"):
            self._latest_c_states = model_out.all_c_states
            self._latest_gate_stats = model_out.all_gate_stats
        elif isinstance(model_out, tuple) and len(model_out) > 2:
            self._latest_c_states = model_out[2]
            self._latest_gate_stats = model_out[3]

        # Project using Causal LM head
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits, new_states)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=new_states if use_cache else None,
            hidden_states=(hidden_states,) if output_hidden_states else None,
            attentions=None,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, **kwargs
    ):
        # If past_key_values is a DynamicCache, we need to extract the underlying list of tuples
        # if the custom cache hasn't taken over yet. But actually, HF doesn't know about our 4-tuples.
        # So we should just let EchoModel handle it. If HF gave us a DynamicCache, it might be empty
        # or mangled.
        if (
            past_key_values is not None
            and not isinstance(past_key_values, (list, tuple))
            and not isinstance(past_key_values, EchoCache)
        ):
            # It's a DynamicCache. It's likely from the first generation step.
            # We can't use it directly because it stripped our (h,c).
            # But wait, on the VERY first generation step, past_key_values is None, then EchoModel returns EchoCache.
            # On subsequent steps we get EchoCache.
            # So if we get a DynamicCache, it means someone passed past_key_values explicitly to generate(),
            # or HF auto-created it on step 0 and passed it to step 1 incorrectly.
            pass

        # In newer transformers, past_key_values could be a DynamicCache.
        # Check if it's effectively empty.
        is_empty = False
        if past_key_values is None:
            is_empty = True
        elif hasattr(past_key_values, "get_seq_length") and past_key_values.get_seq_length() == 0:
            is_empty = True
        elif isinstance(past_key_values, list) and len(past_key_values) == 0:
            is_empty = True

        # If past_key_values is used, we only need the last token
        if not is_empty:
            input_ids = input_ids[:, -1:]

        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": kwargs.get("use_cache"),
        }

        # Pass through extra kwargs like output_dsrn_telemetry
        model_inputs.update({k: v for k, v in kwargs.items() if k not in model_inputs})

        return model_inputs

    def _reorder_cache(self, past_key_values, beam_idx):
        """
        Reorders cache for beam search or contrastive search.
        past_key_values: List[Tuple(h, c, ...)]
        """
        if past_key_values is None:
            return None

        reordered_past = []
        for layer_past in past_key_values:
            # Each layer_past is a tuple of tensors (h, c) or (h, c, k, v)
            reordered_layer_past = tuple(
                p.index_select(0, beam_idx.to(p.device)) for p in layer_past
            )
            reordered_past.append(reordered_layer_past)
        return reordered_past


class EchoClassifier(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        res = super().forward(input)
        if res.ndim == 3 and res.size(1) == 1:
            res = res.squeeze(1)
        return res


class EchoForSequenceClassification(EchoPreTrainedModel):
    """
    Echo-DSRN with a sequence-level classification head.

    This model is the *terminal* form of a fine-tuned classifier: it exposes
    only a ``classify()`` convenience method and a standard HF ``forward()``
    that returns :class:`~transformers.modeling_outputs.SequenceClassifierOutputWithPast`.
    It intentionally does **not** inherit :class:`~transformers.GenerationMixin` so
    chat-completion endpoints cannot be used accidentally.

    Typical construction path
    -------------------------
    1.  Load ``EchoForCausalLM`` + LoRA adapter via :func:`merge_and_export`
        (see ``scripts/merge_clf_adapter.py``).
    2.  The resulting merged weights are saved as ``EchoForSequenceClassification``
        alongside a ``config.json`` that carries ``num_labels``, ``id2label``, and
        ``label2id``.
    3.  End-users load with::

            from echo_dsrn import EchoForSequenceClassification
            model = EchoForSequenceClassification.from_pretrained("your/hub-id")
            label, probs = model.classify("some text")
    """

    # Do NOT add GenerationMixin — this model must not generate text.
    main_input_name = "input_ids"

    def __init__(self, config: EchoConfig):
        super().__init__(config)
        self.num_labels = getattr(config, "num_labels", 2)
        self.model = EchoModel(config)

        classifier_dropout = getattr(config, "classifier_dropout", 0.0)
        self.dropout = nn.Dropout(classifier_dropout) if classifier_dropout > 0.0 else nn.Identity()
        self.classifier = EchoClassifier(config.embed_dim, self.num_labels, bias=True)

        self.post_init()

    @property
    def score(self) -> EchoClassifier:
        return self.classifier

    @score.setter
    def score(self, value: EchoClassifier):
        self.classifier = value

    # ------------------------------------------------------------------
    # HF embedding hooks (required by PreTrainedModel)
    # ------------------------------------------------------------------
    def get_input_embeddings(self):
        return self.model.embedding

    def set_input_embeddings(self, value):
        self.model.embedding = value

    def _set_gradient_checkpointing(self, enable=True, gradient_checkpointing_func=None):
        self.model._set_gradient_checkpointing(enable, gradient_checkpointing_func)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        """
        Parameters
        ----------
        labels:
            - ``num_labels == 1``: regression target (``torch.float``).
            - ``num_labels > 1``, single integer per sample: cross-entropy class index.
            - ``num_labels > 1``, float vector per sample: multi-label BCE.
        """
        return_dict = (
            return_dict
            if return_dict is not None
            else getattr(self.config, "use_return_dict", True)
        )

        kwargs["position_ids"] = position_ids

        model_out = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        hidden_states = model_out[0]  # (B, T, D)
        new_states = model_out[1]

        # --- Pooling: last non-padding token ---
        if attention_mask is not None:
            # Find the index of the last 1 in each row of attention_mask
            seq_lengths = attention_mask.sum(dim=1) - 1  # (B,)
            seq_lengths = seq_lengths.clamp(min=0)
        else:
            # No mask: use the true last token
            if input_ids is not None:
                seq_lengths = torch.full(
                    (hidden_states.size(0),),
                    hidden_states.size(1) - 1,
                    dtype=torch.long,
                    device=hidden_states.device,
                )
            else:
                seq_lengths = torch.full(
                    (hidden_states.size(0),),
                    hidden_states.size(1) - 1,
                    dtype=torch.long,
                    device=hidden_states.device,
                )

        # Gather last-token hidden states: (B, D)
        pooled = hidden_states[
            torch.arange(hidden_states.size(0), device=hidden_states.device), seq_lengths
        ]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # (B, num_labels)

        # --- Loss ---
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.squeeze(-1), labels.float())
            elif labels.dtype in (torch.float, torch.float16, torch.bfloat16):
                # Multi-label binary classification
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels.float())
            else:
                # Standard multi-class
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits, new_states)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=new_states if use_cache else None,
            hidden_states=None,
            attentions=None,
        )

    # ------------------------------------------------------------------
    # Convenience inference API
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def classify(
        self,
        text: str,
        tokenizer,
        device: Optional[str] = None,
        return_probabilities: bool = True,
    ) -> Tuple[str, Optional[torch.Tensor]]:
        """
        High-level classification helper.

        Parameters
        ----------
        text:
            Raw string to classify.
        tokenizer:
            A HuggingFace ``PreTrainedTokenizer`` compatible with the model.
        device:
            Optional device string (e.g. ``"cuda"``).  Defaults to the device
            of the model's first parameter.
        return_probabilities:
            If ``True`` (default), also return a probability tensor (softmax
            for multi-class, sigmoid for binary/multi-label).

        Returns
        -------
        label : str
            The predicted label string from ``config.id2label``.
        probabilities : Tensor or None
            Shape ``(num_labels,)`` probability vector, or ``None`` if
            ``return_probabilities=False``.
        """
        if device is None:
            try:
                device = str(next(self.parameters()).device)
            except StopIteration:
                device = "cpu"

        self.eval()

        # Format text if baked-in templates exist
        sys_prompt = getattr(self.config, "system_prompt", None)
        usr_template = getattr(self.config, "user_template", None)

        if sys_prompt and usr_template:
            messages = [{"role": "system", "content": sys_prompt}]
            messages.append({"role": "user", "content": usr_template.format(text=text)})
            # Format using the tokenizer's chat template
            try:
                formatted_text = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
            except Exception:
                formatted_text = text
        else:
            formatted_text = text

        enc = tokenizer(formatted_text, return_tensors="pt", truncation=True)
        enc = {k: v.to(device) for k, v in enc.items()}

        output = self(**enc)
        logits = output.logits  # (1, num_labels)

        if self.num_labels == 1:
            # Regression: return raw value
            pred_label = str(logits.squeeze().item())
            probs = None
        elif self.num_labels == 2:
            probs_t = torch.softmax(logits, dim=-1).squeeze(0) if return_probabilities else None
            pred_id = int(logits.argmax(dim=-1).item())
            pred_label = getattr(self.config, "id2label", {0: "0", 1: "1"}).get(
                pred_id, str(pred_id)
            )
            probs = probs_t
        else:
            probs_t = torch.softmax(logits, dim=-1).squeeze(0) if return_probabilities else None
            pred_id = int(logits.argmax(dim=-1).item())
            pred_label = getattr(self.config, "id2label", {}).get(pred_id, str(pred_id))
            probs = probs_t

        return pred_label, probs

    @classmethod
    def from_causal_lm(
        cls,
        causal_lm_model,
        num_labels: int = 2,
        id2label: Optional[dict] = None,
        label2id: Optional[dict] = None,
        classifier_dropout: float = 0.0,
        label_token_ids: Optional[List[int]] = None,
        system_prompt: Optional[str] = None,
        user_template: Optional[str] = None,
    ) -> "EchoForSequenceClassification":
        """
        Construct an :class:`EchoForSequenceClassification` from a fully
        merged :class:`EchoForCausalLM` instance (i.e. after LoRA weights
        have been merged via ``peft.merge_adapter``).

        The backbone weights are copied; the ``lm_head`` is discarded.

        Classifier head initialisation
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        If ``label_token_ids`` is provided (one token ID per class), the
        classifier weight rows are seeded directly from the corresponding
        ``lm_head`` weight rows.  This is the correct initialisation for
        **generative** adapters that were fine-tuned to emit a label token
        (e.g. ``"0"`` or ``"1"``): the backbone already knows how to push
        the last hidden state toward those tokens, so we preserve that signal
        instead of starting from random.

        Parameters
        ----------
        causal_lm_model:
            A loaded (and optionally LoRA-merged) ``EchoForCausalLM`` instance.
        num_labels:
            Number of output classes.
        id2label:
            Optional mapping ``{int -> str}`` for label names.
        label2id:
            Optional reverse mapping ``{str -> int}``.
        classifier_dropout:
            Dropout probability before the classification head.
        label_token_ids:
            Optional list of ``num_labels`` token IDs.  When supplied, row
            ``i`` of the ``lm_head`` weight matrix is copied into row ``i``
            of the classifier weight matrix, seeding the head from the
            causal model's learned token distributions.
            Example for Echo-DSRN NSFW adapter::

                label_token_ids=[29900, 29896]  # token IDs for "0" and "1"

        Returns
        -------
        EchoForSequenceClassification
        """
        if id2label is None:
            id2label = {i: str(i) for i in range(num_labels)}
        if label2id is None:
            label2id = {v: k for k, v in id2label.items()}

        # Validate label_token_ids length
        if label_token_ids is not None and len(label_token_ids) != num_labels:
            raise ValueError(
                f"label_token_ids has {len(label_token_ids)} entries but num_labels={num_labels}. "
                "Must provide exactly one token ID per class."
            )

        # Clone config and inject classification fields
        config = causal_lm_model.config
        config.num_labels = num_labels
        config.id2label = {int(k): v for k, v in id2label.items()}
        config.label2id = label2id
        config.classifier_dropout = classifier_dropout

        if system_prompt is not None:
            config.system_prompt = system_prompt
        if user_template is not None:
            config.user_template = user_template

        # Carry dtype forward so save_pretrained serialises it correctly
        if hasattr(causal_lm_model, "dtype"):
            config.torch_dtype = str(causal_lm_model.dtype).replace("torch.", "")
        # Update auto_map so Hub users get the right class on from_pretrained
        config.auto_map = {
            "AutoConfig": "configuration_echo.EchoConfig",
            "AutoModel": "modeling_echo.EchoModel",
            "AutoModelForSequenceClassification": ("modeling_echo.EchoForSequenceClassification"),
        }

        # Build the classifier wrapper
        clf_model = cls(config)

        # Copy backbone weights
        backbone_sd = causal_lm_model.model.state_dict()
        missing, unexpected = clf_model.model.load_state_dict(backbone_sd, strict=True)
        if missing:
            import warnings

            warnings.warn(
                f"EchoForSequenceClassification.from_causal_lm: "
                f"missing backbone keys: {missing}",
                UserWarning,
            )
        if unexpected:
            import warnings

            warnings.warn(
                f"EchoForSequenceClassification.from_causal_lm: "
                f"unexpected backbone keys: {unexpected}",
                UserWarning,
            )

        # --- Seed classifier head from lm_head rows (generative adapter path) ---
        if label_token_ids is not None:
            lm_head_weight = causal_lm_model.lm_head.weight  # (vocab_size, embed_dim)
            with torch.no_grad():
                for label_idx, token_id in enumerate(label_token_ids):
                    clf_model.classifier.weight[label_idx].copy_(lm_head_weight[token_id])
                # Zero-init bias so initial scores are purely from the weight rows
                torch.nn.init.zeros_(clf_model.classifier.bias)

        # --- Cast entire model to the source dtype ---
        # cls(config) initialises weights in float32 by default.
        # We cast everything uniformly AFTER all weight copies so that both
        # the backbone and the seeded classifier head end up in the same precision.
        src_dtype = causal_lm_model.dtype  # e.g. torch.bfloat16
        if src_dtype != torch.float32:
            clf_model = clf_model.to(src_dtype)
            # Persist in config using the current (non-deprecated) field name
            config.dtype = str(src_dtype).replace("torch.", "")

        return clf_model
