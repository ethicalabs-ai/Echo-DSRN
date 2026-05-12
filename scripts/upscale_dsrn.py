import json
import os
import shutil

import click
import torch
import torch.nn as nn
from echo_hf.modeling_echo import EchoConfig, EchoForCausalLM

# Net2Net Initialization Strategy
# -------------------------------
# Core idea: Initialize a larger network to be functionally equivalent (or close)
# to a smaller pre-trained predecessor.
#
# Strategies:
# 1. Net2WiderNet: Expand width (hidden_size, heads). Use Zero-Padding for new weights to preserve output.
# 2. Net2DeeperNet: Expand depth (layers). Initialize new layers as Identity (using skip connections).
#    Since Echo-DSRN uses specific residuals (x + f(x)), pure identity initialization is tough for non-residual blocks.
#    However, for residual blocks x_{l+1} = x_l + F(x_l), we can init F(x_l) near zero.
#    But user requested "Cloning" existing layers with noise, which is a pragmatic approach.


class Net2NetSurgeon:
    def __init__(self, source_model_path, target_config_path, output_path, device="cpu"):
        self.device = device
        self.output_path = output_path

        # Load Source
        print(f"Loading source model from {source_model_path}...")
        try:
            self.source_config = EchoConfig.from_pretrained(source_model_path)
            self.source_model = EchoForCausalLM.from_pretrained(
                source_model_path, config=self.source_config
            ).to(device)
            # Remove any LoRA adapter metadata to ensure clean loading
            # But the object is already instantiated.
        except Exception as e:
            print(f"Error loading source model: {e}")
            raise

        # Load Target Config
        print(f"Loading target config from {target_config_path}...")
        try:
            self.target_config = EchoConfig.from_pretrained(target_config_path)
        except Exception:
            # Maybe it's a raw yaml file, not a model directory
            import yaml

            with open(target_config_path, "r") as f:
                config_dict = yaml.safe_load(f)
            self.target_config = EchoConfig(**config_dict)

        # Validate
        if self.target_config.vocab_size != self.source_config.vocab_size:
            print(
                f"Warning: Vocab size mismatch! Source: {self.source_config.vocab_size}, Target: {self.target_config.vocab_size}"
            )
            # We will handle embedding resizing if needed

        # Instantiate Target
        print("Instantiating target model (Random Init)...")
        self.target_model = EchoForCausalLM(self.target_config).to(device)

        # Mapping Stats
        self.stats = {"copied": 0, "expanded_width": 0, "cloned_depth": 0, "skipped": 0}

    def expand_tensor(self, source_tensor, target_shape):
        """
        Expands a source tensor to target shape using Net2Net padding (zeros for new dims).
        """
        with torch.no_grad():
            s_shape = source_tensor.shape
            t_shape = target_shape

            if s_shape == t_shape:
                return source_tensor.clone()

            # Create target tensor with Zeros (Function Preservation)
            new_tensor = torch.zeros(t_shape, device=self.device, dtype=source_tensor.dtype)

            # Slice logic
            # 1D (Bias, Norm)
            if len(s_shape) == 1:
                # Copy [0:s_dim]
                new_tensor[: s_shape[0]] = source_tensor

            # 2D (Linear Weight: Out, In)
            elif len(s_shape) == 2:
                # Copy [0:s_out, 0:s_in]
                new_tensor[: s_shape[0], : s_shape[1]] = source_tensor

            return new_tensor

    def perform_surgery(self):
        print("Starting surgery...")

        source_sd = self.source_model.state_dict()
        target_sd = self.target_model.state_dict()

        # 1. Globals (Embeddings, Final Norm, LM Head)
        self._transfer_globals(source_sd, target_sd)

        # 2. Layers (Blocks)
        self._transfer_blocks(source_sd, target_sd)

        # Load the new state dict into target model
        print("Loading mapped weights into target model...")
        missing, unexpected = self.target_model.load_state_dict(target_sd, strict=False)
        if missing:
            print(f"Missing keys (initialized randomly): {len(missing)}")
            # print(missing[:5])
        if unexpected:
            print(f"Unexpected keys in target: {len(unexpected)}")

    def _transfer_globals(self, source_sd, target_sd):
        # Embeddings
        self._map_weight(source_sd, target_sd, "model.embedding.weight", "model.embedding.weight")

        # Final Norm
        self._map_weight(source_sd, target_sd, "model.final_norm.weight", "model.final_norm.weight")
        # Handle Bias if it exists (LayerNorm has bias, RMSNorm usually doesn't or has separate scaling)
        if "model.final_norm.bias" in source_sd:
            self._map_weight(source_sd, target_sd, "model.final_norm.bias", "model.final_norm.bias")

        # LM Head
        self._map_weight(source_sd, target_sd, "lm_head.weight", "lm_head.weight")

    def _transfer_blocks(self, source_sd, target_sd):
        n_layers_src = self.source_config.num_layers
        n_layers_tgt = self.target_config.num_layers

        print(f"Mapping {n_layers_src} source layers to {n_layers_tgt} target layers...")

        for i in range(n_layers_tgt):
            src_i = i  # Default
            method = "direct"

            # Depth Expansion Logic
            if i >= n_layers_src:
                method = "cloned_noise"
                extension_len = n_layers_tgt - n_layers_src
                idx_in_extension = i - n_layers_src

                # Clone from the end of the source model to ensure deepest features
                # E.g. Src=18, Tgt=24. Extension=6.
                # i=18 (idx=0) -> src_i = 12
                # i=23 (idx=5) -> src_i = 17
                src_i = (n_layers_src - extension_len + idx_in_extension) % n_layers_src

            prefix_src = f"model.blocks.{src_i}"
            prefix_tgt = f"model.blocks.{i}"

            self._transfer_layer(
                source_sd, target_sd, prefix_src, prefix_tgt, noise=(method == "cloned_noise")
            )

    def _transfer_layer(self, source_sd, target_sd, prefix_src, prefix_tgt, noise=False):
        # Sub-modules in DSRNBlock
        modules = [
            # Norms (RMSNorm has only weight; LayerNorm also has bias — _map_weight skips missing keys)
            "norm_fast.weight",
            "norm_fast.bias",
            "norm_ff.weight",
            "norm_ff.bias",
            # GRU (Special handling via _expand_gru_weight/_expand_gru_bias)
            "gru_cell.weight_ih",
            "gru_cell.bias_ih",
            "gru_cell.weight_hh",
            "gru_cell.bias_hh",
            # Hybrid Attention (qkv_proj has no bias; out_proj has no bias)
            "attn.qkv_proj.weight",
            "attn.out_proj.weight",
            # DSRN Linears
            "linear_read.weight",
            "linear_gate.weight",
            "linear_gate.bias",
            "linear_memory.weight",
            "linear_memory.bias",
            "linear_pred.weight",
            # Surprise
            "surprise_lambda",
            # MLP (canonical names — NOT the old mlp.0/mlp.2 nn.Sequential aliases)
            "mlp_up.weight",
            "mlp_up.bias",
            "mlp_down.weight",
            "mlp_down.bias",
        ]

        for key_suffix in modules:
            key_src = f"{prefix_src}.{key_suffix}"
            key_tgt = f"{prefix_tgt}.{key_suffix}"

            self._map_weight(source_sd, target_sd, key_src, key_tgt, noise_injection=noise)

    def _map_weight(self, source_sd, target_sd, key_src, key_tgt, noise_injection=False):
        if key_src not in source_sd:
            return

        if key_tgt not in target_sd:
            return

        w_src = source_sd[key_src]
        w_tgt = target_sd[key_tgt]

        # 1. Expand/Crop
        # Special logic for GRU weights which are stacked (3*H, H)
        # and QKV proj which is also stacked (3*H, H)
        if "gru_cell.weight_ih" in key_tgt or "gru_cell.weight_hh" in key_tgt:
            w_mapped = self._expand_gru_weight(w_src, w_tgt.shape)
        elif "gru_cell.bias" in key_tgt:
            w_mapped = self._expand_gru_bias(w_src, w_tgt.shape)
        elif "attn.qkv_proj.weight" in key_tgt:
            # QKV proj: (3*H_src, H_src) -> (3*H_tgt, H_tgt) — same chunk logic as GRU
            w_mapped = self._expand_gru_weight(w_src, w_tgt.shape)
        else:
            w_mapped = self.expand_tensor(w_src, w_tgt.shape)

        # 2. RMSNorm scaling (Critical for Function Preservation)
        # If we expand width (H -> H'), zero-padding reduces variance by H/H'.
        # RMSNorm divides by sqrt(Var). So activations increase by sqrt(H'/H).
        # To compensate, we reduce weights by sqrt(H/H').
        if "norm" in key_tgt and "weight" in key_tgt and len(w_src.shape) == 1:
            s_dim = w_src.shape[0]
            t_dim = w_tgt.shape[0]
            if t_dim > s_dim:
                scale_factor = (s_dim / t_dim) ** 0.5
                # Determine slice to scale (copied part)
                # w_mapped[:s_dim] is the copied weight
                w_mapped[:s_dim] *= scale_factor
                # print(f"  Scaled {key_tgt} by {scale_factor:.4f}")

        # 3. Noise Injection (For Depth Cloning)
        if noise_injection:
            # Add small gaussian noise to break symmetry
            noise = torch.randn_like(w_mapped) * 1e-3  # 0.001 std
            w_mapped = w_mapped + noise
            self.stats["cloned_depth"] += 1
        elif w_src.shape != w_tgt.shape:
            self.stats["expanded_width"] += 1
        else:
            self.stats["copied"] += 1

        # Write to target State Dict
        target_sd[key_tgt] = w_mapped

    def _expand_gru_weight(self, source, target_shape):
        # Source: (3*H1, H1)
        # Target: (3*H2, H2)

        chunk_dim = 0
        src_chunks = torch.chunk(source, 3, dim=chunk_dim)

        # Calculate target chunk size
        tgt_chunk_size = target_shape[0] // 3
        tgt_in_size = target_shape[1]

        mapped_chunks = []
        for chunk in src_chunks:
            # Chunk is (H1, H1)
            mapped = self.expand_tensor(chunk, (tgt_chunk_size, tgt_in_size))
            mapped_chunks.append(mapped)

        return torch.cat(mapped_chunks, dim=chunk_dim)

    def _expand_gru_bias(self, source, target_shape):
        # Source: (3*H1)
        # Target: (3*H2)

        chunk_dim = 0
        src_chunks = torch.chunk(source, 3, dim=chunk_dim)
        tgt_chunk_size = target_shape[0] // 3

        mapped_chunks = []
        for chunk in src_chunks:
            mapped = self.expand_tensor(chunk, (tgt_chunk_size,))
            mapped_chunks.append(mapped)

        return torch.cat(mapped_chunks, dim=chunk_dim)

    def save(self):
        # Break weight tying for safetensors compatibility
        # Echo-DSRN typically ties embedding and lm_head
        if (
            self.target_model.get_output_embeddings().weight
            is self.target_model.get_input_embeddings().weight
        ):
            print("Breaking weight tying for SafeTensors compatibility...")
            self.target_model.get_output_embeddings().weight = nn.Parameter(
                self.target_model.get_output_embeddings().weight.clone()
            )

        print(f"Saving upscaled model to {self.output_path}...")
        self.target_model.save_pretrained(self.output_path)
        self.target_config.save_pretrained(self.output_path)

        # Copy modeling files so the output dir is self-contained
        echo_hf_dir = os.path.dirname(os.path.abspath(__file__))
        for fname in ["modeling_echo.py", "configuration_echo.py", "triton_scan.py"]:
            src = os.path.join(echo_hf_dir, fname)
            dst = os.path.join(self.output_path, fname)
            if os.path.exists(src):
                shutil.copy(src, dst)
                print(f"  Copied {fname}")

        print("Done.")
        print("Stats:", json.dumps(self.stats, indent=2))


@click.command()
@click.option("--from-config", required=True, help="Path to source YAML config")
@click.option("--to-config", required=True, help="Path to target YAML config")
@click.option("--input-model", required=True, help="Path to source model directory")
@click.option("--output-model", required=True, help="Path to output model directory")
def main(from_config, to_config, input_model, output_model):
    """
    Upscale an Echo-DSRN model structurally (Net2Net).
    Handles width expansion (padding) and depth expansion (cloning).
    """

    # 1. Verify paths
    if not os.path.exists(input_model):
        raise FileNotFoundError(f"Input model not found: {input_model}")

    # 2. Create output dir
    os.makedirs(output_model, exist_ok=True)

    # 3. Copy Tokenizer (Vocab usually constant)
    print("Copying tokenizer files...")
    for filename in [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ]:
        src = os.path.join(input_model, filename)
        if os.path.exists(src):
            shutil.copy(src, output_model)

    # 4. Perform Surgery
    device = "cuda" if torch.cuda.is_available() else "cpu"
    surgeon = Net2NetSurgeon(input_model, to_config, output_model, device=device)
    surgeon.perform_surgery()
    surgeon.save()

    print(f"\nSUCCESS: Model upscaled and saved to {output_model}")


if __name__ == "__main__":
    main()
