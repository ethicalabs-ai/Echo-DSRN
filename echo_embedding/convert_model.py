import argparse
import json
import os
import shutil

from transformers import AutoTokenizer

from echo_dsrn import EchoConfig, EchoForCausalLM
from echo_embedding.modeling_embedding import EchoModelForSentenceEmbedding


def convert_model(
    base_model_path: str,
    output_dir: str,
    peft_model_path: str = None,
    pooling_mode: str = "c_T",
    attention_masking: str = "causal",
):
    print(f"🔄 Starting conversion of {base_model_path} to embedding model...")
    if os.path.exists(output_dir):
        print(f"🧹 Cleaning existing target directory: {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # SentenceTransformers subdirectories
    transformer_dir = os.path.join(output_dir, "0_Transformer")
    os.makedirs(transformer_dir, exist_ok=True)

    # 1. Load tokenizer and save to transformer directory
    print("⏳ Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(transformer_dir)

    # 2. Load model config and modify auto_map
    print("⏳ Modifying and saving config...")
    config = EchoConfig.from_pretrained(base_model_path)
    config.pooling_mode = pooling_mode
    config.attention_masking = attention_masking

    # Configure auto_map so that AutoModel resolves to modeling_embedding.EchoModelForSentenceEmbedding
    config.auto_map = {
        "AutoConfig": "configuration_echo.EchoConfig",
        "AutoModel": "modeling_embedding.EchoModelForSentenceEmbedding",
        "AutoModelForCausalLM": "modeling_echo.EchoForCausalLM",
    }

    # Update config model name to match the output directory
    basename = os.path.basename(output_dir.rstrip("/"))
    if basename:
        new_model_name = f"ethicalabs/{basename}"
        print(f"✏️ Updating config hf_model_name to: {new_model_name}")
        config.hf_model_name = new_model_name

    # 3. Instantiate and save the embedding model weights
    print("⏳ Loading weights and saving model...")
    # Temporarily clear auto_map for local instantiation to prevent HF recursion
    temp_config = EchoConfig.from_pretrained(base_model_path)
    temp_config.pooling_mode = pooling_mode
    temp_config.attention_masking = attention_masking
    temp_config.auto_map = {}

    if peft_model_path:
        print(f"⏳ Loading base model for PEFT merge from {base_model_path}...")
        base_model = EchoForCausalLM.from_pretrained(base_model_path, config=temp_config)
        from peft import PeftModel

        print(f"⏳ Loading and merging PEFT adapter from {peft_model_path}...")
        peft_model = PeftModel.from_pretrained(base_model, peft_model_path)
        merged_model = peft_model.merge_and_unload()

        # Instantiate empty embedding model and load merged weights
        embed_model = EchoModelForSentenceEmbedding(temp_config)
        embed_model.load_state_dict(merged_model.state_dict(), strict=False)
    else:
        embed_model = EchoModelForSentenceEmbedding.from_pretrained(
            base_model_path, config=temp_config
        )
    embed_model.save_pretrained(transformer_dir)

    # Overwrite the saved config with the custom auto-mapped version
    config.save_pretrained(transformer_dir)

    # 4. Copy modeling code to both root and transformer directories
    print("⏳ Copying modeling code...")
    local_modeling_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "echo_embedding",
        "modeling_embedding.py",
    )
    # Copy modeling_embedding.py
    shutil.copy(local_modeling_path, os.path.join(output_dir, "modeling_embedding.py"))
    shutil.copy(local_modeling_path, os.path.join(transformer_dir, "modeling_embedding.py"))

    # Copy echo_dsrn source files to make both directories self-contained
    echo_dsrn_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "echo_dsrn")
    for filename in ["modeling_echo.py", "configuration_echo.py", "triton_scan.py"]:
        src_file = os.path.join(echo_dsrn_dir, filename)
        if os.path.exists(src_file):
            if filename == "modeling_echo.py":
                with open(src_file, "r", encoding="utf-8") as f:
                    content = f.read()
                # Make the import of triton_scan robust against HF cache namespace changes
                target_import = (
                    "            try:\n"
                    "                from .triton_scan import triton_dsrn_parallel_scan\n"
                    "            except ImportError:\n"
                    "                try:\n"
                    "                    from triton_scan import triton_dsrn_parallel_scan\n"
                    "                except ImportError:\n"
                    "                    from echo_dsrn.triton_scan import triton_dsrn_parallel_scan"
                )
                content = content.replace(
                    "            from .triton_scan import triton_dsrn_parallel_scan",
                    target_import,
                )
                for dest_dir in [output_dir, transformer_dir]:
                    with open(os.path.join(dest_dir, filename), "w", encoding="utf-8") as f:
                        f.write(content)
            else:
                shutil.copy(src_file, os.path.join(output_dir, filename))
                shutil.copy(src_file, os.path.join(transformer_dir, filename))

    # 5. Write modules.json for SentenceTransformers
    print("⏳ Writing SentenceTransformers modules.json...")
    modules = [
        {
            "name": "0",
            "type": "sentence_transformers.models.Transformer",
            "path": "0_Transformer",
        },
        {
            "name": "1",
            "type": "sentence_transformers.models.Pooling",
            "path": "1_Pooling",
        },
    ]
    with open(os.path.join(output_dir, "modules.json"), "w", encoding="utf-8") as f:
        json.dump(modules, f, indent=2)

    # 6. Write 1_Pooling/config.json
    print("⏳ Writing SentenceTransformers pooling configuration...")
    pooling_dir = os.path.join(output_dir, "1_Pooling")
    os.makedirs(pooling_dir, exist_ok=True)

    if getattr(config, "project_embeddings", False) or getattr(config, "projection_mlp", False):
        word_emb_dim = getattr(config, "embedding_dim", config.hidden_size)
    else:
        pooling_mode = getattr(config, "pooling_mode", "c_T")
        if pooling_mode == "hybrid":
            word_emb_dim = config.hidden_size * (config.num_heads + 1)
        elif pooling_mode == "mean_x_out":
            word_emb_dim = config.hidden_size
        else:
            word_emb_dim = config.hidden_size * config.num_heads

    pooling_config = {
        "word_embedding_dimension": word_emb_dim,
        "pooling_mode_cls_token": False,
        "pooling_mode_mean_tokens": True,
        "pooling_mode_max_tokens": False,
    }
    with open(os.path.join(pooling_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(pooling_config, f, indent=2)

    print(f"🎉 Conversion complete! Converted model saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert Echo-DSRN Causal LM to embedding model")
    parser.add_argument(
        "--base_model",
        type=str,
        default="ethicalabs/Echo-DSRN-114M-v0.1.2",
        help="Base model name or path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/Echo-DSRN-v0.1.3-Embed",
        help="Target output directory",
    )
    parser.add_argument(
        "--peft_model",
        type=str,
        default=None,
        help="PEFT adapter name or path to load and merge (optional)",
    )
    parser.add_argument(
        "--pooling_mode",
        type=str,
        default="c_T",
        help="Pooling mode ('c_T', 'mean_c_all', 'mean_x_out', 'hybrid')",
    )
    parser.add_argument(
        "--attention_masking",
        type=str,
        default="causal",
        help="Attention masking strategy ('causal', 'non_causal_window')",
    )
    args = parser.parse_args()

    convert_model(
        args.base_model,
        args.output_dir,
        args.peft_model,
        pooling_mode=args.pooling_mode,
        attention_masking=args.attention_masking,
    )


if __name__ == "__main__":
    main()
