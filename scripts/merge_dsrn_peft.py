import json
import os
import shutil
import sys
import tempfile

import click
import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoTokenizer, GenerationConfig

# Add parent trajectory to path to import EchoModel
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from echo_hf.modeling_echo import EchoForCausalLM

from scripts.rename_weights import rename_mlp_weights


def _is_local_path(path: str) -> bool:
    """Check if the given string is a local directory path (vs a HF Hub model ID)."""
    return os.path.isdir(os.path.abspath(path))


@click.command()
@click.option("--base_model", required=True, help="Path or HF Hub ID of the base model.")
@click.option("--adapter", required=True, help="Path to the PEFT adapter directory.")
@click.option("--output", required=True, help="Directory to save the merged model.")
@click.option("--device", default="cpu", help="Device to use for merging (cpu, cuda, mps).")
@click.option("--safe-serialization", default=True, type=bool, help="Use safe tensors for saving.")
@click.option(
    "--test-gen", is_flag=True, help="Run a test generation after merging to verify quality."
)
def main(base_model, adapter, output, device, safe_serialization, test_gen):
    """
    Merge a PEFT adapter into the Echo-DSRN base model.
    Exactly following the logic of test_merge.py.
    """
    adapter_path = os.path.abspath(adapter)

    print("--- 1. Migrating Weights to Temporary Workspace ---")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_base_dir = os.path.join(temp_dir, "base_model")
        os.makedirs(temp_base_dir)

        if _is_local_path(base_model):
            base_model_path = os.path.abspath(base_model)
            print(f"Using local base model: {base_model_path}")
            # Copy config files
            shutil.copy(os.path.join(base_model_path, "config.json"), temp_base_dir)
            if os.path.exists(os.path.join(base_model_path, "generation_config.json")):
                shutil.copy(os.path.join(base_model_path, "generation_config.json"), temp_base_dir)

            print("Migrating base model weights to temporary directory...")
            input_safetensors = os.path.join(base_model_path, "model.safetensors")
            output_safetensors = os.path.join(temp_base_dir, "model.safetensors")
            rename_mlp_weights(input_safetensors, output_safetensors)
        else:
            print(f"Downloading base model from HF Hub: {base_model}")
            downloaded_path = snapshot_download(
                base_model,
                local_dir=temp_base_dir,
            )
            print(f"Downloaded to: {downloaded_path}")

            # Rename MLP weights in-place
            safetensors_path = os.path.join(temp_base_dir, "model.safetensors")
            if os.path.exists(safetensors_path):
                print("Migrating base model weights...")
                rename_mlp_weights(safetensors_path, safetensors_path)

        print("\n--- 2. Loading Migrated Model ---")
        # Identical to test_merge.py: torch.float32 on CPU for precision
        model = EchoForCausalLM.from_pretrained(
            temp_base_dir, device_map=device, torch_dtype=torch.float32
        )
        print("✅ Base Model loaded successfully.")

        print("\n--- 3. Testing PEFT Adapter Loading ---")
        temp_adapter_dir = os.path.join(temp_dir, "adapter")
        os.makedirs(temp_adapter_dir)

        # Patch and copy adapter config
        adapter_config_orig = os.path.join(adapter_path, "adapter_config.json")
        with open(adapter_config_orig, "r") as f:
            adapter_config = json.load(f)

        if "target_modules" in adapter_config:
            new_targets = []
            for target in adapter_config["target_modules"]:
                if target == "mlp.0":
                    new_targets.append("mlp_up")
                elif target == "mlp.2":
                    new_targets.append("mlp_down")
                else:
                    new_targets.append(target)
            adapter_config["target_modules"] = new_targets

        with open(os.path.join(temp_adapter_dir, "adapter_config.json"), "w") as f:
            json.dump(adapter_config, f)

        # Migrate adapter weights
        print("Patching adapter weights...")
        adapter_weights_in = os.path.join(adapter_path, "adapter_model.safetensors")
        adapter_weights_out = os.path.join(temp_adapter_dir, "adapter_model.safetensors")
        rename_mlp_weights(adapter_weights_in, adapter_weights_out)

        print(f"Loading PeftModel from {temp_adapter_dir}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(temp_base_dir, trust_remote_code=True)

        model.resize_token_embeddings(len(tokenizer))

        peft_model = PeftModel.from_pretrained(model, temp_adapter_dir)
        print("✅ PeftModel attached successfully.")

        print("\n--- 4. PEFT Merge ---")
        print("Merging weights...")
        merged_model = peft_model.merge_and_unload()
        print("✅ Merge successful!")

        # --- 4.5. Test Generation ---
        if test_gen:
            print("\n--- 4.5. Testing Generation Quality ---")
            merged_model.eval()
            try:
                test_tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
            except Exception:
                test_tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

            prompt = "The capital of France is"
            inputs = test_tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = merged_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.4,
                    top_p=0.9,
                    top_k=40,
                    repetition_penalty=1.1,
                    num_beams=4,
                    no_repeat_ngram_size=2,
                    early_stopping=True,
                )
            print(f"Prompt: {prompt}")
            print(f"Output: {test_tokenizer.decode(out[0], skip_special_tokens=True)}")

        print("\n--- 5. Saving Merged Model ---")
        if os.path.exists(output):
            print(f"Warning: Output directory {output} already exists. Overwriting...")
            shutil.rmtree(output, ignore_errors=True)
        os.makedirs(output)

        # Filter out aliased keys to avoid shared tensor error (exactly as in test_merge.py)
        state_dict = merged_model.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items() if ".mlp." not in k}

        # Set final dtype to bfloat16 for the saved model
        merged_model.config.torch_dtype = "bfloat16"
        merged_model.save_pretrained(
            output, safe_serialization=safe_serialization, state_dict=filtered_state_dict
        )

        # Save Tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        tokenizer.save_pretrained(output)

        # Save production files (modeling, etc)
        base_hf_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for f_name in ["modeling_echo.py", "configuration_echo.py", "handler.py", "triton_scan.py"]:
            src = os.path.join(base_hf_dir, f_name)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(output, f_name))

        # Save healthy GenerationConfig
        gen_config = GenerationConfig(
            do_sample=True,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            eos_token_id=[32000, 32007, 32011],
            pad_token_id=32000,
        )
        gen_config.save_pretrained(output)

    print(f"\nSuccess! Merged model saved to: {output}")


if __name__ == "__main__":
    main()
