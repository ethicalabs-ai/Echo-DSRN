import os
import shutil
import sys

import torch
from peft import PeftModel
from transformers import AutoTokenizer, GenerationConfig

# Add parent trajectory to path to import Hybrid components
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from echo_hybrid.modeling_hybrid import HybridEchoForCausalLM

CHAT_TEMPLATE_QWEN = """{%- if tools %}
    {{- '<|im_start|>system\\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {{- messages[0]['content'] }}
    {%- else %}
        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}
    {%- endif %}
    {{- "\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}
    {%- for tool in tools %}
        {{- "\\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}
    {%- else %}
        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}
        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role }}
        {%- if message.content %}
            {{- '\\n' + message.content }}
        {%- endif %}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- '\\n<tool_call>\\n{"name": "' }}
            {{- tool_call.name }}
            {{- '", "arguments": ' }}
            {{- tool_call.arguments | tojson }}
            {{- '}\\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\\n' }}
    {%- elif message.role == "tool" %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\\n<tool_response>\\n' }}
        {{- message.content }}
        {{- '\\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
{%- endif %}"""


def merge_peft(base_model_path, adapter_path, output_path, bf16=False):
    dtype_label = "bfloat16" if bf16 else "float32"
    print("--- Merging PEFT Adapter into Hybrid Engine ---")
    print(f"Base:   {base_model_path}")
    print(f"Adapter:{adapter_path}")
    print(f"Output: {output_path}")
    print(f"Dtype:  {dtype_label} (merge always in float32 for precision, cast after)")

    # 1. Load Base Model (Hybrid)
    print("Loading Hybrid Base Model...")
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)

    model = HybridEchoForCausalLM.from_pretrained(
        base_model_path,
        config=config,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )

    # 2. Load PEFT Adapter
    print("Attaching LoRA Adapters...")
    peft_model = PeftModel.from_pretrained(model, adapter_path)

    # 3. Merge and Unload
    print("Merging weights (Sequential Injection)...")
    merged_model = peft_model.merge_and_unload()
    merged_model.eval()

    # 4. Save Merged Weights
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    print("Saving merged weights...")

    if bf16:
        print("Casting to bfloat16 before saving...")
        merged_model = merged_model.to(torch.bfloat16)

    merged_model.save_pretrained(output_path, safe_serialization=True)

    # 5. Save Tokenizer and enforce Kurtis-EON1 persona
    print("Finalizing Tokenizer (Inheriting Template + Injecting Identity)...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    # Define the preambles with their appropriate quoting for Jinja safety
    # The original Qwen template uses single quotes at line 6
    qwen_single = "'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'"
    qwen_raw = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

    # Bulletproof string: NO APOSTROPHES OR QUOTES inside the text.
    kurtis_raw = (
        'You are Kurtis-EON1, a deeply empathetic and sophisticated multilingual AI assistant. '
        'Your purpose is to provide emotionally intelligent, culturally aware, and highly personalized support. '
        'You listen with genuine care, validate the user perspective, and offer guidance that is '
        'both kindness-driven and technically precise across all languages.'
    )
    kurtis_single = f"'{kurtis_raw}'"

    # 1. Update the tokenizer object's template
    template = tokenizer.chat_template
    if template:
        template = template.replace(qwen_single, kurtis_single)
        template = template.replace(qwen_raw, kurtis_raw)
        tokenizer.chat_template = template

    tokenizer.save_pretrained(output_path)

    # 2. Update the standalone chat_template.jinja if it exists
    jinja_src = os.path.join(base_model_path, "chat_template.jinja")
    if os.path.exists(jinja_src):
        print("  + Synchronizing chat_template.jinja")
        with open(jinja_src, "r") as f:
            jinja_content = f.read()

        jinja_content = jinja_content.replace(qwen_single, kurtis_single)
        jinja_content = jinja_content.replace(qwen_raw, kurtis_raw)

        with open(os.path.join(output_path, "chat_template.jinja"), "w") as f:
            f.write(jinja_content)

    # 6. Copy Hybrid Components for stand-alone deployment
    print("Injecting Stand-alone Components...")
    # These should be in the 'echo_hybrid' directory relative to this script
    hybrid_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    project_root = os.path.abspath(os.path.join(hybrid_root, '..'))

    components = [
        ("modeling_hybrid.py", hybrid_root),
        ("dsrn_memory_block.py", hybrid_root),
        ("configuration_hybrid.py", hybrid_root),
        ("triton_scan.py", os.path.join(project_root, "echo_hf")),  # Fix: look in echo_hf
    ]

    for comp_name, comp_dir in components:
        src = os.path.join(comp_dir, comp_name)
        if os.path.exists(src):
            print(f"  + Copying {comp_name}")
            shutil.copy(src, os.path.join(output_path, comp_name))
        else:
            print(f"  ! Warning: {comp_name} not found in {comp_dir}")

    # 7. Finalize generation config
    gen_config = GenerationConfig.from_pretrained(base_model_path)
    gen_config.save_pretrained(output_path)

    # 8. Enforce Standard Mode (CACHED+DSRN) in the exported config.
    #    use_kv_cache=True ensures the backbone KV cache is active so
    #    model.generate() (and talk.py) work correctly out of the box.
    import json

    config_path = os.path.join(output_path, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)
    cfg["use_kv_cache"] = True
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print("  ✓ config.json patched: use_kv_cache=True (Standard Mode)")

    print(f"✅ STAND-ALONE HYBRID MODEL READY: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge Hybrid PEFT adapters.")
    parser.add_argument("--base", type=str, required=True, help="Base hybrid model path")
    parser.add_argument("--adapter", type=str, required=True, help="Adapter path")
    parser.add_argument("--output", type=str, required=True, help="Output path")
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=False,
        help="Save merged model in bfloat16 (merge is always done in float32 for precision).",
    )

    args = parser.parse_args()
    merge_peft(args.base, args.adapter, args.output, bf16=args.bf16)
