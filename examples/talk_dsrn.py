import os
import time

import click
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextStreamer,
)

from echo_dsrn.modeling_echo import EchoConfig, EchoForCausalLM


class StringStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_strings):
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings

    def __call__(self, input_ids, scores, **kwargs):
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
        # Check if the generated text (stripped) ends with any stop string
        text_to_check = generated_text.strip()
        for stop_string in self.stop_strings:
            if text_to_check.endswith(stop_string):
                return True
        return False


SYSTEM_PROMPT = "You are Kurtis-EON1, built by ethicalabs.ai. " "Be direct, and concise."


@click.command()
@click.option("--model_path", required=True, help="Path to the HF model directory.")
@click.option("--text", default=None, help="Input text for single-turn generation.")
@click.option("--chat/--no-chat", default=True, help="Use chat template for input.")
@click.option("--temperature", default=0.7, help="Sampling temperature.")
@click.option("--max_tokens", default=512, help="Max new tokens to generate.")
@click.option("--tokenizer", default=None, help="Tokenizer path/name (overrides model_path).")
@click.option("--system_prompt", default=SYSTEM_PROMPT, help="System prompt.")
@click.option("--seed", default=42, help="Random seed for reproducibility.")
def main(model_path, text, chat, temperature, max_tokens, tokenizer, system_prompt, seed):
    """
    Run interactive chat with Echo-HF model using full-context re-generation.
    """
    # Set seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    print(f"Loading model from {model_path}...")
    AutoConfig.register("echo", EchoConfig)

    # Check for LoRA
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        from peft import PeftConfig, PeftModel

        print(f"Detected LoRA adapter at {model_path}")
        config = PeftConfig.from_pretrained(model_path)
        base_model_path = config.base_model_name_or_path
        print(f"Loading base model from {base_model_path}...")
        model = EchoForCausalLM.from_pretrained(
            base_model_path, device_map=device, trust_remote_code=True
        )
        print("Loading adapter...")
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()  # Merge for speed/stability
    else:
        model = EchoForCausalLM.from_pretrained(
            model_path, device_map=device, trust_remote_code=True
        )

    # Tokenizer
    tokenizer_path = (
        tokenizer
        if tokenizer
        else (base_model_path if os.path.exists(adapter_config_path) else model_path)
    )
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = 32000  # Default Echo

    # Critical Generation Settings
    # 32000: <|endoftext|>, 32001: <|assistant|>, 32006: <|system|>, 32007: <|end|>, 32010: <|user|>
    eos_token_ids = [32000, 32001, 32006, 32007, 32010]
    stop_strings = ["<|endoftext|>", "<|user|>", "<|assistant|>", "<|system|>", "<|end|>"]
    tokenizer_stop = StringStoppingCriteria(tokenizer, stop_strings)

    # Chat History
    messages = []
    messages.append({"role": "system", "content": system_prompt})

    def generate(current_messages):
        # --- CONTEXT WINDOW MANAGEMENT (Dynamic Truncation) ---
        # Keep SYSTEM_PROMPT + the most recent 10 turns.
        # This prevents instruction-drift in long chat sessions.
        # if len(current_messages) > 11:
        #    current_messages = [current_messages[0]] + current_messages[-10:]

        # Build prompt from History
        if chat:
            inputs = tokenizer.apply_chat_template(
                current_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device)
        else:
            prompt = current_messages[-1]["content"]
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

        input_len = inputs["input_ids"].shape[1]
        print(f"(Context Len: {input_len} tokens)", flush=True)

        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        print("\nEcho:", end=" ", flush=True)
        time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                top_k=50,  # Added to prune the noise tail
                repetition_penalty=1.2,
                # no_repeat_ngram_size=3,  # Added to stop word salad loops
                eos_token_id=eos_token_ids,
                pad_token_id=tokenizer.pad_token_id,
                streamer=streamer,
                stopping_criteria=StoppingCriteriaList([tokenizer_stop]),
                use_cache=True,
            )

        # Decode and Save Response
        output_ids = outputs[0][input_len:]
        response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        # Cleanup response
        for s in stop_strings:
            response = response.split(s)[0]

        print("")
        return response

    if text:
        messages.append({"role": "user", "content": text})
        generate(messages)
    else:
        print(f"Starting interactive chat with {model_path}...")
        print("Type 'exit' or 'quit' to stop.")
        print("Type 'reset' to clear conversation history.")

        while True:
            try:
                user_input = input("\nYou: ")
                if user_input.lower() in ["exit", "quit"]:
                    break
                elif user_input.lower() == "reset":
                    messages = [{"role": "system", "content": system_prompt}]
                    print("Memory cleared.")
                    continue

                messages.append({"role": "user", "content": user_input})
                response = generate(messages)
                messages.append({"role": "assistant", "content": response})

            except KeyboardInterrupt:
                break

    print("Goodbye!")


if __name__ == "__main__":
    main()
