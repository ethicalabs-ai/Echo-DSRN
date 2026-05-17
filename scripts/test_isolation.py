import torch
from transformers import AutoTokenizer

from echo_hybrid.modeling_hybrid import HybridEchoForCausalLM

MODEL_PATH = "models/Kurtis-EON1-Hybrid-0.7B-v0.1.1"


def test_isolation():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = HybridEchoForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    # TEST A: Exact Training Condition (User only, no template overhead)
    prompt_text = "Who are you, and what is your purpose?"
    msgs = [{"role": "user", "content": prompt_text}]

    # Using the exact same template logic as training
    inputs = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to(model.device)

    print("\n--- TEST A: Natural Response (Training Style) ---")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64, temperature=0.7, do_sample=True)
    print(tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True))

    # TEST B: Logic Stress
    prompt_text = "What is 2+2?"
    msgs = [{"role": "user", "content": prompt_text}]
    inputs = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to(model.device)

    print("\n--- TEST B: Logic (No System Prompt) ---")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64, temperature=0.7, do_sample=True)
    print(tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True))


if __name__ == "__main__":
    test_isolation()
