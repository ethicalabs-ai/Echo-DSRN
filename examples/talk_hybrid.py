import click
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    StoppingCriteria,
    TextStreamer,
)

from echo_hybrid.configuration_hybrid import HybridEchoConfig
from echo_hybrid.modeling_hybrid import HybridEchoCache, HybridEchoForCausalLM

# Maximum number of *prompt* tokens allowed before the oldest exchanges are
# dropped.  Leaves ample headroom for max_tokens new tokens on a 32 k-context
# Qwen2 backbone while keeping peak VRAM predictable across many turns.
MAX_HISTORY_TOKENS = 1536


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


def trim_messages(messages, tokenizer, max_tokens: int):
    """Keep the system prompt plus the most-recent user/assistant pairs that fit
    within *max_tokens* prompt tokens.  Oldest pairs are dropped first.

    This prevents the prompt from growing without bound across many chat turns,
    which is the primary driver of the per-turn VRAM escalation (Bug 3).
    """
    system = [m for m in messages if m["role"] == "system"]
    rest = [m for m in messages if m["role"] != "system"]

    while rest:
        candidate = system + rest
        # apply_chat_template with add_generation_prompt=False to measure length
        ids = tokenizer.apply_chat_template(
            candidate,
            tokenize=True,
            add_generation_prompt=False,
        )
        if len(ids) <= max_tokens:
            break
        # Drop the oldest pair (user + assistant); always drop in pairs so the
        # conversation stays structurally valid.
        rest = rest[2:]

    trimmed = system + rest
    if len(trimmed) < len(messages):
        dropped = (len(messages) - len(trimmed)) // 2
        print(
            f"  [history trimmed: dropped {dropped} oldest exchange(s) to stay within {max_tokens} tokens]"
        )
    return trimmed


@click.command()
@click.option("--model_path", required=True, help="Path to the HF model directory.")
@click.option("--text", default=None, help="Input text for single-turn generation.")
@click.option("--chat/--no-chat", default=True, help="Use chat template for input.")
@click.option("--temperature", default=0.7, help="Sampling temperature.")
@click.option("--max_tokens", default=512, help="Max new tokens to generate.")
@click.option("--tokenizer_path", default=None, help="Tokenizer path/name (overrides model_path).")
@click.option("--system_prompt", default=SYSTEM_PROMPT, help="System prompt.")
@click.option("--seed", default=42, help="Random seed for reproducibility.")
@click.option(
    "--use-backbone-cache/--no-backbone-cache",
    default=True,
    help="Enable Qwen2 backbone KV-cache.  On = mode 2 (recommended).  Off = ablation mode (full context re-feed).",
)
@click.option(
    "--use-dsrn-cache/--no-dsrn-cache",
    default=True,
    help="Carry DSRN slow-state across generation steps.  Should almost always be on.",
)
def main(
    model_path,
    text,
    chat,
    temperature,
    max_tokens,
    tokenizer_path,
    system_prompt,
    seed,
    use_backbone_cache,
    use_dsrn_cache,
):
    """
    Run interactive chat with Echo-Hybrid model using full-context re-generation.
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
    # Register our custom config for AutoConfig
    AutoConfig.register("echo_hybrid", HybridEchoConfig)

    # Load Hybrid Model
    model = HybridEchoForCausalLM.from_pretrained(
        model_path, device_map=device, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    model.eval()

    # Override config cache flags from CLI so the session is fully controlled
    # by what was passed on the command line, not by whatever the checkpoint saved.
    model.config.use_kv_cache = use_backbone_cache
    print(
        f"  backbone KV-cache: {'ON' if use_backbone_cache else 'OFF (ablation)'} | "
        f"DSRN state: {'ON' if use_dsrn_cache else 'OFF'}"
    )

    # Tokenizer
    t_path = tokenizer_path if tokenizer_path else model_path
    print(f"Loading tokenizer from {t_path}...")
    tokenizer = AutoTokenizer.from_pretrained(t_path, trust_remote_code=True)

    # Reset peak VRAM counter here so the summary reflects only inference
    # overhead, not the one-time model load cost.
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Qwen2 special tokens
    # <|im_start|>, <|im_end|>, <|endoftext|>
    stop_strings = ["<|im_end|>", "<|endoftext|>", "<|im_start|>", "<|user|>", "<|assistant|>"]
    StringStoppingCriteria(tokenizer, stop_strings)

    # Chat History
    messages = []
    messages.append({"role": "system", "content": system_prompt})

    def generate(current_messages):
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

        TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        print("\nEcho:", end=" ", flush=True)

        with torch.no_grad():
            base_ids = inputs["input_ids"]  # full prompt
            generated_tokens = []
            ablation_mode = not use_backbone_cache

            if ablation_mode:
                # ── ABLATION MODE (use_kv_cache=False) ────────────────────────
                # Attention has NO KV history — each forward must see the full
                # growing context so self-attention is meaningful.
                # DSRN slow-state is carried separately via seen_tokens + dsrn_states.
                ctx_ids = base_ids  # grows each step
                dsrn_states = []  # will be populated after first forward

                for _ in range(max_tokens):
                    if dsrn_states:
                        # Re-feed full context with DSRN states from previous step
                        carry = HybridEchoCache.from_legacy_cache(dsrn_states)
                        carry.seen_tokens = 0  # full re-feed starts at position 0
                        out = model(ctx_ids, past_key_values=carry, use_cache=use_dsrn_cache)
                    else:
                        out = model(ctx_ids, past_key_values=None, use_cache=use_dsrn_cache)

                    # ── FIX (Bug 1+2): extract DSRN states and logits FIRST, then
                    # immediately drop the output and the KV cache reference.
                    # In ablation mode the KV tensors are NOT needed between steps
                    # (ctx_ids is always re-fed in full), so holding pkv alive wastes
                    # O(layers × seq_len × head_dim) VRAM every step.
                    pkv = out.past_key_values
                    dsrn_states = pkv.dsrn_states if pkv else []
                    logits = out.logits[:, -1, :].float()
                    del out  # release the full forward-pass output immediately
                    del pkv  # release KV buffer; next step will allocate a fresh one

                    # Temperature + top-p sampling (avoids repetition loops)
                    logits = logits / max(temperature, 1e-6)
                    probs = torch.softmax(logits, dim=-1)
                    # Top-p (nucleus) filtering
                    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
                    cumsum = torch.cumsum(sorted_probs, dim=-1)
                    top_p = 0.9
                    sorted_probs[cumsum - sorted_probs > top_p] = 0.0
                    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
                    sampled = torch.multinomial(sorted_probs, num_samples=1)
                    next_token = sorted_idx.gather(-1, sampled)
                    token_id = next_token.item()

                    if token_id == tokenizer.eos_token_id or token_id == 151645:
                        break

                    generated_tokens.append(token_id)
                    word = tokenizer.decode([token_id])
                    print(word, end="", flush=True)

                    # Grow the context window with the new token
                    ctx_ids = torch.cat([ctx_ids, next_token], dim=1)

                    # ── FIX (Bug 4): decode only the tail to check stop strings.
                    # Decoding generated_tokens[0:N] every step is O(N²) CPU work.
                    # Stop strings are short tokens — checking the last 12 ids is
                    # always sufficient.
                    tail_text = tokenizer.decode(generated_tokens[-12:], skip_special_tokens=False)
                    if any(tail_text.strip().endswith(s) for s in stop_strings):
                        break

            else:
                # ── CACHED MODE (use_kv_cache=True) ───────────────────────────
                # Standard single-token autoregressive generation with KV cache.
                curr_ids = base_ids
                pkv = None

                for _ in range(max_tokens):
                    out = model(curr_ids, past_key_values=pkv, use_cache=use_dsrn_cache)
                    pkv = out.past_key_values

                    # Temperature + top-p sampling
                    logits = out.logits[:, -1, :].float()
                    del out  # release forward output; pkv already captured above
                    logits = logits / max(temperature, 1e-6)
                    probs = torch.softmax(logits, dim=-1)
                    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
                    cumsum = torch.cumsum(sorted_probs, dim=-1)
                    top_p = 0.9
                    sorted_probs[cumsum - sorted_probs > top_p] = 0.0
                    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
                    sampled = torch.multinomial(sorted_probs, num_samples=1)
                    next_token = sorted_idx.gather(-1, sampled)
                    token_id = next_token.item()

                    if token_id == tokenizer.eos_token_id or token_id == 151645:
                        break

                    generated_tokens.append(token_id)
                    word = tokenizer.decode([token_id])
                    print(word, end="", flush=True)

                    curr_ids = next_token

                    # ── FIX (Bug 4): decode only the tail to check stop strings.
                    tail_text = tokenizer.decode(generated_tokens[-12:], skip_special_tokens=False)
                    if any(tail_text.strip().endswith(s) for s in stop_strings):
                        break

        # Decode and Save Response
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

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
                if not user_input.strip():
                    continue
                if user_input.lower() in ["exit", "quit"]:
                    break
                elif user_input.lower() == "reset":
                    messages = [{"role": "system", "content": system_prompt}]
                    print("Memory cleared.")
                    continue

                messages.append({"role": "user", "content": user_input})
                # ── FIX (Bug 3): trim history to MAX_HISTORY_TOKENS before every
                # generate() call.  Without this the prompt grows turn-by-turn,
                # widening the self-attention matrix quadratically in ablation mode.
                messages = trim_messages(messages, tokenizer, MAX_HISTORY_TOKENS)
                response = generate(messages)
                messages.append({"role": "assistant", "content": response})

            except KeyboardInterrupt:
                break

    if torch.cuda.is_available():
        cur = torch.cuda.memory_allocated() / 1024**2
        peak = torch.cuda.max_memory_allocated() / 1024**2
        res = torch.cuda.memory_reserved() / 1024**2
        print(
            f"\n── VRAM Summary ────────────────────────────────────────────────\n"
            f"  Current   : {cur:>8.1f} MB\n"
            f"  Peak      : {peak:>8.1f} MB   (inference only, model weights excluded)\n"
            f"  Reserved  : {res:>8.1f} MB   (total cache held by allocator)\n"
            f"─────────────────────────────────────────────────────────────"
        )
    print("Goodbye!")


if __name__ == "__main__":
    main()
