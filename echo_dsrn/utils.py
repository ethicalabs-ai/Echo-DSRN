def visualize_masked_samples(dataset, tokenizer, num_samples=2):
    """
    Prints tokenized samples and color-codes tokens based on whether they are masked for loss.
    """
    # ANSI escape codes
    RESET = '\033[0m'
    MASKED = '\033[90m\033[9m'  # Dark gray + strikethrough
    LOSS = '\033[92m\033[1m'  # Bright green + bold

    print("\n" + "=" * 80)
    print(f"🔍 PREVIEWING TOKENIZED DATASET MASKS ({min(num_samples, len(dataset))} samples)")
    print(
        f"Legend: {MASKED}Gray Strikethrough{RESET} = Masked (prompt, ignored), {LOSS}Bright Green{RESET} = Loss Calculated (completion)"
    )
    print("=" * 80 + "\n")

    for i in range(min(num_samples, len(dataset))):
        row = dataset[i]
        ids = row['input_ids']
        labels = row['labels']

        out = []
        span_ids = []
        # Fallback if somehow empty
        if not labels:
            continue

        span_masked = labels[0] == -100

        for tid, lbl in zip(ids, labels):
            is_masked = lbl == -100
            if is_masked == span_masked:
                span_ids.append(tid)
            else:
                text = tokenizer.decode(span_ids, skip_special_tokens=False)
                color = MASKED if span_masked else LOSS
                # Ensure newlines don't break the ANSI formatting
                text = text.replace('\n', f'{RESET}\n{color}')
                out.append(f"{color}{text}{RESET}")
                span_ids = [tid]
                span_masked = is_masked

        if span_ids:
            text = tokenizer.decode(span_ids, skip_special_tokens=False)
            color = MASKED if span_masked else LOSS
            text = text.replace('\n', f'{RESET}\n{color}')
            out.append(f"{color}{text}{RESET}")

        print(f"--- Sample {i} ({len(ids)} tokens, {labels.count(-100)} masked) ---")
        print("".join(out))
        print("\n" + "-" * 80 + "\n")
