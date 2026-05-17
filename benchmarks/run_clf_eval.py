import argparse

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate EchoForSequenceClassification on MASSIVE"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path or name of the EchoForSequenceClassification model",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for evaluation. Supports full batching.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="eliasalbouzidi/NSFW-Safe-Dataset",
        help="Dataset to evaluate on",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading tokenizer and model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Right-padding is strictly required for EchoForSequenceClassification batching!
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ).to(device)
    model.eval()

    print(f"Loading {args.dataset} test set...")
    ds = load_dataset(args.dataset, split="test")

    correct = 0
    total = len(ds)

    print(f"Evaluating on {total} samples...")

    batch_size = args.batch_size
    for i in tqdm(range(0, total, batch_size), desc="Evaluating"):
        batch = ds[i : i + batch_size]
        texts = batch["text"]
        labels = torch.tensor(batch["labels"], dtype=torch.long, device=device)

        # Apply the chat template the adapter was trained on
        sys_prompt = getattr(model.config, "system_prompt", None)
        usr_template = getattr(model.config, "user_template", None)

        formatted_texts = []
        for utt in texts:
            if sys_prompt and usr_template:
                messages = [
                    {"role": "system", "content": sys_prompt},
                    {
                        "role": "user",
                        "content": usr_template.format(text=utt),
                    },
                ]
                formatted = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
            else:
                formatted = utt
            formatted_texts.append(formatted)

        # Tokenize inputs. Right padding is supported perfectly by EchoForSequenceClassification.
        inputs = tokenizer(
            formatted_texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(device)

        with torch.inference_mode():
            outputs = model(**inputs)
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()

    accuracy = correct / total
    print("\n" + "=" * 50)
    print("RESULTS:")
    print("=" * 50)
    print(f"Model:    {args.model}")
    print(f"Dataset:  {args.dataset}")
    print(f"Samples:  {total}")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print("=" * 50)


if __name__ == "__main__":
    main()
