"""QLoRA fine-tuning of Mistral 7B Instruct on the marketing dataset.

Run build_dataset.py first. On a Colab T4 (16 GB) this takes about 15 to 40 minutes,
depending on the number of examples. Only the small LoRA adapter (~80 MB) is saved;
the model server loads it on top of the base model.

Usage:
  python training/train_mistral_lora.py
  python training/train_mistral_lora.py --epochs 2 --model mistralai/Mistral-7B-Instruct-v0.2
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model_server"))

from chat_format import dtype_arg, ensure_chat_template, normalize_messages  # noqa: E402


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []

    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def common_prefix(a: list[int], b: list[int]) -> int:
    size = 0

    for x, y in zip(a, b):
        if x != y:
            break
        size += 1

    return size


class ChatDataset(Dataset):
    """Tokenizes each chat; only the final assistant answer is trained on."""

    def __init__(self, rows: list[dict], tokenizer, max_len: int):
        self.items = []
        skipped = 0

        for row in rows:
            turns = normalize_messages(row["messages"])

            if len(turns) < 2 or turns[-1]["role"] != "assistant":
                skipped += 1
                continue

            prompt = tokenizer.apply_chat_template(turns[:-1], tokenize=False, add_generation_prompt=True)
            full = tokenizer.apply_chat_template(turns, tokenize=False)

            prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
            full_ids = tokenizer(full, add_special_tokens=False).input_ids

            if full_ids[-1] != tokenizer.eos_token_id:
                full_ids.append(tokenizer.eos_token_id)

            if len(full_ids) > max_len:
                skipped += 1
                continue

            start = common_prefix(prompt_ids, full_ids)
            labels = [-100] * start + full_ids[start:]

            self.items.append({"input_ids": full_ids, "labels": labels})

        if skipped:
            print(f"Skipped {skipped} examples (too long or malformed)")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class PadCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        width = max(len(item["input_ids"]) for item in batch)

        def pad(values, fill):
            return [list(v) + [fill] * (width - len(v)) for v in values]

        input_ids = torch.tensor(pad([b["input_ids"] for b in batch], self.pad_id))
        labels = torch.tensor(pad([b["labels"] for b in batch], -100))
        mask = torch.tensor(pad([[1] * len(b["input_ids"]) for b in batch], 0))

        return {"input_ids": input_ids, "labels": labels, "attention_mask": mask}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--train", default=str(ROOT / "training/data/train.jsonl"))
    parser.add_argument("--eval", default=str(ROOT / "training/data/eval.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "training/output/mistral-marketing-lora"))
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported()

    tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.hf_token)
    ensure_chat_template(tokenizer)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_set = ChatDataset(load_rows(Path(args.train)), tokenizer, args.max_len)
    eval_rows = load_rows(Path(args.eval))
    eval_set = ChatDataset(eval_rows, tokenizer, args.max_len) if eval_rows else None

    if len(train_set) == 0:
        sys.exit("No training examples. Run: python training/build_dataset.py")

    print(f"Training on {len(train_set)} examples" + (f", evaluating on {len(eval_set)}" if eval_set else ""))

    if cuda:
        from transformers import BitsAndBytesConfig

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            token=args.hf_token,
            device_map={"": 0},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
                bnb_4bit_use_double_quant=True,
            ),
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        # CPU is only useful for a quick smoke test with a tiny model.
        model = AutoModelForCausalLM.from_pretrained(args.model, token=args.hf_token, **dtype_arg(torch.float32))

    model.config.use_cache = False

    model = get_peft_model(
        model,
        LoraConfig(
            r=args.rank,
            lora_alpha=args.alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    model.print_trainable_parameters()

    total_steps = math.ceil(len(train_set) / (args.batch * args.grad_accum) * args.epochs)
    warmup_steps = max(1, round(total_steps * 0.05))

    training_args = TrainingArguments(
        output_dir=str(Path(args.output) / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        # A step count works on transformers 4.x and 5.x (5.x removed warmup_ratio).
        warmup_steps=warmup_steps,
        weight_decay=0.0,
        max_grad_norm=0.3,
        logging_steps=5,
        save_strategy="no",
        eval_strategy="epoch" if eval_set else "no",
        bf16=bf16,
        fp16=cuda and not bf16,
        optim="paged_adamw_8bit" if cuda else "adamw_torch",
        gradient_checkpointing=cuda,
        gradient_checkpointing_kwargs={"use_reentrant": False} if cuda else None,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=cuda,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=PadCollator(tokenizer.pad_token_id),
    )

    started = time.time()
    result = trainer.train()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))

    info = {
        "base_model": args.model,
        "examples": len(train_set),
        "epochs": args.epochs,
        "rank": args.rank,
        "train_loss": round(result.training_loss, 4),
        "minutes": round((time.time() - started) / 60, 1),
    }

    if eval_set:
        info["eval_loss"] = round(trainer.evaluate()["eval_loss"], 4)

    (output / "training_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    print("Saved adapter to", output)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
