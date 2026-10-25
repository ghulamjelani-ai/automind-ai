"""Builds the Mistral fine-tuning dataset.

Every example is rendered with the same prompt builders the backend uses
(backend/app/core/prompts.py), so the model learns on exactly what it will see.

Sources (all optional except the seed file):
  training/data/seed_examples.jsonl   hand-written examples that ship with the project
  training/data/db_export.jsonl       your real chats, from export_from_db.py
  training/data/custom/*.jsonl        any extra examples you write

Usage:  python training/build_dataset.py
Output: training/data/train.jsonl and training/data/eval.jsonl
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "training" / "data"

sys.path.insert(0, str(ROOT / "backend"))

from app.core.prompts import (  # noqa: E402
    build_chat_messages,
    build_enhance_messages,
    build_memory_messages,
    build_single_post_messages,
    build_times_messages,
    build_week_messages,
)


def render(example: dict) -> list[dict]:
    task = example.get("task", "chat")
    memory = example.get("memory")

    if task == "chat":
        messages = build_chat_messages(
            example["message"],
            example.get("business_name"),
            example.get("industry"),
            example.get("history"),
            memory,
        )
    elif task == "single_post":
        messages = build_single_post_messages(example["request"], memory)
    elif task == "week":
        messages = build_week_messages(
            example.get("description", ""),
            memory,
            int(example["count"]),
            first_day=int(example.get("first_day", 1)),
            avoid_captions=example.get("avoid_captions"),
        )
    elif task == "times":
        messages = build_times_messages(memory)
    elif task == "memory":
        messages = build_memory_messages(example.get("current"), example["message"])
    elif task == "enhance":
        messages = build_enhance_messages(example["idea"], memory)
    else:
        raise ValueError(f"Unknown task: {task}")

    output = str(example["output"]).strip()

    if not output:
        raise ValueError("Empty output")

    return messages + [{"role": "assistant", "content": output}]


def read_jsonl(path: Path) -> list[dict]:
    rows = []

    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"Skipping {path.name} line {number}: {exc}")

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-share", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    # Seed examples are few but carefully written, so they are repeated to weigh more.
    parser.add_argument("--seed-repeat", type=int, default=2)
    args = parser.parse_args()

    sources = [(DATA / "seed_examples.jsonl", args.seed_repeat), (DATA / "db_export.jsonl", 1)]
    sources += [(path, 1) for path in sorted((DATA / "custom").glob("*.jsonl"))]

    examples = []
    counts = {}

    for path, repeat in sources:
        if not path.exists():
            continue

        rendered = []

        for row in read_jsonl(path):
            try:
                rendered.append({"task": row.get("task", "chat"), "messages": render(row)})
            except (KeyError, ValueError, TypeError) as exc:
                print(f"Skipping an example in {path.name}: {exc!r}")

        counts[path.name] = len(rendered)
        examples.extend(rendered * repeat)

    if not examples:
        sys.exit("No training examples found in training/data.")

    random.Random(args.seed).shuffle(examples)

    # The eval set comes only from unique examples so it is not a copy of training rows.
    eval_size = int(len(examples) * args.eval_share) if len(examples) >= 40 else 0
    keys = [json.dumps(example["messages"], ensure_ascii=False) for example in examples]
    repeats = Counter(keys)
    eval_rows, train_rows = [], []

    for key, example in zip(keys, examples):
        if len(eval_rows) < eval_size and repeats[key] == 1:
            eval_rows.append(example)
        else:
            train_rows.append(example)

    for name, rows in (("train.jsonl", train_rows), ("eval.jsonl", eval_rows)):
        with (DATA / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    tasks = {}

    for row in train_rows:
        tasks[row["task"]] = tasks.get(row["task"], 0) + 1

    print("Sources:", counts)
    print(f"Wrote {len(train_rows)} training and {len(eval_rows)} eval examples")
    print("Training examples per task:", tasks)


if __name__ == "__main__":
    main()
