"""Exports good past chats from your PostgreSQL database as training examples.

The original app saved every question and AI answer. Those answers already follow the
post and plan format, so they are useful examples for Mistral. Run this on your PC
(where the database is), then upload training/data/db_export.jsonl with the project.

Usage (from the project folder, with the backend's virtual environment active):
  python training/export_from_db.py
  python training/export_from_db.py --limit 2000
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.db.database import SessionLocal  # noqa: E402
from app.models.models import Conversation, Message  # noqa: E402

GOOD_REPLY = re.compile(r"Caption\s*:|^\s*1\.\s*Business Understanding", re.IGNORECASE | re.MULTILINE)
BAD_REPLY = re.compile(r"openai|as an ai|i cannot|i can't help|error|try again", re.IGNORECASE)


def clean(text: str) -> str:
    text = re.sub(r"\*\*|__|`", "", text or "")
    text = re.sub(r"^\s*#{1,6}\s+", "", text, flags=re.MULTILINE)
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "training/data/db_export.jsonl"))
    parser.add_argument("--limit", type=int, default=1500)
    parser.add_argument("--history", type=int, default=2, help="earlier messages to keep as context")
    args = parser.parse_args()

    db = SessionLocal()
    written = 0
    seen = set()

    try:
        conversations = db.query(Conversation).order_by(Conversation.created_at.desc()).all()

        with open(args.output, "w", encoding="utf-8") as handle:
            for conversation in conversations:
                messages = (
                    db.query(Message)
                    .filter(Message.conversation_id == conversation.id)
                    .order_by(Message.created_at.asc())
                    .all()
                )

                for index, message in enumerate(messages):
                    if message.role != "ai" or index == 0 or messages[index - 1].role != "user":
                        continue

                    question = (messages[index - 1].content or "").strip()
                    answer = clean(message.content)

                    if len(question) < 3 or len(answer) < 120:
                        continue

                    if not GOOD_REPLY.search(answer) or BAD_REPLY.search(answer[:300]):
                        continue

                    key = question.lower()

                    if key in seen:
                        continue

                    seen.add(key)

                    history = [
                        {
                            "role": "assistant" if earlier.role == "ai" else "user",
                            "content": clean(earlier.content),
                        }
                        for earlier in messages[max(0, index - 1 - args.history): index - 1]
                    ]

                    row = {"task": "chat", "message": question, "output": answer}

                    if history:
                        row["history"] = history

                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1

                    if written >= args.limit:
                        break

                if written >= args.limit:
                    break
    finally:
        db.close()

    print(f"Exported {written} chat examples to {args.output}")
    print("Check the file and remove anything private before uploading it.")


if __name__ == "__main__":
    main()
