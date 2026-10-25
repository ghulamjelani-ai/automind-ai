"""Turns chat messages into the shape Mistral's chat template accepts.

Mistral Instruct has no system role and needs strict user/assistant turns that start
with the user. Training and serving both use this function, so the fine-tuned model
sees exactly the same prompt layout in both places.
"""

NL = chr(10)

# Mistral Instruct layout, used only when a tokenizer has no chat template of its own
# (for example a merged model saved without one).
MISTRAL_TEMPLATE = (
    "{{ bos_token }}{% for message in messages %}"
    "{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}"
    "{% else %}{{ ' ' + message['content'] + eos_token }}{% endif %}"
    "{% endfor %}"
)


def ensure_chat_template(tokenizer) -> None:
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = MISTRAL_TEMPLATE


def dtype_arg(dtype) -> dict:
    """from_pretrained calls the argument "dtype" since transformers 4.56 and "torch_dtype" before."""

    import transformers

    major, minor = (int(part) for part in transformers.__version__.split(".")[:2])

    return {"dtype": dtype} if (major, minor) >= (4, 56) else {"torch_dtype": dtype}


def normalize_messages(messages: list[dict], drop_pairs: int = 0) -> list[dict]:
    """Merges the system text into the first user turn and fixes the turn order.

    drop_pairs removes that many of the oldest history turns (used when the prompt is
    too long), while the system text and the latest message are always kept.
    """

    system_parts = []
    turns: list[dict] = []

    for message in messages:
        role = str(message.get("role", "user")).strip().lower()
        content = str(message.get("content") or "").strip()

        if not content:
            continue

        if role == "system":
            system_parts.append(content)
            continue

        role = "assistant" if role in ("assistant", "ai", "model") else "user"

        # Two turns in a row from the same side are joined into one.
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"] += NL + NL + content
        else:
            turns.append({"role": role, "content": content})

    # The conversation must start with the user.
    while turns and turns[0]["role"] != "user":
        turns.pop(0)

    if drop_pairs > 0 and len(turns) > 1:
        keep_from = min(drop_pairs * 2, len(turns) - 1)

        # Never start on an assistant turn after dropping.
        while keep_from < len(turns) - 1 and turns[keep_from]["role"] != "user":
            keep_from += 1

        turns = turns[keep_from:]

    if not turns:
        turns = [{"role": "user", "content": "Hello"}]

    if system_parts:
        system = (NL + NL).join(system_parts)
        turns[0] = {
            "role": "user",
            "content": system + NL + NL + "---" + NL + NL + turns[0]["content"],
        }

    return turns
