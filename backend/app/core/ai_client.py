"""Marketing AI features on top of the GPU model server.

Mistral 7B writes the text (captions, descriptions, hashtags, plans) and Stable
Diffusion 1.5 draws the images. A 7B model follows labelled lines ("Caption: ...")
much more reliably than JSON, so every structured answer uses that format and is
parsed here, with retries when something is missing.
"""

import re

from fastapi import HTTPException, status

from app.core import model_server
from app.core.prompts import (
    IMAGE_QUALITY_TAGS,
    NL,
    build_chat_messages,
    build_enhance_messages,
    build_memory_messages,
    build_single_post_messages,
    build_times_messages,
    build_week_messages,
)

# Most posts are 110 to 150 tokens, so batches of 4 stay well inside the tunnel timeout.
WEEK_BATCH = 4
TOKENS_PER_POST = 170

PLAN_WORDS = re.compile(r"\b(plan|strategy|strategies|campaign|roadmap)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _clean(value: str) -> str:
    value = re.sub(r"\*\*|__|`", "", value)
    return re.sub(r"\s+", " ", value).strip().strip('"“”').strip()


def _strip_markdown(text: str) -> str:
    """The prompt asks for plain text; this removes what slips through anyway."""

    text = re.sub(r"\*\*|__|`", "", text)
    # "## Heading" -> "Heading" (hashtags have no space after the #, so they stay)
    return re.sub(r"^\s*#{1,6}\s+", "", text, flags=re.MULTILINE).strip()


POST_HEADER = re.compile(r"^\s*\**\s*Post\s*\d+\s*\**\s*[:.)\-]?", re.IGNORECASE | re.MULTILINE)


def _field(chunk: str, label: str, following: list[str]) -> str:
    stop = (
        r"(?=\n\s*[-*]?\s*\**(?:" + "|".join(following) + r")\**\s*:|\Z)"
        if following
        else r"\Z"
    )

    match = re.search(
        r"\**" + label + r"\**\s*:\s*([\s\S]*?)" + stop,
        chunk,
        re.IGNORECASE,
    )

    return _clean(match.group(1)) if match else ""


def clean_hashtags(value: str) -> str:
    tags = re.findall(r"#[^\s#,]+", value)

    if not tags:
        tags = ["#" + word for word in re.split(r"[,\s]+", value) if word.strip("#")]

    result: list[str] = []
    seen = set()

    for tag in tags:
        tag = "#" + re.sub(r"[^\w]", "", tag.lstrip("#"))

        if len(tag) < 3 or tag.lower().startswith("#your"):
            continue

        if tag.lower() not in seen:
            seen.add(tag.lower())
            result.append(tag)

    return " ".join(result[:8])


def parse_posts(text: str) -> list[dict]:
    """Reads "Post N / Caption / Description / Hashtags / Image prompt" blocks."""

    text = str(text or "").replace("\r", "")
    chunks = POST_HEADER.split(text)

    posts = []

    for chunk in chunks:
        caption = _field(chunk, "Caption", ["Description", "Hashtags?", "Image prompt"])

        if not caption:
            continue

        description = _field(chunk, "Description", ["Hashtags?", "Image prompt"])
        hashtags = clean_hashtags(_field(chunk, "Hashtags?", ["Image prompt", "Post"]))
        image_prompt = _field(chunk, "Image prompt", ["Post", "Caption"])

        posts.append(
            {
                "caption": caption,
                "description": description,
                "hashtags": hashtags,
                "image_prompt": image_prompt or caption,
            }
        )

    return posts


# ---------------------------------------------------------------------------
# Chat: posts, plans and follow-ups
# ---------------------------------------------------------------------------

def generate_marketing_ai_reply(
    user_message: str,
    business_name: str = "your business",
    industry: str | None = None,
    history: list[dict] | None = None,
    memory: str | None = None,
) -> str:
    messages = build_chat_messages(user_message, business_name, industry, history, memory)

    # A full plan is about twice as long as three posts.
    max_tokens = 1100 if PLAN_WORDS.search(user_message) else 700

    reply = _strip_markdown(model_server.chat(messages, max_new_tokens=max_tokens, temperature=0.7))

    if not reply:
        # One retry with a calmer setting before giving up.
        reply = _strip_markdown(model_server.chat(messages, max_new_tokens=max_tokens, temperature=0.5))

    if not reply:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The AI returned an empty reply. Please try again.",
        )

    return reply


# ---------------------------------------------------------------------------
# Image generation (Stable Diffusion 1.5)
# ---------------------------------------------------------------------------

IMAGE_STYLE_SUFFIX = (
    ", realistic commercial photograph, soft natural light, warm beige and deep maroon "
    "accents, clean composition, " + IMAGE_QUALITY_TAGS
)


def enhance_image_prompt(idea: str, memory: str | None = None) -> str:
    """Turns a short idea into a short, SD 1.5 friendly prompt. Falls back to a style suffix."""

    try:
        text = model_server.chat(
            build_enhance_messages(idea, memory),
            max_new_tokens=110,
            temperature=0.6,
        )

        text = text.strip().split(NL + NL)[0]
        text = re.sub(r"^\s*(prompt|image prompt)\s*:\s*", "", text, flags=re.IGNORECASE)
        text = _clean(text)

        if len(text) >= 20:
            return text[:600].rstrip(" ,.") + ", " + IMAGE_QUALITY_TAGS
    except HTTPException as exc:
        # If the server is offline the image call would fail too, so stop here.
        if exc.status_code in (status.HTTP_503_SERVICE_UNAVAILABLE, status.HTTP_502_BAD_GATEWAY):
            raise

        print("Image prompt enhancement skipped:", exc.detail)

    return idea.strip().rstrip(" .") + IMAGE_STYLE_SUFFIX


def generate_image_png(prompt: str, memory: str | None = None, enhance: bool = True) -> bytes:
    final_prompt = enhance_image_prompt(prompt, memory) if enhance else prompt.strip()

    return model_server.image(final_prompt)


# ---------------------------------------------------------------------------
# Memory: a short profile of the user's business
# ---------------------------------------------------------------------------

def update_memory_summary(current: str | None, user_message: str) -> str:
    text = model_server.chat(
        build_memory_messages(current, user_message),
        max_new_tokens=200,
        temperature=0.2,
    )

    text = re.sub(r"^\s*(updated |current )?profile\s*:\s*", "", text, flags=re.IGNORECASE)
    text = _strip_markdown(text).strip()

    if text.lower().strip("(). ") == "none":
        text = ""

    return text[:900]


# ---------------------------------------------------------------------------
# Several posts for the week
# ---------------------------------------------------------------------------

def draft_week_plan(description: str, memory: str | None, days: int = 7) -> list[dict]:
    posts: list[dict] = []
    seen = set()

    # A few extra calls cover a batch that came back short.
    for _ in range(-(-days // WEEK_BATCH) + 2):
        if len(posts) >= days:
            break

        count = min(WEEK_BATCH, days - len(posts))

        text = model_server.chat(
            build_week_messages(
                description,
                memory,
                count,
                first_day=len(posts) + 1,
                avoid_captions=[post["caption"] for post in posts],
            ),
            max_new_tokens=TOKENS_PER_POST * count + 80,
            temperature=0.8,
        )

        for post in parse_posts(text)[:count]:
            key = post["caption"].lower()

            if key not in seen:
                seen.add(key)
                posts.append(post)

    if len(posts) < days:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The AI returned an incomplete plan. Please try again.",
        )

    return posts[:days]


# ---------------------------------------------------------------------------
# Suggested posting times (a starting point, not measured data)
# ---------------------------------------------------------------------------

TIME_LINE = re.compile(r"(\d{1,2})\s*[:.]\s*(\d{2})\s*(?:[ap]\.?m\.?)?\s*[-–—:]?\s*(.*)", re.IGNORECASE)


def suggest_post_times(memory: str | None) -> list[dict]:
    text = model_server.chat(build_times_messages(memory), max_new_tokens=180, temperature=0.4)

    times = []
    used = set()

    for line in text.splitlines():
        line = re.sub(r"^\s*(\d[.)]\s+|[-*•]\s*)", "", line.strip())
        match = TIME_LINE.match(line)

        if not match:
            continue

        hour, minute = int(match.group(1)), int(match.group(2))

        # "7:00 pm" -> 19:00
        if re.search(r"\d\s*p\.?m", line, re.IGNORECASE) and hour < 12:
            hour += 12

        if hour > 23 or minute > 59:
            continue

        value = f"{hour:02d}:{minute:02d}"

        if value in used:
            continue

        used.add(value)
        times.append({"time": value, "reason": _clean(match.group(3)) or "A time when customers are often online."})

    if not times:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The AI did not return usable times.",
        )

    return times[:3]


# ---------------------------------------------------------------------------
# One ready-to-publish post (used by the one-click Autopilot)
# ---------------------------------------------------------------------------

def draft_single_post(request: str, memory: str | None) -> dict:
    messages = build_single_post_messages(request, memory)

    for temperature in (0.7, 0.5):
        posts = parse_posts(
            model_server.chat(messages, max_new_tokens=260, temperature=temperature)
        )

        if posts:
            return posts[0]

    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail="The AI did not return a usable post. Please try again.",
    )
