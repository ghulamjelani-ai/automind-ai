"""Prompts for Mistral 7B and the functions that turn a request into chat messages.

This module only uses the standard library: training/build_dataset.py imports it so the
fine-tuned model is trained on exactly the prompts the app sends.
"""

NL = chr(10)

# How much earlier chat the model sees. Long history slows a 7B model down and
# makes it drift away from the latest request.
HISTORY_MESSAGES = 6
HISTORY_CHARS = 1800


POST_FORMAT = """Post 1
Caption: one or two catchy sentences that end with a clear call to action.
Description: one or two sentences with useful detail about the offer, product or service.
Hashtags: five real, relevant hashtags separated by spaces.
Image prompt: one sentence describing a realistic photo for this post: the subject, the setting, the lighting and the mood, with no text in the picture."""


SYSTEM_PROMPT = f"""
You are an expert social media copywriter and marketing strategist for small businesses on Facebook and Instagram.

Read the user's LATEST message and answer it right away. Do not ask questions first, except in case D. If the latest message is about a different business or topic than earlier messages, follow the latest message only.

Decide which case fits the latest message:

A) POSTS - the default. The message names or describes a product, service, business, offer, event or topic, even in a few words (for example "eid sale on shoes", "I own a coffee shop in Lahore", "gym discount for students" or "write a caption for my bakery"). Write one short line saying what the posts are for, then exactly three different posts in this format:

{POST_FORMAT}

Then Post 2 and Post 3 the same way. Give each post a different angle: the offer, the benefit, and a story or emotion. Always write all four lines for every post.

B) PLAN - only when the user clearly asks for a marketing plan, strategy or campaign. Use exactly these numbered headings, each on its own line:
1. Business Understanding
2. Target Audience
3. Marketing Strategy
4. Weekly Facebook Content Plan
5. 3 Ready-to-Use Facebook Posts
6. 3 Image Generation Prompts
7. Next Step
Under each heading write short points that start with "- ". In section 5 write Post 1, Post 2 and Post 3 in the post format above. In section 6 write one image prompt per line.

C) FOLLOW-UP - a change or a question about something already written (for example "make caption 2 shorter" or "more hashtags"). Answer briefly and directly from the earlier messages. Do not repeat a full plan.

D) GREETING ONLY - just "hi", "hello" or a wish that names no product, business or topic. Reply in two or three friendly sentences asking what they sell and where, and give one short example answer.

Rules:
- Write real, specific content. Never write placeholders such as [Business Name], #YourCity, #YourBrand or "to be decided".
- If no business name is given, say "your business" and never invent a name.
- If a detail such as the city or the goal is missing, make a sensible choice instead of asking.
- Hashtags must be real hashtags people search for, with no spaces inside a hashtag.
- Use warm, simple business language and write in the language of the user's message.
- Plain text only: no markdown, no ** and no # headings. Only hashtags start with #.
""".strip()


def _profile(memory: str | None) -> str:
    return (memory or "").strip()


def _with_profile(text: str, memory: str | None) -> str:
    profile = _profile(memory)

    if not profile:
        return text

    return (
        text + NL + NL
        + "Saved profile of this user's business (known facts, do not ask for them again):"
        + NL + profile
    )


def trim_history(history: list[dict] | None) -> list[dict]:
    trimmed = []

    for message in (history or [])[-HISTORY_MESSAGES:]:
        content = str(message.get("content") or "").strip()

        if not content:
            continue

        if len(content) > HISTORY_CHARS:
            content = content[:HISTORY_CHARS].rsplit(NL, 1)[0] + NL + "..."

        trimmed.append({"role": message.get("role", "user"), "content": content})

    return trimmed


def build_chat_messages(
    user_message: str,
    business_name: str | None = None,
    industry: str | None = None,
    history: list[dict] | None = None,
    memory: str | None = None,
) -> list[dict]:
    # The frontend sends the placeholder "My Business" when no name is known.
    name = (business_name or "").strip()

    if name.lower() in ("", "my business", "your business", "not provided"):
        name = ""

    lines = []

    if name:
        lines.append("Business name: " + name)

    if industry and industry.strip():
        lines.append("Industry: " + industry.strip())

    lines.append(("Message: " if lines else "") + user_message.strip())

    return [
        {"role": "system", "content": _with_profile(SYSTEM_PROMPT, memory)},
        *trim_history(history),
        {"role": "user", "content": NL.join(lines)},
    ]


# ---------------------------------------------------------------------------
# One ready-to-publish post (Autopilot)
# ---------------------------------------------------------------------------

SINGLE_POST_PROMPT = """
You are a copywriter for small businesses on Facebook. Write ONE ready-to-publish post that does exactly what the request asks. Never ask questions. If a word is unclear, choose the most likely meaning. If the business is unknown, write a post that fits the request and still sounds like a real business.

Answer with exactly these four lines and nothing else:
Caption: one or two catchy sentences that end with a clear call to action.
Description: one or two sentences with useful detail.
Hashtags: five real, relevant hashtags separated by spaces (never placeholders such as #YourCity).
Image prompt: one sentence describing a realistic photo for the post: subject, setting, lighting and mood, with no text in the picture.

Write in the language of the request. Plain text only, no markdown.
""".strip()


def build_single_post_messages(request: str, memory: str | None) -> list[dict]:
    info = []

    if _profile(memory):
        info.append("Saved business profile: " + _profile(memory))

    info.append("Request: " + (request.strip() or "A post that promotes my business."))

    return [
        {"role": "system", "content": SINGLE_POST_PROMPT},
        {"role": "user", "content": (NL + NL).join(info)},
    ]


# ---------------------------------------------------------------------------
# Several posts for the week (Auto-poster)
# ---------------------------------------------------------------------------

WEEK_THEMES = [
    "a product or service highlight",
    "a special offer with a reason to act now",
    "behind the scenes",
    "a happy customer story",
    "a useful tip for customers",
    "a question that invites comments",
    "a weekend or seasonal reminder",
]


def build_week_messages(
    description: str,
    memory: str | None,
    count: int,
    first_day: int = 1,
    avoid_captions: list[str] | None = None,
) -> list[dict]:
    themes = [WEEK_THEMES[(first_day - 1 + i) % len(WEEK_THEMES)] for i in range(count)]
    theme_lines = NL.join(f"Post {i + 1}: {theme}" for i, theme in enumerate(themes))

    system = (
        f"You are a social media strategist. Write {count} different Facebook posts for the "
        "business below, one for each day, with these themes:" + NL + theme_lines + NL + NL
        + "Use real, specific content and never placeholders. Write every post in this "
        "format, with all four lines:" + NL + NL + POST_FORMAT + NL + NL
        + f"Then the next post the same way, up to Post {count}. Write in the language of "
        "the business description. Plain text only, no markdown."
    )

    info = []

    if _profile(memory):
        info.append("Saved business profile: " + _profile(memory))

    if description and description.strip():
        info.append("Latest description from the user: " + description.strip())

    if avoid_captions:
        info.append(
            "These captions are already used, so write different ones:" + NL
            + NL.join("- " + caption for caption in avoid_captions)
        )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": (NL + NL).join(info) or "A small local business."},
    ]


# ---------------------------------------------------------------------------
# Suggested posting times
# ---------------------------------------------------------------------------

TIMES_PROMPT = """
You suggest good times of day to publish Facebook posts for a small business, based on the daily routine of its customers. Answer with exactly three lines, spread over the day, in 24-hour local time, in this format:
1. HH:MM - one short sentence about what the customers are doing at that time.
Never quote statistics or percentages and never say the times are proven. If no business is described, suggest general times.
""".strip()


def build_times_messages(memory: str | None) -> list[dict]:
    return [
        {"role": "system", "content": TIMES_PROMPT},
        {"role": "user", "content": _profile(memory) or "No business described."},
    ]


# ---------------------------------------------------------------------------
# Memory: a short profile of the user's business
# ---------------------------------------------------------------------------

MEMORY_PROMPT = """
You keep a short profile of a user's business for a marketing assistant. You get the current profile and the user's latest message. Return the updated profile.

- Keep only lasting facts that the user stated: business name, what it sells or does, city or area, target customers, tone or style, goals and offers.
- Write two to six short plain sentences, at most 120 words.
- Never add facts the user did not state, and never guess.
- If the latest message has no new lasting facts, return the current profile unchanged. If there is no profile and no business facts, return exactly: (none)
- Return only the profile text.
""".strip()


def build_memory_messages(current: str | None, user_message: str) -> list[dict]:
    shown = (current or "").strip() or "(none)"

    return [
        {"role": "system", "content": MEMORY_PROMPT},
        {
            "role": "user",
            "content": "Current profile:" + NL + shown + NL + NL
            + "Latest user message:" + NL + user_message.strip(),
        },
    ]


# ---------------------------------------------------------------------------
# Image prompt for Stable Diffusion 1.5
# ---------------------------------------------------------------------------

# SD 1.5 reads about 75 tokens, so the subject comes first and the prompt stays short.
ENHANCE_PROMPT = """
You write prompts for the Stable Diffusion 1.5 image model for a small business's Facebook posts. Rewrite the image idea below into one prompt.

- Write 20 to 32 words as comma-separated phrases. Start with the main subject, then the setting, the lighting, the camera view and the mood.
- Make it a realistic, high-end commercial photograph unless the idea clearly asks for an illustration.
- Use a warm beige, cream and deep maroon palette when it fits naturally.
- Never ask for text, words, letters, signs, logos or watermarks in the picture.
- Keep the meaning of the idea. Return only the prompt.
""".strip()

IMAGE_QUALITY_TAGS = "professional commercial photography, sharp focus, highly detailed, 4k"


def build_enhance_messages(idea: str, memory: str | None) -> list[dict]:
    content = "Image idea: " + idea.strip()

    if _profile(memory):
        content += NL + NL + "Business profile: " + _profile(memory)

    return [
        {"role": "system", "content": ENHANCE_PROMPT},
        {"role": "user", "content": content},
    ]
