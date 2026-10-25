"""Tests for the AI layer that need no GPU, database or network.

Run from the backend folder:  python -m unittest discover -s tests -v
"""

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from fastapi import HTTPException  # noqa: E402

from app.core import ai_client, model_server, prompts  # noqa: E402

SEED = BACKEND.parent / "training" / "data" / "seed_examples.jsonl"


def seed_rows():
    with SEED.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class ParsingTests(unittest.TestCase):
    def test_seed_posts_parse_completely(self):
        """Every post-shaped seed answer must parse into full posts."""

        checked = 0

        for row in seed_rows():
            if row["task"] == "week":
                expected = row["count"]
            elif row["task"] == "single_post":
                expected = 1
            elif row["task"] == "chat" and "Post 1" in row["output"] and "Caption:" in row["output"]:
                expected = 3
            else:
                continue

            posts = ai_client.parse_posts(row["output"])
            self.assertEqual(len(posts), expected, row["output"][:80])

            for post in posts:
                self.assertTrue(post["caption"])
                self.assertTrue(post["description"])
                self.assertGreaterEqual(len(post["hashtags"].split()), 5)
                self.assertTrue(post["image_prompt"])
                self.assertNotIn("Image prompt", post["hashtags"])

            checked += 1

        self.assertGreater(checked, 20)

    def test_messy_model_output(self):
        text = (
            "Sure! Here you go.\n\n**Post 1:**\n**Caption:** Big sale today!\n"
            "Description: Everything 20% off.\nHashtags: #Sale, #YourCity, #Deals #sale\n"
            "Image prompt: \"A shop with happy customers\"\n\nPost 2\nCaption: Second one\n"
            "Hashtags: #Two"
        )
        posts = ai_client.parse_posts(text)

        self.assertEqual(len(posts), 2)
        self.assertEqual(posts[0]["caption"], "Big sale today!")
        self.assertEqual(posts[0]["hashtags"], "#Sale #Deals")
        self.assertEqual(posts[0]["image_prompt"], "A shop with happy customers")
        # A missing image prompt falls back to the caption.
        self.assertEqual(posts[1]["image_prompt"], "Second one")

    def test_hashtags_without_hash(self):
        self.assertEqual(ai_client.clean_hashtags("coffee, lahore food"), "#coffee #lahore #food")

    def test_strip_markdown_keeps_hashtags(self):
        text = "## Heading\n**Caption:** Hi\nHashtags: #One #Two"
        self.assertEqual(ai_client._strip_markdown(text), "Heading\nCaption: Hi\nHashtags: #One #Two")


class HistoryTests(unittest.TestCase):
    def test_history_is_trimmed(self):
        history = [{"role": "user", "content": str(i) * 5000} for i in range(10)]
        messages = prompts.build_chat_messages("hi", history=history)

        # system + 6 history messages + the new message
        self.assertEqual(len(messages), 8)
        self.assertLess(len(messages[1]["content"]), prompts.HISTORY_CHARS + 10)

    def test_placeholder_business_name_is_dropped(self):
        messages = prompts.build_chat_messages("eid sale", business_name="My Business")
        self.assertEqual(messages[-1]["content"], "eid sale")


def fake_post(i):
    return f"Post {i}\nCaption: Caption number {i}\nDescription: Detail {i}.\nHashtags: #a{i} #b #c #d #e\nImage prompt: Photo {i}"


class FlowTests(unittest.TestCase):
    def test_week_plan_runs_in_batches(self):
        calls = []

        def chat(messages, max_new_tokens, temperature, prefix=""):
            count = int(messages[0]["content"].split("Write ")[1].split(" ")[0])
            offset = len(calls) * 10
            calls.append(count)
            return "\n\n".join(fake_post(offset + i) for i in range(1, count + 1))

        with mock.patch.object(model_server, "chat", side_effect=chat):
            posts = ai_client.draft_week_plan("A bakery", None, 7)

        self.assertEqual(calls, [4, 3])
        self.assertEqual(len(posts), 7)
        self.assertEqual(len({post["caption"] for post in posts}), 7)

    def test_week_plan_retries_a_short_batch(self):
        answers = iter([
            fake_post(1) + "\n\n" + fake_post(2),  # asked for 3, got 2
            fake_post(3),
        ])

        with mock.patch.object(model_server, "chat", side_effect=lambda *a, **k: next(answers)):
            posts = ai_client.draft_week_plan("A bakery", None, 3)

        self.assertEqual([post["caption"] for post in posts], ["Caption number 1", "Caption number 2", "Caption number 3"])

    def test_single_post_retries_then_fails_cleanly(self):
        with mock.patch.object(model_server, "chat", return_value="I am not sure."):
            with self.assertRaises(HTTPException) as caught:
                ai_client.draft_single_post("sale", None)

        self.assertEqual(caught.exception.status_code, 502)

    def test_single_post_without_post_header(self):
        answer = "Caption: Hello there!\nDescription: More.\nHashtags: #a #b #c #d #e\nImage prompt: A photo."

        with mock.patch.object(model_server, "chat", return_value=answer):
            post = ai_client.draft_single_post("sale", None)

        self.assertEqual(post["caption"], "Hello there!")

    def test_times(self):
        answer = "1. 08:00 - Breakfast time.\n2. 1:30 pm - Lunch break.\n3. 20:30 - Evening relax."

        with mock.patch.object(model_server, "chat", return_value=answer):
            times = ai_client.suggest_post_times(None)

        self.assertEqual([t["time"] for t in times], ["08:00", "13:30", "20:30"])

    def test_memory_none(self):
        with mock.patch.object(model_server, "chat", return_value="(none)"):
            self.assertEqual(ai_client.update_memory_summary(None, "hi"), "")

    def test_enhance_falls_back_on_bad_answer(self):
        with mock.patch.object(model_server, "chat", return_value="ok"):
            prompt = ai_client.enhance_image_prompt("cupcakes on a table")

        self.assertTrue(prompt.startswith("cupcakes on a table, realistic commercial photograph"))

    def test_enhance_stops_when_server_is_offline(self):
        offline = HTTPException(status_code=503, detail="offline")

        with mock.patch.object(model_server, "chat", side_effect=offline):
            with self.assertRaises(HTTPException):
                ai_client.generate_image_png("cupcakes")


class ModelServerClientTests(unittest.TestCase):
    def test_not_configured(self):
        with mock.patch.object(model_server.settings, "MODEL_SERVER_URL", ""):
            with self.assertRaises(HTTPException) as caught:
                model_server.chat([{"role": "user", "content": "hi"}])

        self.assertEqual(caught.exception.status_code, 503)
        self.assertIn("not configured", caught.exception.detail)

    def test_unreachable(self):
        with mock.patch.object(model_server.settings, "MODEL_SERVER_URL", "http://127.0.0.1:9"), \
             mock.patch.object(model_server.time, "sleep"):
            with self.assertRaises(HTTPException) as caught:
                model_server.chat([{"role": "user", "content": "hi"}])

        self.assertEqual(caught.exception.status_code, 503)

    def test_health_never_raises(self):
        with mock.patch.object(model_server.settings, "MODEL_SERVER_URL", "http://127.0.0.1:9"):
            self.assertFalse(model_server.health()["online"])


if __name__ == "__main__":
    unittest.main()
