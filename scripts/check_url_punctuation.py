"""Offline checks for prose punctuation around pasted URLs.

Run: venv_bot/bin/python scripts/check_url_punctuation.py
Reuses temporary SQLite fixtures, blocked external sockets, and controlled HTTP.
Never loads .env or contacts Discord, model servers, or public websites.
"""

import io
import unittest

from PIL import Image

import check_features
import check_regressions as fixtures


class URLPunctuationChecks(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixtures.BotChecks.asyncSetUp
    asyncTearDown = fixtures.BotChecks.asyncTearDown
    completion = staticmethod(fixtures.BotChecks.completion)
    public_http = fixtures.BotChecks.public_http
    chat = check_features.FeatureChecks.chat

    async def test_bare_paths_exclude_sentence_punctuation(self):
        url = "https://public.example/manual.png"
        for suffix in (".", ",", ";", ":", "!", "?", "...", ").", ".)"):
            with self.subTest(suffix=suffix):
                self.assertEqual(self.bot.extract_urls(f"Read {url}{suffix}"), [url])

    async def test_explicit_delimiters_preserve_literal_url_punctuation(self):
        for url in ("https://public.example/manual.png.", "https://public.example/path;!",
                    "https://public.example/image_(draft).png"):
            for formatted in (f"[manual]({url}).", f"<{url}>.", f'"{url}".', f"'{url}'.", f"`{url}`."):
                with self.subTest(formatted=formatted):
                    self.assertEqual(self.bot.extract_urls(f"Read {formatted}"), [url])
        self.assertEqual(self.bot.extract_urls("Read <https://public.example/literal)>"),
                         ["https://public.example/literal)"])

    async def test_balanced_parentheses_apostrophes_and_query_suffixes_survive(self):
        cases = (
            ("Read https://public.example/image_(draft).png.", "https://public.example/image_(draft).png"),
            ("Read https://public.example/page_(draft).", "https://public.example/page_(draft)"),
            ("Read 'https://public.example/O'Reilly.png'.", "https://public.example/O'Reilly.png"),
            ("Read https://public.example/O'Reilly.png.", "https://public.example/O'Reilly.png"),
            ("Read [scan](https://public.example/image.png?name=(draft)).",
             "https://public.example/image.png?name=(draft)"),
            ("Read https://public.example/image.png?signature=abc.",
             "https://public.example/image.png?signature=abc."),
            ("Read https://public.example/page#section.", "https://public.example/page#section."),
        )
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(self.bot.extract_urls(text), [expected])

    async def test_sentence_ending_image_url_is_fetched_directly(self):
        output = io.BytesIO()
        with Image.new("RGB", (4, 4), "white") as image:
            image.save(output, format="PNG")
        data = output.getvalue()
        content = "Read https://public.example/manual.png."
        message = self.chat(content=content)
        async with self.public_http({"/manual.png": (200, {"Content-Type": "image/png"}, data)}) as (seen, _):
            _, images, _, documents = await self.bot.extract_message_context(message, content, "Tester")
        self.assertEqual(seen, ["/manual.png"])
        self.assertEqual(len(images), 1)
        self.assertEqual(await images[0].read(), data)
        self.assertEqual(documents, "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
