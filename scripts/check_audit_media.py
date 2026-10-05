"""Offline regression checks for PDF isolation, URLs, images, and typing failures.

Run: venv_bot/bin/python scripts/check_audit_media.py
Reuses temporary SQLite/model fixtures and a controlled HTTP server. Spawned PDF
workers also block external sockets, .env loading, and Discord login.
"""

import asyncio
import base64
from concurrent.futures.process import BrokenProcessPool
import io
import os
import socket
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import aiohttp
import discord
import dotenv
from PIL import Image, ImageDraw

import check_features
import check_regressions as fixtures


def reject_external_access(*args, **kwargs):
    raise AssertionError("External access is disabled in PDF checks")


def initialize_pdf_checks():
    """Spawn does not inherit the parent fixture's network/config patches."""
    socket.socket.connect = reject_external_access
    socket.socket.connect_ex = reject_external_access
    dotenv.load_dotenv = reject_external_access
    discord.Client.run = reject_external_access


class AuditMediaChecks(unittest.IsolatedAsyncioTestCase):
    asyncTearDown = fixtures.BotChecks.asyncTearDown
    completion = staticmethod(fixtures.BotChecks.completion)
    public_http = fixtures.BotChecks.public_http
    chat = check_features.FeatureChecks.chat

    async def asyncSetUp(self):
        await fixtures.BotChecks.asyncSetUp(self)
        self.create.return_value.choices[0].message.tool_calls = []
        executor_type = self.bot.ProcessPoolExecutor

        def isolated_executor(*args, **kwargs):
            kwargs["initializer"] = initialize_pdf_checks
            return executor_type(*args, **kwargs)

        self.patches.enter_context(patch.object(self.bot, "ProcessPoolExecutor", side_effect=isolated_executor))

    def pdf(self, *pages):
        with self.bot.pymupdf.open() as document:
            for text in pages:
                document.new_page().insert_text((72, 72), text)
            return document.tobytes()

    @staticmethod
    def attachment(name, data, content_type):
        return SimpleNamespace(filename=name, content_type=content_type, size=len(data),
                               read=AsyncMock(return_value=data))

    @staticmethod
    def reference(content="", attachments=()):
        source = SimpleNamespace(author=SimpleNamespace(id=5, display_name="Other"),
                                 content=content, attachments=list(attachments), stickers=[])
        return SimpleNamespace(message_id=123, resolved=source, cached_message=None)

    @staticmethod
    def png(image):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    async def test_uploaded_replied_and_linked_pdfs_parse_outside_bot_process(self):
        direct = self.pdf("Direct synthetic PDF")
        replied = self.pdf("Replied synthetic PDF")
        linked = self.pdf("Linked synthetic PDF")
        first, second = self.chat(channel_id=10), self.chat(channel_id=20)
        first.attachments = [self.attachment("direct.pdf", direct, "application/pdf")]
        second.reference = self.reference(attachments=[self.attachment("reply.pdf", replied, "application/pdf")])

        async with self.public_http({"/linked.pdf": (200, {"Content-Type": "application/pdf"}, linked)}):
            # An asyncio thread would see this patch; a spawned process does not.
            with patch.object(self.bot.pymupdf, "open", side_effect=AssertionError("PDF parsing reached bot process")):
                results = await asyncio.wait_for(asyncio.gather(
                    self.bot.extract_message_context(first, "Read this", "Tester"),
                    self.bot.extract_message_context(second, "Read reply", "Tester"),
                    self.bot.fetch_url_content("https://public.example/linked.pdf"),
                ), 15)

        self.assertIn("Direct synthetic PDF", results[0][3])
        self.assertIn("Replied synthetic PDF", results[1][3])
        self.assertEqual(results[2]["type"], "text")
        self.assertIn("Linked synthetic PDF", results[2]["data"])

    async def test_pdf_worker_obeys_page_limit_and_survives_invalid_input(self):
        data = self.pdf("First page", "Second page", "Third page")
        invalid = await self.bot.extract_pdf_text_async(b"This is not a PDF")
        self.assertIn("Error reading PDF", invalid)
        with patch.object(self.bot, "MAX_PDF_PAGES", 2):
            text = await self.bot.extract_pdf_text_async(data)
        self.assertIn("First page", text)
        self.assertIn("Second page", text)
        self.assertNotIn("Third page", text)
        self.assertIn("Additional pages skipped", text)

    async def test_pdf_worker_starts_lazily_and_stops_when_client_closes(self):
        self.assertIsNone(self.client.pdf_executor)
        text = await self.bot.extract_pdf_text_async(self.pdf("Worker lifecycle"))
        self.assertIn("Worker lifecycle", text)
        executor = self.client.pdf_executor
        worker_pid = await asyncio.get_running_loop().run_in_executor(executor, os.getpid)
        self.assertNotEqual(worker_pid, os.getpid())

        await self.client.close()
        with self.assertRaises(RuntimeError):
            executor.submit(os.getpid)

    async def test_exited_pdf_worker_reports_failure_and_next_upload_recovers(self):
        data = self.pdf("Valid PDF after worker failure")
        await self.bot.extract_pdf_text_async(data)
        broken_executor = self.client.pdf_executor
        # Exit only this test's owned child; do not induce a native parser crash.
        with self.assertRaises(BrokenProcessPool):
            await asyncio.get_running_loop().run_in_executor(broken_executor, os._exit, 23)

        error = await self.bot.extract_pdf_text_async(data)
        self.assertIn("PDF worker stopped", error)
        self.assertIsNone(self.client.pdf_executor)
        recovered = await self.bot.extract_pdf_text_async(data)
        self.assertIn("Valid PDF after worker failure", recovered)
        self.assertIsNot(self.client.pdf_executor, broken_executor)

    async def test_formatted_urls_keep_real_path_characters(self):
        with Image.new("RGB", (4, 4), "white") as image:
            data = self.png(image)
        cases = (
            ("Read https://public.example/image.png", "/image.png"),
            ("Read [scan](https://public.example/image.png)", "/image.png"),
            ("Read [scan](https://public.example/image.png).", "/image.png"),
            ('Read "https://public.example/image.png"', "/image.png"),
            ("Read 'https://public.example/image.png'", "/image.png"),
            ("Read 'https://public.example/O'Reilly.png'.", "/O'Reilly.png"),
            ("Read <https://public.example/image.png>", "/image.png"),
            ("Read [scan](https://public.example/image_(draft).png)", "/image_(draft).png"),
            ("Read https://public.example/O'Reilly.png", "/O'Reilly.png"),
            ("Read [scan](https://public.example/image.png?name=(draft))", "/image.png?name=(draft)"),
            ("Read https://public.example/image.png?signature=abc.", "/image.png?signature=abc."),
        )
        for content, path in cases:
            with self.subTest(content=content):
                message = self.chat(content=content)
                async with self.public_http({path: (200, {"Content-Type": "image/png"}, data)}) as (seen, _):
                    _, images, _, documents = await self.bot.extract_message_context(message, content, "Tester")
                self.assertEqual(seen, [path])
                self.assertEqual(len(images), 1)
                self.assertEqual(await images[0].read(), data)
                self.assertEqual(documents, "")

    async def test_replied_url_is_fetched_before_context_quoting(self):
        with Image.new("RGB", (4, 4), "white") as image:
            data = self.png(image)
        message = self.chat(content="Explain this")
        message.reference = self.reference("https://public.example/image.png")
        async with self.public_http({"/image.png": (200, {"Content-Type": "image/png"}, data)}) as (seen, _):
            text, images, _, documents = await self.bot.extract_message_context(message, "Explain this", "Tester")
        self.assertIn("https://public.example/image.png", text)
        self.assertEqual(seen, ["/image.png"])
        self.assertEqual(len(images), 1)
        self.assertEqual(documents, "")

    async def test_transparency_survives_rgb_grayscale_and_palette_conversion(self):
        for mode in ("RGBA", "LA", "P"):
            with self.subTest(mode=mode):
                if mode == "P":
                    image = Image.new(mode, (128, 128), 1)
                    image.putpalette([0, 0, 0] * 256)
                    image.info["transparency"] = 1
                    opaque = 0
                else:
                    image = Image.new(mode, (128, 128), (0, 0) if mode == "LA" else (0, 0, 0, 0))
                    opaque = (0, 255) if mode == "LA" else (0, 0, 0, 255)
                with image:
                    ImageDraw.Draw(image).rectangle((32, 32, 96, 96), fill=opaque)
                    encoded = self.bot.process_image_bytes(self.png(image))
                self.assertIsNotNone(encoded)
                with Image.open(io.BytesIO(base64.b64decode(encoded))) as processed:
                    self.assertEqual(processed.format, "JPEG")
                    self.assertEqual(processed.mode, "RGB")
                    self.assertTrue(all(value >= 245 for value in processed.getpixel((8, 8))))
                    self.assertTrue(all(value <= 10 for value in processed.getpixel((64, 64))))

    async def test_16bit_png_keeps_dark_midpoint_and_light_intensities(self):
        row = b"\x00\x00" * 64 + b"\x00\x80" * 64 + b"\xff\xff" * 64
        with Image.frombytes("I;16", (192, 64), row * 64) as image:
            encoded = self.bot.process_image_bytes(self.png(image))
        self.assertIsNotNone(encoded)
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as processed, processed.convert("RGB") as rgb:
            pixels = [rgb.getpixel((x, 32))[0] for x in (32, 96, 160)]
        self.assertLessEqual(pixels[0], 5)
        self.assertTrue(122 <= pixels[1] <= 133, pixels)
        self.assertGreaterEqual(pixels[2], 250)

    async def test_failed_typing_requests_do_not_drop_text_attachment_or_answer(self):
        for error_type in (aiohttp.ClientConnectionError, OSError, TimeoutError):
            with self.subTest(error_type=error_type.__name__):
                message = self.chat(content="Summarize")
                message.attachments = [self.attachment("message.txt", b"Synthetic text document", "text/plain")]
                contexts = []

                def typing():
                    ctx = SimpleNamespace(__aenter__=AsyncMock(side_effect=error_type("Typing unavailable")),
                                          __aexit__=AsyncMock())
                    contexts.append(ctx)
                    return ctx

                message.channel.typing = typing
                await self.bot.on_message(message)
                self.assertIn("Synthetic text document", str(self.create.call_args.kwargs["messages"]))
                self.assertEqual(message.reply.await_count, 1)
                self.assertEqual(message.reply.call_args.args[0], "Answer")
                self.assertGreaterEqual(len(contexts), 2)
                for ctx in contexts:
                    ctx.__aexit__.assert_not_awaited()

    async def test_typing_cancellation_still_cancels_chat_and_releases_conversation(self):
        message = self.chat()
        message.channel.typing = lambda: SimpleNamespace(
            __aenter__=AsyncMock(side_effect=asyncio.CancelledError()), __aexit__=AsyncMock(),
        )
        with self.assertRaises(asyncio.CancelledError):
            await self.bot.on_message(message)
        self.create.assert_not_awaited()
        message.reply.assert_not_awaited()
        self.assertFalse(self.client.conversation_locks["channel:10"].locked())


if __name__ == "__main__":
    unittest.main(verbosity=2)
