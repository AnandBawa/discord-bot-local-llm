"""Offline checks for import safety, text history, routing, and shared media handling."""

import asyncio
import importlib.util
import io
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import aiohttp
import aiosqlite
import discord
from PIL import Image, ImageFile

import check_features
import check_regressions as fixtures


class RefactorChecks(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixtures.BotChecks.asyncSetUp
    asyncTearDown = fixtures.BotChecks.asyncTearDown
    completion = staticmethod(fixtures.BotChecks.completion)
    interaction = staticmethod(fixtures.BotChecks.interaction)
    seed = fixtures.BotChecks.seed
    count = fixtures.BotChecks.count
    chat = check_features.FeatureChecks.chat

    @staticmethod
    def answer(text="Answer", calls=None, tokens=10):
        message = SimpleNamespace(content=text, tool_calls=calls or [])
        message.model_dump = lambda **kwargs: {"role": "assistant", "content": text or "", "tool_calls": []}
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=SimpleNamespace(total_tokens=tokens))

    def fallback(self, response):
        create = AsyncMock(return_value=response)
        self.client.config.fallback_model = "cloud-model"
        self.client.fallback_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)), close=AsyncMock(),
        )
        return create

    @staticmethod
    def picture():
        with Image.new("RGB", (2, 2), "white") as image:
            data = io.BytesIO()
            image.save(data, format="PNG")
        return data.getvalue()

    async def test_import_does_not_load_config_create_logs_or_start_clients(self):
        before = set(Path.cwd().iterdir())
        with patch.dict(os.environ, {"DISCORD_BOT_TOKEN": "synthetic", "IMAGEGEN_TIMEOUT": "invalid"}), \
                patch("openai.AsyncOpenAI", side_effect=AssertionError("Clients must be created at startup")), \
                patch("logging.FileHandler", side_effect=AssertionError("Import must not create a log")):
            spec = importlib.util.spec_from_file_location("bot_import_check", fixtures.ROOT / "bot.py")
            imported = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(imported)
        self.assertIsNone(imported.client.lm_client)
        self.assertEqual(imported.client.config.token, "")
        self.assertEqual(set(Path.cwd().iterdir()), before)
        await imported.client.close()

    async def test_main_loads_config_only_when_started_and_requires_token(self):
        with patch.object(self.bot, "load_dotenv", return_value=False) as load, \
                patch.object(self.bot, "configure_logging") as logging_setup, \
                patch.object(self.client, "run") as run, \
                patch.object(ImageFile, "LOAD_TRUNCATED_IMAGES", False):
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(SystemExit):
                    self.bot.main()
            logging_setup.assert_not_called()
            run.assert_not_called()
            with patch.dict(os.environ, {"DISCORD_BOT_TOKEN": "synthetic", "LLM_MODEL_NAME": "configured", "VISION_ENABLED": "false"}, clear=True):
                self.bot.main()
            self.assertEqual(load.call_count, 2)
            logging_setup.assert_called_once()
            run.assert_called_once_with("synthetic")
            self.assertEqual(self.client.config.model, "configured")
            self.assertFalse(self.client.config.vision_enabled)

    async def test_literal_json_and_code_round_trip_as_text(self):
        for answer in ('[{"name":"Alice"}]', '{"key":true}', '[1,2]', '"quoted"', 'null', '```json\n[]\n```'):
            with self.subTest(answer=answer):
                await self.bot.cmd_clear.callback(self.interaction())
                await self.bot.save_and_send_response(self.message, "channel:10", answer, answer)
                context = await self.bot.build_ai_context("channel:10", "Follow up")
                self.assertEqual(context[1], {"role": "user", "content": answer})
                self.assertEqual(context[2], {"role": "assistant", "content": answer})

    async def test_history_merge_keeps_roles_and_current_image(self):
        async with self.bot.history_transaction():
            await self.client.db_conn.executemany(
                "INSERT INTO chat_history (server_id, role, content) VALUES ('channel:10', ?, ?)",
                [("assistant", "Earlier answer"), ("user", "First"), ("user", "Second")],
            )
        payload = [{"type": "text", "text": "Now"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,eA=="}}]
        context = await self.bot.build_ai_context("channel:10", payload)
        self.assertEqual([m["role"] for m in context], ["system", "user", "assistant", "user"])
        self.assertEqual(context[-1]["content"], [{"type": "text", "text": "First\n\nSecond"}, *payload])
        self.assertEqual(len(payload), 2)

    async def test_images_and_stickers_store_only_text_notes(self):
        data = self.picture()
        image = SimpleNamespace(filename="photo.png", read=AsyncMock(return_value=data))
        sticker = SimpleNamespace(name="sticker", format=SimpleNamespace(name="png"), read=AsyncMock(return_value=data))
        api, stored = await self.bot.build_user_payloads("Look", "", [image], [sticker], "Tester")
        self.assertTrue(api[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        self.assertTrue(api[2]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertIsInstance(stored, str)
        self.assertIn("photo.png", stored)
        self.assertIn("sticker", stored)
        self.assertNotIn("base64", stored)

    async def test_disabled_vision_does_not_download_visual_media(self):
        self.client.config.vision_enabled = False
        image = SimpleNamespace(read=AsyncMock())
        api, stored = await self.bot.build_user_payloads("Look", "", [image], [], "Tester")
        image.read.assert_not_awaited()
        self.assertTrue(all(part["type"] == "text" for part in api))
        self.assertIn("Vision is disabled", stored)

    async def test_failed_image_does_not_discard_other_media(self):
        bad = SimpleNamespace(filename="bad.png", read=AsyncMock(side_effect=discord.NotFound(
            SimpleNamespace(status=404, reason="Missing"), "test",
        )))
        good = SimpleNamespace(filename="good.png", read=AsyncMock(return_value=self.picture()))
        api, stored = await self.bot.build_user_payloads("Look", "", [bad, good], [], "Tester")
        self.assertIn("Failed to download", stored)
        self.assertIn("good.png", stored)
        self.assertEqual(sum(part["type"] == "image_url" for part in api), 1)

    async def test_failed_reference_fetch_keeps_the_question(self):
        for error in (TimeoutError("timed out"), aiohttp.ClientConnectionError("disconnected"), OSError("reset")):
            with self.subTest(error=type(error).__name__):
                message = self.chat()
                message.reference = SimpleNamespace(message_id=1, resolved=None, cached_message=None)
                message.channel.fetch_message.side_effect = error
                text, images, stickers, documents = await self.bot.extract_message_context(message, "My question", "Tester")
                self.assertEqual((text, images, stickers, documents), ("My question", [], [], ""))

    async def test_transport_failure_keeps_other_attachments_and_question(self):
        for error_type in (TimeoutError, aiohttp.ClientConnectionError, OSError):
            with self.subTest(error=error_type.__name__):
                def attachment(name, kind):
                    return SimpleNamespace(filename=name, content_type=kind, size=100, read=AsyncMock(side_effect=error_type("offline")))
                message = self.chat()
                message.attachments = [attachment("direct.pdf", "application/pdf"), attachment("bad.png", "image/png")]
                message.reference = SimpleNamespace(message_id=1, cached_message=None, resolved=SimpleNamespace(
                    author=SimpleNamespace(id=5, display_name="Other"), content="Reply context", stickers=[],
                    attachments=[attachment("reply.pdf", "application/pdf"), SimpleNamespace(
                        filename="good.png", content_type="image/png", size=100, read=AsyncMock(return_value=self.picture()),
                    )],
                ))
                text, images, stickers, documents = await self.bot.extract_message_context(message, "My question", "Tester")
                api, stored = await self.bot.build_user_payloads(text, documents, images, stickers, "Tester")
                self.assertIn("My question", stored)
                self.assertIn("Reply context", stored)
                self.assertEqual(stored.count("could not be downloaded"), 2)
                self.assertIn("Failed to download image 'bad.png'", stored)
                self.assertIn("good.png", stored)
                self.assertEqual(sum(part["type"] == "image_url" for part in api), 1)

    async def test_direct_and_replied_pdfs_share_extraction_and_truncation(self):
        with self.bot.pymupdf.open() as document:
            document.new_page().insert_text((72, 72), "Synthetic PDF contents")
            data = document.tobytes()
        def pdf(name):
            return SimpleNamespace(filename=name, content_type="application/pdf", size=len(data), read=AsyncMock(return_value=data))
        message = self.chat()
        message.attachments = [pdf("direct.pdf")]
        referenced = SimpleNamespace(author=SimpleNamespace(id=5, display_name="Other"), content="", attachments=[pdf("reply.pdf")], stickers=[])
        message.reference = SimpleNamespace(message_id=1, resolved=referenced, cached_message=None)
        with patch.object(self.bot, "MAX_TEXT_EXTRACTION_LENGTH", 9):
            text, _, _, documents = await self.bot.extract_message_context(message, "Read these", "Tester")
        self.assertIn("direct.pdf", text)
        self.assertIn("replied reply.pdf", documents)
        self.assertEqual(documents.count("...[Content Truncated]"), 2)

    async def test_direct_and_replied_text_files_share_decoding_and_truncation(self):
        body = "café 日本語 🙂\nmore text"
        data = body.encode("utf-8-sig")
        for name, mime in (("message.TXT", None), ("message.txt", "application/octet-stream"),
                           ("snippet", "Text/Plain; charset=utf-8")):
            with self.subTest(name=name, mime=mime):
                def attachment():
                    return SimpleNamespace(filename=name, content_type=mime, size=len(data),
                                           read=AsyncMock(return_value=data))
                message = self.chat()
                message.attachments = [attachment()]
                referenced = SimpleNamespace(author=SimpleNamespace(id=5, display_name="Other"),
                                             content="", attachments=[attachment()], stickers=[])
                message.reference = SimpleNamespace(message_id=1, resolved=referenced, cached_message=None)
                with patch.object(self.bot, "MAX_TEXT_EXTRACTION_LENGTH", 9):
                    text, images, _, documents = await self.bot.extract_message_context(message, "Read these", "Tester")
                self.assertEqual(images, [])
                self.assertIn(f"Text attached: '{name}'", text)
                self.assertIn(f"Extracted Text Content from replied {name}", documents)
                self.assertEqual(documents.count(body[:9]), 2)
                self.assertEqual(documents.count("...[Content Truncated]"), 2)
                self.assertNotIn("more text", documents)
                self.assertNotIn("\ufeff", documents)

    async def test_unreadable_text_files_keep_valid_companions_and_question(self):
        cases = ((b"\xffbad", "not readable UTF-8"), (b"binary\x00data", "not readable UTF-8"),
                 (b"\xef\xbb\xbf \r\n\t", "is empty"), (b"", "is empty"))
        for data, expected in cases:
            with self.subTest(data=data):
                message = self.chat()
                message.attachments = [
                    SimpleNamespace(filename="bad.txt", content_type="text/plain", size=len(data),
                                    read=AsyncMock(return_value=data)),
                    SimpleNamespace(filename="good.txt", content_type=None, size=10,
                                    read=AsyncMock(return_value=b"Good input")),
                ]
                text, _, _, documents = await self.bot.extract_message_context(message, "My question", "Tester")
                self.assertIn("My question", text)
                self.assertIn(expected, text)
                self.assertIn("Good input", documents)
                self.assertNotIn("bad.txt", documents)

    async def test_text_download_failures_and_oversized_bodies_keep_other_input(self):
        for failure in (TimeoutError("timeout"), aiohttp.ClientConnectionError("offline"), OSError("reset")):
            with self.subTest(error=type(failure).__name__):
                message = self.chat()
                message.attachments = [SimpleNamespace(filename="message.txt", content_type=None, size=10,
                                                       read=AsyncMock(side_effect=failure))]
                text, _, _, documents = await self.bot.extract_message_context(message, "My question", "Tester")
                self.assertIn("My question", text)
                self.assertIn("could not be downloaded", text)
                self.assertEqual(documents, "")
        message = self.chat()
        message.attachments = [SimpleNamespace(filename="message.txt", content_type=None, size=1,
                                               read=AsyncMock(return_value=b"Too large"))]
        with patch.object(self.bot, "MAX_FILE_SIZE", 4):
            text, _, _, documents = await self.bot.extract_message_context(message, "My question", "Tester")
        self.assertIn("exceeds the size limit", text)
        self.assertEqual(documents, "")

    async def test_oversized_direct_and_replied_attachments_are_never_read(self):
        message = self.chat()
        attachments = [SimpleNamespace(filename=name, content_type=kind, size=self.bot.MAX_FILE_SIZE + 1, read=AsyncMock())
                       for name, kind in (("huge.png", "image/png"), ("huge.pdf", "application/pdf"), ("huge.txt", "text/plain"))]
        message.attachments = attachments
        referenced = SimpleNamespace(author=SimpleNamespace(id=5, display_name="Other"), content="", attachments=attachments, stickers=[])
        message.reference = SimpleNamespace(message_id=1, resolved=referenced, cached_message=None)
        text, images, _, documents = await self.bot.extract_message_context(message, "Read these", "Tester")
        self.assertEqual(images, [])
        self.assertEqual(documents, "")
        self.assertEqual(text.count("exceeds the size limit"), 6)
        for attachment in attachments:
            attachment.read.assert_not_awaited()

    async def test_persona_view_and_context_load_saved_value_after_restart(self):
        await self.bot.cmd_role.callback(self.interaction(), "Persisted persona")
        await self.client.db_conn.close()
        self.client.db_conn = await aiosqlite.connect("check.sqlite3")
        interaction = self.interaction()
        await self.bot.cmd_role.callback(interaction)
        self.assertIn("Persisted persona", interaction.followup.send.call_args.args[0])
        context = await self.bot.build_ai_context("channel:10", "Hello")
        self.assertIn("Persisted persona", context[0]["content"])

    async def test_failed_persona_commit_preserves_history_and_persona(self):
        await self.bot.cmd_role.callback(self.interaction(), "Original")
        await self.seed()
        with patch.object(self.client.db_conn, "commit", new=AsyncMock(side_effect=RuntimeError("commit failed"))):
            with self.assertRaises(RuntimeError):
                await self.bot.cmd_role.callback(self.interaction(), "New")
        self.assertEqual(await self.bot.get_persona("channel:10"), "Original")
        self.assertEqual(await self.count("chat_history"), 2)

    async def test_cancelled_commit_persona_lookup_matches_persisted_value(self):
        await self.bot.cmd_role.callback(self.interaction(), "Original")
        await self.seed()
        loop = asyncio.get_running_loop()
        def cancel_at_commit(statement):
            if statement == "COMMIT":
                loop.call_soon_threadsafe(task.cancel)
        await self.client.db_conn.set_trace_callback(cancel_at_commit)
        task = asyncio.create_task(self.bot.cmd_role.callback(self.interaction(), "New"))
        try:
            with self.assertRaises(asyncio.CancelledError):
                await task
        finally:
            await self.client.db_conn.set_trace_callback(None)
        cursor = await self.client.db_conn.execute("SELECT prompt FROM server_config WHERE server_id = 'channel:10'")
        self.assertEqual((await cursor.fetchone())[0], "New")
        self.assertEqual(await self.bot.get_persona("channel:10"), "New")
        self.assertEqual(await self.count("chat_history"), 0)

    async def test_shutdown_closes_remaining_resources_after_failure(self):
        for failing in ("database", "primary", "fallback", "imagegen"):
            with self.subTest(failing=failing):
                closing = self.bot.MyAIClient(intents=discord.Intents.none())
                resources = {name: SimpleNamespace(close=AsyncMock()) for name in ("database", "primary", "fallback", "imagegen")}
                resources[failing].close.side_effect = RuntimeError(f"{failing} close failed")
                closing.db_conn = resources["database"]
                closing.lm_client = resources["primary"]
                closing.fallback_client = resources["fallback"]
                closing.imagegen = resources["imagegen"]
                with patch.object(discord.Client, "close", new=AsyncMock()) as discord_close:
                    with self.assertRaisesRegex(RuntimeError, f"{failing} close failed"):
                        await closing.close()
                    discord_close.assert_awaited_once()
                for resource in resources.values():
                    resource.close.assert_awaited_once()

    async def test_chat_uses_configured_fallback_and_cooldown(self):
        self.create.side_effect = RuntimeError("local offline")
        cloud = self.fallback(self.answer("Cloud answer"))
        for _ in range(2):
            response, used_fallback = await self.bot.request_completion(messages=[{"role": "user", "content": "Hello"}])
            self.assertTrue(used_fallback)
            self.assertEqual(response.choices[0].message.content, "Cloud answer")
        self.assertEqual(self.create.await_count, 1)
        self.assertEqual(cloud.await_count, 2)
        self.assertEqual(cloud.call_args.kwargs["model"], "cloud-model")
        self.assertEqual(cloud.call_args.kwargs["messages"], [{"role": "user", "content": "Hello"}])

    async def test_config_ignores_removed_owner_and_embedding_settings(self):
        settings = {
            "LLM_MODEL_NAME": "configured-chat", "VISION_ENABLED": "false",
            "BOT_OWNER_ID": "obsolete-not-an-integer", "MEMORY_DISTANCE_THRESHOLD": "obsolete-not-a-number",
            "EMB_MODEL_NAME": "obsolete-embedding", "FALLBACK_EMB_API_KEY": "obsolete-key",
        }
        config = self.bot.Config(settings)
        self.assertEqual(config.model, "configured-chat")
        self.assertFalse(config.vision_enabled)
        for name in ("owner_id", "embedding_model", "embedding_key", "memory_distance"):
            self.assertFalse(hasattr(config, name))

    async def test_tool_loop_keeps_fallback_even_if_another_call_resets_cooldown(self):
        call = SimpleNamespace(id="search", function=SimpleNamespace(name="web_search", arguments='{"query":"test"}'))
        self.create.side_effect = RuntimeError("local offline")
        cloud = self.fallback(None)
        cloud.side_effect = [self.answer(None, [call]), self.answer("Found it", tokens=77)]
        async def search(**kwargs):
            self.client.chat_dead_until = 0
            return "URL: https://example.com/source\n"
        with patch.dict(self.bot.AVAILABLE_TOOLS, web_search=search):
            answer = await self.bot.generate_ai_response([], self.chat(), False)
        self.assertEqual(self.create.await_count, 1)
        self.assertEqual(cloud.await_count, 2)
        self.assertIn("https://example.com/source", answer)
        self.assertEqual(self.client.highest_token_count, 77)

    async def test_tool_round_limit_is_preserved(self):
        call = SimpleNamespace(id="search", function=SimpleNamespace(name="web_search", arguments='{"query":"test"}'))
        self.create.return_value = self.answer(None, [call])
        search = AsyncMock(return_value="No results")
        with patch.dict(self.bot.AVAILABLE_TOOLS, web_search=search):
            answer = await self.bot.generate_ai_response([], self.chat(), False)
        self.assertIn("too many", answer)
        self.assertEqual(search.await_count, self.bot.MAX_TOOL_ITERATIONS)
        self.assertEqual(self.create.await_count, self.bot.MAX_TOOL_ITERATIONS + 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
