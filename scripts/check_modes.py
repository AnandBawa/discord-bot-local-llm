"""Offline checks for independent chat and image-generation modes.

Run: venv_bot/bin/python scripts/check_modes.py
Uses synthetic workflows, temporary SQLite files, and mocked endpoints. The
shared fixtures block external sockets, .env loading, and Discord login.
"""

import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, PropertyMock, patch

import discord
from PIL import Image

import check_dms as dm_fixtures
import check_features as feature_fixtures
import check_imagegen as image_fixtures
import check_regressions as fixtures


class ModeChecks(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixtures.BotChecks.asyncSetUp
    asyncTearDown = fixtures.BotChecks.asyncTearDown
    completion = staticmethod(fixtures.BotChecks.completion)
    interaction = staticmethod(fixtures.BotChecks.interaction)
    seed = fixtures.BotChecks.seed
    count = fixtures.BotChecks.count
    chat = feature_fixtures.FeatureChecks.chat
    dm = dm_fixtures.DMChecks.dm
    replies = staticmethod(dm_fixtures.DMChecks.replies)

    async def test_chat_defaults_to_localhost_and_explicit_blank_disables_it(self):
        default = self.bot.Config({})
        self.assertTrue(default.chat_enabled)
        self.assertEqual(default.base_url, "http://localhost:1234/v1")
        for url in ("", " \t\n"):
            with self.subTest(url=repr(url)):
                config = self.bot.Config({"LLM_BASE_URL": url})
                self.assertFalse(config.chat_enabled)
                self.assertEqual(config.base_url, "")
        configured = self.bot.Config({"LLM_BASE_URL": " https://chat.invalid/v1 "})
        self.assertTrue(configured.chat_enabled)
        self.assertEqual(configured.base_url, "https://chat.invalid/v1")

    async def test_image_timeout_is_validated_only_when_images_are_configured(self):
        for timeout in ("invalid", "", "0", "-1", "nan", "inf"):
            for chat_url in ("", "https://chat.invalid/v1"):
                with self.subTest(timeout=timeout, chat=bool(chat_url)):
                    config = self.bot.Config({
                        "LLM_BASE_URL": chat_url, "COMFYUI_BASE_URL": " \t",
                        "IMAGEGEN_TIMEOUT": timeout,
                    })
                    self.assertEqual(config.comfy_url, "")
                    self.assertEqual(config.image_timeout, 600.0)
                    with self.assertRaises(ValueError):
                        self.bot.Config({
                            "LLM_BASE_URL": chat_url, "COMFYUI_BASE_URL": "https://comfy.invalid",
                            "IMAGEGEN_TIMEOUT": timeout,
                        })
        config = self.bot.Config({"COMFYUI_BASE_URL": "https://comfy.invalid/", "IMAGEGEN_TIMEOUT": "12.5"})
        self.assertEqual(config.image_timeout, 12.5)

    async def test_startup_creates_only_configured_backends_without_loading_workflow(self):
        Path("workflow.json").write_text("invalid JSON", encoding="utf-8")
        for chat_enabled, images_enabled in ((True, False), (False, True), (True, True), (False, False)):
            with self.subTest(chat=chat_enabled, images=images_enabled):
                started = self.bot.MyAIClient(intents=self.bot.intents)
                started.config = self.bot.Config({
                    "LLM_BASE_URL": "https://chat.invalid/v1" if chat_enabled else "",
                    "COMFYUI_BASE_URL": "https://comfy.invalid" if images_enabled else "",
                })
                started.db_path = f"startup-{chat_enabled}-{images_enabled}.sqlite3"
                model = SimpleNamespace(close=AsyncMock())
                with patch.object(self.bot, "AsyncOpenAI", return_value=model) as factory, \
                        patch.object(self.bot.tree, "sync", new=AsyncMock()) as sync, \
                        patch.object(Path, "read_text", side_effect=AssertionError("Startup must not read workflows")), \
                        patch.object(self.bot.ImageGeneration, "request", new=AsyncMock(
                            side_effect=AssertionError("Startup must not contact a backend"),
                        )) as request:
                    try:
                        await started.setup_hook()
                        self.assertEqual(factory.call_count, int(chat_enabled))
                        self.assertIs(started.lm_client, model if chat_enabled else None)
                        self.assertEqual(started.imagegen is not None, images_enabled)
                        if chat_enabled:
                            self.assertEqual(factory.call_args.kwargs["base_url"], "https://chat.invalid/v1")
                        sync.assert_awaited_once()
                        request.assert_not_awaited()
                        cursor = await started.db_conn.execute("SELECT COUNT(*) FROM chat_history")
                        self.assertEqual((await cursor.fetchone())[0], 0)
                    finally:
                        await started.close()
                self.assertEqual(model.close.await_count, int(chat_enabled))

    async def test_chat_only_handles_messages_help_and_status_without_any_workflow(self):
        self.client.config = self.bot.Config({"COMFYUI_BASE_URL": "", "IMAGEGEN_TIMEOUT": "invalid"})
        self.create.return_value.choices[0].message.tool_calls = []
        for workflow in (None, "invalid JSON", '{"unsupported": true}'):
            with self.subTest(workflow=workflow):
                if workflow is None:
                    Path("workflow.json").unlink(missing_ok=True)
                else:
                    Path("workflow.json").write_text(workflow, encoding="utf-8")
                message = self.chat()
                self.client._connection.user.name = "Synthetic Bot"
                interaction = self.interaction()
                with patch.object(Path, "read_text", side_effect=AssertionError("Chat must not read a workflow")), \
                        patch.object(self.bot.ImageGeneration, "request", new=AsyncMock(
                            side_effect=AssertionError("Chat-only mode must not contact ComfyUI"),
                        )) as request, \
                        patch.object(discord.Client, "latency", new_callable=PropertyMock, return_value=0.1):
                    await self.bot.on_message(message)
                    await self.bot.cmd_help.callback(interaction)
                    await self.bot.cmd_status.callback(interaction)
                    request.assert_not_awaited()
                self.assertEqual(self.replies(message), ["Answer"])
                self.assertIn("Image generation is disabled", interaction.response.send_message.call_args.args[0])
                self.assertIn("**Image model:** Off", interaction.followup.send.call_args.args[0])
        self.assertEqual(self.create.await_count, 3)
        self.assertEqual(await self.count("chat_history"), 6)

    async def test_help_and_status_describe_only_enabled_capabilities(self):
        self.client._connection.user = SimpleNamespace(id=99, name="Synthetic Bot")
        for chat_enabled, images_enabled in ((True, False), (False, True), (True, True), (False, False)):
            for in_dm in (False, True):
                with self.subTest(chat=chat_enabled, images=images_enabled, dm=in_dm):
                    self.client.config = self.bot.Config({
                        "LLM_BASE_URL": "https://chat.invalid/v1" if chat_enabled else "",
                        "COMFYUI_BASE_URL": "https://comfy.invalid" if images_enabled else "",
                    })
                    interaction = self.interaction()
                    if in_dm:
                        interaction.guild_id = None
                    with patch.object(self.bot.ImageGeneration, "model_name", return_value="synthetic-image") as model, \
                            patch.object(discord.Client, "latency", new_callable=PropertyMock, return_value=0.123):
                        await self.bot.cmd_help.callback(interaction)
                        await self.bot.cmd_status.callback(interaction)
                    help_text = interaction.response.send_message.call_args.args[0]
                    status = interaction.followup.send.call_args.args[0]
                    self.assertIn("/help", help_text)
                    self.assertIn("/status", help_text)
                    self.assertIn("/imagegen", help_text)
                    self.assertIn("123 ms", status)
                    self.assertEqual(model.call_count, int(images_enabled))
                    self.assertIn("**Image model:** " + (
                        "`synthetic-image` (1K–2K)" if images_enabled else "Off"
                    ), status)
                    if chat_enabled:
                        self.assertIn("/role", help_text)
                        self.assertIn("/clear", help_text)
                        self.assertIn("**Chat model:** `local-model`", status)
                        self.assertIn("**History:**", status)
                        if in_dm:
                            self.assertIn("no mention needed", help_text)
                    else:
                        self.assertIn("Chat is disabled", help_text)
                        self.assertIn("**Chat:** Disabled", status)
                        for unavailable in ("/role", "/clear", "persona", "Files and links", "Reply to me"):
                            self.assertNotIn(unavailable, help_text)
                        for unavailable in ("Chat model", "History", "Web search", "Inputs", "Images/stickers", "PDF"):
                            self.assertNotIn(unavailable, status)
        self.create.assert_not_awaited()

    async def test_disabled_chat_commands_preserve_saved_history_and_persona(self):
        await self.seed()
        await self.client.db_conn.execute("INSERT INTO server_config VALUES ('channel:10', 'Saved persona')")
        await self.client.db_conn.commit()
        self.client.config = self.bot.Config({"LLM_BASE_URL": ""})
        for in_dm in (False, True):
            for command, args in ((self.bot.cmd_role, ()), (self.bot.cmd_role, ("replacement",)),
                                  (self.bot.cmd_role, ("clear",)), (self.bot.cmd_clear, ())):
                with self.subTest(command=command.name, args=args, dm=in_dm):
                    interaction = self.interaction()
                    if in_dm:
                        interaction.guild_id = None
                    await command.callback(interaction, *args)
                    interaction.response.send_message.assert_awaited_once_with(
                        "Chat is disabled on this bot.", ephemeral=True,
                    )
                    interaction.response.defer.assert_not_awaited()
                    interaction.followup.send.assert_not_awaited()
        self.assertEqual(await self.count("chat_history"), 2)
        self.assertEqual(await self.bot.get_persona("channel:10"), "Saved persona")
        self.assertEqual(self.client.conversation_versions, {})
        self.create.assert_not_awaited()

    async def test_disabled_chat_messages_do_not_process_input_or_claim_gpu_access(self):
        for images_enabled in (False, True):
            with self.subTest(images=images_enabled):
                self.client.config = self.bot.Config({
                    "LLM_BASE_URL": "", "COMFYUI_BASE_URL": "https://comfy.invalid" if images_enabled else "",
                })
                self.client.imagegen = SimpleNamespace(
                    reserve=Mock(side_effect=AssertionError("Disabled chat must not reserve GPU access")),
                    close=AsyncMock(),
                ) if images_enabled else None
                mentioned, without_history, dm, reply = self.chat(), self.chat(history=False), self.dm(), self.chat()
                reply.mentions = []
                reply.reference = SimpleNamespace(resolved=SimpleNamespace(author=self.client.user))
                with patch.object(self.bot, "extract_message_context", new=AsyncMock(
                    side_effect=AssertionError("Disabled chat must not process input"),
                )) as extract:
                    for message in (mentioned, without_history, dm, reply):
                        message.attachments = [SimpleNamespace(read=AsyncMock())]
                        await self.bot.on_message(message)
                        replies = self.replies(message)
                        self.assertEqual(len(replies), 1)
                        self.assertIn("Chat is disabled", replies[0])
                        self.assertEqual("/imagegen" in replies[0], images_enabled)
                        message.attachments[0].read.assert_not_awaited()
                    extract.assert_not_awaited()
                unmentioned, bot_message = self.chat(), self.dm()
                unmentioned.mentions = []
                bot_message.author.bot = True
                for ignored in (unmentioned, bot_message):
                    await self.bot.on_message(ignored)
                    self.assertEqual(self.replies(ignored), [])
        self.assertEqual(self.client.conversation_locks, {})
        self.assertEqual(self.client.chat_tasks, set())
        self.assertEqual(await self.count("chat_history"), 0)
        self.create.assert_not_awaited()

    async def test_direct_completion_is_rejected_before_touching_disabled_chat_backend(self):
        self.client.config = self.bot.Config({"LLM_BASE_URL": "", "COMFYUI_BASE_URL": "https://comfy.invalid"})
        self.client.lm_client = None
        self.client.imagegen = SimpleNamespace(
            local_request=Mock(side_effect=AssertionError("Disabled chat must not switch backends")),
            close=AsyncMock(),
        )
        with self.assertRaisesRegex(RuntimeError, "Chat is disabled"):
            await self.bot.request_completion(messages=[])
        self.client.imagegen.local_request.assert_not_called()
        self.create.assert_not_awaited()

    async def test_image_command_reports_disabled_images_in_chat_only_mode(self):
        interaction = self.interaction()
        await self.bot.cmd_imagegen.callback(interaction)
        response = interaction.response.send_message.call_args
        self.assertTrue(response.kwargs["ephemeral"])
        self.assertIn("Image generation is not configured", response.args[0])
        self.create.assert_not_awaited()


class ImageOnlyChecks(unittest.IsolatedAsyncioTestCase):
    asyncTearDown = fixtures.BotChecks.asyncTearDown
    completion = staticmethod(fixtures.BotChecks.completion)
    request = image_fixtures.ImageGenerationChecks.request

    async def asyncSetUp(self):
        await image_fixtures.ImageGenerationChecks.asyncSetUp(self)
        self.client.config.base_url = ""
        self.client.lm_client = None
        self.service.backend = None
        self.loaded = {"unrelated-model": ["leave-running"]}

        async def image_only_request(backend, method, path, **kwargs):
            if backend != "comfyui":
                raise AssertionError("Image-only mode must not contact LM Studio")
            return await self.request(backend, method, path, **kwargs)

        self.service.request.side_effect = image_only_request
        self.service.unload_lm = AsyncMock(side_effect=AssertionError("Image-only mode must not unload LM Studio"))

    async def test_repeated_images_succeed_without_a_chat_client_or_lm_studio(self):
        for prompt in ("First image", "Second image"):
            raw, _ = await self.service.generate(prompt, 1024, 1024)
            with Image.open(io.BytesIO(raw)) as image:
                self.assertEqual(image.size, (1024, 1024))
        submitted = [graph for graph in self.jobs.values() if "213" in graph]
        self.assertEqual([graph["6"]["inputs"]["text"] for graph in submitted], ["First image", "Second image"])
        self.assertEqual(self.loaded, {"unrelated-model": ["leave-running"]})
        self.assertTrue(self.calls)
        self.assertTrue(all(backend == "comfyui" for backend, _, _, _ in self.calls))
        self.service.unload_lm.assert_not_awaited()
        self.create.assert_not_awaited()

    async def test_image_only_still_respects_existing_comfyui_jobs(self):
        self.running.append("external-job")
        with self.assertRaisesRegex(self.bot.ModelBusyError, "unfinished jobs"):
            await self.service.generate("Wait for the existing job", 1024, 1024)
        self.assertEqual(self.jobs, {})
        self.assertFalse(any(path == "/prompt" for _, _, path, _ in self.calls))
        self.service.unload_lm.assert_not_awaited()
        self.create.assert_not_awaited()


if __name__ == "__main__":
    unittest.main(verbosity=2)
