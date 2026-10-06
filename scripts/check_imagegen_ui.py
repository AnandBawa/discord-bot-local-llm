"""Offline checks for the image-generation form, permissions, queues, and uploads.

Run: venv_bot/bin/python scripts/check_imagegen_ui.py
Reuses temporary SQLite fixtures and blocked sockets; never reads .env/runtime
databases/logs or contacts Discord, LM Studio, or ComfyUI.
"""

import asyncio
from datetime import datetime, timedelta, timezone
import io
import json
import random
import re
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, PropertyMock, patch

import discord
from PIL import Image, PngImagePlugin

import check_regressions as fixtures
import check_features as feature_fixtures


class ImagegenUIChecks(unittest.IsolatedAsyncioTestCase):
    asyncTearDown = fixtures.BotChecks.asyncTearDown
    completion = staticmethod(fixtures.BotChecks.completion)
    chat = feature_fixtures.FeatureChecks.chat

    async def asyncSetUp(self):
        await fixtures.BotChecks.asyncSetUp(self)
        self.client.config.comfy_url = "http://comfy.invalid:8188"
        self.backend = self.client.imagegen = self.bot.ImageGeneration(self.client.config)
        self.backend.request = AsyncMock(side_effect=AssertionError("Backend requests disabled"))
        self.backend.generate = AsyncMock(return_value=(self.png((1024, 1024)), 83.2))
        self.create.side_effect = AssertionError("Image prompts must not go through the LLM")

    @staticmethod
    def png(size=(64, 64), *, metadata=False, noise=False):
        picture = (Image.frombytes("RGBA", size, random.Random(7).randbytes(size[0] * size[1] * 4))
                   if noise else Image.new("RGB", size, (25, 80, 140)))
        info = PngImagePlugin.PngInfo()
        if metadata:
            info.add_text("workflow", '{"private_workflow": "synthetic"}')
            info.add_text("prompt", '{"private_prompt": "synthetic"}')
        output = io.BytesIO()
        picture.save(output, format="PNG", pnginfo=info)
        return output.getvalue()

    @staticmethod
    def fill(modal, **values):
        # Feed the same values discord.py receives in a modal submission.
        for field, value in values.items():
            getattr(modal, field)._refresh_state(None, {"value": value})
        return modal

    def interaction(self, user_id=42, server_id=1, *, thread=False):
        interaction = fixtures.BotChecks.interaction(user_id, server_id)
        interaction.permissions = discord.Permissions.none()
        interaction.app_permissions = discord.Permissions(
            view_channel=True, send_messages=True, send_messages_in_threads=True, attach_files=True,
        )
        interaction.guild = SimpleNamespace(id=server_id, filesize_limit=8_000_000)
        interaction.filesize_limit = 8_000_000
        interaction.response.send_modal = AsyncMock()
        interaction.edit_original_response = AsyncMock()
        interaction.created_at = datetime.now(timezone.utc)
        interaction.uploads = []

        async def capture_edit(**kwargs):
            for attachment in kwargs.get("attachments", []):
                interaction.uploads.append((attachment.filename, attachment.fp.getvalue()))

        interaction.progress = SimpleNamespace(edit=AsyncMock(side_effect=capture_edit))
        interaction.channel = Mock(spec=discord.Thread) if thread else SimpleNamespace()
        interaction.channel.send = AsyncMock(return_value=interaction.progress)
        return interaction

    def assert_slots_free(self):
        self.assertEqual(self.client.image_users, set())
        self.assertEqual(self.client.image_tasks, set())
        self.assertEqual(self.backend.work, {"lmstudio": 0, "comfyui": 0})

    async def test_chat_declines_images_at_command_and_form_submission_in_any_server(self):
        first = self.chat()
        second = self.chat(author=84)
        other = self.chat(server=2)
        entered, other_entered, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        seen = []

        async def handle(message, *args):
            seen.append(message)
            (other_entered if message is other else entered).set()
            await release.wait()

        with patch.object(self.bot, "handle_server_message", side_effect=handle):
            tasks = [asyncio.create_task(self.bot.on_message(first))]
            try:
                await asyncio.wait_for(entered.wait(), 1)
                tasks += [asyncio.create_task(self.bot.on_message(message)) for message in (second, other)]
                await asyncio.wait_for(other_entered.wait(), 1)
                self.assertNotIn(second, seen)
                for server in (1, 2, 3):
                    for step in ("command", "form"):
                        with self.subTest(server=server, step=step):
                            interaction = self.interaction(server_id=server)
                            if step == "command":
                                await self.bot.cmd_imagegen.callback(interaction)
                            else:
                                modal = self.fill(self.bot.ImageGenerationModal(), width="1024", height="1024", prompt="A tree")
                                await modal.on_submit(interaction)
                            reply = interaction.response.send_message.call_args
                            self.assertIn("Chat is active right now", reply.args[0])
                            self.assertTrue(reply.kwargs["ephemeral"])
                            interaction.response.send_modal.assert_not_awaited()
                            interaction.response.defer.assert_not_awaited()
                            interaction.channel.send.assert_not_awaited()
            finally:
                release.set()
                await asyncio.gather(*tasks)
        self.assertEqual(seen, [first, other, second])
        self.backend.generate.assert_not_awaited()
        self.assert_slots_free()
        await self.bot.cmd_imagegen.callback(self.interaction())

    async def test_image_reserves_before_discord_ack_and_declines_chat_without_history_changes(self):
        interaction = self.interaction()
        entered, release = asyncio.Event(), asyncio.Event()

        async def defer(**kwargs):
            entered.set()
            await release.wait()

        interaction.response.defer.side_effect = defer
        task = asyncio.create_task(self.bot.run_imagegen(interaction, "A tree", 1024, 1024))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            self.backend.generate.assert_not_awaited()
            with patch.object(self.bot, "handle_server_message", new=AsyncMock()) as handle:
                for server in (1, 2):
                    for history in (True, False):
                        message = self.chat(server=server, history=history)
                        await asyncio.wait_for(self.bot.on_message(message), 1)
                        reply = message.reply if history else message.channel.send
                        self.assertIn("Image generation is active right now", reply.call_args.args[0])
                handle.assert_not_awaited()
            self.assertEqual(await fixtures.BotChecks.count(self, "chat_history"), 0)
            self.create.assert_not_awaited()
        finally:
            release.set()
            await task
        self.assert_slots_free()

    async def test_queued_chat_blocks_images_until_cancelled_or_invalidated(self):
        for action in ("cancel", "clear"):
            with self.subTest(action=action):
                lock = self.client.conversation_locks["channel:10"] = asyncio.Lock()
                await lock.acquire()
                with patch.object(self.bot, "handle_server_message", new=AsyncMock()) as handle:
                    task = asyncio.create_task(self.bot.on_message(self.chat()))
                    try:
                        await asyncio.sleep(0)
                        interaction = self.interaction(server_id=2)
                        await self.bot.cmd_imagegen.callback(interaction)
                        self.assertIn("Chat is active right now", interaction.response.send_message.call_args.args[0])
                        if action == "cancel":
                            task.cancel()
                            with self.assertRaises(asyncio.CancelledError):
                                await task
                        else:
                            self.client.conversation_versions["channel:10"] = self.client.conversation_versions.get("channel:10", 0) + 1
                    finally:
                        lock.release()
                        await asyncio.gather(task, return_exceptions=True)
                    handle.assert_not_awaited()
                    self.assert_slots_free()
        with patch.object(self.bot, "handle_server_message", side_effect=RuntimeError("Synthetic failure")):
            with self.assertRaises(RuntimeError):
                await self.bot.on_message(self.chat())
        self.assert_slots_free()

    async def test_four_chats_keep_three_processing_slots_and_queue_the_fourth(self):
        self.backend.backend = "lmstudio"
        entered, release = asyncio.Event(), asyncio.Event()
        active = peak = 0

        async def infer(**kwargs):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            if active == 3:
                entered.set()
            try:
                await release.wait()
                response = self.completion("Answer")
                response.choices[0].message.tool_calls = []
                return response
            finally:
                active -= 1

        self.create.side_effect = infer
        messages = [self.chat(server=server) for server in range(1, 5)]
        with patch.object(self.bot, "build_ai_context", new=AsyncMock(return_value=[])):
            tasks = [asyncio.create_task(self.bot.on_message(message)) for message in messages]
            try:
                await asyncio.wait_for(entered.wait(), 2)
                self.assertEqual(self.create.await_count, 3)
                self.assertTrue(all(not task.done() for task in tasks))
                interaction = self.interaction(server_id=5)
                await self.bot.cmd_imagegen.callback(interaction)
                self.assertIn("Chat is active right now", interaction.response.send_message.call_args.args[0])
            finally:
                release.set()
                await asyncio.gather(*tasks)
        self.assertEqual(self.create.await_count, 4)
        self.assertEqual(peak, 3)
        self.assertTrue(all(message.reply.await_count == 1 for message in messages))
        self.assert_slots_free()

    async def test_single_form_submits_dimensions_and_literal_prompt_without_an_extra_button(self):
        opening = self.interaction()
        self.assertFalse(opening.permissions.administrator)
        self.assertFalse(opening.app_permissions.read_message_history)
        self.assertIsNone(self.bot.cmd_imagegen.default_permissions)
        await self.bot.cmd_imagegen.callback(opening)
        opening.response.send_modal.assert_awaited_once()
        modal = opening.response.send_modal.call_args.args[0]
        self.assertIsInstance(modal, self.bot.ImageGenerationModal)
        self.assertEqual(len(modal.to_dict()["components"]), 3)
        self.assertEqual(modal.prompt.style, discord.TextStyle.paragraph)
        self.assertEqual((modal.prompt.min_length, modal.prompt.max_length), (1, 4000))
        self.backend.generate.assert_not_awaited()
        opening.channel.send.assert_not_awaited()
        self.assert_slots_free()

        submission = self.interaction()
        literal = "  A café in the rain.\nKeep the lettering exactly as given.  "
        self.fill(modal, width="1080", height="1920", prompt=literal)
        self.backend.generate.side_effect = lambda prompt, width, height: (self.png((width, height)), 83.2)
        await modal.on_submit(submission)
        submission.response.defer.assert_awaited_once_with(ephemeral=True, thinking=True)
        submission.response.send_message.assert_not_awaited()
        submission.response.send_modal.assert_not_awaited()
        confirmation = submission.edit_original_response.call_args.kwargs["content"]
        self.assertIn("1088 × 1920", confirmation)
        self.assertIn("2.09 MP", confirmation)
        self.assertIn("adjusted from 1080 × 1920", confirmation)
        progress = submission.channel.send.call_args
        self.assertIn("1088 × 1920", progress.args[0])
        self.assertIn("queued", progress.args[0])
        self.assertNotIn("view", progress.kwargs)
        self.backend.generate.assert_awaited_once_with(literal, 1088, 1920)
        self.assertEqual(len(submission.uploads), 1)
        final = submission.progress.edit.call_args.kwargs
        header, _, spoiler = final["content"].partition("\n")
        self.assertEqual(header, "<@42> · 1088 × 1920 · Generated in 83.2s")
        self.assertTrue(spoiler.startswith("||") and spoiler.endswith("||"))
        self.assertEqual(re.sub(r"\\(.)", r"\1", spoiler[2:-2], flags=re.DOTALL), literal)
        self.assertTrue(final["suppress"])
        submission.channel.send.assert_awaited_once()
        self.create.assert_not_awaited()
        self.assert_slots_free()

    async def test_resolution_rounding_boundaries_and_malformed_form_values(self):
        for requested, expected in (((512, 512), (1024, 1024)), ((64, 2048), (512, 2048)),
                                    ((1080, 1920), (1088, 1920)), ((2048, 2048), (2048, 2048)),
                                    ((3840, 2160), (2048, 1152)), ((1536, 1024), (1536, 1024)),
                                    ((3840, 1080), (2048, 576)), ((4000, 1000), (2048, 512)),
                                    ((1000, 1000), (1024, 1024)), ((2000, 1000), (2000, 1008)),
                                    ((1920, 1088), (1920, 1088)), ((4096, 4096), (2048, 2048))):
            with self.subTest(requested=requested):
                self.assertEqual(self.bot.image_resolution(*requested), expected)
        for invalid in ("", "wide", "64.5", "1e3", "0", "-1", "-64"):
            for field in ("width", "height"):
                with self.subTest(field=field, value=invalid):
                    interaction = self.interaction()
                    values = {"width": "64", "height": "64", "prompt": "A tree", field: invalid}
                    modal = self.fill(self.bot.ImageGenerationModal(), **values)
                    await modal.on_submit(interaction)
                    response = interaction.response.send_message.call_args
                    self.assertTrue(response.kwargs["ephemeral"])
                    self.assertIn("positive whole numbers", response.args[0])
                    self.assertNotIn("view", response.kwargs)
        self.backend.generate.assert_not_awaited()
        self.assert_slots_free()

    async def test_resolution_pixel_limits_idempotence_and_common_aspect_ratios(self):
        rng = random.Random(17)
        requests = [(1, 1), (99999999, 1), (1, 99999999), (1024, 976), (2048, 976)]
        requests += [(rng.randint(1, 99999999), rng.randint(1, 99999999)) for _ in range(200)]
        for requested in requests:
            with self.subTest(requested=requested):
                width, height = self.bot.image_resolution(*requested)
                self.assertGreaterEqual(width * height, 1024 * 1024)
                self.assertLessEqual(width * height, 2048 * 2048)
                self.assertLessEqual(max(width, height), 2048)
                self.assertEqual((width % 16, height % 16), (0, 0))
                self.assertEqual(self.bot.image_resolution(width, height), (width, height))
                self.assertEqual(self.bot.image_resolution(*requested[::-1]), (height, width))
        for x, y in ((1, 1), (16, 9), (9, 16), (3, 2), (2, 3)):
            for scale in (1, 31, 80, 160, 240):
                width, height = self.bot.image_resolution(x * scale, y * scale)
                self.assertLess(abs((width / height) / (x / y) - 1), 0.02)
        interaction = self.interaction()
        modal = self.fill(self.bot.ImageGenerationModal(), width="3840", height="2160", prompt="A tree")
        self.backend.generate.return_value = self.png((2048, 1152)), 83.2
        await modal.on_submit(interaction)
        confirmation = interaction.edit_original_response.call_args.kwargs["content"]
        self.assertIn("2048 × 1152", confirmation)
        self.assertIn("adjusted from 3840 × 2160", confirmation)
        self.backend.generate.assert_awaited_once_with("A tree", 2048, 1152)
        self.assert_slots_free()

    async def test_permissions_are_rechecked_at_each_submission_including_threads(self):
        for thread in (False, True):
            send_permission = "send_messages_in_threads" if thread else "send_messages"
            for missing in ("view_channel", send_permission, "attach_files"):
                for step in ("command", "form"):
                    with self.subTest(thread=thread, missing=missing, step=step):
                        interaction = self.interaction(thread=thread)
                        setattr(interaction.app_permissions, missing, False)
                        if step == "command":
                            await self.bot.cmd_imagegen.callback(interaction)
                        else:
                            modal = self.fill(self.bot.ImageGenerationModal(), width="64", height="64", prompt="A tree")
                            await modal.on_submit(interaction)
                        response = interaction.response.send_message.call_args
                        self.assertTrue(response.kwargs["ephemeral"])
                        self.assertIn("Attach Files", response.args[0])
                        self.assertNotIn("Read Message History", response.args[0])
                        interaction.response.send_modal.assert_not_awaited()
                        interaction.channel.send.assert_not_awaited()
            allowed = self.interaction(thread=thread)
            unrelated = "send_messages" if thread else "send_messages_in_threads"
            setattr(allowed.app_permissions, unrelated, False)
            await self.bot.cmd_imagegen.callback(allowed)
            allowed.response.send_modal.assert_awaited_once()
        for absent in ("guild", "channel"):
            interaction = self.interaction()
            setattr(interaction, absent, None)
            await self.bot.cmd_imagegen.callback(interaction)
            self.assertIn("server channel", interaction.response.send_message.call_args.args[0])
        self.backend.generate.assert_not_awaited()
        self.assert_slots_free()

    async def test_malformed_prompts_and_dimensions_do_not_claim_queue_slots(self):
        cases = [(prompt, 64, 64, "1 and 4000") for prompt in ("", " \n\t ", "x" * 4001, None, 123)]
        for invalid in (True, 64.0, "64", 0, -1, None):
            cases.extend((("A tree", invalid, 64, "positive whole numbers"),
                          ("A tree", 64, invalid, "positive whole numbers")))
        for prompt, width, height, expected in cases:
            with self.subTest(prompt=repr(prompt)[:40], width=width, height=height):
                interaction = self.interaction()
                await self.bot.run_imagegen(interaction, prompt, width, height)
                response = interaction.response.send_message.call_args
                self.assertTrue(response.kwargs["ephemeral"])
                self.assertIn(expected, response.args[0])
                interaction.response.defer.assert_not_awaited()
                interaction.channel.send.assert_not_awaited()
                self.assert_slots_free()
        self.backend.generate.assert_not_awaited()
        self.create.assert_not_awaited()

    async def test_literal_prompt_and_expired_interaction_deliver_via_normal_message(self):
        interaction = self.interaction()
        literal = "  @everyone <@999> <@&123>\nA café in the rain.\n"
        literal += "||spoiler|| ```code``` **bold** https://example.com/a_b\n"
        literal += "[x||VISIBLE||](https://example.com) [x](https://example.com/```VISIBLE```)\n"
        literal += "x" * (4000 - len(literal) - 2) + "  "
        ready, release = asyncio.Event(), asyncio.Event()

        async def generate(prompt, width, height):
            ready.set()
            await release.wait()
            return self.png((width, height), metadata=True), 83.2

        async def require_live_token(*args, **kwargs):
            self.assertLess(datetime.now(timezone.utc) - interaction.created_at, timedelta(minutes=15))

        self.backend.generate.side_effect = generate
        interaction.edit_original_response.side_effect = require_live_token
        interaction.followup.send.side_effect = require_live_token
        modal = self.fill(self.bot.ImageGenerationModal(), width="80", height="112", prompt=literal)
        task = asyncio.create_task(modal.on_submit(interaction))
        try:
            await asyncio.wait_for(ready.wait(), 2)
            interaction.response.defer.assert_awaited_once_with(ephemeral=True, thinking=True)
            interaction.edit_original_response.assert_awaited_once()
            progress = interaction.channel.send.call_args
            self.assertIn("queued", progress.args[0])
            self.assertNotIn(literal, progress.args[0])
            self.assertEqual(progress.kwargs["allowed_mentions"].to_dict()["users"], [42])
            self.assertEqual(progress.kwargs["allowed_mentions"].to_dict()["parse"], [])
            interaction.created_at -= timedelta(minutes=16)
        finally:
            release.set()
            await asyncio.wait_for(task, 2)
        self.backend.generate.assert_awaited_once_with(literal, 864, 1216)
        self.create.assert_not_awaited()
        self.backend.request.assert_not_awaited()
        interaction.followup.send.assert_not_awaited()
        final = interaction.progress.edit.call_args
        self.assertEqual(final.kwargs["allowed_mentions"].to_dict()["parse"], [])
        self.assertTrue(final.kwargs["suppress"])
        self.assertTrue(final.kwargs["content"].startswith("<@42> · 864 × 1216 · Generated in 83.2s\n||"))
        messages = [final.kwargs["content"]]
        for call in interaction.channel.send.call_args_list[1:]:
            self.assertEqual(call.kwargs["allowed_mentions"].to_dict()["parse"], [])
            self.assertTrue(call.kwargs["suppress_embeds"])
            self.assertTrue(call.args[0].startswith("<@42> · Prompt (continued)\n||"))
            messages.append(call.args[0])
        self.assertGreater(len(messages), 1)
        bodies = []
        for text in messages:
            self.assertLessEqual(len(text.encode("utf-16-le")) // 2, 2000)
            body = text.partition("\n")[2]
            self.assertTrue(body.startswith("||") and body.endswith("||"))
            self.assertNotIn("||", body[2:-2])
            bodies.append(body[2:-2])
        self.assertEqual(re.sub(r"\\(.)", r"\1", "".join(bodies), flags=re.DOTALL), literal)
        self.assertEqual(len(interaction.uploads), 1)
        name, data = interaction.uploads[0]
        self.assertEqual(name, "SPOILER_image.png")
        with Image.open(io.BytesIO(data)) as picture:
            self.assertEqual(picture.size, (864, 1216))
            self.assertNotIn("workflow", picture.info)
            self.assertNotIn("prompt", picture.info)
        self.assert_slots_free()

    async def test_prompt_spoilers_preserve_unicode_and_escape_boundaries(self):
        for literal in ("|" * 4000, "\\" * 4000, "😀" * 4000,
                        "[x||VISIBLE||](https://example.com)",
                        "[x](https://example.com/```VISIBLE```)",
                        "[x](https://example.com/a\\||VISIBLE||)",
                        "x" * 1895 + "\\||```\n> quote\n# heading\n- bullet\n"):
            with self.subTest(prompt=literal[:10]):
                chunks = self.bot.image_prompt_chunks(literal, 1900)
                bodies = []
                for chunk in chunks:
                    self.assertLessEqual(len(chunk.encode("utf-16-le")) // 2, 1900)
                    self.assertTrue(chunk.startswith("||") and chunk.endswith("||"))
                    body = chunk[2:-2]
                    self.assertNotIn("||", body)
                    self.assertIsNone(re.search(r"(?<!\\)`", body))
                    self.assertEqual((len(body) - len(body.rstrip("\\"))) % 2, 0)
                    bodies.append(body)
                self.assertEqual(re.sub(r"\\(.)", r"\1", "".join(bodies), flags=re.DOTALL), literal)

    async def test_prompt_continuation_failure_keeps_delivered_image_and_explains(self):
        forbidden = discord.Forbidden(SimpleNamespace(status=403, reason="Forbidden"), "Missing permissions")
        for error in (forbidden, TimeoutError(), OSError(), asyncio.CancelledError()):
            with self.subTest(error=type(error).__name__):
                interaction = self.interaction()
                interaction.channel.send.side_effect = [interaction.progress, error]
                if isinstance(error, asyncio.CancelledError):
                    with self.assertRaises(asyncio.CancelledError):
                        await self.bot.run_imagegen(interaction, "x" * 4000, 1024, 1024)
                else:
                    await self.bot.run_imagegen(interaction, "x" * 4000, 1024, 1024)
                self.assertEqual(len(interaction.uploads), 1)
                self.assertEqual(interaction.uploads[0][0], "SPOILER_image.png")
                final = interaction.progress.edit.call_args.kwargs
                self.assertNotIn("attachments", final)
                self.assertIn("Generated in 83.2s\n||", final["content"])
                self.assertTrue(final["content"].endswith("The image is ready, but the rest of the prompt could not be sent."))
                self.assertLessEqual(len(final["content"]), 2000)
                self.assertTrue(final["suppress"])
                self.assertEqual(final["allowed_mentions"].to_dict()["parse"], [])
                self.assert_slots_free()

    async def test_duplicate_user_and_three_pending_limit_then_slots_are_reusable(self):
        ready, release = asyncio.Event(), asyncio.Event()
        entered = []

        async def generate(prompt, width, height):
            entered.append(prompt)
            if len(entered) == 3:
                ready.set()
            await release.wait()
            return self.png((width, height)), 83.2

        self.backend.generate.side_effect = generate
        interactions = [self.interaction(user_id=user) for user in (42, 84, 126)]
        tasks = [asyncio.create_task(self.bot.run_imagegen(item, str(item.user.id), 64, 64))
                 for item in interactions]
        try:
            await asyncio.wait_for(ready.wait(), 2)
            self.assertEqual(self.client.image_users, {42, 84, 126})
            self.assertEqual(self.client.image_tasks, set(tasks))
            for user, expected in ((42, "already"), (168, "queue is full")):
                rejected = self.interaction(user_id=user, server_id=2)
                await self.bot.run_imagegen(rejected, "Another image", 64, 64)
                response = rejected.response.send_message.call_args
                self.assertIn(expected, response.args[0])
                self.assertTrue(response.kwargs["ephemeral"])
                rejected.response.defer.assert_not_awaited()
                rejected.channel.send.assert_not_awaited()
            self.assertEqual(self.backend.generate.await_count, 3)
        finally:
            release.set()
            await asyncio.wait_for(asyncio.gather(*tasks), 2)
        self.assertTrue(all(len(item.uploads) == 1 for item in interactions))
        self.assert_slots_free()
        retry = self.interaction()
        await self.bot.run_imagegen(retry, "A new image", 64, 64)
        self.assertEqual(len(retry.uploads), 1)
        self.assert_slots_free()

    async def test_failed_acknowledgment_upload_or_generation_releases_slot_for_retry(self):
        for stage in ("defer", "progress", "acknowledgment", "upload", "backend", "timeout", "invalid_image"):
            with self.subTest(stage=stage):
                interaction = self.interaction()
                forbidden = discord.Forbidden(SimpleNamespace(status=403, reason="Forbidden"), "Missing permissions")
                self.backend.generate.side_effect = None
                self.backend.generate.return_value = self.png((1024, 1024)), 83.2
                if stage == "defer":
                    interaction.response.defer.side_effect = forbidden
                elif stage == "progress":
                    interaction.channel.send.side_effect = forbidden
                elif stage == "acknowledgment":
                    interaction.edit_original_response.side_effect = forbidden
                elif stage == "upload":
                    interaction.progress.edit.side_effect = forbidden
                elif stage == "backend":
                    self.backend.generate.side_effect = self.bot.ImageGenerationError("Synthetic backend failure")
                elif stage == "timeout":
                    self.backend.generate.side_effect = TimeoutError()
                else:
                    self.backend.generate.return_value = b"not an image", 83.2
                await self.bot.run_imagegen(interaction, "A tree", 64, 64)
                self.assertEqual(interaction.uploads, [])
                self.assert_slots_free()
                self.backend.generate.side_effect = None
                self.backend.generate.return_value = self.png((1024, 1024)), 83.2
                retry = self.interaction()
                await self.bot.run_imagegen(retry, "Try again", 64, 64)
                self.assertEqual(len(retry.uploads), 1)
                self.assert_slots_free()

    async def test_cancellation_explains_stop_and_releases_pending_slot(self):
        ready = asyncio.Event()

        async def generate(*args):
            ready.set()
            await asyncio.Event().wait()

        self.backend.generate.side_effect = generate
        interaction = self.interaction()
        task = asyncio.create_task(self.bot.run_imagegen(interaction, "A tree", 64, 64))
        try:
            await asyncio.wait_for(ready.wait(), 2)
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertIn("stopped", interaction.progress.edit.call_args.kwargs["content"])
        self.assertEqual(interaction.uploads, [])
        self.assert_slots_free()

    async def test_png_strips_workflow_metadata_without_resizing_or_changing_pixels(self):
        source = self.png((2048, 128), metadata=True)
        with Image.open(io.BytesIO(source)) as picture:
            self.assertIn("workflow", picture.info)
            self.assertIn("prompt", picture.info)
            pixels = picture.tobytes()
        result, name = self.bot.image_attachment(source, 2048, 128, len(source))
        self.assertEqual(name, "image.png")
        with Image.open(io.BytesIO(result)) as picture:
            self.assertEqual(picture.format, "PNG")
            self.assertEqual(picture.size, (2048, 128))
            self.assertEqual(picture.tobytes(), pixels)
            self.assertNotIn("workflow", picture.info)
            self.assertNotIn("prompt", picture.info)

    async def test_attachment_rejects_bad_format_wrong_dimensions_corruption_and_tiny_limit(self):
        jpeg = io.BytesIO()
        Image.new("RGB", (64, 64)).save(jpeg, format="JPEG")
        source = self.png()
        cases = ((jpeg.getvalue(), 64, 64, 100000, "format or resolution"),
                 (source, 80, 64, 100000, "format or resolution"),
                 (b"not an image", 64, 64, 100000, "unreadable"),
                 (source[:len(source) // 2], 64, 64, 100000, "unreadable"),
                 (source, 64, 64, 1, "attachment limit"))
        for data, width, height, limit, expected in cases:
            with self.subTest(expected=expected, width=width, limit=limit):
                with self.assertRaisesRegex(self.bot.ImageGenerationError, expected):
                    self.bot.image_attachment(data, width, height, limit)

    async def test_jpeg_fallback_fits_file_limit_preserves_dimensions_and_strips_metadata(self):
        source = self.png((128, 96), metadata=True, noise=True)
        self.assertGreater(len(source), 8000)
        result, name = self.bot.image_attachment(source, 128, 96, 8000)
        self.assertEqual(name, "image.jpg")
        self.assertLessEqual(len(result), 8000)
        with Image.open(io.BytesIO(result)) as picture:
            self.assertEqual(picture.format, "JPEG")
            self.assertEqual(picture.size, (128, 96))
            self.assertEqual(picture.mode, "RGB")
            self.assertNotIn("workflow", picture.info)
            self.assertNotIn("prompt", picture.info)

    async def test_disabled_feature_is_helpful_and_status_never_probes_backend(self):
        self.client.lm_client.models = SimpleNamespace(
            list=AsyncMock(side_effect=AssertionError("Status must not probe models")),
        )
        with patch.object(discord.Client, "latency", new_callable=PropertyMock, return_value=0.123):
            for configured in (False, True):
                with self.subTest(configured=configured):
                    self.client.imagegen = self.backend if configured else None
                    self.client.config.comfy_url = "http://comfy.invalid:8188" if configured else ""
                    interaction = self.interaction()
                    await self.bot.cmd_imagegen.callback(interaction)
                    if configured:
                        interaction.response.send_modal.assert_awaited_once()
                    else:
                        response = interaction.response.send_message.call_args
                        self.assertTrue(response.kwargs["ephemeral"])
                        self.assertIn("not configured", response.args[0])
                        self.assertIn("COMFYUI_BASE_URL", response.args[0])
                        interaction.response.send_modal.assert_not_awaited()
                    await self.bot.cmd_status.callback(interaction)
                    status = interaction.followup.send.call_args.args[0]
                    model = json.loads((fixtures.ROOT / "krea2.json").read_text())["316"]["inputs"]["unet_name"]
                    self.assertIn("**Image model:** " + (f"`{model.removesuffix('.safetensors')}` (1K–2K)" if configured else "Off"), status)
            # Read the current workflow each time, and keep status available when it cannot be read.
            changed = json.dumps({"316": {"class_type": "UNETLoader", "inputs": {"unet_name": "different-model.safetensors"}}})
            cases = ((changed, "`different-model` (1K–2K)"),
                     ("{}", "Configured (model unavailable)"),
                     ("invalid JSON", "Configured (model unavailable)"),
                     (FileNotFoundError(), "Configured (model unavailable)"))
            for source, expected in cases:
                with self.subTest(workflow=repr(source)):
                    interaction = self.interaction()
                    options = {"side_effect": source} if isinstance(source, Exception) else {"return_value": source}
                    with patch.object(self.bot.Path, "read_text", **options):
                        await self.bot.cmd_status.callback(interaction)
                    self.assertIn("**Image model:** " + expected, interaction.followup.send.call_args.args[0])
        self.backend.request.assert_not_awaited()
        self.backend.generate.assert_not_awaited()
        self.client.lm_client.models.list.assert_not_awaited()
        self.create.assert_not_awaited()


if __name__ == "__main__":
    unittest.main(verbosity=2)
