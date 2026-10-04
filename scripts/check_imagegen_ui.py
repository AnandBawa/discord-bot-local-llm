"""Offline checks for image-generation forms, permissions, queues, and uploads.

Run: venv_bot/bin/python scripts/check_imagegen_ui.py
Reuses temporary SQLite fixtures and blocked sockets; never reads .env/runtime
databases/logs or contacts Discord, LM Studio, or ComfyUI.
"""

import asyncio
from datetime import datetime, timedelta, timezone
import io
import random
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, PropertyMock, patch

import discord
from PIL import Image, PngImagePlugin

import check_regressions as fixtures


class ImagegenUIChecks(unittest.IsolatedAsyncioTestCase):
    asyncTearDown = fixtures.BotChecks.asyncTearDown
    completion = staticmethod(fixtures.BotChecks.completion)

    async def asyncSetUp(self):
        await fixtures.BotChecks.asyncSetUp(self)
        self.client.config.comfy_url = "http://comfy.invalid:8188"
        self.backend = self.client.imagegen = self.bot.ImageGeneration(self.client.config)
        self.backend.request = AsyncMock(side_effect=AssertionError("Backend requests disabled"))
        self.backend.generate = AsyncMock(return_value=self.png())
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

    async def test_resolution_then_private_button_then_prompt_keeps_selected_size(self):
        interaction = self.interaction()
        self.assertFalse(interaction.permissions.administrator)
        self.assertFalse(interaction.app_permissions.read_message_history)
        self.assertIsNone(self.bot.cmd_imagegen.default_permissions)
        await self.bot.cmd_imagegen.callback(interaction)
        resolution = interaction.response.send_modal.call_args.args[0]
        self.assertIsInstance(resolution, self.bot.ImageResolutionModal)
        self.assertEqual(len(resolution.children), 2)
        self.fill(resolution, width="1080", height="1920")
        await resolution.on_submit(interaction)
        response = interaction.response.send_message.call_args
        self.assertTrue(response.kwargs["ephemeral"])
        self.assertIn("1088 × 1920", response.args[0])
        self.assertIn("1080 × 1920", response.args[0])
        view = response.kwargs["view"]
        self.assertIsInstance(view, self.bot.ImagePromptView)
        self.assertTrue(await view.interaction_check(interaction))
        stranger = self.interaction(user_id=84)
        self.assertFalse(await view.interaction_check(stranger))
        self.assertTrue(stranger.response.send_message.call_args.kwargs["ephemeral"])
        self.assertIn("own image", stranger.response.send_message.call_args.args[0])
        interaction.response.send_modal.reset_mock()
        await view.children[0].callback(interaction)
        prompt = interaction.response.send_modal.call_args.args[0]
        self.assertIsInstance(prompt, self.bot.ImagePromptModal)
        self.assertEqual((prompt.width, prompt.height), (1088, 1920))
        self.assertEqual(prompt.prompt.style, discord.TextStyle.paragraph)
        self.assertEqual((prompt.prompt.min_length, prompt.prompt.max_length), (1, 4000))
        self.backend.generate.assert_not_awaited()
        interaction.channel.send.assert_not_awaited()
        self.assert_slots_free()

    async def test_resolution_rounding_boundaries_and_malformed_form_values(self):
        for requested, expected in (((64, 2048), (64, 2048)), ((71, 72), (64, 80)),
                                    ((1080, 1920), (1088, 1920)), ((2040, 2047), (2048, 2048))):
            with self.subTest(requested=requested):
                self.assertEqual(self.bot.image_resolution(*requested), expected)
        for invalid in ("", "wide", "64.5", "1e3", "0", "63", "2049", "-64"):
            for field in ("width", "height"):
                with self.subTest(field=field, value=invalid):
                    interaction = self.interaction()
                    values = {"width": "64", "height": "64", field: invalid}
                    modal = self.fill(self.bot.ImageResolutionModal(), **values)
                    await modal.on_submit(interaction)
                    response = interaction.response.send_message.call_args
                    self.assertTrue(response.kwargs["ephemeral"])
                    self.assertIn("64 and 2048", response.args[0])
                    self.assertNotIn("view", response.kwargs)
        self.backend.generate.assert_not_awaited()
        self.assert_slots_free()

    async def test_permissions_are_rechecked_at_each_submission_including_threads(self):
        for thread in (False, True):
            send_permission = "send_messages_in_threads" if thread else "send_messages"
            for missing in ("view_channel", send_permission, "attach_files"):
                for step in ("command", "resolution", "prompt"):
                    with self.subTest(thread=thread, missing=missing, step=step):
                        interaction = self.interaction(thread=thread)
                        setattr(interaction.app_permissions, missing, False)
                        if step == "command":
                            await self.bot.cmd_imagegen.callback(interaction)
                        elif step == "resolution":
                            modal = self.fill(self.bot.ImageResolutionModal(), width="64", height="64")
                            await modal.on_submit(interaction)
                        else:
                            modal = self.fill(self.bot.ImagePromptModal(64, 64), prompt="A tree")
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
        for invalid in (True, 64.0, "64", 63, 2049, None):
            cases.extend((("A tree", invalid, 64, "64 and 2048"),
                          ("A tree", 64, invalid, "64 and 2048")))
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
        literal += "x" * (4000 - len(literal) - 2) + "  "
        ready, release = asyncio.Event(), asyncio.Event()

        async def generate(prompt, width, height):
            ready.set()
            await release.wait()
            return self.png((width, height), metadata=True)

        async def require_live_token(*args, **kwargs):
            self.assertLess(datetime.now(timezone.utc) - interaction.created_at, timedelta(minutes=15))

        self.backend.generate.side_effect = generate
        interaction.edit_original_response.side_effect = require_live_token
        interaction.followup.send.side_effect = require_live_token
        modal = self.fill(self.bot.ImagePromptModal(80, 112), prompt=literal)
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
        self.backend.generate.assert_awaited_once_with(literal, 80, 112)
        self.create.assert_not_awaited()
        self.backend.request.assert_not_awaited()
        interaction.followup.send.assert_not_awaited()
        interaction.channel.send.assert_awaited_once()
        final = interaction.progress.edit.call_args
        self.assertEqual(final.kwargs["allowed_mentions"].to_dict()["parse"], [])
        self.assertNotIn(literal, final.kwargs["content"])
        self.assertIn("80 × 112", final.kwargs["content"])
        self.assertEqual(len(interaction.uploads), 1)
        name, data = interaction.uploads[0]
        self.assertEqual(name, "image.png")
        with Image.open(io.BytesIO(data)) as picture:
            self.assertEqual(picture.size, (80, 112))
            self.assertNotIn("workflow", picture.info)
            self.assertNotIn("prompt", picture.info)
        self.assert_slots_free()

    async def test_duplicate_user_and_three_pending_limit_then_slots_are_reusable(self):
        ready, release = asyncio.Event(), asyncio.Event()
        entered = []

        async def generate(prompt, width, height):
            entered.append(prompt)
            if len(entered) == 3:
                ready.set()
            await release.wait()
            return self.png((width, height))

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
                self.backend.generate.return_value = self.png()
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
                    self.backend.generate.return_value = b"not an image"
                await self.bot.run_imagegen(interaction, "A tree", 64, 64)
                self.assertEqual(interaction.uploads, [])
                self.assert_slots_free()
                self.backend.generate.side_effect = None
                self.backend.generate.return_value = self.png()
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
                    self.assertIn("**Image generation:** " + ("Configured" if configured else "Off"), status)
        self.backend.request.assert_not_awaited()
        self.backend.generate.assert_not_awaited()
        self.client.lm_client.models.list.assert_not_awaited()
        self.create.assert_not_awaited()


if __name__ == "__main__":
    unittest.main(verbosity=2)
