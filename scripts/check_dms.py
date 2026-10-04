"""Offline checks for private conversations and queues shared with servers.

Run: venv_bot/bin/python scripts/check_dms.py
Uses synthetic users, temporary SQLite files, mocked models and blocked sockets.
Never loads .env, logs into Discord, or contacts LM Studio or ComfyUI.
"""

import asyncio
import io
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, PropertyMock, patch

import aiosqlite
import discord
from PIL import Image

import check_features as feature_fixtures
import check_imagegen_ui as image_fixtures
import check_regressions as fixtures


class DMChecks(unittest.IsolatedAsyncioTestCase):
    asyncTearDown = fixtures.BotChecks.asyncTearDown
    completion = staticmethod(fixtures.BotChecks.completion)
    seed = fixtures.BotChecks.seed
    count = fixtures.BotChecks.count
    png = staticmethod(image_fixtures.ImagegenUIChecks.png)
    fill = staticmethod(image_fixtures.ImagegenUIChecks.fill)

    async def asyncSetUp(self):
        await fixtures.BotChecks.asyncSetUp(self)
        self.create.return_value.choices[0].message.tool_calls = []

    def chat(self, server=1, author=42, content="Hello", history=True, channel_id=None):
        message = feature_fixtures.FeatureChecks.chat(self, server, author, content, history)
        if channel_id is not None:
            message.channel.id = channel_id
        return message

    def dm(self, author=42, content="Hello"):
        message = self.chat(author=author, content=content)
        previous_channel = message.channel
        message.guild = None
        message.mentions = []
        message.content = content
        message.channel = Mock(spec=discord.DMChannel)
        message.channel.id = author * 10 + 9000
        message.channel.type = discord.ChannelType.private
        message.channel.recipient = message.author
        message.channel.me = self.client.user
        message.channel.send = previous_channel.send
        message.channel.fetch_message = previous_channel.fetch_message
        message.channel.typing = previous_channel.typing
        return message

    def interaction(self, user_id=42, server_id=None, channel_id=None):
        fixture_server = 1 if server_id is None else server_id
        interaction = image_fixtures.ImagegenUIChecks.interaction(self, user_id, fixture_server)
        interaction.guild_id = server_id
        interaction.filesize_limit = 8_000_000
        interaction.channel_id = (user_id * 10 + 9000 if server_id is None else server_id * 10)
        if channel_id is not None:
            interaction.channel_id = channel_id
        if server_id is None:
            interaction.guild = None
            previous_send = interaction.channel.send
            interaction.channel = Mock(spec=discord.DMChannel)
            interaction.channel.type = discord.ChannelType.private
            interaction.channel.send = previous_send
            # Guild permission bits do not apply to a private conversation.
            interaction.app_permissions = discord.Permissions.none()
        interaction.channel.id = interaction.channel_id
        return interaction

    def enable_images(self):
        self.client.config.comfy_url = "http://comfy.invalid:8188"
        self.backend = self.client.imagegen = self.bot.ImageGeneration(self.client.config)
        self.backend.backend = "lmstudio"
        self.backend.request = AsyncMock(side_effect=AssertionError("Backend requests disabled"))
        self.backend.generate = AsyncMock(side_effect=lambda prompt, width, height: (
            self.png((width, height)), 12.3,
        ))

    @staticmethod
    def replies(message):
        return [call.args[0] for call in message.reply.call_args_list + message.channel.send.call_args_list]

    async def history(self, key):
        cursor = await self.client.db_conn.execute(
            "SELECT role, content FROM chat_history WHERE server_id = ? ORDER BY id", (key,),
        )
        return await cursor.fetchall()

    async def test_plain_dm_replies_without_a_mention_and_ignores_bots_and_groups(self):
        message = self.dm(content="An ordinary private question")
        await self.bot.on_message(message)
        self.assertEqual(self.replies(message), ["Answer"])
        self.create.assert_awaited_once()
        self.assertEqual(len(await self.history("dm:42")), 2)

        bot_message = self.dm(author=99)
        bot_message.author.bot = True
        group = self.dm(author=84)
        group.channel = Mock(spec=discord.GroupChannel)
        group.mentions = [self.client.user]
        unmentioned_server = self.chat(server=42)
        unmentioned_server.mentions = []
        for ignored in (bot_message, group, unmentioned_server):
            await self.bot.on_message(ignored)
            ignored.reply.assert_not_awaited()
        self.create.assert_awaited_once()

    async def test_dm_users_and_same_numbered_server_channels_keep_separate_contexts(self):
        scopes = ((42, None, None, "Private Alpha", "Alpha persona"),
                  (84, None, None, "Private Beta", "Beta persona"),
                  (42, 42, 42, "Shared Gamma", "Gamma persona"),
                  (42, 42, 84, "Shared Delta", "Delta persona"))
        for user, server, channel, text, persona in scopes:
            await self.bot.cmd_role.callback(self.interaction(user, server, channel), persona)
            message = (self.dm(user, text) if server is None else
                       self.chat(server, user, text, channel_id=channel))
            await self.bot.on_message(message)
            self.assertEqual(self.replies(message), ["Answer"])

        for user, server, channel, own_text, own_persona in scopes:
            message = (self.dm(user, "Continue") if server is None else
                       self.chat(server, user, "Continue", channel_id=channel))
            await self.bot.on_message(message)
            context = str(self.create.call_args.kwargs["messages"])
            self.assertIn(own_text, context)
            self.assertIn(own_persona, context)
            for _, _, _, other_text, other_persona in scopes:
                if other_text != own_text:
                    self.assertNotIn(other_text, context)
                    self.assertNotIn(other_persona, context)
        cursor = await self.client.db_conn.execute("SELECT DISTINCT server_id FROM chat_history")
        self.assertEqual({row[0] for row in await cursor.fetchall()},
                         {"dm:42", "dm:84", "channel:42", "channel:84"})

    async def test_numeric_legacy_records_stay_untouched_in_channels_threads_and_dms(self):
        async with self.bot.history_transaction():
            await self.client.db_conn.executemany(
                "INSERT INTO server_config VALUES (?, ?)",
                [(key, f"Legacy persona {key}") for key in ("42", "84")],
            )
            await self.client.db_conn.executemany(
                "INSERT INTO chat_history (server_id, role, content) VALUES (?, 'user', ?)",
                [(key, f"Legacy history {key}") for key in ("42", "84")],
            )
        legacy = {}
        for table in ("server_config", "chat_history"):
            cursor = await self.client.db_conn.execute(f"SELECT * FROM {table} ORDER BY server_id")
            legacy[table] = await cursor.fetchall()

        parent = self.chat(server=42, channel_id=42, content="Parent channel question")
        thread = self.chat(server=42, channel_id=84, content="Thread question")
        previous_channel = thread.channel
        thread.channel = Mock(spec=discord.Thread)
        thread.channel.id, thread.channel.parent_id = 84, 42
        thread.channel.parent, thread.channel.name = parent.channel, "private-thread-fixture"
        for attribute in ("send", "fetch_message", "typing", "permissions_for"):
            setattr(thread.channel, attribute, getattr(previous_channel, attribute))
        thread_interaction = self.interaction(42, 42, 84)
        thread_interaction.channel = thread.channel
        scopes = (("channel:42", parent, self.interaction(42, 42, 42)),
                  ("channel:84", thread, thread_interaction),
                  ("dm:42", self.dm(42, "Private question"), self.interaction(42)))
        statements = []
        await self.client.db_conn.set_trace_callback(statements.append)
        try:
            for key, message, interaction in scopes:
                await self.bot.on_message(message)
                self.assertEqual(self.replies(message), ["Answer"])
                context = str(self.create.call_args.kwargs["messages"])
                self.assertIn(self.bot.DEFAULT_PERSONA, context)
                self.assertNotIn("Legacy", context)
                await self.bot.cmd_status.callback(interaction)
                self.assertIn("**History:** 2/100 messages", interaction.followup.send.call_args.args[0])
                await self.bot.cmd_role.callback(interaction)
                self.assertIn(self.bot.DEFAULT_PERSONA, interaction.followup.send.call_args.args[0])

            for key, message, interaction in scopes:
                before = {other: await self.history(other) for other, _, _ in scopes if other != key}
                await self.bot.cmd_role.callback(interaction, f"Current persona for {key}")
                self.assertEqual(await self.history(key), [])
                for other, rows in before.items():
                    self.assertEqual(await self.history(other), rows)
                await self.bot.on_message(message)

            await self.client.db_conn.close()
            self.client.db_conn = await aiosqlite.connect("check.sqlite3")
            await self.client.db_conn.set_trace_callback(statements.append)
            await self.bot.init_db(self.client.db_conn)
            self.client.conversation_locks.clear()
            self.client.conversation_versions.clear()
            for key, message, interaction in scopes:
                await self.bot.cmd_role.callback(interaction)
                self.assertIn(f"Current persona for {key}", interaction.followup.send.call_args.args[0])
                await self.bot.cmd_status.callback(interaction)
                self.assertIn("**History:** 2/100 messages", interaction.followup.send.call_args.args[0])
                await self.bot.on_message(message)
                context = str(self.create.call_args.kwargs["messages"])
                self.assertIn(f"Current persona for {key}", context)
                self.assertNotIn("Legacy", context)
                for other, _, _ in scopes:
                    if other != key:
                        self.assertNotIn(f"Current persona for {other}", context)

            for key, message, interaction in scopes:
                before = {other: (await self.history(other), await self.bot.get_persona(other))
                          for other, _, _ in scopes if other != key}
                await self.bot.cmd_clear.callback(interaction)
                self.assertEqual(await self.history(key), [])
                self.assertEqual(await self.bot.get_persona(key), f"Current persona for {key}")
                await self.bot.on_message(message)
                await self.bot.cmd_role.callback(interaction, "clear")
                self.assertEqual(await self.history(key), [])
                self.assertEqual(await self.bot.get_persona(key), self.bot.DEFAULT_PERSONA)
                for other, (rows, persona) in before.items():
                    self.assertEqual(await self.history(other), rows)
                    self.assertEqual(await self.bot.get_persona(other), persona)
        finally:
            await self.client.db_conn.set_trace_callback(None)
        reads = [sql for sql in statements if sql.lstrip().upper().startswith("SELECT")
                 and any(table in sql for table in legacy)]
        self.assertTrue(reads)
        for sql in reads:
            self.assertRegex(sql, r"(?:channel|dm):(?:42|84)")
        for table, rows in legacy.items():
            cursor = await self.client.db_conn.execute(
                f"SELECT * FROM {table} WHERE server_id IN ('42', '84') ORDER BY server_id",
            )
            self.assertEqual(await cursor.fetchall(), rows)

    async def test_command_schemas_enable_servers_and_bot_dms_and_dm_help_needs_no_mention(self):
        commands = {command.name: command.to_dict(self.bot.tree) for command in self.bot.tree.get_commands()}
        self.assertEqual(set(commands), {"help", "status", "role", "clear", "imagegen"})
        for name, schema in commands.items():
            with self.subTest(command=name):
                self.assertEqual(schema["contexts"], [0, 1])
                self.assertTrue(schema["dm_permission"])
        self.assertEqual(commands["clear"]["default_member_permissions"],
                         discord.Permissions(manage_messages=True).value)
        interaction = self.interaction()
        await self.bot.cmd_help.callback(interaction)
        text = interaction.response.send_message.call_args.args[0]
        self.assertIn("no mention needed", text.lower())
        for name in commands:
            self.assertIn(f"/{name}", text)
        self.create.assert_not_awaited()

    async def test_dm_history_and_personas_survive_reopening_sqlite(self):
        for user, marker in ((42, "Alpha"), (84, "Beta")):
            await self.bot.cmd_role.callback(self.interaction(user), f"{marker} persona")
            await self.bot.on_message(self.dm(user, f"{marker} project"))
        requests_before_restart = self.create.await_count
        await self.client.db_conn.close()
        self.client.db_conn = await aiosqlite.connect("check.sqlite3")
        await self.bot.init_db(self.client.db_conn)
        self.client.conversation_locks.clear()
        self.client.conversation_versions.clear()
        for user, marker, other in ((42, "Alpha", "Beta"), (84, "Beta", "Alpha")):
            interaction = self.interaction(user)
            await self.bot.cmd_role.callback(interaction)
            self.assertIn(f"{marker} persona", interaction.followup.send.call_args.args[0])
            context = await self.bot.build_ai_context(f"dm:{user}", "Continue")
            self.assertIn(f"{marker} persona", context[0]["content"])
            self.assertIn(f"{marker} project", str(context))
            self.assertNotIn(other, str(context))
            self.assertEqual(len(await self.history(f"dm:{user}")), 2)
        self.assertEqual(self.create.await_count, requests_before_restart)

    async def test_dm_role_changes_stay_private_and_need_no_public_announcement(self):
        await self.seed(server_id="channel:420")
        await self.seed(server_id="dm:84")
        for prompt in ("Private persona", "clear"):
            with self.subTest(prompt=prompt):
                await self.seed(server_id="dm:42")
                interaction = self.interaction()
                interaction.channel.send.side_effect = AssertionError("No public announcement in a DM")
                await self.bot.cmd_role.callback(interaction, prompt)
                interaction.channel.send.assert_not_awaited()
                interaction.response.defer.assert_awaited_once_with(ephemeral=True)
                self.assertTrue(interaction.followup.send.call_args.kwargs["ephemeral"])
                expected = self.bot.DEFAULT_PERSONA if prompt == "clear" else prompt
                self.assertIn(expected, interaction.followup.send.call_args.args[0])
                self.assertEqual(await self.bot.get_persona("dm:42"), expected)
                self.assertEqual(await self.history("dm:42"), [])
                self.assertEqual(len(await self.history("dm:84")), 2)
                self.assertEqual(len(await self.history("channel:420")), 2)

    async def test_status_counts_only_the_callers_dm_history(self):
        for key, count in (("dm:42", 2), ("dm:84", 4), ("channel:420", 6)):
            await self.seed(count=count, server_id=key)
        with patch.object(discord.Client, "latency", new_callable=PropertyMock, return_value=0.1):
            for user, server, count in ((42, None, 2), (84, None, 4), (42, 42, 6)):
                interaction = self.interaction(user, server)
                await self.bot.cmd_status.callback(interaction)
                self.assertIn(f"**History:** {count}/100 messages", interaction.followup.send.call_args.args[0])
        self.create.assert_not_awaited()

    async def test_clear_and_role_invalidate_only_current_dms_running_and_queued_turns(self):
        for prompt in (None, "New private persona", "clear"):
            with self.subTest(prompt=prompt):
                await self.bot.cmd_role.callback(self.interaction(), "Original private persona")
                await self.bot.cmd_role.callback(self.interaction(84), "Other private persona")
                await self.bot.cmd_role.callback(self.interaction(42, 42), "Server persona")
                first, queued = self.dm(content="Old question"), self.dm(content="Queued question")
                peers = [self.dm(84), self.chat(server=42)]
                entered, peers_entered, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
                peer_count = 0

                async def generate(context, message, has_media):
                    nonlocal peer_count
                    if message is first:
                        entered.set()
                    else:
                        peer_count += 1
                        if peer_count == 2:
                            peers_entered.set()
                    await release.wait()
                    return "Old answer" if message is first else "Peer answer"

                with patch.object(self.bot, "generate_ai_response", side_effect=generate) as model:
                    tasks = [asyncio.create_task(self.bot.on_message(first))]
                    try:
                        await asyncio.wait_for(entered.wait(), 2)
                        tasks += [asyncio.create_task(self.bot.on_message(item)) for item in (queued, *peers)]
                        await asyncio.wait_for(peers_entered.wait(), 2)
                        if prompt is None:
                            await self.bot.cmd_clear.callback(self.interaction())
                        else:
                            await self.bot.cmd_role.callback(self.interaction(), prompt)
                    finally:
                        release.set()
                        await asyncio.gather(*tasks)
                    self.assertEqual(model.await_count, 3)
                self.assertEqual(await self.history("dm:42"), [])
                self.assertEqual(self.replies(first), [])
                self.assertEqual(self.replies(queued), [])
                for key, peer in zip(("dm:84", "channel:420"), peers):
                    self.assertEqual((await self.history(key))[-1], ("assistant", "Peer answer"))
                    self.assertEqual(self.replies(peer), ["Peer answer"])
                expected = ("Original private persona" if prompt is None else
                            self.bot.DEFAULT_PERSONA if prompt == "clear" else prompt)
                self.assertEqual(await self.bot.get_persona("dm:42"), expected)
                self.assertEqual(await self.bot.get_persona("dm:84"), "Other private persona")
                self.assertEqual(await self.bot.get_persona("channel:420"), "Server persona")
                fresh = self.dm(content="A new question")
                await self.bot.on_message(fresh)
                self.assertEqual(self.replies(fresh), ["Answer"])
                self.assertEqual(len(await self.history("dm:42")), 2)

    async def test_dm_turns_serialize_while_another_dm_and_server_progress(self):
        first, second = self.dm(content="Project Orion"), self.dm(content="What name?")
        peers = [self.dm(84), self.chat(server=42)]
        entered, peers_done, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        peer_count = 0
        second_context = []

        async def generate(context, message, has_media):
            nonlocal peer_count
            if message is first:
                entered.set()
                await release.wait()
                return "Orion confirmed"
            if message is second:
                second_context.extend(context)
            else:
                peer_count += 1
                if peer_count == 2:
                    peers_done.set()
            return "Answer"

        with patch.object(self.bot, "generate_ai_response", side_effect=generate):
            tasks = [asyncio.create_task(self.bot.on_message(first))]
            try:
                await asyncio.wait_for(entered.wait(), 2)
                tasks += [asyncio.create_task(self.bot.on_message(item)) for item in (second, *peers)]
                await asyncio.wait_for(peers_done.wait(), 2)
                self.assertEqual(second_context, [])
            finally:
                release.set()
                await asyncio.gather(*tasks)
        self.assertIn("Orion confirmed", str(second_context))
        self.assertEqual(len(await self.history("dm:42")), 4)

    async def test_dm_image_and_pdf_use_media_pipeline_and_save_only_text(self):
        image_data = self.png((2, 2))
        with self.bot.pymupdf.open() as document:
            document.new_page().insert_text((72, 72), "Private synthetic PDF")
            pdf_data = document.tobytes()
        message = self.dm(content="Describe these files")
        message.attachments = [
            SimpleNamespace(filename="private.png", content_type="image/png", size=len(image_data),
                            read=AsyncMock(return_value=image_data)),
            SimpleNamespace(filename="private.pdf", content_type="application/pdf", size=len(pdf_data),
                            read=AsyncMock(return_value=pdf_data)),
        ]
        await self.bot.on_message(message)
        context = self.create.call_args.kwargs["messages"]
        self.assertIn("Private synthetic PDF", str(context))
        image_parts = [part for item in context if isinstance(item["content"], list)
                       for part in item["content"] if part["type"] == "image_url"]
        self.assertEqual(len(image_parts), 1)
        self.assertTrue(image_parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        history = await self.history("dm:42")
        self.assertIn("private.png", str(history))
        self.assertIn("private.pdf", str(history))
        self.assertNotIn("base64", str(history))
        self.assertEqual(await self.history("channel:420"), [])
        self.assertEqual(self.replies(message), ["Answer"])

    async def test_text_file_only_dm_reads_contents_and_keeps_only_attachment_note(self):
        message = self.dm(content="")
        body = "Answer this private question from my uploaded text file."
        data = body.encode("utf-8")
        message.attachments = [SimpleNamespace(filename="message.txt", content_type="text/plain",
                                               size=len(data), read=AsyncMock(return_value=data))]
        await self.bot.on_message(message)
        self.assertIn(body, str(self.create.call_args.kwargs["messages"]))
        self.assertEqual(self.replies(message), ["Answer"])
        saved = str(await self.history("dm:42"))
        self.assertIn("message.txt", saved)
        self.assertNotIn(body, saved)

    async def test_text_truncation_notifies_before_inference_once_for_direct_and_replied_files(self):
        for location in ("dm", "server", "reply"):
            for over_limit in (False, True):
                with self.subTest(location=location, over_limit=over_limit):
                    message = self.dm(content="") if location == "dm" else self.chat(content="")
                    body = "é" * self.bot.MAX_TEXT_EXTRACTION_LENGTH
                    data = (body + ("OMITTED TAIL" if over_limit else "")).encode("utf-8")
                    attachment = SimpleNamespace(filename="message.txt", content_type="text/plain", size=len(data),
                                                 read=AsyncMock(return_value=data))
                    message.attachments = [attachment]
                    if location == "reply":
                        source = SimpleNamespace(author=SimpleNamespace(id=5, display_name="Other"), content="",
                                                 attachments=[attachment], stickers=[])
                        message.reference = SimpleNamespace(message_id=1, resolved=source, cached_message=None)

                    async def infer(**kwargs):
                        replies = self.replies(message)
                        self.assertEqual(len(replies), int(over_limit))
                        if over_limit:
                            self.assertIn("Text file truncated", replies[0])
                            self.assertIn("40,000 characters", replies[0])
                            self.assertIn("remaining text is skipped", replies[0])
                        content = kwargs["messages"][-1]["content"]
                        self.assertIn(body, content)
                        self.assertNotIn("OMITTED TAIL", content)
                        self.assertEqual("[Content Truncated]" in content, over_limit)
                        return self.create.return_value

                    self.create.side_effect = infer
                    await self.bot.on_message(message)
                    replies = self.replies(message)
                    self.assertEqual(len(replies), 1 + int(over_limit))
                    self.assertEqual(replies[-1], "Answer")

    async def test_mixed_dm_and_server_chats_share_three_processing_slots(self):
        self.enable_images()
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
                return self.create.return_value
            finally:
                active -= 1

        self.create.side_effect = infer
        messages = [self.dm(42), self.dm(84), self.chat(server=42), self.chat(server=84)]
        tasks = [asyncio.create_task(self.bot.on_message(message)) for message in messages]
        try:
            await asyncio.wait_for(entered.wait(), 2)
            self.assertEqual(self.create.await_count, 3)
            self.assertTrue(all(not task.done() for task in tasks))
            for server in (None, 42):
                interaction = self.interaction(126, server)
                await self.bot.cmd_imagegen.callback(interaction)
                self.assertIn("Chat is active right now", interaction.response.send_message.call_args.args[0])
                interaction.response.send_modal.assert_not_awaited()
        finally:
            release.set()
            await asyncio.gather(*tasks)
        self.assertEqual(self.create.await_count, 4)
        self.assertEqual(peak, 3)
        self.assertTrue(all(self.replies(message) == ["Answer"] for message in messages))
        self.assertEqual(self.backend.work, {"lmstudio": 0, "comfyui": 0})
        self.backend.generate.assert_not_awaited()
        self.backend.request.assert_not_awaited()

    async def test_dm_and_server_images_block_all_chat_before_discord_acknowledges(self):
        self.enable_images()
        cloud = AsyncMock()
        self.client.fallback_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=cloud)), close=AsyncMock(),
        )
        self.client.chat_dead_until = float("inf")
        for server in (None, 42):
            with self.subTest(image_server=server):
                entered, release = asyncio.Event(), asyncio.Event()
                interaction = self.interaction(126, server)

                async def defer(**kwargs):
                    entered.set()
                    await release.wait()

                interaction.response.defer.side_effect = defer
                task = asyncio.create_task(self.bot.run_imagegen(interaction, "A tree", 1024, 1024))
                try:
                    await asyncio.wait_for(entered.wait(), 2)
                    for message in (self.dm(42), self.dm(84), self.chat(server=42), self.chat(server=84)):
                        await self.bot.on_message(message)
                        replies = self.replies(message)
                        self.assertEqual(len(replies), 1)
                        self.assertIn("Image generation is active right now", replies[0])
                    self.assertEqual(await self.count("chat_history"), 0)
                    self.create.assert_not_awaited()
                    cloud.assert_not_awaited()
                finally:
                    release.set()
                    await task
                self.assertEqual(len(interaction.uploads), 1)
                self.assertEqual(self.client.image_users, set())
                self.assertEqual(self.backend.work, {"lmstudio": 0, "comfyui": 0})

    async def test_dm_and_server_images_share_three_total_and_one_per_user(self):
        self.enable_images()
        entered, release = asyncio.Event(), asyncio.Event()
        prompts = []

        async def generate(prompt, width, height):
            prompts.append(prompt)
            if len(prompts) == 3:
                entered.set()
            await release.wait()
            return self.png((width, height)), 12.3

        self.backend.generate.side_effect = generate
        admitted = [self.interaction(42), self.interaction(84, 42), self.interaction(126)]
        tasks = [asyncio.create_task(self.bot.run_imagegen(item, str(item.user.id), 1024, 1024))
                 for item in admitted]
        try:
            await asyncio.wait_for(entered.wait(), 2)
            self.assertEqual(self.client.image_users, {42, 84, 126})
            for user, server, expected in ((42, 84, "already"), (84, None, "already"),
                                           (168, None, "queue is full"), (210, 84, "queue is full")):
                rejected = self.interaction(user, server)
                await self.bot.run_imagegen(rejected, "Another image", 1024, 1024)
                self.assertIn(expected, rejected.response.send_message.call_args.args[0])
                rejected.response.defer.assert_not_awaited()
                rejected.channel.send.assert_not_awaited()
            self.assertEqual(self.backend.generate.await_count, 3)
        finally:
            release.set()
            await asyncio.gather(*tasks)
        self.assertTrue(all(len(item.uploads) == 1 for item in admitted))
        self.assertEqual(self.client.image_users, set())
        self.assertEqual(self.client.image_tasks, set())
        retry = self.interaction(42, 84)
        await self.bot.run_imagegen(retry, "Same user in a server after finishing", 1024, 1024)
        self.assertEqual(len(retry.uploads), 1)
        self.create.assert_not_awaited()

    async def test_dm_image_form_uses_interaction_upload_limit_and_delivers_without_a_guild(self):
        self.enable_images()
        self.assertFalse(self.bot.cmd_imagegen.guild_only)
        opening = self.interaction()
        await self.bot.cmd_imagegen.callback(opening)
        opening.response.send_modal.assert_awaited_once()
        modal = opening.response.send_modal.call_args.args[0]
        self.fill(modal, width="1024", height="1024", prompt="A private moonlit garden")
        submission = self.interaction()
        submission.filesize_limit = 123_456
        with patch.object(self.bot, "image_attachment", wraps=self.bot.image_attachment) as attachment:
            await modal.on_submit(submission)
        self.assertEqual(attachment.call_args.args[3], submission.filesize_limit)
        self.assertEqual(len(submission.uploads), 1)
        name, data = submission.uploads[0]
        self.assertTrue(name.startswith("SPOILER_"))
        self.assertLessEqual(len(data), submission.filesize_limit)
        with Image.open(io.BytesIO(data)) as picture:
            self.assertEqual(picture.size, (1024, 1024))
        self.assertIn("Generated in 12.3s", submission.progress.edit.call_args.kwargs["content"])
        self.assertIn("||A private moonlit garden||", submission.progress.edit.call_args.kwargs["content"])
        self.create.assert_not_awaited()
        self.backend.request.assert_not_awaited()


if __name__ == "__main__":
    unittest.main(verbosity=2)
