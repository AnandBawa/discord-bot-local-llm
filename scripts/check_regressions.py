"""Focused regression checks for URL fetching and persistent conversation history.

Run: venv_bot/bin/python scripts/check_regressions.py
Uses temporary SQLite files, synthetic model responses, and a controlled HTTP server.
No .env, Discord login, real model requests, or external socket connections.
"""

import asyncio
import contextlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import socket
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import ANY, AsyncMock, patch

import aiohappyeyeballs
import aiohttp
import aiosqlite


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOCKET_CONNECT = socket.socket.connect
START_CONNECTION = aiohappyeyeballs.start_connection

# Connected synthetic graph for offline tests; never read a user's local workflow.
WORKFLOW_FIXTURE = {
    "48": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": "Synthetic prompt"}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": ["48", 0], "clip": ["317", 0]}},
    "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "Synthetic negative", "clip": ["317", 0]}},
    "232": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
    "316": {"class_type": "UNETLoader", "inputs": {"unet_name": "synthetic-model.safetensors"}},
    "317": {"class_type": "CLIPLoader", "inputs": {"clip_name": "synthetic-encoder.safetensors"}},
    "210": {"class_type": "VAELoader", "inputs": {"vae_name": "synthetic-vae.safetensors"}},
    "265": {"class_type": "KSampler", "inputs": {"model": ["316", 0], "positive": ["6", 0],
            "negative": ["7", 0], "latent_image": ["232", 0], "steps": 8, "cfg": 1.0}},
    "323": {"class_type": "VAEDecode", "inputs": {"samples": ["265", 0], "vae": ["210", 0]}},
    "324": {"class_type": "ImageScaleBy", "inputs": {"image": ["323", 0], "scale_by": 0.5}},
    "213": {"class_type": "SaveImage", "inputs": {"images": ["324", 0], "filename_prefix": "synthetic"}},
}


class BotChecks(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="discord-bot-regressions-")
        self.old_directory = Path.cwd()
        os.chdir(self.directory.name)
        self.patches = contextlib.ExitStack()
        self.patches.enter_context(patch("dotenv.load_dotenv", side_effect=AssertionError("Import must not read .env")))
        self.patches.enter_context(patch("discord.Client.run", side_effect=AssertionError("Discord login disabled")))
        self.patches.enter_context(patch("socket.socket.connect", side_effect=AssertionError("External sockets disabled")))
        self.patches.enter_context(patch("socket.socket.connect_ex", side_effect=AssertionError("External sockets disabled")))
        self.create = AsyncMock(return_value=self.completion("Answer"))
        model = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self.create)), close=AsyncMock())
        self.old_logging = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        spec = importlib.util.spec_from_file_location("bot_regression_check", ROOT / "bot.py")
        self.bot = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.bot)
        # Workflow paths are resolved beside __file__; point them at temporary data.
        self.bot.__file__ = str(Path(self.directory.name) / "bot.py")
        Path("workflow.json").write_text(json.dumps(WORKFLOW_FIXTURE), encoding="utf-8")
        self.client = self.bot.client
        self.client.lm_client = model
        self.client.db_lock = asyncio.Lock()
        self.client.llm_queue = asyncio.Semaphore(3)
        self.client.db_conn = await aiosqlite.connect("check.sqlite3")
        await self.bot.init_db(self.client.db_conn)
        self.message = SimpleNamespace(author=SimpleNamespace(id=42), reply=AsyncMock())

    async def asyncTearDown(self):
        await self.client.close()
        logging.disable(self.old_logging)
        self.patches.close()
        os.chdir(self.old_directory)
        self.directory.cleanup()

    @staticmethod
    def completion(content, finish_reason="stop"):
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=content), finish_reason=finish_reason,
        )])

    @staticmethod
    def interaction(user_id=42, server_id=1, *, channel_id=None):
        resolved_channel_id = channel_id if channel_id is not None else server_id * 10
        return SimpleNamespace(
            guild_id=server_id, channel_id=resolved_channel_id,
            channel=SimpleNamespace(id=resolved_channel_id, send=AsyncMock(
                return_value=SimpleNamespace(edit=AsyncMock()),
            )),
            user=SimpleNamespace(id=user_id),
            permissions=SimpleNamespace(administrator=True),
            response=SimpleNamespace(defer=AsyncMock(), send_message=AsyncMock()),
            followup=SimpleNamespace(send=AsyncMock(return_value=SimpleNamespace(
                flags=SimpleNamespace(ephemeral=False), edit=AsyncMock(),
            ))),
            edit_original_response=AsyncMock(),
        )

    async def seed(self, count=2, server_id="channel:10"):
        async with self.bot.history_transaction():
            await self.client.db_conn.executemany(
                "INSERT INTO chat_history (server_id, role, content) VALUES (?, ?, ?)",
                [(server_id, "user", f"Synthetic message {i}") for i in range(count)],
            )

    async def count(self, table):
        cursor = await self.client.db_conn.execute(f"SELECT COUNT(*) FROM {table}")
        return (await cursor.fetchone())[0]

    async def clear_history(self):
        await self.bot.cmd_clear.callback(self.interaction())

    @contextlib.asynccontextmanager
    async def public_http(self, responses):
        """Run real aiohttp requests; route approved public addresses to our test server."""
        seen = []
        handlers = set()

        async def handle(reader, writer):
            handlers.add(asyncio.current_task())
            try:
                request = (await reader.readuntil(b"\r\n\r\n")).decode()
                path = request.split(" ")[1]
                seen.append(path)
                status, headers, body = responses.get(path, (404, {}, b"missing"))
                headers = {"Content-Length": str(len(body)), "Connection": "close", **headers}
                wire = f"HTTP/1.1 {status} Result\r\n" + "".join(f"{k}: {v}\r\n" for k, v in headers.items())
                writer.write(wire.encode() + b"\r\n" + body)
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()
                handlers.discard(asyncio.current_task())

        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]

        def connect(sock, address):
            if address[:2] != ("127.0.0.1", port):
                raise AssertionError(f"Uncontrolled connection: {address}")
            return SOCKET_CONNECT(sock, address)

        async def resolve(resolver, host, port=0, family=socket.AF_INET):
            address = "127.0.0.1" if host == "internal.example" else "93.184.216.34"
            return [{"hostname": host, "host": address, "port": port,
                     "family": socket.AF_INET, "proto": socket.IPPROTO_TCP, "flags": 0}]

        async def route(**kwargs):
            for info in kwargs["addr_infos"]:
                self.assertEqual(info[4][0], "93.184.216.34")
            kwargs["addr_infos"] = [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("127.0.0.1", port))]
            return await START_CONNECTION(**kwargs)

        try:
            with patch("socket.socket.connect", new=connect), \
                    patch("aiohttp.resolver.ThreadedResolver.resolve", new=resolve), \
                    patch("aiohttp.TCPConnector._get_ssl_context", return_value=None), \
                    patch("aiohappyeyeballs.start_connection", side_effect=route) as connections:
                yield seen, connections
        finally:
            server.close()
            await server.wait_closed()
            if handlers:
                await asyncio.gather(*handlers)

    async def test_nonpublic_urls_never_connect(self):
        hosts = ("127.0.0.1", "10.0.0.1", "172.16.0.1", "192.168.1.1", "169.254.169.254",
                 "0.0.0.0", "100.64.0.1", "224.0.0.1", "[::1]", "[fe80::1]", "[fc00::1]",
                 "[::ffff:127.0.0.1]", "127.1", "2130706433", "internal.example")
        async with self.public_http({}) as (seen, connections):
            for host in hosts:
                for path in ("/file.png", "/article"):
                    with self.subTest(host=host, path=path):
                        result = await self.bot.fetch_url_content(f"http://{host}{path}")
                        self.assertEqual(result["type"], "error")
            for url in ("file:///etc/hosts", "ftp://public.example/file.png", "http://user:pass@public.example/file.png"):
                self.assertEqual((await self.bot.fetch_url_content(url))["type"], "error")
            self.assertEqual(seen, [])
            connections.assert_not_called()

    async def test_public_downloads_and_public_redirects_work(self):
        responses = {
            "/ok.png": (200, {"Content-Type": "image/png"}, b"synthetic-image"),
            "/redirect.png": (302, {"Location": "http://public.example/ok.png"}, b""),
            "/https://public.example/article": (200, {"Content-Type": "text/plain"}, b"Synthetic article"),
        }
        async with self.public_http(responses):
            for path in ("ok.png", "redirect.png"):
                self.assertEqual(await self.bot.fetch_url_content(f"http://public.example/{path}"),
                                 {"type": "image", "data": b"synthetic-image"})
            self.assertEqual(await self.bot.fetch_url_content("https://public.example/article"),
                             {"type": "text", "data": "Synthetic article"})

    async def test_redirects_to_private_destinations_are_blocked(self):
        for destination in ("http://127.0.0.1/private", "http://[::1]/private", "http://internal.example/private"):
            with self.subTest(destination=destination):
                async with self.public_http({"/redirect.png": (302, {"Location": destination}, b"")}) as (seen, connections):
                    self.assertEqual((await self.bot.fetch_url_content("http://public.example/redirect.png"))["type"], "error")
                    self.assertEqual(seen, ["/redirect.png"])
                    self.assertEqual(connections.call_count, 1)

    async def test_dns_rebinding_and_mixed_dns_answers_are_blocked(self):
        connector_type = self.bot.PublicURLConnector
        public = {"hostname": "public.example", "host": "93.184.216.34", "port": 80,
                  "family": socket.AF_INET, "proto": socket.IPPROTO_TCP, "flags": 0}
        private = {**public, "host": "127.0.0.1"}
        for answers in (([public], [private]), ([public, private],)):
            with self.subTest(answers=answers), \
                    patch("aiohttp.resolver.ThreadedResolver.resolve", new=AsyncMock(side_effect=answers)), \
                    patch.object(self.bot, "PublicURLConnector", side_effect=lambda: connector_type(use_dns_cache=False)), \
                    patch("aiohappyeyeballs.start_connection", new=AsyncMock()) as connections:
                self.assertEqual((await self.bot.fetch_url_content("http://public.example/file.png"))["type"], "error")
                connections.assert_not_called()

    async def test_download_limit_and_dns_deadline(self):
        async with self.public_http({"/big.png": (200, {"Content-Type": "image/png"}, b"12345")}):
            with patch.object(self.bot, "MAX_FILE_SIZE", 4):
                self.assertEqual((await self.bot.fetch_url_content("http://public.example/big.png"))["type"], "error")
        async def stalled(*args, **kwargs):
            await asyncio.Event().wait()
        with patch("aiohttp.resolver.ThreadedResolver.resolve", new=stalled), patch.object(self.bot, "SCRAPER_TIMEOUT", 0.01):
            result = await self.bot.fetch_url_content("http://public.example/file.png")
            self.assertIn("too long", result["data"])

    async def test_new_database_creates_only_conversation_and_persona_tables(self):
        cursor = await self.client.db_conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        tables = {row[0] for row in await cursor.fetchall() if not row[0].startswith("sqlite_")}
        self.assertEqual(tables, {"server_config", "chat_history"})
        cursor = await self.client.db_conn.execute("PRAGMA table_info(chat_history)")
        self.assertEqual([row[1] for row in await cursor.fetchall()], ["id", "server_id", "role", "content"])

    async def test_eviction_drops_oldest_half_only_in_current_conversation(self):
        await self.seed(count=4, server_id="channel:20")
        await self.seed(count=98)
        await self.bot.save_and_send_response(self.message, "channel:10", "Hello", "Answer")
        cursor = await self.client.db_conn.execute(
            "SELECT role, content FROM chat_history WHERE server_id = 'channel:10' ORDER BY id",
        )
        self.assertEqual(await cursor.fetchall(),
                         [("user", f"Synthetic message {i}") for i in range(50, 98)]
                         + [("user", "Hello"), ("assistant", "Answer")])
        self.assertEqual(await self.count("chat_history"), 54)
        self.create.assert_not_awaited()
        self.message.reply.assert_awaited_once_with("Answer", allowed_mentions=ANY)

    async def test_clear_and_persona_changes_affect_only_current_conversation(self):
        await self.bot.cmd_role.callback(self.interaction(server_id=2), "Other persona")
        await self.seed(server_id="channel:20")
        for prompt in (None, "Friendly", "clear"):
            with self.subTest(prompt=prompt):
                await self.bot.cmd_role.callback(self.interaction(), "Original")
                await self.seed()
                interaction = self.interaction()
                if prompt is None:
                    await self.bot.cmd_clear.callback(interaction)
                else:
                    await self.bot.cmd_role.callback(interaction, prompt)
                expected = "Original" if prompt is None else (self.bot.DEFAULT_PERSONA if prompt == "clear" else prompt)
                self.assertEqual(await self.bot.get_persona("channel:10"), expected)
                self.assertEqual(await self.bot.get_persona("channel:20"), "Other persona")
                cursor = await self.client.db_conn.execute("SELECT server_id FROM chat_history")
                self.assertEqual(await cursor.fetchall(), [("channel:20",), ("channel:20",)])
                confirmation = (interaction.followup.send.call_args.args[0] if prompt is None else
                                interaction.followup.send.return_value.edit.call_args.kwargs["content"])
                self.assertIn("cleared", confirmation)
                self.assertNotIn("extraction", confirmation.lower())
        self.create.assert_not_awaited()

    async def test_role_changes_require_public_reply_before_mutating_shared_state(self):
        for prompt in ("Friendly", "clear", "@everyone ||persona|| 🙂 " * 200):
            with self.subTest(prompt=prompt[:30]):
                await self.bot.cmd_role.callback(self.interaction(), "Original")
                await self.seed()
                version = self.client.conversation_versions["channel:10"]
                interaction = self.interaction(user_id=84)
                # Installation context must not change the shared channel scope.
                interaction.guild = None
                interaction.permissions = self.bot.discord.Permissions.none()
                interaction.app_permissions = self.bot.discord.Permissions.none()
                announcement = SimpleNamespace(flags=SimpleNamespace(ephemeral=False), edit=AsyncMock())

                async def post(content, **kwargs):
                    self.assertEqual(await self.bot.get_persona("channel:10"), "Original")
                    self.assertEqual(await self.count("chat_history"), 2)
                    self.assertEqual(self.client.conversation_versions["channel:10"], version)
                    self.assertFalse(self.client.db_lock.locked())
                    self.assertLess(len(content), 2000)
                    self.assertFalse(kwargs["ephemeral"])
                    self.assertTrue(kwargs["wait"])
                    self.assertEqual(kwargs["allowed_mentions"].to_dict()["parse"], [])
                    return announcement

                interaction.followup.send.side_effect = post
                await self.bot.cmd_role.callback(interaction, prompt)
                persona = self.bot.DEFAULT_PERSONA if prompt == "clear" else prompt
                self.assertEqual(await self.bot.get_persona("channel:10"), persona)
                self.assertEqual(await self.count("chat_history"), 0)
                self.assertEqual(self.client.conversation_versions["channel:10"], version + 1)
                parts = [call.args[0] for call in interaction.followup.send.call_args_list]
                displayed = parts[0].split("**Requested Persona:**\n> ", 1)[1] + "".join(parts[1:])
                self.assertEqual(displayed, persona)
                interaction.channel.send.assert_not_awaited()
                interaction.response.defer.assert_awaited_once_with(ephemeral=False)
                announcement.edit.assert_awaited_once()
                content = announcement.edit.call_args.kwargs["content"]
                action = "Persona removed" if prompt == "clear" else "Saved persona"
                self.assertTrue(content.startswith(f"✅ {action} and history cleared!\n\n**Current Persona:**\n> "))
                self.assertEqual(content.split("**Current Persona:**\n> ", 1)[1] + "".join(parts[1:]), persona)
                self.assertLess(len(content), 2000)
                self.assertEqual(announcement.edit.call_args.kwargs["allowed_mentions"].to_dict()["parse"], [])

    async def test_private_or_unconfirmed_role_reply_preserves_shared_state(self):
        await self.bot.cmd_role.callback(self.interaction(), "Original")
        await self.seed()
        version = self.client.conversation_versions["channel:10"]
        for result in (SimpleNamespace(flags=SimpleNamespace(ephemeral=True)), None,
                       SimpleNamespace(), SimpleNamespace(flags=SimpleNamespace(ephemeral=None))):
            for prompt in ("Changed", "clear"):
                with self.subTest(result=result, prompt=prompt):
                    interaction = self.interaction()
                    interaction.followup.send.return_value = result
                    await self.bot.cmd_role.callback(interaction, prompt)
                    self.assertEqual(await self.bot.get_persona("channel:10"), "Original")
                    self.assertEqual(await self.count("chat_history"), 2)
                    self.assertEqual(self.client.conversation_versions["channel:10"], version)
                    self.assertIn("not changed", interaction.edit_original_response.call_args.kwargs["content"])
                    interaction.channel.send.assert_not_awaited()

    async def test_incomplete_public_persona_display_cannot_change_shared_state(self):
        await self.seed()
        results = [
            self.bot.discord.Forbidden(SimpleNamespace(status=403, reason="Forbidden"), "denied"),
            TimeoutError(), asyncio.CancelledError(), SimpleNamespace(flags=SimpleNamespace(ephemeral=True)),
        ]
        for result in results:
            with self.subTest(result=type(result).__name__):
                interaction = self.interaction()
                announcement = SimpleNamespace(flags=SimpleNamespace(ephemeral=False), edit=AsyncMock())
                interaction.followup.send.side_effect = [announcement, result]
                if isinstance(result, asyncio.CancelledError):
                    with self.assertRaises(asyncio.CancelledError):
                        await self.bot.cmd_role.callback(interaction, "Long persona " * 400)
                    interaction.edit_original_response.assert_not_awaited()
                else:
                    await self.bot.cmd_role.callback(interaction, "Long persona " * 400)
                    self.assertIn("not changed", interaction.edit_original_response.call_args.kwargs["content"])
                self.assertEqual(interaction.followup.send.await_count, 2)
                announcement.edit.assert_not_awaited()
                self.assertEqual(await self.count("server_config"), 0)
                self.assertEqual(await self.count("chat_history"), 2)
                self.assertEqual(self.client.conversation_versions, {})

    async def test_role_public_reply_failures_preserve_persona_history_and_version(self):
        await self.bot.cmd_role.callback(self.interaction(), "Original")
        await self.seed()
        version = self.client.conversation_versions["channel:10"]
        errors = [
            self.bot.discord.Forbidden(SimpleNamespace(status=403, reason="Forbidden"), "denied"),
            self.bot.discord.NotFound(SimpleNamespace(status=404, reason="Not Found"), "deleted thread"),
            self.bot.discord.HTTPException(SimpleNamespace(status=500, reason="Server Error"), "failed"),
            aiohttp.ClientConnectionError("disconnected"), OSError("connection lost"), TimeoutError(),
        ]
        for prompt in ("Changed", "clear"):
            for error in errors:
                with self.subTest(prompt=prompt, error=type(error).__name__):
                    interaction = self.interaction()
                    interaction.followup.send.side_effect = error
                    await self.bot.cmd_role.callback(interaction, prompt)
                    self.assertEqual(await self.bot.get_persona("channel:10"), "Original")
                    self.assertEqual(await self.count("chat_history"), 2)
                    self.assertEqual(self.client.conversation_versions["channel:10"], version)
                    self.assertIn("not changed", interaction.edit_original_response.call_args.kwargs["content"])

    async def test_cancelled_role_announcement_cannot_change_shared_state(self):
        await self.seed()
        entered = asyncio.Event()

        async def blocked_post(*args, **kwargs):
            entered.set()
            await asyncio.Event().wait()

        interaction = self.interaction()
        interaction.followup.send.side_effect = blocked_post
        task = asyncio.create_task(self.bot.cmd_role.callback(interaction, "Changed"))
        try:
            await asyncio.wait_for(entered.wait(), 2)
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(await self.count("server_config"), 0)
        self.assertEqual(await self.count("chat_history"), 2)
        self.assertEqual(self.client.conversation_versions, {})
        interaction.edit_original_response.assert_not_awaited()

    async def test_role_completion_edit_failure_retains_announced_change(self):
        for error in (TimeoutError(), self.bot.discord.Forbidden(
                SimpleNamespace(status=403, reason="Forbidden"), "denied")):
            with self.subTest(error=type(error).__name__):
                interaction = self.interaction()
                interaction.followup.send.return_value.edit.side_effect = error
                await self.seed()
                await self.bot.cmd_role.callback(interaction, "Changed")
                self.assertEqual(await self.bot.get_persona("channel:10"), "Changed")
                self.assertEqual(await self.count("chat_history"), 0)
                self.assertEqual(interaction.followup.send.await_count, 2)
                self.assertIn("Saved persona", interaction.followup.send.call_args.args[0])
                self.assertNotIn("not changed", interaction.followup.send.call_args.args[0])
                self.assertTrue(interaction.followup.send.call_args.kwargs["ephemeral"])

    async def test_role_failed_database_change_does_not_announce_completion(self):
        await self.seed()
        await self.client.db_conn.execute(
            "CREATE TEMP TRIGGER fail_persona BEFORE INSERT ON server_config "
            "BEGIN SELECT RAISE(ABORT, 'synthetic failure'); END",
        )
        interaction = self.interaction()
        with self.assertRaises(aiosqlite.IntegrityError):
            await self.bot.cmd_role.callback(interaction, "Changed")
        self.assertEqual(await self.count("server_config"), 0)
        self.assertEqual(await self.count("chat_history"), 2)
        self.assertEqual(self.client.conversation_versions, {})
        interaction.followup.send.assert_awaited_once()
        interaction.followup.send.return_value.edit.assert_not_awaited()

    async def test_role_view_needs_no_public_post_and_keeps_history(self):
        await self.bot.cmd_role.callback(self.interaction(), "Original")
        await self.seed()
        version = self.client.conversation_versions["channel:10"]
        interaction = self.interaction()
        interaction.followup.send.return_value = SimpleNamespace(flags=SimpleNamespace(ephemeral=True))
        await self.bot.cmd_role.callback(interaction)
        self.assertIn("Original", interaction.followup.send.call_args.args[0])
        interaction.response.defer.assert_awaited_once_with(ephemeral=False)
        interaction.channel.send.assert_not_awaited()
        self.assertEqual(await self.count("chat_history"), 2)
        self.assertEqual(self.client.conversation_versions["channel:10"], version)

    async def test_role_view_unwraps_old_confirmations_without_writing_state(self):
        persona = "You are a neutral AI.\nKeep answers short.\n*Keep this emphasis.*"
        saved = f"✅ Saved persona and history cleared!\n\n**Current Persona:**\n> {persona}"
        wrapped = f"**Current Persona:**\n> *{saved}*"
        for guild_id, channel_id, key, stored in (
            (1, 10, "channel:10", saved),
            (None, 9420, "dm:42", wrapped),
        ):
            with self.subTest(scope=key):
                await self.client.db_conn.execute(
                    "INSERT INTO server_config VALUES (?, ?)", (key, stored),
                )
                await self.seed(server_id=key)
                cursor = await self.client.db_conn.execute("SELECT * FROM chat_history ORDER BY id")
                history = await cursor.fetchall()
                versions = dict(self.client.conversation_versions)
                for _ in range(2):
                    interaction = self.interaction(server_id=guild_id, channel_id=channel_id)
                    await self.bot.cmd_role.callback(interaction)
                    interaction.followup.send.assert_awaited_once_with(
                        f"**Current Persona:**\n> *{persona}*", ephemeral=guild_id is None,
                        allowed_mentions=ANY,
                    )
                    self.assertEqual(await self.bot.get_persona(key), persona)
                    cursor = await self.client.db_conn.execute(
                        "SELECT prompt FROM server_config WHERE server_id = ?", (key,),
                    )
                    self.assertEqual((await cursor.fetchone())[0], stored)
                    cursor = await self.client.db_conn.execute("SELECT * FROM chat_history ORDER BY id")
                    self.assertEqual(await cursor.fetchall(), history)
                    self.assertEqual(self.client.conversation_versions, versions)

    async def test_role_save_unwraps_reply_but_keeps_public_change_requirement(self):
        # A wrapped persona literally named 'clear' must not become a reset command.
        for persona in ("You are a neutral AI.", "clear", self.bot.DEFAULT_PERSONA):
            with self.subTest(persona=persona):
                action = "Persona removed" if persona == self.bot.DEFAULT_PERSONA else "Saved persona"
                copied_reply = f"✅ {action} and history cleared!\n\n**Current Persona:**\n> {persona}"
                await self.bot.cmd_role.callback(self.interaction(), "Original")
                await self.seed()
                interaction = self.interaction()
                interaction.followup.send.return_value.flags.ephemeral = True
                await self.bot.cmd_role.callback(interaction, copied_reply)
                self.assertEqual(await self.bot.get_persona("channel:10"), "Original")
                self.assertEqual(await self.count("chat_history"), 2)

                interaction = self.interaction()
                await self.bot.cmd_role.callback(interaction, copied_reply)
                cursor = await self.client.db_conn.execute(
                    "SELECT prompt FROM server_config WHERE server_id = 'channel:10'",
                )
                self.assertEqual((await cursor.fetchone())[0], persona)
                self.assertEqual(await self.count("chat_history"), 0)
                confirmation = interaction.followup.send.return_value.edit.call_args.kwargs["content"]
                self.assertEqual(confirmation,
                                 f"✅ Saved persona and history cleared!\n\n**Current Persona:**\n> {persona}")

    async def test_persona_reply_cleanup_preserves_ordinary_text_and_incomplete_wrappers(self):
        header = "✅ Saved persona and history cleared!\n\n**Current Persona:**\n> "
        for persona in (
            "  A neutral AI.\n> Keep this quote.\n*Keep these asterisks.*  ",
            "Explain this example:\n" + header + "Example persona",
            "✅ Saved persona and history cleared!\nKeep this literal sentence.",
            "**Current Persona:**\n> Literal text without the view's closing markup",
            "**Current Persona:**\n> **",
            header,
        ):
            with self.subTest(persona=persona):
                await self.client.db_conn.execute(
                    "INSERT OR REPLACE INTO server_config VALUES ('channel:10', ?)", (persona,),
                )
                self.assertEqual(await self.bot.get_persona("channel:10"), persona)

    async def test_failed_clear_rolls_back_history(self):
        await self.seed()
        await self.client.db_conn.execute(
            "CREATE TEMP TRIGGER fail_delete BEFORE DELETE ON chat_history BEGIN SELECT RAISE(ABORT, 'synthetic failure'); END",
        )
        with self.assertRaises(aiosqlite.IntegrityError):
            await self.clear_history()
        self.assertEqual(await self.count("chat_history"), 2)
        await self.client.db_conn.execute("DROP TRIGGER fail_delete")
        await self.clear_history()
        self.assertEqual(await self.count("chat_history"), 0)

    async def test_failed_eviction_rolls_back_new_turn_and_keeps_existing_history(self):
        await self.seed(count=98)
        await self.client.db_conn.execute(
            "CREATE TEMP TRIGGER fail_delete BEFORE DELETE ON chat_history BEGIN SELECT RAISE(ABORT, 'synthetic failure'); END",
        )
        with self.assertRaises(aiosqlite.IntegrityError):
            await self.bot.save_and_send_response(self.message, "channel:10", "New input", "New answer")
        self.assertEqual(await self.count("chat_history"), 98)
        self.message.reply.assert_not_awaited()

    async def test_cancelled_history_transaction_rolls_back_partial_turn(self):
        entered = asyncio.Event()
        async def interrupted_write():
            async with self.bot.history_transaction():
                await self.client.db_conn.execute(
                    "INSERT INTO chat_history (server_id, role, content) VALUES ('channel:10', 'user', 'Partial')",
                )
                entered.set()
                await asyncio.Event().wait()
        task = asyncio.create_task(interrupted_write())
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(await self.count("chat_history"), 0)
        await self.bot.save_and_send_response(self.message, "channel:10", "Next input", "Next answer")
        self.assertEqual(await self.count("chat_history"), 2)

    async def test_restart_retains_recent_conversation_and_persona_without_model_calls(self):
        await self.bot.cmd_role.callback(self.interaction(), "Persisted persona")
        await self.bot.save_and_send_response(self.message, "channel:10", "Project Orion", "Orion confirmed")
        await self.client.db_conn.close()
        self.client.db_conn = await aiosqlite.connect("check.sqlite3")
        await self.bot.init_db(self.client.db_conn)
        context = await self.bot.build_ai_context("channel:10", "What name?")
        self.assertIn("Persisted persona", context[0]["content"])
        self.assertEqual(context[1:], [
            {"role": "user", "content": "Project Orion"},
            {"role": "assistant", "content": "Orion confirmed"},
            {"role": "user", "content": "What name?"},
        ])
        self.create.assert_not_awaited()

    async def test_legacy_history_and_persona_work_without_reading_or_mutating_fact_tables(self):
        await self.client.db_conn.close()
        self.client.db_conn = await aiosqlite.connect("legacy.sqlite3")
        await self.client.db_conn.executescript("""
            CREATE TABLE server_config (server_id TEXT PRIMARY KEY, prompt TEXT);
            CREATE TABLE chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT, server_id TEXT, role TEXT, content TEXT,
                user_id TEXT, user_name TEXT);
            CREATE TABLE pending_memories (
                id INTEGER PRIMARY KEY, server_id TEXT, role TEXT, content TEXT,
                user_id TEXT, user_name TEXT);
            CREATE TABLE explicit_memories (
                id TEXT PRIMARY KEY, server_id TEXT NOT NULL, user_id TEXT NOT NULL,
                user_name TEXT NOT NULL, document TEXT NOT NULL, indexed INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL);
            INSERT INTO server_config VALUES ('channel:10', 'Legacy persona');
            INSERT INTO chat_history VALUES (1, 'channel:10', 'user', 'Recent project Orion', '42', 'Tester');
            INSERT INTO chat_history VALUES (2, 'channel:10', 'assistant', 'Orion confirmed', '42', 'Tester');
            INSERT INTO pending_memories VALUES (100, 'channel:10', 'user', 'Unused pending fact', '42', 'Tester');
            INSERT INTO explicit_memories VALUES ('saved', 'channel:10', '42', 'Tester', 'Unused explicit fact', 0, '2026-01-01');
        """)
        snapshots = {}
        for table in ("pending_memories", "explicit_memories"):
            cursor = await self.client.db_conn.execute(f"SELECT * FROM {table}")
            snapshots[table] = await cursor.fetchall()
        legacy_index = Path("chroma_storage/chroma.sqlite3")
        legacy_index.parent.mkdir()
        legacy_index.write_bytes(b"synthetic legacy index; leave untouched")
        statements = []
        await self.client.db_conn.set_trace_callback(statements.append)
        try:
            await self.bot.init_db(self.client.db_conn)
            context = await self.bot.build_ai_context("channel:10", "What name?")
            self.assertIn("Legacy persona", context[0]["content"])
            self.assertEqual(context[1]["content"], "Recent project Orion")
            self.assertEqual(context[2]["content"], "Orion confirmed")
            self.assertNotIn("Unused", str(context))
            await self.bot.save_and_send_response(self.message, "channel:10", "Next question", "Next answer")
            cursor = await self.client.db_conn.execute(
                "SELECT user_id, user_name FROM chat_history ORDER BY id",
            )
            self.assertEqual(await cursor.fetchall(), [("42", "Tester"), ("42", "Tester"), (None, None), (None, None)])
            await self.clear_history()
            await self.bot.cmd_role.callback(self.interaction(), "Updated persona")
        finally:
            await self.client.db_conn.set_trace_callback(None)
        for table, rows in snapshots.items():
            self.assertFalse(any(table in statement.lower() for statement in statements), statements)
            cursor = await self.client.db_conn.execute(f"SELECT * FROM {table}")
            self.assertEqual(await cursor.fetchall(), rows)
        self.assertEqual(legacy_index.read_bytes(), b"synthetic legacy index; leave untouched")
        self.create.assert_not_awaited()

    async def test_clear_invalidates_every_members_old_reply_but_allows_new_turns(self):
        version = self.client.conversation_versions.get("channel:10", 0)
        await self.clear_history()
        for author_id in (42, 84):
            self.message.author.id = author_id
            await self.bot.save_and_send_response(self.message, "channel:10", "Old input", "Old answer", version)
        self.assertEqual(await self.count("chat_history"), 0)
        self.message.reply.assert_not_awaited()
        await self.bot.save_and_send_response(self.message, "channel:10", "New input", "New answer")
        self.assertEqual(await self.count("chat_history"), 2)
        self.message.reply.assert_awaited_once()

    async def test_message_handler_captures_version_before_loading_context(self):
        bot_user = SimpleNamespace(id=99)
        self.client._connection.user = bot_user
        message = SimpleNamespace(
            author=SimpleNamespace(id=84, bot=False, display_name="Other member"),
            guild=SimpleNamespace(id=1, name="Test guild"), channel=SimpleNamespace(id=10, name="general"),
            mentions=[bot_user], reference=None, content="<@99> Hello", attachments=[], stickers=[],
            reply=AsyncMock(),
        )
        entered, release = asyncio.Event(), asyncio.Event()
        async def context(*args):
            entered.set()
            await release.wait()
            return "Hello", [], [], ""
        with patch.object(self.bot, "extract_message_context", side_effect=context), \
                patch.object(self.bot, "build_ai_context", new=AsyncMock(return_value=[])), \
                patch.object(self.bot, "generate_ai_response", new=AsyncMock(return_value="Old answer")):
            request = asyncio.create_task(self.bot.on_message(message))
            try:
                await asyncio.wait_for(entered.wait(), 2)
                await self.clear_history()
            finally:
                release.set()
                await request
        self.assertEqual(await self.count("chat_history"), 0)
        message.reply.assert_not_awaited()

    async def test_clear_and_role_invalidate_old_replies(self):
        for prompt in (None, "Friendly", "clear"):
            with self.subTest(prompt=prompt):
                version = self.client.conversation_versions.get("channel:10", 0)
                if prompt is None:
                    await self.clear_history()
                else:
                    await self.bot.cmd_role.callback(self.interaction(), prompt)
                await self.bot.save_and_send_response(self.message, "channel:10", "Old", "Old answer", version)
                self.assertEqual(await self.count("chat_history"), 0)
                self.message.reply.assert_not_awaited()


if __name__ == "__main__":
    unittest.main(verbosity=2)
