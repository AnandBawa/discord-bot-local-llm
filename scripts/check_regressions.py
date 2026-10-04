"""Focused regression checks for URL fetching and persistent conversation history.

Run: venv_bot/bin/python scripts/check_regressions.py
Uses temporary SQLite files, synthetic model responses, and a controlled HTTP server.
No .env, Discord login, real model requests, or external socket connections.
"""

import asyncio
import contextlib
import importlib.util
import logging
import os
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import aiohappyeyeballs
import aiohttp
import aiosqlite


ROOT = Path(__file__).resolve().parents[1]
SOCKET_CONNECT = socket.socket.connect
START_CONNECTION = aiohappyeyeballs.start_connection


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
    def interaction(user_id=42, server_id=1):
        return SimpleNamespace(
            guild_id=server_id, user=SimpleNamespace(id=user_id),
            permissions=SimpleNamespace(administrator=True),
            response=SimpleNamespace(defer=AsyncMock(), send_message=AsyncMock()),
            followup=SimpleNamespace(send=AsyncMock()),
        )

    async def seed(self, count=2, server_id="1"):
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

    async def test_eviction_drops_oldest_half_only_in_current_server(self):
        await self.seed(count=4, server_id="2")
        await self.seed(count=98)
        await self.bot.save_and_send_response(self.message, "1", "Hello", "Answer")
        cursor = await self.client.db_conn.execute(
            "SELECT role, content FROM chat_history WHERE server_id = '1' ORDER BY id",
        )
        self.assertEqual(await cursor.fetchall(),
                         [("user", f"Synthetic message {i}") for i in range(50, 98)]
                         + [("user", "Hello"), ("assistant", "Answer")])
        self.assertEqual(await self.count("chat_history"), 54)
        self.create.assert_not_awaited()
        self.message.reply.assert_awaited_once_with("Answer")

    async def test_clear_and_persona_changes_affect_only_current_server(self):
        await self.bot.cmd_role.callback(self.interaction(server_id=2), "Other persona")
        await self.seed(server_id="2")
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
                self.assertEqual(await self.bot.get_persona("1"), expected)
                self.assertEqual(await self.bot.get_persona("2"), "Other persona")
                cursor = await self.client.db_conn.execute("SELECT server_id FROM chat_history")
                self.assertEqual(await cursor.fetchall(), [("2",), ("2",)])
                self.assertIn("cleared", interaction.followup.send.call_args.args[0])
                self.assertNotIn("extraction", interaction.followup.send.call_args.args[0].lower())
        self.create.assert_not_awaited()

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
            await self.bot.save_and_send_response(self.message, "1", "New input", "New answer")
        self.assertEqual(await self.count("chat_history"), 98)
        self.message.reply.assert_not_awaited()

    async def test_cancelled_history_transaction_rolls_back_partial_turn(self):
        entered = asyncio.Event()
        async def interrupted_write():
            async with self.bot.history_transaction():
                await self.client.db_conn.execute(
                    "INSERT INTO chat_history (server_id, role, content) VALUES ('1', 'user', 'Partial')",
                )
                entered.set()
                await asyncio.Event().wait()
        task = asyncio.create_task(interrupted_write())
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(await self.count("chat_history"), 0)
        await self.bot.save_and_send_response(self.message, "1", "Next input", "Next answer")
        self.assertEqual(await self.count("chat_history"), 2)

    async def test_restart_retains_recent_conversation_and_persona_without_model_calls(self):
        await self.bot.cmd_role.callback(self.interaction(), "Persisted persona")
        await self.bot.save_and_send_response(self.message, "1", "Project Orion", "Orion confirmed")
        await self.client.db_conn.close()
        self.client.db_conn = await aiosqlite.connect("check.sqlite3")
        await self.bot.init_db(self.client.db_conn)
        context = await self.bot.build_ai_context("1", "What name?")
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
            INSERT INTO server_config VALUES ('1', 'Legacy persona');
            INSERT INTO chat_history VALUES (1, '1', 'user', 'Recent project Orion', '42', 'Tester');
            INSERT INTO chat_history VALUES (2, '1', 'assistant', 'Orion confirmed', '42', 'Tester');
            INSERT INTO pending_memories VALUES (100, '1', 'user', 'Unused pending fact', '42', 'Tester');
            INSERT INTO explicit_memories VALUES ('saved', '1', '42', 'Tester', 'Unused explicit fact', 0, '2026-01-01');
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
            context = await self.bot.build_ai_context("1", "What name?")
            self.assertIn("Legacy persona", context[0]["content"])
            self.assertEqual(context[1]["content"], "Recent project Orion")
            self.assertEqual(context[2]["content"], "Orion confirmed")
            self.assertNotIn("Unused", str(context))
            await self.bot.save_and_send_response(self.message, "1", "Next question", "Next answer")
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
        version = self.client.conversation_versions.get("1", 0)
        await self.clear_history()
        for author_id in (42, 84):
            self.message.author.id = author_id
            await self.bot.save_and_send_response(self.message, "1", "Old input", "Old answer", version)
        self.assertEqual(await self.count("chat_history"), 0)
        self.message.reply.assert_not_awaited()
        await self.bot.save_and_send_response(self.message, "1", "New input", "New answer")
        self.assertEqual(await self.count("chat_history"), 2)
        self.message.reply.assert_awaited_once()

    async def test_message_handler_captures_version_before_loading_context(self):
        bot_user = SimpleNamespace(id=99)
        self.client._connection.user = bot_user
        message = SimpleNamespace(
            author=SimpleNamespace(id=84, bot=False, display_name="Other member"),
            guild=SimpleNamespace(id=1, name="Test guild"), channel=SimpleNamespace(name="general"),
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
                version = self.client.conversation_versions.get("1", 0)
                if prompt is None:
                    await self.clear_history()
                else:
                    await self.bot.cmd_role.callback(self.interaction(), prompt)
                await self.bot.save_and_send_response(self.message, "1", "Old", "Old answer", version)
                self.assertEqual(await self.count("chat_history"), 0)
                self.message.reply.assert_not_awaited()


if __name__ == "__main__":
    unittest.main(verbosity=2)
