"""Focused regression checks for URL fetching, forgetting, and memory retention.

Run: venv_bot/bin/python scripts/check_regressions.py
Uses temporary SQLite files, synthetic models/memories, and a controlled HTTP server.
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
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import aiohappyeyeballs
import aiohttp
import aiosqlite


ROOT = Path(__file__).resolve().parents[1]
SOCKET_CONNECT = socket.socket.connect
START_CONNECTION = aiohappyeyeballs.start_connection


class MemoryStore:
    def __init__(self):
        self.facts = {}

    def query(self, **kwargs):
        return {"distances": [[]]}

    def upsert(self, *, ids, documents, metadatas, embeddings):
        if len(ids) != len(set(ids)):
            raise ValueError("Chroma requires unique IDs within a write")
        for key, document, metadata in zip(ids, documents, metadatas):
            self.facts[key] = (document, metadata)

    @staticmethod
    def matches(metadata, where):
        filters = where.get("$and", [where])
        return all(all(metadata.get(k) == v for k, v in part.items()) for part in filters)

    def get(self, *, where, include):
        rows = [(key, doc, meta) for key, (doc, meta) in self.facts.items() if self.matches(meta, where)]
        return {"ids": [r[0] for r in rows], "documents": [r[1] for r in rows], "metadatas": [r[2] for r in rows]}

    def delete(self, *, where=None, ids=None):
        for key, (_, metadata) in list(self.facts.items()):
            if (ids is None or key in ids) and (where is None or self.matches(metadata, where)):
                del self.facts[key]


class BotChecks(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="discord-bot-regressions-")
        self.old_directory = Path.cwd()
        os.chdir(self.directory.name)
        self.patches = contextlib.ExitStack()
        self.patches.enter_context(patch.dict(os.environ, {
            "DISCORD_BOT_TOKEN": "", "LLM_API_KEY": "offline-test", "BOT_OWNER_ID": "0",
        }, clear=True))
        self.patches.enter_context(patch("dotenv.load_dotenv", return_value=False))
        self.patches.enter_context(patch("discord.Client.run", side_effect=AssertionError("Discord login disabled")))
        self.patches.enter_context(patch("socket.socket.connect", side_effect=AssertionError("External sockets disabled")))
        self.patches.enter_context(patch("socket.socket.connect_ex", side_effect=AssertionError("External sockets disabled")))
        self.create = AsyncMock(return_value=self.completion('["Tester likes Python"]'))
        model = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self.create)))
        self.patches.enter_context(patch("openai.AsyncOpenAI", return_value=model))
        self.old_logging = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        spec = importlib.util.spec_from_file_location("bot_regression_check", ROOT / "bot.py")
        self.bot = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.bot)
        self.client = self.bot.client
        self.client.db_lock = asyncio.Lock()
        self.client.memory_lock = asyncio.Lock()
        self.client.llm_queue = asyncio.Semaphore(3)
        self.client.db_conn = await aiosqlite.connect("check.sqlite3")
        await self.bot.init_db(self.client.db_conn)
        self.store = self.client.memory_collection = MemoryStore()
        self.client.custom_ef = SimpleNamespace(embed=lambda texts, task: [[1.0, 0.0] for _ in texts])
        self.message = SimpleNamespace(author=SimpleNamespace(id=42), reply=AsyncMock())

    async def asyncTearDown(self):
        await self.client.close()
        self.bot.file_handler.close()
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

    async def seed(self, count=2, user_id="42", server_id="1"):
        async with self.bot.history_transaction():
            await self.client.db_conn.executemany(
                "INSERT INTO chat_history (server_id, role, content, user_id, user_name) VALUES (?, ?, ?, ?, ?)",
                [(server_id, "user", f"Synthetic message {i}", user_id, "Tester") for i in range(count)],
            )

    async def count(self, table):
        cursor = await self.client.db_conn.execute(f"SELECT COUNT(*) FROM {table}")
        return (await cursor.fetchone())[0]

    async def archive(self):
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

    async def test_eviction_retains_input_on_failure_then_retries(self):
        await self.seed(98)
        await self.bot.save_and_send_response(self.message, "1", "Tester", "Hello", "Answer")
        self.assertEqual(await self.count("chat_history"), 50)
        self.assertEqual(await self.count("pending_memories"), 50)
        self.create.side_effect = RuntimeError("Model offline")
        await self.bot.process_pending_memories()
        self.assertEqual(await self.count("pending_memories"), 50)
        self.create.side_effect = None
        await self.bot.process_pending_memories()
        self.assertEqual(await self.count("pending_memories"), 0)
        self.assertEqual(len(self.store.facts), 1)

    async def test_clear_and_persona_changes_retain_input(self):
        for prompt in (None, "Friendly", "clear"):
            with self.subTest(prompt=prompt):
                await self.seed()
                interaction = self.interaction()
                if prompt is None:
                    await self.bot.cmd_clear.callback(interaction)
                else:
                    await self.bot.cmd_role.callback(interaction, prompt)
                self.assertEqual(await self.count("chat_history"), 0)
                self.assertEqual(await self.count("pending_memories"), 2)
                self.assertIn("queued", interaction.followup.send.call_args.args[0])
                await self.bot.process_pending_memories()
                self.assertEqual(await self.count("pending_memories"), 0)

    async def test_invalid_extraction_never_discards_input(self):
        await self.seed()
        await self.archive()
        for content, reason in ((None, "stop"), ("", "stop"), ("not JSON", "stop"),
                                ('{"fact": "x"}', "stop"), ('["ok", 7]', "stop"),
                                ('[""]', "stop"), ('["incomplete"]', "length"),
                                ("preamble [] trailing garbage", "stop")):
            with self.subTest(content=content, reason=reason):
                self.create.return_value = self.completion(content, reason)
                await self.bot.process_pending_memories()
                self.assertEqual(await self.count("pending_memories"), 2)
        self.create.return_value = self.completion("```json\n[]\n```")
        await self.bot.process_pending_memories()
        self.assertEqual(await self.count("pending_memories"), 0)

    async def test_embedding_failures_retain_input(self):
        await self.seed()
        await self.archive()
        for task in ("retrieval.query", "retrieval.passage"):
            for incomplete in (False, True):
                def embed(texts, current_task):
                    if current_task == task:
                        if incomplete:
                            return []
                        raise RuntimeError("Embedding offline")
                    return [[1.0, 0.0] for _ in texts]
                with self.subTest(task=task, incomplete=incomplete), patch.object(self.client.custom_ef, "embed", side_effect=embed):
                    await self.bot.process_pending_memories()
                    self.assertEqual(await self.count("pending_memories"), 2)

    async def test_partial_vector_write_can_retry_without_duplicates(self):
        await self.seed()
        await self.archive()
        upsert = self.store.upsert
        def partial_write(**kwargs):
            upsert(**kwargs)
            raise RuntimeError("Lost response after write")
        with patch.object(self.store, "upsert", side_effect=partial_write):
            await self.bot.process_pending_memories()
        self.assertEqual(await self.count("pending_memories"), 2)
        self.assertEqual(len(self.store.facts), 1)
        await self.bot.process_pending_memories()
        self.assertEqual(await self.count("pending_memories"), 0)
        self.assertEqual(len(self.store.facts), 1)

    async def test_duplicate_facts_in_one_response_do_not_block_extraction(self):
        await self.seed()
        await self.archive()
        self.create.return_value = self.completion('["Tester likes Python", "Tester likes Python", " Tester likes Python "]')
        await self.bot.process_pending_memories()
        self.assertEqual(await self.count("pending_memories"), 0)
        self.assertEqual(len(self.store.facts), 1)

    async def test_failed_archive_rolls_back_both_tables(self):
        await self.seed()
        await self.client.db_conn.execute(
            "CREATE TEMP TRIGGER fail_delete BEFORE DELETE ON chat_history BEGIN SELECT RAISE(ABORT, 'synthetic failure'); END"
        )
        with self.assertRaises(aiosqlite.IntegrityError):
            await self.archive()
        self.assertEqual(await self.count("chat_history"), 2)
        self.assertEqual(await self.count("pending_memories"), 0)
        await self.client.db_conn.execute("DROP TRIGGER fail_delete")
        await self.archive()
        self.assertEqual(await self.count("pending_memories"), 2)

    async def test_restart_resumes_retained_input(self):
        await self.seed()
        await self.archive()
        self.create.side_effect = RuntimeError("Model offline")
        await self.bot.process_pending_memories()
        await self.client.db_conn.close()
        self.client.db_conn = await aiosqlite.connect("check.sqlite3")
        await self.bot.init_db(self.client.db_conn)
        self.create.side_effect = None
        self.client.memory_worker = asyncio.create_task(self.bot.retry_pending_memories())
        async with asyncio.timeout(2):
            while await self.count("pending_memories"):
                await asyncio.sleep(0.01)
        self.assertEqual(len(self.store.facts), 1)

    async def test_worker_retries_without_another_message(self):
        await self.seed()
        await self.archive()
        self.create.side_effect = [RuntimeError("Temporary outage"), self.completion('["Tester likes Python"]')]
        with patch.object(self.bot, "MEMORY_RETRY_INTERVAL", 0.01):
            self.client.memory_worker = asyncio.create_task(self.bot.retry_pending_memories())
            async with asyncio.timeout(2):
                while await self.count("pending_memories"):
                    await asyncio.sleep(0.01)
        self.assertEqual(self.create.await_count, 2)
        self.assertEqual(len(self.store.facts), 1)

    async def test_deletion_during_embedding_cannot_restore_facts(self):
        for wipe in (False, True):
            for phase in ("retrieval.query", "retrieval.passage"):
                with self.subTest(wipe=wipe, phase=phase):
                    await self.seed()
                    await self.archive()
                    entered = asyncio.Event()
                    release = threading.Event()
                    loop = asyncio.get_running_loop()
                    def embed(texts, task):
                        if task == phase:
                            loop.call_soon_threadsafe(entered.set)
                            if not release.wait(5):
                                raise AssertionError("Test failed to release embedding")
                        return [[1.0, 0.0] for _ in texts]
                    with patch.object(self.client.custom_ef, "embed", side_effect=embed):
                        processing = asyncio.create_task(self.bot.process_pending_memories())
                        try:
                            await asyncio.wait_for(entered.wait(), 2)
                            await self.bot.forget_memories("1", None if wipe else "42")
                        finally:
                            release.set()
                            await processing
                    self.assertEqual(await self.count("pending_memories"), 0)
                    self.assertEqual(len(self.store.facts), 0)

    async def test_work_captured_before_deletion_cannot_start_after_it(self):
        version = self.client.memory_version("1", "42")
        await self.bot.forget_memories("1", "42")
        self.assertTrue(await self.bot.update_user_memory("1", "42", "Tester", [
            {"role": "user", "content": "Old fact", "user_id": "42"},
        ], version))
        self.create.assert_not_awaited()

    async def test_forget_invalidates_every_members_old_reply_but_allows_new_turns(self):
        version = self.client.conversation_versions.get("1", 0)
        await self.bot.forget_memories("1", "42")
        for author_id in (42, 84):
            self.message.author.id = author_id
            await self.bot.save_and_send_response(self.message, "1", "Tester", "Old input", "Old shared fact", version)
        self.assertEqual(await self.count("chat_history"), 0)
        self.message.reply.assert_not_awaited()
        await self.bot.save_and_send_response(self.message, "1", "Tester", "New input", "New answer")
        self.assertEqual(await self.count("chat_history"), 2)
        self.message.reply.assert_awaited_once()

    async def test_chat_started_during_delete_is_also_invalidated(self):
        entered = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        def blocked_delete(**kwargs):
            loop.call_soon_threadsafe(entered.set)
            if not release.wait(5):
                raise AssertionError("Test failed to release deletion")
        with patch.object(self.store, "delete", side_effect=blocked_delete):
            deletion = asyncio.create_task(self.bot.forget_memories("1", "42"))
            try:
                await asyncio.wait_for(entered.wait(), 2)
                version = self.client.conversation_versions.get("1", 0)
                self.message.author.id = 84
                response = asyncio.create_task(self.bot.save_and_send_response(
                    self.message, "1", "Other user", "Old context", "Deleted shared fact", version,
                ))
            finally:
                release.set()
                await deletion
            await response
        self.assertEqual(await self.count("chat_history"), 0)
        self.message.reply.assert_not_awaited()

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
                patch.object(self.bot, "generate_ai_response", new=AsyncMock(return_value="Old shared fact")):
            request = asyncio.create_task(self.bot.on_message(message))
            try:
                await asyncio.wait_for(entered.wait(), 2)
                await self.bot.forget_memories("1", "42")
            finally:
                release.set()
                await request
        self.assertEqual(await self.count("chat_history"), 0)
        message.reply.assert_not_awaited()

    async def test_clear_and_role_invalidate_old_replies(self):
        for prompt in (None, "Friendly"):
            with self.subTest(prompt=prompt):
                version = self.client.conversation_versions.get("1", 0)
                if prompt is None:
                    await self.archive()
                else:
                    await self.bot.cmd_role.callback(self.interaction(), prompt)
                await self.bot.save_and_send_response(self.message, "1", "Tester", "Old", "Old answer", version)
                self.assertEqual(await self.count("chat_history"), 0)
                self.message.reply.assert_not_awaited()

    async def test_forget_commands_remove_retained_input_with_correct_scope(self):
        for command in ("self", "force", "server"):
            with self.subTest(command=command):
                await self.seed(user_id="42")
                await self.seed(user_id="84")
                await self.archive()
                await self.seed(user_id="42", server_id="2")
                await self.bot.cmd_clear.callback(self.interaction(server_id=2))
                if command == "self":
                    await self.bot.cmd_memory.callback(self.interaction(), SimpleNamespace(value="clear"))
                elif command == "force":
                    await self.bot.cmd_force_forget.callback(self.interaction(), SimpleNamespace(id=42, mention="@Tester"))
                else:
                    await self.bot.cmd_wipe_server.callback(self.interaction())
                self.assertEqual(await self.count("pending_memories"), 2 if command == "server" else 4)
                await self.bot.process_pending_memories()
                self.assertEqual(await self.count("pending_memories"), 0)
                self.store.facts.clear()

    async def test_cancelled_vector_write_finishes_before_forget(self):
        await self.seed()
        await self.archive()
        entered = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        upsert = self.store.upsert
        def blocked_write(**kwargs):
            loop.call_soon_threadsafe(entered.set)
            if not release.wait(5):
                raise AssertionError("Test failed to release vector write")
            upsert(**kwargs)
        with patch.object(self.store, "upsert", side_effect=blocked_write):
            processing = asyncio.create_task(self.bot.process_pending_memories())
            try:
                await asyncio.wait_for(entered.wait(), 2)
                processing.cancel()
                deletion = asyncio.create_task(self.bot.forget_memories("1", "42"))
                await asyncio.sleep(0.02)
                self.assertFalse(deletion.done())
            finally:
                release.set()
                with contextlib.suppress(asyncio.CancelledError):
                    await processing
            await deletion
        self.assertEqual(len(self.store.facts), 0)
        self.assertEqual(await self.count("pending_memories"), 0)

    async def test_shutdown_preserves_cancellation_when_vector_write_fails(self):
        await self.seed()
        await self.archive()
        entered = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        def failed_write(**kwargs):
            loop.call_soon_threadsafe(entered.set)
            if not release.wait(5):
                raise AssertionError("Test failed to release write")
            raise RuntimeError("Synthetic write failure")
        with patch.object(self.store, "upsert", side_effect=failed_write):
            worker = asyncio.create_task(self.bot.retry_pending_memories())
            try:
                await asyncio.wait_for(entered.wait(), 2)
                worker.cancel()
                await asyncio.sleep(0)
                worker.cancel()
            finally:
                release.set()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(worker, 2)
        self.assertEqual(await self.count("pending_memories"), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
