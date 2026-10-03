"""Offline checks for permission fallback, search, explicit memory, and turn ordering.

Run: venv_bot/bin/python scripts/check_features.py
Reuses the temporary SQLite/model fixtures; never loads .env or logs into Discord.
"""

import asyncio
import contextlib
import json
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, PropertyMock, patch

import aiosqlite
import discord

import check_regressions as fixtures


class FeatureChecks(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixtures.BotChecks.asyncSetUp
    asyncTearDown = fixtures.BotChecks.asyncTearDown
    completion = staticmethod(fixtures.BotChecks.completion)
    interaction = staticmethod(fixtures.BotChecks.interaction)
    seed = fixtures.BotChecks.seed
    count = fixtures.BotChecks.count
    archive = fixtures.BotChecks.archive

    def chat(self, server=1, author=42, content="Hello", history=True):
        bot_user = SimpleNamespace(id=99)
        self.client._connection.user = bot_user
        channel = SimpleNamespace(
            id=server * 10, name="general", send=AsyncMock(), fetch_message=AsyncMock(),
            permissions_for=lambda member: SimpleNamespace(read_message_history=history),
        )
        @contextlib.asynccontextmanager
        async def typing():
            yield
        channel.typing = typing
        return SimpleNamespace(
            author=SimpleNamespace(id=author, bot=False, display_name=f"Member{author}"),
            guild=SimpleNamespace(id=server, name=f"Server{server}", me=bot_user),
            channel=channel, mentions=[bot_user], reference=None,
            content=f"<@99> {content}", attachments=[], stickers=[], reply=AsyncMock(),
        )

    def put_fact(self, key="existing", text="Tester lives in Delhi", user="42", server="1", name="Tester"):
        self.store.facts[key] = (f"[Recorded on 2026-01-01]: {text}",
                                 {"user_id": user, "server_id": server, "user_name": name})
        return key

    async def test_status_distinguishes_server_response_and_configured_backups(self):
        self.client.config.model = "primary-chat"
        self.client.config.embedding_model = "primary-embedding"
        self.client.config.fallback_model = "backup-chat"
        models = self.client.lm_client.models = SimpleNamespace(list=AsyncMock(return_value=[]))
        with patch.object(discord.Client, "latency", new_callable=PropertyMock, return_value=0.123):
            for responding in (True, False):
                for chat_backup in (True, False):
                    for memory_backup in (True, False):
                        with self.subTest(responding=responding, chat_backup=chat_backup, memory_backup=memory_backup):
                            models.list.side_effect = None if responding else RuntimeError("server check failed")
                            self.client.fallback_client = SimpleNamespace(close=AsyncMock()) if chat_backup else None
                            self.client.config.embedding_key = "synthetic" if memory_backup else ""
                            interaction = self.interaction()
                            await self.bot.cmd_status.callback(interaction)
                            text = interaction.followup.send.call_args.args[0]
                            self.assertIn("**Primary AI server:** " + ("Responding" if responding else "Check failed"), text)
                            self.assertIn("**Chat model (configured):** `primary-chat`", text)
                            self.assertIn("**Memory search model (configured):** `primary-embedding`", text)
                            self.assertIn("**Chat backup:** " + ("Configured (backup-chat)" if chat_backup else "Not configured"), text)
                            self.assertIn("**Memory search backup:** " + ("Configured (Jina)" if memory_backup else "Not configured"), text)
                            self.assertIn("does not test model responses or backups", text)
        self.assertEqual(models.list.await_count, 8)
        self.create.assert_not_awaited()

    async def test_status_reports_scoped_history_usage_and_configured_input_limits(self):
        self.client.lm_client.models = SimpleNamespace(list=AsyncMock(return_value=[]))
        self.client.highest_token_count = 4096
        await self.seed(count=2, server_id="1")
        await self.seed(count=4, server_id="2")
        with patch.object(self.bot, "MAX_FILE_SIZE", 4_000_000), \
                patch.object(self.bot, "MAX_PDF_PAGES", 7), \
                patch.object(self.bot, "MAX_TEXT_EXTRACTION_LENGTH", 12345):
            for latency, vision in ((float("nan"), False), (float("inf"), True), (0.123, True)):
                with self.subTest(latency=latency, vision=vision), \
                        patch.object(discord.Client, "latency", new_callable=PropertyMock, return_value=latency):
                    self.client.config.vision_enabled = vision
                    interaction = self.interaction()
                    await self.bot.cmd_status.callback(interaction)
                    text = interaction.followup.send.call_args.args[0]
                    self.assertIn("123 ms" if latency == 0.123 else "Not available yet", text)
                    self.assertIn("Enabled; the chat model must support images" if vision else "Disabled in bot settings", text)
                    self.assertIn("**This server's history:** 2/100 messages", text)
                    self.assertIn("4,096 tokens (input + output)", text)
                    self.assertIn("all servers since the bot started", text)
                    self.assertIn("4.0 MB per file", text)
                    self.assertIn("first 7 pages", text)
                    self.assertIn("12,345 characters", text)
                    self.assertLessEqual(len(text), 2000)

    async def test_status_delivers_when_server_probe_stalls(self):
        cancelled = asyncio.Event()
        async def stalled():
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        self.client.lm_client.models = SimpleNamespace(list=AsyncMock(side_effect=stalled))
        interaction = self.interaction()
        with patch.object(discord.Client, "latency", new_callable=PropertyMock, return_value=0.1):
            await asyncio.wait_for(self.bot.cmd_status.callback(interaction), timeout=6.0)
        self.assertTrue(cancelled.is_set())
        text = interaction.followup.send.call_args.args[0]
        self.assertIn("**Primary AI server:** Check failed", text)
        self.assertIn("What you can send", text)
        self.assertIn("No usage reported yet", text)

    async def test_status_with_long_model_names_fits_discord_messages(self):
        self.client.lm_client.models = SimpleNamespace(list=AsyncMock(return_value=[]))
        self.client.config.model = "chat-" + "a" * 2100
        self.client.config.embedding_model = "memory-" + "b" * 2100
        self.client.fallback_client = SimpleNamespace(close=AsyncMock())
        self.client.config.fallback_model = "backup-" + "c" * 2100
        interaction = self.interaction()
        interaction.channel = self.chat().channel
        with patch.object(discord.Client, "latency", new_callable=PropertyMock, return_value=0.1), \
                patch.object(self.bot, "CHUNK_MESSAGE_DELAY", 0):
            await self.bot.cmd_status.callback(interaction)
        chunks = [call.args[0] for call in interaction.followup.send.call_args_list + interaction.channel.send.call_args_list]
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 2000 for chunk in chunks))
        text = "".join(chunks)
        for model in (self.client.config.model, self.client.config.embedding_model, self.client.config.fallback_model):
            self.assertIn(model, text)
        self.assertIn("What you can send", text)
        self.assertTrue(text.endswith("not supported."))

    async def test_missing_history_uses_normal_message_and_fits_discord_limit(self):
        message = self.chat(history=False)
        await self.bot.send_chunked_message(message, "```python\n" + "x" * 4100 + "\n```")
        message.reply.assert_not_awaited()
        self.assertGreater(message.channel.send.await_count, 1)
        for call in message.channel.send.call_args_list:
            self.assertLessEqual(len(call.args[0]), 2000)
        first = message.channel.send.call_args_list[0]
        self.assertTrue(first.args[0].startswith("<@42> "))
        allowed = first.kwargs["allowed_mentions"].to_dict()
        self.assertEqual(allowed["users"], [42])
        self.assertEqual(allowed["parse"], [])

    async def test_forbidden_or_deleted_native_reply_falls_back(self):
        for error in (discord.Forbidden, discord.NotFound):
            message = self.chat()
            message.reply.side_effect = error(SimpleNamespace(status=403, reason="Denied"), "test")
            await self.bot.send_chunked_message(message, "Answer")
            self.assertEqual(message.channel.send.call_args.args[0], "<@42> Answer")

    async def test_other_delivery_errors_are_not_silently_retried(self):
        message = self.chat()
        message.reply.side_effect = discord.HTTPException(SimpleNamespace(status=500, reason="Failure"), "test")
        with self.assertRaises(discord.HTTPException):
            await self.bot.reply_or_send(message, "Answer")
        message.channel.send.assert_not_awaited()

    async def test_reply_context_uses_delivered_message_without_history_fetch(self):
        message = self.chat(history=False)
        referenced = SimpleNamespace(author=SimpleNamespace(id=5, display_name="Other"),
                                     content="Our project is Orion", attachments=[], stickers=[])
        message.reference = SimpleNamespace(message_id=123, resolved=referenced, cached_message=None)
        text, *_ = await self.bot.extract_message_context(message, "What name?", "Tester")
        self.assertIn("Orion", text)
        message.channel.fetch_message.assert_not_awaited()
        message.reference.resolved = None
        text, *_ = await self.bot.extract_message_context(message, "What name?", "Tester")
        self.assertEqual(text, "What name?")
        message.channel.fetch_message.assert_not_awaited()

    async def test_search_preserves_query_and_returns_source_urls(self):
        search = Mock(return_value=[{"title": "C++ docs", "href": "https://example.com/cpp", "body": "An excerpt"}])
        with patch.object(self.bot, "DDGS", return_value=SimpleNamespace(text=search)):
            for query in ('C++ std::vector "2010"', 'C# vs .NET', 'history of Python 1991'):
                result = await self.bot.perform_web_search(query)
                search.assert_called_with(query, max_results=self.bot.WEB_SEARCH_MAX_RESULTS)
                self.assertIn("URL: https://example.com/cpp", result)

    async def test_invalid_tool_arguments_return_recoverable_errors(self):
        search = AsyncMock(return_value="result")
        with patch.dict(self.bot.AVAILABLE_TOOLS, web_search=search):
            for arguments in ('{}', '[]', 'null', '{broken', '{"query": 3}',
                              '{"query":"ok","extra":true}', '{"query":" "}'):
                result = await self.bot.execute_tool_call("web_search", arguments)
                self.assertIn("Tool error", result)
            self.assertIn("Tool error", await self.bot.execute_tool_call("unknown", '{"query":"ok"}'))
            search.assert_not_awaited()

    async def test_generation_keeps_sources_even_if_model_omits_citations(self):
        call = SimpleNamespace(id="search-1", function=SimpleNamespace(name="web_search", arguments='{"query":"test"}'))
        tool_message = SimpleNamespace(content=None, tool_calls=[call], model_dump=lambda **kwargs: {
            "role": "assistant", "tool_calls": [],
        })
        answer_message = SimpleNamespace(content="The answer.", tool_calls=[])
        self.create.side_effect = [SimpleNamespace(choices=[SimpleNamespace(message=m)], usage=None)
                                   for m in (tool_message, answer_message)]
        with patch.dict(self.bot.AVAILABLE_TOOLS, web_search=AsyncMock(return_value="URL: https://example.com/source\n")):
            answer = await self.bot.generate_ai_response([], self.chat(), False)
        self.assertIn("https://example.com/source", answer)
        self.assertTrue(answer.startswith("The answer."))

    async def test_supplied_document_still_allows_web_search(self):
        message = self.chat()
        self.create.return_value = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="Answer", tool_calls=[]),
        )], usage=None)
        with patch.object(self.bot, "extract_message_context", new=AsyncMock(return_value=(
                "Question", [], [], "[Extracted PDF Content]: incomplete document"))):
            await self.bot.on_message(message)
        self.assertEqual(self.create.call_args.kwargs["tools"][0]["function"]["name"], "web_search")
        self.assertIn("incomplete document", str(self.create.call_args.kwargs["messages"]))

    async def test_remember_saves_and_recalls_during_embedding_outage(self):
        interaction = self.interaction()
        interaction.user.display_name = "Tester"
        await self.bot.cmd_remember.callback(interaction, "I prefer Python")
        self.assertIn("Saved", interaction.followup.send.call_args.args[0])
        with patch.object(self.client.custom_ef, "embed", side_effect=RuntimeError("offline")):
            await self.bot.sync_manual_memories()
            context = await self.bot.build_ai_context("1", "What do I prefer?")
        self.assertIn("I prefer Python", context[0]["content"])
        self.assertEqual(await self.count("explicit_memories"), 1)
        await self.client.db_conn.close()
        self.client.db_conn = await aiosqlite.connect("check.sqlite3")
        await self.bot.init_db(self.client.db_conn)
        await self.bot.sync_manual_memories()
        self.assertEqual(len(self.store.facts), 1)

    async def test_removed_memory_actions_are_unavailable_and_cannot_mutate_data(self):
        self.put_fact()
        schema = self.bot.cmd_memory.to_dict(self.bot.tree)
        options = {option["name"]: option for option in schema["options"]}
        self.assertEqual(set(options), {"action", "target_user"})
        self.assertEqual({choice["value"] for choice in options["action"]["choices"]},
                         {"list", "read", "clear"})
        for action in ("edit", "delete"):
            interaction = self.interaction()
            await self.bot.cmd_memory.callback(interaction, SimpleNamespace(value=action))
            self.assertIn("no longer available", interaction.followup.send.call_args.args[0])
        self.assertEqual(await self.count("explicit_memories"), 0)
        self.assertIn("Delhi", self.store.facts["existing"][0])
        cursor = await self.client.db_conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        tables = {row[0] for row in await cursor.fetchall()}
        self.assertTrue({"memory_overrides", "memory_suppressions"}.isdisjoint(tables))

    async def test_read_defaults_to_self_and_can_read_another_member_in_same_server(self):
        self.put_fact()
        self.put_fact("other", "Other likes Python", user="84", name="Other")
        self.put_fact("elsewhere", "Tester likes games", server="2")
        interaction = self.interaction()
        await self.bot.cmd_memory.callback(interaction, SimpleNamespace(value="read"))
        text = interaction.followup.send.call_args.args[0]
        self.assertIn("Delhi", text)
        self.assertNotIn("Other", text)
        self.assertNotIn("games", text)
        await self.bot.cmd_memory.callback(interaction, SimpleNamespace(value="read"), target_user="Other")
        text = interaction.followup.send.call_args.args[0]
        self.assertIn("Other likes Python", text)
        self.assertNotIn("games", text)

    async def test_explicit_memory_survives_failed_indexing_and_retries(self):
        key = await self.bot.remember_fact("1", "42", "Tester", "I like hiking")
        with patch.object(self.store, "upsert", side_effect=RuntimeError("index failure")):
            await self.bot.sync_manual_memories()
        self.assertEqual(self.store.facts, {})
        self.assertIn(key, await self.bot.list_memories("1", "42"))
        context = await self.bot.build_ai_context("1", "What do I like?")
        self.assertIn("I like hiking", context[0]["content"])
        await self.bot.sync_manual_memories()
        self.assertIn("I like hiking", self.store.facts[key][0])

    async def test_remember_during_extraction_retries_with_new_explicit_fact(self):
        await self.seed()
        await self.archive()
        entered, release = asyncio.Event(), asyncio.Event()
        async def extract(**kwargs):
            entered.set()
            await release.wait()
            return self.completion('["Tester likes Python", "Tester likes hiking"]')
        self.create.side_effect = extract
        task = asyncio.create_task(self.bot.process_pending_memories())
        await asyncio.wait_for(entered.wait(), 2)
        await self.bot.remember_fact("1", "42", "Tester", "I prefer Rust")
        release.set()
        await task
        self.assertEqual(await self.count("pending_memories"), 2)
        self.create.side_effect = None
        self.create.return_value = self.completion('["Tester likes Python", "Tester likes hiking"]')
        await self.bot.process_pending_memories()
        facts = await self.bot.list_memories("1", "42")
        text = str(facts)
        self.assertIn("Rust", text)
        self.assertIn("hiking", text)
        self.assertIn("likes Python", text)
        self.assertIn("I prefer Rust", self.create.call_args.kwargs["messages"][0]["content"])
        self.assertEqual(await self.count("explicit_memories"), 1)

    async def test_forget_during_manual_embedding_prevents_index_restoration(self):
        await self.bot.remember_fact("1", "42", "Tester", "I like Python")
        entered, release = asyncio.Event(), threading.Event()
        loop = asyncio.get_running_loop()
        def embed(*args):
            loop.call_soon_threadsafe(entered.set)
            if not release.wait(5):
                raise AssertionError("Unreleased embedding")
            return [[1.0, 0.0]]
        with patch.object(self.client.custom_ef, "embed", side_effect=embed):
            task = asyncio.create_task(self.bot.sync_manual_memories())
            try:
                await asyncio.wait_for(entered.wait(), 2)
                await self.bot.forget_memories("1", "42")
            finally:
                release.set()
                await task
        self.assertEqual(self.store.facts, {})
        self.assertEqual(await self.count("explicit_memories"), 0)

    async def test_failed_sqlite_save_leaves_existing_facts_intact(self):
        self.put_fact()
        await self.client.db_conn.execute(
            "CREATE TEMP TRIGGER fail_save BEFORE INSERT ON explicit_memories BEGIN SELECT RAISE(ABORT, 'failure'); END",
        )
        with self.assertRaises(aiosqlite.IntegrityError):
            await self.bot.remember_fact("1", "42", "Tester", "I like hiking")
        self.assertEqual(await self.count("explicit_memories"), 0)
        self.assertIn("Delhi", (await self.bot.list_memories("1", "42"))["existing"][0])

    async def test_turns_order_within_server_while_other_servers_progress(self):
        first, second, other = self.chat(content="Project Orion"), self.chat(author=84, content="What name?"), self.chat(server=2)
        entered, release, other_done = asyncio.Event(), asyncio.Event(), asyncio.Event()
        contexts = {}
        async def context(server, payload):
            cursor = await self.client.db_conn.execute("SELECT content FROM chat_history WHERE server_id = ? ORDER BY id", (server,))
            contexts.setdefault(server, []).append([row[0] for row in await cursor.fetchall()])
            return []
        async def generate(messages, message, *args):
            if message is first:
                entered.set()
                await release.wait()
                return "Orion confirmed"
            if message is other:
                other_done.set()
            return "Answer"
        with patch.object(self.bot, "build_ai_context", side_effect=context), \
                patch.object(self.bot, "generate_ai_response", side_effect=generate):
            a = asyncio.create_task(self.bot.on_message(first))
            await asyncio.wait_for(entered.wait(), 2)
            b = asyncio.create_task(self.bot.on_message(second))
            c = asyncio.create_task(self.bot.on_message(other))
            try:
                await asyncio.wait_for(other_done.wait(), 2)
                self.assertEqual(len(contexts["1"]), 1)
            finally:
                release.set()
                await asyncio.gather(a, b, c)
        self.assertIn("Orion confirmed", contexts["1"][1])

    async def test_clear_invalidates_running_and_queued_turns(self):
        first, second = self.chat(), self.chat(author=84)
        entered, release = asyncio.Event(), asyncio.Event()
        async def generate(*args):
            entered.set()
            await release.wait()
            return "Old answer"
        with patch.object(self.bot, "build_ai_context", new=AsyncMock(return_value=[])), \
                patch.object(self.bot, "generate_ai_response", side_effect=generate) as model:
            a = asyncio.create_task(self.bot.on_message(first))
            await asyncio.wait_for(entered.wait(), 2)
            b = asyncio.create_task(self.bot.on_message(second))
            await asyncio.sleep(0)
            await self.archive()
            release.set()
            await asyncio.gather(a, b)
            self.assertEqual(model.await_count, 1)
        self.assertEqual(await self.count("chat_history"), 0)
        first.reply.assert_not_awaited()
        second.reply.assert_not_awaited()

    async def test_cancelled_turn_releases_server_for_next_request(self):
        first, second = self.chat(), self.chat(author=84)
        entered = asyncio.Event()
        async def generate(messages, message, *args):
            if message is first:
                entered.set()
                await asyncio.Event().wait()
            return "Next answer"
        with patch.object(self.bot, "build_ai_context", new=AsyncMock(return_value=[])), \
                patch.object(self.bot, "generate_ai_response", side_effect=generate):
            a = asyncio.create_task(self.bot.on_message(first))
            await asyncio.wait_for(entered.wait(), 2)
            b = asyncio.create_task(self.bot.on_message(second))
            a.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await a
            await asyncio.wait_for(b, 2)
        second.reply.assert_awaited_once_with("Next answer")

    async def test_remember_appends_and_clear_keeps_other_members_and_servers(self):
        key = self.put_fact()
        own = await self.bot.remember_fact("1", "42", "Tester", "I like hiking")
        other = await self.bot.remember_fact("1", "84", "Other", "I like Python")
        elsewhere = await self.bot.remember_fact("2", "42", "Tester", "I like games")
        self.assertEqual(set(await self.bot.list_memories("1", "42")), {key, own})
        await self.bot.sync_manual_memories()
        await self.bot.cmd_memory.callback(self.interaction(), SimpleNamespace(value="clear"), target_user="Other")
        self.assertEqual(set(self.store.facts), {other, elsewhere})
        self.assertEqual(await self.bot.list_memories("1", "42"), {})
        self.assertEqual(await self.count("explicit_memories"), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
