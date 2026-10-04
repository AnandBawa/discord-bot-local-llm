"""Offline checks for permission fallback, search, command schemas, and turn ordering.

Run: venv_bot/bin/python scripts/check_features.py
Reuses the temporary SQLite/model fixtures; never loads .env or logs into Discord.
"""

import asyncio
import contextlib
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, PropertyMock, patch

import discord

import check_regressions as fixtures


class FeatureChecks(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixtures.BotChecks.asyncSetUp
    asyncTearDown = fixtures.BotChecks.asyncTearDown
    completion = staticmethod(fixtures.BotChecks.completion)
    interaction = staticmethod(fixtures.BotChecks.interaction)
    seed = fixtures.BotChecks.seed
    count = fixtures.BotChecks.count
    clear_history = fixtures.BotChecks.clear_history

    def chat(self, server=1, author=42, content="Hello", history=True, *, channel_id=None):
        bot_user = SimpleNamespace(id=99)
        self.client._connection.user = bot_user
        channel = SimpleNamespace(
            id=channel_id if channel_id is not None else server * 10, name="general", send=AsyncMock(), fetch_message=AsyncMock(),
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

    async def test_status_reports_chat_model_from_last_successful_request(self):
        self.client.config.model = "primary-chat"
        self.client.config.fallback_model = "fallback-chat"
        cloud = AsyncMock(return_value=self.completion("Answer"))
        self.client.fallback_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=cloud)), close=AsyncMock())
        models = self.client.lm_client.models = SimpleNamespace(list=AsyncMock())
        with patch.object(discord.Client, "latency", new_callable=PropertyMock, return_value=0.123):
            for chat_fallback in (True, False):
                with self.subTest(chat_fallback=chat_fallback):
                    self.client.chat_dead_until = 0
                    self.create.side_effect = RuntimeError("chat unavailable") if chat_fallback else None
                    await self.bot.request_completion(messages=[])
                    # Cooldown expiry does not change the provider that handled the last request.
                    self.client.chat_dead_until = 0
                    interaction = self.interaction()
                    await self.bot.cmd_status.callback(interaction)
                    text = interaction.followup.send.call_args.args[0]
                    self.assertIn("**Chat model:** `" + ("fallback-chat (fallback)" if chat_fallback else "primary-chat") + "`", text)
                    self.assertNotIn("Memory model", text)
                    self.assertNotIn("backup:", text.lower())
        models.list.assert_not_awaited()

    async def test_status_keeps_primary_model_until_fallback_succeeds(self):
        self.client.config.model = "primary-chat"
        self.client.config.fallback_model = "fallback-chat"
        cloud = AsyncMock(side_effect=RuntimeError("fallback unavailable"))
        self.client.fallback_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=cloud)), close=AsyncMock())
        self.client.lm_client.models = SimpleNamespace(list=AsyncMock(side_effect=AssertionError("Status must not probe providers")))
        with patch.object(discord.Client, "latency", new_callable=PropertyMock, return_value=0.1):
            for failed_attempt in (False, True):
                if failed_attempt:
                    self.create.side_effect = RuntimeError("primary unavailable")
                    with self.assertRaises(RuntimeError):
                        await self.bot.request_completion(messages=[])
                interaction = self.interaction()
                await self.bot.cmd_status.callback(interaction)
                text = interaction.followup.send.call_args.args[0]
                self.assertIn("**Chat model:** `primary-chat`", text)
                self.assertNotIn("Memory model", text)
                self.assertNotIn("fallback-chat", text)
            cloud.side_effect = None
            cloud.return_value = self.completion("Answer")
            await self.bot.request_completion(messages=[])
            interaction = self.interaction()
            await self.bot.cmd_status.callback(interaction)
            self.assertIn("**Chat model:** `fallback-chat (fallback)`", interaction.followup.send.call_args.args[0])
            self.client.chat_dead_until = 0
            self.create.side_effect = None
            await self.bot.request_completion(messages=[])
            interaction = self.interaction()
            await self.bot.cmd_status.callback(interaction)
            self.assertIn("**Chat model:** `primary-chat`", interaction.followup.send.call_args.args[0])
        self.client.lm_client.models.list.assert_not_awaited()

    async def test_commands_and_help_expose_only_retained_features(self):
        names = {command.name for command in self.bot.tree.get_commands()}
        self.assertEqual(names, {"help", "status", "role", "clear", "imagegen"})
        for name in ("remember", "memory", "force-forget", "admin_wipe_server"):
            self.assertIsNone(self.bot.tree.get_command(name))
        interaction = self.interaction()
        self.client._connection.user = SimpleNamespace(id=99, name="Synthetic Bot")
        await self.bot.cmd_help.callback(interaction)
        text = str(interaction.response.send_message.call_args) + str(interaction.followup.send.call_args)
        for name in ("status", "role", "clear", "imagegen"):
            self.assertIn("/" + name, text)
        for name in ("remember", "memory", "force-forget", "admin_wipe_server"):
            self.assertNotIn("/" + name, text)

    async def test_status_reports_scoped_history_and_configured_input_limits(self):
        await self.seed(count=2, server_id="channel:10")
        await self.seed(count=4, server_id="channel:20")
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
                    self.assertIn("123 ms" if latency == 0.123 else "Unavailable", text)
                    self.assertIn("**Images/stickers:** " + ("On" if vision else "Off"), text)
                    self.assertIn("**History:** 2/100 messages", text)
                    self.assertIn("4.0 MB per image/PDF/text file", text)
                    self.assertIn("text files/PDFs", text)
                    self.assertIn("7 PDF pages", text)
                    self.assertIn("12,345 characters", text)
                    self.assertLess(len(text), 600)

    async def test_status_with_long_model_names_fits_discord_messages(self):
        self.client.config.model = "chat-" + "a" * 2100
        interaction = self.interaction()
        interaction.channel = self.chat().channel
        with patch.object(discord.Client, "latency", new_callable=PropertyMock, return_value=0.1), \
                patch.object(self.bot, "CHUNK_MESSAGE_DELAY", 0):
            await self.bot.cmd_status.callback(interaction)
        chunks = [call.args[0] for call in interaction.followup.send.call_args_list + interaction.channel.send.call_args_list]
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 2000 for chunk in chunks))
        text = "".join(chunks)
        self.assertIn(self.client.config.model, text)
        self.assertTrue(text.endswith("compatible chat model."))

    async def test_long_message_text_attachment_reaches_model_without_a_caption(self):
        body = "Please review this long message.\n" + "café 日本語 🙂\n" * 300
        data = body.encode("utf-8-sig")
        message = self.chat(content="")  # Only the bot mention remains outside the file.
        message.attachments = [SimpleNamespace(
            filename="message.txt", content_type="text/plain; charset=utf-8", size=len(data),
            read=AsyncMock(return_value=data),
        )]
        self.create.return_value.choices[0].message.tool_calls = []
        await self.bot.on_message(message)
        context = str(self.create.call_args.kwargs["messages"])
        self.assertIn(body, self.create.call_args.kwargs["messages"][-1]["content"])
        self.assertIn("Extracted Text Content from message.txt", context)
        message.reply.assert_awaited_once_with("Answer")
        cursor = await self.client.db_conn.execute("SELECT content FROM chat_history ORDER BY id")
        saved = str(await cursor.fetchall())
        self.assertIn("message.txt", saved)
        self.assertNotIn("Please review this long message", saved)
        self.assertNotIn("日本語", saved)

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

    async def test_turns_order_within_channel_while_other_channels_progress(self):
        first, second, other = self.chat(content="Project Orion"), self.chat(author=84, content="What name?"), self.chat(channel_id=11)
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
                self.assertEqual(len(contexts["channel:10"]), 1)
            finally:
                release.set()
                await asyncio.gather(a, b, c)
        self.assertIn("Orion confirmed", contexts["channel:10"][1])

    async def test_clear_and_role_invalidate_running_and_queued_turns(self):
        for prompt in (None, "Friendly", "clear"):
            with self.subTest(prompt=prompt):
                first, second = self.chat(), self.chat(author=84)
                entered, release = asyncio.Event(), asyncio.Event()
                async def generate(*args):
                    entered.set()
                    await release.wait()
                    return "Old answer"
                with patch.object(self.bot, "build_ai_context", new=AsyncMock(return_value=[])), \
                        patch.object(self.bot, "generate_ai_response", side_effect=generate) as model:
                    first_task = asyncio.create_task(self.bot.on_message(first))
                    await asyncio.wait_for(entered.wait(), 2)
                    second_task = asyncio.create_task(self.bot.on_message(second))
                    try:
                        await asyncio.sleep(0)
                        if prompt is None:
                            await self.clear_history()
                        else:
                            await self.bot.cmd_role.callback(self.interaction(), prompt)
                    finally:
                        release.set()
                        await asyncio.gather(first_task, second_task)
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



if __name__ == "__main__":
    unittest.main(verbosity=2)
