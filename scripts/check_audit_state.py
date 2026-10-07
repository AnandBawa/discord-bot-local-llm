"""Offline regression checks for audit findings A04 and A08–A10.

Run: venv_bot/bin/python scripts/check_audit_state.py
Uses temporary SQLite, synthetic Discord SDK objects, mocked transports/providers,
and the existing blocked-socket fixtures. Never reads .env or runtime data.
"""

import asyncio
import contextlib
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import ANY, AsyncMock, patch

import aiosqlite
import discord
from discord.webhook.async_ import async_context

import check_features as features
import check_regressions as fixtures


USER = {"id": "42", "username": "Synthetic User", "discriminator": "0", "avatar": None}
BOT = {"id": "99", "username": "Synthetic Bot", "discriminator": "0", "avatar": None, "bot": True}


class InteractionTransport:
    """Exercise real webhook serialization while enforcing Discord's text limit."""
    def __init__(self, channel_id):
        self.channel_id = channel_id
        self.sent = []
        self.next_id = 500

    async def create_interaction_response(self, interaction_id, token, **kwargs):
        return {"interaction": {"id": str(interaction_id), "response_message_loading": True}}

    def message(self, payload):
        content = payload.get("content", "")
        if len(content) > 2000:
            raise discord.HTTPException(
                SimpleNamespace(status=400, reason="Bad Request"),
                {"code": 50035, "message": "Message content exceeds 2000 characters"},
            )
        self.next_id += 1
        return {
            "id": str(self.next_id), "channel_id": str(self.channel_id), "author": BOT,
            "content": content, "type": 20, "flags": payload.get("flags", 0),
            "attachments": [], "embeds": [], "mentions": [], "mention_roles": [],
            "timestamp": "2026-10-05T00:00:00+00:00", "edited_timestamp": None,
            "tts": False, "mention_everyone": False,
        }

    async def execute_webhook(self, *args, **kwargs):
        payload = kwargs["payload"]
        result = self.message(payload)
        self.sent.append(payload)
        return result

    async def edit_webhook_message(self, *args, **kwargs):
        return self.message(kwargs["payload"])

    async def edit_original_interaction_response(self, *args, **kwargs):
        return self.message(kwargs["payload"])


class AuditStateChecks(unittest.IsolatedAsyncioTestCase):
    asyncTearDown = fixtures.BotChecks.asyncTearDown
    completion = staticmethod(fixtures.BotChecks.completion)
    interaction = staticmethod(fixtures.BotChecks.interaction)
    seed = fixtures.BotChecks.seed
    count = fixtures.BotChecks.count
    chat = features.FeatureChecks.chat

    async def asyncSetUp(self):
        await fixtures.BotChecks.asyncSetUp(self)
        self.create.return_value.choices[0].message.tool_calls = []
        self.patches.enter_context(patch.object(self.bot, "CHUNK_MESSAGE_DELAY", 0))

    def sdk_interaction(self, *, guild=True):
        self.client._connection.user = discord.ClientUser(state=self.client._connection, data=BOT)
        data = {
            "id": "1234567890123456789", "application_id": "99", "token": "synthetic-token",
            "type": 2, "version": 1, "locale": "en-US", "attachment_size_limit": 10485760,
            "context": 0 if guild else 1, "data": {"id": "25", "name": "role", "type": 1},
            "channel": {
                "id": "10" if guild else "9420", "type": 0 if guild else 1,
                "name": "general", "position": 0, "permission_overwrites": [], "recipients": [USER],
            },
        }
        if guild:
            data["guild_id"] = "1"
            data["member"] = {
                "user": USER, "roles": [], "joined_at": "2026-01-01T00:00:00+00:00",
                "deaf": False, "mute": False, "flags": 0,
            }
        else:
            data["user"] = USER
        return discord.Interaction(data=data, state=self.client._connection)

    async def history_count(self, key):
        cursor = await self.client.db_conn.execute(
            "SELECT COUNT(*) FROM chat_history WHERE server_id = ?", (key,),
        )
        return (await cursor.fetchone())[0]

    async def test_generated_replies_restrict_mentions_in_serialized_discord_payloads(self):
        transport = InteractionTransport(10)
        payloads = []

        async def send(channel_id, *, params):
            payloads.append(dict(params.payload))
            return transport.message(params.payload)

        @contextlib.asynccontextmanager
        async def typing(channel):
            yield

        def message(guild):
            channel = self.sdk_interaction(guild=guild).channel
            data = transport.message({"content": "Synthetic incoming message"})
            data.update(author=USER, type=0)
            return discord.Message(state=self.client._connection, channel=channel, data=data)

        self.client.http.send_message = AsyncMock(side_effect=send)
        mentions = "@everyone @here <@&123> <@456>"
        with patch.object(self.bot, "safe_typing", typing), \
                patch.object(self.bot, "can_read_history", return_value=True):
            await self.bot.send_chunked_message(message(True), mentions)
            await self.bot.send_chunked_message(message(True), "a " * 975 + mentions)
            await self.bot.send_chunked_message(message(False), mentions)
        self.assertEqual(len(payloads), 4)
        for index, payload in enumerate(payloads):
            self.assertEqual(payload["allowed_mentions"]["parse"], [])
            self.assertFalse(payload["allowed_mentions"].get("users"))
            self.assertEqual(payload["allowed_mentions"].get("replied_user", False), index < 2)
            self.assertEqual("message_reference" in payload, index < 2)

        token = async_context.set(transport)
        try:
            await self.bot.send_chunked_message(self.sdk_interaction(), mentions,
                                                is_interaction_followup=True)
        finally:
            async_context.reset(token)
        self.assertEqual(transport.sent[-1]["allowed_mentions"], {"parse": []})

        with patch.object(self.bot, "can_read_history", return_value=False):
            await self.bot.send_chunked_message(message(True), mentions)
        self.assertEqual(payloads[-1]["allowed_mentions"]["parse"], [])
        self.assertEqual(payloads[-1]["allowed_mentions"]["users"], [42])

    async def test_long_server_role_can_be_saved_and_viewed_in_full(self):
        transport = InteractionTransport(10)
        token = async_context.set(transport)
        persona = "A detailed persona. " * 200
        try:
            await self.bot.cmd_role.callback(self.sdk_interaction(), persona)
            await self.seed()
            version = dict(self.client.conversation_versions)
            transport.sent.clear()
            await self.bot.cmd_role.callback(self.sdk_interaction())
            self.assertGreater(len(transport.sent), 1)
            self.assertTrue(all(0 < len(part["content"]) <= 2000 for part in transport.sent))
            self.assertIn(persona, "".join(part["content"] for part in transport.sent))
            self.assertEqual(await self.history_count("channel:10"), 2)
            self.assertEqual(self.client.conversation_versions, version)
        finally:
            async_context.reset(token)

    async def test_long_dm_role_confirmation_keeps_every_followup_private(self):
        transport = InteractionTransport(9420)
        token = async_context.set(transport)
        persona = "A private detailed persona. " * 150
        try:
            await self.seed(server_id="dm:42")
            await self.bot.cmd_role.callback(self.sdk_interaction(guild=False), persona)
            self.assertEqual(await self.bot.get_persona("dm:42"), persona)
            self.assertEqual(await self.history_count("dm:42"), 0)
            self.assertGreater(len(transport.sent), 1)
            self.assertTrue(all(part.get("flags", 0) & 64 for part in transport.sent))
            self.assertTrue(all(0 < len(part["content"]) <= 2000 for part in transport.sent))
            self.assertIn(persona, "".join(part["content"] for part in transport.sent))
            transport.sent.clear()
            await self.bot.cmd_role.callback(self.sdk_interaction(guild=False))
            self.assertGreater(len(transport.sent), 1)
            self.assertTrue(all(part.get("flags", 0) & 64 for part in transport.sent))
            self.assertIn(persona, "".join(part["content"] for part in transport.sent))
        finally:
            async_context.reset(token)

    async def test_clear_and_role_during_chunk_delay_stop_only_old_continuations(self):
        real_sleep = asyncio.sleep
        for prompt in (None, "New persona"):
            with self.subTest(prompt=prompt):
                message = self.chat()
                other = self.chat(channel_id=20)
                version = self.client.conversation_versions.get("channel:10", 0)
                delay_entered, release_delay = asyncio.Event(), asyncio.Event()

                async def delayed(seconds):
                    if seconds == 0.125:
                        delay_entered.set()
                        await release_delay.wait()
                    else:
                        await real_sleep(seconds)

                with patch.object(self.bot, "CHUNK_MESSAGE_DELAY", 0.125), \
                        patch.object(self.bot.asyncio, "sleep", new=delayed):
                    delivery = asyncio.create_task(self.bot.save_and_send_response(
                        message, "channel:10", "Old input", "x" * 4500, version,
                    ))
                    try:
                        await asyncio.wait_for(delay_entered.wait(), 2)
                        message.reply.assert_awaited_once()
                        if prompt is None:
                            await self.bot.cmd_clear.callback(self.interaction())
                        else:
                            await self.bot.cmd_role.callback(self.interaction(), prompt)
                        await self.bot.save_and_send_response(
                            other, "channel:20", "Independent input", "Independent answer",
                        )
                        other.reply.assert_awaited_once_with("Independent answer", allowed_mentions=ANY)
                    finally:
                        release_delay.set()
                        await asyncio.wait_for(delivery, 2)
                message.channel.send.assert_not_awaited()
                self.assertEqual(await self.history_count("channel:10"), 0)

    async def test_clear_after_saving_prevents_the_first_undispatched_reply(self):
        message = self.chat()
        original_send = self.bot.send_chunked_message

        async def clear_then_deliver(*args, **kwargs):
            await self.bot.cmd_clear.callback(self.interaction())
            return await original_send(*args, **kwargs)

        with patch.object(self.bot, "send_chunked_message", new=clear_then_deliver):
            await self.bot.save_and_send_response(message, "channel:10", "Old input", "Old answer", 0)
        message.reply.assert_not_awaited()
        message.channel.send.assert_not_awaited()
        self.assertEqual(await self.history_count("channel:10"), 0)

    async def test_clear_during_failed_native_reply_prevents_stale_fallback(self):
        message = self.chat()

        async def failed_reply(text, **kwargs):
            await self.bot.cmd_clear.callback(self.interaction())
            raise discord.Forbidden(SimpleNamespace(status=403, reason="Forbidden"), "Synthetic denial")

        message.reply.side_effect = failed_reply
        await self.bot.save_and_send_response(message, "channel:10", "Old input", "Old answer", 0)
        message.reply.assert_awaited_once()
        message.channel.send.assert_not_awaited()
        self.assertEqual(await self.history_count("channel:10"), 0)

    async def test_real_failed_commit_preserves_version_and_allows_pending_answer(self):
        await self.client.db_conn.execute("PRAGMA foreign_keys = ON")
        await self.client.db_conn.execute("CREATE TABLE constraint_parent (id INTEGER PRIMARY KEY)")
        await self.client.db_conn.execute(
            "CREATE TABLE constraint_child (id INTEGER REFERENCES constraint_parent(id) "
            "DEFERRABLE INITIALLY DEFERRED)",
        )
        await self.client.db_conn.commit()
        for channel_id, prompt in ((10, None), (20, "Changed persona")):
            with self.subTest(prompt=prompt):
                key = f"channel:{channel_id}"
                interaction = self.interaction(channel_id=channel_id)
                await self.bot.cmd_role.callback(interaction, "Original persona")
                await self.seed(server_id=key)
                versions = dict(self.client.conversation_versions)
                await self.client.db_conn.execute(
                    "CREATE TRIGGER deferred_failure AFTER DELETE ON chat_history "
                    "BEGIN INSERT INTO constraint_child VALUES (999); END",
                )
                await self.client.db_conn.commit()
                with self.assertRaises(aiosqlite.IntegrityError):
                    if prompt is None:
                        await self.bot.cmd_clear.callback(self.interaction(channel_id=channel_id))
                    else:
                        await self.bot.cmd_role.callback(self.interaction(channel_id=channel_id), prompt)
                self.assertEqual(await self.bot.get_persona(key), "Original persona")
                self.assertEqual(await self.history_count(key), 2)
                self.assertEqual(self.client.conversation_versions, versions)
                message = self.chat(channel_id=channel_id)
                await self.bot.save_and_send_response(
                    message, key, "Accepted before failure", "Still a valid answer", versions[key],
                )
                message.reply.assert_awaited_once_with("Still a valid answer", allowed_mentions=ANY)
                self.assertEqual(await self.history_count(key), 4)
                await self.client.db_conn.execute("DROP TRIGGER deferred_failure")
                await self.client.db_conn.commit()

    async def test_cancellation_at_real_commit_still_invalidates_old_answers(self):
        for prompt in (None, "Changed persona"):
            with self.subTest(prompt=prompt):
                await self.bot.cmd_role.callback(self.interaction(), "Original persona")
                await self.seed()
                version = self.client.conversation_versions["channel:10"]
                loop = asyncio.get_running_loop()

                def cancel_at_commit(statement):
                    if statement == "COMMIT":
                        loop.call_soon_threadsafe(task.cancel)

                await self.client.db_conn.set_trace_callback(cancel_at_commit)
                callback = (self.bot.cmd_clear.callback(self.interaction()) if prompt is None else
                            self.bot.cmd_role.callback(self.interaction(), prompt))
                task = asyncio.create_task(callback)
                try:
                    with self.assertRaises(asyncio.CancelledError):
                        await asyncio.wait_for(task, 2)
                finally:
                    await self.client.db_conn.set_trace_callback(None)
                self.assertEqual(self.client.conversation_versions["channel:10"], version + 1)
                self.assertEqual(await self.history_count("channel:10"), 0)
                self.assertEqual(await self.bot.get_persona("channel:10"), prompt or "Original persona")
                message = self.chat()
                await self.bot.save_and_send_response(message, "channel:10", "Old input", "Old answer", version)
                message.reply.assert_not_awaited()
                self.assertEqual(await self.history_count("channel:10"), 0)

    async def test_repeated_cancellation_drains_commit_before_releasing_database_lock(self):
        await self.seed()
        entered, release_commit = asyncio.Event(), threading.Event()
        loop = asyncio.get_running_loop()
        version = self.client.conversation_versions.get("channel:10", 0)

        def hold_commit(statement):
            if statement == "COMMIT":
                loop.call_soon_threadsafe(entered.set)
                release_commit.wait(5)

        async def unrelated_write():
            async with self.bot.history_transaction():
                await self.client.db_conn.execute(
                    "INSERT INTO chat_history (server_id, role, content) VALUES ('channel:20', 'user', 'Other')",
                )

        await self.client.db_conn.set_trace_callback(hold_commit)
        task = asyncio.create_task(self.bot.cmd_clear.callback(self.interaction()))
        other = None
        try:
            await asyncio.wait_for(entered.wait(), 2)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            other = asyncio.create_task(unrelated_write())
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            self.assertFalse(other.done())
            self.assertTrue(self.client.db_lock.locked())
            self.assertEqual(self.client.conversation_versions.get("channel:10", 0), version)
        finally:
            release_commit.set()
            await asyncio.gather(task, *([other] if other else []), return_exceptions=True)
            await self.client.db_conn.set_trace_callback(None)
        self.assertTrue(task.cancelled())
        if other is not None:
            other.result()
        self.assertEqual(self.client.conversation_versions["channel:10"], version + 1)
        self.assertEqual(await self.history_count("channel:10"), 0)
        self.assertEqual(await self.history_count("channel:20"), 1)

    async def test_shutdown_drains_active_and_waiting_chats_before_closing_resources(self):
        started, cleaned = asyncio.Event(), asyncio.Event()

        async def blocked_completion(**kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                # Cancellation cleanup must still have a working database.
                await self.client.db_conn.execute("SELECT 1")
                cleaned.set()

        self.create.side_effect = blocked_completion
        active = asyncio.create_task(self.bot.on_message(self.chat()))
        queued = None
        original_close = self.client.db_conn.close

        async def close_database():
            self.assertTrue(cleaned.is_set())
            self.assertTrue(active.done())
            self.assertTrue(queued.done())
            await original_close()

        try:
            await asyncio.wait_for(started.wait(), 2)
            queued = asyncio.create_task(self.bot.on_message(self.chat(content="Queued question")))
            await asyncio.sleep(0)
            self.create.assert_awaited_once()
            with patch.object(self.client.db_conn, "close", new=close_database):
                await asyncio.wait_for(self.client.close(), 2)
            self.assertTrue(active.cancelled())
            self.assertTrue(queued.cancelled())
        finally:
            for task in (active, queued):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(active, *([queued] if queued else []), return_exceptions=True)

    async def test_shutdown_stops_new_chat_admission_while_draining(self):
        started, cancelling, release_cleanup = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def blocked_completion(**kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelling.set()
                await release_cleanup.wait()
                raise

        self.create.side_effect = blocked_completion
        active = asyncio.create_task(self.bot.on_message(self.chat()))
        closing = None
        try:
            await asyncio.wait_for(started.wait(), 2)
            closing = asyncio.create_task(self.client.close())
            await asyncio.wait_for(cancelling.wait(), 2)
            late = self.chat(channel_id=20)
            await asyncio.wait_for(self.bot.on_message(late), 2)
            self.create.assert_awaited_once()
            late.reply.assert_not_awaited()
            late.channel.send.assert_not_awaited()
            self.assertFalse(closing.done())
        finally:
            release_cleanup.set()
            if (closing is None or closing.done()) and not active.done():
                active.cancel()
            await asyncio.gather(active, *([closing] if closing else []), return_exceptions=True)
        if closing is not None:
            closing.result()

    async def test_shutdown_keeps_model_reservation_until_cancelled_request_finishes(self):
        service = self.client.imagegen = self.bot.ImageGeneration(self.client.config)
        service.backend = "lmstudio"
        service.request = AsyncMock(side_effect=AssertionError("Backend requests disabled"))
        started, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def completing(**kwargs):
            started.set()
            await release.wait()
            finished.set()
            return self.create.return_value

        self.create.side_effect = completing
        active = asyncio.create_task(self.bot.on_message(self.chat()))
        closing = None
        try:
            await asyncio.wait_for(started.wait(), 2)
            closing = asyncio.create_task(self.client.close())
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            self.assertFalse(finished.is_set())
            self.assertFalse(closing.done())
            self.assertGreater(service.work["lmstudio"], 0)
            self.assertIsNotNone(service.busy_message("comfyui"))
            await self.client.db_conn.execute("SELECT 1")
        finally:
            release.set()
            if closing is None and not active.done():
                active.cancel()
            await asyncio.gather(active, *([closing] if closing else []), return_exceptions=True)
        if closing is not None:
            closing.result()
        self.assertTrue(active.cancelled())
        self.assertTrue(finished.is_set())
        self.assertEqual(service.work, {"lmstudio": 0, "comfyui": 0})
        service.request.assert_not_awaited()


if __name__ == "__main__":
    unittest.main(verbosity=2)
