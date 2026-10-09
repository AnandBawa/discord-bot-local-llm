"""Offline backend availability, Strata handoff, and optional authentication checks."""

import io
import ssl
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import httpx2
from openai import AsyncOpenAI
from PIL import Image

import check_imagegen as image_fixtures
import check_features as feature_fixtures
import check_regressions as fixtures


class BackendChecks(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = image_fixtures.ImageGenerationChecks.asyncSetUp
    asyncTearDown = fixtures.BotChecks.asyncTearDown
    completion = staticmethod(fixtures.BotChecks.completion)
    request = image_fixtures.ImageGenerationChecks.request

    def reset_service(self, previous=None):
        self.service = self.client.imagegen = self.bot.ImageGeneration(self.client.config)
        self.service.backend = previous
        self.service.request = AsyncMock(side_effect=self.request)
        self.calls.clear()
        self.create.reset_mock(side_effect=True)
        self.strata_releases = 0

    def outages(self):
        connection = SimpleNamespace(host="offline.invalid", port=8188, ssl=False)
        return (
            self.bot.aiohttp.ClientConnectorError(connection, ConnectionRefusedError(111, "refused")),
            self.bot.aiohttp.ConnectionTimeoutError("connection establishment timed out"),
        )

    async def test_chat_works_when_comfy_is_offline_at_startup_or_after_images(self):
        for previous in (None, "comfyui"):
            for error in self.outages():
                with self.subTest(previous=previous, error=type(error).__name__):
                    self.reset_service(previous)
                    self.service.request.side_effect = error
                    for _ in range(2):
                        result = await self.bot.request_completion(messages=[])
                        self.assertIs(result, self.create.return_value)
                    self.assertEqual(self.create.await_count, 2)
                    self.assertEqual(self.service.request.await_count, 2)
                    self.assertTrue(all(call.args == ("comfyui", "GET", "/queue")
                                        for call in self.service.request.await_args_list))
                    self.assertEqual(self.service.backend, "lmstudio")
                    self.assertEqual(self.service.work, {"lmstudio": 0, "comfyui": 0})

    async def test_images_work_when_chat_is_offline_at_startup_or_after_chat(self):
        for previous in (None, "lmstudio"):
            for error in self.outages():
                with self.subTest(previous=previous, error=type(error).__name__):
                    self.reset_service(previous)

                    async def offline_chat(backend, method, path, **kwargs):
                        if backend != "comfyui":
                            raise error
                        return await self.request(backend, method, path, **kwargs)

                    self.service.request.side_effect = offline_chat
                    for _ in range(2):
                        data, _ = await self.service.generate("Synthetic", 1024, 1024)
                        with Image.open(io.BytesIO(data)) as image:
                            self.assertEqual(image.size, (1024, 1024))
                    probes = [call for call in self.service.request.await_args_list if call.args[0] != "comfyui"]
                    self.assertEqual(len(probes), 2)
                    self.assertEqual(self.service.backend, "comfyui")
                    self.create.assert_not_awaited()

    async def test_same_backend_rechecks_and_unloads_a_restarted_peer(self):
        for active in ("lmstudio", "comfyui"):
            for provider in ("lmstudio", "strata"):
                with self.subTest(active=active, provider=provider):
                    self.reset_service()
                    self.loaded = {"native-chat": ["chat-alias"]}
                    self.strata_loaded = True
                    offline = True

                    async def recovering(backend, method, path, **kwargs):
                        if offline and ((backend == "comfyui") == (active == "lmstudio")):
                            raise self.outages()[1]
                        handler = self.strata_request if provider == "strata" else self.request
                        return await handler(backend, method, path, **kwargs)

                    async def request_active():
                        if active == "lmstudio":
                            await self.bot.request_completion(messages=[])
                        else:
                            await self.service.generate("Synthetic", 1024, 1024)

                    self.service.request.side_effect = recovering
                    await request_active()
                    self.assertTrue(self.service.peer_offline)
                    offline = False
                    await request_active()
                    self.assertFalse(self.service.peer_offline)
                    if active == "lmstudio":
                        self.assertTrue(self.finished_free)
                    elif provider == "strata":
                        self.assertFalse(self.strata_loaded)
                    else:
                        self.assertFalse(self.loaded["native-chat"])
                    calls = len(self.service.request.await_args_list)
                    releases = self.strata_releases
                    await request_active()
                    if active == "lmstudio":
                        self.assertEqual(len(self.service.request.await_args_list), calls)
                    self.assertEqual(self.strata_releases, releases)

    async def test_chat_server_restart_between_images_is_checked_without_an_observed_outage(self):
        for provider in ("lmstudio", "strata"):
            with self.subTest(provider=provider):
                self.reset_service()
                self.loaded = {"native-chat": ["chat-alias"]}
                self.strata_loaded = True
                self.service.request.side_effect = self.strata_request if provider == "strata" else self.request
                await self.service.generate("First image", 1024, 1024)
                self.assertFalse(self.service.peer_offline)
                self.loaded = {"native-chat": ["chat-alias"]}
                self.strata_loaded = True  # Normal server startup may eagerly load.
                await self.service.generate("After server restart", 1024, 1024)
                if provider == "strata":
                    self.assertFalse(self.strata_loaded)
                    self.assertEqual(self.strata_releases, 2)
                else:
                    self.assertFalse(self.loaded["native-chat"])

    async def test_offline_image_request_does_not_unload_chat_and_can_recover(self):
        for error in self.outages():
            with self.subTest(error=type(error).__name__):
                self.reset_service()
                self.service.request.side_effect = error
                with self.assertRaises(type(error)):
                    await self.service.generate("Offline", 1024, 1024)
                self.service.request.assert_awaited_once_with("comfyui", "GET", "/queue")
                await self.bot.request_completion(messages=[])
                self.service.request.side_effect = self.request
                await self.service.generate("Online again", 1024, 1024)
                self.assertEqual(self.service.backend, "comfyui")
                self.assertEqual(self.service.work, {"lmstudio": 0, "comfyui": 0})

    async def test_unconfirmed_comfy_job_cannot_be_bypassed_as_offline(self):
        self.reset_service("comfyui")
        self.service.active_job = "unconfirmed-job"
        self.service.submission_uncertain = True
        self.service.request.side_effect = self.outages()[1]
        with self.assertRaises(self.bot.aiohttp.ConnectionTimeoutError):
            await self.bot.request_completion(messages=[])
        self.create.assert_not_awaited()

    async def test_ssl_http_and_response_timeouts_are_not_mistaken_for_offline(self):
        connection = SimpleNamespace(host="offline.invalid", port=8188, ssl=True)
        errors = [
            self.bot.aiohttp.ClientConnectorSSLError(connection, ssl.SSLError("bad certificate")),
            self.bot.aiohttp.SocketTimeoutError("server accepted connection but did not answer"),
            self.bot.ImageGenerationError("Unauthorized", status=401),
            self.bot.ImageGenerationError("Server error", status=503),
        ]
        for target in ("lmstudio", "comfyui"):
            for error in errors:
                with self.subTest(target=target, error=type(error).__name__):
                    self.reset_service()

                    async def failed_other(backend, method, path, **kwargs):
                        if (backend == "comfyui") == (target == "lmstudio"):
                            raise error
                        return await self.request(backend, method, path, **kwargs)

                    self.service.request.side_effect = failed_other
                    with self.assertRaises(type(error)):
                        if target == "lmstudio":
                            await self.bot.request_completion(messages=[])
                        else:
                            await self.service.generate("Synthetic", 1024, 1024)
                    self.create.assert_not_awaited()
                    self.assertFalse(any(path == "/prompt" for _, _, path, _ in self.calls))

    async def strata_request(self, backend, method, path, **kwargs):
        if backend == "lmstudio":
            raise self.bot.ImageGenerationError("Not found", status=404)
        if backend == "strata":
            if (method, path) == ("GET", "/health"):
                return {"service": "strata", "loaded": self.strata_loaded, "model": self.client.config.model}
            if (method, path) == ("POST", "/unload"):
                self.assertEqual(kwargs["body"], {})
                status = "unloaded" if self.strata_loaded else "not loaded"
                self.strata_releases += int(self.strata_loaded)
                self.strata_loaded = False
                return {"status": status}
            raise AssertionError(f"Unexpected Strata call: {method} {path}")
        return await self.request(backend, method, path, **kwargs)

    async def test_strata_unloads_only_on_switch_and_chat_releases_comfy_first(self):
        self.reset_service("lmstudio")
        self.strata_loaded = True
        self.service.request.side_effect = self.strata_request
        await self.service.generate("First image", 1024, 1024)
        self.assertFalse(self.strata_loaded)
        self.assertEqual(self.service.chat_provider, "strata")
        self.assertFalse(self.service.unload_pending)
        self.assertEqual(self.strata_releases, 1)
        await self.service.generate("Second image", 1024, 1024)
        self.assertEqual(self.strata_releases, 1)

        async def chat(**kwargs):
            self.assertTrue(self.finished_free)
            self.assertEqual(kwargs["model"], self.client.config.model)
            self.strata_loaded = True  # Strata loads on the completion request.
            return self.completion("Answer")

        self.create.side_effect = chat
        await self.bot.request_completion(messages=[])
        await self.service.generate("After chat", 1024, 1024)
        controls = [call.args for call in self.service.request.await_args_list if call.args[1] == "POST"]
        self.assertEqual(self.strata_releases, 2)
        self.assertNotIn(("lmstudio", "POST", "/models/unload"), controls)

    async def test_strata_already_unloaded_still_checks_that_it_is_not_busy(self):
        self.reset_service()
        self.strata_loaded = False
        self.service.request.side_effect = self.strata_request
        await self.service.generate("Synthetic", 1024, 1024)
        self.assertEqual(sum(call.args == ("strata", "POST", "/unload")
                             for call in self.service.request.await_args_list), 1)
        self.assertFalse(self.service.unload_pending)

    async def test_strata_still_loading_or_unsupported_unload_blocks_images(self):
        for result in (self.bot.ImageGenerationError("busy", status=409), {"status": "unsupported"}, {}):
            with self.subTest(result=str(result)):
                self.reset_service()
                self.strata_loaded = False

                async def loading(backend, method, path, **kwargs):
                    if backend == "strata" and path == "/unload":
                        if isinstance(result, Exception):
                            raise result
                        return result
                    return await self.strata_request(backend, method, path, **kwargs)

                self.service.request.side_effect = loading
                with self.assertRaises(self.bot.ImageGenerationError):
                    await self.service.generate("Synthetic", 1024, 1024)
                self.assertIn("chat", self.service.unload_pending)
                self.assertFalse(any(path == "/prompt" for _, _, path, _ in self.calls))

    async def test_strata_must_confirm_unloaded_before_an_image_is_submitted(self):
        self.reset_service()
        self.strata_loaded = True

        async def still_loaded(backend, method, path, **kwargs):
            if backend == "strata" and path == "/unload":
                return {"status": "unloaded"}  # Acknowledgement alone is not enough.
            return await self.strata_request(backend, method, path, **kwargs)

        self.service.request.side_effect = still_loaded
        with patch.object(self.bot, "IMAGEGEN_SWITCH_TIMEOUT", 0.03):
            with self.assertRaises(TimeoutError):
                await self.service.generate("Synthetic", 1024, 1024)
        self.assertIn("chat", self.service.unload_pending)
        self.assertFalse(any(path == "/prompt" for _, _, path, _ in self.calls))

    async def test_strata_identity_and_model_ownership_are_checked(self):
        invalid_states = [
            {"service": "other", "loaded": True, "model": "chat-alias"},
            {"service": "strata", "loaded": "false", "model": "chat-alias"},
            {"service": "strata", "loaded": True, "model": "another-apps-model"},
        ]
        for state in invalid_states:
            with self.subTest(state=state):
                self.reset_service()

                async def bad_state(backend, method, path, **kwargs):
                    if backend == "strata" and path == "/health":
                        return state
                    return await self.strata_request(backend, method, path, **kwargs)

                self.service.request.side_effect = bad_state
                with self.assertRaises(self.bot.ImageGenerationError):
                    await self.service.generate("Synthetic", 1024, 1024)
                self.assertFalse(any(call.args[1] == "POST" for call in self.service.request.await_args_list))

    async def test_failed_chat_unload_must_be_confirmed_before_images(self):
        for provider in ("lmstudio", "strata"):
            for failure in (*self.outages(), self.bot.ImageGenerationError("busy", status=409)):
                with self.subTest(provider=provider, failure=type(failure).__name__):
                    self.reset_service()
                    self.loaded = {"native-chat": ["chat-alias"]}
                    self.strata_loaded = True

                    async def failed_unload(backend, method, path, **kwargs):
                        if backend != "comfyui" and method == "POST":
                            raise failure
                        handler = self.strata_request if provider == "strata" else self.request
                        return await handler(backend, method, path, **kwargs)

                    self.service.request.side_effect = failed_unload
                    with self.assertRaises(type(failure)):
                        await self.service.generate("Synthetic", 1024, 1024)
                    self.assertIn("chat", self.service.unload_pending)

                    async def now_offline(backend, method, path, **kwargs):
                        if backend != "comfyui":
                            raise self.outages()[1]
                        return await self.request(backend, method, path, **kwargs)

                    self.service.request.side_effect = now_offline
                    with self.assertRaises(self.bot.aiohttp.ConnectionTimeoutError):
                        await self.service.generate("Still uncertain", 1024, 1024)
                    self.assertFalse(any(path == "/prompt" for _, _, path, _ in self.calls))
                    self.service.request.side_effect = self.strata_request if provider == "strata" else self.request
                    await self.service.generate("Recovered", 1024, 1024)
                    self.assertFalse(self.service.unload_pending)


class AuthenticationChecks(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixtures.BotChecks.asyncSetUp
    asyncTearDown = fixtures.BotChecks.asyncTearDown
    completion = staticmethod(fixtures.BotChecks.completion)
    chat = feature_fixtures.FeatureChecks.chat

    async def test_real_sdk_accepts_optional_key_and_sends_only_configured_credentials(self):
        for key in (None, "", " \t ", " synthetic-key "):
            with self.subTest(key=key):
                env = {"LLM_BASE_URL": "https://chat.invalid/v1"}
                if key is not None:
                    env["LLM_API_KEY"] = key
                expected = f"Bearer {key.strip()}" if key and key.strip() else None

                def respond(request):
                    self.assertEqual(request.headers.get("Authorization"), expected)
                    return httpx2.Response(200, json={
                        "id": "synthetic", "object": "chat.completion", "created": 0, "model": "local-model",
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Answer"},
                                     "finish_reason": "stop"}],
                    })

                def factory(**kwargs):
                    return AsyncOpenAI(**kwargs, http_client=httpx2.AsyncClient(
                        transport=httpx2.MockTransport(respond),
                    ))

                started = self.bot.MyAIClient(intents=self.bot.intents)
                started.config = self.bot.Config(env)
                started.db_path = "optional-key.sqlite3"
                with patch.object(self.bot, "AsyncOpenAI", side_effect=factory), \
                        patch.object(self.bot.tree, "sync", new=AsyncMock()), \
                        patch.dict("os.environ", {"OPENAI_API_KEY": "unrelated-environment-key"}):
                    try:
                        await started.setup_hook()
                        with patch.object(self.bot, "client", started):
                            result = await self.bot.request_completion(messages=[{"role": "user", "content": "Hi"}])
                            self.assertEqual(result.choices[0].message.content, "Answer")
                    finally:
                        await started.close()

    async def test_chat_connection_failure_has_a_clear_message(self):
        self.create.side_effect = self.bot.APIConnectionError(request=httpx2.Request("POST", "https://chat.invalid/v1/chat/completions"))
        message = self.chat()
        result = await self.bot.generate_ai_response([], message, False)
        self.assertIsNone(result)
        self.assertIn("Chat is unavailable", message.reply.call_args.args[0])

    async def test_native_api_routes_and_headers_do_not_send_chat_credentials_to_comfy(self):
        responses = {path: (200, {"Content-Type": "application/json"}, b'{}')
                     for path in ("/api/v1/models", "/health", "/unload", "/queue")}
        for key in ("", "synthetic-key"):
            with self.subTest(authenticated=bool(key)):
                config = self.bot.Config({"LLM_BASE_URL": "http://public.example/v1",
                                          "COMFYUI_BASE_URL": "http://public.example", "LLM_API_KEY": key})
                service = self.bot.ImageGeneration(config)
                observed = []
                trace = self.bot.aiohttp.TraceConfig()

                async def on_request_start(session, context, params):
                    observed.append((params.url.path, params.headers.get("Authorization")))

                trace.on_request_start.append(on_request_start)
                service.session = self.bot.aiohttp.ClientSession(trace_configs=[trace])
                try:
                    async with fixtures.BotChecks.public_http(self, responses):
                        await service.request("lmstudio", "GET", "/models")
                        await service.request("strata", "GET", "/health")
                        await service.request("strata", "POST", "/unload", body={})
                        await service.request("comfyui", "GET", "/queue")
                finally:
                    await service.close()
                auth = f"Bearer {key}" if key else None
                self.assertEqual(observed, [("/api/v1/models", auth), ("/health", auth),
                                            ("/unload", auth), ("/queue", None)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
