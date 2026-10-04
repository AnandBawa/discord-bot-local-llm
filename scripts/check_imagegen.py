"""Offline ComfyUI API and GPU handoff checks; no live models or Discord login."""

import asyncio
import contextlib
import copy
import io
import json
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from PIL import Image

import check_regressions as fixtures


class ImageGenerationChecks(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await fixtures.BotChecks.asyncSetUp(self)
        self.client.config.comfy_url = "http://comfy.invalid:8188"
        self.client.config.model = "chat-alias"
        self.client.config.embedding_model = "embedding-alias"
        self.service = self.client.imagegen = self.bot.ImageGeneration(self.client.config)
        self.service.backend = "lmstudio"
        self.patches.enter_context(patch.object(self.bot, "IMAGEGEN_POLL_INTERVAL", 0.001))
        self.service.request = AsyncMock(side_effect=self.request)
        self.calls, self.jobs, self.running = [], {}, []
        self.loaded = {"native-chat": ["chat-alias"], "native-embedding": ["embedding-alias"]}
        self.hold_image = False
        self.started = asyncio.Event()
        self.finished_free = False
        self.barriers = 0

    asyncTearDown = fixtures.BotChecks.asyncTearDown
    completion = staticmethod(fixtures.BotChecks.completion)

    async def request(self, backend, method, path, *, body=None, params=None, binary=False):
        self.calls.append((backend, method, path, copy.deepcopy(body)))
        if backend == "lmstudio":
            if path == "/models":
                return {"models": [{"key": key, "loaded_instances": [{"id": ident} for ident in ids]}
                                   for key, ids in self.loaded.items()]}
            if path == "/models/unload":
                for ids in self.loaded.values():
                    if body["instance_id"] in ids:
                        ids.remove(body["instance_id"])
                return body
        if path == "/queue":
            return {"queue_running": [[0, ident] for ident in self.running], "queue_pending": []}
        if path == "/free":
            self.finished_free = False
            self.barriers = 0
            return {}
        if path == "/prompt":
            job_id = body["prompt_id"]
            self.jobs[job_id] = copy.deepcopy(body["prompt"])
            self.running.append(job_id)
            if "213" in body["prompt"]:
                self.started.set()
            return {"prompt_id": job_id}
        if path.startswith("/history/"):
            job_id = path.rsplit("/", 1)[1]
            workflow = self.jobs.get(job_id)
            if workflow is None:
                return {}
            if "213" in workflow and self.hold_image:
                return {}
            if job_id in self.running:
                self.running.remove(job_id)
            if "213" not in workflow:
                self.barriers += 1
                # The first completion can precede /free processing. Only the
                # second one proves the worker has traversed that stage.
                self.finished_free = self.barriers >= 2
                outputs = {"2": {"text": ["barrier"]}}
            else:
                outputs = {"213": {"images": [{"filename": job_id + ".png", "subfolder": "krea", "type": "output"}]},
                           "999": {"images": [{"filename": "wrong.png", "subfolder": "", "type": "output"}]}}
            return {job_id: {"status": {"completed": True, "status_str": "success"}, "outputs": outputs}}
        if path == "/view":
            job_id = params["filename"].removesuffix(".png")
            workflow = self.jobs[job_id]
            width, height = (workflow["232"]["inputs"][name] for name in ("width", "height"))
            output = io.BytesIO()
            with Image.new("RGB", (width, height), "blue") as image:
                image.save(output, format="PNG")
            return output.getvalue()
        if path.startswith("/api/jobs/") and path.endswith("/cancel"):
            job_id = path.split("/")[3]
            if job_id not in self.jobs:
                return {"cancelled": False}
            was_running = job_id in self.running
            if was_running:
                self.running.remove(job_id)
            return {"cancelled": was_running}
        raise AssertionError(f"Unexpected API operation: {backend} {method} {path}")

    async def test_workflow_uses_one_dynamic_resolution_and_literal_prompt(self):
        prompt = '  A sign saying "hello"\n{"48": "literal prompt text"} @everyone  '
        graph = self.service.workflow(prompt, 1080, 1920)
        self.assertEqual(graph["48"]["inputs"]["value"], prompt)
        self.assertEqual(graph["232"]["inputs"], {"width": 1056, "height": 1888, "batch_size": 1})
        self.assertEqual(graph["324"]["class_type"], "ImageScaleBy")
        self.assertEqual(graph["324"]["inputs"]["scale_by"], 0.5)
        self.assertEqual(graph["324"]["inputs"]["image"], ["323", 0])
        self.assertNotIn("width", graph["324"]["inputs"])
        self.assertNotIn("height", graph["324"]["inputs"])
        self.assertEqual([nid for nid, node in graph.items() if node["class_type"] == "EmptyLatentImage"], ["232"])
        self.assertEqual(graph["213"]["inputs"]["images"], ["324", 0])
        graph["48"]["inputs"]["value"] = "changed"
        next_graph = self.service.workflow("next", 64, 64)
        self.assertEqual(next_graph["48"]["inputs"]["value"], "next")
        self.assertEqual(next_graph["323"], graph["323"])
        self.assertEqual(next_graph["324"], graph["324"])

    async def test_repeated_images_reuse_comfy_without_any_lm_inference(self):
        for prompt, requested, expected in (("first", (64, 80), (896, 1120)),
                                            ("second\nexactly", (3840, 2160), (1888, 1056))):
            raw = await self.service.generate(prompt, *requested)
            with Image.open(io.BytesIO(raw)) as picture:
                self.assertEqual(picture.size, expected)
        submitted = [graph for graph in self.jobs.values() if "213" in graph]
        self.assertEqual([graph["48"]["inputs"]["value"] for graph in submitted], ["first", "second\nexactly"])
        self.assertEqual(sum(path == "/models/unload" for _, _, path, _ in self.calls), 2)
        self.assertFalse(any(path == "/free" for _, _, path, _ in self.calls))
        self.assertEqual(self.service.backend, "comfyui")
        self.create.assert_not_awaited()

    async def test_round_trip_preserves_chat_and_embedding_custom_aliases(self):
        await self.service.generate("first", 64, 64)
        async def infer(**kwargs):
            self.assertTrue(self.finished_free)
            self.assertEqual(kwargs["model"], "native-chat")
            self.loaded["native-chat"] = ["native-chat"]
            return self.completion("hello")
        self.create.side_effect = infer
        await self.bot.request_completion(messages=[])
        await self.bot.request_completion(messages=[])
        self.assertEqual(sum(path == "/free" for _, _, path, _ in self.calls), 1)
        self.assertEqual(self.barriers, 2)
        provider = self.bot.LocalAPIEmbeddingFunction("http://lm.invalid/v1", "synthetic", "embedding-alias", self.service.resolve_model)
        self.client.custom_ef = self.bot.ResilientEmbeddingFunction(provider)
        response = Mock()
        response.json.return_value = {"data": [{"embedding": [1.0]}]}
        with patch.object(self.bot.requests, "post", return_value=response) as post:
            await self.bot.request_embeddings(["hello"], "retrieval.query")
        self.assertEqual(post.call_args.kwargs["json"]["model"], "native-embedding")
        self.loaded["native-embedding"] = ["native-embedding"]
        await self.service.generate("next", 64, 64)
        unloaded = [body["instance_id"] for _, _, path, body in self.calls if path == "/models/unload"]
        self.assertEqual(unloaded, ["chat-alias", "embedding-alias", "native-chat", "native-embedding"])

    async def test_concurrent_local_requests_drain_before_image_and_new_chat_waits(self):
        release = asyncio.Event()
        two_running = asyncio.Event()
        async def infer(**kwargs):
            if self.service.local_active == 2:
                two_running.set()
            await release.wait()
            return self.completion("answer")
        self.create.side_effect = infer
        first = asyncio.create_task(self.bot.request_completion(messages=[]))
        second = asyncio.create_task(self.bot.request_completion(messages=[]))
        await asyncio.wait_for(two_running.wait(), 1)
        self.hold_image = True
        image = asyncio.create_task(self.service.generate("wait", 64, 64))
        await asyncio.sleep(0.01)
        third = asyncio.create_task(self.bot.request_completion(messages=[]))
        self.assertEqual(self.create.await_count, 2)
        self.assertFalse(self.started.is_set())
        release.set()
        await asyncio.gather(first, second)
        await asyncio.wait_for(self.started.wait(), 1)
        self.assertEqual(self.create.await_count, 2)
        self.hold_image = False
        await image
        await third
        self.assertEqual(self.create.await_count, 3)
        self.assertTrue(self.finished_free)

    async def test_cancelled_embedding_thread_keeps_gpu_until_it_finishes(self):
        started, release = threading.Event(), threading.Event()
        def embed(*args):
            started.set()
            release.wait(2)
            return [[1.0]]
        self.client.custom_ef = SimpleNamespace(embed=embed)
        embedding = asyncio.create_task(self.bot.request_embeddings(["fact"], "retrieval.query"))
        try:
            await asyncio.to_thread(started.wait, 1)
            self.assertTrue(started.is_set())
            embedding.cancel()
            image = asyncio.create_task(self.service.generate("after embedding", 64, 64))
            await asyncio.sleep(0.02)
            self.assertFalse(self.started.is_set())
            self.assertFalse(embedding.done())
        finally:
            release.set()
        with self.assertRaises(asyncio.CancelledError):
            await embedding
        await image

    async def test_cloud_embedding_stays_cloud_when_cooldown_expires_before_thread(self):
        primary = SimpleNamespace(embed=Mock(side_effect=AssertionError("local GPU is busy")))
        fallback = SimpleNamespace(embed=Mock(return_value=[[2.0]]))
        ef = self.client.custom_ef = self.bot.ResilientEmbeddingFunction(primary, fallback)
        ef.dead_until = time.monotonic() + 100
        original = asyncio.to_thread
        async def delayed(function, *args, **kwargs):
            ef.dead_until = 0
            return await original(function, *args, **kwargs)
        async with self.service.entry:
            self.service.backend = "comfyui"
            with patch.object(asyncio, "to_thread", side_effect=delayed):
                result = await asyncio.wait_for(self.bot.request_embeddings(["cloud"], "retrieval.query"), 1)
        self.assertEqual(result, [[2.0]])
        self.assertTrue(ef.last_used_fallback)
        primary.embed.assert_not_called()

    async def test_startup_only_ignores_an_initial_refused_connection(self):
        connection = SimpleNamespace(host="comfy.invalid", port=8188, ssl=False)
        refused = self.bot.aiohttp.ClientConnectorError(connection, ConnectionRefusedError(111, "refused"))
        for failed_path in ("/queue", "/free", "/prompt"):
            with self.subTest(failed_path=failed_path):
                self.service.backend = None
                self.service.comfy_contacted = False
                self.create.reset_mock()
                async def disconnected(backend, method, path, **kwargs):
                    if path == failed_path:
                        raise refused
                    return await self.request(backend, method, path, **kwargs)
                self.service.request.side_effect = disconnected
                if failed_path == "/queue":
                    await self.bot.request_completion(messages=[])
                    self.assertEqual(self.service.backend, "lmstudio")
                    self.create.assert_awaited_once()
                else:
                    with self.assertRaises(self.bot.aiohttp.ClientConnectorError):
                        await self.bot.request_completion(messages=[])
                    self.assertIsNone(self.service.backend)
                    self.create.assert_not_awaited()

    async def test_failed_startup_handoff_cannot_bypass_unload_on_later_retry(self):
        connection = SimpleNamespace(host="comfy.invalid", port=8188, ssl=False)
        refused = self.bot.aiohttp.ClientConnectorError(connection, ConnectionRefusedError(111, "refused"))
        for failed_path in ("/free", "/prompt"):
            with self.subTest(failed_path=failed_path):
                self.service.backend = None
                self.service.comfy_contacted = False
                self.service.active_job = None
                self.service.submission_uncertain = False
                async def disconnected(backend, method, path, **kwargs):
                    if path == failed_path:
                        raise refused
                    return await self.request(backend, method, path, **kwargs)
                self.service.request.side_effect = disconnected
                with self.assertRaises(self.bot.aiohttp.ClientConnectorError):
                    await self.bot.request_completion(messages=[])
                self.assertTrue(self.service.comfy_contacted)
                self.service.request.side_effect = refused
                with self.assertRaises(self.bot.aiohttp.ClientConnectorError):
                    await self.bot.request_completion(messages=[])
                self.assertIsNone(self.service.backend)
                self.create.assert_not_awaited()

    async def test_embedding_admission_failure_uses_cloud_without_local_inference(self):
        primary = SimpleNamespace(embed=Mock(side_effect=AssertionError("must not touch local GPU")))
        fallback = SimpleNamespace(embed=Mock(return_value=[[2.0]]))
        ef = self.client.custom_ef = self.bot.ResilientEmbeddingFunction(primary, fallback)
        @contextlib.asynccontextmanager
        async def unavailable():
            raise self.bot.ImageGenerationError("GPU handoff failed")
            yield
        self.service.local_request = unavailable
        self.assertEqual(await self.bot.request_embeddings(["cloud"], "retrieval.query"), [[2.0]])
        primary.embed.assert_not_called()
        fallback.embed.assert_called_once_with(["cloud"], "retrieval.query")
        self.assertTrue(ef.last_used_fallback)
        self.assertGreater(ef.dead_until, time.monotonic())
        ef.fallback_ef = None
        with self.assertRaisesRegex(self.bot.ImageGenerationError, "handoff failed"):
            await self.bot.request_embeddings(["local"], "retrieval.query")

    async def test_failed_embedding_fallback_is_not_retried_after_admission(self):
        primary = SimpleNamespace(embed=Mock(side_effect=RuntimeError("local failed")))
        fallback = SimpleNamespace(embed=Mock(side_effect=RuntimeError("cloud failed")))
        self.client.custom_ef = self.bot.ResilientEmbeddingFunction(primary, fallback)
        with self.assertRaisesRegex(RuntimeError, "cloud failed"):
            await self.bot.request_embeddings(["fact"], "retrieval.query")
        primary.embed.assert_called_once()
        fallback.embed.assert_called_once()

    async def test_cancellation_drains_late_submission_before_targeted_cancel(self):
        entered, accepted = asyncio.Event(), asyncio.Event()
        async def delayed(backend, method, path, **kwargs):
            if path == "/prompt":
                entered.set()
                await accepted.wait()
            return await self.request(backend, method, path, **kwargs)
        self.service.request.side_effect = delayed
        image = asyncio.create_task(self.service.generate("late", 64, 64))
        await asyncio.wait_for(entered.wait(), 1)
        image.cancel()
        await asyncio.sleep(0.01)
        self.assertFalse(image.done())
        self.assertFalse(any(path.endswith("/cancel") for _, _, path, _ in self.calls))
        accepted.set()
        with self.assertRaises(asyncio.CancelledError):
            await image
        self.assertFalse(self.running)
        self.assertIsNone(self.service.active_job)
        self.assertTrue(any(path.startswith("/api/jobs/") and path.endswith("/cancel") for _, _, path, _ in self.calls))

    async def test_lost_submission_ack_does_not_treat_empty_queue_as_safe(self):
        async def lost(backend, method, path, **kwargs):
            if path == "/prompt":
                raise TimeoutError("unknown submission result")
            return await self.request(backend, method, path, **kwargs)
        self.service.request.side_effect = lost
        with self.assertRaises(TimeoutError):
            await self.service.generate("unknown", 64, 64)
        self.assertTrue(self.service.submission_uncertain)
        self.assertIsNotNone(self.service.active_job)
        self.assertTrue(any(path.startswith("/history/") for _, _, path, _ in self.calls))
        with self.assertRaises(self.bot.ImageGenerationError):
            await self.bot.request_completion(messages=[])
        self.create.assert_not_awaited()

    async def test_timeout_cancels_only_our_job(self):
        self.client.config.image_timeout = 0.025
        self.hold_image = True
        with self.assertRaises(TimeoutError):
            await self.service.generate("timeout", 64, 64)
        self.assertIsNone(self.service.active_job)
        self.assertFalse(self.running)
        self.assertFalse(any(path == "/interrupt" for _, _, path, _ in self.calls))

    async def test_unrelated_lm_or_comfy_jobs_are_not_unloaded_or_interrupted(self):
        self.loaded["unrelated"] = ["other-app"]
        with self.assertRaisesRegex(self.bot.ImageGenerationError, "Another LM Studio"):
            await self.service.generate("busy", 64, 64)
        self.assertFalse(any(path == "/models/unload" for _, _, path, _ in self.calls))
        self.loaded.pop("unrelated")
        self.running = ["another-comfy-job"]
        with self.assertRaisesRegex(self.bot.ImageGenerationError, "unfinished jobs"):
            await self.service.generate("busy", 64, 64)
        self.assertFalse(any(path.endswith("/cancel") for _, _, path, _ in self.calls))

    async def test_real_http_empty_json_binary_limits_and_redirect_rejection(self):
        self.service.request = self.bot.ImageGeneration.request.__get__(self.service)
        self.service.comfy_url = "http://public.example"
        responses = {
            "/queue": (200, {"Content-Type": "application/json"}, b'{"queue_running": [], "queue_pending": []}'),
            "/free": (200, {}, b""),
            "/redirect": (302, {"Location": "http://internal.example/secret"}, b""),
            "/view": (200, {"Content-Type": "image/png"}, b"0123456789"),
        }
        async with fixtures.BotChecks.public_http(self, responses):
            self.assertEqual(await self.service.queue(), {"queue_running": [], "queue_pending": []})
            self.assertEqual(await self.service.request("comfyui", "POST", "/free"), {})
            with self.assertRaises(self.bot.ImageGenerationError):
                await self.service.request("comfyui", "GET", "/redirect")
            with patch.object(self.bot, "IMAGEGEN_DOWNLOAD_LIMIT", 8):
                with self.assertRaisesRegex(self.bot.ImageGenerationError, "too large"):
                    await self.service.request("comfyui", "GET", "/view", binary=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
