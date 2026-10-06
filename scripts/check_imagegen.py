"""Offline ComfyUI API and GPU handoff checks; no live models or Discord login."""

import asyncio
import copy
import io
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from PIL import Image

import check_regressions as fixtures
import check_features as feature_fixtures


class ImageGenerationChecks(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await fixtures.BotChecks.asyncSetUp(self)
        self.client.config.comfy_url = "http://comfy.invalid:8188"
        self.client.config.model = "chat-alias"
        self.service = self.client.imagegen = self.bot.ImageGeneration(self.client.config)
        self.service.backend = "lmstudio"
        self.patches.enter_context(patch.object(self.bot, "IMAGEGEN_POLL_INTERVAL", 0.001))
        self.service.request = AsyncMock(side_effect=self.request)
        self.calls, self.jobs, self.running = [], {}, []
        self.loaded = {"native-chat": ["chat-alias"]}
        self.hold_image = False
        self.started = asyncio.Event()
        self.finished_free = False
        self.barriers = 0

    asyncTearDown = fixtures.BotChecks.asyncTearDown
    completion = staticmethod(fixtures.BotChecks.completion)
    chat = feature_fixtures.FeatureChecks.chat

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
        self.assertEqual(graph["232"]["inputs"], {"width": 1088, "height": 1920, "batch_size": 1})
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
        for prompt, requested, expected in (("first", (64, 80), (912, 1152)),
                                            ("second\nexactly", (3840, 2160), (2048, 1152))):
            raw, _ = await self.service.generate(prompt, *requested)
            with Image.open(io.BytesIO(raw)) as picture:
                self.assertEqual(picture.size, expected)
        submitted = [graph for graph in self.jobs.values() if "213" in graph]
        self.assertEqual([graph["48"]["inputs"]["value"] for graph in submitted], ["first", "second\nexactly"])
        self.assertEqual(sum(path == "/models/unload" for _, _, path, _ in self.calls), 1)
        self.assertFalse(any(path == "/free" for _, _, path, _ in self.calls))
        self.assertEqual(self.service.backend, "comfyui")
        self.create.assert_not_awaited()

    async def test_generation_time_excludes_queue_handoff_and_download(self):
        clock = SimpleNamespace(now=0.0)

        async def timed_request(backend, method, path, **kwargs):
            # Only submission and workflow completion belong in generation time.
            if path == "/prompt":
                clock.now += 3.0
            elif path.startswith("/history/"):
                clock.now += 4.0
            else:
                clock.now += 100.0
            return await self.request(backend, method, path, **kwargs)

        self.service.request.side_effect = timed_request
        with patch.object(self.bot, "time", SimpleNamespace(monotonic=lambda: clock.now)):
            async with self.service.entry:
                task = asyncio.create_task(self.service.generate("queued image", 1024, 1024))
                await asyncio.sleep(0)
                clock.now += 1000.0
            raw, duration = await asyncio.wait_for(task, 1)
        self.assertEqual(duration, 7.0)
        self.assertGreater(clock.now, 1000.0)
        with Image.open(io.BytesIO(raw)) as picture:
            self.assertEqual(picture.size, (1024, 1024))

    async def test_round_trip_preserves_custom_chat_alias(self):
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
        await self.service.generate("next", 64, 64)
        unloaded = [body["instance_id"] for _, _, path, body in self.calls if path == "/models/unload"]
        self.assertEqual(unloaded, ["chat-alias", "native-chat"])

    async def test_concurrent_local_calls_decline_images_and_keep_accepting_chat(self):
        release = asyncio.Event()
        two_running = asyncio.Event()
        three_running = asyncio.Event()
        async def infer(**kwargs):
            if self.create.await_count >= 2:
                two_running.set()
            if self.create.await_count == 3:
                three_running.set()
            await release.wait()
            return self.completion("answer")
        self.create.side_effect = infer
        first = asyncio.create_task(self.bot.request_completion(messages=[]))
        second = asyncio.create_task(self.bot.request_completion(messages=[]))
        tasks = [first, second]
        try:
            await asyncio.wait_for(two_running.wait(), 1)
            with self.assertRaisesRegex(self.bot.ModelBusyError, "Chat is active right now"):
                await asyncio.wait_for(self.service.generate("declined", 64, 64), 1)
            tasks.append(asyncio.create_task(self.bot.request_completion(messages=[])))
            await asyncio.wait_for(three_running.wait(), 1)
            self.assertEqual(self.create.await_count, 3)
            self.assertFalse(self.started.is_set())
            self.assertFalse(self.calls)
        finally:
            release.set()
            await asyncio.gather(*tasks)
        await self.service.generate("accepted after chat", 64, 64)
        self.assertEqual(self.create.await_count, 3)
        self.assertEqual(self.service.work, {"lmstudio": 0, "comfyui": 0})

    async def test_images_queue_but_decline_chat_until_all_images_finish(self):
        self.hold_image = True
        images = [asyncio.create_task(self.service.generate(str(i), 64, 64)) for i in range(3)]
        try:
            await asyncio.wait_for(self.started.wait(), 1)
            await asyncio.sleep(0)
            self.assertEqual(len(self.running), 1)
            with self.assertRaisesRegex(self.bot.ModelBusyError, "Image generation is active right now"):
                await asyncio.wait_for(self.bot.request_completion(messages=[]), 1)
            self.create.assert_not_awaited()
            self.assertFalse(any(path == "/free" for _, _, path, _ in self.calls))
        finally:
            self.hold_image = False
            await asyncio.gather(*images)
        self.assertEqual([graph["48"]["inputs"]["value"] for graph in self.jobs.values()], ["0", "1", "2"])
        self.assertEqual(sum(path == "/models/unload" for _, _, path, _ in self.calls), 1)
        self.assertEqual(self.service.work, {"lmstudio": 0, "comfyui": 0})
        await self.bot.request_completion(messages=[])
        self.assertTrue(self.finished_free)

    async def test_model_switch_already_reserves_the_selected_request_type(self):
        original_switch = self.service.switch
        for selected in ("lmstudio", "comfyui"):
            with self.subTest(selected=selected):
                entered, release = asyncio.Event(), asyncio.Event()
                async def switch(backend):
                    entered.set()
                    await release.wait()
                    await original_switch(backend)
                with patch.object(self.service, "switch", side_effect=switch):
                    request = (self.bot.request_completion(messages=[]) if selected == "lmstudio"
                               else self.service.generate("switching", 64, 64))
                    task = asyncio.create_task(request)
                    try:
                        await asyncio.wait_for(entered.wait(), 1)
                        rejected = (self.service.generate("declined", 64, 64) if selected == "lmstudio"
                                    else self.bot.request_completion(messages=[]))
                        with self.assertRaises(self.bot.ModelBusyError):
                            await asyncio.wait_for(rejected, 1)
                    finally:
                        release.set()
                        await task
                self.assertEqual(self.service.work, {"lmstudio": 0, "comfyui": 0})

    async def test_cancelled_chat_keeps_gpu_until_request_finishes_even_if_it_fails(self):
        for failed in (False, True):
            with self.subTest(failed=failed):
                started, release = asyncio.Event(), asyncio.Event()
                self.started.clear()
                self.create.reset_mock()
                async def infer(**kwargs):
                    started.set()
                    await release.wait()
                    if failed:
                        raise RuntimeError("Synthetic inference failure")
                    return self.completion("Answer")
                self.create.side_effect = infer
                chat = asyncio.create_task(self.bot.request_completion(messages=[]))
                try:
                    await asyncio.wait_for(started.wait(), 1)
                    chat.cancel()
                    await asyncio.sleep(0)
                    chat.cancel()
                    with self.assertRaisesRegex(self.bot.ModelBusyError, "Chat is active right now"):
                        await asyncio.wait_for(self.service.generate("declined", 64, 64), 1)
                    self.assertFalse(self.started.is_set())
                    self.assertFalse(chat.done())
                finally:
                    release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(chat, 2)
                self.create.assert_awaited_once()
                await self.service.generate("after cancelled chat", 64, 64)
                self.assertEqual(self.service.work, {"lmstudio": 0, "comfyui": 0})

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

    async def test_chat_admission_failure_blocks_inference_and_releases_gpu(self):
        with patch.object(self.service, "switch", new=AsyncMock(
            side_effect=self.bot.ImageGenerationError("GPU handoff failed"),
        )) as switch:
            with self.assertRaisesRegex(self.bot.ImageGenerationError, "handoff failed"):
                await self.bot.request_completion(messages=[])
            switch.assert_awaited_once_with("lmstudio")
            self.create.assert_not_awaited()
        self.assertEqual(self.service.work, {"lmstudio": 0, "comfyui": 0})
        self.assertFalse(self.service.entry.locked())
        response = await self.bot.request_completion(messages=[])
        self.assertIs(response, self.create.return_value)
        self.create.assert_awaited_once_with(model="chat-alias", messages=[])

    async def test_failed_local_chat_is_not_retried_and_releases_gpu(self):
        self.create.side_effect = RuntimeError("local failed")
        with self.assertRaisesRegex(RuntimeError, "local failed"):
            await self.bot.request_completion(messages=[])
        self.create.assert_awaited_once_with(model="chat-alias", messages=[])
        self.assertEqual(self.service.work, {"lmstudio": 0, "comfyui": 0})
        self.assertFalse(self.service.entry.locked())
        self.create.side_effect = None
        response = await self.bot.request_completion(messages=[])
        self.assertIs(response, self.create.return_value)
        self.assertEqual(self.create.await_count, 2)

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

    async def test_connection_failure_before_submission_allows_next_image_or_chat(self):
        connection = SimpleNamespace(host="comfy.invalid", port=8188, ssl=False)
        failures = (
            self.bot.aiohttp.ClientConnectorError(connection, ConnectionRefusedError(111, "refused")),
            self.bot.aiohttp.ConnectionTimeoutError("connection establishment timed out"),
        )
        for failure in failures:
            for next_backend in ("comfyui", "lmstudio"):
                with self.subTest(failure=type(failure).__name__, next_backend=next_backend):
                    self.service = self.client.imagegen = self.bot.ImageGeneration(self.client.config)
                    self.service.backend = "lmstudio"
                    self.loaded = {"native-chat": ["chat-alias"]}
                    self.create.reset_mock()
                    attempted_ids = []

                    async def disconnected(backend, method, path, **kwargs):
                        if path == "/prompt":
                            attempted_ids.append(kwargs["body"]["prompt_id"])
                            raise failure
                        return await self.request(backend, method, path, **kwargs)

                    self.service.request = AsyncMock(side_effect=disconnected)
                    with self.assertRaises(type(failure)):
                        await self.service.generate("not submitted", 64, 64)
                    self.assertEqual(len(attempted_ids), 1)
                    self.assertNotIn(attempted_ids[0], self.jobs)
                    self.assertIsNone(self.service.active_job)
                    self.assertFalse(self.service.submission_uncertain)
                    self.assertEqual(self.service.work, {"lmstudio": 0, "comfyui": 0})

                    # Connectivity recovers without restarting the bot. Either
                    # request type must be able to use the local GPU next.
                    self.service.request.side_effect = self.request
                    if next_backend == "comfyui":
                        data, _ = await self.service.generate("retry image", 64, 64)
                        with Image.open(io.BytesIO(data)) as picture:
                            self.assertEqual(picture.size, (1024, 1024))
                        self.create.assert_not_awaited()
                    else:
                        response = await self.bot.request_completion(messages=[])
                        self.assertIs(response, self.create.return_value)
                        self.create.assert_awaited_once()
                        self.assertTrue(self.finished_free)
                    self.assertEqual(self.service.backend, next_backend)
                    self.assertIsNone(self.service.active_job)
                    self.assertFalse(self.service.submission_uncertain)
                    self.assertEqual(self.service.work, {"lmstudio": 0, "comfyui": 0})

    async def test_lost_submission_ack_does_not_treat_empty_queue_as_safe(self):
        failures = (
            TimeoutError("unknown submission result"),
            self.bot.aiohttp.ClientOSError(104, "connection reset after submission"),
            self.bot.aiohttp.ServerDisconnectedError("disconnected before acknowledgement"),
        )
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                self.service = self.client.imagegen = self.bot.ImageGeneration(self.client.config)
                self.service.backend = "lmstudio"

                async def lost(backend, method, path, **kwargs):
                    if path == "/prompt":
                        raise failure
                    return await self.request(backend, method, path, **kwargs)

                self.service.request = AsyncMock(side_effect=lost)
                with self.assertRaises(type(failure)):
                    await self.service.generate("unknown", 64, 64)
                self.assertTrue(self.service.submission_uncertain)
                self.assertIsNotNone(self.service.active_job)
                job_id = self.service.active_job
                self.assertTrue(any(path == f"/history/{job_id}" for _, _, path, _ in self.calls))

                # A restored connection and empty queue do not establish whether
                # the server is still validating an already-sent submission.
                self.service.request.side_effect = self.request
                with self.assertRaisesRegex(self.bot.ImageGenerationError, "not confirmed"):
                    await self.service.generate("unsafe retry", 64, 64)
                with self.assertRaisesRegex(self.bot.ImageGenerationError, "not confirmed"):
                    await self.bot.request_completion(messages=[])
                self.assertEqual(self.service.active_job, job_id)
                self.assertTrue(self.service.submission_uncertain)
                self.assertFalse(self.jobs)
                self.create.assert_not_awaited()

    async def test_timeout_cancels_only_our_job(self):
        self.client.config.image_timeout = 0.025
        self.hold_image = True
        with self.assertRaises(TimeoutError):
            await self.service.generate("timeout", 64, 64)
        self.assertIsNone(self.service.active_job)
        self.assertFalse(self.running)
        self.assertFalse(any(path == "/interrupt" for _, _, path, _ in self.calls))

    async def test_failed_image_cancellation_declines_chat_until_remote_job_finishes(self):
        self.client.config.image_timeout = 0.025
        self.hold_image = True
        cancellations = []

        async def failed_cancel(backend, method, path, **kwargs):
            if path.endswith("/cancel"):
                cancellations.append(path)
                raise TimeoutError("Cancellation could not be confirmed")
            return await self.request(backend, method, path, **kwargs)

        self.service.request.side_effect = failed_cancel
        with self.assertRaises(TimeoutError):
            await self.service.generate("timed out", 64, 64)
        self.assertEqual(len(self.running), 1)
        self.assertIsNotNone(self.service.active_job)
        for uncertain in (False, True):
            self.service.submission_uncertain = uncertain
            with self.assertRaisesRegex(self.bot.ModelBusyError, "unfinished jobs"):
                await self.bot.request_completion(messages=[])
        self.assertEqual(len(cancellations), 1)
        message = self.chat(server=2)
        await self.bot.on_message(message)
        self.assertIn("unfinished jobs", message.reply.call_args.args[0])
        with patch.object(self.bot, "build_ai_context", new=AsyncMock(return_value=[])):
            message = self.chat(server=3)
            await self.bot.on_message(message)
            self.assertIn("unfinished jobs", message.reply.call_args.args[0])
        self.assertEqual(await fixtures.BotChecks.count(self, "chat_history"), 0)
        self.create.assert_not_awaited()
        # A later retry checks the remote queue again, instead of remaining locked.
        self.hold_image = False
        self.running.clear()
        self.service.request.side_effect = self.request
        response = await self.bot.request_completion(messages=[])
        self.assertIs(response, self.create.return_value)
        self.create.assert_awaited_once()
        self.assertTrue(self.finished_free)
        self.assertEqual(self.service.work, {"lmstudio": 0, "comfyui": 0})

    async def test_unrelated_lm_or_comfy_jobs_are_not_unloaded_or_interrupted(self):
        for key, alias in (("unrelated", "other-app"), ("native-embedding", "embedding-alias")):
            with self.subTest(model=key):
                self.loaded[key] = [alias]
                with self.assertRaisesRegex(self.bot.ImageGenerationError, "Another LM Studio"):
                    await self.service.generate("busy", 64, 64)
                self.assertFalse(any(path == "/models/unload" for _, _, path, _ in self.calls))
                self.loaded.pop(key)
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
