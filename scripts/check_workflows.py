"""Offline workflow discovery checks using synthetic graphs with arbitrary IDs."""

import copy
import unittest

import check_regressions as fixtures
from comfy_workflow import ComfyWorkflow, WorkflowError


class WorkflowChecks(unittest.TestCase):
    def graph(self):
        return copy.deepcopy(fixtures.WORKFLOW_FIXTURE)

    def test_connected_positive_is_found_without_ids_or_titles(self):
        source = self.graph()
        mapping = {key: f"node-{index}" for index, key in enumerate(source)}
        graph = {}
        for key, node in source.items():
            node["_meta"] = {"title": "Negative" if key == "6" else "Positive"}
            for name, value in node["inputs"].items():
                if isinstance(value, list) and len(value) == 2 and value[0] in mapping:
                    node["inputs"][name] = [mapping[value[0]], value[1]]
            graph[mapping[key]] = node
        before = copy.deepcopy(graph)
        workflow = ComfyWorkflow(graph)
        result = workflow.prepare("  literal @everyone\n{prompt}  ", 1088, 1920)
        self.assertEqual(result[mapping["6"]]["inputs"]["text"], "  literal @everyone\n{prompt}  ")
        self.assertEqual(result[mapping["232"]]["inputs"], {"width": 1088, "height": 1920, "batch_size": 1})
        self.assertEqual(workflow.output_id, mapping["213"])
        self.assertEqual(workflow.model_name(), "synthetic-model")
        self.assertEqual(graph, before)
        for key in ("7", "265", "316", "323", "324"):
            self.assertEqual(result[mapping[key]], before[mapping[key]])

    def test_shared_upstream_string_does_not_rewrite_negative_text(self):
        graph = self.graph()
        graph["7"]["inputs"]["text"] = ["48", 0]
        result = ComfyWorkflow(graph).prepare("new positive", 1024, 1024)
        self.assertEqual(result["6"]["inputs"]["text"], "new positive")
        self.assertEqual(result["7"], graph["7"])
        self.assertEqual(result["48"], graph["48"])

    def test_two_sampler_stages_can_share_one_prompt_and_latent(self):
        graph = self.graph()
        graph["second"] = copy.deepcopy(graph["265"])
        graph["second"]["inputs"]["latent_image"] = ["265", 0]
        graph["second"]["inputs"]["steps"] = 4
        graph["323"]["inputs"]["samples"] = ["second", 0]
        result = ComfyWorkflow(graph).prepare("new", 1024, 1024)
        self.assertEqual(result["6"]["inputs"]["text"], "new")
        self.assertEqual(result["265"], graph["265"])
        self.assertEqual(result["second"], graph["second"])

    def test_unsupported_second_guider_is_not_hidden_by_supported_first_stage(self):
        graph = self.graph()
        graph["other_positive"] = copy.deepcopy(graph["6"])
        graph["other_condition"] = copy.deepcopy(graph["7"])
        graph["guider"] = {"class_type": "DualCFGGuider", "inputs": {
            "model": ["316", 0], "cond1": ["other_positive", 0],
            "cond2": ["other_condition", 0], "negative": ["6", 0]}}
        graph["second"] = {"class_type": "SamplerCustomAdvanced", "inputs": {
            "guider": ["guider", 0], "latent_image": ["265", 0]}}
        graph["323"]["inputs"]["samples"] = ["second", 0]
        with self.assertRaisesRegex(WorkflowError, "Unsupported guider"):
            ComfyWorkflow(graph).prepare("new prompt", 1024, 1024)

    def test_model_name_follows_model_transforms_and_ignores_clip_checkpoint(self):
        graph = self.graph()
        graph["317"] = {"class_type": "CheckpointLoaderSimple", "inputs": {
            "ckpt_name": "clip-only.safetensors"}}
        graph["6"]["inputs"]["clip"] = ["317", 1]
        graph["7"]["inputs"]["clip"] = ["317", 1]
        graph["lora"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["316", 0], "lora_name": "adapter.safetensors", "strength_model": 0.75}}
        graph["265"]["inputs"]["model"] = ["lora", 0]
        workflow = ComfyWorkflow(graph)
        result = workflow.prepare("new", 1024, 1024)
        self.assertEqual(workflow.model_name(), "synthetic-model")
        self.assertEqual(result["lora"], graph["lora"])
        graph["316"] = {"class_type": "UnsupportedLoader", "inputs": {"model_file": "unknown"}}
        workflow = ComfyWorkflow(graph)
        self.assertIsNone(workflow.model_name())
        with self.assertRaisesRegex(WorkflowError, "image model"):
            workflow.prepare("new", 1024, 1024)

    def test_each_sampling_stage_must_have_a_discoverable_model(self):
        graph = self.graph()
        graph["second"] = copy.deepcopy(graph["265"])
        graph["second"]["inputs"].update(latent_image=["265", 0], model=["unknown_model", 0])
        graph["unknown_model"] = {"class_type": "UnsupportedLoader", "inputs": {"model_file": "unknown"}}
        graph["323"]["inputs"]["samples"] = ["second", 0]
        with self.assertRaisesRegex(WorkflowError, "image model"):
            ComfyWorkflow(graph).prepare("new", 1024, 1024)

    def test_inline_prompt_and_checkpoint_loader(self):
        graph = self.graph()
        graph["6"]["inputs"]["text"] = "inline positive"
        graph["316"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "checkpoint.safetensors"}}
        graph["6"]["inputs"]["clip"] = ["316", 1]
        graph["7"]["inputs"]["clip"] = ["316", 1]
        graph["323"]["inputs"]["vae"] = ["316", 2]
        workflow = ComfyWorkflow(graph)
        result = workflow.prepare("literal positive", 1024, 1024)
        self.assertEqual(result["6"]["inputs"]["text"], "literal positive")
        self.assertEqual(workflow.model_name(), "checkpoint")

    def test_flux_basic_guider_and_multi_field_encoder(self):
        graph = self.graph()
        graph["6"] = {"class_type": "CLIPTextEncodeFlux", "inputs": {
            "clip": ["317", 0], "clip_l": "original clip", "t5xxl": "original t5", "guidance": 3.5}}
        graph["232"]["class_type"] = "EmptyFlux2LatentImage"
        graph["guider"] = {"class_type": "BasicGuider", "inputs": {
            "model": ["316", 0], "conditioning": ["6", 0]}}
        graph["265"] = {"class_type": "SamplerCustomAdvanced", "inputs": {
            "guider": ["guider", 0], "latent_image": ["232", 0]}}
        result = ComfyWorkflow(graph).prepare("new prompt", 1536, 1024)
        self.assertEqual(result["6"]["inputs"], {
            "clip": ["317", 0], "clip_l": "new prompt", "t5xxl": "new prompt", "guidance": 3.5})
        self.assertEqual(result["232"]["inputs"]["width"], 1536)

    def test_cfg_guider_with_linked_dimensions_and_model_transforms(self):
        graph = self.graph()
        graph["size"] = {"class_type": "FluxResolutionNode", "inputs": {"megapixel": "1.0"}}
        graph["232"]["class_type"] = "EmptyFlux2LatentImage"
        graph["232"]["inputs"].update(width=["size", 0], height=["size", 1])
        graph["model_patch"] = {"class_type": "ModelSamplingAuraFlow", "inputs": {
            "model": ["316", 0], "shift": 1.0}}
        graph["guider"] = {"class_type": "CFGGuider", "inputs": {
            "model": ["model_patch", 0], "positive": ["6", 0], "negative": ["7", 0], "cfg": 1.0}}
        graph["scheduler"] = {"class_type": "Flux2Scheduler", "inputs": {
            "steps": 4, "width": ["size", 0], "height": ["size", 1]}}
        graph["265"] = {"class_type": "SamplerCustomAdvanced", "inputs": {
            "guider": ["guider", 0], "latent_image": ["232", 0], "sigmas": ["scheduler", 0]}}
        graph["preview"] = {"class_type": "PreviewImage", "inputs": {"images": ["size", 3]}}
        workflow = ComfyWorkflow(graph)
        result = workflow.prepare("new", 1536, 1024)
        self.assertEqual(result["232"]["inputs"], {"width": 1536, "height": 1024, "batch_size": 1})
        self.assertEqual(result["scheduler"]["inputs"], {"steps": 4, "width": 1536, "height": 1024})
        self.assertEqual(result["6"]["inputs"]["text"], "new")
        for key in ("7", "model_patch", "guider", "265"):
            self.assertEqual(result[key], graph[key])
        self.assertNotIn("preview", result)
        self.assertEqual(workflow.model_name(), "synthetic-model")

    def test_sdxl_updates_conditioning_dimensions_and_keeps_crops(self):
        graph = self.graph()
        graph["6"] = {"class_type": "CLIPTextEncodeSDXL", "inputs": {
            "clip": ["317", 0], "text_g": "old g", "text_l": "old l", "width": 512,
            "height": 512, "target_width": 512, "target_height": 512, "crop_w": 16, "crop_h": 32}}
        result = ComfyWorkflow(graph).prepare("new prompt", 1536, 1024)["6"]["inputs"]
        self.assertEqual([result[k] for k in ("width", "height", "target_width", "target_height")], [1536, 1024, 1536, 1024])
        self.assertEqual((result["text_g"], result["text_l"]), ("new prompt", "new prompt"))
        self.assertEqual((result["crop_w"], result["crop_h"]), (16, 32))

    def test_sd3_encoder_and_latent(self):
        graph = self.graph()
        graph["6"] = {"class_type": "CLIPTextEncodeSD3", "inputs": {
            "clip": ["317", 0], "clip_l": "old", "clip_g": "old", "t5xxl": "old", "empty_padding": "none"}}
        graph["232"]["class_type"] = "EmptySD3LatentImage"
        result = ComfyWorkflow(graph).prepare("new", 1024, 1536)
        self.assertEqual([result["6"]["inputs"][k] for k in ("clip_l", "clip_g", "t5xxl")], ["new"] * 3)
        self.assertEqual(result["6"]["inputs"]["empty_padding"], "none")

    def test_sdxl_negative_size_hints_follow_dimensions_without_changing_text(self):
        graph = self.graph()
        graph["7"] = {"class_type": "CLIPTextEncodeSDXLRefiner", "inputs": {
            "clip": ["317", 0], "text": "keep negative", "width": 512, "height": 512, "ascore": 2.5}}
        result = ComfyWorkflow(graph).prepare("new", 1536, 1024)
        self.assertEqual(result["7"]["inputs"], {
            "clip": ["317", 0], "text": "keep negative", "width": 1536, "height": 1024, "ascore": 2.5})

    def test_controlnet_routes_positive_and_negative_by_output_port(self):
        for kind in ("ControlNetApplyAdvanced", "ControlNetApplySD3"):
            with self.subTest(kind=kind):
                graph = self.graph()
                graph["control"] = {"class_type": kind, "inputs": {"positive": ["6", 0], "negative": ["7", 0]}}
                graph["265"]["inputs"].update(positive=["control", 0], negative=["control", 1])
                result = ComfyWorkflow(graph).prepare("new", 1024, 1024)
                self.assertEqual(result["6"]["inputs"]["text"], "new")
                self.assertEqual(result["7"], graph["7"])

    def test_shared_encoder_requires_zeroed_negative_conditioning(self):
        graph = self.graph()
        graph["265"]["inputs"]["negative"] = ["6", 0]
        with self.assertRaisesRegex(WorkflowError, "share a text encoder"):
            ComfyWorkflow(graph).prepare("new", 1024, 1024)
        graph["zero"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["6", 0]}}
        graph["265"]["inputs"]["negative"] = ["zero", 0]
        result = ComfyWorkflow(graph).prepare("new", 1024, 1024)
        self.assertEqual(result["zero"], graph["zero"])
        self.assertEqual(result["6"]["inputs"]["text"], "new")

    def test_multiple_positive_encoders_are_ambiguous(self):
        graph = self.graph()
        graph["extra"] = copy.deepcopy(graph["6"])
        graph["combine"] = {"class_type": "ConditioningCombine", "inputs": {
            "conditioning_1": ["6", 0], "conditioning_2": ["extra", 0]}}
        graph["265"]["inputs"]["positive"] = ["combine", 0]
        with self.assertRaisesRegex(WorkflowError, "one positive prompt encoder"):
            ComfyWorkflow(graph).prepare("new", 1024, 1024)

    def test_disconnected_presets_loaders_and_previews_are_not_submitted(self):
        graph = self.graph()
        graph["unused_size"] = copy.deepcopy(graph["232"])
        graph["unused_model"] = {"class_type": "UNETLoader", "inputs": {"unet_name": "unused-model.safetensors"}}
        graph["unused_preview"] = {"class_type": "PreviewImage", "inputs": {"images": ["unused_model", 0]}}
        workflow = ComfyWorkflow(graph)
        result = workflow.prepare("new", 1024, 1024)
        self.assertFalse(any(key.startswith("unused") for key in result))
        self.assertEqual(workflow.model_name(), "synthetic-model")

    def test_missing_or_ambiguous_required_nodes_report_errors(self):
        mutations = (
            (lambda g: g["213"].update(class_type="PreviewImage"), "SaveImage"),
            (lambda g: g.update(extra_output=copy.deepcopy(g["213"])), "SaveImage"),
            (lambda g: g["232"].update(class_type="UnsupportedSize"), "image-size input"),
            (lambda g: g["6"].update(class_type="UnsupportedTextEncode"), "conditioning node"),
            (lambda g: g["316"]["inputs"].clear(), "image model"),
            (lambda g: g["265"]["inputs"].pop("positive"), "positive prompt encoder"),
        )
        for mutation, expected in mutations:
            with self.subTest(expected=expected):
                graph = self.graph()
                mutation(graph)
                with self.assertRaisesRegex(WorkflowError, expected):
                    ComfyWorkflow(graph).prepare("new", 1024, 1024)

    def test_multiple_connected_latent_sources_are_ambiguous(self):
        graph = self.graph()
        graph["extra"] = copy.deepcopy(graph["232"])
        graph["merge"] = {"class_type": "LatentComposite", "inputs": {
            "samples_to": ["232", 0], "samples_from": ["extra", 0]}}
        graph["265"]["inputs"]["latent_image"] = ["merge", 0]
        with self.assertRaisesRegex(WorkflowError, "one image-size input"):
            ComfyWorkflow(graph).prepare("new", 1024, 1024)

    def test_custom_prompt_transform_is_not_silently_bypassed(self):
        graph = self.graph()
        graph["48"]["class_type"] = "CustomPromptTransform"
        with self.assertRaisesRegex(WorkflowError, "custom text-processing"):
            ComfyWorkflow(graph).prepare("new", 1024, 1024)

    def test_incomplete_api_export_identifies_missing_node_type(self):
        for missing in (None, "", " "):
            graph = self.graph()
            graph["size"] = {"inputs": {"width": 1024}}
            if missing is not None:
                graph["size"]["class_type"] = missing
            graph["232"]["inputs"]["width"] = ["size", 0]
            with self.assertRaisesRegex(WorkflowError, "node 'size' is missing class_type"):
                ComfyWorkflow(graph)

    def test_reference_image_path_explains_text_to_image_limitation(self):
        graph = self.graph()
        graph["reference"] = {"class_type": "ReferenceLatent", "inputs": {
            "conditioning": ["6", 0], "latent": ["232", 0]}}
        graph["265"]["inputs"]["positive"] = ["reference", 0]
        with self.assertRaisesRegex(WorkflowError, "Reference-image conditioning.*text-to-image"):
            ComfyWorkflow(graph).prepare("new", 1024, 1024)

    def test_invalid_api_graphs_broken_links_and_cycles_are_rejected(self):
        for graph in ({}, [], {"nodes": []}, {"node": {"class_type": "SaveImage", "inputs": {}}}):
            with self.subTest(graph=graph), self.assertRaises(WorkflowError):
                ComfyWorkflow(graph)
        graph = self.graph()
        graph["6"]["inputs"]["clip"] = ["missing", 0]
        with self.assertRaisesRegex(WorkflowError, "missing node"):
            ComfyWorkflow(graph)
        graph = self.graph()
        graph["265"]["inputs"]["latent_image"] = ["265", 0]
        with self.assertRaisesRegex(WorkflowError, "cycle"):
            ComfyWorkflow(graph)


if __name__ == "__main__":
    unittest.main(verbosity=2)
