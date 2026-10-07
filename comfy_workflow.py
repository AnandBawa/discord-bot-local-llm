"""Discover supported ComfyUI API inputs by their connections, never by node IDs."""

import copy
from graphlib import CycleError, TopologicalSorter


class WorkflowError(ValueError):
    """A workflow problem that can be shown without exposing its prompt or settings."""


TEXT_INPUTS = {
    "CLIPTextEncode": ("text",),
    "CLIPTextEncodeSDXL": ("text_g", "text_l"),
    "CLIPTextEncodeSDXLRefiner": ("text",),
    "CLIPTextEncodeSD3": ("clip_l", "clip_g", "t5xxl"),
    "CLIPTextEncodeFlux": ("clip_l", "t5xxl"),
}
TEXT_NODE_HINT = "Expected a connected text encoder: " + ", ".join(TEXT_INPUTS) + "."
LATENT_NODES = {"EmptyLatentImage", "EmptySD3LatentImage", "EmptyFlux2LatentImage"}
CONDITION_INPUTS = {
    "conditioning", "conditioning_1", "conditioning_2",
    "conditioning_to", "conditioning_from",
}


def link(value):
    return (isinstance(value, list) and len(value) == 2
            and isinstance(value[0], str) and type(value[1]) is int and value[1] >= 0)


class ComfyWorkflow:
    def __init__(self, graph):
        if (not isinstance(graph, dict) or not graph
                or any(not isinstance(node, dict) or not isinstance(node.get("inputs"), dict)
                       for node in graph.values())):
            raise WorkflowError("Export the workflow using ComfyUI's File → Export (API), then save it as workflow.json.")
        for key, node in graph.items():
            if not isinstance(node.get("class_type"), str) or not node["class_type"].strip():
                raise WorkflowError(f"Workflow node {key!r} is missing class_type. Check that custom node in ComfyUI and export (API) again.")
        outputs = [key for key, node in graph.items() if node["class_type"] == "SaveImage"]
        if len(outputs) != 1:
            raise WorkflowError("The workflow needs exactly one SaveImage node as its final image output.")
        self.output_id = outputs[0]
        if not link(graph[self.output_id]["inputs"].get("images")):
            raise WorkflowError("Connect the final image to the workflow's SaveImage node.")
        self.graph = graph
        self.parents = {}
        pending = [self.output_id]
        while pending:
            key = pending.pop()
            if key in self.parents:
                continue
            if key not in graph:
                raise WorkflowError("The workflow has a connection to a missing node. Re-export a working workflow.")
            self.parents[key] = {value[0] for value in graph[key]["inputs"].values() if link(value)}
            pending.extend(self.parents[key])
        try:
            tuple(TopologicalSorter(self.parents).static_order())
        except CycleError as exc:
            raise WorkflowError("The workflow contains a connection cycle. Re-export a working workflow.") from exc
        # Only the selected output's dependencies may run. Unrelated previews and
        # unused presets must not start another model or influence discovery.
        self.graph = copy.deepcopy({key: node for key, node in graph.items() if key in self.parents})

    def conditioning_encoders(self, roots):
        encoders, visited = set(), set()
        pending = list(roots)
        while pending:
            reference = pending.pop()
            if not link(reference):
                raise WorkflowError("The sampler's conditioning is not connected correctly. " + TEXT_NODE_HINT)
            key, port = reference
            if (key, port) in visited:
                continue
            visited.add((key, port))
            node = self.graph[key]
            kind, inputs = node["class_type"], node["inputs"]
            if kind in TEXT_INPUTS and port == 0:
                encoders.add(key)
                continue
            if kind == "ConditioningZeroOut" and port == 0:
                continue
            if kind == "ReferenceLatent":
                raise WorkflowError("Reference-image conditioning (ReferenceLatent) is unsupported. Use a text-to-image workflow for /imagegen.")
            if kind in {"ControlNetApplyAdvanced", "ControlNetApplySD3"} and port in (0, 1):
                fields = ("positive" if port == 0 else "negative",)
            elif port == 0 and (kind.startswith("Conditioning")
                               or kind in {"FluxGuidance", "FluxDisableGuidance", "ControlNetApply"}):
                fields = tuple(name for name in inputs if name in CONDITION_INPUTS)
            else:
                fields = ()
            if not fields:
                raise WorkflowError("Cannot follow the prompt through this conditioning node. " + TEXT_NODE_HINT)
            pending.extend(inputs.get(name) for name in fields)
        return encoders

    def sampling_inputs(self):
        positive, negative, models = [], [], []
        for node in self.graph.values():
            inputs = node["inputs"]
            if "guider" in inputs:
                reference = inputs["guider"]
                if (not link(reference) or reference[1] != 0
                        or self.graph[reference[0]]["class_type"] not in {"BasicGuider", "CFGGuider"}):
                    raise WorkflowError("Unsupported guider. Use BasicGuider or CFGGuider with one positive prompt.")
            if "model" in inputs and "positive" in inputs:
                positive.append(inputs["positive"])
                if "negative" in inputs:
                    negative.append(inputs["negative"])
                models.append(inputs["model"])
            elif node["class_type"] == "BasicGuider":
                positive.append(inputs.get("conditioning"))
                models.append(inputs.get("model"))
        return positive, negative, models

    def model_name(self):
        # Follow MODEL connections only: a checkpoint used just for CLIP or VAE
        # must not stand in for a missing diffusion-model loader.
        pending = list(self.sampling_inputs()[2])
        names, visited = [], set()
        while pending:
            reference = pending.pop()
            if not link(reference) or reference[1] != 0:
                return None
            key = reference[0]
            if key in visited:
                continue
            visited.add(key)
            inputs = self.graph[key]["inputs"]
            sources = [inputs[field] for field in ("model", "model1", "model2") if field in inputs]
            if sources:
                pending.extend(sources)
                continue
            loaders = [inputs.get(field) for field in ("unet_name", "ckpt_name")]
            loaders = [name.strip().removesuffix(".safetensors")
                       for name in loaders if isinstance(name, str) and name.strip()]
            if not loaders:
                return None
            names.extend(name for name in loaders if name not in names)
        return ", ".join(names) or None

    def prepare(self, prompt, width, height):
        positive, negative, _ = self.sampling_inputs()
        encoders = self.conditioning_encoders(positive)
        if len(encoders) != 1:
            raise WorkflowError("Cannot identify one positive prompt encoder for the sampler. " + TEXT_NODE_HINT)
        negative_encoders = self.conditioning_encoders(negative)
        if encoders & negative_encoders:
            raise WorkflowError("Positive and negative conditioning share a text encoder. Use separate encoders or zeroed negative conditioning.")
        encoder = self.graph[next(iter(encoders))]
        text_fields = TEXT_INPUTS[encoder["class_type"]]
        for name in text_fields:
            value = encoder["inputs"].get(name)
            if isinstance(value, str):
                continue
            source = self.graph[value[0]] if link(value) else None
            if (source is None or value[1] != 0
                    or source["class_type"] not in {"PrimitiveString", "PrimitiveStringMultiline"}
                    or not isinstance(source["inputs"].get("value"), str)):
                raise WorkflowError("The positive prompt needs editable text or a PrimitiveString/PrimitiveStringMultiline node; custom text-processing paths are unsupported.")
        latents = [node for node in self.graph.values() if node["class_type"] in LATENT_NODES]
        if len(latents) != 1 or not {"width", "height"}.issubset(latents[0]["inputs"]):
            raise WorkflowError("Cannot identify one image-size input. Expected one EmptyLatentImage, EmptySD3LatentImage, or EmptyFlux2LatentImage node with width and height inputs.")
        if self.model_name() is None:
            raise WorkflowError("Cannot identify the image model. Expected a connected UNETLoader (unet_name), CheckpointLoaderSimple (ckpt_name), or equivalent model loader.")
        # Override only this encoder's text fields, preserving negative text even
        # when its encoder shares the original upstream string source.
        encoder["inputs"].update({name: prompt for name in text_fields})
        for key in encoders | negative_encoders:
            conditioning = self.graph[key]
            if conditioning["class_type"] in {"CLIPTextEncodeSDXL", "CLIPTextEncodeSDXLRefiner"}:
                for name, value in (("width", width), ("height", height),
                                    ("target_width", width), ("target_height", height)):
                    if name in conditioning["inputs"]:
                        conditioning["inputs"][name] = value
        for node in self.graph.values():
            if node["class_type"] == "Flux2Scheduler":
                node["inputs"].update(width=width, height=height)
        latents[0]["inputs"].update(width=width, height=height, batch_size=1)
        return self.graph
