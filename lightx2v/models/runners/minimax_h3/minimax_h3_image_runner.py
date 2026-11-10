"""MiniMax-H3-Image runner: text-to-image (``t2i``) and reference-image editing (``i2i``).

One request packs ``[text | reference images | target image]`` into a single
sequence for the H3 DiT without audio, samples the fixed 6-step PDD grid
without guidance, and decodes with the image VAE. The request flow follows the
Diffusers ``MiniMaxH3ImageBlocks``:

1. resolve the generation canvas (requests below ``min_generation_area`` pixels
   are generated larger and resized back with LANCZOS);
2. encode the prompt (edit: ``Picture i: `` label + vision block per reference);
3. edit only: encode every reference with the VAE at its own size;
4. draw the target noise, then the reference noise, off one CPU generator;
5. denoise, decode, quantize to 8 bits and resize to the requested size.

``h3_image_rng_order`` selects whether the VAE posterior samples of step 3 are
drawn before (``diffusers``, default) or after (``sglang``) the noise of step 4.
"""

import gc
import os
from contextlib import suppress

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image, ImageOps
from loguru import logger

from lightx2v.models.input_encoders.hf.minimax_h3.qwen3vl_image import MiniMaxH3ImageQwen3VLTextEncoder
from lightx2v.models.networks.minimax_h3.image.model import MiniMaxH3ImageModel
from lightx2v.models.networks.minimax_h3.image.packing import (
    CANVAS_MULTIPLE,
    DEFAULT_MIN_GENERATION_AREA,
    MAX_REFERENCE_IMAGES,
    REFERENCE_VAE_SHORT_EDGE,
    bound_short_edge,
    get_image_canvas_size,
    reference_latent_grid,
    unpatchify_image_rows,
)
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.runners.request_fields import COMMON_REQUEST_FIELDS
from lightx2v.models.schedulers.minimax_h3.image_scheduler import MiniMaxH3ImageScheduler, draw_image_noise
from lightx2v.models.video_encoders.hf.minimax_h3.image_vae import MiniMaxH3ImageVAE
from lightx2v.utils.envs import DTYPE_MAP
from lightx2v.utils.profiler import ProfilingContext4DebugL1, ProfilingContext4DebugL2
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)
RNG_ORDERS = ("diffusers", "sglang")


def load_rgb_image(value):
    image = value if isinstance(value, Image.Image) else Image.open(value)
    return ImageOps.exif_transpose(image).convert("RGB")


def split_image_paths(value):
    """Normalize ``image_path`` (comma-separated string, list, or images) into a list."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return []
    if isinstance(value, str):
        return [path.strip() for path in value.split(",") if path.strip()]
    if isinstance(value, (list, tuple)):
        return [item.strip() if isinstance(item, str) else item for item in value if item is not None and not (isinstance(item, str) and not item.strip())]
    return [value]


def resize_vae_reference(image, short_edge_bounds=REFERENCE_VAE_SHORT_EDGE):
    """The VAE copy of an edit reference: original pixels unless the short edge is out of bounds."""
    size = bound_short_edge(*image.size, *short_edge_bounds)
    if size == image.size:
        return image
    return image.resize(size, Image.Resampling.LANCZOS)


def quantize_image(image):
    """``[1, 3, H, W]`` in ``[0, 1]`` to a PIL image, truncating to 8 bits as released."""
    array = (image[0] * 255).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(np.ascontiguousarray(array))


@RUNNER_REGISTER("minimax_h3_image")
class MiniMaxH3ImageRunner(DefaultRunner):
    supported_request_fields_by_task = {
        "t2i": COMMON_REQUEST_FIELDS | {"prompt", "size"},
        "i2i": COMMON_REQUEST_FIELDS | {"prompt", "size", "image_path"},
    }

    def __init__(self, config):
        if config.get("lazy_load", False) or config.get("unload_modules", False):
            raise NotImplementedError("MiniMax-H3-Image does not support lazy_load or unload_modules; use model or block CPU offload.")
        if config.get("enable_cfg", False) or config.get("cfg_parallel", False):
            raise ValueError("MiniMax-H3-Image is guidance-distilled and has no unconditional branch")
        self.rng_order = config.get("h3_image_rng_order", "diffusers")
        if self.rng_order not in RNG_ORDERS:
            raise ValueError(f"h3_image_rng_order must be one of {RNG_ORDERS}, got {self.rng_order!r}")
        super().__init__(config)
        self.generator = None
        self.reference_latents = []
        self.target_noise = None
        self.reference_noise = []

    def get_supported_tasks(self):
        return ("t2i", "i2i")

    def init_scheduler(self):
        self.scheduler = MiniMaxH3ImageScheduler(self.config)

    def init_modules(self):
        super().init_modules()
        self.run_input_encoder = self._run_input_encoder_local_image

    @ProfilingContext4DebugL2("Load models")
    def load_model(self):
        self.model = self.load_transformer()
        self.text_encoders = self.load_text_encoder()
        self.vae = self.load_vae()

    def load_transformer(self):
        return MiniMaxH3ImageModel(self.config["model_path"], self.config, self.init_device)

    def load_text_encoder(self):
        return [MiniMaxH3ImageQwen3VLTextEncoder(self.config)]

    def load_vae(self):
        return MiniMaxH3ImageVAE.from_pretrained(
            self.config["model_path"],
            device=AI_DEVICE,
            cpu_offload=self.config.get("vae_cpu_offload", self.config.get("cpu_offload", False)),
            decode_dtype=DTYPE_MAP[self.config.get("vae_decode_dtype", "bf16")],
            attn_type=self.config.get("vae_attn_type", "torch_sdpa"),
            tile_size=self.config.get("vae_tile_size"),
            tile_overlap=self.config.get("vae_tile_overlap"),
        )

    def maybe_empty_cache(self, *, force=False, collect_garbage=False):
        if AI_DEVICE == "cpu":
            if force or collect_garbage:
                gc.collect()
            return False
        return super().maybe_empty_cache(force=force, collect_garbage=collect_garbage)

    # ---------------------------------------------------------------- request
    def _resolve_geometry(self):
        size = self.input_info.size or self.config.get("size") or [2048, 2048]
        if len(size) != 2:
            raise ValueError(f"MiniMax-H3-Image size must be [height, width], got {size}")
        self.output_height, self.output_width = (int(value) for value in size)
        min_area = int(self.config.get("min_generation_area", DEFAULT_MIN_GENERATION_AREA))
        self.canvas_height, self.canvas_width = get_image_canvas_size(self.output_height, self.output_width, min_area, CANVAS_MULTIPLE)
        self.input_info.size = [self.output_height, self.output_width]
        logger.info(f"MiniMax-H3-Image output {self.output_width}x{self.output_height}, canvas {self.canvas_width}x{self.canvas_height}")

    def _load_references(self):
        if self.input_info.task != "i2i":
            return []
        values = split_image_paths(getattr(self.input_info, "image_path", None))
        if not values:
            raise ValueError("MiniMax-H3-Image editing (task i2i) requires at least one --image_path")
        if len(values) > MAX_REFERENCE_IMAGES:
            raise ValueError(f"MiniMax-H3-Image accepts at most {MAX_REFERENCE_IMAGES} reference images, got {len(values)}")
        images = [load_rgb_image(value) for value in values]
        for image in images:
            if min(image.size) < CANVAS_MULTIPLE:
                raise ValueError(f"A reference image needs at least {CANVAS_MULTIPLE} pixels on each side, got {image.size}")
        return images

    def _new_generator(self):
        device = self.config.get("generator_device", "cpu")
        return torch.Generator(device=device).manual_seed(int(self.input_info.seed))

    def _encode_references(self, images):
        latents = []
        patch_size = tuple(self.config.get("patch_size", (1, 2, 2)))
        for image in images:
            pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)[None].float().div_(255.0)
            latent = self.vae.encode(pixels, self.generator, patch_size=patch_size)
            if tuple(latent.shape[-2:]) != reference_latent_grid(*image.size, patch_size):
                raise RuntimeError(f"MiniMax-H3-Image reference latent {tuple(latent.shape)} does not match its grid {reference_latent_grid(*image.size, patch_size)}")
            latents.append(latent)
        return latents

    def _draw_noise(self, reference_grids):
        vae_scale = self.vae.spatial_compression_ratio
        self.target_noise, self.reference_noise = draw_image_noise(
            self.generator,
            int(self.config.get("in_channels", 24)),
            self.canvas_height // vae_scale,
            self.canvas_width // vae_scale,
            reference_grids,
        )

    @ProfilingContext4DebugL2("Run Input Encoder")
    def _run_input_encoder_local_image(self):
        self._resolve_geometry()
        references = self._load_references()
        with ProfilingContext4DebugL1("Run Text Encoder"):
            text_encoder_output = self.text_encoders[0].infer(self.input_info.prompt, image_list=references or None)
        tags = text_encoder_output["text_token_tags"]
        if self.input_info.task == "t2i" and not bool((tags == 1).all()):
            raise ValueError("MiniMax-H3-Image t2i conditioner returned non-text rows")

        patch_size = tuple(self.config.get("patch_size", (1, 2, 2)))
        vae_bounds = tuple(self.config.get("reference_vae_short_edge", REFERENCE_VAE_SHORT_EDGE))
        vae_images = [resize_vae_reference(image, vae_bounds) for image in references]
        reference_grids = [reference_latent_grid(*image.size, patch_size) for image in vae_images]
        self.generator = self._new_generator()
        with ProfilingContext4DebugL1("Run VAE Encoder"):
            if self.rng_order == "diffusers":
                self.reference_latents = self._encode_references(vae_images)
                self._draw_noise(reference_grids)
            else:
                self._draw_noise(reference_grids)
                self.reference_latents = self._encode_references(vae_images)
        return {"text_encoder_output": text_encoder_output}

    @ProfilingContext4DebugL2("Prepare DiT")
    def init_run(self):
        text_encoder_output = self.inputs["text_encoder_output"]
        self.scheduler.prepare(
            text_encoder_output["text_token_tags"],
            self.target_noise,
            reference_latents=self.reference_latents,
            reference_noise=self.reference_noise,
            patch_size=tuple(self.config.get("patch_size", (1, 2, 2))),
        )
        logger.info(
            f"MiniMax-H3-Image packed layout: text={text_encoder_output['prompt_embeds'].shape[0]}, "
            f"reference={self.scheduler.num_condition_video_rows}, target={self.scheduler.target_rows().shape[0]}, total={self.scheduler.layout.sequence_length}"
        )
        if self.config.get("cpu_offload", False) and self.config.get("offload_granularity", "model") == "model":
            self.model.to_cuda()

    def run_segment(self, segment_idx=0):
        infer_steps = self.scheduler.infer_steps
        for step_index in range(infer_steps):
            with ProfilingContext4DebugL1("Run Dit every step"):
                if hasattr(self, "check_stop"):
                    self.check_stop()
                logger.info(f"==> MiniMax-H3-Image step: {step_index + 1} / {infer_steps}")
                self.scheduler.step_pre(step_index)
                self.model.infer(self.inputs)
                self.scheduler.step_post()
            if self.progress_callback:
                self.progress_callback((step_index + 1) / infer_steps * 100, 100)
        return self.scheduler.target_rows()

    def _offload_transformer(self):
        if not self.config.get("cpu_offload", False):
            return
        if self.model.block_offload:
            if not self.model.prepost_resident:
                self.model.pre_weight.to_cpu()
                self.model.post_weight.to_cpu()
        else:
            self.model.to_cpu()
        if AI_DEVICE != "cpu":
            torch_device_module.synchronize()
        self.maybe_empty_cache(force=True, collect_garbage=True)

    @ProfilingContext4DebugL1("Run VAE Decoder")
    def run_vae_decoder(self, rows):
        latents = unpatchify_image_rows(
            rows.float(),
            self.scheduler.latent_height,
            self.scheduler.latent_width,
            channels=int(self.config.get("in_channels", 24)),
            patch_size=tuple(self.config.get("patch_size", (1, 2, 2))),
        )
        image = quantize_image(self.vae.decode(latents))
        if image.size != (self.output_width, self.output_height):
            image = image.resize((self.output_width, self.output_height), Image.Resampling.LANCZOS)
        return image

    def _save_image(self, image):
        path = self.input_info.save_result_path
        if self.input_info.return_result_tensor or not path:
            return
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        image.save(path)
        logger.info(f"MiniMax-H3-Image saved to {path}")

    @ProfilingContext4DebugL1("RUN pipeline")
    def run_pipeline(self, input_info):
        self.input_info = input_info
        offloaded = not self.config.get("cpu_offload", False)
        try:
            self.inputs = self.run_input_encoder()
            self.init_run()
            try:
                rows = self.run_segment()
            finally:
                if not offloaded:
                    self._offload_transformer()
                    offloaded = True
            image = self.run_vae_decoder(rows)
            self._save_image(image)
            if input_info.return_result_tensor:
                return {"images": [image]}
            return {"images": None}
        finally:
            if not offloaded:
                with suppress(Exception):
                    self._offload_transformer()
            self.generator = None
            self.reference_latents = []
            self.target_noise = None
            self.reference_noise = []
            self.end_run()
