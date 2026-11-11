# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
# Copyright 2026 The LightX2V Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native MiniMax-H3-Image VAE: the H3 causal CNN encoder and a 2D ViT decoder.

The encoder is the MiniMax-H3 video encoder run over a single frame; the
decoder is an image-only ViT with ``(h, w)`` rotary positions. Both reuse the
video VAE modules. The released ``vae/diffusion_pytorch_model.safetensors`` is
stored in the original MiniMax layout and is converted while it is loaded
(per-head interleaved fused QKV, ``[gate; up]`` SwiGLU, CompVis encoder names),
following ``AutoencoderKLMiniMaxH3Image`` of Diffusers.

Pixels are ImageNet-normalized RGB over ``[0, 1]``; latents are normalized
with the per-channel ``latents_mean``/``latents_std`` of ``vae/config.json``.
Encoding and decoding are tiled (256 px tiles, at least 64 px overlap) as in
the release, which is what makes 16 MP images fit. When the default
``torch.distributed`` group has more than one rank, decode tiles are split into
balanced contiguous blocks over its ranks, and every rank gathers all tiles and
stitches the same full image (``vae_decode_parallel: false`` in the VAE config
turns this off).
"""

from __future__ import annotations

import gc
import json
import math
import re
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from loguru import logger
from safetensors import safe_open

from lightx2v.models.video_encoders.hf.minimax_h3.video_vae import (
    MINIMAX_H3_PIXEL_MEAN,
    MINIMAX_H3_PIXEL_STD,
    MiniMaxH3VideoEncoder3d,
    MiniMaxH3VideoVAE,
    MiniMaxH3VideoViTDecoder3d,
    _component_dir,
    _empty_device_cache,
)
from lightx2v.models.video_encoders.hf.minimax_h3.weights import _assign_tensor, _component_files
from lightx2v_platform.base.global_var import AI_DEVICE

IMAGE_VAE_DROPPED_KEYS = ("decoder.mask_token",)
_ENCODER_LEVEL = re.compile(r"^encoder\.down\.(\d+)\.(.+)$")


class MiniMaxH3ImageViTDecoder2d(MiniMaxH3VideoViTDecoder3d):
    """Non-causal 2D ViT decoder: one token per latent pixel, ``(h, w)`` rotary positions."""

    def __init__(self, *, rope_dim_ratio: float = 0.5, **kwargs) -> None:
        super().__init__(patch_size_t=1, rope_dim_ratio=rope_dim_ratio, rope_num_axes=2, **kwargs)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_size, height * width, num_channels)
        hidden_states = self.proj_in(hidden_states)
        num_patches = hidden_states.shape[1]
        # The register tokens follow the projection precision before the
        # residual stream is lifted to the sensitive dtype, as in the reference.
        register_tokens = self.register_tokens.expand(batch_size, -1, -1).to(hidden_states.dtype)
        if self.sensitive_layer_dtype != self.infer_dtype:
            hidden_states = hidden_states.to(self.sensitive_layer_dtype)
            register_tokens = register_tokens.to(self.sensitive_layer_dtype)
        cls_token = torch.zeros_like(hidden_states[:, :1, :])
        hidden_states = torch.cat([hidden_states, register_tokens, cls_token], dim=1)

        grids = [2.0 * (torch.arange(0.5, size, dtype=torch.float32, device=hidden_states.device) / size) - 1.0 for size in (height, width)]
        position_ids = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).flatten(0, 1)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1, -1)
        suffix_ids = position_ids.new_zeros((batch_size, self.num_register_tokens + 1, 2))
        rotary_emb = self.rope(torch.cat([position_ids, suffix_ids], dim=1))

        for block_index, block in enumerate(self.transformer_blocks):
            hidden_states = self._run_block(block_index, block, hidden_states, rotary_emb)

        hidden_states = self.norm_out(hidden_states)
        if self.sensitive_layer_dtype != self.infer_dtype:
            hidden_states = hidden_states.to(self.infer_dtype)
        hidden_states = self.proj_out(hidden_states)[:, :num_patches, :]
        patch_size = self.patch_size
        hidden_states = hidden_states.view(batch_size, height, width, self.out_channels, patch_size, patch_size)
        hidden_states = hidden_states.permute(0, 3, 1, 4, 2, 5).contiguous()
        return hidden_states.reshape(batch_size, self.out_channels, height * patch_size, width * patch_size)


def native_image_vae_names(source_key: str) -> tuple[str, ...]:
    """LightX2V module names of one original image-VAE key (empty when dropped)."""
    if source_key in IMAGE_VAE_DROPPED_KEYS:
        return ()
    if ".attn.to_qkv." in source_key:
        prefix, suffix = source_key.split(".attn.to_qkv.")
        return tuple(f"{prefix}.attn.{name}.{suffix}" for name in ("to_q", "to_k", "to_v"))
    target = source_key
    match = _ENCODER_LEVEL.fullmatch(target)
    if match is not None:
        rest = match.group(2).replace("block.", "resnets.", 1).replace("nin_shortcut.", "conv_shortcut.", 1).replace("downsample.", "downsamplers.0.", 1)
        target = f"encoder.down_blocks.{match.group(1)}.{rest}"
    target = target.replace("decoder.x_embedder.", "decoder.proj_in.")
    target = target.replace(".attn.to_out.", ".attn.to_out.0.")
    target = target.replace(".ff.w1.", ".ff.net.0.proj.")
    target = target.replace(".ff.w2.", ".ff.net.2.")
    return (target,)


def convert_native_image_vae_tensor(source_key: str, tensor: torch.Tensor, num_heads: int, head_dim: int) -> list[tuple[str, torch.Tensor]]:
    """Convert one original image-VAE key/tensor pair into LightX2V pairs."""
    targets = native_image_vae_names(source_key)
    if not targets:
        return []
    if ".attn.to_qkv." in source_key:
        # Same per-head interleave as the DiT: [head0: q k v, head1: q k v, ...].
        grouped = tensor.reshape(num_heads, 3, head_dim, *tensor.shape[1:])
        return [(target, grouped[:, index].reshape(num_heads * head_dim, *tensor.shape[1:]).contiguous()) for index, target in enumerate(targets)]
    if ".ff.w1." in source_key:
        gate, up = tensor.chunk(2, dim=0)
        tensor = torch.cat((up, gate), dim=0).contiguous()
    return [(targets[0], tensor)]


class MiniMaxH3ImageVAE(nn.Module):
    """H3-Image VAE with tiled single-frame encode and tiled 2D decode."""

    def __init__(
        self,
        config: dict,
        *,
        device: str | torch.device | None = None,
        cpu_offload: bool = False,
        decode_dtype: torch.dtype = torch.bfloat16,
        attn_type: str = "torch_sdpa",
    ) -> None:
        super().__init__()
        if not config.get("vit_decoder_2d", True):
            raise ValueError("MiniMax-H3-Image VAE requires vit_decoder_2d=true")
        decoder_kwargs = dict(config.get("vit_decoder_kwargs", {}))
        if decoder_kwargs.get("norm_type", "rms_norm") != "rms_norm" or decoder_kwargs.get("qk_norm_type", "rms_norm") != "rms_norm":
            raise ValueError(f"MiniMax-H3-Image VAE decoder expects RMSNorm, got {decoder_kwargs}")
        if not decoder_kwargs.get("ffn_use_gated", True) or decoder_kwargs.get("ffn_activation_fn", "silu") != "silu":
            raise ValueError(f"MiniMax-H3-Image VAE decoder expects a SiLU-gated FFN, got {decoder_kwargs}")
        self.config = dict(config)
        self.execution_device = torch.device(device or AI_DEVICE)
        self.cpu_offload = cpu_offload
        self.infer_dtype = decode_dtype
        self.sensitive_layer_dtype = torch.float32

        latent_channels = int(config.get("latent_channels", 24))
        spatial_factors = tuple(int(value) for value in config.get("spatial_downsample_factors", (2, 2, 2, 2, 1, 1)))
        temporal_factors = tuple(int(value) for value in config.get("temporal_downsample_factors", (1, 2, 2, 1, 1, 1)))
        self.spatial_compression_ratio = math.prod(spatial_factors)
        self.latent_channels = latent_channels
        self.decoder_num_heads = int(decoder_kwargs.get("heads", 16))
        self.decoder_head_dim = int(decoder_kwargs.get("dim_head", 64))

        # The encoder runs entirely in FP32, as released.
        self.encoder = MiniMaxH3VideoEncoder3d(
            in_channels=int(config.get("in_channels", 3)),
            out_channels=2 * latent_channels,
            block_out_channels=tuple(config.get("block_out_channels", (128, 256, 256, 512, 512, 1024))),
            layers_per_block=int(config.get("layers_per_block", 2)),
            spatial_downsample_factors=spatial_factors,
            temporal_downsample_factors=temporal_factors,
            norm_num_groups=int(config.get("norm_num_groups", 32)),
            norm_eps=float(config.get("norm_eps", 1e-6)),
            spatial_padding_mode=config.get("spatial_padding_mode", "reflect"),
            infer_dtype=torch.float32,
            sensitive_layer_dtype=torch.float32,
        )
        self.quant_conv = nn.Conv3d(2 * latent_channels, 2 * latent_channels, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)
        self.decoder = MiniMaxH3ImageViTDecoder2d(
            in_channels=latent_channels,
            out_channels=int(config.get("out_channels", 3)),
            patch_size=self.spatial_compression_ratio,
            num_layers=int(decoder_kwargs.get("num_layers", 24)),
            num_attention_heads=self.decoder_num_heads,
            attention_head_dim=self.decoder_head_dim,
            num_register_tokens=int(config.get("decoder_num_register_tokens", 4)),
            ffn_mult=int(config.get("decoder_ffn_mult", 4)),
            rope_theta=float(decoder_kwargs.get("rope_theta", 100.0)),
            rope_dim_ratio=float(decoder_kwargs.get("rope_dim_ratio", 0.5)),
            norm_eps=float(config.get("decoder_norm_eps", 1e-5)),
            infer_dtype=self.infer_dtype,
            sensitive_layer_dtype=self.sensitive_layer_dtype,
            attn_type=attn_type,
        )

        self.use_encoder_tiling = bool(config.get("vae_encoder_tiling", 1))
        self.use_decoder_tiling = bool(config.get("vae_decoder_tiling", 1))
        # Every rank of a multi-rank image run decodes the same latents, so the
        # tiles are split over the default group whenever it has several ranks.
        self.decode_parallel = bool(config.get("vae_decode_parallel", True))
        self.tile_sample_size = int(config.get("vae_tile_size", 256))
        self.tile_sample_min_overlap = int(config.get("vae_tile_overlap_min", 64))
        for name in ("tile_sample_size", "tile_sample_min_overlap"):
            if getattr(self, name) % self.spatial_compression_ratio:
                raise ValueError(f"MiniMax-H3-Image VAE {name} must be a multiple of {self.spatial_compression_ratio}")

        mean = config.get("latents_mean", [0.0] * latent_channels)
        std = config.get("latents_std", [1.0] * latent_channels)
        if len(mean) != latent_channels or len(std) != latent_channels:
            raise ValueError(f"Image latent statistics must contain {latent_channels} values, got mean={len(mean)}, std={len(std)}")
        self.register_buffer("latents_mean", torch.tensor(mean, dtype=torch.float32), persistent=False)
        self.register_buffer("latents_std", torch.tensor(std, dtype=torch.float32), persistent=False)
        self.register_buffer("pixel_mean", torch.tensor(MINIMAX_H3_PIXEL_MEAN, dtype=torch.float32), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(MINIMAX_H3_PIXEL_STD, dtype=torch.float32), persistent=False)

    # ------------------------------------------------------------------ loading
    def _reset_runtime_buffers(self) -> None:
        latent_channels = self.latent_channels
        self._buffers["latents_mean"] = torch.tensor(self.config.get("latents_mean", [0.0] * latent_channels), dtype=torch.float32)
        self._buffers["latents_std"] = torch.tensor(self.config.get("latents_std", [1.0] * latent_channels), dtype=torch.float32)
        self._buffers["pixel_mean"] = torch.tensor(MINIMAX_H3_PIXEL_MEAN, dtype=torch.float32)
        self._buffers["pixel_std"] = torch.tensor(MINIMAX_H3_PIXEL_STD, dtype=torch.float32)

    def expected_native_keys(self) -> set[str]:
        """Original checkpoint keys this module consumes (dropped keys excluded)."""
        state = set(self.state_dict())
        native = set()
        for name in state:
            source = name
            if ".attn.to_q." in name or ".attn.to_k." in name or ".attn.to_v." in name:
                source = re.sub(r"\.attn\.to_[qkv]\.", ".attn.to_qkv.", name)
            else:
                match = re.fullmatch(r"encoder\.down_blocks\.(\d+)\.(.+)", name)
                if match is not None:
                    rest = match.group(2).replace("resnets.", "block.", 1).replace("conv_shortcut.", "nin_shortcut.", 1).replace("downsamplers.0.", "downsample.", 1)
                    source = f"encoder.down.{match.group(1)}.{rest}"
                source = source.replace("decoder.proj_in.", "decoder.x_embedder.").replace(".attn.to_out.0.", ".attn.to_out.").replace(".ff.net.0.proj.", ".ff.w1.").replace(".ff.net.2.", ".ff.w2.")
            native.add(source)
        return native

    def load_native_weights(self, component_dir: str | Path) -> None:
        expected = {name: (tuple(value.shape), value.dtype) for name, value in self.state_dict().items()}
        loaded = set()
        for filename in _component_files(component_dir):
            with safe_open(filename, framework="pt", device="cpu") as checkpoint:
                for key in checkpoint.keys():
                    converted = convert_native_image_vae_tensor(key, checkpoint.get_tensor(key), self.decoder_num_heads, self.decoder_head_dim)
                    if key not in IMAGE_VAE_DROPPED_KEYS and not converted:
                        raise KeyError(f"Unexpected MiniMax-H3-Image VAE key: {key}")
                    for name, tensor in converted:
                        if name not in expected:
                            raise KeyError(f"MiniMax-H3-Image VAE checkpoint key {key!r} maps to unknown module tensor {name!r}")
                        if tuple(tensor.shape) != expected[name][0]:
                            raise ValueError(f"Shape mismatch for {name!r}: module expects {expected[name][0]}, checkpoint has {tuple(tensor.shape)}")
                        _assign_tensor(self, name, tensor.to(torch.float32))
                        loaded.add(name)
        missing = sorted(set(expected) - loaded)
        if missing:
            raise RuntimeError(f"MiniMax-H3-Image VAE checkpoint is missing {len(missing)} tensors, e.g. {missing[:8]}")

    def _prepare_inference_weights(self) -> None:
        # Norms, layer scales, register tokens, the output norm and the whole
        # encoder stay FP32; decoder projections run in the decode dtype.
        self.post_quant_conv.to(dtype=self.infer_dtype)
        for module in self.decoder.modules():
            if isinstance(module, nn.Linear):
                module.to(dtype=self.infer_dtype)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        device: str | torch.device | None = None,
        cpu_offload: bool = False,
        decode_dtype: torch.dtype = torch.bfloat16,
        attn_type: str = "torch_sdpa",
        tile_size: int | None = None,
        tile_overlap: int | None = None,
    ) -> "MiniMaxH3ImageVAE":
        vae_dir = _component_dir(model_path, "vae")
        with (vae_dir / "config.json").open(encoding="utf-8") as handle:
            config = json.load(handle)
        if tile_size is not None:
            config["vae_tile_size"] = tile_size
        if tile_overlap is not None:
            config["vae_tile_overlap_min"] = tile_overlap
        with torch.device("meta"):
            model = cls(config, device=device, cpu_offload=cpu_offload, decode_dtype=decode_dtype, attn_type=attn_type)
        model._reset_runtime_buffers()
        model.load_native_weights(vae_dir)
        model._prepare_inference_weights()
        model.eval().requires_grad_(False)
        if not cpu_offload:
            model.to(model.execution_device)
        logger.info(f"Loaded MiniMax-H3-Image VAE from {vae_dir} (decode dtype {decode_dtype}, tiles {model.tile_sample_size}/{model.tile_sample_min_overlap})")
        return model

    # ------------------------------------------------------------------ tiling
    def _split_tiles(self, length: int, tile_size: int, min_overlap: int) -> tuple[list[int], list[int], list[int]]:
        if tile_size >= length:
            return [0], [length], []
        num_tiles = math.ceil(length / tile_size)
        while tile_size * num_tiles - min_overlap * (num_tiles - 1) - length < 0:
            num_tiles += 1
        overlaps = [min_overlap] * (num_tiles - 1)
        remaining = tile_size * num_tiles - sum(overlaps) - length
        for index in range(remaining // self.spatial_compression_ratio):
            overlaps[index % (num_tiles - 1)] += self.spatial_compression_ratio
        starts = [0]
        for index in range(num_tiles - 1):
            starts.append(starts[-1] + tile_size - overlaps[index])
        return starts, [tile_size] * num_tiles, overlaps

    @staticmethod
    def _blend(a: torch.Tensor, b: torch.Tensor, blend_extent: int, dim: int) -> torch.Tensor:
        blend_extent = min(a.shape[dim], b.shape[dim], blend_extent)
        positions = torch.arange(blend_extent, device=b.device, dtype=b.dtype)
        shape = [1] * a.ndim
        shape[dim] = blend_extent
        weight_a = (1 - positions / blend_extent).view(shape)
        weight_b = (positions / blend_extent).view(shape)
        slice_a = [slice(None)] * a.ndim
        slice_a[dim] = slice(-blend_extent, None)
        slice_b = [slice(None)] * b.ndim
        slice_b[dim] = slice(0, blend_extent)
        blended = a[tuple(slice_a)] * weight_a + b[tuple(slice_b)] * weight_b
        if blend_extent == b.shape[dim]:
            return blended
        slice_rest = [slice(None)] * b.ndim
        slice_rest[dim] = slice(blend_extent, None)
        return torch.cat([blended, b[tuple(slice_rest)]], dim=dim)

    def _stitch_tiles(self, tiles: list[list[torch.Tensor]], height_overlaps: list[int], width_overlaps: list[int]) -> torch.Tensor:
        result_rows = []
        for i, row in enumerate(tiles):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self._blend(tiles[i - 1][j], tile, height_overlaps[i - 1], dim=-2)
                if j > 0:
                    tile = self._blend(row[j - 1], tile, width_overlaps[j - 1], dim=-1)
                if i < len(tiles) - 1:
                    tile = tile[..., : -height_overlaps[i], :]
                if j < len(row) - 1:
                    tile = tile[..., :, : -width_overlaps[j]]
                result_row.append(tile)
            result_rows.append(torch.cat(result_row, dim=-1))
        return torch.cat(result_rows, dim=-2)

    # ---------------------------------------------------------------- parallel
    def _use_decode_parallel(self) -> bool:
        return self.decode_parallel and self.use_decoder_tiling and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

    @staticmethod
    def _run_tiles(tiles: list[torch.Tensor], run, output_shape: tuple[int, ...], output_dtype: torch.dtype, parallel: bool) -> list[torch.Tensor]:
        """Run ``run`` over equally shaped tiles, optionally split over all ranks.

        In parallel mode each rank runs one balanced contiguous block of the
        row-major tile list, then an all-gather returns every result to every
        rank in the global order, so all ranks stitch the same output.
        """
        if not parallel or len(tiles) == 1:
            return [run(tile) for tile in tiles]
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        task_counts = MiniMaxH3VideoVAE._balanced_task_counts(len(tiles), world_size)
        local_start = sum(task_counts[:rank])
        # Ranks past the last tile send a zero-filled block of the same size.
        send_buffer = torch.zeros((max(task_counts), *output_shape), dtype=output_dtype, device=tiles[0].device)
        for index, tile in enumerate(tiles[local_start : local_start + task_counts[rank]]):
            output = run(tile)
            if tuple(output.shape) != output_shape or output.dtype != output_dtype:
                raise RuntimeError(f"MiniMax-H3-Image VAE tile output {tuple(output.shape)}/{output.dtype} differs from the expected {output_shape}/{output_dtype}")
            send_buffer[index] = output
        receive_buffers = [torch.empty_like(send_buffer) for _ in range(world_size)]
        dist.all_gather(receive_buffers, send_buffer)
        return [tile for buffer, count in zip(receive_buffers, task_counts) for tile in buffer[:count]]

    def _encode_tile(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.quant_conv(self.encoder(pixels[:, :, None]))[:, :, -1]

    def _encode_moments(self, pixels: torch.Tensor) -> torch.Tensor:
        if not self.use_encoder_tiling:
            return self._encode_tile(pixels)
        height, width = pixels.shape[-2:]
        y_starts, y_lengths, y_overlaps = self._split_tiles(height, self.tile_sample_size, self.tile_sample_min_overlap)
        x_starts, x_lengths, x_overlaps = self._split_tiles(width, self.tile_sample_size, self.tile_sample_min_overlap)
        rows = [[self._encode_tile(pixels[..., y : y + h, x : x + w]) for x, w in zip(x_starts, x_lengths)] for y, h in zip(y_starts, y_lengths)]
        ratio = self.spatial_compression_ratio
        return self._stitch_tiles(rows, [value // ratio for value in y_overlaps], [value // ratio for value in x_overlaps])

    def _decode_tile(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.post_quant_conv(latents))

    def _decode_pixels(self, latents: torch.Tensor) -> torch.Tensor:
        if not self.use_decoder_tiling:
            return self._decode_tile(latents)
        ratio = self.spatial_compression_ratio
        height, width = latents.shape[-2] * ratio, latents.shape[-1] * ratio
        y_starts, y_lengths, y_overlaps = self._split_tiles(height, self.tile_sample_size, self.tile_sample_min_overlap)
        x_starts, x_lengths, x_overlaps = self._split_tiles(width, self.tile_sample_size, self.tile_sample_min_overlap)
        tiles = [latents[..., y // ratio : (y + h) // ratio, x // ratio : (x + w) // ratio] for y, h in zip(y_starts, y_lengths) for x, w in zip(x_starts, x_lengths)]
        # The decoder returns its projections' dtype (the decode dtype).
        output_shape = (latents.shape[0], self.decoder.out_channels, y_lengths[0], x_lengths[0])
        tiles = self._run_tiles(tiles, self._decode_tile, output_shape, self.decoder.infer_dtype, self._use_decode_parallel())
        rows = [tiles[index : index + len(x_starts)] for index in range(0, len(tiles), len(x_starts))]
        return self._stitch_tiles(rows, y_overlaps, x_overlaps)

    # ---------------------------------------------------------------- residency
    def _activate(self) -> torch.device:
        if self.cpu_offload:
            self.to(self.execution_device)
        return self.execution_device

    def offload(self) -> None:
        self.to("cpu")
        _empty_device_cache(self.execution_device)
        gc.collect()

    # ------------------------------------------------------------------- public
    def normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        mean = self.latents_mean.to(latents.device).view(1, -1, 1, 1)
        std = self.latents_std.to(latents.device).view(1, -1, 1, 1)
        return (latents.float() - mean) / std

    def denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        mean = self.latents_mean.to(latents.device).view(1, -1, 1, 1)
        std = self.latents_std.to(latents.device).view(1, -1, 1, 1)
        return latents.float() * std + mean

    @staticmethod
    def sample_posterior(moments: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
        """Diffusers ``DiagonalGaussianDistribution.sample`` with the generator's own device."""
        mean, logvar = torch.chunk(moments, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        noise_device = generator.device if generator is not None else mean.device
        noise = torch.randn(mean.shape, generator=generator, device=noise_device, dtype=mean.dtype).to(mean.device)
        return mean + torch.exp(0.5 * logvar) * noise

    @torch.no_grad()
    def encode(self, pixels: torch.Tensor, generator: torch.Generator | None, *, patch_size: tuple[int, int, int] = (1, 2, 2)) -> torch.Tensor:
        """Encode RGB ``[1, 3, H, W]`` in ``[0, 1]`` into normalized ``[1, C, h, w]`` latents.

        The released recipe: crop the top-left to whole VAE cells, sample the
        posterior off ``generator``, round the sample to FP16, crop it to whole
        DiT patches, then normalize.
        """
        if pixels.ndim != 4 or pixels.shape[:2] != (1, 3):
            raise ValueError(f"MiniMax-H3-Image VAE encode expects [1, 3, H, W] pixels, got {tuple(pixels.shape)}")
        try:
            device = self._activate()
            ratio = self.spatial_compression_ratio
            pixels = pixels[..., : pixels.shape[-2] // ratio * ratio, : pixels.shape[-1] // ratio * ratio]
            pixels = pixels.to(device=device, dtype=torch.float32)
            pixels = (pixels - self.pixel_mean.to(device).view(1, -1, 1, 1)) / self.pixel_std.to(device).view(1, -1, 1, 1)
            moments = self._encode_moments(pixels)
            latents = self.sample_posterior(moments, generator).to(torch.float16).float()
            _, patch_h, patch_w = patch_size
            latents = latents[..., : latents.shape[-2] // patch_h * patch_h, : latents.shape[-1] // patch_w * patch_w]
            return self.normalize_latents(latents).cpu()
        finally:
            if self.cpu_offload:
                self.offload()

    @torch.no_grad()
    def decode_raw(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode VAE-space latents to ImageNet-normalized RGB in the decode dtype."""
        return self._decode_pixels(latents.to(self.post_quant_conv.weight.dtype))

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode normalized ``[1, C, h, w]`` latents into FP32 RGB ``[1, 3, H, W]`` in ``[0, 1]``."""
        if latents.ndim != 4 or latents.shape[1] != self.latent_channels:
            raise ValueError(f"MiniMax-H3-Image VAE decode expects [B, {self.latent_channels}, h, w] latents, got {tuple(latents.shape)}")
        try:
            device = self._activate()
            latents = self.denormalize_latents(latents.to(device))
            image = self.decode_raw(latents).float()
            image = image * self.pixel_std.to(device).view(1, -1, 1, 1) + self.pixel_mean.to(device).view(1, -1, 1, 1)
            return image.clamp(0, 1).cpu()
        finally:
            if self.cpu_offload:
                self.offload()
