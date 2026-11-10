"""MiniMax-H3-Image packed-sequence geometry.

One request is one unbatched sequence ``[text | reference images | target image]``.
Every image is a single latent frame on the rotary clock, which starts where the
text rows end: reference ``i`` takes time slot ``num_text + i`` and the target
takes the slot after the last reference. The spatial axes reuse the
aspect-normalized H3 grid, built per image from that image's own latent size.
This mirrors ``build_image_packed_sequence`` of the Diffusers MiniMax-H3-Image
pipeline.
"""

import math

import numpy as np
import torch

from lightx2v.models.networks.minimax_h3.packing import VIDEO_TAG, MiniMaxH3PackedSequence, _spatial_position_grid, patchify_video_latents

IMAGE_TAG = VIDEO_TAG
VAE_SPATIAL_SCALE = 16
CANVAS_MULTIPLE = 32
REFERENCE_NOISE_AUG = 0.999
DEFAULT_MIN_GENERATION_AREA = 2048 * 2048
MIN_GENERATION_AREA_RANGE = (32 * 32, 4096 * 4096)
MAX_REFERENCE_IMAGES = 16
MAX_SEQUENCE_TOKENS = 4096
# Short-edge bounds of the two copies of an edit reference: the VAE copy and
# the Qwen3-VL copy. A short edge inside the bounds keeps the original pixels.
REFERENCE_VAE_SHORT_EDGE = (1024, 2048)
REFERENCE_VISION_SHORT_EDGE = (1024, 1024)


def get_image_canvas_size(height, width, min_generation_area=DEFAULT_MIN_GENERATION_AREA, multiple=CANVAS_MULTIPLE):
    """The ``(height, width)`` canvas a ``height x width`` request is generated on.

    A request below ``min_generation_area`` pixels is generated on a larger
    canvas of the same aspect ratio; both sides round to the nearest
    ``multiple`` (Python round, ties to even).
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"MiniMax-H3-Image height and width must be positive, got {height}x{width}")
    low, high = MIN_GENERATION_AREA_RANGE
    if not low <= min_generation_area <= high:
        raise ValueError(f"min_generation_area must be within [{low}, {high}], got {min_generation_area}")
    scale = max(1.0, math.sqrt(min_generation_area / (width * height)))
    return (
        max(multiple, round(height * scale / multiple) * multiple),
        max(multiple, round(width * scale / multiple) * multiple),
    )


def bound_short_edge(width, height, min_short_edge, max_short_edge, multiple=CANVAS_MULTIPLE):
    """The ``(width, height)`` an edit reference is resized to for one consumer."""
    short_edge = min(width, height)
    if min_short_edge <= short_edge <= max_short_edge:
        return width, height
    scale = (min_short_edge if short_edge < min_short_edge else max_short_edge) / short_edge
    return (
        max(multiple, round(width * scale / multiple) * multiple),
        max(multiple, round(height * scale / multiple) * multiple),
    )


def reference_latent_grid(width, height, patch_size=(1, 2, 2)):
    """``(latent_height, latent_width)`` of a reference: whole VAE cells, whole patches."""
    _, patch_h, patch_w = patch_size
    return height // VAE_SPATIAL_SCALE // patch_h * patch_h, width // VAE_SPATIAL_SCALE // patch_w * patch_w


def frame_position_grid(latent_height, latent_width, patch_h, patch_w):
    """The float64 ``(h, w)`` rotary coordinates of one latent frame."""
    sqrt_area = np.sqrt(latent_height * latent_width)
    height_grid = _spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_position_grid(latent_width, patch_w, sqrt_area)
    grids = torch.meshgrid(height_grid, width_grid, indexing="ij")
    return torch.stack([grid.reshape(-1) for grid in grids], dim=-1)


def build_image_packed_sequence(text_token_tags, reference_grids, latent_height, latent_width, patch_size=(1, 2, 2)):
    """Build the ``[text | references | target]`` layout and its float64 rotary grid."""
    _, patch_h, patch_w = patch_size
    for height, width in [*reference_grids, (latent_height, latent_width)]:
        if height % patch_h or width % patch_w or height <= 0 or width <= 0:
            raise ValueError(f"latent grid {height}x{width} is not a positive multiple of the patch {patch_size}")
    num_text_tokens = int(text_token_tags.shape[0])
    num_reference_rows = sum((height // patch_h) * (width // patch_w) for height, width in reference_grids)
    num_target_rows = (latent_height // patch_h) * (latent_width // patch_w)
    sequence_length = num_text_tokens + num_reference_rows + num_target_rows

    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)
    cursor = num_text_tokens
    rotary_time = float(num_text_tokens)
    for height, width in reference_grids:
        rows = slice(cursor, cursor + (height // patch_h) * (width // patch_w))
        cursor = rows.stop
        position_ids[rows, 0] = rotary_time
        position_ids[rows, 1:] = frame_position_grid(height, width, patch_h, patch_w)
        rotary_time += 1.0
    position_ids[cursor:, 0] = rotary_time
    position_ids[cursor:, 1:] = frame_position_grid(latent_height, latent_width, patch_h, patch_w)

    text_indices = torch.arange(num_text_tokens, dtype=torch.long)
    image_indices = torch.arange(num_text_tokens, sequence_length, dtype=torch.long)
    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = text_token_tags.to(torch.long)
    token_tags[image_indices] = IMAGE_TAG
    return MiniMaxH3PackedSequence(
        sequence_length=sequence_length,
        position_ids=position_ids,
        token_tags=token_tags,
        video_indices=image_indices,
        audio_indices=torch.empty(0, dtype=torch.long),
        text_indices=text_indices,
        num_condition_video_rows=num_reference_rows,
        num_condition_audio_rows=0,
    )


def build_image_row_timesteps(layout, timestep, reference_timestep=REFERENCE_NOISE_AUG):
    """Per-row timesteps: text and target rows at ``timestep``, references held at ``reference_timestep``."""
    row_timesteps = torch.full((layout.sequence_length,), float(timestep), dtype=torch.float32)
    row_timesteps[layout.video_indices[: layout.num_condition_video_rows]] = reference_timestep
    return torch.unique(row_timesteps, sorted=True, return_inverse=True)


def patchify_image_latents(latents, patch_size=(1, 2, 2)):
    """``[1, C, H, W]`` (or ``[1, C, 1, H, W]``) latents to unbatched patch rows."""
    if latents.ndim == 4:
        latents = latents[:, :, None]
    return patchify_video_latents(latents, patch_size)


def unpatchify_image_rows(rows, latent_height, latent_width, channels=24, patch_size=(1, 2, 2)):
    """Inverse of :func:`patchify_image_latents`, returning ``[1, C, H, W]``."""
    _, patch_h, patch_w = patch_size
    rows = rows.reshape(1, latent_height // patch_h, latent_width // patch_w, channels, patch_h, patch_w)
    return rows.permute(0, 3, 1, 4, 2, 5).reshape(1, channels, latent_height, latent_width).contiguous()
