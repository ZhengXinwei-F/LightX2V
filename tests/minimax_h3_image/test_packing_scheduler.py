"""Packing, canvas, RoPE positions and the PDD schedule, checked against Diffusers where installed."""

import pytest
import torch

from lightx2v.models.networks.minimax_h3.image.packing import (
    IMAGE_TAG,
    REFERENCE_NOISE_AUG,
    bound_short_edge,
    build_image_packed_sequence,
    build_image_row_timesteps,
    get_image_canvas_size,
    patchify_image_latents,
    reference_latent_grid,
    unpatchify_image_rows,
)
from lightx2v.models.schedulers.minimax_h3.image_scheduler import draw_image_noise, make_pdd_schedule

RELEASE_SIGMAS = [1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0]


def _text_tags(num_text, vision_rows=()):
    tags = torch.ones(num_text, dtype=torch.long)
    for start, stop in vision_rows:
        tags[start:stop] = IMAGE_TAG
    return tags


def test_layout_order_and_time_axis():
    tags = _text_tags(7, [(2, 5)])
    layout = build_image_packed_sequence(tags, [(4, 6), (2, 2)], 8, 10)
    num_ref = 2 * 3 + 1 * 1
    num_target = 4 * 5
    assert layout.sequence_length == 7 + num_ref + num_target
    assert torch.equal(layout.text_indices, torch.arange(7))
    assert torch.equal(layout.video_indices, torch.arange(7, layout.sequence_length))
    assert layout.audio_indices.numel() == 0
    assert layout.num_condition_video_rows == num_ref
    assert torch.equal(layout.token_tags[:7], tags)
    assert bool((layout.token_tags[7:] == IMAGE_TAG).all())
    time = layout.position_ids[:, 0]
    assert torch.equal(time[:7], torch.arange(7, dtype=torch.float64))
    assert bool((time[7:13] == 7).all()) and bool((time[13:14] == 8).all()) and bool((time[14:] == 9).all())
    # A square target spans [0, 32) on both spatial axes.
    square = build_image_packed_sequence(torch.ones(1, dtype=torch.long), [], 8, 8)
    assert square.position_ids[1:, 1:].min() == 0 and square.position_ids[1:, 1:].max() == 32 * 3 / 4


def test_row_timesteps_hold_references():
    layout = build_image_packed_sequence(torch.ones(3, dtype=torch.long), [(2, 2)], 2, 4)
    unique, inverse = build_image_row_timesteps(layout, 0.25)
    assert unique.tolist() == pytest.approx([0.25, REFERENCE_NOISE_AUG])
    row = unique[inverse]
    assert bool((row[:3] == 0.25).all()) and float(row[3]) == pytest.approx(REFERENCE_NOISE_AUG) and bool((row[4:] == 0.25).all())


def test_patchify_round_trip():
    latents = torch.randn(1, 24, 6, 10)
    rows = patchify_image_latents(latents)
    assert rows.shape == (15, 96)
    assert torch.equal(unpatchify_image_rows(rows, 6, 10), latents)


def test_canvas_and_reference_bounds():
    assert get_image_canvas_size(4096, 4096) == (4096, 4096)
    assert get_image_canvas_size(1024, 1024) == (2048, 2048)
    assert get_image_canvas_size(1000, 1000, 1024) == (992, 992)
    assert bound_short_edge(1500, 1200, 1024, 2048) == (1500, 1200)
    assert bound_short_edge(800, 400, 1024, 1024) == (2048, 1024)
    assert reference_latent_grid(1500, 1210) == (74, 92)


def test_pdd_schedule_rounds_through_timesteps():
    sigmas, timesteps = make_pdd_schedule(RELEASE_SIGMAS)
    assert timesteps.dtype == torch.float32 and timesteps.numel() == 6
    assert torch.equal(sigmas[:-1], 1.0 - timesteps) and float(sigmas[-1]) == 0.0
    assert float(timesteps[0]) == 0.0


def test_noise_draw_order():
    generator = torch.Generator().manual_seed(3)
    target, references = draw_image_noise(generator, 24, 4, 6, [(2, 4), (4, 2)])
    replay = torch.Generator().manual_seed(3)
    assert torch.equal(target, torch.randn((1, 24, 1, 4, 6), generator=replay))
    for noise, shape in zip(references, [(2, 4), (4, 2)]):
        assert torch.equal(noise, torch.randn((1, 24, 1, *shape), generator=replay))
