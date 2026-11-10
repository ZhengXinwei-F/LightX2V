"""Numerical parity of the MiniMax-H3-Image port against the Diffusers reference.

Requires a Diffusers build that ships MiniMax-H3-Image (``huggingface/diffusers``
main, PR #15230). The checkpoint conversion is compared with the official
``scripts/convert_minimax_h3_to_diffusers.py``; set ``DIFFUSERS_SCRIPTS_DIR`` when
Diffusers is not an editable checkout.
"""

import importlib.util
import json
import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

pytest.importorskip("diffusers.models.transformers.transformer_minimax_h3_image", reason="Diffusers with MiniMax-H3-Image is not installed")

import diffusers  # noqa: E402
from diffusers import AutoencoderKLMiniMaxH3Image, MiniMaxH3ImageTransformer2DModel, MiniMaxH3Scheduler  # noqa: E402
from diffusers.modular_pipelines.minimax_h3_image import before_denoise as diffusers_denoise  # noqa: E402
from fixtures import RELEASE_SIGMAS, TINY_TRANSFORMER_CONFIG, TINY_VAE_CONFIG, tiny_native_transformer_state, tiny_native_vae_state, tiny_pdd_heads, tiny_runner_config  # noqa: E402

from lightx2v.models.networks.minimax_h3.image.checkpoint import convert_native_tensor  # noqa: E402
from lightx2v.models.networks.minimax_h3.image.packing import (  # noqa: E402
    IMAGE_TAG,
    bound_short_edge,
    build_image_packed_sequence,
    build_image_row_timesteps,
    get_image_canvas_size,
    patchify_image_latents,
)
from lightx2v.models.schedulers.minimax_h3.image_scheduler import make_pdd_schedule  # noqa: E402
from lightx2v.models.video_encoders.hf.minimax_h3.image_vae import MiniMaxH3ImageVAE  # noqa: E402


def _conversion_script():
    candidates = [os.environ.get("DIFFUSERS_SCRIPTS_DIR"), str(Path(diffusers.__file__).resolve().parents[2] / "scripts")]
    for directory in filter(None, candidates):
        path = Path(directory) / "convert_minimax_h3_to_diffusers.py"
        if path.is_file():
            spec = importlib.util.spec_from_file_location("convert_minimax_h3_to_diffusers", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    pytest.skip("Diffusers scripts/convert_minimax_h3_to_diffusers.py not found; set DIFFUSERS_SCRIPTS_DIR")


def _diffusers_transformer_config():
    c = TINY_TRANSFORMER_CONFIG
    return {
        "num_attention_heads": c["num_attention_heads"],
        "attention_head_dim": c["attention_head_dim"],
        "hidden_size": c["hidden_size"],
        "num_layers": c["num_layers"],
        "num_refiner_layers": c["token_refiner_num_layers"],
        "ffn_dim": c["ffn_hidden_size"],
        "in_channels": c["latents_dim"],
        "patch_size": c["patch_size"],
        "text_dim": c["text_dim"],
        "freq_dim": c["timestep_input_dim"],
        "time_embed_hidden_dim": c["time_embed_hidden_size"],
        "time_embed_dim": c["time_embed_dim"],
        "rope_freq_dim": c["rope_inv_freq_len"],
        "num_pdd_heads": 6,
    }


def _official_transformer_state(script, native, heads):
    config = _diffusers_transformer_config()
    state = {}
    for key, tensor in native.items():
        if key.endswith(".attn.qkv_proj.weight"):
            tensor = script.reorder_interleaved_qkv(tensor, config["num_attention_heads"], config["attention_head_dim"])
        for target, value in script.convert_transformer_key(key, tensor, config):
            state[target] = value
    for suffix in ("weight", "bias"):
        for step in range(config["num_pdd_heads"]):
            state[f"pdd_proj_out.{step}.{suffix}"] = heads[f"video_out.{suffix}"][step].contiguous()
    return state


def test_transformer_conversion_matches_official_script():
    script = _conversion_script()
    native = tiny_native_transformer_state()
    heads = tiny_pdd_heads()
    expected = _official_transformer_state(script, native, heads)
    ours = {}
    for key, tensor in native.items():
        ours.update(convert_native_tensor(key, tensor, TINY_TRANSFORMER_CONFIG["num_attention_heads"], TINY_TRANSFORMER_CONFIG["attention_head_dim"]))
    pdd = {key: value for key, value in expected.items() if key.startswith("pdd_proj_out.")}
    assert set(ours) == set(expected) - set(pdd)
    for key, value in ours.items():
        assert value.dtype == expected[key].dtype and torch.equal(value, expected[key]), key


def _vae_script_config(script):
    v = TINY_VAE_CONFIG
    kwargs = v["vit_decoder_kwargs"]
    config = dict(script.MINIMAX_H3_IMAGE_VAE_CONFIG)
    config.update(
        block_out_channels=v["block_out_channels"],
        layers_per_block=v["layers_per_block"],
        spatial_downsample_factors=v["spatial_downsample_factors"],
        temporal_downsample_factors=v["temporal_downsample_factors"],
        decoder_num_layers=kwargs["num_layers"],
        decoder_num_attention_heads=kwargs["heads"],
        decoder_attention_head_dim=kwargs["dim_head"],
    )
    return config


@pytest.fixture(scope="module")
def vae_pair(tmp_path_factory):
    script = _conversion_script()
    tmp = tmp_path_factory.mktemp("vae")
    native = tiny_native_vae_state()
    save_file(native, str(tmp / "diffusion_pytorch_model.safetensors"))
    with open(tmp / "config.json", "w") as handle:
        json.dump(TINY_VAE_CONFIG, handle)
    config = _vae_script_config(script)
    reference = AutoencoderKLMiniMaxH3Image(**config, latents_mean=TINY_VAE_CONFIG["latents_mean"], latents_std=TINY_VAE_CONFIG["latents_std"])
    state = {}
    for key, tensor in native.items():
        state.update(dict(script.convert_video_vae_key(key, tensor, config)))
    missing, unexpected = reference.load_state_dict(state, strict=False)
    assert not unexpected and not [key for key in missing if "rope" not in key]
    reference.enable_tiling(64, 64, 32, 32)
    reference.eval()
    converted_dir = tmp / "diffusers"
    reference.save_pretrained(str(converted_dir))

    def ours(decode_dtype):
        module = MiniMaxH3ImageVAE(TINY_VAE_CONFIG, device="cpu", decode_dtype=decode_dtype)
        module.load_native_weights(tmp)
        module._prepare_inference_weights()
        return module.eval()

    return reference, ours, converted_dir


def test_vae_encode_decode_fp32_match_diffusers(vae_pair):
    reference, ours_factory, _ = vae_pair
    ours = ours_factory(torch.float32)
    generator = torch.Generator().manual_seed(0)
    pixels = torch.randn(1, 3, 112, 176, generator=generator)
    with torch.no_grad():
        expected_moments = reference.encode(pixels).latent_dist.parameters
        moments = ours._encode_moments(pixels)
        torch.testing.assert_close(moments, expected_moments, rtol=1e-4, atol=1e-4)
        latents = torch.randn(1, 24, 7, 11, generator=generator)
        expected = reference.decode(latents).sample
        decoded = ours.decode_raw(latents)
        torch.testing.assert_close(decoded, expected, rtol=1e-4, atol=1e-4)


def test_vae_bf16_decode_close_to_diffusers(vae_pair):
    _, ours_factory, converted_dir = vae_pair
    reference = AutoencoderKLMiniMaxH3Image.from_pretrained(str(converted_dir), torch_dtype=torch.bfloat16).eval()
    reference.enable_tiling(64, 64, 32, 32)
    ours = ours_factory(torch.bfloat16)
    latents = torch.randn(1, 24, 7, 11, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        expected = reference.decode(latents).sample.float()
        decoded = ours.decode_raw(latents).float()
    assert decoded.dtype == expected.dtype
    error = (decoded - expected).norm() / expected.norm()
    assert error < 1e-2, float(error)


def test_vae_reference_recipe_matches_diffusers_edit_encoder(vae_pair):
    reference, ours_factory, _ = vae_pair
    ours = ours_factory(torch.float32)
    pixels = torch.rand(1, 3, 101, 150, generator=torch.Generator().manual_seed(2))
    latents = ours.encode(pixels, torch.Generator().manual_seed(9))

    # MiniMaxH3ImageEditVaeEncoderStep, written out on a [0, 1] tensor.
    ratio = 16
    cropped = pixels[..., : 101 // ratio * ratio, : 150 // ratio * ratio]
    mean = torch.tensor((0.485, 0.456, 0.406)).view(1, -1, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(1, -1, 1, 1)
    with torch.no_grad():
        posterior = reference.encode((cropped - mean) / std, return_dict=False)[0]
    expected = posterior.sample(generator=torch.Generator().manual_seed(9)).to(torch.float16).float()
    expected = expected[..., : expected.shape[-2] // 2 * 2, : expected.shape[-1] // 2 * 2]
    latents_mean = torch.tensor(TINY_VAE_CONFIG["latents_mean"]).view(1, -1, 1, 1)
    latents_std = torch.tensor(TINY_VAE_CONFIG["latents_std"]).view(1, -1, 1, 1)
    expected = (expected - latents_mean) / latents_std
    assert latents.shape == expected.shape == (1, 24, 6, 8)
    # FP16 rounding of the sample makes ulp-level encoder differences visible only as rare 1-ulp flips.
    torch.testing.assert_close(latents, expected, rtol=0, atol=2e-3)


# ------------------------------------------------------------------- transformer and loop
@pytest.fixture(scope="module")
def transformer_pair(tiny_release, tmp_path_factory):
    script = _conversion_script()
    state = _official_transformer_state(script, tiny_native_transformer_state(), tiny_pdd_heads())
    reference = MiniMaxH3ImageTransformer2DModel(**_diffusers_transformer_config())
    reference.load_state_dict(state, strict=True)
    converted_dir = tmp_path_factory.mktemp("transformer")
    reference.save_pretrained(str(converted_dir))
    # Loading in bfloat16 applies the release's FP32 islands (_keep_in_fp32_modules).
    reference = MiniMaxH3ImageTransformer2DModel.from_pretrained(str(converted_dir), torch_dtype=torch.bfloat16).eval()

    from lightx2v.models.networks.minimax_h3.image.model import MiniMaxH3ImageModel
    from lightx2v.models.schedulers.minimax_h3.image_scheduler import MiniMaxH3ImageScheduler

    config = tiny_runner_config(tiny_release, "i2i")
    model = MiniMaxH3ImageModel(tiny_release, config, torch.device("cpu"))
    scheduler = MiniMaxH3ImageScheduler(config)
    model.set_scheduler(scheduler)
    model.to_cuda()
    return reference, model, scheduler


def _edit_inputs(seed=0):
    generator = torch.Generator().manual_seed(seed)
    tags = torch.ones(23, dtype=torch.long)
    tags[3:11] = IMAGE_TAG
    prompt_embeds = (torch.randn(23, TINY_TRANSFORMER_CONFIG["text_dim"], generator=generator) * 0.5).to(torch.bfloat16)
    reference_latents = [torch.randn(1, 24, 4, 6, generator=generator), torch.randn(1, 24, 2, 4, generator=generator)]
    target_noise = torch.randn(1, 24, 1, 6, 8, generator=generator)
    reference_noise = [torch.randn(1, 24, 1, 4, 6, generator=generator), torch.randn(1, 24, 1, 2, 4, generator=generator)]
    return tags, prompt_embeds, reference_latents, target_noise, reference_noise


def _reference_forward(reference, scheduler, prompt_embeds, step):
    layout = scheduler.layout_cpu
    with torch.no_grad():
        return reference(
            hidden_states=scheduler.video_latents[None],
            encoder_hidden_states=prompt_embeds[None],
            timestep=scheduler.unique_timesteps_cpu,
            timestep_indices=scheduler.timestep_indices,
            token_tags=layout.token_tags,
            position_ids=layout.position_ids,
            image_indices=layout.video_indices,
            text_indices=layout.text_indices,
            pdd_step_index=step,
            return_dict=False,
        )[0][0].float()


@pytest.mark.parametrize("step", [0, 3, 5])
def test_transformer_step_matches_diffusers(transformer_pair, step):
    reference, model, scheduler = transformer_pair
    tags, prompt_embeds, reference_latents, target_noise, reference_noise = _edit_inputs()
    scheduler.prepare(tags, target_noise, reference_latents, reference_noise)
    scheduler.step_pre(step)
    model.infer({"text_encoder_output": {"prompt_embeds": prompt_embeds}})
    ours = scheduler.video_noise_pred.float()
    expected = _reference_forward(reference, scheduler, prompt_embeds, step)
    assert ours.shape == expected.shape
    error = (ours - expected).norm() / expected.norm()
    assert error < 2e-2, f"step {step}: relative error {float(error)}"
    # The head of another step must give a clearly different velocity.
    other = _reference_forward(reference, scheduler, prompt_embeds, (step + 1) % 6)
    assert (ours - other).norm() / other.norm() > 10 * error


def test_edit_denoising_loop_matches_diffusers(transformer_pair):
    reference, model, scheduler = transformer_pair
    tags, prompt_embeds, reference_latents, target_noise, reference_noise = _edit_inputs(seed=4)
    scheduler.prepare(tags, target_noise, reference_latents, reference_noise)

    # Reference side: the Diffusers blocks' layout, noising, timestep plan and update.
    reference_scheduler = MiniMaxH3Scheduler(shift_provided_sigmas=False, euler_update_form="velocity")
    timesteps = torch.tensor([1.0 - sigma for sigma in RELEASE_SIGMAS[:-1]], dtype=torch.float32)
    reference_scheduler.set_timesteps(sigmas=torch.cat([1.0 - timesteps, torch.zeros(1)]))
    position_ids, token_tags, image_indices, text_indices, num_reference_rows = diffusers_denoise.build_image_packed_sequence(tags, [(4, 6), (2, 4)], 6, 8, (1, 2, 2), 0)
    rows = [
        diffusers_denoise.patchify_video_latents(reference_scheduler.scale_noise(latent, torch.tensor(0.999), noise[:, :, 0])[:, :, None], (1, 2, 2))
        for latent, noise in zip(reference_latents, reference_noise)
    ]
    latents = torch.cat([*rows, diffusers_denoise.patchify_video_latents(target_noise, (1, 2, 2))])
    assert torch.equal(latents, scheduler.video_latents)

    for index, timestep in enumerate(reference_scheduler.timesteps):
        row = torch.full((position_ids.shape[0],), float(timestep), dtype=torch.float32)
        row[image_indices[:num_reference_rows]] = 0.999
        unique, inverse = torch.unique(row, sorted=True, return_inverse=True)
        with torch.no_grad():
            velocity = reference(
                hidden_states=latents[None],
                encoder_hidden_states=prompt_embeds[None],
                timestep=unique,
                timestep_indices=inverse,
                token_tags=token_tags,
                position_ids=position_ids,
                image_indices=image_indices,
                text_indices=text_indices,
                pdd_step_index=index,
                return_dict=False,
            )[0][0]
        latents[num_reference_rows:] = reference_scheduler.step(velocity[num_reference_rows:].float(), timestep, latents[num_reference_rows:], return_dict=False)[0]

        scheduler.step_pre(index)
        model.infer({"text_encoder_output": {"prompt_embeds": prompt_embeds}})
        scheduler.step_post()

    assert torch.equal(scheduler.video_latents[: scheduler.num_condition_video_rows], latents[:num_reference_rows]), "reference rows must stay fixed"
    target = scheduler.target_rows()
    error = (target - latents[num_reference_rows:]).norm() / latents[num_reference_rows:].norm()
    assert error < 3e-2, float(error)


def test_canvas_rounding_matches_released_sizes():
    for height, width in [(4096, 4096), (3552, 4736), (3360, 5024), (3072, 5472), (2688, 6272)]:
        assert get_image_canvas_size(height, width) == diffusers_denoise.get_image_canvas_size(height, width, 2048 * 2048, 32) == (height, width)
    layout = build_image_packed_sequence(torch.ones(4, dtype=torch.long), [], 4096 // 16, 4096 // 16)
    assert layout.video_indices.numel() == 128 * 128


# ------------------------------------------------------------------ packing and schedule
def _text_tags(num_text, vision_rows=()):
    tags = torch.ones(num_text, dtype=torch.long)
    for start, stop in vision_rows:
        tags[start:stop] = IMAGE_TAG
    return tags


@pytest.mark.parametrize(
    "text_vision, reference_grids, target",
    [
        ((), [], (256, 256)),
        ((), [], (168, 384)),
        (((3, 40), (45, 60)), [(64, 96), (128, 50)], (222, 296)),
    ],
)
def test_layout_matches_diffusers(text_vision, reference_grids, target):
    tags = _text_tags(80, text_vision)
    ours = build_image_packed_sequence(tags, reference_grids, *target)
    position_ids, token_tags, image_indices, text_indices, num_reference_rows = diffusers_denoise.build_image_packed_sequence(tags, reference_grids, target[0], target[1], (1, 2, 2), 0)
    assert torch.equal(ours.position_ids, position_ids)
    assert torch.equal(ours.token_tags, token_tags)
    assert torch.equal(ours.video_indices, image_indices)
    assert torch.equal(ours.text_indices, text_indices)
    assert ours.num_condition_video_rows == num_reference_rows


@pytest.mark.parametrize("size", [(4096, 4096), (3552, 4736), (2688, 6272), (1000, 700), (32, 4000)])
def test_canvas_matches_diffusers(size):
    assert get_image_canvas_size(*size, 2048 * 2048, 32) == diffusers_denoise.get_image_canvas_size(*size, 2048 * 2048, 32)
    assert get_image_canvas_size(*size, 1024, 32) == diffusers_denoise.get_image_canvas_size(*size, 1024, 32)


def test_reference_bounds_match_diffusers():
    from diffusers.modular_pipelines.minimax_h3_image.before_encoder import bound_short_edge as reference

    for width, height in [(640, 480), (1500, 1200), (5000, 3000), (33, 900), (1024, 4096)]:
        assert bound_short_edge(width, height, 1024, 2048) == reference(width, height, 1024, 2048, 32)
        assert bound_short_edge(width, height, 1024, 1024) == reference(width, height, 1024, 1024, 32)


def test_schedule_and_row_plan_match_diffusers():
    from diffusers import MiniMaxH3Scheduler

    scheduler = MiniMaxH3Scheduler(shift_provided_sigmas=False, euler_update_form="velocity")
    timesteps = torch.tensor([1.0 - sigma for sigma in RELEASE_SIGMAS[:-1]], dtype=torch.float32)
    scheduler.set_timesteps(sigmas=torch.cat([1.0 - timesteps, torch.zeros(1)]))
    sigmas, ours = make_pdd_schedule(RELEASE_SIGMAS)
    assert torch.equal(scheduler.timesteps, ours)
    assert torch.equal(scheduler.sigmas, sigmas)

    layout = build_image_packed_sequence(_text_tags(9, [(1, 4)]), [(4, 4)], 6, 8)
    for timestep in scheduler.timesteps:
        row = torch.full((layout.sequence_length,), float(timestep), dtype=torch.float32)
        row[layout.video_indices[: layout.num_condition_video_rows]] = 0.999
        expected_unique, expected_inverse = torch.unique(row, sorted=True, return_inverse=True)
        unique, inverse = build_image_row_timesteps(layout, float(timestep))
        assert torch.equal(unique, expected_unique) and torch.equal(inverse, expected_inverse)


@pytest.mark.parametrize("step_update, form", [("velocity", "velocity"), ("reference_blend", "blend")])
def test_scheduler_steps_match_diffusers(step_update, form):
    from diffusers import MiniMaxH3Scheduler

    from lightx2v.models.schedulers.minimax_h3.image_scheduler import MiniMaxH3ImageScheduler

    reference = MiniMaxH3Scheduler(shift_provided_sigmas=False, euler_update_form=form)
    timesteps = torch.tensor([1.0 - sigma for sigma in RELEASE_SIGMAS[:-1]], dtype=torch.float32)
    reference.set_timesteps(sigmas=torch.cat([1.0 - timesteps, torch.zeros(1)]))
    ours = MiniMaxH3ImageScheduler({"infer_steps": 6, "pdd_sigmas": RELEASE_SIGMAS, "h3_image_step_update": step_update})

    generator = torch.Generator().manual_seed(0)
    reference_latents = torch.randn(1, 24, 4, 4, generator=generator)
    target_noise = torch.randn(1, 24, 1, 6, 8, generator=generator)
    reference_noise = torch.randn(1, 24, 1, 4, 4, generator=generator)
    ours.prepare(_text_tags(5), target_noise, [reference_latents], [reference_noise])
    condition = patchify_image_latents(reference.scale_noise(reference_latents, torch.tensor(0.999), reference_noise[:, :, 0]))
    expected = torch.cat([condition, patchify_image_latents(target_noise)])
    assert torch.equal(ours.video_latents, expected)

    num_reference = ours.num_condition_video_rows
    for index, timestep in enumerate(reference.timesteps):
        velocity = torch.randn(expected.shape, generator=generator)
        ours.step_pre(index)
        ours.video_noise_pred = velocity
        ours.step_post()
        expected[num_reference:] = reference.step(velocity[num_reference:], timestep, expected[num_reference:], return_dict=False)[0]
        assert torch.equal(ours.video_latents, expected), f"step {index}"
