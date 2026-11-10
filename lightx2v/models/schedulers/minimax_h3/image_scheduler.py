"""Fixed 6-step PDD schedule of MiniMax-H3-Image.

The checkpoint is distilled to the explicit sigma grid of ``pdd_config.json``
with one transformer output head per step, so the grid is not configurable.
Each sigma goes through its float32 timestep ``t = 1 - sigma`` and back, as the
released sampler does. The transformer predicts a data-ward velocity; the
default update is ``x_next = x + (sigma - sigma_next) * v`` in float32, the form
the PDD heads were distilled with (Diffusers ``euler_update_form="velocity"``).
``h3_image_step_update="reference_blend"`` selects the ``x_t``/``x0`` blend of
the MiniMax-H3 video scheduler, which SGLang uses; both are the same Euler
update up to float32 rounding.
"""

from dataclasses import replace

import torch

from lightx2v.models.networks.minimax_h3.image.packing import (
    REFERENCE_NOISE_AUG,
    build_image_packed_sequence,
    build_image_row_timesteps,
    patchify_image_latents,
)
from lightx2v.models.schedulers.scheduler import BaseScheduler
from lightx2v_platform.base.global_var import AI_DEVICE

STEP_UPDATES = ("velocity", "reference_blend")


def make_pdd_schedule(pdd_sigmas, device="cpu"):
    """Return ``(sigmas, timesteps)``: the grid rounded through float32 ``1 - sigma``."""
    timesteps = torch.tensor([1.0 - float(sigma) for sigma in pdd_sigmas[:-1]], dtype=torch.float32)
    sigmas = torch.cat((1.0 - timesteps, torch.zeros(1, dtype=torch.float32)))
    return sigmas.to(device), timesteps.to(device)


def draw_image_noise(generator, latent_channels, latent_height, latent_width, reference_grids):
    """Draw the target noise, then one noise tensor per reference, off one generator."""
    target = torch.randn((1, latent_channels, 1, latent_height, latent_width), generator=generator, device=generator.device, dtype=torch.float32).cpu()
    references = [torch.randn((1, latent_channels, 1, height, width), generator=generator, device=generator.device, dtype=torch.float32).cpu() for height, width in reference_grids]
    return target, references


class MiniMaxH3ImageScheduler(BaseScheduler):
    """Data-ward rectified-flow Euler sampler over the PDD grid of MiniMax-H3-Image."""

    def __init__(self, config):
        pdd_sigmas = config.get("pdd_sigmas")
        if pdd_sigmas is None:
            raise ValueError("MiniMax-H3-Image requires pdd_sigmas (read from pdd_config.json)")
        config_steps = int(config.get("infer_steps", len(pdd_sigmas) - 1))
        if config_steps != len(pdd_sigmas) - 1:
            raise ValueError(f"MiniMax-H3-Image has {len(pdd_sigmas) - 1} PDD output heads, so infer_steps must be {len(pdd_sigmas) - 1}, got {config_steps}")
        super().__init__(config)
        self.step_update = config.get("h3_image_step_update", "velocity")
        if self.step_update not in STEP_UPDATES:
            raise ValueError(f"h3_image_step_update must be one of {STEP_UPDATES}, got {self.step_update!r}")
        self.reference_timestep = float(config.get("reference_noise_aug", REFERENCE_NOISE_AUG))
        self.sigmas, self.timesteps = make_pdd_schedule(pdd_sigmas, AI_DEVICE)
        self.infer_steps = int(self.timesteps.numel())
        self.start_step_index = 0
        self.draft_steps = None
        self.video_latents = None
        self.audio_latents = None
        self.video_noise_pred = None
        self.audio_noise_pred = None
        self.layout = None
        self.layout_cpu = None

    @property
    def pdd_step_index(self):
        """The output head of the current step."""
        return self.step_index

    def prepare(self, text_token_tags, target_noise, reference_latents=(), reference_noise=(), patch_size=(1, 2, 2)):
        """Build the packed layout and the initial image rows.

        Args:
            text_token_tags: ``[num_text]`` modality tags of the conditioner rows.
            target_noise: ``[1, C, 1, H, W]`` noise of the generated image.
            reference_latents: normalized ``[1, C, h, w]`` latents per reference.
            reference_noise: ``[1, C, 1, h, w]`` noise per reference.
        """
        if len(reference_latents) != len(reference_noise):
            raise ValueError(f"MiniMax-H3-Image needs one noise tensor per reference, got {len(reference_noise)} for {len(reference_latents)}")
        _, channels, _, latent_height, latent_width = target_noise.shape
        reference_grids = [tuple(latents.shape[-2:]) for latents in reference_latents]
        condition_rows = []
        for latents, noise in zip(reference_latents, reference_noise):
            latents = latents.to(device=AI_DEVICE, dtype=torch.float32)
            noise = noise.to(device=AI_DEVICE, dtype=torch.float32).reshape(latents.shape)
            # References are noised to t = 0.999 and held there. The level is a
            # float32 tensor so that 1 - t is evaluated in float32, as released.
            level = torch.tensor(self.reference_timestep, dtype=torch.float32, device=latents.device)
            condition_rows.append(patchify_image_latents(level * latents + (1.0 - level) * noise, patch_size))
        target_rows = patchify_image_latents(target_noise.to(device=AI_DEVICE, dtype=torch.float32), patch_size)
        self.video_latents = torch.cat([*condition_rows, target_rows])
        self.audio_latents = None

        self.layout_cpu = build_image_packed_sequence(text_token_tags.cpu(), reference_grids, latent_height, latent_width, patch_size)
        if self.video_latents.shape[0] != self.layout_cpu.video_indices.numel():
            raise RuntimeError(f"MiniMax-H3-Image packed {self.video_latents.shape[0]} image rows for a layout of {self.layout_cpu.video_indices.numel()}")
        self.layout = replace(
            self.layout_cpu,
            position_ids=self.layout_cpu.position_ids.to(device=AI_DEVICE, dtype=torch.float32),
            token_tags=self.layout_cpu.token_tags.to(AI_DEVICE),
            video_indices=self.layout_cpu.video_indices.to(AI_DEVICE),
            audio_indices=self.layout_cpu.audio_indices.to(AI_DEVICE),
            text_indices=self.layout_cpu.text_indices.to(AI_DEVICE),
        )
        self.latent_channels = channels
        self.latent_height = latent_height
        self.latent_width = latent_width
        self.num_condition_video_rows = self.layout_cpu.num_condition_video_rows
        self.num_condition_audio_rows = 0
        self.step_index = 0
        self.video_noise_pred = None

    def step_pre(self, step_index):
        self.step_index = int(step_index)
        unique, inverse = build_image_row_timesteps(self.layout_cpu, float(self.timesteps[self.step_index]), self.reference_timestep)
        self.unique_timesteps_cpu = unique
        self.unique_timesteps = unique.to(AI_DEVICE)
        self.timestep_indices = inverse.to(AI_DEVICE)

    def step_post(self):
        if self.video_noise_pred is None:
            raise RuntimeError("MiniMax-H3-Image transformer did not populate the velocity prediction")
        start = self.num_condition_video_rows
        sample = self.video_latents[start:]
        velocity = self.video_noise_pred[start:].float()
        sigma = self.sigmas[self.step_index].to(device=sample.device, dtype=torch.float32)
        sigma_next = self.sigmas[self.step_index + 1].to(device=sample.device, dtype=torch.float32)
        if self.step_update == "velocity":
            updated = sample.float() + (sigma - sigma_next) * velocity
        else:
            sigma_from_timestep = 1.0 - self.timesteps[self.step_index].to(device=sample.device, dtype=torch.float32)
            denoised = sample.float() + sigma_from_timestep * velocity
            ratio = sigma_next / sigma
            updated = ratio * sample.float() + (1.0 - ratio) * denoised
        self.video_latents[start:] = updated.to(self.video_latents.dtype)

    def target_rows(self):
        return self.video_latents[self.num_condition_video_rows :]

    def clear(self):
        self.step_index = 0
        for name in ("video_latents", "audio_latents", "video_noise_pred", "audio_noise_pred", "layout", "layout_cpu", "unique_timesteps_cpu", "unique_timesteps", "timestep_indices"):
            setattr(self, name, None)
        self.generator = None
