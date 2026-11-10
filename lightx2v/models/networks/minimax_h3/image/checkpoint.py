"""Map the released MiniMax-H3-Image checkpoint onto LightX2V's H3 weight names.

The MiniMax-H3-Image release (``MiniMaxAI/MiniMax-H3-Image``) ships its DiT in
the original MiniMax-H3 layout (``blocks.*.attn.qkv_proj``, ``mlp.fc1``,
``video_patch_proj`` ...), while the native LightX2V H3 network consumes the
Diffusers layout of the MiniMax-H3 video release. This module converts one
tensor at a time while the checkpoint is streamed, so the 66 GB state dict is
never materialized twice. The conversion follows
``scripts/convert_minimax_h3_to_diffusers.py --version image`` in Diffusers:

* the fused QKV rows are stored per-head interleaved
  (``[head0: q k v, head1: q k v, ...]``) and are split into ``to_q/to_k/to_v``;
* the fused SwiGLU ``mlp.fc1`` is stored ``[gate; value]`` while the native
  feed-forward reads ``[value; gate]``;
* ``final_layer.video_out`` is replaced by the per-step PDD heads of
  ``pdd_fused_heads.safetensors`` and is not loaded;
* ``rope.inv_freq`` equals the recomputed frequencies and is not loaded.
"""

import json
import os
import re

import torch
from safetensors import safe_open

PDD_HEADS_FILENAME = "pdd_fused_heads.safetensors"
PDD_CONFIG_FILENAME = "pdd_config.json"
PDD_HEAD_PREFIX = "pdd_proj_out"

# Original checkpoint keys that LightX2V does not load.
DROPPED_KEYS = ("rope.inv_freq", "final_layer.video_out.weight", "final_layer.video_out.bias")

# The original transformer/config.json uses the SGLang-native names.
NATIVE_CONFIG_NAMES = {
    "token_refiner_num_layers": "num_refiner_layers",
    "ffn_hidden_size": "ffn_dim",
    "latents_dim": "in_channels",
    "timestep_input_dim": "freq_dim",
    "time_embed_hidden_size": "time_embed_hidden_dim",
    "rope_inv_freq_len": "rope_freq_dim",
}

_GLOBAL_NAMES = {
    "video_patch_proj.weight": "proj_in.weight",
    "video_patch_proj.bias": "proj_in.bias",
    "condition_proj.weight": "context_embedder.weight",
    "condition_proj.bias": "context_embedder.bias",
    "time_embedder.proj_in.weight": "time_embedder.linear_1.weight",
    "time_embedder.proj_in.bias": "time_embedder.linear_1.bias",
    "time_embedder.proj_out.weight": "time_embedder.linear_2.weight",
    "time_embedder.proj_out.bias": "time_embedder.linear_2.bias",
    "final_layer.norm.weight": "norm_out.norm.weight",
    "final_layer.adaln_proj.linear.weight": "norm_out.linear.weight",
    "final_layer.adaln_proj.linear.bias": "norm_out.linear.bias",
    "token_refiner.final_norm.weight": "token_refiner.final_norm.weight",
}

_BLOCK_SUFFIXES = {
    "norm1.weight": "norm1.weight",
    "norm2.weight": "norm2.weight",
    "adaln_proj.linear.weight": "adaln_proj.linear.weight",
    "adaln_proj.linear.bias": "adaln_proj.linear.bias",
    "attn.q_norm.weight": "attn.norm_q.weight",
    "attn.k_norm.weight": "attn.norm_k.weight",
    "attn.out_proj.weight": "attn.to_out.0.weight",
    "mlp.fc1.weight": "ff.net.0.proj.weight",
    "mlp.fc2.weight": "ff.net.2.weight",
}
_QKV_SUFFIX = "attn.qkv_proj.weight"
_QKV_TARGETS = ("attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight")
_BLOCK = re.compile(r"^blocks\.(\d+)\.(.+)$")
_REFINER_BLOCK = re.compile(r"^token_refiner\.blocks\.(\d+)\.(.+)$")


def normalize_native_transformer_config(native_config):
    """Translate the release's ``transformer/config.json`` into LightX2V H3 names."""
    config = {}
    for key, value in native_config.items():
        if key.startswith("_"):
            continue
        config[NATIVE_CONFIG_NAMES.get(key, key)] = value
    return config


def is_native_checkpoint_key(key):
    """Whether ``key`` is spelled in the original MiniMax-H3 layout."""
    return key in DROPPED_KEYS or key in _GLOBAL_NAMES or _BLOCK.fullmatch(key) is not None or _REFINER_BLOCK.fullmatch(key) is not None


def native_target_names(source_key):
    """Return the LightX2V names one original key maps to (empty when dropped)."""
    if source_key in DROPPED_KEYS:
        return ()
    target = _GLOBAL_NAMES.get(source_key)
    if target is not None:
        return (target,)
    match = _BLOCK.fullmatch(source_key)
    if match is not None:
        prefix = f"transformer_blocks.{match.group(1)}."
    else:
        match = _REFINER_BLOCK.fullmatch(source_key)
        if match is None:
            raise KeyError(f"Unrecognized MiniMax-H3-Image checkpoint key: {source_key}")
        prefix = f"token_refiner.refiner_blocks.{match.group(1)}."
    suffix = match.group(2)
    if suffix == _QKV_SUFFIX:
        return tuple(prefix + name for name in _QKV_TARGETS)
    target = _BLOCK_SUFFIXES.get(suffix)
    if target is None:
        raise KeyError(f"Unrecognized MiniMax-H3-Image checkpoint key: {source_key}")
    return (prefix + target,)


def split_interleaved_qkv(weight, num_heads, head_dim):
    """Split a per-head interleaved fused QKV weight into contiguous Q, K and V."""
    if weight.shape[0] != 3 * num_heads * head_dim:
        raise ValueError(f"fused QKV weight has {weight.shape[0]} rows, expected 3 * {num_heads} heads * {head_dim}")
    grouped = weight.reshape(num_heads, 3, head_dim, *weight.shape[1:])
    return tuple(grouped[:, index].reshape(num_heads * head_dim, *weight.shape[1:]).contiguous() for index in range(3))


def swap_swiglu_halves(weight):
    """Reorder a fused SwiGLU projection from ``[gate; value]`` to ``[value; gate]``."""
    if weight.shape[0] % 2:
        raise ValueError(f"fused SwiGLU weight has an odd row count: {tuple(weight.shape)}")
    gate, value = weight.chunk(2, dim=0)
    return torch.cat((value, gate), dim=0).contiguous()


def convert_native_tensor(source_key, tensor, num_heads, head_dim):
    """Convert one original key/tensor pair into LightX2V ``(key, tensor)`` pairs."""
    targets = native_target_names(source_key)
    if not targets:
        return []
    if source_key.endswith(_QKV_SUFFIX):
        return list(zip(targets, split_interleaved_qkv(tensor, num_heads, head_dim)))
    if source_key.endswith(".mlp.fc1.weight"):
        tensor = swap_swiglu_halves(tensor)
    return [(targets[0], tensor)]


def native_target_shapes(source_key, shape, num_heads, head_dim):
    """Shapes of the converted tensors, derived from the original shape only."""
    targets = native_target_names(source_key)
    if source_key.endswith(_QKV_SUFFIX):
        return {name: (num_heads * head_dim, *shape[1:]) for name in targets}
    return {name: tuple(shape) for name in targets}


def resolve_pdd_paths(model_path, config=None):
    """Return ``(heads_path, pdd_config_path)`` of the release's PDD heads."""
    config = config or {}
    heads_path = config.get("pdd_heads_path")
    if heads_path is None:
        filename = PDD_HEADS_FILENAME
        index_path = os.path.join(model_path, "model_index.json")
        if os.path.isfile(index_path):
            with open(index_path, encoding="utf-8") as handle:
                filename = json.load(handle).get("_minimax_h3", {}).get("pdd_fused_heads", filename)
        heads_path = os.path.join(model_path, filename)
    pdd_config_path = config.get("pdd_config_path") or os.path.join(os.path.dirname(heads_path), PDD_CONFIG_FILENAME)
    return heads_path, pdd_config_path


def read_pdd_config(path):
    """Read and validate ``pdd_config.json``: the step count and its sigma grid."""
    with open(path, encoding="utf-8") as handle:
        pdd_config = json.load(handle)
    steps = int(pdd_config["num_inference_steps"])
    sigmas = [float(value) for value in pdd_config["sigmas"]]
    if len(sigmas) != steps + 1 or sigmas[0] != 1.0 or sigmas[-1] != 0.0 or any(b >= a for a, b in zip(sigmas, sigmas[1:])):
        raise ValueError(f"{path} must list num_inference_steps + 1 strictly decreasing sigmas from 1.0 to 0.0, got {sigmas}")
    return steps, sigmas


def load_pdd_heads(path, num_steps, hidden_size, out_features, device="cpu"):
    """Split the stacked PDD heads into one ``pdd_proj_out.{i}`` linear per step (FP32)."""
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        if keys != {"video_out.weight", "video_out.bias"}:
            raise KeyError(f"{path} must hold exactly video_out.weight and video_out.bias, got {sorted(keys)}")
        weight = handle.get_tensor("video_out.weight")
        bias = handle.get_tensor("video_out.bias")
    expected = {"weight": (num_steps, out_features, hidden_size), "bias": (num_steps, out_features)}
    for name, tensor in (("weight", weight), ("bias", bias)):
        if tuple(tensor.shape) != expected[name]:
            raise ValueError(f"PDD video_out.{name} must have shape {expected[name]}, got {tuple(tensor.shape)}")
    heads = {}
    for step in range(num_steps):
        heads[f"{PDD_HEAD_PREFIX}.{step}.weight"] = weight[step].to(device=device, dtype=torch.float32).contiguous()
        heads[f"{PDD_HEAD_PREFIX}.{step}.bias"] = bias[step].to(device=device, dtype=torch.float32).contiguous()
    return heads
