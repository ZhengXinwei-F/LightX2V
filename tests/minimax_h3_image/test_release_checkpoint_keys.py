"""Check LightX2V's expected weights against the real MiniMax-H3-Image checkpoint layout.

Only metadata is read: ``transformer/model.safetensors.index.json``, the
configs, and the safetensors *headers* of every shard, of ``vae/`` and of
``pdd_fused_heads.safetensors``. With ``MINIMAX_H3_IMAGE_PATH`` set to a local
copy of the release the files are read from disk; otherwise the headers are
fetched from Hugging Face with HTTP range requests (no tensor data), and the
test is skipped when the network is unavailable.
"""

import json
import os
import struct
import urllib.error
import urllib.request

import pytest
import torch

from lightx2v.models.networks.minimax_h3.image.checkpoint import DROPPED_KEYS, PDD_HEAD_PREFIX, native_target_names, native_target_shapes, normalize_native_transformer_config
from lightx2v.models.networks.minimax_h3.image.weights import MiniMaxH3ImagePostWeights, MiniMaxH3ImagePreWeights
from lightx2v.models.networks.minimax_h3.model import MiniMaxH3Model
from lightx2v.models.networks.minimax_h3.weights import MiniMaxH3TransformerWeights
from lightx2v.models.video_encoders.hf.minimax_h3.image_vae import IMAGE_VAE_DROPPED_KEYS, MiniMaxH3ImageVAE, native_image_vae_names

REPO_URL = os.environ.get("MINIMAX_H3_IMAGE_URL", "https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/resolve/main/")


class _Release:
    def __init__(self):
        self.local = os.environ.get("MINIMAX_H3_IMAGE_PATH")

    def _read(self, name, byte_range=None):
        if self.local:
            with open(os.path.join(self.local, name), "rb") as handle:
                if byte_range is None:
                    return handle.read()
                handle.seek(byte_range[0])
                return handle.read(byte_range[1] - byte_range[0] + 1)
        headers = {} if byte_range is None else {"Range": f"bytes={byte_range[0]}-{byte_range[1]}"}
        request = urllib.request.Request(REPO_URL + name, headers=headers)
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()

    def json(self, name):
        return json.loads(self._read(name))

    def header(self, name):
        length = struct.unpack("<Q", self._read(name, (0, 7)))[0]
        header = json.loads(self._read(name, (8, 8 + length - 1)))
        header.pop("__metadata__", None)
        return header


@pytest.fixture(scope="module")
def release():
    source = _Release()
    try:
        source.json("model_index.json")
    except (OSError, urllib.error.URLError) as error:
        pytest.skip(f"MiniMax-H3-Image metadata is unavailable ({error}); set MINIMAX_H3_IMAGE_PATH to a local copy")
    return source


@pytest.fixture(scope="module")
def transformer_header(release):
    index = release.json("transformer/model.safetensors.index.json")["weight_map"]
    header = {}
    for shard in sorted(set(index.values())):
        header.update(release.header(f"transformer/{shard}"))
    assert set(header) == set(index), "index.json and the shard headers disagree"
    return header


def _expected_weights(objects):
    names = set()
    for weight in MiniMaxH3Model._iter_weight_objects(*objects):
        for name, _, _ in getattr(weight, "base_attrs", ()):
            names.add(name)
    return names


def test_transformer_keys_shapes_and_dtypes(release, transformer_header):
    native_config = release.json("transformer/config.json")
    assert native_config["audio_enabled"] is False
    config = {**normalize_native_transformer_config(native_config), "attn_type": "torch_sdpa", "infer_steps": 6}
    hidden, heads, head_dim = config["hidden_size"], config["num_attention_heads"], config["attention_head_dim"]
    inner, ffn, temb = heads * head_dim, config["ffn_dim"], config["time_embed_dim"]

    pre, blocks, post = MiniMaxH3ImagePreWeights(config), MiniMaxH3TransformerWeights(config), MiniMaxH3ImagePostWeights(config)
    expected = _expected_weights([pre, blocks, post])
    pdd_names = {name for name in expected if name.startswith(PDD_HEAD_PREFIX)}
    assert len(pdd_names) == 12

    converted = {}
    for key, meta in transformer_header.items():
        if key in DROPPED_KEYS:
            continue
        for name, shape in native_target_shapes(key, meta["shape"], heads, head_dim).items():
            assert name not in converted, f"{key} collides on {name}"
            converted[name] = (shape, meta["dtype"])
    assert set(converted) == expected - pdd_names, (sorted(set(converted) ^ (expected - pdd_names)))[:10]
    assert set(DROPPED_KEYS) <= set(transformer_header)

    rules = [
        (".attn.to_q.weight", (inner, hidden)),
        (".attn.to_k.weight", (inner, hidden)),
        (".attn.to_v.weight", (inner, hidden)),
        (".attn.to_out.0.weight", (hidden, inner)),
        (".attn.norm_q.weight", (head_dim,)),
        (".attn.norm_k.weight", (head_dim,)),
        (".ff.net.0.proj.weight", (2 * ffn, hidden)),
        (".ff.net.2.weight", (hidden, ffn)),
        (".adaln_proj.linear.weight", (18 * hidden, temb)),
        (".adaln_proj.linear.bias", (18 * hidden,)),
        ("norm1.weight", (hidden,)),
        ("norm2.weight", (hidden,)),
    ]
    fixed = {
        "proj_in.weight": ((hidden, config["in_channels"] * 4), "F32"),
        "proj_in.bias": ((hidden,), "F32"),
        "context_embedder.weight": ((hidden, config["text_dim"]), "BF16"),
        "context_embedder.bias": ((hidden,), "BF16"),
        "time_embedder.linear_1.weight": ((config["time_embed_hidden_dim"], config["freq_dim"]), "F32"),
        "time_embedder.linear_1.bias": ((config["time_embed_hidden_dim"],), "F32"),
        "time_embedder.linear_2.weight": ((temb, config["time_embed_hidden_dim"]), "F32"),
        "time_embedder.linear_2.bias": ((temb,), "F32"),
        "token_refiner.final_norm.weight": ((hidden,), "BF16"),
        "norm_out.norm.weight": ((hidden,), "BF16"),
        "norm_out.linear.weight": ((2 * hidden, temb), "BF16"),
        "norm_out.linear.bias": ((2 * hidden,), "BF16"),
    }
    for name, (shape, dtype) in converted.items():
        if name in fixed:
            assert (shape, dtype) == fixed[name], name
            continue
        matches = [rule_shape for suffix, rule_shape in rules if name.endswith(suffix)]
        assert matches and tuple(shape) == matches[0], (name, shape)
        assert dtype == "BF16", (name, dtype)
    assert sum(1 for name in converted if name.startswith("transformer_blocks.") and name.endswith(".norm1.weight")) == config["num_layers"] == 50
    assert sum(1 for name in converted if name.startswith("token_refiner.refiner_blocks.") and name.endswith(".norm1.weight")) == config["num_refiner_layers"] == 2


def test_pdd_heads_and_schedule(release):
    index = release.json("model_index.json")
    heads_name = index["_minimax_h3"]["pdd_fused_heads"]
    header = release.header(heads_name)
    pdd_config = release.json("pdd_config.json")
    steps = pdd_config["num_inference_steps"]
    assert steps == 6 and len(pdd_config["sigmas"]) == steps + 1
    assert {key: (meta["dtype"], tuple(meta["shape"])) for key, meta in header.items()} == {
        "video_out.weight": ("F32", (steps, 96, 5376)),
        "video_out.bias": ("F32", (steps, 96)),
    }


def test_vae_keys_and_shapes(release):
    config = release.json("vae/config.json")
    header = release.header("vae/diffusion_pytorch_model.safetensors")
    with torch.device("meta"):
        module = MiniMaxH3ImageVAE(config, device="cpu")
    state = {name: tuple(value.shape) for name, value in module.state_dict().items()}
    assert module.expected_native_keys() | set(IMAGE_VAE_DROPPED_KEYS) == set(header), sorted(set(header) ^ (module.expected_native_keys() | set(IMAGE_VAE_DROPPED_KEYS)))[:10]
    heads = config["vit_decoder_kwargs"]["heads"]
    head_dim = config["vit_decoder_kwargs"]["dim_head"]
    for key, meta in header.items():
        assert meta["dtype"] == "F32", key
        names = native_image_vae_names(key)
        for name in names:
            shape = list(meta["shape"])
            if ".attn.to_qkv." in key:
                assert shape[0] == 3 * heads * head_dim
                shape[0] //= 3
            assert tuple(shape) == state[name], (key, name, shape, state[name])
    assert len(config["latents_mean"]) == len(config["latents_std"]) == config["latent_channels"] == 24


def test_native_names_cover_every_released_key(transformer_header):
    for key in transformer_header:
        if key not in DROPPED_KEYS:
            assert native_target_names(key), key
