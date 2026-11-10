"""Tiny random-weight MiniMax-H3-Image releases for CPU tests.

The fixture writes a directory in the layout of ``MiniMaxAI/MiniMax-H3-Image``:
an original-layout transformer split over two shards, ``pdd_fused_heads``, a
``pdd_config.json`` with the released sigma grid, and an original-layout image
VAE. Only the dimensions are shrunk; the checkpoint-tied ones (24 latent
channels, the ``[1, 2, 2]`` patch, 16x VAE compression, 3-way AdaLN table) are
kept. The Qwen3-VL conditioner is replaced by :class:`StubImageTextEncoder`,
which builds the real image presentation layout (labels, vision blocks, tags)
with deterministic embeddings.
"""

import json
import os

import torch
from safetensors.torch import save_file

RELEASE_SIGMAS = [1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0]

TINY_TRANSFORMER_CONFIG = {
    "_class_name": "MiniMaxH3DiTModel",
    "hidden_size": 64,
    "num_layers": 2,
    "token_refiner_num_layers": 2,
    "num_attention_heads": 2,
    "attention_head_dim": 32,
    "ffn_hidden_size": 128,
    "latents_dim": 24,
    "audio_latents_dim": 32,
    "patch_size": [1, 2, 2],
    "text_dim": 48,
    "timestep_input_dim": 16,
    "time_embed_hidden_size": 64,
    "time_embed_dim": 32,
    "adaln_out_features": 6 * 3 * 64,
    "final_adaln_out_features": 2 * 64,
    "rope_inv_freq_len": 4,
    "norm_eps": 1e-05,
    "qk_norm_eps": 1e-05,
    "final_norm_eps": 1e-05,
    "audio_enabled": False,
}

TINY_VAE_CONFIG = {
    "_class_name": "MiniMaxH3ImageVAE",
    "latent_channels": 24,
    "block_out_channels": [32, 32, 32, 32],
    "layers_per_block": 1,
    "spatial_downsample_factors": [2, 2, 2, 2],
    "temporal_downsample_factors": [1, 2, 2, 1],
    "latents_mean": [0.1 * (index % 5) - 0.2 for index in range(24)],
    "latents_std": [1.0 + 0.05 * index for index in range(24)],
    "vae_decoder_tiling": 1,
    "vae_encoder_tiling": 1,
    "vae_tile_overlap_min": 32,
    "vae_tile_size": 64,
    "vit_decoder_2d": True,
    "vit_decoder_kwargs": {
        "dim_head": 32,
        "ffn_activation_fn": "silu",
        "ffn_use_gated": True,
        "heads": 2,
        "norm_affine": True,
        "norm_type": "rms_norm",
        "num_layers": 2,
        "qk_norm_affine": False,
        "qk_norm_type": "rms_norm",
        "rope_dim_ratio": 0.5,
        "rope_theta": 100.0,
    },
}


def tiny_native_transformer_state(config=TINY_TRANSFORMER_CONFIG, seed=0):
    """Random tensors under the original MiniMax-H3 DiT key names and dtypes."""
    generator = torch.Generator().manual_seed(seed)
    hidden = config["hidden_size"]
    inner = config["num_attention_heads"] * config["attention_head_dim"]
    head_dim = config["attention_head_dim"]
    ffn = config["ffn_hidden_size"]
    temb = config["time_embed_dim"]
    patch_dim = config["latents_dim"] * 4

    def rand(*shape, dtype=torch.bfloat16, scale=0.05):
        return (torch.randn(shape, generator=generator) * scale).to(dtype)

    def norm(size):
        return (1.0 + torch.randn(size, generator=generator) * 0.05).to(torch.bfloat16)

    state = {
        "video_patch_proj.weight": rand(hidden, patch_dim, dtype=torch.float32, scale=0.1),
        "video_patch_proj.bias": rand(hidden, dtype=torch.float32),
        "condition_proj.weight": rand(hidden, config["text_dim"], scale=0.1),
        "condition_proj.bias": rand(hidden),
        "time_embedder.proj_in.weight": rand(config["time_embed_hidden_size"], config["timestep_input_dim"], dtype=torch.float32, scale=0.2),
        "time_embedder.proj_in.bias": rand(config["time_embed_hidden_size"], dtype=torch.float32),
        "time_embedder.proj_out.weight": rand(temb, config["time_embed_hidden_size"], dtype=torch.float32, scale=0.2),
        "time_embedder.proj_out.bias": rand(temb, dtype=torch.float32),
        "token_refiner.final_norm.weight": norm(hidden),
        "final_layer.norm.weight": norm(hidden),
        "final_layer.adaln_proj.linear.weight": rand(2 * hidden, temb),
        "final_layer.adaln_proj.linear.bias": rand(2 * hidden),
        "final_layer.video_out.weight": rand(patch_dim, hidden, dtype=torch.float32),
        "final_layer.video_out.bias": rand(patch_dim, dtype=torch.float32),
        "rope.inv_freq": 1.0 / (10000.0 ** (torch.arange(0, 2 * config["rope_inv_freq_len"], 2, dtype=torch.float32) / (2 * config["rope_inv_freq_len"]))),
    }
    blocks = [("blocks", config["num_layers"], True), ("token_refiner.blocks", config["token_refiner_num_layers"], False)]
    for prefix, count, has_adaln in blocks:
        for index in range(count):
            p = f"{prefix}.{index}"
            state[f"{p}.norm1.weight"] = norm(hidden)
            state[f"{p}.norm2.weight"] = norm(hidden)
            state[f"{p}.attn.qkv_proj.weight"] = rand(3 * inner, hidden, scale=0.1)
            state[f"{p}.attn.q_norm.weight"] = norm(head_dim)
            state[f"{p}.attn.k_norm.weight"] = norm(head_dim)
            state[f"{p}.attn.out_proj.weight"] = rand(hidden, inner, scale=0.1)
            state[f"{p}.mlp.fc1.weight"] = rand(2 * ffn, hidden, scale=0.1)
            state[f"{p}.mlp.fc2.weight"] = rand(hidden, ffn, scale=0.1)
            if has_adaln:
                state[f"{p}.adaln_proj.linear.weight"] = rand(6 * 3 * hidden, temb, scale=0.1)
                state[f"{p}.adaln_proj.linear.bias"] = rand(6 * 3 * hidden)
    return state


def tiny_pdd_heads(config=TINY_TRANSFORMER_CONFIG, steps=6, seed=1):
    generator = torch.Generator().manual_seed(seed)
    patch_dim = config["latents_dim"] * 4
    return {
        "video_out.weight": torch.randn((steps, patch_dim, config["hidden_size"]), generator=generator) * 0.05,
        "video_out.bias": torch.randn((steps, patch_dim), generator=generator) * 0.01,
    }


def tiny_native_vae_state(config=TINY_VAE_CONFIG, seed=2):
    """Random tensors under the original image-VAE key names (all FP32)."""
    from lightx2v.models.video_encoders.hf.minimax_h3.image_vae import MiniMaxH3ImageVAE

    torch.manual_seed(seed)
    module = MiniMaxH3ImageVAE(config, device="cpu", decode_dtype=torch.float32)
    heads = config["vit_decoder_kwargs"]["heads"]
    head_dim = config["vit_decoder_kwargs"]["dim_head"]
    state = {}
    with torch.no_grad():
        for name, tensor in module.state_dict().items():
            # Layer scales are zero-initialized; give them weight so the
            # decoder blocks contribute in tests.
            if name.endswith((".scale1", ".scale2")):
                tensor = torch.randn_like(tensor) * 0.5
            elif name.endswith("register_tokens"):
                tensor = torch.randn_like(tensor)
            state[name] = tensor.float().clone()
    native = {"decoder.mask_token": torch.zeros(1, 1, heads * head_dim)}
    for name, tensor in state.items():
        if ".attn.to_q." in name:
            parts = [state[name.replace(".to_q.", f".to_{kind}.")] for kind in "qkv"]
            stacked = torch.stack([part.reshape(heads, head_dim, *part.shape[1:]) for part in parts], dim=1)
            native[name.replace(".to_q.", ".to_qkv.")] = stacked.reshape(3 * heads * head_dim, *parts[0].shape[1:]).contiguous()
            continue
        if ".attn.to_k." in name or ".attn.to_v." in name:
            continue
        source = name
        if source.startswith("encoder.down_blocks."):
            level, rest = source.removeprefix("encoder.down_blocks.").split(".", 1)
            rest = rest.replace("resnets.", "block.", 1).replace("conv_shortcut.", "nin_shortcut.", 1).replace("downsamplers.0.", "downsample.", 1)
            source = f"encoder.down.{level}.{rest}"
        source = source.replace("decoder.proj_in.", "decoder.x_embedder.").replace(".attn.to_out.0.", ".attn.to_out.")
        if ".ff.net.0.proj." in source:
            up, gate = tensor.chunk(2, dim=0)
            tensor = torch.cat((gate, up), dim=0)
            source = source.replace(".ff.net.0.proj.", ".ff.w1.")
        source = source.replace(".ff.net.2.", ".ff.w2.")
        native[source] = tensor.contiguous()
    return native


def build_tiny_release(root, *, transformer_config=TINY_TRANSFORMER_CONFIG, vae_config=TINY_VAE_CONFIG):
    """Write a tiny MiniMax-H3-Image release under ``root`` and return its path."""
    root = os.path.join(root, "MiniMax-H3-Image-tiny")
    os.makedirs(os.path.join(root, "transformer"), exist_ok=True)
    os.makedirs(os.path.join(root, "vae"), exist_ok=True)
    with open(os.path.join(root, "model_index.json"), "w") as handle:
        json.dump({"_class_name": "MiniMaxH3ImagePipeline", "_minimax_h3": {"partition": "image", "pdd_fused_heads": "pdd_fused_heads.safetensors", "tasks": ["t2i", "edit"]}}, handle)
    with open(os.path.join(root, "pdd_config.json"), "w") as handle:
        json.dump({"num_inference_steps": 6, "sigmas": RELEASE_SIGMAS}, handle)
    with open(os.path.join(root, "transformer", "config.json"), "w") as handle:
        json.dump(transformer_config, handle)
    state = tiny_native_transformer_state(transformer_config)
    keys = sorted(state)
    half = len(keys) // 2
    save_file({key: state[key] for key in keys[:half]}, os.path.join(root, "transformer", "model-00001-of-00002.safetensors"))
    save_file({key: state[key] for key in keys[half:]}, os.path.join(root, "transformer", "model-00002-of-00002.safetensors"))
    save_file(tiny_pdd_heads(transformer_config), os.path.join(root, "pdd_fused_heads.safetensors"))
    with open(os.path.join(root, "vae", "config.json"), "w") as handle:
        json.dump(vae_config, handle)
    save_file(tiny_native_vae_state(vae_config), os.path.join(root, "vae", "diffusion_pytorch_model.safetensors"))
    return root


class _StubTokenizer:
    """Whitespace tokenizer with the Qwen vision special tokens."""

    special = {"<|vision_start|>": 1, "<|image_pad|>": 2, "<|vision_end|>": 3}

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [10 + (sum(map(ord, word)) % 1000) for word in text.split()]}

    def convert_tokens_to_ids(self, token):
        return self.special[token]


class StubImageTextEncoder:
    """Stands in for the 50-layer Qwen3-VL conditioner on CPU.

    The presentation (token ids and modality tags) is built by the real
    ``build_image_edit_presentation``; the embeddings are a deterministic
    function of the token ids. Each reference contributes ``tokens_per_image``
    vision rows.
    """

    def __init__(self, text_dim, tokens_per_image=6):
        self.text_dim = text_dim
        self.tokens_per_image = tokens_per_image
        self.tokenizer = _StubTokenizer()
        self.calls = []

    def infer(self, prompt, image_list=None, references=None):
        from lightx2v.models.input_encoders.hf.minimax_h3.qwen3vl_image import build_image_edit_presentation

        images = image_list or []
        token_ids, token_tags = build_image_edit_presentation(self.tokenizer, prompt, [self.tokens_per_image] * len(images))
        self.calls.append({"prompt": prompt, "num_images": len(images), "image_sizes": [image.size for image in images]})
        ids = torch.tensor(token_ids, dtype=torch.float32)
        basis = torch.arange(self.text_dim, dtype=torch.float32)
        embeds = torch.sin(ids[:, None] * 0.37 + basis[None] * 0.11)
        return {"prompt_embeds": embeds.to(torch.bfloat16), "text_token_tags": torch.tensor(token_tags, dtype=torch.long)}


def tiny_runner_config(root, task, overrides=None):
    from lightx2v.utils.set_config import build_startup_config

    config_path = os.path.join(root, f"runner_{task}.json")
    deployment = {
        "attn_type": "torch_sdpa",
        "refiner_attn_type": "torch_sdpa",
        "vae_attn_type": "torch_sdpa",
        "rms_type": "torch_native",
        "rope_type": "torch_real_rope",
        "vae_decode_dtype": "fp32",
        "min_generation_area": 1024,
        "size": [96, 128],
        "text_encoder_load_on_init": False,
        # LightX2V keeps host weights pinned and moves them to AI_DEVICE on
        # to_cuda(); model-granularity offload exercises exactly that on CPU.
        "cpu_offload": True,
        "offload_granularity": "model",
        # Keep the tiny references at their own size on CPU.
        "reference_vae_short_edge": [32, 512],
        "reference_vision_short_edge": [32, 512],
    }
    deployment.update(overrides or {})
    with open(config_path, "w") as handle:
        json.dump(deployment, handle)
    return build_startup_config({"model_cls": "minimax_h3_image", "model_path": root, "task": task, "config_json": config_path})


def build_tiny_runner(root, task, overrides=None, monkeypatch=None):
    """Build the registered runner with the stub conditioner."""
    from lightx2v.models.runners.minimax_h3.minimax_h3_image_runner import MiniMaxH3ImageRunner
    from lightx2v.models.runners.runner_factory import build_runner

    config = tiny_runner_config(root, task, overrides)
    stub = StubImageTextEncoder(TINY_TRANSFORMER_CONFIG["text_dim"])
    original = MiniMaxH3ImageRunner.load_text_encoder
    MiniMaxH3ImageRunner.load_text_encoder = lambda self: [stub]
    try:
        runner = build_runner(config)
    finally:
        MiniMaxH3ImageRunner.load_text_encoder = original
    return runner, stub


def write_reference_images(root, sizes=((80, 72), (40, 120))):
    """Write deterministic RGB references (``(width, height)`` each) and return their paths."""
    from PIL import Image

    paths = []
    for index, (width, height) in enumerate(sizes):
        yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        pixels = torch.stack(((xx * 7 + index * 40) % 256, (yy * 5) % 256, ((xx + yy) * 3) % 256), dim=-1).to(torch.uint8)
        path = os.path.join(root, f"reference_{index}.png")
        Image.fromarray(pixels.numpy()).save(path)
        paths.append(path)
    return paths


def run_tiny_pipeline(root, task, workdir, overrides=None, seed=7):
    runner, stub = build_tiny_runner(root, task, overrides)
    request = {"task": task, "prompt": "a red fox sitting in the snow at dusk", "seed": seed, "return_result_tensor": True}
    if task == "i2i":
        request["image_path"] = ",".join(write_reference_images(workdir))
        request["prompt"] = "Keep everything in <Picture 1> and put the object of <Picture 2> on the left"
    input_info = runner.prepare_request(request)
    result = runner.run_request(input_info)
    return result["images"][0]
