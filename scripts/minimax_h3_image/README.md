# MiniMax-H3-Image

[MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image) generates images of about 16 megapixels from text (text-to-image) and edits up to 16 reference images (image editing) with one checkpoint. Its diffusion transformer (DiT) is the MiniMax-H3 DiT without the audio branch; it is distilled to a fixed schedule of 6 sampling steps with Parallel Decoding Distillation (PDD), which keeps one output head per step, and it runs without classifier-free guidance (CFG).

In LightX2V the model is `--model_cls minimax_h3_image`. It reuses the native MiniMax-H3 transformer blocks, attention backends, offload paths and Qwen3-VL text encoder.

## Weights

| Component | Source |
| --- | --- |
| DiT, PDD heads, image VAE, tokenizer, processor | [MiniMaxAI/MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image), used as `--model_path` |
| Text encoder | The unchanged MiniMax-H3 Qwen3-VL text encoder: `text_encoder/` of the image repository, or of [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) through `text_encoder_path` in the JSON config |

```bash
# The original-layout components; transformer_diffusers/, vae_diffusers/ and text_encoder/ are not needed.
hf download MiniMaxAI/MiniMax-H3-Image --exclude "transformer_diffusers/*" "vae_diffusers/*" "text_encoder/*" --local-dir /path/to/MiniMax-H3-Image
# The text encoder, if no MiniMax-H3 copy exists yet.
hf download MiniMaxAI/MiniMax-H3 --include "text_encoder/*" --local-dir /path/to/MiniMax-H3
```

Then add `"text_encoder_path": "/path/to/MiniMax-H3/text_encoder"` to the JSON config. Without `text_encoder_path` the text encoder is read from `<model_path>/text_encoder`.

The release is read in its original layout: the transformer shards, `pdd_fused_heads.safetensors`, `pdd_config.json` and `vae/` are converted while loading, so no conversion step is needed. The Diffusers-layout `transformer_diffusers/` folder uses the LightX2V weight names directly and can be selected with `"dit_original_ckpt": "<model_path>/transformer_diffusers"`. The step count and the sigma grid come from `pdd_config.json`; `infer_steps` other than 6 is rejected.

## Tasks

| Task | Request inputs | Notes |
| --- | --- | --- |
| `t2i` | `prompt`, `size` | |
| `i2i` (image editing) | `prompt`, `size`, `image_path` | 1 to 16 comma-separated references; their order defines `<Picture 1>`, `<Picture 2>`, ... in the prompt |

`size` is `[height, width]` in pixels (`--size HEIGHT WIDTH`) and is the size of the returned image. A request below `min_generation_area` pixels (default 4194304, i.e. 2048x2048) is generated on a larger canvas of the same aspect ratio and resized back with LANCZOS. The model gives its best results at about 16 MP:

| Aspect ratio | `--size` (height width) |
| --- | --- |
| 1:1 | `4096 4096` |
| 4:3 / 3:4 | `3552 4736` / `4736 3552` |
| 3:2 / 2:3 | `3360 5024` / `5024 3360` |
| 16:9 / 9:16 | `3072 5472` / `5472 3072` |
| 21:9 | `2688 6272` |

The prompt is used verbatim and truncated to 4096 tokens; the model expects a long, complete English description. `negative_prompt` is not supported.

## Offline inference

Set `lightx2v_path` and `model_path` in the scripts (and `text_encoder_path` in the JSON config when the text encoder lives outside `model_path`), then run from the repository root:

```bash
bash scripts/minimax_h3_image/run_minimax_h3_image_t2i.sh    # examples/request.t2i.json, 4736x3552, seed 42
bash scripts/minimax_h3_image/run_minimax_h3_image_edit.sh   # examples/reference1.png, 4096x4096, seed 1234
```

| Configuration under `configs/minimax_h3_image/` | Mode |
| --- | --- |
| `minimax_h3_image.json` | One GPU, BF16 DiT resident, text encoder offloaded between requests |
| `minimax_h3_image_block_offload.json` | One GPU, DiT and text encoder block offload |

The configs use dense FlashAttention 3 and SGL kernels; choose operators that match the installed kernels.

## Matching the reference implementations

| Option | Default | Alternative |
| --- | --- | --- |
| `h3_image_step_update` | `velocity`: `x + (sigma - sigma_next) * v`, the form the PDD heads were distilled with (Diffusers) | `reference_blend`: the `x_t`/`x0` blend of the H3 video scheduler (SGLang) |
| `h3_image_rng_order` | `diffusers`: VAE posterior samples of the references, then the target noise, then the reference noise | `sglang`: target noise, reference noise, then the posterior samples |
| `generator_device` | `cpu` | Same seed gives the same initial noise on every machine |
| `vae_decode_dtype` | `bf16` | `fp32` |

Both step updates are the same Euler step up to float32 rounding. The random-number order only affects editing.

## Tests

The CPU tests use tiny random-weight releases and need no GPU or weights:

```bash
pytest tests/minimax_h3_image
```

`test_parity_diffusers.py` compares against Diffusers (with MiniMax-H3-Image, `huggingface/diffusers` main); `test_release_checkpoint_keys.py` reads only the safetensors headers of the release, either from Hugging Face or from `MINIMAX_H3_IMAGE_PATH`.
