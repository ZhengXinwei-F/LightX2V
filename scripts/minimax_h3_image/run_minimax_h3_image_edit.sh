#!/bin/bash

# set paths first; also set text_encoder_path in the JSON config to the
# text_encoder/ folder of MiniMaxAI/MiniMax-H3 (not shipped with the image release)
lightx2v_path=/path/to/LightX2V
model_path=/path/to/MiniMax-H3-Image

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source "${lightx2v_path}/scripts/base/base.sh"
export DTYPE=BF16

# Image editing uses task i2i. Pass up to 16 comma-separated references; their
# order defines <Picture 1>, <Picture 2>, ... in the prompt.
python -m lightx2v.infer \
  --model_cls minimax_h3_image \
  --task i2i \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/minimax_h3_image/minimax_h3_image.json" \
  --image_path "${model_path}/examples/reference1.png" \
  --prompt "$(cat "${model_path}/examples/revised_prompt.edit.txt")" \
  --size 4096 4096 \
  --save_result_path "${lightx2v_path}/save_results/output_lightx2v_minimax_h3_image_edit.png" \
  --seed 1234
