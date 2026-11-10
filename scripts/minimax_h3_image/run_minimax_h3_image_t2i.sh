#!/bin/bash

# set paths first. The text encoder is read from ${model_path}/text_encoder; to use
# the text_encoder/ folder of an existing MiniMax-H3 download instead, add
# "text_encoder_path": "/path/to/MiniMax-H3/text_encoder" to the JSON config.
lightx2v_path=/path/to/LightX2V
model_path=/path/to/MiniMax-H3-Image

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source "${lightx2v_path}/scripts/base/base.sh"
export DTYPE=BF16

# --size is HEIGHT WIDTH. The prompt is examples/request.t2i.json of the release.
python -m lightx2v.infer \
  --model_cls minimax_h3_image \
  --task t2i \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/minimax_h3_image/minimax_h3_image.json" \
  --prompt "$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["prompt"])' "${model_path}/examples/request.t2i.json")" \
  --size 3552 4736 \
  --save_result_path "${lightx2v_path}/save_results/output_lightx2v_minimax_h3_image_t2i.png" \
  --seed 42
