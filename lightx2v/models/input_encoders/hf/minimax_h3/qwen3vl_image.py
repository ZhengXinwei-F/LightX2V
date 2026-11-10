"""MiniMax-H3-Image conditioning with the unchanged MiniMax-H3 Qwen3-VL encoder.

Only the presentation differs from the video release:

* text-to-image: the prompt verbatim (no chat template, no special tokens),
  truncated to 4096 tokens;
* image editing: per reference ``Picture i: `` (no angle brackets) followed by
  its vision block, then the prompt verbatim; every label and the prompt are
  truncated to 4096 tokens. The Qwen3-VL copy of each reference is resized so
  that its short edge is 1024 px (sides on 32 px multiples) when it is outside
  the configured bounds.

The vision-block rows are tagged as image rows (tag 0) for the DiT's AdaLN.
"""

import torch
from PIL import Image

from lightx2v.models.input_encoders.hf.minimax_h3.qwen3vl import MINIMAX_H3_TEXT_TAG, MiniMaxH3Qwen3VLTextEncoder
from lightx2v.models.networks.minimax_h3.image.packing import IMAGE_TAG, MAX_SEQUENCE_TOKENS, REFERENCE_VISION_SHORT_EDGE, bound_short_edge


def build_image_edit_presentation(tokenizer, prompt, image_token_counts, max_tokens=MAX_SEQUENCE_TOKENS):
    """Return ``(token_ids, token_tags)`` of the edit presentation."""
    vision_start = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    image_pad = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    vision_end = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    token_ids, token_tags = [], []
    for index, count in enumerate(image_token_counts, start=1):
        label = tokenizer(f"Picture {index}: ", add_special_tokens=False)["input_ids"][:max_tokens]
        block = [vision_start] + [image_pad] * int(count) + [vision_end]
        token_ids += label + block
        token_tags += [MINIMAX_H3_TEXT_TAG] * len(label) + [IMAGE_TAG] * len(block)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"][:max_tokens]
    token_ids += prompt_ids
    token_tags += [MINIMAX_H3_TEXT_TAG] * len(prompt_ids)
    return token_ids, token_tags


def resize_vision_reference(image, short_edge_bounds=REFERENCE_VISION_SHORT_EDGE):
    """The Qwen3-VL copy of an edit reference (LANCZOS, kept as is inside the bounds)."""
    size = bound_short_edge(*image.size, *short_edge_bounds)
    if size == image.size:
        return image
    return image.resize(size, Image.Resampling.LANCZOS)


class MiniMaxH3ImageQwen3VLTextEncoder(MiniMaxH3Qwen3VLTextEncoder):
    """The MiniMax-H3 conditioner with the image release's presentation."""

    def __init__(self, config):
        self.max_sequence_tokens = int(config.get("max_sequence_tokens", MAX_SEQUENCE_TOKENS))
        bounds = config.get("reference_vision_short_edge", REFERENCE_VISION_SHORT_EDGE)
        self.vision_short_edge = (int(bounds[0]), int(bounds[1]))
        super().__init__(config)

    def _prepare_t2av_input_ids(self, prompt, device):
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("MiniMax-H3-Image expects one non-empty prompt string")
        token_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"][: self.max_sequence_tokens]
        if not token_ids:
            raise ValueError("MiniMax-H3-Image prompt must encode to at least one token")
        return torch.tensor(token_ids, dtype=torch.long, device=device)

    def _prepare_keyframe_inputs(self, prompt, images):
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("MiniMax-H3-Image expects one non-empty prompt string")
        processor = self.load_processor()
        images = [resize_vision_reference(image, self.vision_short_edge) for image in images]
        vision = processor.image_processor(images=images, return_tensors="pt")
        image_grid_thw = vision["image_grid_thw"]
        merge_unit = processor.image_processor.merge_size**2
        counts = [int(grid.prod()) // merge_unit for grid in image_grid_thw]
        token_ids, token_tags = build_image_edit_presentation(self.tokenizer, prompt, counts, self.max_sequence_tokens)
        return token_ids, token_tags, vision["pixel_values"], image_grid_thw, None, None
