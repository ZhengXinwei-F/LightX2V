"""The image release's Qwen3-VL presentation (no real tokenizer or weights needed)."""

from PIL import Image
from fixtures import _StubTokenizer

from lightx2v.models.input_encoders.hf.minimax_h3.qwen3vl_image import build_image_edit_presentation, resize_vision_reference


class _RecordingTokenizer(_StubTokenizer):
    def __init__(self):
        self.texts = []

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        self.texts.append(text)
        return super().__call__(text, add_special_tokens=add_special_tokens)


def test_edit_presentation_layout_and_tags():
    tokenizer = _RecordingTokenizer()
    ids, tags = build_image_edit_presentation(tokenizer, "make <Picture 2> blue", [3, 2])
    # Labels carry no angle brackets, unlike the prompt's references.
    assert tokenizer.texts == ["Picture 1: ", "Picture 2: ", "make <Picture 2> blue"]
    label = tokenizer("Picture 1: ")["input_ids"]
    first_block = [1, 2, 2, 2, 3]
    assert ids[: len(label) + 5] == label + first_block
    assert tags[: len(label)] == [1] * len(label) and tags[len(label) : len(label) + 5] == [0] * 5
    assert ids.count(2) == 5 and tags.count(0) == 5 + 4
    assert tags[-3:] == [1, 1, 1]


def test_prompt_and_labels_truncate_independently():
    tokenizer = _StubTokenizer()
    prompt = " ".join(f"w{index}" for index in range(50))
    ids, tags = build_image_edit_presentation(tokenizer, prompt, [1], max_tokens=4)
    label = tokenizer("Picture 1: ")["input_ids"][:4]
    assert ids == label + [1, 2, 3] + tokenizer(prompt)["input_ids"][:4]
    ids, tags = build_image_edit_presentation(tokenizer, prompt, [], max_tokens=7)
    assert len(ids) == 7 and set(tags) == {1}


def test_vision_copy_short_edge_is_fixed_to_1024():
    assert resize_vision_reference(Image.new("RGB", (1600, 1024))).size == (1600, 1024)
    assert resize_vision_reference(Image.new("RGB", (640, 480))).size == (1376, 1024)
    assert resize_vision_reference(Image.new("RGB", (3000, 2000))).size == (1536, 1024)
