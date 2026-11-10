"""End-to-end ``t2i`` and ``i2i`` runs of the registered runner on a tiny random release (CPU)."""

import numpy as np
import pytest
from fixtures import build_tiny_runner, run_tiny_pipeline, write_reference_images


def _array(image):
    return np.asarray(image).astype(np.int16)


def test_t2i_end_to_end(tiny_release, tmp_path):
    image = run_tiny_pipeline(tiny_release, "t2i", str(tmp_path))
    assert image.size == (128, 96) and image.mode == "RGB"
    assert _array(image).std() > 0
    again = run_tiny_pipeline(tiny_release, "t2i", str(tmp_path))
    assert np.array_equal(_array(image), _array(again)), "same seed must reproduce the image"
    other = run_tiny_pipeline(tiny_release, "t2i", str(tmp_path), seed=8)
    assert not np.array_equal(_array(image), _array(other))


def test_i2i_end_to_end_with_two_references(tiny_release, tmp_path):
    runner, stub = build_tiny_runner(tiny_release, "i2i")
    paths = write_reference_images(str(tmp_path))
    request = {"task": "i2i", "prompt": "Keep <Picture 1> and add <Picture 2>", "seed": 3, "image_path": ",".join(paths), "return_result_tensor": True}
    result = runner.run_request(runner.prepare_request(request))
    image = result["images"][0]
    assert image.size == (128, 96)
    assert stub.calls[-1]["num_images"] == 2
    # The references keep their own size (inside the configured bounds).
    assert stub.calls[-1]["image_sizes"] == [(80, 72), (40, 120)]


def test_i2i_reference_rows_follow_each_reference_grid(tiny_release, tmp_path):
    runner, _ = build_tiny_runner(tiny_release, "i2i")
    paths = write_reference_images(str(tmp_path))
    runner.input_info = runner.prepare_request({"task": "i2i", "prompt": "edit", "seed": 1, "image_path": paths})
    runner.inputs = runner.run_input_encoder()
    # (width x height) 80x72 -> 4x4 latent cells after the patch crop; 40x120 -> 6x2.
    assert [tuple(latent.shape[-2:]) for latent in runner.reference_latents] == [(4, 4), (6, 2)]
    runner.init_run()
    assert runner.scheduler.num_condition_video_rows == 2 * 2 + 1 * 3
    tags = runner.scheduler.layout_cpu.token_tags
    assert int((tags == 0).sum()) == runner.scheduler.layout_cpu.video_indices.numel() + 2 * (6 + 2)
    runner.end_run()


def test_rng_order_only_changes_edits(tiny_release, tmp_path):
    workdir = str(tmp_path)
    t2i_a = run_tiny_pipeline(tiny_release, "t2i", workdir)
    t2i_b = run_tiny_pipeline(tiny_release, "t2i", workdir, overrides={"h3_image_rng_order": "sglang"})
    assert np.array_equal(_array(t2i_a), _array(t2i_b))
    edit_a = run_tiny_pipeline(tiny_release, "i2i", workdir)
    edit_b = run_tiny_pipeline(tiny_release, "i2i", workdir, overrides={"h3_image_rng_order": "sglang"})
    assert not np.array_equal(_array(edit_a), _array(edit_b))


def test_small_request_is_generated_larger_and_resized(tiny_release, tmp_path):
    runner, _ = build_tiny_runner(tiny_release, "t2i", overrides={"min_generation_area": 160 * 160, "size": [64, 96]})
    result = runner.run_request(runner.prepare_request({"task": "t2i", "prompt": "a lighthouse", "seed": 0, "return_result_tensor": True}))
    assert result["images"][0].size == (96, 64)


def test_saves_png(tiny_release, tmp_path):
    runner, _ = build_tiny_runner(tiny_release, "t2i")
    path = tmp_path / "out" / "t2i.png"
    result = runner.run_request(runner.prepare_request({"task": "t2i", "prompt": "a lighthouse", "seed": 0, "save_result_path": str(path)}))
    assert result == {"images": None}
    assert path.is_file()


@pytest.mark.parametrize("bad", [{"infer_steps": 8}, {"enable_cfg": True}, {"h3_image_rng_order": "other"}])
def test_rejects_unsupported_configs(tiny_release, bad):
    with pytest.raises((ValueError, NotImplementedError)):
        build_tiny_runner(tiny_release, "t2i", overrides=bad)


def test_rejects_negative_prompt_and_missing_references(tiny_release):
    runner, _ = build_tiny_runner(tiny_release, "i2i")
    with pytest.raises(ValueError):
        runner.prepare_request({"task": "t2i", "prompt": "x", "negative_prompt": "blurry"})
    with pytest.raises(ValueError):
        runner.run_request(runner.prepare_request({"task": "i2i", "prompt": "x"}))
