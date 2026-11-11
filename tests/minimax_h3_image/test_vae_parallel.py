"""Tile-parallel MiniMax-H3-Image VAE decode over gloo matches one process.

Two CPU ranks first decode with parallelism turned off, then with it on (the
default once a multi-rank group exists). Each rank checks that it decoded only
its own balanced block of tiles and that both images are bitwise equal; the
parent then compares every rank's image with its own single-process decode.
"""

import json
import os
import socket
import sys

sys.path.insert(0, os.path.dirname(__file__))

import cpu_platform  # noqa: E402,F401  (must precede lightx2v in spawned ranks)
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

WORLD_SIZE = 2
# Latent sizes in VAE cells (16 px). With 64 px tiles and >= 32 px overlap:
# 12x16 -> 5x7 = 35 tiles and 4x12 -> 1x5 = 5 tiles (both odd), 4x4 -> 1 tile.
LATENT_SIZES = ((12, 16), (4, 12), (4, 4))
DECODE_DTYPES = ("fp32", "bf16")


def _load(vae_dir, dtype_name):
    from lightx2v.models.video_encoders.hf.minimax_h3.image_vae import MiniMaxH3ImageVAE

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[dtype_name]
    return MiniMaxH3ImageVAE.from_pretrained(vae_dir, device="cpu", decode_dtype=dtype)


def _latents(latent_height, latent_width):
    generator = torch.Generator().manual_seed(1000 * latent_height + latent_width)
    return torch.randn((1, 24, latent_height, latent_width), generator=generator)


def _worker(rank, port, vae_dir, out_dir):
    from lightx2v.models.video_encoders.hf.minimax_h3.video_vae import MiniMaxH3VideoVAE

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=WORLD_SIZE)
    try:
        results = {}
        for dtype_name in DECODE_DTYPES:
            vae = _load(vae_dir, dtype_name)
            assert vae._use_decode_parallel()
            vae.decode_parallel = False
            serial = {size: vae.decode(_latents(*size)) for size in LATENT_SIZES}
            vae.decode_parallel = True
            calls = []
            decode_tile = vae._decode_tile
            vae._decode_tile = lambda tile: calls.append(1) or decode_tile(tile)
            for size in LATENT_SIZES:
                calls.clear()
                image = vae.decode(_latents(*size))
                num_tiles = 1
                for length in size:
                    num_tiles *= len(vae._split_tiles(16 * length, vae.tile_sample_size, vae.tile_sample_min_overlap)[0])
                expected = 1 if num_tiles == 1 else MiniMaxH3VideoVAE._balanced_task_counts(num_tiles, WORLD_SIZE)[rank]
                assert len(calls) == expected, (dtype_name, size, rank, len(calls), expected)
                assert torch.equal(image, serial[size]), (dtype_name, size, rank, (image - serial[size]).abs().max())
                results[(dtype_name, size)] = image
        torch.save(results, os.path.join(out_dir, f"rank{rank}.pt"))
    finally:
        dist.destroy_process_group()


def test_parallel_decode_matches_single_process(tmp_path, monkeypatch):
    from fixtures import TINY_VAE_CONFIG, tiny_native_vae_state

    vae_dir = tmp_path / "vae"
    vae_dir.mkdir()
    (vae_dir / "config.json").write_text(json.dumps(TINY_VAE_CONFIG))
    save_file(tiny_native_vae_state(), str(vae_dir / "diffusion_pytorch_model.safetensors"))

    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port, str(vae_dir), str(tmp_path)), nprocs=WORLD_SIZE, join=True)

    # The ranks run single-threaded; CPU kernels may round differently with more threads.
    num_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for dtype_name in DECODE_DTYPES:
            vae = _load(str(vae_dir), dtype_name)
            for size in LATENT_SIZES:
                image = vae.decode(_latents(*size))
                assert image.shape[-2:] == (16 * size[0], 16 * size[1])
                for rank in range(WORLD_SIZE):
                    assert torch.equal(torch.load(os.path.join(tmp_path, f"rank{rank}.pt"))[(dtype_name, size)], image), (dtype_name, size, rank)
    finally:
        torch.set_num_threads(num_threads)
