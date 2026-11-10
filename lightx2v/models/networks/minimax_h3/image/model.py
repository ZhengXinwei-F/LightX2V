import os

import torch
from loguru import logger
from safetensors import safe_open

from lightx2v.models.networks.base_model import SAFETENSORS_DTYPE_MAP, BaseTransformerModel
from lightx2v.models.networks.minimax_h3.image.checkpoint import (
    convert_native_tensor,
    is_native_checkpoint_key,
    load_pdd_heads,
    native_target_shapes,
    resolve_pdd_paths,
)
from lightx2v.models.networks.minimax_h3.image.infer import MiniMaxH3ImagePostInfer, MiniMaxH3ImagePreInfer
from lightx2v.models.networks.minimax_h3.image.weights import MiniMaxH3ImagePostWeights, MiniMaxH3ImagePreWeights
from lightx2v.models.networks.minimax_h3.model import MiniMaxH3Model

# Options of the video H3 network that depend on the audio branch, the video
# AdaLN timestep cache, or a converted checkpoint layout.
_UNSUPPORTED_OPTIONS = (
    "use_adaln_cache",
    "dit_quantized",
    "dit_disk_streaming",
    "shared_cpu_weights",
    "lazy_load",
    "lora_configs",
    "cpu_offload_block_contiguous",
    "offload_resident_blocks",
)


class MiniMaxH3ImageModel(MiniMaxH3Model):
    """MiniMax-H3 DiT without audio, with one output head per PDD sampling step.

    The transformer blocks, attention backends, tensor/sequence parallelism and
    block offload are the video H3 ones. Only the pre/post stages differ: no
    audio projection, and the post stage projects with the head of the current
    sampling step. The released checkpoint is read in its original layout and
    converted tensor by tensor while it is loaded.
    """

    pre_weight_class = MiniMaxH3ImagePreWeights
    post_weight_class = MiniMaxH3ImagePostWeights

    def __init__(self, model_path, config, device):
        enabled = [option for option in _UNSUPPORTED_OPTIONS if config.get(option)]
        if enabled:
            raise NotImplementedError(f"MiniMax-H3-Image does not support {enabled} yet")
        if config.get("feature_caching", "NoCaching") != "NoCaching":
            raise NotImplementedError("MiniMax-H3-Image does not support feature caching")
        # The image release has no AdaLN timestep cache: CPU offload computes
        # the modulation online, exactly as without offload.
        config.setdefault("allow_online_adaln_offload", True)
        self.pdd_model_path = model_path
        super().__init__(model_path, config, device)

    def _init_infer_class(self):
        super()._init_infer_class()
        self.pre_infer_class = MiniMaxH3ImagePreInfer
        self.post_infer_class = MiniMaxH3ImagePostInfer

    def _convert_native(self, key, tensor):
        if not is_native_checkpoint_key(key):
            # Already in LightX2V layout, e.g. a converted checkpoint.
            return [(key, tensor)]
        return convert_native_tensor(
            key,
            tensor,
            int(self.config.get("num_attention_heads", 56)),
            int(self.config.get("attention_head_dim", 128)),
        )

    def _load_safetensor_to_dict(self, file_path, unified_dtype, sensitive_layer):
        """Stream the original-layout shards, convert, then shard for TP."""
        del unified_dtype, sensitive_layer
        if os.path.splitext(file_path)[-1] != ".safetensors":
            raise ValueError(f"MiniMax-H3-Image expects the released safetensors checkpoint; got {file_path}")
        load_device = self._checkpoint_load_device()
        weight_dict = {}
        with safe_open(file_path, framework="pt", device="cpu") as source:
            self._record_checkpoint_metadata(source)
            for key in source.keys():
                for target, tensor in self._convert_native(key, source.get_tensor(key)):
                    tensor = self._select_tensor_parallel_shard(target, tensor)
                    if torch.device(load_device).type != "cpu":
                        tensor = tensor.to(load_device)
                    weight_dict[target] = tensor
        self._validate_checkpoint_devices(weight_dict, load_device)
        return weight_dict

    def _record_checkpoint_metadata(self, tensors):
        # Block-offload validation compares checkpoint shapes against the
        # LightX2V weight names, so record the converted names and shapes.
        if self._checkpoint_metadata is None or not hasattr(tensors, "get_slice"):
            return super()._record_checkpoint_metadata(tensors)
        from lightx2v_platform.ops.offload.weight_storage import TensorMetadata

        num_heads = int(self.config.get("num_attention_heads", 56))
        head_dim = int(self.config.get("attention_head_dim", 128))
        for key in tensors.keys():
            view = tensors.get_slice(key)
            shape = tuple(view.get_shape())
            dtype = SAFETENSORS_DTYPE_MAP.get(view.get_dtype()) or tensors.get_tensor(key).dtype
            targets = native_target_shapes(key, shape, num_heads, head_dim) if is_native_checkpoint_key(key) else {key: shape}
            for target, target_shape in targets.items():
                if target in self._checkpoint_metadata:
                    raise ValueError(f"Duplicate checkpoint weight: {target}")
                self._checkpoint_metadata[target] = TensorMetadata(tuple(target_shape), dtype)

    def _load_dummy_ckpt(self, unified_dtype, sensitive_layer):
        weight_dict = BaseTransformerModel._load_dummy_ckpt(self, unified_dtype, sensitive_layer)
        converted = {}
        for key, tensor in weight_dict.items():
            for target, value in self._convert_native(key, tensor):
                converted[target] = self._select_tensor_parallel_shard(target, value)
        return converted

    def _load_adapter_ckpt(self):
        """Load the per-step PDD heads that replace the transformer's output layer."""
        heads_path, _ = resolve_pdd_paths(self.pdd_model_path, self.config)
        num_steps = int(self.config.get("num_pdd_heads", self.config.get("infer_steps", 6)))
        out_features = int(self.config.get("in_channels", 24)) * int(torch.tensor(self.config.get("patch_size", (1, 2, 2))).prod())
        hidden_size = int(self.config.get("hidden_size", 5376))
        if self.config.get("dummy_model", False) and not os.path.isfile(heads_path):
            logger.info("[DummyModel] Generating random MiniMax-H3-Image PDD heads")
            return {
                **{f"pdd_proj_out.{step}.weight": torch.randn(out_features, hidden_size, dtype=torch.float32, device=self.device) for step in range(num_steps)},
                **{f"pdd_proj_out.{step}.bias": torch.randn(out_features, dtype=torch.float32, device=self.device) for step in range(num_steps)},
            }
        logger.info(f"Loading MiniMax-H3-Image PDD heads from {heads_path}")
        return load_pdd_heads(heads_path, num_steps, hidden_size, out_features, device=self._checkpoint_load_device())

    @torch.no_grad()
    def infer(self, inputs):
        prepost_offload = self.block_offload and not self.prepost_resident
        if prepost_offload and self.scheduler.step_index == 0:
            self.pre_weight.to_cuda()
            self.post_weight.to_cuda()
        output = self._infer_cond_uncond(inputs, infer_condition=True)
        self.scheduler.video_noise_pred = output.video
        if prepost_offload and self.scheduler.step_index == self.scheduler.infer_steps - 1:
            self.pre_weight.to_cpu()
            self.post_weight.to_cpu()
