from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.models.networks.minimax_h3.image.checkpoint import PDD_HEAD_PREFIX
from lightx2v.models.networks.minimax_h3.weights import MiniMaxH3PreWeights
from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER, RMS_WEIGHT_REGISTER


class MiniMaxH3ImagePreWeights(MiniMaxH3PreWeights):
    """H3 input projections, time MLP and text refiner, without the audio projection."""

    def __init__(self, config):
        super().__init__(config)
        self._modules.pop("audio_proj_in")
        del self.audio_proj_in


class MiniMaxH3ImagePostWeights(WeightModule):
    """Shared final norm plus one FP32 output head per PDD sampling step."""

    def __init__(self, config):
        super().__init__()
        self.add_module(
            "norm_out",
            RMS_WEIGHT_REGISTER[config.get("rms_type", "torch_native")]("norm_out.norm.weight", eps=float(config.get("final_norm_eps", 1e-5))),
        )
        self.add_module("norm_out_linear", MM_WEIGHT_REGISTER["Default"]("norm_out.linear.weight", "norm_out.linear.bias"))
        num_heads = int(config.get("num_pdd_heads", config.get("infer_steps", 6)))
        # Registered one by one: WeightModuleList expects WeightModule children,
        # and each head is a single matmul weight. ``pdd_proj_out[step]`` indexes them.
        heads = []
        for step in range(num_heads):
            head = MM_WEIGHT_REGISTER["Default-ForceFp32"](f"{PDD_HEAD_PREFIX}.{step}.weight", f"{PDD_HEAD_PREFIX}.{step}.bias")
            self.add_module(f"{PDD_HEAD_PREFIX}_{step}", head)
            heads.append(head)
        self.pdd_proj_out = tuple(heads)
