from lightx2v.models.networks.minimax_h3.infer.module_io import MiniMaxH3VelocityOutput
from lightx2v.models.networks.minimax_h3.infer.post_infer import MiniMaxH3PostInfer
from lightx2v.models.networks.minimax_h3.infer.pre_infer import MiniMaxH3PreInfer
from lightx2v.utils.envs import GET_DTYPE


class MiniMaxH3ImagePreInfer(MiniMaxH3PreInfer):
    """Pack ``[text | references | target]`` rows; there is no audio stream."""

    def prepare_inputs(self, weights, prompt_embeds):
        layout = self.scheduler.layout
        bulk_dtype = GET_DTYPE()
        # The patch projection is FP32 in the release; the packed sequence
        # follows the BF16 text stream.
        image_embeds = weights.proj_in.apply(self.scheduler.video_latents.float()).to(bulk_dtype)
        hidden_states = image_embeds.new_zeros((layout.sequence_length, self.hidden_size))
        if layout.text_indices.numel():
            text_embeds = weights.context_embedder.apply(prompt_embeds.to(bulk_dtype))
            text_embeds = self._refine_text(weights, text_embeds)
            hidden_states.index_copy_(0, layout.text_indices, text_embeds)
        hidden_states.index_copy_(0, layout.video_indices, image_embeds)
        return hidden_states, self._rotary_embedding(layout.position_ids)


class MiniMaxH3ImagePostInfer(MiniMaxH3PostInfer):
    """Shared final norm, then the PDD head of the current sampling step."""

    def infer(self, weights, hidden_states, pre_infer_out):
        step = self.scheduler.pdd_step_index
        if not 0 <= step < len(weights.pdd_proj_out):
            raise IndexError(f"MiniMax-H3-Image has {len(weights.pdd_proj_out)} PDD heads but the sampler is at step {step}")
        # The head runs over every row before the image rows are selected,
        # matching the reference; the FP32 head consumes the FP32 norm output.
        hidden_states = self.norm_out(weights, hidden_states, pre_infer_out)
        image = weights.pdd_proj_out[step].apply(hidden_states).index_select(0, pre_infer_out.video_indices)
        return MiniMaxH3VelocityOutput(video=image, audio=None)
