from typing import Optional

import diffusers
import torch


def patch_time_text_image_embedding_forward() -> None:
    _patch_time_text_image_embedding_forward()


def _patch_time_text_image_embedding_forward() -> None:
    diffusers.models.transformers.transformer_wan.WanTimeTextImageEmbedding.forward = (
        _patched_WanTimeTextImageEmbedding_forward
    )


def _patched_WanTimeTextImageEmbedding_forward(
    self,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_image: Optional[torch.Tensor] = None,
):
    # Some code has been removed compared to original implementation in Diffusers
    # Also, timestep is typed as that of encoder_hidden_states
    timestep = self.timesteps_proj(timestep).type_as(encoder_hidden_states)
    temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
    timestep_proj = self.time_proj(self.act_fn(temb))

    encoder_hidden_states = self.text_embedder(encoder_hidden_states)
    if encoder_hidden_states_image is not None:
        encoder_hidden_states_image = self.image_embedder(encoder_hidden_states_image)

    return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image
