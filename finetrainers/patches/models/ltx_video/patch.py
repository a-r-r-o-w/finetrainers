from typing import Any, Dict, Optional, Tuple

import diffusers
import torch
from diffusers import LTXVideoTransformer3DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils.import_utils import is_torch_version


def patch_transformer_forward() -> None:
    _perform_ltx_transformer_forward_patch()


def patch_apply_rotary_emb_for_tp_compatibility() -> None:
    _perform_ltx_apply_rotary_emb_tensor_parallel_compatibility_patch()


def _perform_ltx_transformer_forward_patch() -> None:
    LTXVideoTransformer3DModel.forward = _patched_LTXVideoTransformer3D_forward


def _perform_ltx_apply_rotary_emb_tensor_parallel_compatibility_patch() -> None:
    def apply_rotary_emb(x, freqs):
        cos, sin = freqs
        # ======== THIS IS CHANGED FROM THE ORIGINAL IMPLEMENTATION ========
        # The change is made due to unsupported DTensor operation aten.ops.unbind
        # FIXME: Once aten.ops.unbind support lands, this will no longer be required
        # x_real, x_imag = x.unflatten(2, (-1, 2)).unbind(-1)  # [B, S, H, D // 2]
        x_real, x_imag = x.unflatten(2, (-1, 2)).chunk(2, dim=-1)  # [B, S, H, D // 2]
        # ==================================================================
        x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(2)
        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
        return out

    diffusers.models.transformers.transformer_ltx.apply_rotary_emb = apply_rotary_emb


def _patched_LTXVideoTransformer3D_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_attention_mask: torch.Tensor,
    num_frames: int,
    height: int,
    width: int,
    rope_interpolation_scale: Optional[Tuple[float, float, float]] = None,
    return_dict: bool = True,
    *args,
    **kwargs,
) -> torch.Tensor:
    image_rotary_emb = self.rope(hidden_states, num_frames, height, width, rope_interpolation_scale)

    # convert encoder_attention_mask to a bias the same way we do for attention_mask
    if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
        encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
        encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

    batch_size = hidden_states.size(0)

    # ===== This is modified compared to Diffusers =====
    # This is done because the Diffusers pipeline will pass in a 1D tensor for timestep
    if timestep.ndim == 1:
        timestep = timestep.view(-1, 1, 1).expand(-1, *hidden_states.shape[1:-1], -1)
    # ==================================================

    temb, embedded_timestep = self.time_embed(
        timestep.flatten(),
        batch_size=batch_size,
        hidden_dtype=hidden_states.dtype,
    )

    # ===== This is modified compared to Diffusers =====
    # temb = temb.view(batch_size, -1, temb.size(-1))
    # embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.size(-1))
    # ==================================================
    # This is done to make it possible to use per-token timestep embedding
    temb = temb.view(batch_size, *hidden_states.shape[1:-1], temb.size(-1))
    embedded_timestep = embedded_timestep.view(batch_size, *hidden_states.shape[1:-1], embedded_timestep.size(-1))
    # ==================================================

    hidden_states = self.proj_in(hidden_states)

    encoder_hidden_states = self.caption_projection(encoder_hidden_states)
    encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.size(-1))

    for block in self.transformer_blocks:
        if torch.is_grad_enabled() and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states,
                encoder_hidden_states,
                temb,
                image_rotary_emb,
                encoder_attention_mask,
                **ckpt_kwargs,
            )
        else:
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                encoder_attention_mask=encoder_attention_mask,
            )

    scale_shift_values = self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
    shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

    hidden_states = self.norm_out(hidden_states)
    hidden_states = hidden_states * (1 + scale) + shift
    output = self.proj_out(hidden_states)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)
