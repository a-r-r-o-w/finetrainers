import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import LTXVideoTransformer3DModel
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import is_torch_version
from conditioned_residual_adapter_bottleneck import ConditionedResidualAdapterBottleneck
@maybe_allow_in_graph
class LTXVideoConditionedTransformer3DModel(LTXVideoTransformer3DModel):
    def __init__(self,
        in_channels: int = 128,
        out_channels: int = 128,
        patch_size: int = 1,
        patch_size_t: int = 1,
        num_attention_heads: int = 32,
        attention_head_dim: int = 64,
        cross_attention_dim: int = 2048,
        num_layers: int = 28,
        activation_fn: str = "gelu-approximate",
        qk_norm: str = "rms_norm_across_heads",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        caption_channels: int = 4096,
        attention_bias: bool = True,
        attention_out_bias: bool = True,
        adapter_in_dim:int = 256):

        self.adapter = ConditionedResidualAdapterBottleneck(
            input_dim=adapter_in_dim,
            output_dim=128,
            bottleneck_dim=64,
            adapter_dropout=0.1,
            adapter_init_scale=1e-3
        )

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            cross_attention_dim=cross_attention_dim,
            num_layers=num_layers,
            activation_fn=activation_fn,
            qk_norm=qk_norm,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            caption_channels=caption_channels,
            attention_bias=attention_bias,
            attention_out_bias=attention_out_bias
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        new_model = cls(**model.config.__dict__)
        new_model.load_state_dict(model.state_dict())
        return new_model

    # def save_pretrained(self, save_directory: str, **kwargs):
    #     """
    #     Saves model weights (including adapter) + config to disk
    #     in a format compatible with .from_pretrained()
    #     """
    #     # 1) Save config
    #     self.config.save_pretrained(save_directory)

    #     # 2) Save PyTorch state dict
    #     state_dict = self.state_dict()
    #     torch.save(state_dict, os.path.join(save_directory, "pytorch_model.bin"))

def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_attention_mask: torch.Tensor,
        residual_x: torch.Tensor,
        num_frames: int,
        height: int,
        width: int,
        rope_interpolation_scale: Optional[Tuple[float, float, float]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True
    ) -> torch.Tensor:

        image_rotary_emb = self.rope(hidden_states, num_frames, height, width, rope_interpolation_scale)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        batch_size = hidden_states.size(0)

        # inject the condition and the residual then project it into the pretrained proj_in
        hidden_states = self.adapter(residual_x=residual_x,
                                     conditioned_x=hidden_states)

        hidden_states = self.proj_in(hidden_states)

        temb, embedded_timestep = self.time_embed(
            timestep.flatten(),
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )

        temb = temb.view(batch_size, -1, temb.size(-1))
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.size(-1))

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


def apply_rotary_emb(x, freqs):
    cos, sin = freqs
    x_real, x_imag = x.unflatten(2, (-1, 2)).unbind(-1)  # [B, S, H, D // 2]
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(2)
    out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
    return out

