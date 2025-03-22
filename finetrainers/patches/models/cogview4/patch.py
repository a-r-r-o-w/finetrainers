from typing import Optional

import diffusers
import torch
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb


def patch_cogview4_attn_processor_for_tp_compatibility() -> None:
    diffusers.models.transformers.transformer_cogview4.CogView4AttnProcessor.__call__ = (
        _patched_CogView4AttnProcessor___call__
    )


def _patched_CogView4AttnProcessor___call__(
    self,
    attn: Attention,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    image_rotary_emb: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    batch_size, text_seq_length, embed_dim = encoder_hidden_states.shape
    batch_size, image_seq_length, embed_dim = hidden_states.shape
    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

    # 1. QKV projections
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    query = query.unflatten(2, (-1, attn.inner_dim // attn.heads)).transpose(1, 2)
    key = key.unflatten(2, (-1, attn.inner_dim // attn.heads)).transpose(1, 2)
    value = value.unflatten(2, (-1, attn.inner_dim // attn.heads)).transpose(1, 2)

    # 2. QK normalization
    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    # 3. Rotational positional embeddings applied to latent stream
    if image_rotary_emb is not None:
        query[:, :, text_seq_length:, :] = apply_rotary_emb(
            query[:, :, text_seq_length:, :], image_rotary_emb, use_real_unbind_dim=-2
        )
        key[:, :, text_seq_length:, :] = apply_rotary_emb(
            key[:, :, text_seq_length:, :], image_rotary_emb, use_real_unbind_dim=-2
        )

    # 4. Attention
    if attention_mask is not None:
        text_attention_mask = attention_mask.float().to(query.device)
        actual_text_seq_length = text_attention_mask.size(1)
        new_attention_mask = torch.zeros((batch_size, text_seq_length + image_seq_length), device=query.device)
        new_attention_mask[:, :actual_text_seq_length] = text_attention_mask
        new_attention_mask = new_attention_mask.unsqueeze(2)
        attention_mask_matrix = new_attention_mask @ new_attention_mask.transpose(1, 2)
        attention_mask = (attention_mask_matrix > 0).unsqueeze(1).to(query.dtype)

    hidden_states = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
    )
    hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
    hidden_states = hidden_states.type_as(query)

    # 5. Output projection
    hidden_states = attn.to_out[0](hidden_states)
    hidden_states = attn.to_out[1](hidden_states)

    encoder_hidden_states, hidden_states = hidden_states.split(
        [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
    )
    return hidden_states, encoder_hidden_states
