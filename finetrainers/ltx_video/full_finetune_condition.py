from typing import Dict, List, Optional

import torch
import torch.nn as nn

from diffusers import FlowMatchEulerDiscreteScheduler, LTXVideoTransformer3DModel
from finetrainers.conditioning.LTXVideoConditionedTransformer3DModel import LTXVideoConditionedTransformer3DModel
from finetrainers.conditioning.conditioned_pipeline import LTXConditionedPipeline
from PIL import Image

# Exisiting Pipeline
from .lora import (
    initialize_pipeline,
    load_conditioned_diffusion_models, # use this for conditioning
    load_latent_models,
    post_latent_preparation,
    prepare_conditions,
    prepare_latents,
    load_condition_models
)

def load_conditioned_diffusion_models(
    model_id: str = "Lightricks/LTX-Video",
    transformer_dtype: torch.dtype = torch.bfloat16,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> Dict[str, nn.Module]:
    transformer = LTXVideoConditionedTransformer3DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=transformer_dtype, revision=revision, cache_dir=cache_dir
    )
    scheduler = FlowMatchEulerDiscreteScheduler()
    return {"transformer": transformer, "scheduler": scheduler}


def load_diffusion_models(
    model_id: str = "Lightricks/LTX-Video",
    transformer_dtype: torch.dtype = torch.bfloat16,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> Dict[str, nn.Module]:
    transformer = LTXVideoTransformer3DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=transformer_dtype, revision=revision, cache_dir=cache_dir
    )
    scheduler = FlowMatchEulerDiscreteScheduler()
    return {"transformer": transformer, "scheduler": scheduler}


def collate_fn_t2v_cond(batch: List[List[Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
    return {
        "prompts": [x["prompt"] for x in batch[0]],
        "videos": torch.stack([x["video"] for x in batch[0]]),
        "poses": torch.stack([x["pose"] for x in batch[0]]),
        "img_refs": torch.stack([x["img_ref"] for x in batch[0]])
    }

def conditioned_forward_pass(
    transformer: LTXVideoConditionedTransformer3DModel,
    prompt_embeds: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    latents: torch.Tensor,
    noisy_latents: torch.Tensor,
    timesteps: torch.LongTensor,
    num_frames: int,
    height: int,
    width: int,
    noisy_latents_residual:torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    rope_interpolation_scale = [1 / 25, 32, 32]
   
    denoised_latents = transformer(
        hidden_states=noisy_latents,
        encoder_hidden_states=prompt_embeds,
        timestep=timesteps,
        encoder_attention_mask=prompt_attention_mask,
        num_frames=num_frames,
        height=height,
        width=width,
        rope_interpolation_scale=rope_interpolation_scale,
        return_dict=False,
        residual_x=noisy_latents_residual,
    )[0]

    return {"latents": denoised_latents}

def conditional_validation(
    pipeline: LTXConditionedPipeline,
    prompt: str,
    image: Optional[Image.Image] = None,
    video: Optional[List[Image.Image]] = None,
    pose_video: Optional[List[Image.Image]] = None,
    image_ref_video: Optional[List[Image.Image]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_frames: Optional[int] = None,
    frame_rate: int = 24,
    num_videos_per_prompt: int = 1,
    generator: Optional[torch.Generator] = None,
    **kwargs,
):
    generation_kwargs = {
        "prompt": prompt,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "frame_rate": frame_rate,
        "num_videos_per_prompt": num_videos_per_prompt,
        "generator": generator,
        "return_dict": True,
        "output_type": "pil",
        "pose_video": pose_video,
        "image_ref_video": image_ref_video,
    }

    # pose template noisey input [cat] img_ref + pose video 
      
    # pose template ref video latent + patchify video latent.
    # img ref patchify video latent +
    # add them together as input 
    # residual x latent 

    generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}
    video = pipeline(**generation_kwargs).frames[0]
    return [("video", video)]


LTX_VIDEO_COND_T2V_FULL_FINETUNE_CONFIG = {
    "pipeline_cls": LTXConditionedPipeline,
    "load_condition_models": load_condition_models,
    "load_latent_models": load_latent_models,
    "load_diffusion_models": load_conditioned_diffusion_models,
    "initialize_pipeline": initialize_pipeline,
    "prepare_conditions": prepare_conditions,
    "prepare_latents": prepare_latents,
    "post_latent_preparation": post_latent_preparation,
    "collate_fn": collate_fn_t2v_cond,
    "forward_pass": conditioned_forward_pass,
    "validation": conditional_validation,
}
