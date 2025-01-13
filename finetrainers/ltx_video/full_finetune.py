from diffusers import LTXPipeline

from .ltx_video_lora import (
    collate_fn_t2v,
    forward_pass,
    initialize_pipeline,
    load_condition_models,
    load_diffusion_models,
    load_latent_models,
    post_latent_preparation,
    prepare_conditions,
    prepare_latents,
    validation,
)


# TODO(aryan): refactor into model specs for better re-use
LTX_VIDEO_T2V_FULL_FINETUNE_CONFIG = {
    "pipeline_cls": LTXPipeline,
    "load_condition_models": load_condition_models,
    "load_latent_models": load_latent_models,
    "load_diffusion_models": load_diffusion_models,
    "initialize_pipeline": initialize_pipeline,
    "prepare_conditions": prepare_conditions,
    "prepare_latents": prepare_latents,
    "post_latent_preparation": post_latent_preparation,
    "collate_fn": collate_fn_t2v,
    "forward_pass": forward_pass,
    "validation": validation,
}
