from diffusers import LTXPipeline

from .lora import (
    collate_fn_t2v,
    forward_pass,
    initialize_pipeline,
    load_condition_models,
    load_diffusion_models,
    load_conditioned_diffusion_models, # use this for conditioning
    load_latent_models,
    post_latent_preparation,
    prepare_conditions,
    prepare_latents,
    validation,
    conditional_validation,
    collate_fn_t2v_cond,
    conditioned_forward_pass # use this for conditioning
)


# TODO(aryan): refactor into model specs for better re-use
LTX_VIDEO_T2V_FULL_FINETUNE_CONFIG = {
    "pipeline_cls": LTXPipeline,
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
