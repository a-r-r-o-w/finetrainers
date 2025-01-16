from typing import Any, Dict, List, Optional, Union

import torch
from diffusers import AutoencoderKLMochi, FlowMatchEulerDiscreteScheduler, MochiPipeline, MochiTransformer3DModel
from PIL import Image
from transformers import T5EncoderModel, T5Tokenizer


# Following the official implementation.
def cast_dit(model, dtype):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            assert any(
                n in name for n in ["time_embed", "proj_out", "blocks", "norm_out"]
            ), f"Unexpected linear layer: {name}"
            module.to(dtype=dtype)
        elif isinstance(module, torch.nn.Conv2d):
            module.to(dtype=dtype)
    return model


def load_condition_models(
    model_id: str = "genmo/mochi-1-preview",
    text_encoder_dtype: torch.dtype = torch.float32,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
):
    tokenizer = T5Tokenizer.from_pretrained(model_id, subfolder="tokenizer", revision=revision, cache_dir=cache_dir)
    text_encoder = T5EncoderModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=text_encoder_dtype, revision=revision, cache_dir=cache_dir
    )
    return {"tokenizer": tokenizer, "text_encoder": text_encoder}


def load_latent_models(
    model_id: str = "genmo/mochi-1-preview",
    vae_dtype: torch.dtype = torch.float32,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
):
    vae = AutoencoderKLMochi.from_pretrained(
        model_id, subfolder="vae", torch_dtype=vae_dtype, revision=revision, cache_dir=cache_dir
    )
    return {"vae": vae}


def load_diffusion_models(
    model_id: str = "genmo/mochi-1-preview",
    transformer_dtype: torch.dtype = torch.float32,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
):
    transformer = MochiTransformer3DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=transformer_dtype, revision=revision, cache_dir=cache_dir
    )
    # TODO (sayakpaul): 
    # 1. test if this is necessary by doing a pure bf16 (casting with to()) and this way of casting.
    # 2. this is likely fine for LoRA but for full fine-tuning this could be revisited.
    transformer = cast_dit(transformer, torch.bfloat16)
    
    # Instead of doing a `from_pretrained()` we simply initialize the scheduler. This is so that the 
    # `invert_sigmas` flag in the original config does not mess with any
    # of the downstream reversing of sigmas we apply.
    scheduler = FlowMatchEulerDiscreteScheduler()
    return {"transformer": transformer, "scheduler": scheduler}


def initialize_pipeline(
    model_id: str = "genmo/mochi-1-preview",
    text_encoder_dtype: torch.dtype = torch.float32,
    transformer_dtype: torch.dtype = torch.float32,
    vae_dtype: torch.dtype = torch.float32,
    tokenizer: Optional[T5Tokenizer] = None,
    text_encoder: Optional[T5EncoderModel] = None,
    transformer: Optional[MochiTransformer3DModel] = None,
    vae: Optional[AutoencoderKLMochi] = None,
    scheduler: Optional[FlowMatchEulerDiscreteScheduler] = None,
    device: Optional[torch.device] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    enable_slicing: bool = False,
    enable_tiling: bool = False,
    enable_model_cpu_offload: bool = False,
    is_training: bool = False,
    **kwargs,
) -> MochiPipeline:
    component_name_pairs = [
        ("tokenizer", tokenizer),
        ("text_encoder", text_encoder),
        ("transformer", transformer),
        ("vae", vae),
        ("scheduler", scheduler),
    ]
    components = {}
    for name, component in component_name_pairs:
        if component is not None:
            components[name] = component

    pipe = MochiPipeline.from_pretrained(model_id, **components, revision=revision, cache_dir=cache_dir)
    pipe.text_encoder = pipe.text_encoder.to(dtype=text_encoder_dtype)
    pipe.vae = pipe.vae.to(dtype=vae_dtype)

    # The transformer should already be in the correct dtype when training, so we don't need to cast it here.
    # If we cast, whilst using fp8 layerwise upcasting hooks, it will lead to an error in the training during
    # DDP optimizer step.
    if not is_training:
        pipe.transformer = pipe.transformer.to(dtype=transformer_dtype)

    if enable_slicing:
        pipe.vae.enable_slicing()
    if enable_tiling:
        pipe.vae.enable_tiling()

    if enable_model_cpu_offload:
        pipe.enable_model_cpu_offload(device=device)
    else:
        pipe = pipe.to(device=device)

    return pipe


def prepare_conditions(
    tokenizer,
    text_encoder,
    prompt: Union[str, List[str]],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    max_sequence_length: int = 256,  # TODO: this should be configurable
    **kwargs,
):
    device = device or text_encoder.device
    dtype = dtype or text_encoder.dtype
    return _get_t5_prompt_embeds(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        prompt=prompt,
        max_sequence_length=max_sequence_length,
        device=device,
        dtype=dtype,
    )


# Following original implementation.
@torch.autocast("cuda", dtype=torch.bfloat16)
@torch.inference_mode()
def prepare_latents(
    vae: AutoencoderKLMochi,
    image_or_video: torch.Tensor,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    generator: Optional[torch.Generator] = None,
    precompute: bool = False,
    **kwargs,
) -> torch.Tensor:
    device = device or vae.device
    dtype = dtype or vae.dtype

    if image_or_video.ndim == 4:
        image_or_video = image_or_video.unsqueeze(2)
    assert image_or_video.ndim == 5, f"Expected 5D tensor, got {image_or_video.ndim}D tensor"

    assert (image_or_video.shape[1] - 1) % 6 == 0, "Expected number of frames to be 1 mod 6"
    image_or_video = image_or_video.to(device=device, dtype=vae.dtype)
    image_or_video = image_or_video.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]

    if not precompute:
        latents = vae.encode(image_or_video).latent_dist.sample(generator=generator)
        latents = latents.to(dtype=dtype)
        return {"latents": latents}
    else:
        # handle vae scaling in the `train()` method directly.
        if vae.use_slicing and image_or_video.shape[0] > 1:
            encoded_slices = [vae._encode(x_slice) for x_slice in image_or_video.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = vae._encode(image_or_video)
        return {"latents": h}


def post_latent_preparation(
    vae_config: Dict[str, Any], latents: torch.Tensor, patch_size_t: Optional[int] = None, **kwargs
) -> torch.Tensor:
    return {"latents": latents}


def collate_fn_t2v(batch: List[List[Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
    return {
        "prompts": [x["prompt"] for x in batch[0]],
        "videos": torch.stack([x["video"] for x in batch[0]]),
    }


# Following original implementation.
@torch.autocast("cuda", torch.bfloat16)
def forward_pass(
    transformer: MochiTransformer3DModel,
    scheduler: FlowMatchEulerDiscreteScheduler,
    prompt_embeds: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    latents: torch.Tensor,
    noisy_latents: torch.Tensor,
    timesteps: torch.LongTensor,
    **kwargs,
) -> torch.Tensor:
    denoised_latents = transformer(
        hidden_states=noisy_latents,
        encoder_hidden_states=prompt_embeds,
        encoder_attention_mask=prompt_attention_mask,
        # TODO: revisit if needed as Mochi has a weird way of doing `timesteps`.
        timestep=scheduler.config.num_train_timesteps - timesteps,
        return_dict=False,
    )[0]
    # TODO: revisit if needed. We do this because of 
    # https://github.com/genmoai/mochi/blob/aba74c1b5e0755b1fa3343d9e4bd22e89de77ab1/src/genmo/mochi_preview/dit/joint_model/asymm_models_joint.py#L656
    # In short, Mochi operates on reversed targets which is why we need to negate 
    # the predictions.
    denoised_latents = -denoised_latents

    return {"latents": denoised_latents}


def validation(
    pipeline: MochiPipeline,
    prompt: str,
    image: Optional[Image.Image] = None,
    video: Optional[List[Image.Image]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_frames: Optional[int] = None,
    num_videos_per_prompt: int = 1,
    generator: Optional[torch.Generator] = None,
    **kwargs,
):
    generation_kwargs = {
        "prompt": prompt,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "num_videos_per_prompt": num_videos_per_prompt,
        "generator": generator,
        "return_dict": True,
        "output_type": "pil",
    }
    generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}
    output = pipeline(**generation_kwargs).frames[0]
    return [("video", output)]


def _get_t5_prompt_embeds(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]] = None,
    max_sequence_length: int = 256,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids
    prompt_attention_mask = text_inputs.attention_mask
    prompt_attention_mask = prompt_attention_mask.bool().to(device)

    prompt_embeds = text_encoder(text_input_ids.to(device), attention_mask=prompt_attention_mask)[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    return prompt_embeds, prompt_attention_mask


# TODO: model specs
MOCHI_T2V_LORA_CONFIG = {
    "pipeline_cls": MochiPipeline,
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
