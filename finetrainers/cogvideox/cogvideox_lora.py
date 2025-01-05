from typing import Any, Dict, List, Optional, Union

import torch
from diffusers import AutoencoderKLCogVideoX, CogVideoXDDIMScheduler, CogVideoXPipeline, CogVideoXTransformer3DModel
from PIL import Image
from transformers import T5EncoderModel, T5Tokenizer

from .utils import prepare_rotary_positional_embeddings


def load_condition_models(
    model_id: str = "THUDM/CogVideoX-5b",
    text_encoder_dtype: torch.dtype = torch.bfloat16,
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
    model_id: str = "THUDM/CogVideoX-5b",
    vae_dtype: torch.dtype = torch.bfloat16,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
):
    vae = AutoencoderKLCogVideoX.from_pretrained(
        model_id, subfolder="vae", torch_dtype=vae_dtype, revision=revision, cache_dir=cache_dir
    )
    return {"vae": vae}


def load_diffusion_models(
    model_id: str = "THUDM/CogVideoX-5b",
    transformer_dtype: torch.dtype = torch.bfloat16,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
):
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=transformer_dtype, revision=revision, cache_dir=cache_dir
    )
    scheduler = CogVideoXDDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    return {"transformer": transformer, "scheduler": scheduler}


def initialize_pipeline(
    model_id: str = "THUDM/CogVideoX-5b",
    text_encoder_dtype: torch.dtype = torch.bfloat16,
    transformer_dtype: torch.dtype = torch.bfloat16,
    vae_dtype: torch.dtype = torch.bfloat16,
    tokenizer: Optional[T5Tokenizer] = None,
    text_encoder: Optional[T5EncoderModel] = None,
    transformer: Optional[CogVideoXTransformer3DModel] = None,
    vae: Optional[AutoencoderKLCogVideoX] = None,
    scheduler: Optional[CogVideoXDDIMScheduler] = None,
    device: Optional[torch.device] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    enable_slicing: bool = False,
    enable_tiling: bool = False,
    enable_model_cpu_offload: bool = False,
    **kwargs,
) -> CogVideoXPipeline:
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

    pipe = CogVideoXPipeline.from_pretrained(model_id, **components, revision=revision, cache_dir=cache_dir)
    pipe.text_encoder = pipe.text_encoder.to(dtype=text_encoder_dtype)
    pipe.transformer = pipe.transformer.to(dtype=transformer_dtype)
    pipe.vae = pipe.vae.to(dtype=vae_dtype)

    if enable_slicing:
        pipe.vae.enable_slicing()
    if enable_tiling:
        pipe.vae.enable_tiling()

    if enable_model_cpu_offload:
        pipe.enable_model_cpu_offload(device=device)
    else:
        pipe.to(device=device)

    return pipe


def prepare_conditions(
    tokenizer,
    text_encoder,
    prompt: Union[str, List[str]],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    max_sequence_length: int = 226,  # TODO: this should be configurable
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


def prepare_latents(
    vae: AutoencoderKLCogVideoX,
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

    image_or_video = image_or_video.to(device=device, dtype=vae.dtype)
    image_or_video = image_or_video.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
    if not precompute:
        latents = vae.encode(image_or_video).latent_dist.sample(generator=generator)
        if not vae.config.invert_scale_latents:
            latents = latents * vae.config.scaling_factor
        # For training Cog 1.5, we don't need to handle the scaling factor here.
        # The CogVideoX team forgot to multiply here, so we should not do it too. Invert scale latents
        # is probably only needed for image-to-video training.
        # TODO(aryan): investigate this
        # else:
        #     latents = 1 / vae.config.scaling_factor * latents
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
    if not vae_config.invert_scale_latents:
        latents = latents * vae_config.scaling_factor
    # For training Cog 1.5, we don't need to handle the scaling factor here.
    # The CogVideoX team forgot to multiply here, so we should not do it too. Invert scale latents
    # is probably only needed for image-to-video training.
    # TODO(aryan): investigate this
    # else:
    #     latents = 1 / vae_config.scaling_factor * latents
    latents = _pad_frames(latents, patch_size_t)
    latents = latents.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
    return {"latents": latents}


def collate_fn_t2v(batch: List[List[Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
    return {
        "prompts": [x["prompt"] for x in batch[0]],
        "videos": torch.stack([x["video"] for x in batch[0]]),
    }


def calculate_noisy_latents(
    scheduler: CogVideoXDDIMScheduler,
    noise: torch.Tensor,
    latents: torch.Tensor,
    timesteps: torch.LongTensor,
) -> torch.Tensor:
    noisy_latents = scheduler.add_noise(latents, noise, timesteps)
    return noisy_latents


def forward_pass(
    transformer: CogVideoXTransformer3DModel,
    scheduler: CogVideoXDDIMScheduler,
    prompt_embeds: torch.Tensor,
    latents: torch.Tensor,
    noisy_latents: torch.Tensor,
    timesteps: torch.LongTensor,
    ofs_emb: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    # Just hardcode for now. In Diffusers, we will refactor such that RoPE would be handled within the model itself.
    VAE_SPATIAL_SCALE_FACTOR = 8
    transformer_config = transformer.module.config if hasattr(transformer, "module") else transformer.config
    batch_size, num_frames, num_channels, height, width = noisy_latents.shape
    rope_base_height = transformer_config.sample_height * VAE_SPATIAL_SCALE_FACTOR
    rope_base_width = transformer_config.sample_width * VAE_SPATIAL_SCALE_FACTOR

    image_rotary_emb = (
        prepare_rotary_positional_embeddings(
            height=height * VAE_SPATIAL_SCALE_FACTOR,
            width=width * VAE_SPATIAL_SCALE_FACTOR,
            num_frames=num_frames,
            vae_scale_factor_spatial=VAE_SPATIAL_SCALE_FACTOR,
            patch_size=transformer_config.patch_size,
            patch_size_t=transformer_config.patch_size_t if hasattr(transformer_config, "patch_size_t") else None,
            attention_head_dim=transformer_config.attention_head_dim,
            device=transformer.device,
            base_height=rope_base_height,
            base_width=rope_base_width,
        )
        if transformer_config.use_rotary_positional_embeddings
        else None
    )
    ofs_emb = None if transformer_config.ofs_embed_dim is None else latents.new_full((batch_size,), fill_value=2.0)

    velocity = transformer(
        hidden_states=noisy_latents,
        timestep=timesteps,
        encoder_hidden_states=prompt_embeds,
        ofs=ofs_emb,
        image_rotary_emb=image_rotary_emb,
        return_dict=False,
    )[0]
    # For CogVideoX, the transformer predicts the velocity. The denoised output is calculated by applying the same
    # code paths as scheduler.get_velocity(), which can be confusing to understand.
    denoised_latents = scheduler.get_velocity(velocity, noisy_latents, timesteps)

    return {"latents": denoised_latents}


def validation(
    pipeline: CogVideoXPipeline,
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
    max_sequence_length: int = 226,
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

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    return {"prompt_embeds": prompt_embeds}


def _pad_frames(latents: torch.Tensor, patch_size_t: int):
    if patch_size_t is None or patch_size_t == 1:
        return latents

    # `latents` should be of the following format: [B, C, F, H, W].
    # For CogVideoX 1.5, the latent frames should be padded to make it divisible by patch_size_t
    latent_num_frames = latents.shape[2]
    additional_frames = patch_size_t - latent_num_frames % patch_size_t

    if additional_frames > 0:
        last_frame = latents[:, :, -1:, :, :]
        padding_frames = last_frame.repeat(1, 1, additional_frames, 1, 1)
        latents = torch.cat([latents, padding_frames], dim=2)

    return latents


COGVIDEOX_T2V_LORA_CONFIG = {
    "pipeline_cls": CogVideoXPipeline,
    "load_condition_models": load_condition_models,
    "load_latent_models": load_latent_models,
    "load_diffusion_models": load_diffusion_models,
    "initialize_pipeline": initialize_pipeline,
    "prepare_conditions": prepare_conditions,
    "prepare_latents": prepare_latents,
    "post_latent_preparation": post_latent_preparation,
    "collate_fn": collate_fn_t2v,
    "calculate_noisy_latents": calculate_noisy_latents,
    "forward_pass": forward_pass,
    "validation": validation,
}
