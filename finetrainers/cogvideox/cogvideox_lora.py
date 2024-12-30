from typing import Dict, List, Optional, Union

import torch
from diffusers import AutoencoderKLCogVideoX, CogVideoXDPMScheduler, CogVideoXPipeline, CogVideoXTransformer3DModel
from PIL import Image
from transformers import T5EncoderModel, T5Tokenizer


def load_condition_models(
    model_id: str = "THUDM/CogVideoX-2b",
    text_encoder_dtype: torch.dtype = torch.float16,
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
    model_id: str = "THUDM/CogVideoX-2b",
    vae_dtype: torch.dtype = torch.float16,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
):
    vae = AutoencoderKLCogVideoX.from_pretrained(
        model_id, subfolder="vae", torch_dtype=vae_dtype, revision=revision, cache_dir=cache_dir
    )
    return {"vae": vae}


def load_diffusion_models(
    model_id: str = "THUDM/CogVideoX-2b",
    transformer_dtype: torch.dtype = torch.float16,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
):
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=transformer_dtype, revision=revision, cache_dir=cache_dir
    )
    scheduler = CogVideoXDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    return {"transformer": transformer, "scheduler": scheduler}


def initialize_pipeline(
    model_id: str = "THUDM/CogVideoX-2b",
    text_encoder_dtype: torch.dtype = torch.float16,
    transformer_dtype: torch.dtype = torch.float16,
    vae_dtype: torch.dtype = torch.float16,
    tokenizer: Optional[T5Tokenizer] = None,
    text_encoder: Optional[T5EncoderModel] = None,
    transformer: Optional[CogVideoXTransformer3DModel] = None,
    vae: Optional[AutoencoderKLCogVideoX] = None,
    scheduler: Optional[CogVideoXDPMScheduler] = None,
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
        else:
            latents = 1 / vae.config.scaling_factor * latents
        latents = latents.to(dtype=dtype)
        return {"latents": latents}
    else:
        if vae.use_slicing and image_or_video.shape[0] > 1:
            encoded_slices = [vae._encode(x_slice) for x_slice in image_or_video.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = vae._encode(image_or_video)
            return {"latents": h}


def post_latent_preparation(latents: torch.Tensor, **kwargs) -> torch.Tensor:
    return {"latents": latents}


def collate_fn_t2v(batch: List[List[Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
    return {
        "prompts": [x["prompt"] for x in batch[0]],
        "videos": torch.stack([x["video"] for x in batch[0]]),
    }


def forward_pass(
    transformer: CogVideoXTransformer3DModel,
    prompt_embeds: torch.Tensor,
    noisy_latents: torch.Tensor,
    timesteps: torch.LongTensor,
    image_rotary_emb: torch.Tensor,
    ofs_emb: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    noisy_latents = noisy_latents.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
    denoised_latents = transformer(
        hidden_states=noisy_latents,
        timestep=timesteps,
        encoder_hidden_states=prompt_embeds,
        ofs_emb=ofs_emb,
        image_rotary_emb=image_rotary_emb,
        return_dict=False,
    )[0]

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
    "forward_pass": forward_pass,
    "validation": validation,
}
