from typing import Dict, List, Optional, Union

import torch
from diffusers import AutoencoderKLCogVideoX, CogVideoXDPMScheduler, CogVideoXPipeline, CogVideoXTransformer3DModel
from PIL import Image
from transformers import T5EncoderModel, T5Tokenizer
from .utils import prepare_rotary_positional_embeddings
from ..utils.torch_utils import expand_tensor_to_dims


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
        else:
            latents = 1 / vae.config.scaling_factor * latents
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


def post_latent_preparation(latents: torch.Tensor, **kwargs) -> torch.Tensor:
    if kwargs.get("precompute_conditions", False) and kwargs.get("vae_config", None) is not None:
        latents = _scale_latents(latents, kwargs.get("vae_config"))
    latents = _pad_frames(latents, kwargs.get("denoier_config", None))
    latents = latents.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
    return {"latents": latents}


def collate_fn_t2v(batch: List[List[Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
    return {
        "prompts": [x["prompt"] for x in batch[0]],
        "videos": torch.stack([x["video"] for x in batch[0]]),
    }

def calculate_noisy_latents(scheduler, latent_conditions, timesteps, state):
    noise = torch.randn(
        latent_conditions["latents"].shape,
        generator=state.generator,
        device=state.accelerator.device,
        dtype=state.weight_dtype,
    )
    noisy_latents = scheduler.add_noise(latent_conditions["latents"], noise, timesteps)
    return noise, noisy_latents

def forward_pass(
    transformer: CogVideoXTransformer3DModel,
    prompt_embeds: torch.Tensor,
    noisy_latents: torch.Tensor,
    timesteps: torch.LongTensor,
    ofs_emb: Optional[torch.Tensor] = None,
    latents: torch.Tensor = None,
    **kwargs
) -> torch.Tensor:
    denoiser_config = transformer.module.config if hasattr(transformer, "module") else transformer.config
    _, num_channels, num_frames, height, width = noisy_latents.shape
    vae_config = kwargs.get("vae_config")
    cog_vae_scale_factor_spatial = 2 ** (len(vae_config["block_out_channels"]) - 1)
    cog_rope_base_height = denoiser_config.sample_height * cog_vae_scale_factor_spatial
    cog_rope_base_width = denoiser_config.sample_width * cog_vae_scale_factor_spatial

    image_rotary_emb = (
        prepare_rotary_positional_embeddings(
            height=height * cog_vae_scale_factor_spatial,
            width=width * cog_vae_scale_factor_spatial,
            num_frames=num_frames,
            vae_scale_factor_spatial=cog_vae_scale_factor_spatial,
            patch_size=denoiser_config.patch_size,
            patch_size_t=denoiser_config.patch_size_t
            if hasattr(denoiser_config, "patch_size_t")
            else None,
            attention_head_dim=denoiser_config.attention_head_dim,
            device=transformer.device,
            base_height=cog_rope_base_height,
            base_width=cog_rope_base_width,
        )
        if denoiser_config.use_rotary_positional_embeddings
        else None
    )
    
    denoised_latents = transformer(
        hidden_states=noisy_latents,
        timestep=timesteps,
        encoder_hidden_states=prompt_embeds,
        ofs=ofs_emb,
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

def calculate_timesteps(scheduler, latent_conditions, generator):
    batch_size = latent_conditions["latents"].shape[0]
    timesteps = torch.randint(
        0,
        scheduler.config.num_train_timesteps,
        (batch_size,),
        dtype=torch.int64,
        device=latent_conditions["latents"].device,
        generator=generator,
    )
    return timesteps

def calculate_loss(denoiser, model_config, scheduler, latent_conditions, text_conditions, timesteps):
    batch_size = latent_conditions["noisy_latents"].shape[0]
    scheduler_alphas_cumprod = (
        scheduler.alphas_cumprod.clone().to(denoiser.device, dtype=torch.float32)
    )

    model_pred = model_config["forward_pass"](
        transformer=denoiser, timesteps=timesteps, **latent_conditions, **text_conditions
    )["latents"]
    model_pred = scheduler.get_velocity(model_pred, latent_conditions["noisy_latents"], timesteps)
    
    weights = 1 / (1 - scheduler_alphas_cumprod[timesteps])
    weights = expand_tensor_to_dims(weights, latent_conditions["noisy_latents"].ndim)
    target = latent_conditions["latents"].view_as(model_pred)
    
    loss = torch.mean(
        (weights * (model_pred - target) ** 2).reshape(batch_size, -1),
        dim=1,
    )
    loss = loss.mean()
    return loss


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


def _pad_frames(latents, denoiser_config=None):
    if denoiser_config:
        # `latents` should be of the following format: [B, C, F, H, W].
        # For CogVideoX 1.5, the latent frames should be padded to make it divisible by patch_size_t
        latent_num_frames = latents.shape[2]
        additional_frames = 0
        patch_size_t = denoiser_config.patch_size_t
        if patch_size_t is not None and latent_num_frames % patch_size_t != 0:
            additional_frames = patch_size_t - latent_num_frames % patch_size_t
        
        if additional_frames:
            last_frame = latents[:, :, -1:, :, :]
            padding_frames = last_frame.repeat(1, 1, additional_frames, 1, 1)
            latents = torch.cat([latents, padding_frames], dim=2)

    return latents


def _scale_latents(latents, vae_config):
    if not vae_config["invert_scale_latents"]:
        latents = latents * vae_config["scaling_factor"]
    else:
        latents = 1 / vae_config["scaling_factor"] * latents
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
    "calculate_timesteps": calculate_timesteps,
    "calculate_loss": calculate_loss,
    "validation": validation,
}
