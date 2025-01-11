from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from accelerate.logging import get_logger
from diffusers import (
    AutoencoderKLHunyuanVideo,
    FlowMatchEulerDiscreteScheduler,
    HunyuanVideoPipeline,
    HunyuanVideoTransformer3DModel,
)
from PIL import Image
from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer, LlamaModel, LlamaTokenizer


logger = get_logger("finetrainers")  # pylint: disable=invalid-name


def load_condition_models(
    model_id: str = "hunyuanvideo-community/HunyuanVideo",
    text_encoder_dtype: torch.dtype = torch.float16,
    text_encoder_2_dtype: torch.dtype = torch.float16,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> Dict[str, nn.Module]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer", revision=revision, cache_dir=cache_dir)
    text_encoder = LlamaModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=text_encoder_dtype, revision=revision, cache_dir=cache_dir
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        model_id, subfolder="tokenizer_2", revision=revision, cache_dir=cache_dir
    )
    text_encoder_2 = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder_2", torch_dtype=text_encoder_2_dtype, revision=revision, cache_dir=cache_dir
    )
    return {
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "tokenizer_2": tokenizer_2,
        "text_encoder_2": text_encoder_2,
    }


def load_latent_models(
    model_id: str = "hunyuanvideo-community/HunyuanVideo",
    vae_dtype: torch.dtype = torch.float16,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> Dict[str, nn.Module]:
    vae = AutoencoderKLHunyuanVideo.from_pretrained(
        model_id, subfolder="vae", torch_dtype=vae_dtype, revision=revision, cache_dir=cache_dir
    )
    return {"vae": vae}


def load_diffusion_models(
    model_id: str = "hunyuanvideo-community/HunyuanVideo",
    transformer_dtype: torch.dtype = torch.bfloat16,
    shift: float = 1.0,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> Dict[str, Union[nn.Module, FlowMatchEulerDiscreteScheduler]]:
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=transformer_dtype, revision=revision, cache_dir=cache_dir
    )
    scheduler = FlowMatchEulerDiscreteScheduler(shift=shift)
    return {"transformer": transformer, "scheduler": scheduler}


def initialize_pipeline(
    model_id: str = "hunyuanvideo-community/HunyuanVideo",
    text_encoder_dtype: torch.dtype = torch.float16,
    text_encoder_2_dtype: torch.dtype = torch.float16,
    transformer_dtype: torch.dtype = torch.bfloat16,
    vae_dtype: torch.dtype = torch.float16,
    tokenizer: Optional[LlamaTokenizer] = None,
    text_encoder: Optional[LlamaModel] = None,
    tokenizer_2: Optional[CLIPTokenizer] = None,
    text_encoder_2: Optional[CLIPTextModel] = None,
    transformer: Optional[HunyuanVideoTransformer3DModel] = None,
    vae: Optional[AutoencoderKLHunyuanVideo] = None,
    scheduler: Optional[FlowMatchEulerDiscreteScheduler] = None,
    device: Optional[torch.device] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    enable_slicing: bool = False,
    enable_tiling: bool = False,
    enable_model_cpu_offload: bool = False,
    **kwargs,
) -> HunyuanVideoPipeline:
    component_name_pairs = [
        ("tokenizer", tokenizer),
        ("text_encoder", text_encoder),
        ("tokenizer_2", tokenizer_2),
        ("text_encoder_2", text_encoder_2),
        ("transformer", transformer),
        ("vae", vae),
        ("scheduler", scheduler),
    ]
    components = {}
    for name, component in component_name_pairs:
        if component is not None:
            components[name] = component

    pipe = HunyuanVideoPipeline.from_pretrained(model_id, **components, revision=revision, cache_dir=cache_dir)
    pipe.text_encoder = pipe.text_encoder.to(dtype=text_encoder_dtype)
    pipe.text_encoder_2 = pipe.text_encoder_2.to(dtype=text_encoder_2_dtype)
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
    tokenizer: LlamaTokenizer,
    text_encoder: LlamaModel,
    tokenizer_2: CLIPTokenizer,
    text_encoder_2: CLIPTextModel,
    prompt: Union[str, List[str]],
    guidance: float = 1.0,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    max_sequence_length: int = 256,
    # TODO(aryan): make configurable
    prompt_template: Dict[str, Any] = {
        "template": (
            "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: "
            "1. The main content and theme of the video."
            "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
            "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
            "4. background environment, light, style and atmosphere."
            "5. camera angles, movements, and transitions used in the video:<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
        ),
        "crop_start": 95,
    },
    **kwargs,
) -> torch.Tensor:
    device = device or text_encoder.device
    dtype = dtype or text_encoder.dtype

    if isinstance(prompt, str):
        prompt = [prompt]

    conditions = {}
    conditions.update(
        _get_llama_prompt_embeds(tokenizer, text_encoder, prompt, prompt_template, device, dtype, max_sequence_length)
    )
    conditions.update(_get_clip_prompt_embeds(tokenizer_2, text_encoder_2, prompt, device, dtype))

    guidance = torch.tensor([guidance], device=device, dtype=dtype) * 1000.0
    conditions["guidance"] = guidance

    return conditions


def prepare_latents(
    vae: AutoencoderKLHunyuanVideo,
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
    image_or_video = image_or_video.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, F, H, W] -> [B, F, C, H, W]
    if not precompute:
        latents = vae.encode(image_or_video).latent_dist.sample(generator=generator)
        latents = latents * vae.config.scaling_factor
        latents = latents.to(dtype=dtype)
        return {"latents": latents}
    else:
        if vae.use_slicing and image_or_video.shape[0] > 1:
            encoded_slices = [vae._encode(x_slice) for x_slice in image_or_video.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = vae._encode(image_or_video)
        return {"latents": h}


def post_latent_preparation(
    vae_config: Dict[str, Any],
    latents: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    latents = latents * vae_config.scaling_factor
    return {"latents": latents}


def collate_fn_t2v(batch: List[List[Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
    return {
        "prompts": [x["prompt"] for x in batch[0]],
        "videos": torch.stack([x["video"] for x in batch[0]]),
    }


def forward_pass(
    transformer: HunyuanVideoTransformer3DModel,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    guidance: torch.Tensor,
    latents: torch.Tensor,
    noisy_latents: torch.Tensor,
    timesteps: torch.LongTensor,
    **kwargs,
) -> torch.Tensor:
    denoised_latents = transformer(
        hidden_states=noisy_latents,
        timestep=timesteps,
        encoder_hidden_states=prompt_embeds,
        pooled_projections=pooled_prompt_embeds,
        encoder_attention_mask=prompt_attention_mask,
        guidance=guidance,
        return_dict=False,
    )[0]

    return {"latents": denoised_latents}


def validation(
    pipeline: HunyuanVideoPipeline,
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


def _get_llama_prompt_embeds(
    tokenizer: LlamaTokenizer,
    text_encoder: LlamaModel,
    prompt: List[str],
    prompt_template: Dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    max_sequence_length: int = 256,
    num_hidden_layers_to_skip: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(prompt)
    prompt = [prompt_template["template"].format(p) for p in prompt]

    crop_start = prompt_template.get("crop_start", None)
    if crop_start is None:
        prompt_template_input = tokenizer(
            prompt_template["template"],
            padding="max_length",
            return_tensors="pt",
            return_length=False,
            return_overflowing_tokens=False,
            return_attention_mask=False,
        )
        crop_start = prompt_template_input["input_ids"].shape[-1]
        # Remove <|eot_id|> token and placeholder {}
        crop_start -= 2

    max_sequence_length += crop_start
    text_inputs = tokenizer(
        prompt,
        max_length=max_sequence_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_length=False,
        return_overflowing_tokens=False,
        return_attention_mask=True,
    )
    text_input_ids = text_inputs.input_ids.to(device=device)
    prompt_attention_mask = text_inputs.attention_mask.to(device=device)

    prompt_embeds = text_encoder(
        input_ids=text_input_ids,
        attention_mask=prompt_attention_mask,
        output_hidden_states=True,
    ).hidden_states[-(num_hidden_layers_to_skip + 1)]
    prompt_embeds = prompt_embeds.to(dtype=dtype)

    if crop_start is not None and crop_start > 0:
        prompt_embeds = prompt_embeds[:, crop_start:]
        prompt_attention_mask = prompt_attention_mask[:, crop_start:]

    prompt_attention_mask = prompt_attention_mask.view(batch_size, -1)

    return {"prompt_embeds": prompt_embeds, "prompt_attention_mask": prompt_attention_mask}


def _get_clip_prompt_embeds(
    tokenizer_2: CLIPTokenizer,
    text_encoder_2: CLIPTextModel,
    prompt: Union[str, List[str]],
    device: torch.device,
    dtype: torch.dtype,
    max_sequence_length: int = 77,
) -> torch.Tensor:
    text_inputs = tokenizer_2(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )

    prompt_embeds = text_encoder_2(text_inputs.input_ids.to(device), output_hidden_states=False).pooler_output
    prompt_embeds = prompt_embeds.to(dtype=dtype)

    return {"pooled_prompt_embeds": prompt_embeds}


# TODO(aryan): refactor into model specs for better re-use
HUNYUAN_VIDEO_T2V_LORA_CONFIG = {
    "pipeline_cls": HunyuanVideoPipeline,
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
