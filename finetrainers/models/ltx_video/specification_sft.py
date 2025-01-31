import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from diffusers import (
    AutoencoderKLLTXVideo,
    FlowMatchEulerDiscreteScheduler,
    LTXImageToVideoPipeline,
    LTXPipeline,
    LTXVideoTransformer3DModel,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils.import_utils import is_torch_version
from PIL.Image import Image
from transformers import AutoModel, AutoTokenizer, T5EncoderModel, T5Tokenizer

from ... import functional as FF
from ...processors import get_condition
from ...utils import get_non_null_items
from ..modeling_utils import ModelSpecification


class LTXVideoModelSpecification(ModelSpecification):
    def __init__(
        self,
        pretrained_model_name_or_path: str = "Lightricks/LTX-Video",
        tokenizer_id: Optional[str] = None,
        text_encoder_id: Optional[str] = None,
        transformer_id: Optional[str] = None,
        vae_id: Optional[str] = None,
        text_encoder_dtype: torch.dtype = torch.bfloat16,
        transformer_dtype: torch.dtype = torch.bfloat16,
        vae_dtype: torch.dtype = torch.bfloat16,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            tokenizer_id=tokenizer_id,
            text_encoder_id=text_encoder_id,
            transformer_id=transformer_id,
            vae_id=vae_id,
            text_encoder_dtype=text_encoder_dtype,
            transformer_dtype=transformer_dtype,
            vae_dtype=vae_dtype,
            revision=revision,
            cache_dir=cache_dir,
        )

        LTXVideoTransformer3DModel.forward = _patched_LTXVideoTransformer3Dforward

    def load_condition_models(self, condition_types: List[str], *args, **kwargs) -> Dict[str, torch.nn.Module]:
        if self.tokenizer_id is not None:
            tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_id, revision=self.revision, cache_dir=self.cache_dir
            )
        else:
            tokenizer = T5Tokenizer.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="tokenizer",
                revision=self.revision,
                cache_dir=self.cache_dir,
            )

        if self.text_encoder_id is not None:
            text_encoder = AutoModel.from_pretrained(
                self.text_encoder_id,
                torch_dtype=self.text_encoder_dtype,
                revision=self.revision,
                cache_dir=self.cache_dir,
            )
        else:
            text_encoder = T5EncoderModel.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="text_encoder",
                torch_dtype=self.text_encoder_dtype,
                revision=self.revision,
                cache_dir=self.cache_dir,
            )

        for condition_type in condition_types:
            condition = get_condition(condition_type, {"tokenizer": tokenizer, "text_encoder": text_encoder})
            self._add_condition(condition_type, condition)

        return {"tokenizer": tokenizer, "text_encoder": text_encoder}

    def load_latent_models(self, *args, **kwargs) -> Dict[str, torch.nn.Module]:
        if self.vae_id is not None:
            vae = AutoencoderKLLTXVideo.from_pretrained(
                self.vae_id,
                torch_dtype=self.vae_dtype,
                revision=self.revision,
                cache_dir=self.cache_dir,
            )
        else:
            vae = AutoencoderKLLTXVideo.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="vae",
                torch_dtype=self.vae_dtype,
                revision=self.revision,
                cache_dir=self.cache_dir,
            )

        return {"vae": vae}

    def load_diffusion_models(self, *args, **kwargs) -> Dict[str, torch.nn.Module]:
        if self.transformer_id is not None:
            transformer = LTXVideoTransformer3DModel.from_pretrained(
                self.transformer_id,
                torch_dtype=self.transformer_dtype,
                revision=self.revision,
                cache_dir=self.cache_dir,
            )
        else:
            transformer = LTXVideoTransformer3DModel.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="transformer",
                torch_dtype=self.transformer_dtype,
                revision=self.revision,
                cache_dir=self.cache_dir,
            )

        scheduler = FlowMatchEulerDiscreteScheduler()

        return {"transformer": transformer, "scheduler": scheduler}

    def load_pipeline(
        self,
        tokenizer: Optional[T5Tokenizer] = None,
        text_encoder: Optional[T5EncoderModel] = None,
        transformer: Optional[LTXVideoTransformer3DModel] = None,
        vae: Optional[AutoencoderKLLTXVideo] = None,
        scheduler: Optional[FlowMatchEulerDiscreteScheduler] = None,
        enable_slicing: bool = False,
        enable_tiling: bool = False,
        enable_model_cpu_offload: bool = False,
        training: bool = False,
        device: Optional[torch.device] = None,
        *args,
        **kwargs,
    ) -> LTXPipeline:
        components = {
            "tokenizer": tokenizer,
            "text_encoder": text_encoder,
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
        }
        components = get_non_null_items(components)

        pipe = LTXPipeline.from_pretrained(
            self.pretrained_model_name_or_path, **components, revision=self.revision, cache_dir=self.cache_dir
        )
        pipe.text_encoder.to(self.text_encoder_dtype)
        pipe.vae.to(self.vae_dtype)

        if not training:
            pipe.transformer.to(self.transformer_dtype)

        if enable_slicing:
            pipe.vae.enable_slicing()
        if enable_tiling:
            pipe.vae.enable_tiling()
        if enable_model_cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

        return pipe

    def collate_fn(self, batch: List[List[Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        if "image" in batch[0][0]:
            image_or_video = torch.stack([x["image"] for x in batch[0]])
        else:
            image_or_video = torch.stack([x["video"] for x in batch[0]])
        return {
            "text": [x["text"] for x in batch[0]],
            "image_or_video": image_or_video,
        }

    @torch.no_grad()
    def prepare_conditions(
        self,
        caption: str,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        max_sequence_length: int = 128,
        *args,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        conditions = {}
        for _, condition in self.conditions.items():
            result = self._prepare_condition(
                condition,
                caption=caption,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                max_sequence_length=max_sequence_length,
            )
            conditions.update(result)
        return conditions

    @torch.no_grad()
    def prepare_latents(
        self,
        vae: AutoencoderKLLTXVideo,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        precompute: bool = False,
        *args,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        device = vae.device
        dtype = vae.dtype

        # TODO(aryan): remove this
        video = video[:, :49, :, :, :]

        if image is not None:
            video = image.unsqueeze(1)

        assert video.ndim == 5, f"Expected 5D tensor, got {video.ndim}D tensor"
        video = video.to(device=device, dtype=vae.dtype)
        video = video.permute(0, 2, 1, 3, 4).contiguous()  # [B, F, C, H, W] -> [B, C, F, H, W]

        if not precompute:
            latents = vae.encode(video).latent_dist.sample(generator=generator)
            latents = latents.to(dtype=dtype)
        else:
            if vae.use_slicing and video.shape[0] > 1:
                encoded_slices = [vae._encode(x_slice) for x_slice in video.split(1)]
                h = torch.cat(encoded_slices)
            else:
                h = vae._encode(video)
            latents = h

        _, _, num_frames, height, width = latents.shape

        return {
            "latents": latents,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "latents_mean": vae.latents_mean,
            "latents_std": vae.latents_std,
        }

    def forward(
        self,
        transformer: LTXVideoTransformer3DModel,
        condition_model_conditions: Dict[str, torch.Tensor],
        latent_model_conditions: Dict[str, torch.Tensor],
        sigmas: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        # TODO(aryan): make this configurable? Should it be?
        first_frame_conditioning_p = 0.1
        min_first_frame_sigma = 0.25

        latents = latent_model_conditions.pop("latents")
        latents_mean = latent_model_conditions.pop("latents_mean")
        latents_std = latent_model_conditions.pop("latents_std")

        latents = self._normalize_latents(latents, latents_mean, latents_std)
        noise = torch.zeros_like(latents).normal_(generator=generator)

        if random.random() < first_frame_conditioning_p:
            # Based on Section 2.4 of the paper, it mentions that the first frame timesteps should be a small random value.
            # Making as estimated guess, we limit the sigmas to be at least 0.2.
            # torch.rand_like returns values in [0, 1). We want to make sure that the first frame sigma is <= actual sigmas
            # for image conditioning. In order to do this, we rescale by multiplying with sigmas so the range is [0, sigmas).
            first_frame_sigma = torch.rand_like(sigmas) * sigmas
            first_frame_sigma = torch.min(first_frame_sigma, sigmas.new_full(sigmas.shape, min_first_frame_sigma))

            latents_first_frame, latents_rest = latents[:, :, :1], latents[:, :, 1:]
            noisy_latents_first_frame = FF.flow_match_xt(latents_first_frame, noise[:, :, :1], first_frame_sigma)
            noisy_latents_remaining = FF.flow_match_xt(latents_rest, noise[:, :, 1:], sigmas)
            noisy_latents = torch.cat([noisy_latents_first_frame, noisy_latents_remaining], dim=2)
        else:
            noisy_latents = FF.flow_match_xt(latents, noise, sigmas)

        patch_size = self.transformer_config.patch_size
        patch_size_t = self.transformer_config.patch_size_t

        latents = self._pack_latents(latents, patch_size, patch_size_t)
        noise = self._pack_latents(noise, patch_size, patch_size_t)
        noisy_latents = self._pack_latents(noisy_latents, patch_size, patch_size_t)

        sigmas = sigmas.view(-1, 1, 1).expand(-1, *noisy_latents.shape[1:-1], -1)

        latent_model_conditions["hidden_states"] = noisy_latents.to(latents)
        condition_model_conditions["encoder_hidden_states"] = condition_model_conditions.pop("prompt_embeds")
        condition_model_conditions["encoder_attention_mask"] = condition_model_conditions.pop("prompt_attention_mask")

        # TODO(aryan): make this configurable
        frame_rate = 25
        temporal_compression_ratio = 8
        vae_spatial_compression_ratio = 32
        latent_frame_rate = frame_rate / temporal_compression_ratio

        rope_interpolation_scale = [
            1 / latent_frame_rate,
            vae_spatial_compression_ratio,
            vae_spatial_compression_ratio,
        ]
        timesteps = (sigmas * 1000.0).long()

        pred = transformer(
            **latent_model_conditions,
            **condition_model_conditions,
            timestep=timesteps,
            rope_interpolation_scale=rope_interpolation_scale,
            return_dict=False,
        )[0]
        target = FF.flow_match_target(noise, latents)

        return pred, target, sigmas

    def validation(
        self,
        pipeline: LTXPipeline,
        prompt: str,
        image: Optional[Image] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        frame_rate: int = 25,
        num_videos_per_prompt: int = 1,
        generator: Optional[torch.Generator] = None,
        *args,
        **kwargs,
    ) -> List[Tuple[str, torch.Tensor]]:
        if image is not None:
            pipeline = LTXImageToVideoPipeline.from_pipe(pipeline)

        generation_kwargs = {
            "prompt": prompt,
            "image": image,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "frame_rate": frame_rate,
            "num_videos_per_prompt": num_videos_per_prompt,
            "generator": generator,
            "return_dict": True,
            "output_type": "pil",
        }
        generation_kwargs = get_non_null_items(generation_kwargs)
        video = pipeline(**generation_kwargs).frames[0]
        return [("video", video)]

    def save_lora_weights(self, directory: str, transformer_layers: List[torch.nn.Parameter]) -> None:
        LTXPipeline.save_lora_weights(directory, transformer_layers)

    def save_model(
        self,
        directory: str,
        transformer: Optional[LTXVideoTransformer3DModel] = None,
        scheduler: Optional[FlowMatchEulerDiscreteScheduler] = None,
        *args,
        **kwargs,
    ) -> None:
        directory = Path(directory)
        if transformer is not None:
            transformer.save_pretrained((directory / "transformer").as_posix())
        if scheduler is not None:
            scheduler.save_pretrained((directory / "scheduler").as_posix())

    @staticmethod
    def _normalize_latents(
        latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0
    ) -> torch.Tensor:
        # Normalize latents across the channel dimension [B, C, F, H, W]
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents = (latents - latents_mean) * scaling_factor / latents_std
        return latents

    @staticmethod
    def _pack_latents(latents: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1) -> torch.Tensor:
        # Unpacked latents of shape are [B, C, F, H, W] are patched into tokens of shape [B, C, F // p_t, p_t, H // p, p, W // p, p].
        # The patch dimensions are then permuted and collapsed into the channel dimension of shape:
        # [B, F // p_t * H // p * W // p, C * p_t * p * p] (an ndim=3 tensor).
        # dim=0 is the batch size, dim=1 is the effective video sequence length, dim=2 is the effective number of input features
        batch_size, num_channels, num_frames, height, width = latents.shape
        post_patch_num_frames = num_frames // patch_size_t
        post_patch_height = height // patch_size
        post_patch_width = width // patch_size
        latents = latents.reshape(
            batch_size,
            -1,
            post_patch_num_frames,
            patch_size_t,
            post_patch_height,
            patch_size,
            post_patch_width,
            patch_size,
        )
        latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
        return latents


def _patched_LTXVideoTransformer3Dforward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_attention_mask: torch.Tensor,
    num_frames: int,
    height: int,
    width: int,
    rope_interpolation_scale: Optional[Tuple[float, float, float]] = None,
    return_dict: bool = True,
    *args,
    **kwargs,
) -> torch.Tensor:
    image_rotary_emb = self.rope(hidden_states, num_frames, height, width, rope_interpolation_scale)

    # convert encoder_attention_mask to a bias the same way we do for attention_mask
    if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
        encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
        encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

    batch_size = hidden_states.size(0)

    # ===== This is modified compared to Diffusers =====
    # This is done because the Diffusers pipeline will pass in a 1D tensor for timestep
    if timestep.ndim == 1:
        timestep = timestep.view(-1, 1, 1).expand(-1, *hidden_states.shape[1:-1], -1)
    # ==================================================

    temb, embedded_timestep = self.time_embed(
        timestep.flatten(),
        batch_size=batch_size,
        hidden_dtype=hidden_states.dtype,
    )

    # ===== This is modified compared to Diffusers =====
    # temb = temb.view(batch_size, -1, temb.size(-1))
    # embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.size(-1))
    # ==================================================
    # This is done to make it possible to use per-token timestep embedding
    temb = temb.view(batch_size, *hidden_states.shape[1:-1], temb.size(-1))
    embedded_timestep = embedded_timestep.view(batch_size, *hidden_states.shape[1:-1], embedded_timestep.size(-1))
    # ==================================================

    hidden_states = self.proj_in(hidden_states)

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
