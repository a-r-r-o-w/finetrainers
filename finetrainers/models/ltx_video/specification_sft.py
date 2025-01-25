from typing import Dict, List, Optional, Tuple

import torch
from diffusers import AutoencoderKLLTXVideo, FlowMatchEulerDiscreteScheduler, LTXPipeline, LTXVideoTransformer3DModel
from transformers import AutoModel, AutoTokenizer, T5EncoderModel, T5Tokenizer

from ... import functional as FF
from ...utils import get_non_null_items
from ..modeling_utils import ModelSpecification


class LTXVideoModelSpecification(ModelSpecification):
    pipeline_cls = LTXPipeline

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

    def load_condition_models(self, *args, **kwargs) -> Dict[str, torch.nn.Module]:
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
            "text": [x["prompt"] for x in batch[0]],
            "image_or_video": image_or_video,
        }

    def prepare_conditions(
        self,
        text: str,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        max_sequence_length: int = 128,
        *args,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        conditions = {}
        for condition in self.conditions:
            result = self._prepare_condition(
                condition,
                text=text,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                max_sequence_length=max_sequence_length,
            )
            conditions.update(result)
        return conditions

    def prepare_latents(
        self,
        vae: AutoencoderKLLTXVideo,
        image_or_video: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        precompute: bool = False,
        *args,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        device = vae.device
        dtype = vae.dtype

        if image_or_video.ndim == 4:
            image_or_video = image_or_video.unsqueeze(2)
        assert image_or_video.ndim == 5, f"Expected 5D tensor, got {image_or_video.ndim}D tensor"

        image_or_video = image_or_video.to(device=device, dtype=vae.dtype)
        image_or_video = image_or_video.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, F, H, W] -> [B, F, C, H, W]

        if not precompute:
            patch_size = self.transformer_config.patch_size
            patch_size_t = self.transformer_config.patch_size_t
            latents = vae.encode(image_or_video).latent_dist.sample(generator=generator)
            latents = latents.to(dtype=dtype)
            _, _, num_frames, height, width = latents.shape
            latents = self._normalize_latents(latents, vae.latents_mean, vae.latents_std)
            latents = self._pack_latents(latents, patch_size, patch_size_t)

            return {"latents": latents, "num_frames": num_frames, "height": height, "width": width}
        else:
            if vae.use_slicing and image_or_video.shape[0] > 1:
                encoded_slices = [vae._encode(x_slice) for x_slice in image_or_video.split(1)]
                h = torch.cat(encoded_slices)
            else:
                h = vae._encode(image_or_video)
            _, _, num_frames, height, width = h.shape

            # TODO(aryan): This is very stupid that we might possibly be storing the latents_mean and latents_std in every file
            # if precomputation is enabled. We should probably have a single file where re-usable properties like this are stored
            # so as to reduce the disk memory requirements of the precomputed files.
            return {
                "latents": h,
                "num_frames": num_frames,
                "height": height,
                "width": width,
                "latents_mean": vae.latents_mean,
                "latents_std": vae.latents_std,
            }

    def postprocess_precomputed_conditions(
        self,
        condition_model_conditions: Dict[str, torch.Tensor],
        latent_model_conditions: Dict[str, torch.Tensor],
        *args,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        latents = latent_model_conditions["latents"]
        latents_mean = latent_model_conditions["latents_mean"]
        latents_std = latent_model_conditions["latents_std"]

        patch_size = self.transformer_config.patch_size
        patch_size_t = self.transformer_config.patch_size_t
        latents = self._normalize_latents(latents, latents_mean, latents_std)
        latents = self._pack_latents(latents, patch_size, patch_size_t)

        latent_model_conditions["latents"] = latents
        return condition_model_conditions, latent_model_conditions

    def forward(
        self,
        transformer: LTXVideoTransformer3DModel,
        condition_model_conditions: Dict[str, torch.Tensor],
        latent_model_conditions: Dict[str, torch.Tensor],
        sigmas: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        *args,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        latents = latent_model_conditions.pop("latents")
        noise = torch.zeros_like(latents).normal_(generator=generator)
        noisy_latents = FF.flow_match_xt(latents, noise, sigmas)
        latent_model_conditions["hidden_states"] = noisy_latents.to(latents)

        timesteps = (sigmas * 1000.0).long()

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

        pred = transformer(
            **latent_model_conditions,
            **condition_model_conditions,
            timestep=timesteps,
            rope_interpolation_scale=rope_interpolation_scale,
            return_dict=False,
        )[0]
        target = FF.flow_match_target(noise, latents)

        return {
            "pred": pred,
            "target": target,
        }

    def validation(
        self,
        pipeline: LTXPipeline,
        prompt: str,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        frame_rate: int = 25,
        num_videos_per_prompt: int = 1,
        generator: Optional[torch.Generator] = None,
        *args,
        **kwargs,
    ) -> List[Tuple[str, torch.Tensor]]:
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
        }
        generation_kwargs = get_non_null_items(generation_kwargs)
        video = pipeline(**generation_kwargs).frames[0]
        return [("video", video)]

    def _normalize_latents(
        latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0
    ) -> torch.Tensor:
        # Normalize latents across the channel dimension [B, C, F, H, W]
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents = (latents - latents_mean) * scaling_factor / latents_std
        return latents

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
