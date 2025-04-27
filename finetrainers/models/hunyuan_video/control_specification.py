import functools
import os
from typing import Any, Dict, List, Optional, Tuple

import safetensors
import torch
from accelerate import init_empty_weights
from diffusers import (
    AutoencoderKLHunyuanVideo,
    FlowMatchEulerDiscreteScheduler,
    HunyuanVideoPipeline,
    HunyuanVideoTransformer3DModel,
)
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer, LlamaModel

from finetrainers.data._artifact import VideoArtifact
from finetrainers.models.utils import _expand_conv3d_with_zeroed_weights
from finetrainers.trainer.control_trainer.config import FrameConditioningType
from finetrainers.utils.serialization import safetensors_torch_save_function

from ... import functional as FF
from ...logging import get_logger
from ...patches.dependencies.diffusers.control import control_channel_concat
from ...processors import CLIPPooledProcessor, LlamaProcessor, ProcessorMixin
from ...typing import ArtifactType, SchedulerType
from ...utils import _enable_vae_memory_optimizations, get_non_null_items
from ..modeling_utils import ControlModelSpecification
from .base_specification import HunyuanLatentEncodeProcessor


logger = get_logger()


class HunyuanVideoControlModelSpecification(ControlModelSpecification):
    def __init__(
        self,
        pretrained_model_name_or_path: str = "hunyuanvideo-community/HunyuanVideo",
        tokenizer_id: Optional[str] = None,
        text_encoder_id: Optional[str] = None,
        transformer_id: Optional[str] = None,
        vae_id: Optional[str] = None,
        text_encoder_dtype: torch.dtype = torch.bfloat16,
        transformer_dtype: torch.dtype = torch.bfloat16,
        vae_dtype: torch.dtype = torch.bfloat16,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        condition_model_processors: List[ProcessorMixin] = None,
        latent_model_processors: List[ProcessorMixin] = None,
        control_model_processors: List[ProcessorMixin] = None,
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

        if condition_model_processors is None:
            condition_model_processors = [
                LlamaProcessor(["encoder_hidden_states", "encoder_attention_mask"]),
                CLIPPooledProcessor(
                    ["pooled_projections"],
                    input_names={"tokenizer_2": "tokenizer", "text_encoder_2": "text_encoder"},
                ),
            ]
        if latent_model_processors is None:
            latent_model_processors = [HunyuanLatentEncodeProcessor(["latents", "latents_mean", "latents_std"])]
        if control_model_processors is None:
            control_model_processors = [HunyuanLatentEncodeProcessor(["control_latents", "__drop__", "__drop__"])]

        self.condition_model_processors = condition_model_processors
        self.latent_model_processors = latent_model_processors
        self.control_model_processors = control_model_processors

    @property
    def control_injection_layer_name(self) -> str:
        return "x_embedder.proj"

    @property
    def _resolution_dim_keys(self):
        return {"latents": (2, 3, 4)}

    def load_condition_models(self) -> Dict[str, torch.nn.Module]:
        common_kwargs = {"revision": self.revision, "cache_dir": self.cache_dir}

        if self.tokenizer_id is not None:
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id, **common_kwargs)
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                self.pretrained_model_name_or_path, subfolder="tokenizer", **common_kwargs
            )

        if self.tokenizer_2_id is not None:
            tokenizer_2 = AutoTokenizer.from_pretrained(self.tokenizer_2_id, **common_kwargs)
        else:
            tokenizer_2 = CLIPTokenizer.from_pretrained(
                self.pretrained_model_name_or_path, subfolder="tokenizer_2", **common_kwargs
            )

        if self.text_encoder_id is not None:
            text_encoder = LlamaModel.from_pretrained(
                self.text_encoder_id, torch_dtype=self.text_encoder_dtype, **common_kwargs
            )
        else:
            text_encoder = LlamaModel.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="text_encoder",
                torch_dtype=self.text_encoder_dtype,
                **common_kwargs,
            )

        if self.text_encoder_2_id is not None:
            text_encoder_2 = CLIPTextModel.from_pretrained(
                self.text_encoder_2_id, torch_dtype=self.text_encoder_2_dtype, **common_kwargs
            )
        else:
            text_encoder_2 = CLIPTextModel.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="text_encoder_2",
                torch_dtype=self.text_encoder_2_dtype,
                **common_kwargs,
            )

        return {
            "tokenizer": tokenizer,
            "tokenizer_2": tokenizer_2,
            "text_encoder": text_encoder,
            "text_encoder_2": text_encoder_2,
        }

    def load_latent_models(self) -> Dict[str, torch.nn.Module]:
        common_kwargs = {"revision": self.revision, "cache_dir": self.cache_dir}

        if self.vae_id is not None:
            vae = AutoencoderKLHunyuanVideo.from_pretrained(self.vae_id, torch_dtype=self.vae_dtype, **common_kwargs)
        else:
            vae = AutoencoderKLHunyuanVideo.from_pretrained(
                self.pretrained_model_name_or_path, subfolder="vae", torch_dtype=self.vae_dtype, **common_kwargs
            )

        return {"vae": vae}

    def load_diffusion_models(self, new_in_features: int) -> Dict[str, torch.nn.Module]:
        common_kwargs = {"revision": self.revision, "cache_dir": self.cache_dir}

        if self.transformer_id is not None:
            transformer = HunyuanVideoTransformer3DModel.from_pretrained(
                self.transformer_id, torch_dtype=self.transformer_dtype, **common_kwargs
            )
        else:
            transformer = HunyuanVideoTransformer3DModel.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="transformer",
                torch_dtype=self.transformer_dtype,
                **common_kwargs,
            )

        transformer.x_embedder.proj = _expand_conv3d_with_zeroed_weights(
            transformer.x_embedder.proj, new_in_channels=new_in_features
        )
        transformer.register_to_config(in_channels=new_in_features)
        scheduler = FlowMatchEulerDiscreteScheduler()

        return {"transformer": transformer, "scheduler": scheduler}

    def load_pipeline(
        self,
        tokenizer: Optional[AutoTokenizer] = None,
        tokenizer_2: Optional[CLIPTokenizer] = None,
        text_encoder: Optional[LlamaModel] = None,
        text_encoder_2: Optional[CLIPTextModel] = None,
        transformer: Optional[HunyuanVideoTransformer3DModel] = None,
        vae: Optional[AutoencoderKLHunyuanVideo] = None,
        scheduler: Optional[FlowMatchEulerDiscreteScheduler] = None,
        enable_slicing: bool = False,
        enable_tiling: bool = False,
        enable_model_cpu_offload: bool = False,
        training: bool = False,
        **kwargs,
    ) -> HunyuanVideoPipeline:
        components = {
            "tokenizer": tokenizer,
            "tokenizer_2": tokenizer_2,
            "text_encoder": text_encoder,
            "text_encoder_2": text_encoder_2,
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
        }
        components = get_non_null_items(components)

        pipe = HunyuanVideoPipeline.from_pretrained(
            self.pretrained_model_name_or_path, **components, revision=self.revision, cache_dir=self.cache_dir
        )
        pipe.text_encoder.to(self.text_encoder_dtype)
        pipe.text_encoder_2.to(self.text_encoder_2_dtype)
        pipe.vae.to(self.vae_dtype)

        _enable_vae_memory_optimizations(pipe.vae, enable_slicing, enable_tiling)
        if not training:
            pipe.transformer.to(self.transformer_dtype)

        # TODO(aryan): add support in diffusers
        # if enable_slicing:
        #     pipe.vae.enable_slicing()
        # if enable_tiling:
        #     pipe.vae.enable_tiling()
        if enable_model_cpu_offload:
            pipe.enable_model_cpu_offload()

        return pipe

    @torch.no_grad()
    def prepare_conditions(
        self,
        tokenizer: AutoTokenizer,
        tokenizer_2: CLIPTokenizer,
        text_encoder: LlamaModel,
        text_encoder_2: CLIPTextModel,
        caption: str,
        max_sequence_length: int = 256,
        **kwargs,
    ) -> Dict[str, Any]:
        conditions = {
            "tokenizer": tokenizer,
            "tokenizer_2": tokenizer_2,
            "text_encoder": text_encoder,
            "text_encoder_2": text_encoder_2,
            "caption": caption,
            "max_sequence_length": max_sequence_length,
            **kwargs,
        }
        input_keys = set(conditions.keys())
        conditions = super().prepare_conditions(**conditions)
        conditions = {k: v for k, v in conditions.items() if k not in input_keys}
        return conditions

    @torch.no_grad()
    def prepare_latents(
        self,
        vae: AutoencoderKLHunyuanVideo,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        control_image: Optional[torch.Tensor] = None,
        control_video: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        compute_posterior: bool = True,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        common_kwargs = {
            "vae": vae,
            "generator": generator,
            "compute_posterior": compute_posterior,
            **kwargs,
        }
        conditions = {"image": image, "video": video, **common_kwargs}
        input_keys = set(conditions.keys())
        conditions = super().prepare_latents(**conditions)
        conditions = {k: v for k, v in conditions.items() if k not in input_keys}

        control_conditions = {"image": control_image, "video": control_video, **common_kwargs}
        input_keys = set(control_conditions.keys())
        control_conditions = ControlModelSpecification.prepare_latents(
            self, self.control_model_processors, **control_conditions
        )
        control_conditions = {k: v for k, v in control_conditions.items() if k not in input_keys}

        return {**control_conditions, **conditions}

    def forward(
        self,
        transformer: HunyuanVideoTransformer3DModel,
        condition_model_conditions: Dict[str, torch.Tensor],
        latent_model_conditions: Dict[str, torch.Tensor],
        sigmas: torch.Tensor,
        guidance: float = 1.0,
        generator: Optional[torch.Generator] = None,
        compute_posterior: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        from finetrainers.trainer.control_trainer.data import apply_frame_conditioning_on_latents

        compute_posterior = False
        if compute_posterior:
            latents = latent_model_conditions.pop("latents")
            control_latents = latent_model_conditions.pop("control_latents")
        else:
            latents = latent_model_conditions.pop("latents")
            control_latents = latent_model_conditions.pop("control_latents")
            latents_mean = latent_model_conditions.pop("latents_mean")
            latents_std = latent_model_conditions.pop("latents_std")

            mu, logvar = torch.chunk(latents, 2, dim=1)
            mu = self._normalize_latents(mu, latents_mean, latents_std)
            logvar = self._normalize_latents(logvar, latents_mean, latents_std)
            latents = torch.cat([mu, logvar], dim=1)

            mu, logvar = torch.chunk(control_latents, 2, dim=1)
            mu = self._normalize_latents(mu, latents_mean, latents_std)
            logvar = self._normalize_latents(logvar, latents_mean, latents_std)
            control_latents = torch.cat([mu, logvar], dim=1)

            posterior = DiagonalGaussianDistribution(latents)
            latents = posterior.mode()
            del posterior

            control_posterior = DiagonalGaussianDistribution(control_latents)
            control_latents = control_posterior.mode()
            del control_posterior

        noise = torch.zeros_like(latents).normal_(generator=generator)
        timesteps = (sigmas.flatten() * 1000.0).long()
        guidance = latents.new_full((latents.size(0),), fill_value=guidance) * 1000.0

        noisy_latents = FF.flow_match_xt(latents, noise, sigmas)
        control_latents = apply_frame_conditioning_on_latents(
            control_latents,
            noisy_latents.shape[2],
            channel_dim=1,
            frame_dim=2,
            frame_conditioning_type=self.frame_conditioning_type,
            frame_conditioning_index=self.frame_conditioning_index,
            concatenate_mask=self.frame_conditioning_concatenate_mask,
        )
        noisy_latents = torch.cat([noisy_latents, control_latents], dim=1)

        latent_model_conditions["hidden_states"] = noisy_latents.to(latents)
        latent_model_conditions["guidance"] = guidance

        pred = transformer(
            **latent_model_conditions,
            **condition_model_conditions,
            timestep=timesteps,
            return_dict=False,
        )[0]
        target = FF.flow_match_target(noise, latents)

        return pred, target, sigmas

    def validation(
        self,
        pipeline: HunyuanVideoPipeline,
        prompt: str,
        control_image: Optional[torch.Tensor] = None,
        control_video: Optional[torch.Tensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = None,
        frame_conditioning_type: "FrameConditioningType" = "full",
        frame_conditioning_index: int = 0,
        **kwargs,
    ) -> List[ArtifactType]:
        from finetrainers.trainer.control_trainer.data import apply_frame_conditioning_on_latents

        with torch.no_grad():
            dtype = pipeline.vae.dtype
            device = pipeline._execution_device
            in_channels = self.transformer_config.in_channels  # We need to use the original in_channels
            latents = pipeline.prepare_latents(1, in_channels, height, width, num_frames, dtype, device, generator)
            latents_mean = (
                torch.tensor(self.vae_config.latents_mean)
                .view(1, self.vae_config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae_config.latents_std).view(1, self.vae_config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )

            if control_image is not None:
                control_video = pipeline.video_processor.preprocess(
                    control_image, height=height, width=width
                ).unsqueeze(2)
            else:
                control_video = pipeline.video_processor.preprocess_video(control_video, height=height, width=width)

            control_video = control_video.to(device=device, dtype=dtype)
            control_latents = pipeline.vae.encode(control_video).latent_dist.mode()
            control_latents = self._normalize_latents(control_latents, latents_mean, latents_std)
            control_latents = apply_frame_conditioning_on_latents(
                control_latents,
                latents.shape[2],
                channel_dim=1,
                frame_dim=2,
                frame_conditioning_type=frame_conditioning_type,
                frame_conditioning_index=frame_conditioning_index,
                concatenate_mask=self.frame_conditioning_concatenate_mask,
            )

        generation_kwargs = {
            "latents": latents,
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "generator": generator,
            "return_dict": True,
            "output_type": "pil",
        }
        generation_kwargs = get_non_null_items(generation_kwargs)

        with control_channel_concat(pipeline.transformer, ["hidden_states"], [control_latents], dims=[1]):
            video = pipeline(**generation_kwargs).frames[0]

        return [VideoArtifact(value=video)]

    def _save_lora_weights(
        self,
        directory: str,
        transformer_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        norm_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        scheduler: Optional[SchedulerType] = None,
        metadata: Optional[Dict[str, str]] = None,
        *args,
        **kwargs,
    ) -> None:
        # TODO(aryan): this needs refactoring
        if transformer_state_dict is not None:
            HunyuanVideoPipeline.save_lora_weights(
                directory,
                transformer_state_dict,
                save_function=functools.partial(safetensors_torch_save_function, metadata=metadata),
                safe_serialization=True,
            )
        if norm_state_dict is not None:
            safetensors.torch.save_file(norm_state_dict, os.path.join(directory, "norm_state_dict.safetensors"))
        if scheduler is not None:
            scheduler.save_pretrained(os.path.join(directory, "scheduler"))

    def _save_model(
        self,
        directory: str,
        transformer: HunyuanVideoTransformer3DModel,
        transformer_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        scheduler: Optional[SchedulerType] = None,
    ) -> None:
        # TODO(aryan): this needs refactoring
        if transformer_state_dict is not None:
            with init_empty_weights():
                transformer_copy = HunyuanVideoTransformer3DModel.from_config(transformer.config)
            transformer_copy.load_state_dict(transformer_state_dict, strict=True, assign=True)
            transformer_copy.save_pretrained(os.path.join(directory, "transformer"))
        if scheduler is not None:
            scheduler.save_pretrained(os.path.join(directory, "scheduler"))

    @staticmethod
    def _normalize_latents(
        latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor
    ) -> torch.Tensor:
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(device=latents.device)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(device=latents.device)
        latents = ((latents.float() - latents_mean) * latents_std).to(latents)
        return latents

    @property
    def _original_control_layer_in_features(self):
        return self.transformer_config.in_channels

    @property
    def _original_control_layer_out_features(self):
        return self.transformer_config.num_attention_heads * self.transformer_config.attention_head_dim

    @property
    def _qk_norm_identifiers(self):
        return ["norm_q", "norm_k", "norm_added_q", "norm_added_k"]
