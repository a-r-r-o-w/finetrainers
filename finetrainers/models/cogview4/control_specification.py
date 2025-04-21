import functools
import os
from typing import Any, Dict, List, Optional, Tuple

import safetensors.torch
import torch
from accelerate import init_empty_weights
from diffusers import AutoencoderKL, CogView4Pipeline, CogView4Transformer2DModel, FlowMatchEulerDiscreteScheduler
from transformers import AutoTokenizer, GlmModel

import finetrainers.functional as FF
from finetrainers.data import ImageArtifact
from finetrainers.models.modeling_utils import ControlModelSpecification
from finetrainers.models.utils import DiagonalGaussianDistribution, _expand_linear_with_zeroed_weights
from finetrainers.patches.dependencies.diffusers.control import control_channel_concat
from finetrainers.processors import CogView4GLMProcessor, ProcessorMixin
from finetrainers.typing import ArtifactType, SchedulerType
from finetrainers.utils import _enable_vae_memory_optimizations, get_non_null_items, safetensors_torch_save_function

from .base_specification import CogView4LatentEncodeProcessor


class CogView4ControlModelSpecification(ControlModelSpecification):
    def __init__(
        self,
        pretrained_model_name_or_path: str = "THUDM/CogView4-6B",
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
            condition_model_processors = [CogView4GLMProcessor(["encoder_hidden_states"])]
        if latent_model_processors is None:
            latent_model_processors = [
                CogView4LatentEncodeProcessor(["latents", "original_size", "target_size", "crop_coords"])
            ]
        if control_model_processors is None:
            control_model_processors = [
                CogView4LatentEncodeProcessor(["control_latents", "original_size", "target_size", "crop_coords"])
            ]

        self.condition_model_processors = condition_model_processors
        self.latent_model_processors = latent_model_processors
        self.control_model_processors = control_model_processors

    @property
    def control_injection_layer_name(self):
        return "patch_embed.proj"

    @property
    def _resolution_dim_keys(self):
        return {"latents": (2, 3)}

    def load_condition_models(self) -> Dict[str, torch.nn.Module]:
        common_kwargs = {"revision": self.revision, "cache_dir": self.cache_dir}

        if self.tokenizer_id is not None:
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id, **common_kwargs)
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                self.pretrained_model_name_or_path, subfolder="tokenizer", **common_kwargs
            )

        if self.text_encoder_id is not None:
            text_encoder = GlmModel.from_pretrained(
                self.text_encoder_id, torch_dtype=self.text_encoder_dtype, **common_kwargs
            )
        else:
            text_encoder = GlmModel.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="text_encoder",
                torch_dtype=self.text_encoder_dtype,
                **common_kwargs,
            )

        return {"tokenizer": tokenizer, "text_encoder": text_encoder}

    def load_latent_models(self) -> Dict[str, torch.nn.Module]:
        common_kwargs = {"revision": self.revision, "cache_dir": self.cache_dir}

        if self.vae_id is not None:
            vae = AutoencoderKL.from_pretrained(self.vae_id, torch_dtype=self.vae_dtype, **common_kwargs)
        else:
            vae = AutoencoderKL.from_pretrained(
                self.pretrained_model_name_or_path, subfolder="vae", torch_dtype=self.vae_dtype, **common_kwargs
            )

        return {"vae": vae}

    def load_diffusion_models(self, new_in_features: int) -> Dict[str, torch.nn.Module]:
        common_kwargs = {"revision": self.revision, "cache_dir": self.cache_dir}

        if self.transformer_id is not None:
            transformer = CogView4Transformer2DModel.from_pretrained(
                self.transformer_id, torch_dtype=self.transformer_dtype, **common_kwargs
            )
        else:
            transformer = CogView4Transformer2DModel.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="transformer",
                torch_dtype=self.transformer_dtype,
                **common_kwargs,
            )

        actual_new_in_features = new_in_features * transformer.config.patch_size**2
        transformer.patch_embed.proj = _expand_linear_with_zeroed_weights(
            transformer.patch_embed.proj, new_in_features=actual_new_in_features
        )
        transformer.register_to_config(in_channels=new_in_features)

        scheduler = FlowMatchEulerDiscreteScheduler()

        return {"transformer": transformer, "scheduler": scheduler}

    def load_pipeline(
        self,
        tokenizer: Optional[AutoTokenizer] = None,
        text_encoder: Optional[GlmModel] = None,
        transformer: Optional[CogView4Transformer2DModel] = None,
        vae: Optional[AutoencoderKL] = None,
        scheduler: Optional[FlowMatchEulerDiscreteScheduler] = None,
        enable_slicing: bool = False,
        enable_tiling: bool = False,
        enable_model_cpu_offload: bool = False,
        training: bool = False,
        **kwargs,
    ) -> CogView4Pipeline:
        components = {
            "tokenizer": tokenizer,
            "text_encoder": text_encoder,
            "transformer": transformer,
            "vae": vae,
            # Load the scheduler based on CogView4's config instead of using the default initialization being used for training
            # "scheduler": scheduler,
        }
        components = get_non_null_items(components)

        pipe = CogView4Pipeline.from_pretrained(
            self.pretrained_model_name_or_path, **components, revision=self.revision, cache_dir=self.cache_dir
        )
        pipe.text_encoder.to(self.text_encoder_dtype)
        pipe.vae.to(self.vae_dtype)

        _enable_vae_memory_optimizations(pipe.vae, enable_slicing, enable_tiling)
        if not training:
            pipe.transformer.to(self.transformer_dtype)
        if enable_model_cpu_offload:
            pipe.enable_model_cpu_offload()

        return pipe

    @torch.no_grad()
    def prepare_conditions(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: GlmModel,
        caption: str,
        max_sequence_length: int = 1024,
        **kwargs,
    ) -> Dict[str, Any]:
        conditions = {
            "tokenizer": tokenizer,
            "text_encoder": text_encoder,
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
        vae: AutoencoderKL,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        control_image: Optional[torch.Tensor] = None,
        control_video: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        compute_posterior: bool = True,
        _original_height: Optional[int] = None,
        _original_width: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        common_kwargs = {
            "vae": vae,
            "generator": generator,
            "compute_posterior": compute_posterior,
            "_original_height": _original_height,
            "_original_width": _original_width,
            **kwargs,
        }
        conditions = {"image": image, "video": video, **common_kwargs}
        input_keys = set(conditions.keys())
        conditions = ControlModelSpecification.prepare_latents(self, self.latent_model_processors, **conditions)
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
        transformer: CogView4Transformer2DModel,
        condition_model_conditions: Dict[str, torch.Tensor],
        latent_model_conditions: Dict[str, torch.Tensor],
        sigmas: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        compute_posterior: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        base_image_sequence_length = 256
        base_shift = 0.25
        max_shift = 0.75

        if compute_posterior:
            latents = latent_model_conditions.pop("latents")
            control_latents = latent_model_conditions.pop("control_latents")
        else:
            posterior = DiagonalGaussianDistribution(latent_model_conditions.pop("latents"))
            latents = posterior.sample(generator=generator)
            del posterior

            control_posterior = DiagonalGaussianDistribution(latent_model_conditions.pop("control_latents"))
            control_latents = control_posterior.sample(generator=generator)
            del control_posterior

        if getattr(self.vae_config, "shift_factor") is not None:
            latents = (latents - self.vae_config.shift_factor) * self.vae_config.scaling_factor
            control_latents = (control_latents - self.vae_config.shift_factor) * self.vae_config.scaling_factor

        noise = torch.zeros_like(latents).normal_(generator=generator)
        timesteps = (sigmas.flatten() * 1000.0).long()

        image_sequence_length = latents.size(2) * latents.size(3) // self.transformer_config.patch_size**2
        mu = (image_sequence_length / base_image_sequence_length) ** 0.5
        mu = mu * max_shift + base_shift
        shifted_sigmas = mu / (mu + (1 / sigmas - 1) ** 1.0)
        noisy_latents = FF.flow_match_xt(latents, noise, shifted_sigmas)
        noisy_latents = torch.cat([noisy_latents, control_latents], dim=1)

        latent_model_conditions["hidden_states"] = noisy_latents.to(latents)

        pred = transformer(
            **latent_model_conditions,
            **condition_model_conditions,
            timestep=timesteps,
            return_dict=False,
        )[0]
        target = FF.flow_match_target(noise, latents)

        # NOTE: shifted_sigmas loss weighting seems to work better than sigmas. Needs more investigation
        # but let's keep it this way for now. Longer training runs should reveal more insights.
        # return pred, target, sigmas
        return pred, target, shifted_sigmas

    def validation(
        self,
        pipeline: CogView4Pipeline,
        prompt: str,
        control_image: torch.Tensor,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> List[ArtifactType]:
        with torch.no_grad():
            dtype = pipeline.vae.dtype
            device = pipeline._execution_device
            in_channels = self.transformer_config.in_channels  # We need to use the original in_channels
            latents = pipeline.prepare_latents(1, in_channels, height, width, dtype, device, generator)
            control_image = pipeline.image_processor.preprocess(control_image, height=height, width=width)
            control_image = control_image.to(device=device, dtype=dtype)
            control_latents = pipeline.vae.encode(control_image).latent_dist.sample(generator=generator)
            if getattr(self.vae_config, "shift_factor") is not None:
                control_latents = (control_latents - self.vae_config.shift_factor) * self.vae_config.scaling_factor

        generation_kwargs = {
            "latents": latents,
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "generator": generator,
            "return_dict": True,
            "output_type": "pil",
        }
        generation_kwargs = get_non_null_items(generation_kwargs)

        with control_channel_concat(pipeline.transformer, ["hidden_states"], [control_latents], dims=[1]):
            image = pipeline(**generation_kwargs).images[0]

        return [ImageArtifact(value=image)]

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
            CogView4Pipeline.save_lora_weights(
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
        transformer: CogView4Transformer2DModel,
        transformer_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        scheduler: Optional[SchedulerType] = None,
    ) -> None:
        # TODO(aryan): this needs refactoring
        if transformer_state_dict is not None:
            with init_empty_weights():
                transformer_copy = CogView4Transformer2DModel.from_config(transformer.config)
            transformer_copy.load_state_dict(transformer_state_dict, strict=True, assign=True)
            transformer_copy.save_pretrained(os.path.join(directory, "transformer"))
        if scheduler is not None:
            scheduler.save_pretrained(os.path.join(directory, "scheduler"))

    @property
    def _original_control_layer_in_features(self):
        return self.transformer_config.in_channels

    @property
    def _original_control_layer_out_features(self):
        return self.transformer_config.num_attention_heads * self.transformer_config.attention_head_dim

    @property
    def _qk_norm_identifiers(self):
        return ["attn1.norm_q", "attn1.norm_k"]
