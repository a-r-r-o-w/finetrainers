import os
from typing import Dict, List, Optional, Tuple

import safetensors.torch
import torch
from accelerate import init_empty_weights
from diffusers import AutoencoderKL, CogView4Pipeline, CogView4Transformer2DModel, FlowMatchEulerDiscreteScheduler

from ... import functional as FF
from ...processors import CogView4GLMProcessor, ProcessorMixin
from ...typing import SchedulerType
from ..modeling_utils import ControlModelSpecification
from ..utils import DiagonalGaussianDistribution, _expand_linear_with_zeroed_weights
from .base_specification import CogView4LatentEncodeProcessor, CogView4ModelSpecification


class CogView4ControlModelSpecification(CogView4ModelSpecification, ControlModelSpecification):
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

    def load_diffusion_models(self, new_in_features: int) -> Dict[str, torch.nn.Module]:
        if self.transformer_id is not None:
            transformer = CogView4Transformer2DModel.from_pretrained(
                self.transformer_id,
                torch_dtype=self.transformer_dtype,
                revision=self.revision,
                cache_dir=self.cache_dir,
            )
        else:
            transformer = CogView4Transformer2DModel.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="transformer",
                torch_dtype=self.transformer_dtype,
                revision=self.revision,
                cache_dir=self.cache_dir,
            )

        actual_new_in_features = new_in_features * transformer.config.patch_size**2
        transformer.patch_embed.proj = _expand_linear_with_zeroed_weights(
            transformer.patch_embed.proj, new_in_features=actual_new_in_features
        )
        transformer.register_to_config(in_channels=new_in_features)

        scheduler = FlowMatchEulerDiscreteScheduler()

        return {"transformer": transformer, "scheduler": scheduler}

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
        conditions = {"image": image, "video": video, **common_kwargs, **kwargs}
        input_keys = set(conditions.keys())
        conditions = ControlModelSpecification.prepare_latents(self, self.latent_model_processors, **conditions)
        conditions = {k: v for k, v in conditions.items() if k not in input_keys}

        control_conditions = {"image": control_image, "video": control_video, **common_kwargs, **kwargs}
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

    def _save_lora_weights(
        self,
        directory: str,
        transformer_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        norm_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        scheduler: Optional[SchedulerType] = None,
        *args,
        **kwargs,
    ) -> None:
        # TODO(aryan): this needs refactoring
        if transformer_state_dict is not None:
            CogView4Pipeline.save_lora_weights(directory, transformer_state_dict, safe_serialization=True)
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
    def _original_in_features(self):
        return self.transformer_config.in_channels

    @property
    def _control_layer_pattern(self):
        return "patch_embed.proj"

    @property
    def _qk_norm_identifiers(self):
        return ["attn1.norm_q", "attn1.norm_k"]
