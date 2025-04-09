import functools
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from accelerate import init_empty_weights
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxPipeline, FluxTransformer2DModel
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer, T5EncoderModel

import finetrainers.functional as FF
from finetrainers.data import ImageArtifact
from finetrainers.logging import get_logger
from finetrainers.models.modeling_utils import ModelSpecification
from finetrainers.processors import CLIPPooledProcessor, ProcessorMixin, T5Processor
from finetrainers.typing import ArtifactType, SchedulerType
from finetrainers.utils import _enable_vae_memory_optimizations, get_non_null_items, safetensors_torch_save_function


logger = get_logger()


class FluxLatentEncodeProcessor(ProcessorMixin):
    r"""
    Processor to encode image/video into latents using the Flux VAE.

    Args:
        output_names (`List[str]`):
            The names of the outputs that the processor returns. The outputs are in the following order:
            - latents: The latents of the input image/video.
    """

    def __init__(self, output_names: List[str]):
        super().__init__()

        self.output_names = output_names
        assert len(self.output_names) == 1

    def forward(
        self,
        vae: AutoencoderKL,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        compute_posterior: bool = True,
    ) -> Dict[str, torch.Tensor]:
        device = vae.device
        dtype = vae.dtype

        if video is not None:
            # TODO(aryan): perhaps better would be to flatten(0, 1), but need to account for reshaping sigmas accordingly
            image = video[:, 0]  # [B, F, C, H, W] -> [B, 1, C, H, W]

        assert image.ndim == 4, f"Expected 4D tensor, got {image.ndim}D tensor"
        image = image.to(device=device, dtype=vae.dtype)

        if compute_posterior:
            latents = vae.encode(image).latent_dist.sample(generator=generator)
            latents = latents.to(dtype=dtype)
        else:
            if vae.use_slicing and image.shape[0] > 1:
                encoded_slices = [vae._encode(x_slice) for x_slice in image.split(1)]
                moments = torch.cat(encoded_slices)
            else:
                moments = vae._encode(image)
            latents = moments.to(dtype=dtype)

        return {self.output_names[0]: latents}


class FluxModelSpecification(ModelSpecification):
    def __init__(
        self,
        pretrained_model_name_or_path: str = "black-forest-labs/FLUX.1-dev",
        tokenizer_id: Optional[str] = None,
        tokenizer_2_id: Optional[str] = None,
        text_encoder_id: Optional[str] = None,
        text_encoder_2_id: Optional[str] = None,
        transformer_id: Optional[str] = None,
        vae_id: Optional[str] = None,
        text_encoder_dtype: torch.dtype = torch.bfloat16,
        transformer_dtype: torch.dtype = torch.bfloat16,
        vae_dtype: torch.dtype = torch.bfloat16,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        condition_model_processors: List[ProcessorMixin] = None,
        latent_model_processors: List[ProcessorMixin] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            tokenizer_id=tokenizer_id,
            tokenizer_2_id=tokenizer_2_id,
            text_encoder_id=text_encoder_id,
            text_encoder_2_id=text_encoder_2_id,
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
                CLIPPooledProcessor(["pooled_projections"]),
                T5Processor(
                    ["encoder_hidden_states", "prompt_attention_mask"],
                    input_names={"tokenizer_2": "tokenizer", "text_encoder_2": "text_encoder"},
                ),
            ]
        if latent_model_processors is None:
            latent_model_processors = [FluxLatentEncodeProcessor(["latents"])]

        self.condition_model_processors = condition_model_processors
        self.latent_model_processors = latent_model_processors

    @property
    def _resolution_dim_keys(self):
        return {"latents": (2, 3)}

    def load_condition_models(self) -> Dict[str, torch.nn.Module]:
        common_kwargs = {"revision": self.revision, "cache_dir": self.cache_dir}

        if self.tokenizer_id is not None:
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id, **common_kwargs)
        else:
            tokenizer = CLIPTokenizer.from_pretrained(
                self.pretrained_model_name_or_path, subfolder="tokenizer", **common_kwargs
            )

        if self.tokenizer_2_id is not None:
            tokenizer_2 = AutoTokenizer.from_pretrained(self.tokenizer_2_id, **common_kwargs)
        else:
            tokenizer_2 = AutoTokenizer.from_pretrained(
                self.pretrained_model_name_or_path, subfolder="tokenizer_2", **common_kwargs
            )

        if self.text_encoder_id is not None:
            text_encoder = CLIPTextModel.from_pretrained(
                self.text_encoder_id, torch_dtype=self.text_encoder_dtype, **common_kwargs
            )
        else:
            text_encoder = CLIPTextModel.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="text_encoder",
                torch_dtype=self.text_encoder_dtype,
                **common_kwargs,
            )

        if self.text_encoder_2_id is not None:
            text_encoder_2 = T5EncoderModel.from_pretrained(
                self.text_encoder_2_id, torch_dtype=self.text_encoder_2_dtype, **common_kwargs
            )
        else:
            text_encoder_2 = T5EncoderModel.from_pretrained(
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
            vae = AutoencoderKL.from_pretrained(self.vae_id, torch_dtype=self.vae_dtype, **common_kwargs)
        else:
            vae = AutoencoderKL.from_pretrained(
                self.pretrained_model_name_or_path, subfolder="vae", torch_dtype=self.vae_dtype, **common_kwargs
            )

        return {"vae": vae}

    def load_diffusion_models(self) -> Dict[str, torch.nn.Module]:
        common_kwargs = {"revision": self.revision, "cache_dir": self.cache_dir}

        if self.transformer_id is not None:
            transformer = FluxTransformer2DModel.from_pretrained(
                self.transformer_id, torch_dtype=self.transformer_dtype, **common_kwargs
            )
        else:
            transformer = FluxTransformer2DModel.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="transformer",
                torch_dtype=self.transformer_dtype,
                **common_kwargs,
            )

        scheduler = FlowMatchEulerDiscreteScheduler()

        return {"transformer": transformer, "scheduler": scheduler}

    def load_pipeline(
        self,
        tokenizer: Optional[AutoTokenizer] = None,
        tokenizer_2: Optional[CLIPTokenizer] = None,
        text_encoder: Optional[CLIPTextModel] = None,
        text_encoder_2: Optional[T5EncoderModel] = None,
        transformer: Optional[FluxTransformer2DModel] = None,
        vae: Optional[AutoencoderKL] = None,
        scheduler: Optional[FlowMatchEulerDiscreteScheduler] = None,
        enable_slicing: bool = False,
        enable_tiling: bool = False,
        enable_model_cpu_offload: bool = False,
        training: bool = False,
        **kwargs,
    ) -> FluxPipeline:
        components = {
            "tokenizer": tokenizer,
            "tokenizer_2": tokenizer_2,
            "text_encoder": text_encoder,
            "text_encoder_2": text_encoder_2,
            "transformer": transformer,
            "vae": vae,
            # Load the scheduler based on Flux's config instead of using the default initialization being used for training
            # "scheduler": scheduler,
        }
        components = get_non_null_items(components)

        pipe = FluxPipeline.from_pretrained(
            self.pretrained_model_name_or_path, **components, revision=self.revision, cache_dir=self.cache_dir
        )
        pipe.text_encoder.to(self.text_encoder_dtype)
        pipe.text_encoder_2.to(self.text_encoder_2_dtype)
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
        tokenizer_2: CLIPTokenizer,
        text_encoder: CLIPTextModel,
        text_encoder_2: T5EncoderModel,
        caption: str,
        max_sequence_length: int = 512,
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
        vae: AutoencoderKL,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        compute_posterior: bool = True,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        conditions = {
            "vae": vae,
            "image": image,
            "video": video,
            "generator": generator,
            "compute_posterior": compute_posterior,
            **kwargs,
        }
        input_keys = set(conditions.keys())
        conditions = super().prepare_latents(**conditions)
        conditions = {k: v for k, v in conditions.items() if k not in input_keys}
        return conditions

    def forward(
        self,
        transformer: FluxTransformer2DModel,
        condition_model_conditions: Dict[str, torch.Tensor],
        latent_model_conditions: Dict[str, torch.Tensor],
        sigmas: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        compute_posterior: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        if compute_posterior:
            latents = latent_model_conditions.pop("latents")
        else:
            posterior = DiagonalGaussianDistribution(latent_model_conditions.pop("latents"))
            latents = posterior.sample(generator=generator)
            del posterior

        if getattr(self.vae_config, "shift_factor", None) is not None:
            latents = (latents - self.vae_config.shift_factor) * self.vae_config.scaling_factor
        else:
            latents = latents * self.vae_config.scaling_factor

        noise = torch.zeros_like(latents).normal_(generator=generator)
        timesteps = (sigmas.flatten() * 1000.0).long()
        img_ids = FluxPipeline._prepare_latent_image_ids(
            latents.size(0), latents.size(2) // 2, latents.size(3) // 2, latents.device, latents.dtype
        )
        text_ids = latents.new_zeros(condition_model_conditions["encoder_hidden_states"].shape[1], 3)

        if self.transformer_config.guidance_embeds:
            guidance_scale = 1.0
            guidance = latents.new_full((1,), guidance_scale).expand(latents.shape[0])
        else:
            guidance = None

        noisy_latents = FF.flow_match_xt(latents, noise, sigmas)
        noisy_latents = FluxPipeline._pack_latents(noisy_latents, *latents.shape)

        latent_model_conditions["hidden_states"] = noisy_latents.to(latents)
        condition_model_conditions.pop("prompt_attention_mask", None)

        spatial_compression_ratio = 2 ** len(self.vae_config.block_out_channels)
        pred = transformer(
            **latent_model_conditions,
            **condition_model_conditions,
            timestep=timesteps / 1000.0,
            guidance=guidance,
            img_ids=img_ids,
            txt_ids=text_ids,
            return_dict=False,
        )[0]
        pred = FluxPipeline._unpack_latents(
            pred,
            latents.size(2) * spatial_compression_ratio,
            latents.size(3) * spatial_compression_ratio,
            spatial_compression_ratio,
        )
        target = FF.flow_match_target(noise, latents)

        return pred, target, sigmas

    def validation(
        self,
        pipeline: FluxPipeline,
        prompt: str,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.5,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> List[ArtifactType]:
        generation_kwargs = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "return_dict": True,
            "output_type": "pil",
        }
        generation_kwargs = get_non_null_items(generation_kwargs)
        image = pipeline(**generation_kwargs).images[0]
        return [ImageArtifact(value=image)]

    def _save_lora_weights(
        self,
        directory: str,
        transformer_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        scheduler: Optional[SchedulerType] = None,
        metadata: Optional[Dict[str, str]] = None,
        *args,
        **kwargs,
    ) -> None:
        # TODO(aryan): this needs refactoring
        if transformer_state_dict is not None:
            FluxPipeline.save_lora_weights(
                directory,
                transformer_state_dict,
                save_function=functools.partial(safetensors_torch_save_function, metadata=metadata),
                safe_serialization=True,
            )
        if scheduler is not None:
            scheduler.save_pretrained(os.path.join(directory, "scheduler"))

    def _save_model(
        self,
        directory: str,
        transformer: FluxTransformer2DModel,
        transformer_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        scheduler: Optional[SchedulerType] = None,
    ) -> None:
        # TODO(aryan): this needs refactoring
        if transformer_state_dict is not None:
            with init_empty_weights():
                transformer_copy = FluxTransformer2DModel.from_config(transformer.config)
            transformer_copy.load_state_dict(transformer_state_dict, strict=True, assign=True)
            transformer_copy.save_pretrained(os.path.join(directory, "transformer"))
        if scheduler is not None:
            scheduler.save_pretrained(os.path.join(directory, "scheduler"))
