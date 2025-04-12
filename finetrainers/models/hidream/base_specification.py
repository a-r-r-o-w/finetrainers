import functools
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from accelerate import init_empty_weights
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    HiDreamImagePipeline,
    HiDreamImageTransformer2DModel,
)
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from transformers import (
    AutoTokenizer,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    LlamaForCausalLM,
    T5EncoderModel,
    T5Tokenizer,
)

import finetrainers.functional as FF
from finetrainers.data import ImageArtifact
from finetrainers.logging import get_logger
from finetrainers.models.modeling_utils import ModelSpecification
from finetrainers.processors import (
    CLIPTextModelWithProjectionPooledProcessor,
    ProcessorMixin,
    T5Processor,
)
from finetrainers.typing import ArtifactType, SchedulerType
from finetrainers.utils import _enable_vae_memory_optimizations, get_non_null_items, safetensors_torch_save_function


logger = get_logger()


class HiDreamImageLlamaProcessor(ProcessorMixin):
    r"""
    Processor for the Llama family of models specific to HiDream. This processor is used to encode text inputs
    and return the embeddings and attention masks for the input text.

    Args:
        output_names (`List[str]`):
            The names of the outputs that the processor should return. The first output is the embeddings of the input
            text and the second output is the attention mask for the input text.
    """

    def __init__(
        self, output_names: List[str], input_names: Optional[Dict[str, Any]] = None, use_attention_mask: bool = False
    ):
        super().__init__()

        self.output_names = output_names
        self.input_names = input_names
        self.use_attention_mask = use_attention_mask

        assert input_names is None or len(input_names) <= 4
        assert len(self.output_names) == 1

    def forward(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: LlamaForCausalLM,
        caption: Union[str, List[str]],
        max_sequence_length: int,
    ) -> torch.Tensor:
        if isinstance(caption, str):
            caption = [caption]

        device = text_encoder.device
        dtype = text_encoder.dtype

        text_inputs = tokenizer(
            caption,
            padding="max_length",
            max_length=min(max_sequence_length, tokenizer.model_max_length),
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.to(device)

        outputs = text_encoder(
            text_input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True,
        )
        prompt_embeds = outputs.hidden_states[1:]
        prompt_embeds = torch.stack(prompt_embeds, dim=0).to(dtype=dtype)

        return {self.output_names[0]: prompt_embeds}


class HiDreamImageLatentEncodeProcessor(ProcessorMixin):
    r"""
    Processor to encode image/video into latents using the HiDream VAE.

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


class HiDreamImageModelSpecification(ModelSpecification):
    def __init__(
        self,
        pretrained_model_name_or_path: str = "HiDream-ai/HiDream-I1-Full",
        tokenizer_id: Optional[str] = None,
        tokenizer_2_id: Optional[str] = None,
        tokenizer_3_id: Optional[str] = None,
        tokenizer_4_id: Optional[str] = None,
        text_encoder_id: Optional[str] = None,
        text_encoder_2_id: Optional[str] = None,
        text_encoder_3_id: Optional[str] = None,
        text_encoder_4_id: Optional[str] = None,
        transformer_id: Optional[str] = None,
        vae_id: Optional[str] = None,
        text_encoder_dtype: torch.dtype = torch.bfloat16,
        text_encoder_2_dtype: torch.dtype = torch.bfloat16,
        text_encoder_3_dtype: torch.dtype = torch.bfloat16,
        text_encoder_4_dtype: torch.dtype = torch.bfloat16,
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
            tokenizer_3_id=tokenizer_3_id,
            tokenizer_4_id=tokenizer_4_id,
            text_encoder_id=text_encoder_id,
            text_encoder_2_id=text_encoder_2_id,
            text_encoder_3_id=text_encoder_3_id,
            text_encoder_4_id=text_encoder_4_id,
            transformer_id=transformer_id,
            vae_id=vae_id,
            text_encoder_dtype=text_encoder_dtype,
            text_encoder_2_dtype=text_encoder_2_dtype,
            text_encoder_3_dtype=text_encoder_3_dtype,
            text_encoder_4_dtype=text_encoder_4_dtype,
            transformer_dtype=transformer_dtype,
            vae_dtype=vae_dtype,
            revision=revision,
            cache_dir=cache_dir,
        )

        if condition_model_processors is None:
            condition_model_processors = [
                CLIPTextModelWithProjectionPooledProcessor(["pooled_prompt_embeds_1"]),
                CLIPTextModelWithProjectionPooledProcessor(
                    ["pooled_prompt_embeds_2"],
                    input_names={"tokenizer_2": "tokenizer", "text_encoder_2": "text_encoder"},
                ),
                T5Processor(
                    ["encoder_hidden_states_t5", "__drop__"],
                    input_names={"tokenizer_3": "tokenizer", "text_encoder_3": "text_encoder"},
                    use_attention_mask=True,
                ),
                HiDreamImageLlamaProcessor(
                    ["encoder_hidden_states_llama"],
                    input_names={"tokenizer_4": "tokenizer", "text_encoder_4": "text_encoder"},
                ),
            ]
        if latent_model_processors is None:
            latent_model_processors = [HiDreamImageLatentEncodeProcessor(["latents"])]

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
            tokenizer_2 = CLIPTokenizer.from_pretrained(
                self.pretrained_model_name_or_path, subfolder="tokenizer_2", **common_kwargs
            )
        if self.tokenizer_3_id is not None:
            tokenizer_3 = AutoTokenizer.from_pretrained(self.tokenizer_3_id, **common_kwargs)
        else:
            tokenizer_3 = T5Tokenizer.from_pretrained(
                self.pretrained_model_name_or_path, subfolder="tokenizer_3", **common_kwargs
            )
        if self.tokenizer_4_id is not None:
            tokenizer_4 = AutoTokenizer.from_pretrained(self.tokenizer_4_id, **common_kwargs)
        else:
            tokenizer_4 = AutoTokenizer.from_pretrained(
                self.pretrained_model_name_or_path, subfolder="tokenizer_4", **common_kwargs
            )

        if self.text_encoder_id is not None:
            text_encoder = CLIPTextModelWithProjection.from_pretrained(
                self.text_encoder_id, torch_dtype=self.text_encoder_dtype, **common_kwargs
            )
        else:
            text_encoder = CLIPTextModelWithProjection.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="text_encoder",
                torch_dtype=self.text_encoder_dtype,
                **common_kwargs,
            )
        if self.text_encoder_2_id is not None:
            text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
                self.text_encoder_2_id, torch_dtype=self.text_encoder_2_dtype, **common_kwargs
            )
        else:
            text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="text_encoder_2",
                torch_dtype=self.text_encoder_2_dtype,
                **common_kwargs,
            )
        if self.text_encoder_3_id is not None:
            text_encoder_3 = T5EncoderModel.from_pretrained(
                self.text_encoder_3_id, torch_dtype=self.text_encoder_3_dtype, **common_kwargs
            )
        else:
            text_encoder_3 = T5EncoderModel.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="text_encoder_3",
                torch_dtype=self.text_encoder_3_dtype,
                **common_kwargs,
            )
        if self.text_encoder_4_id is not None:
            text_encoder_4 = LlamaForCausalLM.from_pretrained(
                self.text_encoder_4_id,
                torch_dtype=self.text_encoder_4_dtype,
                output_hidden_states=True,
                output_attentions=True,
                **common_kwargs,
            )
        else:
            text_encoder_4 = LlamaForCausalLM.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="text_encoder_4",
                torch_dtype=self.text_encoder_4_dtype,
                output_hidden_states=True,
                output_attentions=True,
                **common_kwargs,
            )

        return {
            "tokenizer": tokenizer,
            "tokenizer_2": tokenizer_2,
            "tokenizer_3": tokenizer_3,
            "tokenizer_4": tokenizer_4,
            "text_encoder": text_encoder,
            "text_encoder_2": text_encoder_2,
            "text_encoder_3": text_encoder_3,
            "text_encoder_4": text_encoder_4,
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
            transformer = HiDreamImageTransformer2DModel.from_pretrained(
                self.transformer_id, torch_dtype=self.transformer_dtype, **common_kwargs
            )
        else:
            transformer = HiDreamImageTransformer2DModel.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="transformer",
                torch_dtype=self.transformer_dtype,
                **common_kwargs,
            )

        scheduler = FlowMatchEulerDiscreteScheduler()

        return {"transformer": transformer, "scheduler": scheduler}

    def load_pipeline(
        self,
        tokenizer: Optional[CLIPTokenizer] = None,
        tokenizer_2: Optional[CLIPTokenizer] = None,
        tokenizer_3: Optional[T5Tokenizer] = None,
        tokenizer_4: Optional[AutoTokenizer] = None,
        text_encoder: Optional[CLIPTextModelWithProjection] = None,
        text_encoder_2: Optional[CLIPTextModelWithProjection] = None,
        text_encoder_3: Optional[T5EncoderModel] = None,
        text_encoder_4: Optional[LlamaForCausalLM] = None,
        transformer: Optional[HiDreamImageTransformer2DModel] = None,
        vae: Optional[AutoencoderKL] = None,
        scheduler: Optional[FlowMatchEulerDiscreteScheduler] = None,
        enable_slicing: bool = False,
        enable_tiling: bool = False,
        enable_model_cpu_offload: bool = False,
        training: bool = False,
        **kwargs,
    ) -> HiDreamImagePipeline:
        components = {
            "tokenizer": tokenizer,
            "tokenizer_2": tokenizer_2,
            "tokenizer_3": tokenizer_3,
            "tokenizer_4": tokenizer_4,
            "text_encoder": text_encoder,
            "text_encoder_2": text_encoder_2,
            "text_encoder_3": text_encoder_3,
            "text_encoder_4": text_encoder_4,
            "transformer": transformer,
            "vae": vae,
            # Load the scheduler based on HiDream's config instead of using the default initialization being used for training
            # "scheduler": scheduler,
        }
        components = get_non_null_items(components)

        pipe = HiDreamImagePipeline.from_pretrained(
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
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        tokenizer_3: T5Tokenizer,
        tokenizer_4: AutoTokenizer,
        text_encoder: CLIPTextModelWithProjection,
        text_encoder_2: CLIPTextModelWithProjection,
        text_encoder_3: T5EncoderModel,
        text_encoder_4: LlamaForCausalLM,
        caption: str,
        max_sequence_length: int = 128,
        **kwargs,
    ) -> Dict[str, Any]:
        conditions = {
            "tokenizer": tokenizer,
            "tokenizer_2": tokenizer_2,
            "tokenizer_3": tokenizer_3,
            "tokenizer_4": tokenizer_4,
            "text_encoder": text_encoder,
            "text_encoder_2": text_encoder_2,
            "text_encoder_3": text_encoder_3,
            "text_encoder_4": text_encoder_4,
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
        transformer: HiDreamImageTransformer2DModel,
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
        noisy_latents = FF.flow_match_xt(latents, noise, sigmas)

        batch_size, num_channels, height, width = latents.shape
        img_sizes = img_ids = None
        p = self.transformer_config.patch_size
        if height != width:
            patch_height, patch_width = height // p, width // p

            img_sizes = torch.tensor([patch_height, patch_width], dtype=torch.int64).reshape(-1)
            img_ids = torch.zeros(patch_height, patch_width, 3)
            img_ids[..., 1] = img_ids[..., 1] + torch.arange(patch_height)[:, None]
            img_ids[..., 2] = img_ids[..., 2] + torch.arange(patch_width)[None, :]
            img_ids = img_ids.reshape(patch_height * patch_width, -1)
            img_ids_pad = torch.zeros(self.transformer.max_seq, 3)
            img_ids_pad[: patch_height * patch_width, :] = img_ids

            img_sizes = img_sizes.unsqueeze(0).to(latents.device)
            img_ids = img_ids_pad.unsqueeze(0).to(latents.device)

        pooled_prompt_embeds_1 = condition_model_conditions.pop("pooled_prompt_embeds_1")
        pooled_prompt_embeds_2 = condition_model_conditions.pop("pooled_prompt_embeds_2")
        pooled_embeds = torch.cat([pooled_prompt_embeds_1, pooled_prompt_embeds_2], dim=-1)
        encoder_hidden_states_t5 = condition_model_conditions.pop("encoder_hidden_states_t5")
        encoder_hidden_states_llama = condition_model_conditions.pop("encoder_hidden_states_llama")
        encoder_hidden_states = [encoder_hidden_states_t5, encoder_hidden_states_llama]

        latent_model_conditions["hidden_states"] = noisy_latents.to(latents)
        latent_model_conditions["encoder_hidden_states"] = encoder_hidden_states
        latent_model_conditions["pooled_embeds"] = pooled_embeds
        latent_model_conditions["img_ids"] = img_ids
        latent_model_conditions["img_sizes"] = img_sizes

        pred = transformer(
            **latent_model_conditions,
            **condition_model_conditions,
            timesteps=timesteps,
            return_dict=False,
        )[0]
        pred = pred.flatten(-2, -1).unflatten(-1, (height, width))
        pred = -pred
        target = FF.flow_match_target(noise, latents)

        return pred, target, sigmas

    def validation(
        self,
        pipeline: HiDreamImagePipeline,
        prompt: str,
        prompt_2: Optional[str] = None,
        prompt_3: Optional[str] = None,
        prompt_4: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        negative_prompt_2: Optional[str] = None,
        negative_prompt_3: Optional[str] = None,
        negative_prompt_4: Optional[str] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> List[ArtifactType]:
        generation_kwargs = {
            "prompt": prompt,
            "prompt_2": prompt_2,
            "prompt_3": prompt_3,
            "prompt_4": prompt_4,
            "negative_prompt": negative_prompt,
            "negative_prompt_2": negative_prompt_2,
            "negative_prompt_3": negative_prompt_3,
            "negative_prompt_4": negative_prompt_4,
            "height": height,
            "width": width,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
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
            HiDreamImagePipeline.save_lora_weights(
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
        transformer: HiDreamImageTransformer2DModel,
        transformer_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        scheduler: Optional[SchedulerType] = None,
    ) -> None:
        # TODO(aryan): this needs refactoring
        if transformer_state_dict is not None:
            with init_empty_weights():
                transformer_copy = HiDreamImageTransformer2DModel.from_config(transformer.config)
            transformer_copy.load_state_dict(transformer_state_dict, strict=True, assign=True)
            transformer_copy.save_pretrained(os.path.join(directory, "transformer"))
        if scheduler is not None:
            scheduler.save_pretrained(os.path.join(directory, "scheduler"))
