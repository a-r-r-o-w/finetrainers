import functools
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
from accelerate import init_empty_weights
from diffusers import (
    AutoencoderKLLTXVideo,
    FlowMatchEulerDiscreteScheduler,
    LTXImageToVideoPipeline,
    LTXPipeline,
    LTXVideoTransformer3DModel,
)
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from PIL.Image import Image
from transformers import AutoModel, AutoTokenizer, T5EncoderModel, T5Tokenizer

import finetrainers.functional as FF
from finetrainers.data import VideoArtifact
from finetrainers.logging import get_logger
from finetrainers.models.modeling_utils import ModelSpecification
from finetrainers.parallel import ParallelBackendEnum
from finetrainers.processors import ProcessorMixin, T5Processor
from finetrainers.typing import ArtifactType, SchedulerType
from finetrainers.utils import _enable_vae_memory_optimizations, get_non_null_items, safetensors_torch_save_function


logger = get_logger()


class LTXLatentEncodeProcessor(ProcessorMixin):
    r"""
    Processor to encode image/video into latents using the LTX VAE.

    Args:
        output_names (`List[str]`):
            The names of the outputs that the processor returns. The outputs are in the following order:
            - latents: The latents of the input image/video.
            - num_frames: The number of frames in the input video.
            - height: The height of the input image/video.
            - width: The width of the input image/video.
            - latents_mean: The latent channel means from the VAE state dict.
            - latents_std: The latent channel standard deviations from the VAE state dict.
    """

    def __init__(self, output_names: List[str]):
        super().__init__()
        self.output_names = output_names
        assert len(self.output_names) == 6

    def forward(
        self,
        vae: AutoencoderKLLTXVideo,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        compute_posterior: bool = True,
    ) -> Dict[str, torch.Tensor]:
        device = vae.device
        dtype = vae.dtype

        if image is not None:
            video = image.unsqueeze(1)

        assert video.ndim == 5, f"Expected 5D tensor, got {video.ndim}D tensor"
        video = video.to(device=device, dtype=vae.dtype)
        video = video.permute(0, 2, 1, 3, 4).contiguous()  # [B, F, C, H, W] -> [B, C, F, H, W]

        if compute_posterior:
            latents = vae.encode(video).latent_dist.sample(generator=generator)
            latents = latents.to(dtype=dtype)
        else:
            if vae.use_slicing and video.shape[0] > 1:
                encoded_slices = [vae._encode(x_slice) for x_slice in video.split(1)]
                moments = torch.cat(encoded_slices)
            else:
                moments = vae._encode(video)
            latents = moments.to(dtype=dtype)

        _, _, num_frames, height, width = latents.shape

        return {
            self.output_names[0]: latents,
            self.output_names[1]: num_frames,
            self.output_names[2]: height,
            self.output_names[3]: width,
            self.output_names[4]: vae.latents_mean,
            self.output_names[5]: vae.latents_std,
        }


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
        condition_model_processors: List[ProcessorMixin] = None,
        latent_model_processors: List[ProcessorMixin] = None,
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
            condition_model_processors = [T5Processor(["encoder_hidden_states", "encoder_attention_mask"])]
        if latent_model_processors is None:
            latent_model_processors = [
                LTXLatentEncodeProcessor(["latents", "num_frames", "height", "width", "latents_mean", "latents_std"])
            ]

        self.condition_model_processors = condition_model_processors
        self.latent_model_processors = latent_model_processors

    @property
    def _resolution_dim_keys(self):
        return {"latents": (2, 3, 4)}

    def load_condition_models(self) -> Dict[str, torch.nn.Module]:
        common_kwargs = {"revision": self.revision, "cache_dir": self.cache_dir}

        if self.tokenizer_id is not None:
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id, **common_kwargs)
        else:
            tokenizer = T5Tokenizer.from_pretrained(
                self.pretrained_model_name_or_path, subfolder="tokenizer", **common_kwargs
            )

        if self.text_encoder_id is not None:
            text_encoder = AutoModel.from_pretrained(
                self.text_encoder_id, torch_dtype=self.text_encoder_dtype, **common_kwargs
            )
        else:
            text_encoder = T5EncoderModel.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="text_encoder",
                torch_dtype=self.text_encoder_dtype,
                **common_kwargs,
            )

        return {"tokenizer": tokenizer, "text_encoder": text_encoder}

    def load_latent_models(self) -> Dict[str, torch.nn.Module]:
        common_kwargs = {"revision": self.revision, "cache_dir": self.cache_dir}

        if self.vae_id is not None:
            vae = AutoencoderKLLTXVideo.from_pretrained(self.vae_id, torch_dtype=self.vae_dtype, **common_kwargs)
        else:
            vae = AutoencoderKLLTXVideo.from_pretrained(
                self.pretrained_model_name_or_path, subfolder="vae", torch_dtype=self.vae_dtype, **common_kwargs
            )

        return {"vae": vae}

    def load_diffusion_models(self) -> Dict[str, torch.nn.Module]:
        common_kwargs = {"revision": self.revision, "cache_dir": self.cache_dir}

        if self.transformer_id is not None:
            transformer = LTXVideoTransformer3DModel.from_pretrained(
                self.transformer_id, torch_dtype=self.transformer_dtype, **common_kwargs
            )
        else:
            transformer = LTXVideoTransformer3DModel.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="transformer",
                torch_dtype=self.transformer_dtype,
                **common_kwargs,
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

        _enable_vae_memory_optimizations(pipe.vae, enable_slicing, enable_tiling)
        if not training:
            pipe.transformer.to(self.transformer_dtype)
        if enable_model_cpu_offload:
            pipe.enable_model_cpu_offload()
        return pipe

    @torch.no_grad()
    def prepare_conditions(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        caption: str,
        max_sequence_length: int = 128,
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
        vae: AutoencoderKLLTXVideo,
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
        transformer: LTXVideoTransformer3DModel,
        condition_model_conditions: Dict[str, torch.Tensor],
        latent_model_conditions: Dict[str, torch.Tensor],
        sigmas: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        compute_posterior: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        # TODO(aryan): make this configurable? Should it be?
        first_frame_conditioning_p = 0.1
        min_first_frame_sigma = 0.25

        if compute_posterior:
            latents = latent_model_conditions.pop("latents")
        else:
            posterior = DiagonalGaussianDistribution(latent_model_conditions.pop("latents"))
            latents = posterior.sample(generator=generator)
            del posterior

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
        timesteps = (sigmas * 1000.0).long()

        latent_model_conditions["hidden_states"] = noisy_latents.to(latents)

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
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> List[ArtifactType]:
        if image is not None:
            pipeline = LTXImageToVideoPipeline.from_pipe(pipeline)

        generation_kwargs = {
            "prompt": prompt,
            "image": image,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "frame_rate": frame_rate,
            "num_inference_steps": num_inference_steps,
            "generator": generator,
            "return_dict": True,
            "output_type": "pil",
        }
        generation_kwargs = get_non_null_items(generation_kwargs)
        video = pipeline(**generation_kwargs).frames[0]
        return [VideoArtifact(value=video)]

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
            LTXPipeline.save_lora_weights(
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
        transformer: LTXVideoTransformer3DModel,
        transformer_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        scheduler: Optional[SchedulerType] = None,
    ) -> None:
        # TODO(aryan): this needs refactoring
        if transformer_state_dict is not None:
            with init_empty_weights():
                transformer_copy = LTXVideoTransformer3DModel.from_config(transformer.config)
            transformer_copy.load_state_dict(transformer_state_dict, strict=True, assign=True)
            transformer_copy.save_pretrained(os.path.join(directory, "transformer"))
        if scheduler is not None:
            scheduler.save_pretrained(os.path.join(directory, "scheduler"))

    def apply_tensor_parallel(
        self,
        backend: ParallelBackendEnum,
        device_mesh: torch.distributed.DeviceMesh,
        transformer: LTXVideoTransformer3DModel,
        **kwargs,
    ) -> None:
        if backend == ParallelBackendEnum.PTD:
            _apply_tensor_parallel_ptd(device_mesh, transformer)
        else:
            raise NotImplementedError(f"Parallel backend {backend} is not supported for LTXVideoModelSpecification")

    @staticmethod
    def _normalize_latents(
        latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0
    ) -> torch.Tensor:
        # Normalize latents across the channel dimension [B, C, F, H, W]
        batch_size = latents.shape[0]
        latents_mean = latents_mean.view(batch_size, -1, 1, 1, 1).to(device=latents.device)
        latents_std = latents_std.view(batch_size, -1, 1, 1, 1).to(device=latents.device)
        latents = ((latents.float() - latents_mean) * scaling_factor / latents_std).to(latents)
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


def _apply_tensor_parallel_ptd(
    device_mesh: torch.distributed.device_mesh.DeviceMesh, transformer: LTXVideoTransformer3DModel
) -> None:
    from torch.distributed.tensor.parallel import parallelize_module
    from torch.distributed.tensor.parallel.style import ColwiseParallel, RowwiseParallel

    transformer_plan = {
        # ===== Condition embeddings =====
        # "time_embed.emb.timestep_embedder.linear_1": ColwiseParallel(),
        # "time_embed.emb.timestep_embedder.linear_2": RowwiseParallel(output_layouts=Shard(-1)),
        # "time_embed.linear": ColwiseParallel(input_layouts=Shard(-1), output_layouts=Replicate()),
        # "time_embed": PrepareModuleOutput(output_layouts=(Replicate(), Shard(-1)), desired_output_layouts=(Replicate(), Replicate())),
        # "caption_projection.linear_1": ColwiseParallel(),
        # "caption_projection.linear_2": RowwiseParallel(),
        # "rope": PrepareModuleOutput(output_layouts=(Replicate(), Replicate()), desired_output_layouts=(Shard(1), Shard(1)), use_local_output=False),
        # ===== =====
    }

    for block in transformer.transformer_blocks:
        block_plan = {}

        # ===== Attention =====
        # 8 all-to-all, 3 all-reduce
        # block_plan["attn1.to_q"] = ColwiseParallel(use_local_output=False)
        # block_plan["attn1.to_k"] = ColwiseParallel(use_local_output=False)
        # block_plan["attn1.to_v"] = ColwiseParallel(use_local_output=False)
        # block_plan["attn1.norm_q"] = SequenceParallel()
        # block_plan["attn1.norm_k"] = SequenceParallel()
        # block_plan["attn1.to_out.0"] = RowwiseParallel(input_layouts=Shard(1))
        # block_plan["attn2.to_q"] = ColwiseParallel(use_local_output=False)
        # block_plan["attn2.to_k"] = ColwiseParallel(use_local_output=False)
        # block_plan["attn2.to_v"] = ColwiseParallel(use_local_output=False)
        # block_plan["attn2.norm_q"] = SequenceParallel()
        # block_plan["attn2.norm_k"] = SequenceParallel()
        # block_plan["attn2.to_out.0"] = RowwiseParallel(input_layouts=Shard(1))
        # ===== =====

        block_plan["ff.net.0.proj"] = ColwiseParallel()
        block_plan["ff.net.2"] = RowwiseParallel()

        parallelize_module(block, device_mesh, block_plan)

    parallelize_module(transformer, device_mesh, transformer_plan)
