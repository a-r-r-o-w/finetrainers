from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch
from diffusers import DiffusionPipeline
from diffusers.configuration_utils import FrozenDict
from PIL.Image import Image

from finetrainers.logging import get_logger
from finetrainers.parallel import ParallelBackendEnum
from finetrainers.processors import ProcessorMixin
from finetrainers.typing import ArtifactType, SchedulerType, TokenizerType
from finetrainers.utils import resolve_component_cls


if TYPE_CHECKING:
    from finetrainers.trainer.control_trainer.config import FrameConditioningType

logger = get_logger()

# TODO(aryan): we most likely don't need this. take a look after refactoring more
# fmt: off
IGNORE_KEYS_FOR_COLLATION = {"height", "width", "num_frames", "frame_rate", "rope_interpolation_scale", "return_dict", "attention_kwargs", "cross_attention_kwargs", "joint_attention_kwargs", "latents_mean", "latents_std"}
# fmt: on


class ModelSpecification:
    r"""
    The ModelSpecification class is an interface to be used for Diffusion training recipes. It provides
    loose structure about how to organize the code for training. The trainer implementations will
    make use of this interface to load models, prepare conditions, prepare latents, forward pass, etc.
    """

    def __init__(
        self,
        pretrained_model_name_or_path: Optional[str] = None,
        tokenizer_id: Optional[str] = None,
        tokenizer_2_id: Optional[str] = None,
        tokenizer_3_id: Optional[str] = None,
        text_encoder_id: Optional[str] = None,
        text_encoder_2_id: Optional[str] = None,
        text_encoder_3_id: Optional[str] = None,
        transformer_id: Optional[str] = None,
        vae_id: Optional[str] = None,
        text_encoder_dtype: torch.dtype = torch.bfloat16,
        text_encoder_2_dtype: torch.dtype = torch.bfloat16,
        text_encoder_3_dtype: torch.dtype = torch.bfloat16,
        transformer_dtype: torch.dtype = torch.bfloat16,
        vae_dtype: str = torch.bfloat16,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        condition_model_processors: List[ProcessorMixin] = None,
        latent_model_processors: List[ProcessorMixin] = None,
    ) -> None:
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.tokenizer_id = tokenizer_id
        self.tokenizer_2_id = tokenizer_2_id
        self.tokenizer_3_id = tokenizer_3_id
        self.text_encoder_id = text_encoder_id
        self.text_encoder_2_id = text_encoder_2_id
        self.text_encoder_3_id = text_encoder_3_id
        self.transformer_id = transformer_id
        self.vae_id = vae_id
        self.text_encoder_dtype = text_encoder_dtype
        self.text_encoder_2_dtype = text_encoder_2_dtype
        self.text_encoder_3_dtype = text_encoder_3_dtype
        self.transformer_dtype = transformer_dtype
        self.vae_dtype = vae_dtype
        self.revision = revision
        self.cache_dir = cache_dir
        self.condition_model_processors = condition_model_processors or []
        self.latent_model_processors = latent_model_processors or []

        self.transformer_config: Dict[str, Any] = None
        self.vae_config: Dict[str, Any] = None

        self._load_configs()

    def _trainer_init(self, *args, **kwargs):
        pass

    # TODO(aryan): revisit how to do this better without user having to worry about it
    @property
    def _resolution_dim_keys(self) -> Dict[str, Tuple[int, ...]]:
        raise NotImplementedError(
            f"ModelSpecification::_resolution_dim_keys is not implemented for {self.__class__.__name__}"
        )

    def load_condition_models(self) -> Dict[str, torch.nn.Module]:
        raise NotImplementedError(
            f"ModelSpecification::load_condition_models is not implemented for {self.__class__.__name__}"
        )

    def load_latent_models(self) -> Dict[str, torch.nn.Module]:
        raise NotImplementedError(
            f"ModelSpecification::load_latent_models is not implemented for {self.__class__.__name__}"
        )

    def load_diffusion_models(self) -> Dict[str, Union[torch.nn.Module]]:
        raise NotImplementedError(
            f"ModelSpecification::load_diffusion_models is not implemented for {self.__class__.__name__}"
        )

    def load_pipeline(
        self,
        tokenizer: Optional[TokenizerType] = None,
        tokenizer_2: Optional[TokenizerType] = None,
        tokenizer_3: Optional[TokenizerType] = None,
        text_encoder: Optional[torch.nn.Module] = None,
        text_encoder_2: Optional[torch.nn.Module] = None,
        text_encoder_3: Optional[torch.nn.Module] = None,
        transformer: Optional[torch.nn.Module] = None,
        vae: Optional[torch.nn.Module] = None,
        scheduler: Optional[SchedulerType] = None,
        enable_slicing: bool = False,
        enable_tiling: bool = False,
        enable_model_cpu_offload: bool = False,
        training: bool = False,
        **kwargs,
    ) -> DiffusionPipeline:
        raise NotImplementedError(
            f"ModelSpecification::load_pipeline is not implemented for {self.__class__.__name__}"
        )

    def prepare_conditions(self, processors: Optional[ProcessorMixin] = None, **kwargs) -> Dict[str, Any]:
        if processors is None:
            processors = self.condition_model_processors
        for processor in processors:
            result = processor(**kwargs)
            result_keys = set(result.keys())
            repeat_keys = result_keys.intersection(kwargs.keys())
            if repeat_keys:
                logger.warning(
                    f"Processor {processor.__class__.__name__} returned keys that already exist in "
                    f"conditions: {repeat_keys}. Overwriting the existing values, but this may not "
                    f"be intended. Please rename the keys in the processor to avoid conflicts."
                )
            kwargs.update(result)
        return kwargs

    def prepare_latents(self, processors: Optional[ProcessorMixin] = None, **kwargs) -> Dict[str, Any]:
        if processors is None:
            processors = self.latent_model_processors
        for processor in processors:
            result = processor(**kwargs)
            result_keys = set(result.keys())
            repeat_keys = result_keys.intersection(kwargs.keys())
            if repeat_keys:
                logger.warning(
                    f"Processor {processor.__class__.__name__} returned keys that already exist in "
                    f"conditions: {repeat_keys}. Overwriting the existing values, but this may not "
                    f"be intended. Please rename the keys in the processor to avoid conflicts."
                )
            kwargs.update(result)
        return kwargs

    def collate_conditions(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        keys = list(data[0].keys())
        collated_data = {}
        for key in keys:
            if key in IGNORE_KEYS_FOR_COLLATION:
                collated_data[key] = data[0][key]
                continue
            collated_d = [d[key] for d in data]
            if isinstance(collated_d[0], torch.Tensor):
                collated_d = torch.cat(collated_d)
            collated_data[key] = collated_d
        return collated_data

    def collate_latents(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        keys = list(data[0].keys())
        collated_data = {}
        for key in keys:
            if key in IGNORE_KEYS_FOR_COLLATION:
                collated_data[key] = data[0][key]
                continue
            collated_d = [d[key] for d in data]
            # TODO(aryan): Support multi-resolution collation
            if isinstance(collated_d[0], torch.Tensor):
                collated_d = torch.cat(collated_d)
            collated_data[key] = collated_d
        return collated_data

    def forward(
        self, transformer: torch.nn.Module, generator: Optional[torch.Generator] = None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError(f"ModelSpecification::forward is not implemented for {self.__class__.__name__}")

    def validation(
        self,
        pipeline: DiffusionPipeline,
        prompt: Optional[str] = None,
        image: Optional[Image] = None,
        video: Optional[List[Image]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        frame_rate: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> List[ArtifactType]:
        raise NotImplementedError(f"ModelSpecification::validation is not implemented for {self.__class__.__name__}")

    def _save_lora_weights(
        self,
        directory: str,
        transformer: torch.nn.Module,
        transformer_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        scheduler: Optional[SchedulerType] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        r"""
        Save the lora state dicts of the model to the given directory.

        This API is not backwards compatible and will be changed in near future.
        """
        raise NotImplementedError(
            f"ModelSpecification::save_lora_weights is not implemented for {self.__class__.__name__}"
        )

    def _save_model(
        self,
        directory: str,
        transformer: torch.nn.Module,
        transformer_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        scheduler: Optional[SchedulerType] = None,
    ) -> None:
        r"""
        Save the state dicts to the given directory.

        This API is not backwards compatible and will be changed in near future.
        """
        raise NotImplementedError(f"ModelSpecification::save_model is not implemented for {self.__class__.__name__}")

    def apply_tensor_parallel(
        self,
        backend: ParallelBackendEnum,
        device_mesh: torch.distributed.DeviceMesh,
        text_encoder: torch.nn.Module,
        text_encoder_2: torch.nn.Module,
        text_encoder_3: torch.nn.Module,
        transformer: torch.nn.Module,
        vae: torch.nn.Module,
    ) -> None:
        raise NotImplementedError(
            f"ModelSpecification::apply_tensor_parallel is not implemented for {self.__class__.__name__}"
        )

    def _load_configs(self) -> None:
        self._load_transformer_config()
        self._load_vae_config()

    def _load_transformer_config(self) -> None:
        if self.transformer_id is not None:
            transformer_cls = resolve_component_cls(
                self.transformer_id,
                component_name="_class_name",
                filename="config.json",
                revision=self.revision,
                cache_dir=self.cache_dir,
            )
            self.transformer_config = transformer_cls.load_config(
                self.transformer_id, revision=self.revision, cache_dir=self.cache_dir
            )
        else:
            transformer_cls = resolve_component_cls(
                self.pretrained_model_name_or_path,
                component_name="transformer",
                filename="model_index.json",
                revision=self.revision,
                cache_dir=self.cache_dir,
            )
            self.transformer_config = transformer_cls.load_config(
                self.pretrained_model_name_or_path,
                subfolder="transformer",
                revision=self.revision,
                cache_dir=self.cache_dir,
            )
        self.transformer_config = FrozenDict(**self.transformer_config)

    def _load_vae_config(self) -> None:
        if self.vae_id is not None:
            vae_cls = resolve_component_cls(
                self.vae_id,
                component_name="_class_name",
                filename="config.json",
                revision=self.revision,
                cache_dir=self.cache_dir,
            )
            self.vae_config = vae_cls.load_config(self.vae_id, revision=self.revision, cache_dir=self.cache_dir)
        else:
            vae_cls = resolve_component_cls(
                self.pretrained_model_name_or_path,
                component_name="vae",
                filename="model_index.json",
                revision=self.revision,
                cache_dir=self.cache_dir,
            )
            self.vae_config = vae_cls.load_config(
                self.pretrained_model_name_or_path, subfolder="vae", revision=self.revision, cache_dir=self.cache_dir
            )
        self.vae_config = FrozenDict(**self.vae_config)


class ControlModelSpecification(ModelSpecification):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.frame_conditioning_type: "FrameConditioningType" = None
        self.frame_conditioning_index: int = None
        self.frame_conditioning_concatenate_mask: bool = False

    def _trainer_init(
        self, frame_conditioning_type: "FrameConditioningType", frame_conditioning_index: int, concatenate_mask: bool
    ) -> None:
        self.frame_conditioning_type = frame_conditioning_type
        self.frame_conditioning_index = frame_conditioning_index
        self.frame_conditioning_concatenate_mask = concatenate_mask

    @property
    def control_injection_layer_name(self):
        r"""Must return the FQN (fully-qualified name) of the control injection layer."""
        raise NotImplementedError(
            f"ControlModelSpecification::control_injection_layer_name is not implemented for {self.__class__.__name__}"
        )

    def load_diffusion_models(self, new_in_features: int) -> Dict[str, Union[torch.nn.Module]]:
        raise NotImplementedError(
            f"ControlModelSpecification::load_diffusion_models is not implemented for {self.__class__.__name__}"
        )

    def _save_lora_weights(
        self,
        directory: str,
        transformer: torch.nn.Module,
        transformer_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        norm_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        scheduler: Optional[SchedulerType] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        r"""
        Save the lora state dicts of the model to the given directory.

        This API is not backwards compatible and will be changed in near future.
        """
        raise NotImplementedError(
            f"ControlModelSpecification::save_lora_weights is not implemented for {self.__class__.__name__}"
        )

    def _save_model(
        self,
        directory: str,
        transformer: torch.nn.Module,
        transformer_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        scheduler: Optional[SchedulerType] = None,
    ) -> None:
        r"""
        Save the state dicts to the given directory.

        This API is not backwards compatible and will be changed in near future.
        """
        raise NotImplementedError(
            f"ControlModelSpecification::save_model is not implemented for {self.__class__.__name__}"
        )

    @property
    def _original_control_layer_in_features(self):
        """
        Original in_features of the input projection layer where control is injected.
        """
        raise NotImplementedError(
            f"ControlModelSpecification::_original_control_layer_in_features is not implemented for {self.__class__.__name__}"
        )

    @property
    def _original_control_layer_out_features(self):
        """
        Original out_features of the input projection layer where control is injected.

        This will be used as the rank for control injection layer when performing low-rank training and unused otherwise.
        """
        raise NotImplementedError(
            f"ControlModelSpecification::_original_control_layer_out_features is not implemented for {self.__class__.__name__}"
        )

    @property
    def _qk_norm_identifiers(self):
        raise NotImplementedError(
            f"ControlModelSpecification::_qk_norm_identifiers is not implemented for {self.__class__.__name__}"
        )
