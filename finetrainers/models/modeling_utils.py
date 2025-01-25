from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from diffusers import DiffusionPipeline
from PIL.Image import Image

from ..conditions import Condition, get_condition_parameters_from_dict
from ..typing import SchedulerType, TokenizerType
from ..utils import get_parameter_names, resolve_component_cls


class ModelSpecification:
    pipeline_cls = None

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
    ) -> None:
        if self.pipeline_cls is None:
            raise ValueError(f"ModelSpecification {self.__class__.__name__} must define a pipeline_cls")

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

        self.transformer_config: Dict[str, Any] = None
        self.vae_config: Dict[str, Any] = None

        self.conditions: Dict[str, Condition] = {}

        self._load_configs()

    def load_condition_models(self, *args, **kwargs) -> Dict[str, torch.nn.Module]:
        raise NotImplementedError(
            f"ModelSpecification::load_condition_models is not implemented for {self.__class__.__name__}"
        )

    def load_latent_models(self, *args, **kwargs) -> Dict[str, torch.nn.Module]:
        raise NotImplementedError(
            f"ModelSpecification::load_latent_models is not implemented for {self.__class__.__name__}"
        )

    def load_diffusion_models(self, *args, **kwargs) -> Dict[str, Union[torch.nn.Module]]:
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
        device: Optional[torch.device] = None,
        *args,
        **kwargs,
    ) -> DiffusionPipeline:
        raise NotImplementedError(
            f"ModelSpecification::load_pipeline is not implemented for {self.__class__.__name__}"
        )

    def collate_fn(self, batch: List[List[Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError(f"ModelSpecification::collate_fn is not implemented for {self.__class__.__name__}")

    def prepare_conditions(
        self,
        tokenizer: Optional[TokenizerType] = None,
        tokenizer_2: Optional[TokenizerType] = None,
        tokenizer_3: Optional[TokenizerType] = None,
        text_encoder: Optional[torch.nn.Module] = None,
        text_encoder_2: Optional[torch.nn.Module] = None,
        text_encoder_3: Optional[torch.nn.Module] = None,
        max_sequence_length: Optional[int] = None,
        *args,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError(
            f"ModelSpecification::prepare_conditions is not implemented for {self.__class__.__name__}"
        )

    def prepare_latents(
        self,
        vae: Optional[torch.nn.Module] = None,
        image_or_video: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        precompute: bool = False,
        *args,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError(
            f"ModelSpecification::prepare_latents is not implemented for {self.__class__.__name__}"
        )

    def postprocess_precomputed_conditions(
        self,
        condition_model_conditions: Dict[str, torch.Tensor],
        latent_model_conditions: Dict[str, torch.Tensor],
        *args,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError(
            f"ModelSpecification::postprocess_precomputed_latents is not implemented for {self.__class__.__name__}"
        )

    def forward(
        self, transformer: torch.nn.Module, generator: Optional[torch.Generator] = None, *args, **kwargs
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError(f"ModelSpecification::forward is not implemented for {self.__class__.__name__}")

    def loss(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError(f"ModelSpecification::loss is not implemented for {self.__class__.__name__}")

    def validation(
        self,
        pipeline: DiffusionPipeline,
        prompt: Optional[str] = None,
        image: Optional[Image] = None,
        video: Optional[List[Image]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_images_per_prompt: int = 1,
        num_videos_per_prompt: int = 1,
        generator: Optional[torch.Generator] = None,
    ) -> List[Tuple[str, Union[Image, List[Image]]]]:
        raise NotImplementedError(f"ModelSpecification::validation is not implemented for {self.__class__.__name__}")

    def _add_condition(self, name: str, condition: Condition) -> None:
        self.conditions[name] = condition

    def _remove_condition(self, name: str) -> None:
        if name in self.conditions:
            self.conditions.pop(name)

    @staticmethod
    def _prepare_condition(
        condition: Condition,
        *args,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        accepted_parameters = get_parameter_names(condition, "__call__")
        condition_parameters = get_condition_parameters_from_dict(accepted_parameters, kwargs)
        return condition(*args, **condition_parameters)

    def _load_configs(self) -> None:
        self._load_transformer_config()
        self._load_vae_config()

    def _load_transformer_config(self) -> None:
        if self.transformer_id is not None:
            self.transformer_config = resolve_component_cls(
                self.transformer_id,
                component_name="_class_name",
                filename="config.json",
                revision=self.revision,
                cache_dir=self.cache_dir,
            )
        else:
            self.transformer_config = resolve_component_cls(
                self.pretrained_model_name_or_path,
                component_name="transformer",
                filename="model_index.json",
                revision=self.revision,
                cache_dir=self.cache_dir,
            )

    def _load_vae_config(self) -> None:
        if self.vae_id is not None:
            self.vae_config = resolve_component_cls(
                self.vae_id,
                component_name="_class_name",
                filename="config.json",
                revision=self.revision,
                cache_dir=self.cache_dir,
            )
        else:
            self.vae_config = resolve_component_cls(
                self.pretrained_model_name_or_path,
                component_name="vae",
                filename="model_index.json",
                revision=self.revision,
                cache_dir=self.cache_dir,
            )
