from enum import Enum
from typing import Type

from .models import ModelSpecification
from .models.cogvideox import CogVideoXModelSpecification
from .models.hunyuan_video import HUNYUAN_VIDEO_T2V_FULL_FINETUNE_CONFIG, HUNYUAN_VIDEO_T2V_LORA_CONFIG
from .models.ltx_video import LTXVideoModelSpecification


class ModelType(str, Enum):
    HUNYUAN_VIDEO = "hunyuan_video"
    LTX_VIDEO = "ltx_video"
    COGVIDEOX = "cogvideox"


class TrainingType(str, Enum):
    LORA = "lora"
    FULL_FINETUNE = "full-finetune"


SUPPORTED_MODEL_CONFIGS = {
    ModelType.HUNYUAN_VIDEO: {
        TrainingType.LORA: HUNYUAN_VIDEO_T2V_LORA_CONFIG,
        TrainingType.FULL_FINETUNE: HUNYUAN_VIDEO_T2V_FULL_FINETUNE_CONFIG,
    },
    ModelType.LTX_VIDEO: {
        TrainingType.LORA: LTXVideoModelSpecification,
        TrainingType.FULL_FINETUNE: LTXVideoModelSpecification,
    },
    ModelType.COGVIDEOX: {
        TrainingType.LORA: CogVideoXModelSpecification,
        TrainingType.FULL_FINETUNE: CogVideoXModelSpecification,
    },
}


def _get_model_specifiction_cls(model_name: str, training_type: str) -> Type[ModelSpecification]:
    if model_name not in SUPPORTED_MODEL_CONFIGS:
        raise ValueError(
            f"Model {model_name} not supported. Supported models are: {list(SUPPORTED_MODEL_CONFIGS.keys())}"
        )
    if training_type not in SUPPORTED_MODEL_CONFIGS[model_name]:
        raise ValueError(
            f"Training type {training_type} not supported for model {model_name}. Supported training types are: {list(SUPPORTED_MODEL_CONFIGS[model_name].keys())}"
        )
    return SUPPORTED_MODEL_CONFIGS[model_name][training_type]
