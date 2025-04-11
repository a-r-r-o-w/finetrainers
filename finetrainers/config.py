from enum import Enum
from typing import Type

from .models import ModelSpecification
from .models.cogvideox import CogVideoXModelSpecification
from .models.cogview4 import CogView4ControlModelSpecification, CogView4ModelSpecification
from .models.flux import FluxModelSpecification
from .models.hunyuan_video import HunyuanVideoModelSpecification
from .models.ltx_video import LTXVideoModelSpecification
from .models.wan import WanControlModelSpecification, WanModelSpecification


class ModelType(str, Enum):
    COGVIDEOX = "cogvideox"
    COGVIEW4 = "cogview4"
    FLUX = "flux"
    HUNYUAN_VIDEO = "hunyuan_video"
    LTX_VIDEO = "ltx_video"
    WAN = "wan"


class TrainingType(str, Enum):
    # SFT
    LORA = "lora"
    FULL_FINETUNE = "full-finetune"

    # Control
    CONTROL_LORA = "control-lora"
    CONTROL_FULL_FINETUNE = "control-full-finetune"


SUPPORTED_MODEL_CONFIGS = {
    # TODO(aryan): autogenerate this
    # SFT
    ModelType.COGVIDEOX: {
        TrainingType.LORA: CogVideoXModelSpecification,
        TrainingType.FULL_FINETUNE: CogVideoXModelSpecification,
    },
    ModelType.COGVIEW4: {
        TrainingType.LORA: CogView4ModelSpecification,
        TrainingType.FULL_FINETUNE: CogView4ModelSpecification,
        TrainingType.CONTROL_LORA: CogView4ControlModelSpecification,
        TrainingType.CONTROL_FULL_FINETUNE: CogView4ControlModelSpecification,
    },
    ModelType.FLUX: {
        TrainingType.LORA: FluxModelSpecification,
        TrainingType.FULL_FINETUNE: FluxModelSpecification,
    },
    ModelType.HUNYUAN_VIDEO: {
        TrainingType.LORA: HunyuanVideoModelSpecification,
        TrainingType.FULL_FINETUNE: HunyuanVideoModelSpecification,
    },
    ModelType.LTX_VIDEO: {
        TrainingType.LORA: LTXVideoModelSpecification,
        TrainingType.FULL_FINETUNE: LTXVideoModelSpecification,
    },
    ModelType.WAN: {
        TrainingType.LORA: WanModelSpecification,
        TrainingType.FULL_FINETUNE: WanModelSpecification,
        TrainingType.CONTROL_LORA: WanControlModelSpecification,
        TrainingType.CONTROL_FULL_FINETUNE: WanControlModelSpecification,
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
