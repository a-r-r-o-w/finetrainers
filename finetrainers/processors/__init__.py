from enum import Enum
from typing import Any, Dict

from ..utils import get_parameter_names
from .text import CaptionEmbeddingDropoutProcessor, CaptionTextDropoutProcessor, T5Processor
from .utils import Processor, get_processor_parameters_from_dict


class ProcessorType(str, Enum):
    # Text conditions
    CLIP = "clip"
    T5 = "t5"

    # Dropout conditions
    CAPTION_TEXT_DROPOUT = "caption_text_dropout"
    CAPTION_EMBEDDING_DROPOUT = "caption_embedding_dropout"


SUPPORTED_CONDITIONS = {condition_type.value for condition_type in ProcessorType.__members__.values()}

# fmt: off
_PROCESSOR_TYPE_TO_PROCESSOR_MAPPING = {
    # Text conditions
    ProcessorType.T5: T5Processor,

    # Dropout conditions
    ProcessorType.CAPTION_EMBEDDING_DROPOUT: CaptionEmbeddingDropoutProcessor,
    ProcessorType.CAPTION_TEXT_DROPOUT: CaptionTextDropoutProcessor,
}
# fmt: on


def get_condition_cls(condition_type: ProcessorType) -> Processor:
    return _PROCESSOR_TYPE_TO_PROCESSOR_MAPPING[condition_type]


def get_condition(condition_type: ProcessorType, condition_parameters: Dict[str, Any]) -> Processor:
    condition_cls = get_condition_cls(condition_type)
    accepted_parameters = get_parameter_names(condition_cls.__init__)
    condition_parameters = get_processor_parameters_from_dict(accepted_parameters, condition_parameters)
    return condition_cls(**condition_parameters)
