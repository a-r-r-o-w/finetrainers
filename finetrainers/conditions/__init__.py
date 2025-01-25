from enum import Enum
from typing import Any, Dict

from ..utils import get_parameter_names
from .condition_utils import Condition, get_condition_parameters_from_dict
from .text import CaptionEmbeddingDropoutCondition, CaptionTextDropoutCondition, T5Condition


class ConditionType(str, Enum):
    # Text conditions
    CLIP = "clip"
    T5 = "t5"

    # Dropout conditions
    CAPTION_TEXT_DROPOUT = "caption_text_dropout"
    CAPTION_EMBEDDING_DROPOUT = "caption_embedding_dropout"


SUPPORTED_CONDITIONS = {condition_type.value for condition_type in ConditionType.__members__.values()}

# fmt: off
_CONDITION_TYPE_TO_CONDITION_MAPPING = {
    # Text conditions
    ConditionType.T5: T5Condition,

    # Dropout conditions
    ConditionType.CAPTION_EMBEDDING_DROPOUT: CaptionEmbeddingDropoutCondition,
    ConditionType.CAPTION_TEXT_DROPOUT: CaptionTextDropoutCondition,
}
# fmt: on


def get_condition_cls(condition_type: ConditionType) -> Condition:
    return _CONDITION_TYPE_TO_CONDITION_MAPPING[condition_type]


def get_condition(condition_type: ConditionType, condition_parameters: Dict[str, Any]) -> Condition:
    condition_cls = get_condition_cls(condition_type)
    accepted_parameters = get_parameter_names(condition_cls.__init__)
    condition_parameters = get_condition_parameters_from_dict(accepted_parameters, condition_parameters)
    return condition_cls(**condition_parameters)
