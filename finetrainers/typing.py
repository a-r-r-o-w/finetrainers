from enum import Enum
from typing import Union

from diffusers import CogVideoXDDIMScheduler, FlowMatchEulerDiscreteScheduler
from transformers import CLIPTokenizer, LlamaTokenizer, LlamaTokenizerFast, T5Tokenizer, T5TokenizerFast

from .data import ImageArtifact, VideoArtifact


ArtifactType = Union[ImageArtifact, VideoArtifact]
SchedulerType = Union[CogVideoXDDIMScheduler, FlowMatchEulerDiscreteScheduler]
TokenizerType = Union[CLIPTokenizer, T5Tokenizer, T5TokenizerFast, LlamaTokenizer, LlamaTokenizerFast]


class ControlType(str, Enum):
    r"""
    Enum class for the control types.
    """

    CANNY = "canny"
    CUSTOM = "custom"


class FrameConditioningType(str, Enum):
    r"""
    Enum class for the frame conditioning types.
    """

    INDEX = "index"
    PREFIX = "prefix"
    RANDOM = "random"
    SUFFIX = "suffix"
