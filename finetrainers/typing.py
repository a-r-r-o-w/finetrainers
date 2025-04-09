from typing import Union

from diffusers import CogVideoXDDIMScheduler, FlowMatchEulerDiscreteScheduler
from transformers import CLIPTokenizer, LlamaTokenizer, LlamaTokenizerFast, T5Tokenizer, T5TokenizerFast

from .data import ImageArtifact, VideoArtifact


ArtifactType = Union[ImageArtifact, VideoArtifact]
SchedulerType = Union[CogVideoXDDIMScheduler, FlowMatchEulerDiscreteScheduler]
TokenizerType = Union[CLIPTokenizer, T5Tokenizer, T5TokenizerFast, LlamaTokenizer, LlamaTokenizerFast]
