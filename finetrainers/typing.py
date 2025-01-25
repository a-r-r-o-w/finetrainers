from typing import Union

from diffusers import CogVideoXDDIMScheduler, FlowMatchEulerDiscreteScheduler
from transformers import CLIPTokenizer, LlamaTokenizer, LlamaTokenizerFast, T5Tokenizer, T5TokenizerFast


SchedulerType = Union[CogVideoXDDIMScheduler, FlowMatchEulerDiscreteScheduler]
TokenizerType = Union[CLIPTokenizer, T5Tokenizer, T5TokenizerFast, LlamaTokenizer, LlamaTokenizerFast]
