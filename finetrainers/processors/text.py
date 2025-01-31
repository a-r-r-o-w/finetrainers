from typing import Any, Dict, List, Optional, Union

import torch
from transformers import T5EncoderModel, T5Tokenizer, T5TokenizerFast

from .. import functional as FF
from .utils import Processor


class CaptionEmbeddingDropoutProcessor(Processor):
    _input_names = ["embedding", "masks"]
    _output_names = ["embedding"]

    def __init__(self, dropout_p: float = 0.0) -> None:
        self.dropout_p = dropout_p

    def __call__(
        self, embedding: torch.Tensor, masks: Optional[List[torch.BoolTensor]] = None, **kwargs
    ) -> Dict[str, Any]:
        embedding = FF.dropout_embeddings_to_zero(embedding, masks, self.dropout_p)
        kwargs.update({"embedding": embedding})
        return kwargs


class CaptionTextDropoutProcessor(Processor):
    _input_names = ["caption"]
    _output_names = ["caption"]

    def __init__(self, dropout_p: float = 0.0) -> None:
        self.dropout_p = dropout_p

    def __call__(self, caption: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        caption = FF.dropout_caption(caption, self.dropout_p)
        kwargs.update({"caption": caption})
        return kwargs


class T5Processor(Processor):
    _input_names = ["caption"]
    _output_names = ["prompt_embeds", "prompt_attention_mask"]

    def __init__(
        self,
        tokenizer: Union[T5Tokenizer, T5TokenizerFast],
        text_encoder: T5EncoderModel,
    ) -> None:
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder

    def __call__(self, caption: Union[str, List[str]], max_sequence_length: int, **kwargs) -> Dict[str, Any]:
        if isinstance(caption, str):
            caption = [caption]

        device = self.text_encoder.device
        dtype = self.text_encoder.dtype

        batch_size = len(caption)
        text_inputs = self.tokenizer(
            caption,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask
        prompt_attention_mask = prompt_attention_mask.bool().to(device)

        prompt_embeds = self.text_encoder(text_input_ids.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_attention_mask = prompt_attention_mask.view(batch_size, -1)

        kwargs.update({"prompt_embeds": prompt_embeds, "prompt_attention_mask": prompt_attention_mask})
        return kwargs
