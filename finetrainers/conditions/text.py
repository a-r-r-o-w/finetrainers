from typing import Dict, List, Optional, Union

import torch
from transformers import T5EncoderModel, T5Tokenizer, T5TokenizerFast

from .. import functional as FF
from .condition_utils import Condition


class CaptionEmbeddingDropoutCondition(Condition):
    def __init__(self, dropout_p: float = 0.0) -> None:
        self.dropout_p = dropout_p

    def __call__(self, embedding: torch.Tensor, masks: Optional[List[torch.BoolTensor]] = None) -> torch.Tensor:
        return FF.dropout_embeddings_to_zero(embedding, masks, self.dropout_p)


class CaptionTextDropoutCondition(Condition):
    def __init__(self, dropout_p: float = 0.0) -> None:
        self.dropout_p = dropout_p

    def __call__(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        return FF.dropout_text(text, self.dropout_p)


class T5Condition(Condition):
    def __init__(
        self,
        tokenizer: Union[T5Tokenizer, T5TokenizerFast],
        text_encoder: T5EncoderModel,
    ) -> None:
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder

    def __call__(self, text: Union[str, List[str]], max_sequence_length: int) -> Dict[str, torch.Tensor]:
        if isinstance(text, str):
            text = [text]

        device = self.text_encoder.device
        dtype = self.text_encoder.dtype

        batch_size = len(text)
        text_inputs = self.tokenizer(
            text,
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

        return {"prompt_embeds": prompt_embeds, "prompt_attention_mask": prompt_attention_mask}
