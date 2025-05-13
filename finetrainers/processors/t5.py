from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import T5EncoderModel, T5Tokenizer, T5TokenizerFast

from .base import ProcessorMixin


class T5Processor(ProcessorMixin):
    r"""
    Processor for the T5 family of models. This processor is used to encode text inputs and return the embeddings
    and attention masks for the input text.

    Args:
        output_names (`List[str]`):
            The names of the outputs that the processor should return. The first output is the embeddings of the input
            text and the second output is the attention mask for the input text.
    """

    def __init__(
        self,
        output_names: List[str],
        input_names: Optional[Dict[str, Any]] = None,
        *,
        use_attention_mask: bool = False,
    ):
        super().__init__()

        self.output_names = output_names
        self.input_names = input_names
        self.use_attention_mask = use_attention_mask

        if input_names is not None:
            assert len(input_names) <= 4
        assert len(self.output_names) == 2

    def forward(
        self,
        tokenizer: Union[T5Tokenizer, T5TokenizerFast],
        text_encoder: T5EncoderModel,
        caption: Union[str, List[str]],
        max_sequence_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Encode the input text and return the embeddings and attention mask for the input text.

        Args:
            tokenizer (`Union[T5Tokenizer, T5TokenizerFast]`):
                The tokenizer used to tokenize the input text.
            text_encoder (`T5EncoderModel`):
                The text encoder used to encode the input text.
            caption (`Union[str, List[str]]`):
                The input text to be encoded.
            max_sequence_length (`int`):
                The maximum sequence length of the input text.
        """
        if isinstance(caption, str):
            caption = [caption]

        device = text_encoder.device
        dtype = text_encoder.dtype

        batch_size = len(caption)
        text_inputs = tokenizer(
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

        te_mask = None
        if self.use_attention_mask:
            te_mask = prompt_attention_mask

        prompt_embeds = text_encoder(text_input_ids.to(device), te_mask)[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_attention_mask = prompt_attention_mask.view(batch_size, -1)

        return {
            self.output_names[0]: prompt_embeds,
            self.output_names[1]: prompt_attention_mask,
        }
