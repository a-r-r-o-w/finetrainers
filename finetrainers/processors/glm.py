from typing import List, Tuple, Union

import torch
from transformers import AutoTokenizer, GlmModel

from .base import ProcessorMixin


class CogView4GLMProcessor(ProcessorMixin):
    r"""
    Processor for the GLM family of models. This processor is used to encode text inputs and return the embeddings
    and attention masks for the input text.

    This processor is specific to CogView4 but can be used with any other model.

    Args:
        output_names (`List[str]`):
            The names of the outputs that the processor should return. The first output is the embeddings of the input
            text and the second output is the attention mask for the input text.
    """

    def __init__(self, output_names: List[str]):
        super().__init__()

        self.output_names = output_names

        assert len(self.output_names) == 1

    def forward(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: GlmModel,
        caption: Union[str, List[str]],
        max_sequence_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Encode the input text and return the embeddings and attention mask for the input text.

        Args:
            tokenizer (`AutoTokenizer`):
                The tokenizer used to tokenize the input text.
            text_encoder (`GlmModel`):
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

        text_inputs = tokenizer(
            caption,
            padding="longest",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)

        current_length = text_input_ids.size(1)
        pad_length = 16 - current_length % 16
        if pad_length > 0:
            pad_ids = text_input_ids.new_full((text_input_ids.shape[0], pad_length), fill_value=tokenizer.pad_token_id)
            text_input_ids = torch.cat([pad_ids, text_input_ids], dim=1)

        prompt_embeds = text_encoder(text_input_ids, output_hidden_states=True).hidden_states[-2]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        return {self.output_names[0]: prompt_embeds}
