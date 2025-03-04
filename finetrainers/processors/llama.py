from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import LlamaModel, LlamaTokenizer, LlamaTokenizerFast

from .base import ProcessorMixin


DEFAULT_PROMPT_TEMPLATE = {
    "template": (
        "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: "
        "1. The main content and theme of the video."
        "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
        "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
        "4. background environment, light, style and atmosphere."
        "5. camera angles, movements, and transitions used in the video:<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
    ),
    "crop_start": 95,
}


class LlamaProcessor(ProcessorMixin):
    r"""
    Processor for the Llama family of models. This processor is used to encode text inputs and return the embeddings
    and attention masks for the input text.

    Args:
        output_names (`List[str]`):
            The names of the outputs that the processor should return. The first output is the embeddings of the input
            text and the second output is the attention mask for the input text.
    """

    def __init__(self, output_names: List[str] = None):
        super().__init__()

        self.output_names = output_names

        assert len(output_names) == 2

    def forward(
        self,
        tokenizer: Union[LlamaTokenizer, LlamaTokenizerFast],
        text_encoder: LlamaModel,
        caption: Union[str, List[str]],
        max_sequence_length: int,
        prompt_template: Optional[Dict[str, Any]] = None,
        num_layers_to_skip: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Encode the input text and return the embeddings and attention mask for the input text.

        Args:
            tokenizer (`Union[LlamaTokenizer, LlamaTokenizerFast]`):
                The tokenizer used to tokenize the input text.
            text_encoder (`LlamaModel`):
                The text encoder used to encode the input text.
            caption (`Union[str, List[str]]`):
                The input text to be encoded.
            max_sequence_length (`int`):
                The maximum sequence length of the input text.
            prompt_template (`Optional[Dict[str, Any]]`):
                The prompt template to be used to encode the input text.
        """
        if prompt_template is None:
            prompt_template = DEFAULT_PROMPT_TEMPLATE
        if isinstance(caption, str):
            caption = [caption]

        device = text_encoder.device
        dtype = text_encoder.dtype

        batch_size = len(caption)
        caption = [prompt_template["template"].format(c) for c in caption]

        crop_start = prompt_template.get("crop_start", None)
        if crop_start is None:
            prompt_template_input = tokenizer(
                prompt_template["template"],
                padding="max_length",
                return_tensors="pt",
                return_length=False,
                return_overflowing_tokens=False,
                return_attention_mask=False,
            )
            crop_start = prompt_template_input["input_ids"].shape[-1]
            # Remove <|eot_id|> token and placeholder {}
            crop_start -= 2

        max_sequence_length += crop_start
        text_inputs = tokenizer(
            caption,
            max_length=max_sequence_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_length=False,
            return_overflowing_tokens=False,
            return_attention_mask=True,
        )
        text_input_ids = text_inputs.input_ids.to(device)
        prompt_attention_mask = text_inputs.attention_mask.bool().to(device)

        prompt_embeds = text_encoder(
            text_input_ids, attention_mask=prompt_attention_mask, output_hidden_states=True
        ).hidden_states[-(num_layers_to_skip + 1)]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        if crop_start is not None and crop_start > 0:
            prompt_embeds = prompt_embeds[:, crop_start:]
            prompt_attention_mask = prompt_attention_mask[:, crop_start:]

        prompt_attention_mask = prompt_attention_mask.view(batch_size, -1)

        return {
            self.output_names[0]: prompt_embeds,
            self.output_names[1]: prompt_attention_mask,
        }
