import random
from typing import List, Optional, Union

import torch


def dropout_text(text: Union[str, List[str]], dropout_p: float = 0) -> Union[str, List[str]]:
    if random.random() >= dropout_p:
        return text
    if isinstance(text, str):
        return ""
    return [""] * len(text)


def dropout_embeddings_to_zero(
    embed: torch.Tensor,
    masks: Optional[List[torch.BoolTensor]] = None,
    dropout_p: float = 0,
) -> torch.Tensor:
    if random.random() >= dropout_p:
        return embed
    embed = torch.zeros_like(embed)
    if masks is not None:
        masks = [torch.zeros_like(mask, dtype=torch.bool) for mask in masks]
    return embed, masks
