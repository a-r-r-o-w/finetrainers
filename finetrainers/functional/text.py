import random
from typing import List, Union

import torch


def dropout_caption(caption: Union[str, List[str]], dropout_p: float = 0) -> Union[str, List[str]]:
    if random.random() >= dropout_p:
        return caption
    if isinstance(caption, str):
        return ""
    return [""] * len(caption)


def dropout_embeddings_to_zero(embed: torch.Tensor, dropout_p: float = 0) -> torch.Tensor:
    if random.random() >= dropout_p:
        return embed
    embed = torch.zeros_like(embed)
    return embed


def remove_prefix(text: str, prefixes: List[str]) -> str:
    for prefix in prefixes:
        if text.startswith(prefix):
            return text.removeprefix(prefix).strip()
    return text
