import random
from typing import List, Union

import torch


def convert_byte_str_to_str(s: str, encoding: str = "utf-8") -> str:
    """
    Extracts the actual string from a stringified bytes array (common in some webdatasets).

    Example: "b'hello world'" -> "hello world"
    """
    try:
        s = s[2:-1]
        s = s.encode("utf-8").decode(encoding)
    except (UnicodeDecodeError, UnicodeEncodeError, IndexError):
        pass
    return s


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
