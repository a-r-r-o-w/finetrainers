from typing import List, Union

import torch

import finetrainers.functional as FF

from .base import ProcessorMixin


class CaptionTextDropoutProcessor(ProcessorMixin):
    def __init__(self, dropout_p: float = 0.0) -> None:
        self.dropout_p = dropout_p

    def forward(self, caption: Union[str, List[str]]) -> Union[str, List[str]]:
        return FF.dropout_caption(caption, self.dropout_p)


class CaptionEmbeddingDropoutProcessor(ProcessorMixin):
    def __init__(self, dropout_p: float = 0.0) -> None:
        self.dropout_p = dropout_p

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return FF.dropout_embeddings_to_zero(embedding, self.dropout_p)
