from typing import Any, Dict, List, Optional

from .base import ProcessorMixin
from .canny import CannyProcessor
from .clip import CLIPPooledProcessor
from .glm import CogView4GLMProcessor
from .llama import LlamaProcessor
from .t5 import T5Processor
from .text import CaptionEmbeddingDropoutProcessor, CaptionTextDropoutProcessor


class CopyProcessor(ProcessorMixin):
    r"""Processor that copies the input data unconditionally to the output."""

    def __init__(self, output_names: List[str] = None, input_names: Optional[Dict[str, Any]] = None):
        super().__init__()

        self.output_names = output_names
        self.input_names = input_names
        assert len(output_names) == 1

    def forward(self, input: Any) -> Any:
        return {self.output_names[0]: input}
