from typing import Any, Dict, List, Optional

import torch

from ..utils.import_utils import is_kornia_available
from .base import ProcessorMixin


if is_kornia_available():
    import kornia


class CannyProcessor(ProcessorMixin):
    r"""
    Processor for obtaining the Canny edge detection of an image.

    Args:
        output_names (`List[str]`):
            The names of the outputs that the processor should return. The first output is the Canny edge detection of
            the input image.
    """

    def __init__(self, output_names: List[str] = None, input_names: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()

        self.output_names = output_names
        self.input_names = input_names
        assert len(output_names) == 1

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""
        Obtain the Canny edge detection of the input image.

        Args:
            input (`torch.Tensor`):
                The input tensor for which the Canny edge detection should be obtained. Must be a 3D tensor (CHW)
                or 4D tensor (BCHW) or 5D tensor (BTCHW).
        """
        ndim = input.ndim
        assert ndim in [3, 4, 5]

        if ndim == 3:
            input = input.unsqueeze(0)
            batch_size = 1
        elif ndim == 5:
            input = input.flatten(0, 1)
            batch_size = input.size(0)
        else:
            batch_size = input.size(0)

        output = kornia.filters.canny(input)[1].repeat(1, 3, 1, 1)
        if ndim == 3:
            output = output[0]
        elif ndim == 5:
            output = output.unflatten(0, (batch_size, -1))

        # TODO(aryan): think about how one can pass parameters to the underlying function from
        # a UI perspective. It's important to think about ProcessorMixin in terms of a Graph-based
        # data processing pipeline.
        return {self.output_names[0]: output}
