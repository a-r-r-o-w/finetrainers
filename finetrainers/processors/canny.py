from typing import Any, Dict, List, Optional, Union

import numpy as np
import PIL.Image
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

    def __init__(
        self,
        output_names: List[str] = None,
        input_names: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.output_names = output_names
        self.input_names = input_names
        self.device = device
        assert len(output_names) == 1

    def forward(self, input: Union[torch.Tensor, PIL.Image.Image, List[PIL.Image.Image]]) -> torch.Tensor:
        r"""
        Obtain the Canny edge detection of the input image.

        Args:
            input (`torch.Tensor`, `PIL.Image.Image`, or `List[PIL.Image.Image]`):
                The input tensor, image or list of images for which the Canny edge detection should be obtained.
                If a tensor, must be a 3D (CHW) or 4D (BCHW) or 5D (BTCHW) tensor. The input tensor should have
                values in the range [0, 1].

        Returns:
            torch.Tensor:
                The Canny edge detection of the input image. The output has the same shape as the input tensor. If
                the input is an image, the output is a 3D tensor. If the input is a list of images, the output is a 5D
                tensor. The output tensor has values in the range [0, 1].
        """
        if isinstance(input, PIL.Image.Image):
            input = kornia.utils.image.image_to_tensor(np.array(input)).unsqueeze(0) / 255.0
            input = input.to(self.device)
            output = kornia.filters.canny(input)[1].repeat(1, 3, 1, 1).squeeze(0)
        elif isinstance(input, list):
            input = kornia.utils.image.image_list_to_tensor([np.array(img) for img in input]) / 255.0
            output = kornia.filters.canny(input)[1].repeat(1, 3, 1, 1)
        else:
            ndim = input.ndim
            assert ndim in [3, 4, 5]

            batch_size = 1 if ndim == 3 else input.size(0)

            if ndim == 3:
                input = input.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
            elif ndim == 5:
                input = input.flatten(0, 1)  # [B, F, C, H, W] -> [B*F, C, H, W]

            output = kornia.filters.canny(input)[1].repeat(1, 3, 1, 1)
            output = output[0] if ndim == 3 else output.unflatten(0, (batch_size, -1)) if ndim == 5 else output

        # TODO(aryan): think about how one can pass parameters to the underlying function from
        # a UI perspective. It's important to think about ProcessorMixin in terms of a Graph-based
        # data processing pipeline.
        return {self.output_names[0]: output}
