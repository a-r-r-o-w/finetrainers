from typing import Optional

import torch


def normalize(x: torch.Tensor, min: float = -1.0, max: float = 1.0, dim: Optional[int] = None) -> torch.Tensor:
    """
    Normalize a tensor to the range [min_val, max_val].

    Args:
        x (`torch.Tensor`):
            The input tensor to normalize.
        min (`float`, defaults to `-1.0`):
            The minimum value of the normalized range.
        max (`float`, defaults to `1.0`):
            The maximum value of the normalized range.
        dim (`int`, *optional*):
            The dimension along which to normalize. If `None`, the entire tensor is normalized.

    Returns:
        The normalized tensor of the same shape as `x`.
    """
    if dim is None:
        x_min = x.min()
        x_max = x.max()
        if torch.isclose(x_min, x_max).any():
            x = torch.full_like(x, min)
        else:
            x = min + (max - min) * (x - x_min) / (x_max - x_min)
    else:
        x_min = x.amin(dim=dim, keepdim=True)
        x_max = x.amax(dim=dim, keepdim=True)
        if torch.isclose(x_min, x_max).any():
            x = torch.full_like(x, min)
        else:
            x = min + (max - min) * (x - x_min) / (x_max - x_min)
    return x
