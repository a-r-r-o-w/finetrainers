import torch


def normalize(x: torch.Tensor, min: float = -1.0, max: float = 1.0) -> torch.Tensor:
    """Normalize a tensor to the range [min_val, max_val]."""
    x_min = x.min()
    x_max = x.max()
    if torch.isclose(x_min, x_max).any():
        x = torch.full_like(x, min)
    else:
        x = min + (max - min) * (x - x_min) / (x_max - x_min)
    return x
