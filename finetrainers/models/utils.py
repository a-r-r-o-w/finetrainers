from typing import Optional, Tuple

import numpy as np
import torch
from diffusers.utils.torch_utils import randn_tensor


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False, _dim: int = 1):
        # Note: _dim is the new argument added here after copying from diffusers
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=_dim)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        # make sure sample is on the same device as the parameters and has same dtype
        sample = randn_tensor(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        x = self.mean + self.std * sample
        return x

    def kl(self, other: "DiagonalGaussianDistribution" = None) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2, 3],
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(self, sample: torch.Tensor, dims: Tuple[int, ...] = [1, 2, 3]) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self) -> torch.Tensor:
        return self.mean


@torch.no_grad()
def _expand_linear_with_zeroed_weights(
    module: torch.nn.Linear, new_in_features: Optional[int] = None, new_out_features: Optional[int] = None
) -> torch.nn.Linear:
    if new_in_features is None:
        new_in_features = module.in_features
    if new_out_features is None:
        new_out_features = module.out_features
    bias = getattr(module, "bias", None)
    new_module = torch.nn.Linear(new_in_features, new_out_features, bias=bias is not None)
    new_module.to(device=module.weight.device, dtype=module.weight.dtype)
    new_module.weight.zero_()
    new_module.weight.data[: module.weight.data.shape[0], : module.weight.data.shape[1]].copy_(module.weight.data)
    if bias is not None:
        new_module.bias.zero_()
        new_module.bias.data[: bias.data.shape[0]].copy_(bias.data)
    return new_module


@torch.no_grad()
def _expand_conv3d_with_zeroed_weights(
    module: torch.nn.Linear, new_in_channels: Optional[int] = None, new_out_channels: Optional[int] = None
) -> torch.nn.Conv3d:
    if new_in_channels is None:
        new_in_channels = module.in_channels
    if new_out_channels is None:
        new_out_channels = module.out_channels
    bias = getattr(module, "bias", None)
    new_module = torch.nn.Conv3d(
        new_in_channels,
        new_out_channels,
        kernel_size=module.kernel_size,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=module.groups,
        bias=bias is not None,
    )
    new_module.to(device=module.weight.device, dtype=module.weight.dtype)
    new_module.weight.zero_()
    new_module.weight.data[: module.weight.data.shape[0], : module.weight.data.shape[1]].copy_(module.weight.data)
    if bias is not None:
        new_module.bias.zero_()
        new_module.bias.data[: bias.data.shape[0]].copy_(bias.data)
    return new_module
