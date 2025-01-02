from typing import Dict, Optional, Union

import torch
from accelerate import Accelerator
from diffusers.utils.torch_utils import is_compiled_module


def unwrap_model(accelerator: Accelerator, model):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def align_device_and_dtype(
    x: Union[torch.Tensor, Dict[str, torch.Tensor]],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    if isinstance(x, torch.Tensor):
        if device is not None:
            x = x.to(device)
        if dtype is not None:
            x = x.to(dtype)
    elif isinstance(x, dict):
        if device is not None:
            x = {k: align_device_and_dtype(v, device, dtype) for k, v in x.items()}
        if dtype is not None:
            x = {k: align_device_and_dtype(v, device, dtype) for k, v in x.items()}
    return x


def expand_tensor_to_dims(tensor, ndim):
    while len(tensor.shape) < ndim:
        tensor = tensor.unsqueeze(-1)
    return tensor
