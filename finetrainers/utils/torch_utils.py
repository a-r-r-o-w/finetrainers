from typing import Dict, Optional, Union

import torch
import torch.backends
import torch.distributed.tensor
from accelerate import Accelerator
from diffusers.utils.torch_utils import is_compiled_module


def get_device_info():
    from torch._utils import _get_available_device_type, _get_device_module

    device_type = _get_available_device_type()
    if device_type is None:
        device_type = "cuda"
    device_module = _get_device_module(device_type)
    return device_type, device_module


def synchronize_device() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()


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


def expand_tensor_dims(tensor, ndim):
    while len(tensor.shape) < ndim:
        tensor = tensor.unsqueeze(-1)
    return tensor


def get_dtype_from_string(dtype: str):
    return _STRING_TO_DTYPE[dtype]


def get_string_from_dtype(dtype: torch.dtype):
    return _DTYPE_TO_STRING[dtype]


_STRING_TO_DTYPE = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

_DTYPE_TO_STRING = {v: k for k, v in _STRING_TO_DTYPE.items()}
