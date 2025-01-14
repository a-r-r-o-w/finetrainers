import functools

import torch
from accelerate.logging import get_logger
from peft.tuners.tuners_utils import BaseTunerLayer

from .constants import FINETRAINERS_LOG_LEVEL


logger = get_logger("finetrainers")  # pylint: disable=invalid-name
logger.setLevel(FINETRAINERS_LOG_LEVEL)


def perform_peft_patches() -> None:
    _perform_patch_move_adapter_to_device_of_base_layer()


def _perform_patch_move_adapter_to_device_of_base_layer() -> None:
    # We don't patch the method for torch.float32 and torch.bfloat16 because it is okay to train with them. If the model weights
    # are in torch.float16, torch.float8_e4m3fn or torch.float8_e5m2, we need to patch this method to avoid conversion of
    # LoRA weights from higher precision dtype.
    BaseTunerLayer._move_adapter_to_device_of_base_layer = _patched_move_adapter_to_device_of_base_layer(
        BaseTunerLayer._move_adapter_to_device_of_base_layer
    )


def _patched_move_adapter_to_device_of_base_layer(func) -> None:
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        with DisableTensorToDtype():
            return func(self, *args, **kwargs)

    return wrapper


class DisableTensorToDtype:
    def __enter__(self):
        self.original_to = torch.Tensor.to

        def modified_to(tensor, *args, **kwargs):
            # remove dtype from args if present
            args = [arg if not isinstance(arg, torch.dtype) else None for arg in args]
            if "dtype" in kwargs:
                kwargs.pop("dtype")
            return self.original_to(tensor, *args, **kwargs)

        torch.Tensor.to = modified_to

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.Tensor.to = self.original_to
