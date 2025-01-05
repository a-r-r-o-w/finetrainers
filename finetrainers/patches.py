import functools
from typing import Any, Union

from peft.tuners.tuners_utils import BaseTunerLayer
from peft.tuners.lora.layer import Linear as LoRALinear
import torch
from accelerate.logging import get_logger

from .constants import FINETRAINERS_LOG_LEVEL


logger = get_logger("finetrainers")  # pylint: disable=invalid-name
logger.setLevel(FINETRAINERS_LOG_LEVEL)


def perform_peft_patches() -> None:
    _perform_patch_move_adapter_to_device_of_base_layer()
    
    # We probably do not need this because we are now preventing the conversion of LoRA weights to lower precision.
    # So, LoRA weights are probably already in correct dtype of bfloat16 or float32.
    # _perform_patch_lora_linear_forward()



def _perform_patch_move_adapter_to_device_of_base_layer() -> None:
    # We don't patch the method for torch.float32 and torch.bfloat16 because it is okay to train with them. If the model weights
    # are in torch.float16, torch.float8_e4m3fn or torch.float8_e5m2, we need to patch this method to avoid conversion of
    # LoRA weights from higher precision dtype.
    BaseTunerLayer._move_adapter_to_device_of_base_layer = _patched_move_adapter_to_device_of_base_layer(BaseTunerLayer._move_adapter_to_device_of_base_layer)


def _perform_patch_lora_linear_forward() -> None:
    LoRALinear.forward = _patched_LoRALinear_forward


def _patched_move_adapter_to_device_of_base_layer(func) -> None:
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        with DisableTensorToDtype():
            print("called")
            return func(self, *args, **kwargs)

    return wrapper


def _patched_LoRALinear_forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    self._check_forward_args(x, *args, **kwargs)
    adapter_names = kwargs.pop("adapter_names", None)

    if self.disable_adapters:
        if self.merged:
            self.unmerge()
        result = self.base_layer(x, *args, **kwargs)
    elif adapter_names is not None:
        result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
    elif self.merged:
        result = self.base_layer(x, *args, **kwargs)
    else:
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            
            # This line is removed because ...
            # x = x.to(lora_A.weight.dtype)

            if not self.use_dora[active_adapter]:
                result = result + lora_B(lora_A(dropout(x))) * scaling
            else:
                if isinstance(dropout, torch.nn.Identity) or not self.training:
                    base_result = result
                else:
                    x = dropout(x)
                    base_result = None

                result = result + self.lora_magnitude_vector[active_adapter](
                    x,
                    lora_A=lora_A,
                    lora_B=lora_B,
                    scaling=scaling,
                    base_layer=self.get_base_layer(),
                    base_result=base_result,
                )

        result = result.to(torch_result_dtype)

    return result


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
