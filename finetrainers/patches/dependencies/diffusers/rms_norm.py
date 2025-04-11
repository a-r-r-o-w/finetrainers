import torch
import torch.nn as nn
from diffusers.utils import is_torch_npu_available, is_torch_version


def _patched_rms_norm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    if is_torch_npu_available():
        import torch_npu

        if self.weight is not None:
            # convert into half-precision if necessary
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                hidden_states = hidden_states.to(self.weight.dtype)
        hidden_states = torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.eps)[0]
        if self.bias is not None:
            hidden_states = hidden_states + self.bias
    elif is_torch_version(">=", "2.4"):
        ### ===== <Modified> =======
        input_dtype = hidden_states.dtype
        if self.weight is not None:
            # convert into half-precision if necessary
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                hidden_states = hidden_states.to(self.weight.dtype)
        hidden_states = nn.functional.rms_norm(
            hidden_states, normalized_shape=(hidden_states.shape[-1],), weight=self.weight, eps=self.eps
        )
        if self.bias is not None:
            hidden_states = hidden_states + self.bias
        hidden_states = hidden_states.to(input_dtype)
        ### ===== </Modified> =====
    else:
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        if self.weight is not None:
            # convert into half-precision if necessary
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                hidden_states = hidden_states.to(self.weight.dtype)
            hidden_states = hidden_states * self.weight
            if self.bias is not None:
                hidden_states = hidden_states + self.bias
        else:
            hidden_states = hidden_states.to(input_dtype)

    return hidden_states
