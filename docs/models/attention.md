# Attention backends

Finetrainers supports multiple attention backends to support different hardware and tradeoff between speed and memory usage. The following attention implementations are supported:
- Training:
  - If model uses attention masks: `flash_varlen`, `flex`, `native`
  - If model does not use attention masks: `flash`, `flex`, `native`, `xformers`
- Inference:
  - If model uses attention masks: `flash_varlen`, `flex`, `native`, `sage_varlen`
  - If model does not use attention masks: `flash`, `flash_varlen`, `flex`, `native`, `sage`, `sage_varlen`, `xformers`

Additionally, some specialized methods are available for debugging-specific purposes: `_native_cudnn`, `_native_efficient`, `_native_flash`, `_native_math`, `_sage_qk_int8_pv_fp8_cuda`, `_sage_qk_int8_pv_fp8_cuda_sm90`, `_sage_qk_int8_pv_fp16_cuda`, `_sage_qk_int8_pv_fp16_triton`. With time, more attention-specific optimizations and custom implementations will be supported. Contributions are welcome!

Unfortunately, due to limited time for testing, only specific versions of packages that provide these implementations are supported. Other versions may work. The supported versions will be gradually made lower for more flexibility, but for now, please use the following versions:
- `flash-attn>=2.6.3`
- `sageattention>=2.1.1`
- `xformers>=0.0.29.post3`

This guide will help you quickly install flash-attn, sageattention, and xformers to make your models run faster and use less memory for training/inference. We'll cover installation on Linux (Ubuntu 22.04) and Windows (using WSL).

Before you start, make sure to use a clean python virtual environment to not mess up your system seriously, or to avoid conflicting dependencies leading to failed installations which might leave the environment in hard-to-recover state.

### Flash attention

Providers covered: `flash`, `flash_varlen`

The installation steps have only been tested with Ubuntu 22.04; CUDA version higher than 12.2 and 12.6.
- Check your CUDA version: look at the output of `nvidia-smi` or run `nvcc --version`.
- You might need the following packages: `pip install packaging ninja`
- Linux: Run: `pip install flash-attn --no-build-isolation`. Verify the version with `pip show flash-attn`
- WSL: Same instruction as above should work. Native Windows might require building from source - check community guiders and follow the instruction [here](https://github.com/Dao-AILab/flash-attention).

### Sage attention

Providers covered: `sage`, `sage_varlen`, `_sage_qk_int8_pv_fp8_cuda`, `_sage_qk_int8_pv_fp8_cuda_sm90`, `_sage_qk_int8_pv_fp16_cuda`, `_sage_qk_int8_pv_fp16_triton`

FP8 implementations will require CUDA compute capability of 90 or higher (H100, RTX 5090, etc.). Some may work on compute capability 89 as well (RTX 4090, for example). For FP16 implementations, compute capability of atleast 80 is required (A100, RTX 3090, etc.). For other GPUs, FP16 implementations may or may not work (this is untested by me).

- Check your compute capability with the following command:
  ```bash
  python -c "import torch; print(torch.cuda.get_device_capability())"
  ```
- Check your CUDA version: look at the output of `nvidia-smi` or run `nvcc --version`.
- You might need the following packages: `pip install triton`. For Windows, check out the [triton-windows](https://github.com/woct0rdho/triton-windows) project.
- Linux/WSL: Run: `pip install git+https://github.com/thu-ml/SageAttention`. Verify the version with `pip show sageattention`.
- Make sure to look at the official installation guide in [SageAttention](https://github.com/thu-ml/SageAttention) too!

### xformers

Providers covered: `xformers`

- Check your CUDA version: look at the output of `nvidia-smi` or run `nvcc --version`.
- Linux/WSL: Run: `pip install -U xformers --index-url https://download.pytorch.org/whl/cu126` (assuming CUDA 12.6). Verify the version with `pip show xformers`.
- Make sure to look at the official installation guide in [xformers](https://github.com/facebookresearch/xformers) too!

----------

All other providers are either native PyTorch implementations or require a specific PyTorch version (for example, Flex Attention requires torch version of atleast 2.5.0).

----------

## Usage

There are two ways to use the attention dispatcher mechanism:
- Replace `scaled_dot_product_attention` globally:
  ```python
  import torch.nn.functional as F
  from finetrainers.models.attention_dispatch import attention_dispatch

  F.scaled_dot_product_attention = attention_dispatch
  ```
- Replace all occurrences of `scaled_dot_product_attention` in your code with `attention_dispatch`.

```python
# Use dispatcher directly
from finetrainers.models.attention_dispatch import attention_provider, AttentionProvider

with attention_provider(AttentionProvider.FLASH_VARLEN):
    model(...)

# or,
with attention_provider("sage_varlen"):
    model(...)
```

## Context Parallel

TODO
