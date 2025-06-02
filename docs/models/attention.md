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

References and reading material:
- https://docs.pytorch.org/tutorials/prototype/context_parallel.html
- https://insujang.github.io/2024-09-20/introducing-context-parallelism/
- https://www.youtube.com/watch?v=ws7angQYIxI
- https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
- https://arxiv.org/abs/2309.14509

There are three steps to enabling context parallelism with any model:
- Defining the context parallel plan: This is a dictionary that mentions what tensors to split and gather across CP region at different layers in the model
- Applying the CP plan with `apply_context_parallel` function: This registers the necessary hooks to split and gather tensors at the right places in the model without having to manually modify the model code.
- Running model under the `attention_provider` context manager

For a quick example, refer to the [inference example](#inference) below.

The CP plan is a dictionary that maps the name of the module to a list of `CPInput` or `CPOutput` objects. The keys in the dictionary are the names of the internal modules in the model, and the values are dictionaries that map a parameter identifier (either as an argument index or keyword argument as used in the forward method) to a `CPInput` or `CPOutput` object. The `CPInput` object specifies the input tensor to be split, and the `CPOutput` object specifies the output tensor to be gathered.

```python
class ParamId:
    name: Optional[str] = None
    index: Optional[int] = None

class CPInput:
    split_dim: int
    expected_dims: Optional[int] = None
    split_output: bool = False

class CPOutput:
    gather_dim: int
    expected_dims: Optional[int] = None
```

- The `split_dim` and `gather_dim` parameters specify the dimension along which to split or gather the tensor. When using CP with native scaled dot product attention from pytorch, the tensor shape is `[B, N, S, D]`, so the `split_dim` and `gather_dim` parameters should be set to `2` as it is the sequence dimension.
- The `expected_dims` parameter is an optional parameter that is used for sanity checking if the tensor contains the expected number of dimensions.
- By default, `CPInput`'s are split in a pre-forward hook and `CPOutput`'s are gathered in a post-forward hook. If you want to split the output of a module, you can set the `split_output` parameter to `True`. This will split the output tensor in the post-forward hook instead of the pre-forward hook.

- Attention providers supported for training with CP: `flash`, `_native_cudnn`, `_native_efficient`, `_native_flash`
- Attention providers supported for inference with CP: `flash`, `_native_cudnn`, `_native_efficient`, `_native_flash`

### Training

To enable training with context parallelism, you need to make sure a suitable CP plan is registered for the model you are using and launch training with `--cp_degree 2`. For models supported in finetrainers, this is internally done in the [transformer metadata](https://github.com/a-r-r-o-w/finetrainers/tree/main/finetrainers/models/_metadata/transformer.py) file. For custom models, make sure to pass the `plan` argument to the `apply_context_parallel` function.

Currently supported models include: CogVideoX, CogView4, Flux, Wan 2.1. Support for more models and attention providers is in progress.

### Inference

The following example shows how to run context parallel inference. For more examples and ready-to-use inference scripts, check out the [examples/inference](https://github.com/a-r-r-o-w/finetrainers/tree/main/examples/inference/) folder.

<details>
<summary> Example </summary>

```python
import torch
import torch.distributed as dist
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video

from finetrainers._metadata import ParamId, CPInput, CPOutput
from finetrainers.parallel.ptd import apply_context_parallel
from finetrainers.models.attention_dispatch import attention_provider, attention_dispatch

torch.nn.functional.scaled_dot_product_attention = attention_dispatch


def apply_compile(model: torch.nn.Module, compile_scope: str) -> torch.nn.Module:
    r"""Apply torch.compile to a model or its submodules if not already compiled."""
    if getattr(model, "_torch_compiled", False):
        return model  # Already compiled

    if compile_scope == "full":
        model = torch.compile(model)
        setattr(model, "_torch_compiled", True)
    elif compile_scope == "regional":
        if isinstance(model, torch.nn.ModuleList):
            for name, module in model.named_children():
                if not getattr(module, "_torch_compiled", False):
                    compiled_module = torch.compile(module, mode="max-autotune-no-cudagraphs", fullgraph=False, dynamic=False)
                    setattr(compiled_module, "_torch_compiled", True)
                    model.register_module(name, compiled_module)
        else:
            for name, module in model.named_children():
                apply_compile(module, compile_scope)
    else:
        raise ValueError(f"Unknown compile mode: {compile_scope}. Use 'full' or 'regional'.")

    return model


torch.manual_seed(0)
dist.init_process_group("nccl")
rank, world_size = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank)
cp_mesh = dist.device_mesh.init_device_mesh("cuda", [world_size], mesh_dim_names=["cp"])

cp_plan = {
    "rope": {
        ParamId(index=0): CPInput(2, 4, split_output=True),
    },
    "blocks.*": {
        ParamId("encoder_hidden_states", 1): CPInput(1, 3),
    },
    "blocks.0": {
        ParamId("hidden_states", 0): CPInput(1, 3),
    },
    "proj_out": [CPOutput(1, 3)],
}

try:
    model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
    pipe.to("cuda")

    apply_context_parallel(pipe.transformer, mesh=cp_mesh, plan=cp_plan)

    apply_compile(pipe.transformer, compile_scope="regional")

    prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

    with torch.no_grad():
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt=prompt, negative_prompt=negative_prompt, device="cuda",
        )
    
    attention_backend = "_native_flash"
    generator = torch.Generator().manual_seed(0)
    
    # Warmup for compilation
    with attention_provider(attention_backend, mesh=cp_mesh, convert_to_fp32=True, rotate_method="alltoall"):
        latents = pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            height=480,
            width=832,
            num_frames=81,
            num_inference_steps=2,
            guidance_scale=5.0,
            output_type="latent",
            generator=generator,
        ).frames[0]

    # Inference
    with attention_provider(attention_backend, mesh=cp_mesh, convert_to_fp32=True, rotate_method="allgather"):
        latents = pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            height=480,
            width=832,
            num_frames=81,
            guidance_scale=5.0,
            num_inference_steps=30,
            output_type="latent",
            generator=generator,
        ).frames[0]
    
    with torch.no_grad():
        latents = latents.to(pipe.vae.dtype)
        latents_mean = (
            torch.tensor(pipe.vae.config.latents_mean)
            .view(1, pipe.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        video = pipe.vae.decode(latents, return_dict=False)[0]
        video = pipe.video_processor.postprocess_video(video, output_type="pil")[0]
    
    if rank == 0:
        export_to_video(video, "output.mp4", fps=16)
finally:
    dist.destroy_process_group()
```

</details>
