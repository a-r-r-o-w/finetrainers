# Environment

Finetrainers has only been widely tested with the following environment (output obtained by running `diffusers-cli env`):

```shell
- 🤗 Diffusers version: 0.33.0.dev0
- Platform: Linux-5.4.0-166-generic-x86_64-with-glibc2.31
- Running on Google Colab?: No
- Python version: 3.10.14
- PyTorch version (GPU?): 2.5.1+cu124 (True)
- Flax version (CPU?/GPU?/TPU?): 0.8.5 (cpu)
- Jax version: 0.4.31
- JaxLib version: 0.4.31
- Huggingface_hub version: 0.28.1
- Transformers version: 4.48.0.dev0
- Accelerate version: 1.1.0.dev0
- PEFT version: 0.14.1.dev0
- Bitsandbytes version: 0.43.3
- Safetensors version: 0.4.5
- xFormers version: not installed
- Accelerator: NVIDIA A100-SXM4-80GB, 81920 MiB
NVIDIA A100-SXM4-80GB, 81920 MiB
NVIDIA A100-SXM4-80GB, 81920 MiB
NVIDIA DGX Display, 4096 MiB
NVIDIA A100-SXM4-80GB, 81920 MiB
```

Other versions of dependencies may or may not work as expected. We would like to make finetrainers work on a wider range of environments, but due to the complexity of testing at the early stages of development, we are unable to do so. The long term goals include compatibility with most pytorch versions on CUDA, MPS, ROCm and XLA devices.

> [!IMPORTANT]
>
> For context parallelism, PyTorch 2.6+ is required.

## Configuration

The following environment variables may be configured to change the default behaviour of finetrainers:

`FINETRAINERS_ATTN_PROVIDER`: Sets the default attention provider for training/validation. Defaults to `native`, as in native PyTorch SDPA. See [attention docs](./models/attention.md) for more information.
`FINETRAINERS_ATTN_CHECKS`: Whether or not to run basic sanity checks when using different attention providers. This is useful for debugging but you should leave it disabled for longer training runs. Defaults to `"0"`. Can be set to a truthy env value.
