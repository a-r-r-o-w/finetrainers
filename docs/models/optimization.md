# Memory optimizations

To lower memory requirements during training:

- `--precompute_conditions`: this precomputes the conditions and latents, and loads them as required during training, which saves a significant amount of time and memory.
- `--gradient_checkpointing`: this saves memory by recomputing activations during the backward pass.
- `--layerwise_upcasting_modules transformer`: naively casts the model weights to `torch.float8_e4m3fn` or `torch.float8_e5m2`. This halves the memory requirement for model weights. Computation is performed in the dtype set by `--transformer_dtype` (which defaults to `bf16`)
- `--use_8bit_bnb`: this is only applicable to Adam and AdamW optimizers, and makes use of 8-bit precision to store optimizer states.
- Use a DeepSpeed config to launch training (refer to [`accelerate_configs/deepspeed.yaml`](./accelerate_configs/deepspeed.yaml) as an example).
- Do not perform validation/testing. This saves a significant amount of memory, which can be used to focus solely on training if you're on smaller VRAM GPUs.

We will continue to add more features that help to reduce memory consumption.
