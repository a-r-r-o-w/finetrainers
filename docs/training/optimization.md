To lower memory requirements during training:

- Use a DeepSpeed config to launch training (refer to [`accelerate_configs/deepspeed.yaml`](./accelerate_configs/deepspeed.yaml) as an example).
- Pass `--precompute_conditions` when launching training.
- Pass `--gradient_checkpointing` when launching training.
- Pass `--use_8bit_bnb` when launching training. Note that this is only applicable to Adam and AdamW optimizers.
- Do not perform validation/testing. This saves a significant amount of memory, which can be used to focus solely on training if you're on smaller VRAM GPUs.

We will continue to add more features that help to reduce memory consumption.