# Notes for Future Me

>![NOTE]
> This doc page is intended for developers and contributors.

FSDP dump:
- https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
- https://github.com/pytorch/pytorch/issues/114299
- Using FSDP1 requires that all FSDP flat parameters are of the same dtype. For LoRA training, we default lora parameters to fp32 and transformer parameters to dtype chosen by user. There seems to be no easy workaround than performing lora training in same dtype.
- https://github.com/pytorch/pytorch/issues/100945
- https://github.com/pytorch/torchtune/blob/9b3836028fd0b48f593ea43474b86880c49a4d74/recipes/lora_finetune_distributed.py
- https://github.com/KellerJordan/modded-nanogpt/pull/68
- https://github.com/pytorch/pytorch/pull/125394: monkey-patch method for FSDP pre/post-hooks to be triggered for method other than `forward`
- https://github.com/pytorch/pytorch/pull/127786:
- https://github.com/pytorch/pytorch/pull/130949:
- Sanity saver: create optimizers after parallelizing/activation-checkpointing models

DTensor:
- https://github.com/pytorch/pytorch/issues/88838
- https://github.com/pytorch/pytorch/blob/main/test/distributed/tensor/parallel/test_parallelize_api.py
