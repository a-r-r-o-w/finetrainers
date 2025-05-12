# Finetrainers Parallel Backends

Finetrainers supports parallel training on multiple GPUs & nodes. This is done using the Pytorch DTensor backend. To run parallel training, `torchrun` is utilized.

As an experiment for comparing performance of different training backends, Finetrainers has implemented multi-backend support. These backends may or may not fully rely on Pytorch's distributed DTensor solution. Currently, only [🤗 Accelerate](https://github.com/huggingface/accelerate) is supported for backwards-compatibility reasons (as we initially started Finetrainers with only Accelerate). In the near future, there are plans for integrating with:
- [DeepSpeed](https://github.com/deepspeedai/DeepSpeed)
- [Nanotron](https://github.com/huggingface/nanotron)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)

> [!IMPORTANT]
> The multi-backend support is completely experimental and only serves to satisfy my curiosity of how much of a tradeoff there is between performance and ease of use. The Pytorch DTensor backend is the only one with stable support, following Accelerate.
>
> Users will not have to worry about backwards-breaking changes or dependencies if they stick to the Pytorch DTensor backend.

## Support matrix

Currently supported parallelizations include: 
- [DDP](https://pytorch.org/docs/stable/notes/ddp.html)
- [FSDP2](https://pytorch.org/docs/stable/fsdp.html)
- [HSDP](https://pytorch.org/docs/stable/fsdp.html)
- [CP](https://docs.pytorch.org/tutorials/prototype/context_parallel.html)
<!-- - [TP](https://pytorch.org/docs/stable/distributed.tensor.parallel.html) -->

## Training

The following parameters are relevant for launching training:

- `parallel_backend`: The backend to use for parallel training. Available options are `ptd` & `accelerate`.
- `pp_degree`: The degree of pipeline parallelism. Currently unsupported.
- `dp_degree`: The degree of data parallelis/replicas. Defaults to `1`.
- `dp_shards`: The number of shards for data parallelism. Defaults to `1`.
- `cp_degree`: The degree of context parallelism.
- `tp_degree`: The degree of tensor parallelism.

For launching training with the Pytorch DTensor backend, use the following:

```bash
# Single node - 8 GPUs available
torchrun --standalone --nodes=1 --nproc_per_node=8 --rdzv_backend c10d --rdzv_endpoint="localhost:0" train.py <YOUR_OTHER_ARGS>

# Single node - 8 GPUs but only 4 available
export CUDA_VISIBLE_DEVICES=0,2,4,5
torchrun --standalone --nodes=1 --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint="localhost:0" train.py <YOUR_OTHER_ARGS>

# Multi-node - Nx8 GPUs available
# TODO(aryan): Add slurm script
```

For launching training with the Accelerate backend, use the following:

```bash
# Single node - 8 GPUs available
accelerate launch --config_file accelerate_configs/uncompiled_8.yaml --gpu_ids 0,1,2,3,4,5,6,7 train.py <YOUR_OTHER_ARGS>

# Single node - 8 GPUs but only 4 available
accelerate launch --config_file accelerate_configs/uncompiled_4.yaml --gpu_ids 0,2,4,5 train.py <YOUR_OTHER_ARGS>

# Multi-node - Nx8 GPUs available
# TODO(aryan): Add slurm script
```

## Inference

For inference-only purposes, the example implementation can be found in the [examples/inference/](../../examples/inference/) directory.
