from typing import Optional

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.tensor
from diffusers.utils import is_accelerate_available
from torch.distributed._composable.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
from torch.distributed._composable.replicate import replicate

from ..utils._common import DIFFUSERS_TRANSFORMER_BLOCK_NAMES


if is_accelerate_available():
    from accelerate import Accelerator
    from accelerate.utils import (
        DataLoaderConfiguration,
        DistributedDataParallelKwargs,
        InitProcessGroupKwargs,
        ProjectConfiguration,
    )


def apply_fsdp2_ptd(
    model: torch.nn.Module,
    dp_mesh: torch.distributed.device_mesh.DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    output_dtype: torch.dtype,
    pp_enabled: bool = False,
    cpu_offload: bool = False,
) -> None:
    r"""Apply FSDP2 on a model."""
    mp_policy = MixedPrecisionPolicy(param_dtype, reduce_dtype, output_dtype, cast_forward_inputs=True)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}

    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy(pin_memory=True)

    def apply_fully_shard(blocks):
        for layer_index, block in enumerate(blocks):
            if pp_enabled:
                # For PP, do not reshard after forward to avoid per-microbatch
                # all-gathers, which can be expensive and non-overlapped
                reshard_after_forward = False
            else:
                # As an optimization, do not reshard after forward for the last
                # transformer block since FSDP would prefetch it immediately
                reshard_after_forward = layer_index < len(blocks) - 1
            fully_shard(block, **fsdp_config, reshard_after_forward=reshard_after_forward)

    for transformer_block_name in DIFFUSERS_TRANSFORMER_BLOCK_NAMES:
        blocks = getattr(model, transformer_block_name, None)
        if blocks is not None:
            apply_fully_shard(blocks)

    fully_shard(model, **fsdp_config, reshard_after_forward=not pp_enabled)


def apply_ddp_accelerate(
    model: torch.nn.Module,
    project_config: Optional[ProjectConfiguration] = None,
    ddp_kwargs: Optional[DistributedDataParallelKwargs] = None,
    init_process_group_kwargs: Optional[InitProcessGroupKwargs] = None,
    dataloader_config: Optional[DataLoaderConfiguration] = None,
    gradient_accumulation_steps: Optional[int] = None,
    accelerator: Optional[Accelerator] = None,
) -> torch.nn.Module:
    if accelerator is None:
        accelerator = Accelerator(
            project_config=project_config,
            dataloader_config=dataloader_config,
            gradient_accumulation_steps=gradient_accumulation_steps,
            log_with=None,
            kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
        )
        if torch.backends.mps.is_available():
            accelerator.native_amp = False
    accelerator.prepare_model(model)
    return accelerator, model


def apply_ddp_ptd(model: torch.nn.Module, dp_mesh: torch.distributed.device_mesh.DeviceMesh) -> None:
    replicate(model, device_mesh=dp_mesh, bucket_cap_mb=100)


def dist_reduce(x: torch.Tensor, reduceOp: str, mesh: torch.distributed.device_mesh.DeviceMesh) -> float:
    if isinstance(x, torch.distributed.tensor.DTensor):
        # functional collectives do not support DTensor inputs
        x = x.full_tensor()
    assert x.numel() == 1  # required by `.item()`
    return funcol.all_reduce(x, reduceOp=reduceOp, group=mesh).item()


def dist_max(x: torch.Tensor, mesh: torch.distributed.device_mesh.DeviceMesh) -> float:
    return dist_reduce(x, reduceOp=torch.distributed.distributed_c10d.ReduceOp.MAX.name, mesh=mesh)


def dist_mean(x: torch.Tensor, mesh: torch.distributed.device_mesh.DeviceMesh) -> float:
    return dist_reduce(x, reduceOp=torch.distributed.distributed_c10d.ReduceOp.AVG.name, mesh=mesh)
