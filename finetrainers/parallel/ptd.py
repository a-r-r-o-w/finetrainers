import datetime
import functools
import os
import pathlib
import shutil
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import datasets.distributed
import torch
import torch.distributed.checkpoint
import torch.distributed.checkpoint.stateful
from torch.distributed._composable.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
from torch.distributed._composable.replicate import replicate
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)

from finetrainers.data import DPDataLoader
from finetrainers.logging import get_logger
from finetrainers.utils import enable_determinism, get_device_info
from finetrainers.utils._common import DIFFUSERS_TRANSFORMER_BLOCK_NAMES

from .base import BaseCheckpointer, BaseParallelBackend


if TYPE_CHECKING:
    from finetrainers import optimizer


_device_type, _device_module = get_device_info()
logger = get_logger()


class PytorchDTensorParallelBackend(BaseParallelBackend):
    def __init__(
        self,
        world_size: int,
        pp_degree: int = 1,
        dp_degree: int = 1,
        dp_shards: int = -1,
        cp_degree: int = 1,
        tp_degree: int = 1,
        backend: str = "nccl",
        timeout: int = 180,
        logging_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        gradient_accumulation_steps: Optional[int] = None,
    ) -> None:
        super().__init__()

        self._world_size = world_size
        self._pp_degree = pp_degree
        self._dp_degree = dp_degree
        self._dp_shards = dp_shards
        self._cp_degree = cp_degree
        self._tp_degree = tp_degree
        self._output_dir = pathlib.Path(output_dir) if output_dir is not None else None
        self._logging_dir = (
            self._output_dir / logging_dir if output_dir is not None and logging_dir is not None else None
        )
        self._backend = backend
        self._timeout = timeout

        for degree in [pp_degree, dp_degree, dp_shards, cp_degree, tp_degree]:
            if degree < 1:
                raise ValueError(f"Parallel degree must be at least 1, got {degree}.")

        if dp_shards * pp_degree * dp_degree * cp_degree * tp_degree != world_size:
            raise ValueError(
                f"World size {world_size} must be divisible by the product of all parallel degrees and data parallel shards."
            )

        torch.distributed.init_process_group(backend=self._backend, timeout=datetime.timedelta(seconds=self._timeout))
        _device_module.set_device(self.local_rank)

        logger.info(
            f"Initialized parallel state with:\n"
            f"  - World size: {world_size}\n"
            f"  - Pipeline parallel degree: {pp_degree}\n"
            f"  - Data parallel degree: {dp_degree}\n"
            f"  - Context parallel degree: {cp_degree}\n"
            f"  - Tensor parallel degree: {tp_degree}\n"
            f"  - Data parallel shards: {dp_shards}\n"
        )

        self._mesh: torch.distributed.DeviceMesh = None

    def enable_determinism(self, seed):
        world_mesh = self.get_mesh()
        enable_determinism(seed, world_mesh)

    def apply_ddp(
        self, model: torch.nn.Module, device_mesh: Optional[torch.distributed.DeviceMesh] = None
    ) -> torch.nn.Module:
        if device_mesh is None:
            device_mesh = self.get_mesh()
        apply_ddp(model, device_mesh)
        logger.debug("Applied PytorchDTensorParallel::apply_ddp to model.")
        return model

    def apply_fsdp2(
        self,
        model: torch.nn.Module,
        param_dtype: torch.dtype,
        reduce_dtype: torch.dtype,
        output_dtype: torch.dtype,
        pp_enabled: bool = False,
        cpu_offload: bool = False,
        device_mesh: Optional[torch.distributed.DeviceMesh] = None,
    ) -> torch.nn.Module:
        if device_mesh is None:
            device_mesh = self.get_mesh()
        apply_fsdp2(model, device_mesh, param_dtype, reduce_dtype, output_dtype, pp_enabled, cpu_offload)
        logger.debug("Applied PytorchDTensorParallel::apply_fsdp2 to model.")
        return model

    def prepare_model(self, model: torch.nn.Module) -> torch.nn.Module:
        return model

    def prepare_dataset(self, dataset: torch.utils.data.IterableDataset) -> torch.utils.data.IterableDataset:
        dp_mesh = self.get_mesh("dp_replicate")
        if dp_mesh is None:
            dp_mesh = self.get_mesh()
        if self.world_size > 1:
            dp_local_rank, dp_world_size = dp_mesh.get_local_rank(), dp_mesh.size()
        else:
            dp_local_rank, dp_world_size = 0, 1
        dataset._data = datasets.distributed.split_dataset_by_node(dataset._data, dp_local_rank, dp_world_size)
        logger.debug("PytorchDTensorParallelBackend::prepare_dataset completed!")
        return dataset

    def prepare_dataloader(
        self, dataset: torch.utils.data.IterableDataset, batch_size: int, num_workers: int, pin_memory: bool
    ) -> DPDataLoader:
        dp_mesh = self.get_mesh("dp_replicate")
        if dp_mesh is None:
            dp_mesh = self.get_mesh()
        if self.world_size > 1:
            dp_local_rank = dp_mesh.get_local_rank()
        else:
            dp_local_rank = 0
        dataloader = DPDataLoader(dp_local_rank, dataset, batch_size=batch_size, num_workers=num_workers)
        logger.debug("PytorchDTensorParallelBackend::prepare_dataloader completed!")
        return dataloader

    def prepare_optimizer(self, optimizer, lr_scheduler):
        logger.debug("PytorchDTensorParallelBackend::prepare_optimizer completed!")
        return optimizer, lr_scheduler

    def get_mesh(self, name: Optional[str] = None) -> torch.distributed.DeviceMesh:
        def _get_mesh():
            if name is None:
                return self._mesh
            try:
                return self._mesh[name]
            except (KeyError, RuntimeError):
                if self._mesh.ndim == 0:
                    return None
                return self._mesh

        if self._mesh is not None:
            return _get_mesh()

        mesh_list = [
            ("pp", self._pp_degree),
            ("dp_replicate", self._dp_degree),
            ("dp_shard", self._dp_shards),
            ("cp", self._cp_degree),
            ("tp", self._tp_degree),
        ]
        mesh_list = [(name, degree) for name, degree in mesh_list if degree > 1]
        names = [x[0] for x in mesh_list]
        degrees = [x[1] for x in mesh_list]
        mesh = torch.distributed.device_mesh.init_device_mesh(_device_type, mesh_shape=degrees, mesh_dim_names=names)

        dp_mesh_names, dp_cp_mesh_names, dp_shard_cp_mesh_names = [], [], []

        if self.data_replication_enabled:
            dp_mesh_names.append("dp_replicate")
            dp_cp_mesh_names.append("dp_replicate")
        if self.data_sharding_enabled:
            dp_mesh_names.append("dp_shard")
            dp_cp_mesh_names.append("dp_shard")
            dp_shard_cp_mesh_names.append("dp_shard")
        if self.context_parallel_enabled:
            dp_cp_mesh_names.append("cp")
            dp_shard_cp_mesh_names.append("cp")

        if len(dp_mesh_names) > 0:
            mesh[tuple(dp_mesh_names)]._flatten(mesh_dim_name="dp")
        if len(dp_cp_mesh_names) > 0:
            mesh[tuple(dp_cp_mesh_names)]._flatten(mesh_dim_name="dp_cp")
        if len(dp_shard_cp_mesh_names) > 0:
            mesh[tuple(dp_shard_cp_mesh_names)]._flatten(mesh_dim_name="dp_shard_cp")

        logger.debug(f"Device mesh: {mesh}")
        self._mesh = mesh
        return _get_mesh()

    def get_checkpointer(self, *args, **kwargs):
        return PTDCheckpointer(*args, **kwargs)

    @property
    def world_size(self):
        return torch.distributed.get_world_size()

    @property
    def rank(self):
        return torch.distributed.get_rank()

    @property
    def local_rank(self):
        return int(os.environ.get("LOCAL_RANK", 0))

    @property
    def is_main_process(self):
        r"""Returns `True` if the current process is the main process on the master node."""
        return self.rank == 0

    @property
    def is_local_main_process(self):
        r"""Returns `True` if the current process is the main process on local node."""
        return self.local_rank == 0

    @property
    def device(self):
        return torch.device(_device_type, self.local_rank)

    def wait_for_everyone(self):
        return torch.distributed.barrier()

    # @contextmanager
    # def main_process_first(self):
    #     if self.is_main_process:
    #         yield
    #         self.wait_for_everyone()
    #     else:
    #         self.wait_for_everyone()
    #         yield

    def destroy(self):
        if self.is_main_process and self.tracker is not None:
            self.tracker.finish()
        return torch.distributed.destroy_process_group()

    @property
    def pipeline_parallel_enabled(self):
        return self._pp_degree > 1

    @property
    def data_parallel_enabled(self):
        return self._dp_degree > 1 or self._dp_shards > 1

    @property
    def data_replication_enabled(self):
        return self._dp_degree > 1

    @property
    def data_sharding_enabled(self):
        return self._dp_shards > 1

    @property
    def context_parallel_enabled(self):
        return self._cp_degree > 1

    @property
    def tensor_parallel_enabled(self):
        return self._tp_degree > 1


class ModelWrapper(torch.distributed.checkpoint.stateful.Stateful):
    def __init__(self, model: Union[torch.nn.Module, List[torch.nn.Module]]) -> None:
        self.model = [model] if isinstance(model, torch.nn.Module) else model

    def state_dict(self) -> Dict[str, Any]:
        return {k: v for sd in map(get_model_state_dict, self.model) for k, v in sd.items()}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        func = functools.partial(
            set_model_state_dict,
            model_state_dict=state_dict,
            options=StateDictOptions(strict=False),
        )
        list(map(func, self.model))


class PTDCheckpointer(BaseCheckpointer):
    def __init__(
        self,
        dataloader: torch.utils.data.DataLoader,
        model_parts: List[torch.nn.Module],
        optimizers: "optimizer.OptimizerWrapper",
        schedulers: "optimizer.SchedulerWrapper",
        states: Dict[str, Any],
        checkpointing_steps: int,
        checkpointing_limit: int,
        output_dir: str,
        enable: bool = True,
        _callback_fn: Callable[[Dict[str, Any]], Dict[str, Any]] = None,
        _prefix: str = "finetrainers_step",
    ) -> None:
        self.states = states
        self.states.update(
            {
                "model": ModelWrapper(model_parts),
                "optimizer": optimizers,
                "dataloader": dataloader,
            }
        )
        self.states.update(schedulers.get_lr_scheduler_state())

        self.checkpointing_steps = checkpointing_steps
        self.checkpointing_limit = checkpointing_limit
        self.output_dir = pathlib.Path(output_dir)
        self.enable = enable
        self._callback_fn = _callback_fn
        self._prefix = _prefix

        logger.info(f"Checkpointing enabled. Checkpoints will be stored in '{self.output_dir}'")

    def save(self, step: int = -1, force: bool = False, *, _device: torch.device, _is_main_process: bool) -> str:
        if not self._should_checkpoint(step, force):
            return None

        checkpoint_dir = self._get_checkpoint_dir(step)
        begin_time = time.monotonic()
        torch.distributed.checkpoint.save(self.states, checkpoint_id=checkpoint_dir.as_posix())
        end_time = time.monotonic()
        logger.info(
            f"Saved checkpoint in {end_time - begin_time:.2f} seconds at step {step}. Directory: {checkpoint_dir}"
        )
        self._purge_stale_checkpoints()

        state_dicts = [
            gather_state_dict_on_cpu_rank0(model, _device, is_main_process=_is_main_process)
            for model in self.states["model"].model
        ]
        if self._callback_fn is not None:
            list(map(self._callback_fn, state_dicts))

        return checkpoint_dir.as_posix()

    def load(self, step: int = -1) -> bool:
        if not self.enable:
            return False
        if not self.output_dir.exists():
            return False
        if step != -1 and not self._get_checkpoint_dir(step).exists():
            return False

        if step == -1:
            latest_checkpoint_dir = self._find_latest_checkpoint_dir()
            if latest_checkpoint_dir is None:
                return False
            step = int(latest_checkpoint_dir.name.split("_")[-1])

        checkpoint_dir = self._get_checkpoint_dir(step)
        logger.info(f"Loading checkpoint from '{checkpoint_dir}' at step {step}")

        # For step 0, optimizers/schedulers are not available as they are created during training after first step
        states = {"model": self.states["model"]} if step == 0 else self.states

        # See bug: https://github.com/pytorch/pytorch/pull/138575
        original_stateful_states = {
            k: v for k, v in states.items() if isinstance(v, torch.distributed.checkpoint.stateful.Stateful)
        }
        begin_time = time.monotonic()
        torch.distributed.checkpoint.load(states, checkpoint_id=checkpoint_dir.as_posix())
        end_time = time.monotonic()
        logger.info(f"Loaded checkpoint in {end_time - begin_time:.2f} seconds.")

        # bugfix from above: restore the original stateful objects, whose states were already updated in-place by dcp.load()
        states.update(original_stateful_states)

        return True

    def _should_checkpoint(self, step: int, force: bool) -> bool:
        if not self.enable:
            return False
        if not force:
            if step % self.checkpointing_steps != 0:
                return False
        return True

    def _get_checkpoint_dir(self, step: int) -> pathlib.Path:
        return self.output_dir / f"{self._prefix}_{step}"

    def _find_latest_checkpoint_dir(self) -> Optional[pathlib.Path]:
        checkpoints = sorted(self.output_dir.glob(f"{self._prefix}_*"), key=lambda x: int(x.name.split("_")[-1]))
        return checkpoints[-1] if len(checkpoints) > 0 else None

    def _purge_stale_checkpoints(self) -> None:
        if self.checkpointing_limit is None or self.checkpointing_limit <= 0:
            return
        checkpoints = sorted(
            self.output_dir.glob(f"{self._prefix}_*"), key=lambda x: int(x.name.split("_")[-1]), reverse=True
        )
        for checkpoint in checkpoints[self.checkpointing_limit :]:
            logger.info(f"Deleting stale checkpoint: {checkpoint}")
            shutil.rmtree(checkpoint, ignore_errors=True)


def gather_state_dict_on_cpu_rank0(
    model, device: Optional[torch.device] = None, *, is_main_process: bool
) -> Dict[str, Any]:
    cpu_state_dict = {}
    sharded_sd = model.state_dict()
    for param_name, param in sharded_sd.items():
        if param.is_cpu:
            # Move back to device if offloaded to CPU
            param = param.to(device)
        if hasattr(param, "_local_tensor"):
            # Gather DTensor
            param = param.full_tensor()
        if is_main_process:
            cpu_state_dict[param_name] = param.cpu()
        torch.distributed.barrier()
    return cpu_state_dict


# # Copied from pytorch (torch/distributed/checkpoint/format_utils.py) to support callbacks to modify state_dict
# def dcp_to_torch_save(
#     dcp_checkpoint_dir: Union[str, os.PathLike],
#     torch_save_path: Union[str, os.PathLike],
#     callback_fn: Callable[[Dict[str, Any]], Dict[str, Any]] = None,
# ):
#     """
#     Given a directory containing a DCP checkpoint, this function will convert it into a
#     Torch save file.

#     Args:
#         dcp_checkpoint_dir: Directory containing the DCP checkpoint.
#         torch_save_path: Filename to store the converted Torch save file.
#         callback_fn: Optional callback function that takes the state_dict as input and returns a modified state_dict.

#     .. warning::
#         To avoid OOM, it's recommended to only run this function on a single rank.
#     """
#     state_dict = {}
#     _load_state_dict(
#         state_dict,
#         storage_reader=FileSystemReader(dcp_checkpoint_dir),
#         planner=_EmptyStateDictLoadPlanner(),
#         no_dist=True,
#     )
#     if callback_fn is not None:
#         state_dict = callback_fn(state_dict)
#     torch.save(state_dict, torch_save_path)


def apply_ddp(model: torch.nn.Module, dp_mesh: torch.distributed.device_mesh.DeviceMesh) -> None:
    replicate(model, device_mesh=dp_mesh, bucket_cap_mb=100)


def apply_fsdp2(
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
