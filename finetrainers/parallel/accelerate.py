import datetime
import pathlib
from typing import Optional

import torch
from diffusers.utils import is_accelerate_available

from ..logging import get_logger
from ..utils import get_device_info
from .base import BaseParallelBackend
from .utils import apply_ddp_accelerate


if not is_accelerate_available():
    raise ImportError(
        "Please install the accelerate package using `pip install accelerate` to use the AccelerateParallelBackend."
    )

from accelerate import Accelerator
from accelerate.data_loader import DataLoader
from accelerate.utils import (
    DataLoaderConfiguration,
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
)


logger = get_logger()
_device_type, _device_module = get_device_info()


class AccelerateParallelBackend(BaseParallelBackend):
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
        self._gradient_accumulation_steps = gradient_accumulation_steps

        if pp_degree > 1 or dp_shards > 1 or cp_degree > 1 or tp_degree > 1:
            raise ValueError(
                "AccelerateParallelBackend does not support anything but Distributed Data Parallelism at the moment."
            )
        if dp_degree != world_size:
            raise ValueError("Data parallel degree must be equal to world size.")

        self._accelerator: Accelerator = None
        self._mesh: torch.distributed.DeviceMesh = None

    def apply_ddp(self, model: torch.nn.Module, *args, **kwargs) -> torch.nn.Module:
        project_config = None
        ddp_kwargs = None
        init_process_group_kwargs = None
        if self._accelerator is None:
            project_config = ProjectConfiguration(project_dir=self._output_dir, logging_dir=self._logging_dir)
            ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
            dataloader_config = DataLoaderConfiguration(
                split_batches=False, dispatch_batches=False, use_stateful_dataloader=True
            )
            init_process_group_kwargs = InitProcessGroupKwargs(
                backend=self._backend, timeout=datetime.timedelta(seconds=self._timeout)
            )
        self._accelerator, model = apply_ddp_accelerate(
            model,
            project_config,
            ddp_kwargs,
            init_process_group_kwargs,
            dataloader_config,
            self._gradient_accumulation_steps,
            accelerator=self._accelerator,
        )
        logger.debug("Applied AccelerateParallel::apply_ddp to model.")
        return model

    def prepare_dataset(self, dataset: torch.utils.data.IterableDataset) -> torch.utils.data.IterableDataset:
        logger.debug("AccelerateParallelBackend::prepare_dataset completed!")
        return dataset

    def prepare_dataloader(
        self,
        dataset: torch.utils.data.IterableDataset,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> DataLoader:
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory
        )
        dataloader = self._accelerator.prepare_data_loader(dataloader)
        logger.debug("AccelerateParallelBackend::prepare_dataloader completed!")
        return dataloader

    def prepare_optimizer(self, optimizer, lr_scheduler):
        optimizer = self._accelerator.prepare_optimizer(optimizer)
        lr_scheduler = self._accelerator.prepare_scheduler(lr_scheduler)
        return optimizer, lr_scheduler

    def get_mesh(self, name: Optional[str] = None) -> torch.distributed.DeviceMesh:
        def _get_mesh():
            if name is None:
                return self._mesh
            try:
                return self._mesh[name]
            except (KeyError, RuntimeError):
                return self._mesh

        if self._mesh is not None:
            return _get_mesh()

        mesh_list = [("dp_replicate", self._dp_degree), ("dp_shard", self._dp_shards)]
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

    @property
    def world_size(self):
        return self._accelerator.num_processes

    @property
    def rank(self):
        return self._accelerator.process_index

    @property
    def local_rank(self):
        return self._accelerator.local_process_index

    @property
    def is_main_process(self):
        r"""Returns `True` if the current process is the main process on the master node."""
        return self._accelerator.is_main_process

    @property
    def is_local_main_process(self):
        r"""Returns `True` if the current process is the main process on local node."""
        return self._accelerator.is_local_main_process

    @property
    def device(self):
        return self._accelerator.device

    def wait_for_everyone(self):
        self._accelerator.wait_for_everyone()

    def destroy(self):
        self._accelerator.end_training()

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
