import datetime
import os
import pathlib
from typing import Optional

import datasets.distributed
import torch

from ..data import DPDataLoader
from ..logging import get_logger
from ..utils import get_device_info
from .base import BaseParallelBackend
from .utils import apply_ddp_ptd


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

    def apply_ddp(
        self, model: torch.nn.Module, device_mesh: Optional[torch.distributed.DeviceMesh] = None
    ) -> torch.nn.Module:
        if device_mesh is None:
            device_mesh = self.get_mesh()
        apply_ddp_ptd(model, device_mesh)
        logger.debug("Applied PytorchDTensorParallel::apply_ddp to model.")
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
        if self.is_main_process:
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
