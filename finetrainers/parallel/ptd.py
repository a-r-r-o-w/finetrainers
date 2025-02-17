import datetime
import os

import torch

from ..logging import logger
from ..utils import get_device_info
from .base import BaseParallelState


_device_type, _device_module = get_device_info()


class PytorchDTensorParallelState(BaseParallelState):
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
    ) -> None:
        super().__init__()

        self._world_size = world_size
        self._pp_degree = pp_degree
        self._dp_degree = dp_degree
        self._dp_shards = dp_shards
        self._cp_degree = cp_degree
        self._tp_degree = tp_degree
        self._backend = backend
        self._timeout = timeout

        for degree in [pp_degree, dp_degree, cp_degree, tp_degree]:
            if degree < 1:
                raise ValueError(f"Parallel degree must be at least 1, got {degree}.")

        if dp_shards == -1:
            self._dp_shards = world_size // (pp_degree * dp_degree * cp_degree * tp_degree)

        if self._dp_shards * pp_degree * dp_degree * cp_degree * tp_degree != world_size:
            raise ValueError(
                f"World size {world_size} must be divisible by the product of all parallel degrees and data parallel shards."
            )

        self._init_dist()

        logger.info(
            f"Initialized parallel state with:\n"
            f"  - World size: {world_size}\n"
            f"  - Pipeline parallel degree: {pp_degree}\n"
            f"  - Data parallel degree: {dp_degree}\n"
            f"  - Context parallel degree: {cp_degree}\n"
            f"  - Tensor parallel degree: {tp_degree}\n"
            f"  - Data parallel shards: {dp_shards}\n"
        )

        self.mesh = None

    def _init_dist(self):
        torch.distributed.init_process_group(backend=self._backend, timeout=datetime.timedelta(seconds=self._timeout))
        _device_module.set_device(self.local_rank)

    def get_mesh(self) -> torch.distributed.DeviceMesh:
        if self.mesh is not None:
            return self.mesh

        mesh_list = [
            ("pp", self._pp_degree),
            ("dp_replicate", self._dp_degree),
            ("dp_shard", self._dp_shards),
            ("cp", self._cp_degree),
            ("tp", self._tp_degree),
        ]
        mesh_list = [(name, degree) for name, degree in mesh_list if degree > 1]
        logger.info(f"Creating device mesh with {dict(mesh_list)}")

        names = [x[0] for x in mesh_list]
        degrees = [x[1] for x in mesh_list]
        mesh = torch.distributed.device_mesh.init_device_mesh(_device_type, mesh_shape=degrees, mesh_dim_names=names)

        dp_mesh_names = []
        dp_cp_mesh_names = []
        dp_shard_cp_mesh_names = []

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

        self.mesh = mesh
        return mesh

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
