import io
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch
import torch.distributed.checkpoint.stateful

from .logging import logger
from .utils import get_device_info


_device_type, _ = get_device_info()


@dataclass
class TrainState(torch.distributed.checkpoint.stateful.Stateful):
    step: int = 0
    observed_data_samples: int = 0
    global_avg_losses: List[float] = field(default_factory=list)
    global_max_losses: List[float] = field(default_factory=list)
    log_steps: List[int] = field(default_factory=list)

    def state_dict(self) -> Dict[str, Any]:
        # Only checkpoint global_avg_losses and global_max_losses per log frequency
        # to avoid sync overhead in every iteration.
        global_avg_losses_bytes = io.BytesIO()
        torch.save(self.global_avg_losses, global_avg_losses_bytes)
        global_max_losses_bytes = io.BytesIO()
        torch.save(self.global_max_losses, global_max_losses_bytes)
        log_steps_bytes = io.BytesIO()
        torch.save(self.log_steps, log_steps_bytes)
        return {
            "step": torch.tensor(self.step, dtype=torch.int32),
            "observed_data_samples": torch.tensor(self.observed_data_samples, dtype=torch.int32),
            "global_avg_losses": global_avg_losses_bytes,
            "global_max_losses": global_max_losses_bytes,
            "log_steps": log_steps_bytes,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        state_dict["global_avg_losses"].seek(0)
        state_dict["global_max_losses"].seek(0)
        state_dict["log_steps"].seek(0)

        self.step = state_dict["step"].item()
        self.observed_data_samples = state_dict["observed_data_samples"].item()
        self.global_avg_losses = torch.load(state_dict["global_avg_losses"], weights_only=False)
        self.global_max_losses = torch.load(state_dict["global_max_losses"], weights_only=False)
        self.log_steps = torch.load(state_dict["log_steps"], weights_only=False)


class ParallelState:
    def __init__(
        self,
        world_size: int,
        pipeline_parallel_degree: int = 1,
        data_parallel_degree: int = 1,
        data_parallel_shards: int = -1,
        context_parallel_degree: int = 1,
        tensor_parallel_degree: int = 1,
    ) -> None:
        self._world_size = world_size
        self.data_parallel_degree = data_parallel_degree
        self.context_parallel_degree = context_parallel_degree
        self.tensor_parallel_degree = tensor_parallel_degree
        self.pipeline_parallel_degree = pipeline_parallel_degree
        self.data_parallel_shards = data_parallel_shards

        for degree in [
            data_parallel_degree,
            context_parallel_degree,
            tensor_parallel_degree,
            pipeline_parallel_degree,
        ]:
            if degree < 1:
                raise ValueError(f"Parallel degree must be at least 1, got {degree}.")

        if data_parallel_shards == -1:
            self.data_parallel_shards = world_size // (
                data_parallel_degree * context_parallel_degree * tensor_parallel_degree * pipeline_parallel_degree
            )

        if (
            self.data_parallel_shards
            * data_parallel_degree
            * context_parallel_degree
            * tensor_parallel_degree
            * pipeline_parallel_degree
            != world_size
        ):
            raise ValueError(
                f"World size {world_size} must be divisible by the product of all parallel degrees and data parallel shards."
            )

        logger.info(
            f"Initialized parallel state with:\n"
            f"  - World size: {world_size}\n"
            f"  - Pipeline parallel degree: {pipeline_parallel_degree}\n"
            f"  - Data parallel degree: {data_parallel_degree}\n"
            f"  - Context parallel degree: {context_parallel_degree}\n"
            f"  - Tensor parallel degree: {tensor_parallel_degree}\n"
            f"  - Data parallel shards: {data_parallel_shards}\n"
        )

        self.mesh = None

    def get_mesh(self, device_type: str = "cuda") -> torch.distributed.DeviceMesh:
        if self.mesh is not None:
            return self.mesh

        mesh_list = [
            ("pp", self.pipeline_parallel_degree),
            ("dp_replicate", self.data_parallel_degree),
            ("dp_shard", self.data_parallel_shards),
            ("cp", self.context_parallel_degree),
            ("tp", self.tensor_parallel_degree),
        ]
        mesh_list = [(name, degree) for name, degree in mesh_list if degree > 1]
        logger.info(f"Creating device mesh with {dict(mesh_list)}")

        names = [x[0] for x in mesh_list]
        degrees = [x[1] for x in mesh_list]
        mesh = torch.distributed.device_mesh.init_device_mesh(device_type, mesh_shape=degrees, mesh_dim_names=names)

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

    def wait_for_everyone(self):
        return torch.distributed.barrier()

    def destroy(self):
        return torch.distributed.destroy_process_group()

    @property
    def pipeline_parallel_enabled(self):
        return self.pipeline_parallel_degree > 1

    @property
    def data_parallel_enabled(self):
        return self.data_parallel_degree > 1 or self.data_parallel_shards > 1

    @property
    def data_replication_enabled(self):
        return self.data_parallel_degree > 1

    @property
    def data_sharding_enabled(self):
        return self.data_parallel_shards > 1

    @property
    def context_parallel_enabled(self):
        return self.context_parallel_degree > 1

    @property
    def tensor_parallel_enabled(self):
        return self.tensor_parallel_degree > 1

    @property
    def non_data_parallel_size(self):
        return self.pipeline_parallel_degree * self.context_parallel_degree * self.tensor_parallel_degree

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


@dataclass
class State:
    # Parallel state
    parallel: ParallelState = None
    dp_mesh: torch.distributed.DeviceMesh = None
    pp_mesh: torch.distributed.DeviceMesh = None
    dp_degree: int = None
    dp_rank: int = None

    # Training state
    train_state: TrainState = None
    num_trainable_parameters: int = 0
    generator: torch.Generator = None

    # Hub state
    repo_id: str = None

    # Artifacts state
    output_dir: str = None
