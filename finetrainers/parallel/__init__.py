from enum import Enum
from typing import Union

from .accelerate import AccelerateParallelState
from .ptd import PytorchDTensorParallelState
from .utils import apply_ddp_ptd, apply_fsdp2_ptd, dist_max, dist_mean


ParallelBackendType = Union[AccelerateParallelState, PytorchDTensorParallelState]


class ParallelBackend(str, Enum):
    ACCELERATE = "accelerate"
    PTD = "ptd"


def get_parallel_backend_cls(backend: ParallelBackend) -> ParallelBackendType:
    if backend == ParallelBackend.ACCELERATE:
        return AccelerateParallelState
    if backend == ParallelBackend.PTD:
        return PytorchDTensorParallelState
    raise ValueError(f"Unknown parallel backend: {backend}")
