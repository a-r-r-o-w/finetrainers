from enum import Enum

from .base import BaseParallelState
from .ptd import PytorchDTensorParallelState
from .utils import apply_ddp, apply_fsdp2, dist_max, dist_mean


class ParallelBackend(str, Enum):
    PTD = "ptd"


def get_parallel_state_cls(backend: ParallelBackend) -> BaseParallelState:
    if backend == ParallelBackend.PTD:
        return PytorchDTensorParallelState
    raise ValueError(f"Unknown parallel backend: {backend}")
