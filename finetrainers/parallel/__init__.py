from enum import Enum

from .base import BaseParallelState
from .finetrainers import FinetrainersParallelState
from .utils import apply_ddp, apply_fsdp2, dist_max, dist_mean


class ParallelBackend(str, Enum):
    FINETRAINERS = "finetrainers"


def get_parallel_state_cls(backend: ParallelBackend) -> BaseParallelState:
    if backend == ParallelBackend.FINETRAINERS:
        return FinetrainersParallelState
    raise ValueError(f"Unknown parallel backend: {backend}")
