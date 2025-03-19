from enum import Enum
from typing import Union

from .accelerate import AccelerateParallelBackend
from .ptd import PytorchDTensorParallelBackend
from .utils import dist_max, dist_mean


ParallelBackendType = Union[AccelerateParallelBackend, PytorchDTensorParallelBackend]


class ParallelBackendEnum(str, Enum):
    ACCELERATE = "accelerate"
    PTD = "ptd"


def get_parallel_backend_cls(backend: ParallelBackendEnum) -> ParallelBackendType:
    if backend == ParallelBackendEnum.ACCELERATE:
        return AccelerateParallelBackend
    if backend == ParallelBackendEnum.PTD:
        return PytorchDTensorParallelBackend
    raise ValueError(f"Unknown parallel backend: {backend}")
