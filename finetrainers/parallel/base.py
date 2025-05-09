from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional

import torch

from finetrainers.trackers import DummyTracker, TrackerType, initialize_trackers


class BaseParallelBackend:
    r"""
    Base class that contains properties and methods that should be implemented by different parallel backends.
    """

    def __init__(self):
        self.tracker = None

    def enable_determinism(self, seed: int) -> None:
        raise NotImplementedError("Method `enable_determinism` must be implemented by subclass.")

    def apply_ddp(self, *args, **kwargs) -> torch.nn.Module:
        raise NotImplementedError("Method `apply_ddp` must be implemented by subclass.")

    def apply_fsdp2(self, *args, **kwargs) -> torch.nn.Module:
        raise NotImplementedError("Method `apply_fsdp2` must be implemented by subclass.")

    def apply_context_parallel(self, *args, **kwargs) -> torch.nn.Module:
        raise NotImplementedError("Method `apply_context_parallel` must be implemented by subclass.")

    def prepare_model(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Method `prepare_model` must be implemented by subclass.")

    def prepare_dataset(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Method `prepare_dataset` must be implemented by subclass.")

    def prepare_dataloader(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Method `prepare_dataloader` must be implemented by subclass.")

    def prepare_optimizer(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Method `prepare_optimizer` must be implemented by subclass.")

    def get_mesh(self, name: Optional[str] = None) -> torch.distributed.DeviceMesh:
        raise NotImplementedError("Method `get_mesh` must be implemented by subclass.")

    def get_checkpointer(self, *args, **kwargs) -> None:
        raise NotImplementedError("Method `get_checkpointer` must be implemented by subclass.")

    def initialize_trackers(
        self, trackers: List[str], experiment_name: str, config: Dict[str, Any], log_dir: str
    ) -> TrackerType:
        if self.is_main_process:
            self.tracker = initialize_trackers(trackers, experiment_name, config, log_dir)
        else:
            self.tracker = DummyTracker()

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        if self.is_main_process:
            self.tracker.log(metrics, step)

    def wait_for_everyone(self):
        raise NotImplementedError("Method `wait_for_everyone` must be implemented by subclass.")

    @contextmanager
    def main_process_first(self):
        raise NotImplementedError("Method `main_process_first` must be implemented by subclass.")

    def destroy(self):
        raise NotImplementedError("Method `destroy` must be implemented by subclass.")

    @property
    def world_size(self):
        raise NotImplementedError("Method `world_size` must be implemented by subclass.")

    @property
    def rank(self):
        raise NotImplementedError("Method `rank` must be implemented by subclass.")

    @property
    def local_rank(self):
        raise NotImplementedError("Method `local_rank` must be implemented by subclass.")

    @property
    def is_main_process(self):
        raise NotImplementedError("Method `is_main_process` must be implemented by subclass.")

    @property
    def is_local_main_process(self):
        raise NotImplementedError("Method `is_local_main_process` must be implemented by subclass.")

    @property
    def device(self):
        raise NotImplementedError("Method `device` must be implemented by subclass.")

    @property
    def pipeline_parallel_enabled(self):
        raise NotImplementedError("Property `pipeline_parallel_enabled` must be implemented by subclass.")

    @property
    def data_parallel_enabled(self):
        raise NotImplementedError("Property `data_parallel_enabled` must be implemented by subclass.")

    @property
    def data_replication_enabled(self):
        raise NotImplementedError("Property `data_replication_enabled` must be implemented by subclass.")

    @property
    def data_sharding_enabled(self):
        raise NotImplementedError("Property `data_sharding_enabled` must be implemented by subclass.")

    @property
    def context_parallel_enabled(self):
        raise NotImplementedError("Property `context_parallel_enabled` must be implemented by subclass.")

    @property
    def tensor_parallel_enabled(self):
        raise NotImplementedError("Property `tensor_parallel_enabled` must be implemented by subclass.")


class BaseCheckpointer:
    r"""
    Base class that contains properties and methods that should be implemented by different parallel backends.
    """

    def __init__(
        self,
        dataloader: torch.utils.data.DataLoader,
        model_parts: List[torch.nn.Module],
        optimizers: Any,
        schedulers: Any,
        states: Dict[str, Any],
        checkpointing_steps: int,
        checkpointing_limit: int,
        output_dir: str,
        enable: bool = True,
        _callback_fn: Callable[[Dict[str, Any]], Dict[str, Any]] = None,
        _prefix: str = "finetrainers_step",
        *args,
        **kwargs,
    ) -> None:
        raise NotImplementedError("Method `__init__` must be implemented by subclass.")

    def save(self, step: int, force: bool, *, _device: Optional[torch.device] = None, _is_main_process: bool) -> str:
        raise NotImplementedError("Method `save` must be implemented by subclass.")

    def load(self, step: int = -1) -> bool:
        raise NotImplementedError("Method `load` must be implemented by subclass.")
