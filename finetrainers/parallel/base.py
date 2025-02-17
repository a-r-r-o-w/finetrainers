from contextlib import contextmanager
from typing import Any, Dict, List

from ..trackers import TrackerType, initialize_trackers


class BaseParallelState:
    r"""
    Base class that contains properties and methods that should be implemented by different parallel backends.
    """

    def get_mesh(self, device_type: str = "cuda"):
        raise NotImplementedError("Method `get_mesh` must be implemented by subclass.")

    def initialize_trackers(
        self, trackers: List[str], experiment_name: str, config: Dict[str, Any], log_dir: str
    ) -> TrackerType:
        self.tracker = None
        if self.is_main_process:
            self.tracker = initialize_trackers(trackers, experiment_name, config, log_dir)

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
