import contextlib
import copy
import pathlib
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from .logging import get_logger
from .utils import Timer, TimerDevice


logger = get_logger()


class BaseTracker:
    r"""Base class for loggers. Does nothing by default, so it is useful when you want to disable logging."""

    def __init__(self):
        self._timed_metrics = {}

    @contextlib.contextmanager
    def timed(self, name: str, device: TimerDevice = TimerDevice.CPU, device_sync: bool = False):
        r"""Context manager to track time for a specific operation."""
        timer = Timer(name, device, device_sync)
        timer.start()
        yield timer
        timer.end()
        elapsed_time = timer.elapsed_time
        if name in self._timed_metrics:
            # If the timer name already exists, add the elapsed time to the existing value since a log has not been invoked yet
            self._timed_metrics[name] += elapsed_time
        else:
            self._timed_metrics[name] = elapsed_time

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        pass

    def finish(self) -> None:
        pass

    def get_wandb_run_id(self) -> Optional[str]:
        r"""Get the wandb run ID if available."""
        return None


class DummyTracker(BaseTracker):
    def __init__(self):
        super().__init__()

    def log(self, *args, **kwargs):
        pass

    def finish(self) -> None:
        pass


class WandbTracker(BaseTracker):
    r"""Logger implementation for Weights & Biases."""

    def __init__(self, experiment_name: str, log_dir: str, config: Optional[Dict[str, Any]] = None, resume_run_id: Optional[str] = None) -> None:
        super().__init__()

        import wandb

        self.wandb = wandb

        # WandB does not create a directory if it does not exist and instead starts using the system temp directory.
        pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)

        if resume_run_id is not None:
            logger.info(f"Resuming WandB run with ID: {resume_run_id}")
            self.run = wandb.init(project=experiment_name, dir=log_dir, config=config, id=resume_run_id, resume="must")
        else:
            self.run = wandb.init(project=experiment_name, dir=log_dir, config=config)
        logger.info("WandB logging enabled")

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        metrics = {**self._timed_metrics, **metrics}
        self.run.log(metrics, step=step)
        self._timed_metrics = {}

    def finish(self) -> None:
        self.run.finish()

    @property
    def run_id(self) -> Optional[str]:
        """Return the current wandb run ID for checkpointing purposes."""
        return self.run.id if self.run is not None else None

    def get_wandb_run_id(self) -> Optional[str]:
        """Return the wandb run ID if this tracker supports it."""
        return self.run_id


class SequentialTracker(BaseTracker):
    r"""Sequential tracker that logs to multiple trackers in sequence."""

    def __init__(self, trackers: List[BaseTracker]) -> None:
        super().__init__()
        self.trackers = trackers

    @contextlib.contextmanager
    def timed(self, name: str, device: TimerDevice = TimerDevice.CPU, device_sync: bool = False):
        r"""Context manager to track time for a specific operation."""
        timer = Timer(name, device, device_sync)
        timer.start()
        yield timer
        timer.end()
        elapsed_time = timer.elapsed_time
        if name in self._timed_metrics:
            # If the timer name already exists, add the elapsed time to the existing value since a log has not been invoked yet
            self._timed_metrics[name] += elapsed_time
        else:
            self._timed_metrics[name] = elapsed_time
        for tracker in self.trackers:
            tracker._timed_metrics = copy.deepcopy(self._timed_metrics)

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        for tracker in self.trackers:
            tracker.log(metrics, step)
        self._timed_metrics = {}

    def finish(self) -> None:
        for tracker in self.trackers:
            tracker.finish()

    def get_wandb_run_id(self) -> Optional[str]:
        """Return the wandb run ID from the first WandB tracker in the sequence."""
        for tracker in self.trackers:
            run_id = tracker.get_wandb_run_id()
            if run_id is not None:
                return run_id
        return None


class Trackers(str, Enum):
    r"""Enum for supported trackers."""

    NONE = "none"
    WANDB = "wandb"


_SUPPORTED_TRACKERS = [tracker.value for tracker in Trackers.__members__.values()]


def initialize_trackers(
    trackers: List[str], experiment_name: str, config: Dict[str, Any], log_dir: str, resume_run_id: Optional[str] = None
) -> Union[BaseTracker, SequentialTracker]:
    r"""Initialize loggers based on the provided configuration."""

    logger.info(f"Initializing trackers: {trackers}. Logging to {log_dir=}")

    if len(trackers) == 0:
        return BaseTracker()

    if any(tracker_name not in _SUPPORTED_TRACKERS for tracker_name in set(trackers)):
        raise ValueError(f"Unsupported tracker(s) provided. Supported trackers: {_SUPPORTED_TRACKERS}")

    tracker_instances = []
    for tracker_name in set(trackers):
        if tracker_name == Trackers.NONE:
            tracker = BaseTracker()
        elif tracker_name == Trackers.WANDB:
            tracker = WandbTracker(experiment_name, log_dir, config, resume_run_id=resume_run_id)
        tracker_instances.append(tracker)

    tracker = SequentialTracker(tracker_instances)
    return tracker


TrackerType = Union[BaseTracker, SequentialTracker, WandbTracker]
