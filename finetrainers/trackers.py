import pathlib
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from .logging import get_logger


logger = get_logger()


class BaseTracker:
    r"""Base class for loggers. Does nothing by default, so it is useful when you want to disable logging."""

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        pass

    def finish(self) -> None:
        pass


class WandbTracker(BaseTracker):
    r"""Logger implementation for Weights & Biases."""

    def __init__(self, experiment_name: str, log_dir: str, config: Optional[Dict[str, Any]] = None) -> None:
        import wandb

        self.wandb = wandb

        # WandB does not create a directory if it does not exist and instead starts using the system temp directory.
        pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)

        self.run = wandb.init(project=experiment_name, dir=log_dir, config=config)
        logger.info("WandB logging enabled")

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        self.run.log(metrics, step=step)

    def finish(self) -> None:
        self.run.finish()


class SequentialTracker(BaseTracker):
    r"""Sequential tracker that logs to multiple trackers in sequence."""

    def __init__(self, trackers: List[BaseTracker]) -> None:
        self.trackers = trackers

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        for tracker in self.trackers:
            tracker.log(metrics, step)

    def finish(self) -> None:
        for tracker in self.trackers:
            tracker.finish()


class Trackers(str, Enum):
    r"""Enum for supported trackers."""

    NONE = "none"
    WANDB = "wandb"


_SUPPORTED_TRACKERS = [tracker.value for tracker in Trackers.__members__.values()]


def initialize_trackers(
    trackers: List[str], experiment_name: str, config: Dict[str, Any], log_dir: str
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
            tracker = WandbTracker(experiment_name, log_dir, config)
        tracker_instances.append(tracker)

    tracker = SequentialTracker(tracker_instances)
    return tracker


TrackerType = Union[BaseTracker, SequentialTracker, WandbTracker]
