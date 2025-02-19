import io
from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch
import torch.distributed.checkpoint.stateful

from .parallel import ParallelBackendType
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


@dataclass
class State:
    # Parallel state
    parallel_backend: ParallelBackendType = None

    # Training state
    train_state: TrainState = None
    num_trainable_parameters: int = 0
    generator: torch.Generator = None

    # Hub state
    repo_id: str = None

    # Artifacts state
    output_dir: str = None
