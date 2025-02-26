import functools
import pathlib
import shutil
import time
from typing import TYPE_CHECKING, Any, Dict, List, Union

import torch
import torch.distributed.checkpoint
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful

from ..logging import get_logger


if TYPE_CHECKING:
    from .. import optimizer


logger = get_logger()


class ModelWrapper(Stateful):
    def __init__(self, model: Union[torch.nn.Module, List[torch.nn.Module]]) -> None:
        self.model = [model] if isinstance(model, torch.nn.Module) else model

    def state_dict(self) -> Dict[str, Any]:
        return {k: v for sd in map(get_model_state_dict, self.model) for k, v in sd.items()}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        func = functools.partial(
            set_model_state_dict,
            model_state_dict=state_dict,
            options=StateDictOptions(strict=False),
        )
        list(map(func, self.model))


class PTDCheckpointManager:
    def __init__(
        self,
        dataloader: torch.utils.data.DataLoader,
        model_parts: List[torch.nn.Module],
        optimizers: "optimizer.OptimizerWrapper",
        schedulers: "optimizer.SchedulerWrapper",
        states: Dict[str, Any],
        checkpointing_steps: int,
        checkpointing_limit: int,
        output_dir: str,
        enable: bool = True,
        _prefix: str = "finetrainers_step",
    ) -> None:
        self.states = states
        self.states.update(
            {
                "model": ModelWrapper(model_parts),
                "optimizer": optimizers,
                "dataloader": dataloader,
            }
        )
        self.states.update(schedulers.get_lr_scheduler_state())

        self.checkpointing_steps = checkpointing_steps
        self.checkpointing_limit = checkpointing_limit
        self.output_dir = pathlib.Path(output_dir)
        self.enable = enable
        self._prefix = _prefix

        logger.info(f"Checkpointing enabled. Checkpoints will be stored in '{self.output_dir}'")

    def save(self, step: int = -1, force: bool = False) -> str:
        if not self._should_checkpoint(step, force):
            return None

        checkpoint_dir = self._get_checkpoint_dir(step)
        begin_time = time.monotonic()
        torch.distributed.checkpoint.save(self.states, checkpoint_id=checkpoint_dir.as_posix())
        end_time = time.monotonic()
        logger.info(
            f"Saved checkpoint in {end_time - begin_time:.2f} seconds at step {step}. Directory: {checkpoint_dir}"
        )

        self._purge_stale_checkpoints()
        return checkpoint_dir.as_posix()

    def load(self, step: int = -1) -> bool:
        if not self.enable:
            return False
        if not self.output_dir.exists():
            return False
        if step != -1 and not self._get_checkpoint_dir(step).exists():
            return False

        if step == -1:
            latest_checkpoint_dir = self._find_latest_checkpoint_dir()
            if latest_checkpoint_dir is None:
                return False
            step = int(latest_checkpoint_dir.name.split("_")[-1])

        checkpoint_dir = self._get_checkpoint_dir(step)
        logger.info(f"Loading checkpoint from '{checkpoint_dir}' at step {step}")

        # For step 0, optimizers/schedulers are not available as they are created during training after first step
        states = {"model": self.states["model"]} if step == 0 else self.states

        # See bug: https://github.com/pytorch/pytorch/pull/138575
        original_stateful_states = {k: v for k, v in states.items() if isinstance(v, Stateful)}
        begin_time = time.monotonic()
        torch.distributed.checkpoint.load(states, checkpoint_id=checkpoint_dir.as_posix())
        end_time = time.monotonic()
        logger.info(f"Loaded checkpoint in {end_time - begin_time:.2f} seconds.")

        # bugfix from above: restore the original stateful objects, whose states were already updated in-place by dcp.load()
        states.update(original_stateful_states)

        return True

    def _should_checkpoint(self, step: int, force: bool) -> bool:
        if not self.enable:
            return False

        if not force:
            if step % self.checkpointing_steps != 0:
                return False

        return True

    def _get_checkpoint_dir(self, step: int) -> pathlib.Path:
        return self.output_dir / f"{self._prefix}_{step}"

    def _find_latest_checkpoint_dir(self) -> Union[pathlib.Path, None]:
        checkpoints = sorted(self.output_dir.glob(f"{self._prefix}_*"), key=lambda x: int(x.name.split("_")[-1]))
        return checkpoints[-1] if len(checkpoints) > 0 else None

    def _purge_stale_checkpoints(self) -> None:
        if self.checkpointing_limit <= 0:
            return
        checkpoints = sorted(
            self.output_dir.glob(f"{self._prefix}_*"), key=lambda x: int(x.name.split("_")[-1]), reverse=True
        )
        for checkpoint in checkpoints[self.checkpointing_limit :]:
            logger.info(f"Deleting stale checkpoint: {checkpoint}")
            shutil.rmtree(checkpoint, ignore_errors=True)
