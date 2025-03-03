import functools
import pathlib
import shutil
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

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
        _callback_fn: Callable[[Dict[str, Any]], Dict[str, Any]] = None,
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
        self._callback_fn = _callback_fn
        self._prefix = _prefix

        logger.info(f"Checkpointing enabled. Checkpoints will be stored in '{self.output_dir}'")

    def save(self, step: int = -1, force: bool = False, *, _device: torch.device, _is_main_process: bool) -> str:
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

        state_dicts = [
            gather_state_dict_on_cpu_rank0(model, _device, is_main_process=_is_main_process)
            for model in self.states["model"].model
        ]
        if self._callback_fn is not None:
            list(map(self._callback_fn, state_dicts))

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
        if self.checkpointing_limit is None or self.checkpointing_limit <= 0:
            return
        checkpoints = sorted(
            self.output_dir.glob(f"{self._prefix}_*"), key=lambda x: int(x.name.split("_")[-1]), reverse=True
        )
        for checkpoint in checkpoints[self.checkpointing_limit :]:
            logger.info(f"Deleting stale checkpoint: {checkpoint}")
            shutil.rmtree(checkpoint, ignore_errors=True)


def gather_state_dict_on_cpu_rank0(
    model, device: Optional[torch.device] = None, *, is_main_process: bool
) -> Dict[str, Any]:
    cpu_state_dict = {}
    sharded_sd = model.state_dict()
    for param_name, param in sharded_sd.items():
        if param.is_cpu:
            # Move back to device if offloaded to CPU
            param = param.to(device)
        if hasattr(param, "_local_tensor"):
            # Gather DTensor
            param = param.full_tensor()
        if is_main_process:
            cpu_state_dict[param_name] = param.cpu()
        torch.distributed.barrier()
    return cpu_state_dict


# # Copied from pytorch (torch/distributed/checkpoint/format_utils.py) to support callbacks to modify state_dict
# def dcp_to_torch_save(
#     dcp_checkpoint_dir: Union[str, os.PathLike],
#     torch_save_path: Union[str, os.PathLike],
#     callback_fn: Callable[[Dict[str, Any]], Dict[str, Any]] = None,
# ):
#     """
#     Given a directory containing a DCP checkpoint, this function will convert it into a
#     Torch save file.

#     Args:
#         dcp_checkpoint_dir: Directory containing the DCP checkpoint.
#         torch_save_path: Filename to store the converted Torch save file.
#         callback_fn: Optional callback function that takes the state_dict as input and returns a modified state_dict.

#     .. warning::
#         To avoid OOM, it's recommended to only run this function on a single rank.
#     """
#     state_dict = {}
#     _load_state_dict(
#         state_dict,
#         storage_reader=FileSystemReader(dcp_checkpoint_dir),
#         planner=_EmptyStateDictLoadPlanner(),
#         no_dist=True,
#     )
#     if callback_fn is not None:
#         state_dict = callback_fn(state_dict)
#     torch.save(state_dict, torch_save_path)
