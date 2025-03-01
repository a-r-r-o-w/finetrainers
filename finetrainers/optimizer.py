import functools
import math
from typing import Any, Callable, Dict, List, Optional, Type, Union

import torch
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful

from .parallel import ParallelBackendEnum
from .utils.import_utils import is_bitsandbytes_available


class OptimizerWrapper(Stateful):
    r"""
    Optimizer wrapper that:
        - allows step/zero_grad on multiple optimizers needed for virtual pipeline stages
        - saves/loading optimizer state_dict at checkpoint
    """

    def __init__(
        self,
        model_parts: List[torch.nn.Module],
        optimizer_cls: Type[torch.optim.Optimizer],
        optimizer_kwargs: Dict[str, Any],
    ) -> None:
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs

        self.optimizers = []
        self.model_parts = model_parts

        for model in self.model_parts:
            optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)
            self.optimizers.append(optimizer)

    def step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()

    def zero_grad(self) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def state_dict(self) -> Dict[str, Any]:
        func = functools.partial(
            get_optimizer_state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        return {k: v for sd in map(func, self.model_parts, self.optimizers) for k, v in sd.items()}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        func = functools.partial(
            set_optimizer_state_dict,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        list(map(func, self.model_parts, self.optimizers))


class SchedulerWrapper:
    def __init__(
        self, optimizers, scheduler_lambda_fn: Type[torch.optim.lr_scheduler.LRScheduler], last_epoch: int
    ) -> None:
        self.schedulers = []
        for optimizer in optimizers:
            self.schedulers.append(torch.optim.lr_scheduler.LambdaLR(optimizer, scheduler_lambda_fn, last_epoch))

    def step(self) -> None:
        for scheduler in self.schedulers:
            scheduler.step()

    def get_last_lr(self) -> List[float]:
        # TODO(aryan): look into this later. Currently calling it leads to NCCL hang?????
        return {f"lr_{idx}": scheduler.get_last_lr() for idx, scheduler in enumerate(self.schedulers)}

    def get_lr_scheduler_state(self) -> Dict[str, Any]:
        state_dict = {}
        if len(self.schedulers) == 1:
            state_dict["lr_scheduler"] = self.schedulers[0]
        else:
            # For now, pipeline-parallel with looped schedules does not support resharding for lr_scheduler.
            # It should only support saving and loading a distributed checkpoint with the same number of pp ranks
            for idx, lr_scheduler in enumerate(self.schedulers):
                state_dict[f"lr_scheduler_{idx}"] = lr_scheduler
        return state_dict


def get_optimizer(
    parallel_backend: ParallelBackendEnum,
    name: str,
    model_parts: List[torch.nn.Module],
    learning_rate: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.95,
    beta3: float = 0.999,
    epsilon: float = 1e-8,
    weight_decay: float = 1e-4,
    fused: bool = False,
) -> Union[torch.optim.Optimizer, OptimizerWrapper]:
    name = name.lower()

    _raise_errors_if_packages_not_available(name)

    if name == "adam":
        optimizer_cls = torch.optim.Adam
        optimizer_kwargs = {
            "lr": learning_rate,
            "betas": (beta1, beta2),
            "eps": epsilon,
            "weight_decay": weight_decay,
            "fused": fused,
        }
    elif name == "adamw":
        optimizer_cls = torch.optim.AdamW
        optimizer_kwargs = {
            "lr": learning_rate,
            "betas": (beta1, beta2),
            "eps": epsilon,
            "weight_decay": weight_decay,
            "fused": fused,
        }
    elif name == "adam-bnb":
        from bitsandbytes.optim import Adam

        optimizer_cls = Adam
        optimizer_kwargs = {
            "lr": learning_rate,
            "betas": (beta1, beta2),
            "eps": epsilon,
            "weight_decay": weight_decay,
        }
    elif name == "adamw-bnb":
        from bitsandbytes.optim import AdamW

        optimizer_cls = AdamW
        optimizer_kwargs = {
            "lr": learning_rate,
            "betas": (beta1, beta2),
            "eps": epsilon,
            "weight_decay": weight_decay,
        }
    elif name == "adam-bnb-8bit":
        from bitsandbytes.optim import Adam8bit

        optimizer_cls = Adam8bit
        optimizer_kwargs = {
            "lr": learning_rate,
            "betas": (beta1, beta2),
            "eps": epsilon,
            "weight_decay": weight_decay,
        }
    elif name == "adamw-bnb-8bit":
        from bitsandbytes.optim import AdamW8bit

        optimizer_cls = AdamW8bit
        optimizer_kwargs = {
            "lr": learning_rate,
            "betas": (beta1, beta2),
            "eps": epsilon,
            "weight_decay": weight_decay,
        }

    # TODO(aryan): handle bitsandbytes and torchao
    else:
        raise ValueError(f"Unsupported optimizer: {name}")

    if parallel_backend == ParallelBackendEnum.ACCELERATE:
        return get_optimizer_accelerate(model_parts, optimizer_cls, optimizer_kwargs)
    elif parallel_backend == ParallelBackendEnum.PTD:
        return get_optimizer_ptd(model_parts, optimizer_cls, optimizer_kwargs)


def get_optimizer_accelerate(
    model_parts: List[torch.nn.Module], optimizer_cls: Type[torch.optim.Optimizer], optimizer_kwargs: Dict[str, Any]
) -> torch.optim.Optimizer:
    params = [param for model in model_parts for param in model.parameters() if param.requires_grad]
    optimizer = optimizer_cls(params, **optimizer_kwargs)
    return optimizer


def get_optimizer_ptd(
    model_parts: List[torch.nn.Module], optimizer_cls: Type[torch.optim.Optimizer], optimizer_kwargs: Dict[str, Any]
) -> OptimizerWrapper:
    return OptimizerWrapper(model_parts, optimizer_cls, optimizer_kwargs)


def get_lr_scheduler(
    parallel_backend: ParallelBackendEnum,
    name: str,
    optimizer: Union[torch.optim.Optimizer, OptimizerWrapper],
    step_rules: Optional[str] = None,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    num_cycles: int = 1,
    power: float = 1.0,
    lr_init: float = 1e-3,
    lr_end: float = 1e-7,
    last_epoch: int = -1,
) -> Union[torch.optim.lr_scheduler.LambdaLR, SchedulerWrapper]:
    name = name.lower()
    if name == "constant":
        scheduler_lambda_fn = get_constant_schedule()
    elif name == "constant_with_warmup":
        scheduler_lambda_fn = get_constant_schedule_with_warmup(num_warmup_steps)
    elif name == "piecewise_constant":
        scheduler_lambda_fn = get_piecewise_constant_schedule(step_rules)
    elif name == "linear":
        scheduler_lambda_fn = get_linear_schedule_with_warmup(num_warmup_steps, num_training_steps)
    elif name == "cosine":
        scheduler_lambda_fn = get_cosine_schedule_with_warmup(num_warmup_steps, num_training_steps, num_cycles)
    elif name == "cosine_with_restarts":
        scheduler_lambda_fn = get_cosine_with_hard_restarts_schedule_with_warmup(
            num_warmup_steps, num_training_steps, num_cycles
        )
    elif name == "polynomial":
        scheduler_lambda_fn = get_polynomial_decay_schedule_with_warmup(
            num_warmup_steps, num_training_steps, lr_init, lr_end, power
        )
    else:
        raise ValueError(f"Unsupported scheduler: {name}")

    if parallel_backend == ParallelBackendEnum.ACCELERATE:
        return get_lr_scheduler_accelerate(optimizer, scheduler_lambda_fn, last_epoch)
    elif parallel_backend == ParallelBackendEnum.PTD:
        return get_lr_scheduler_ptd(optimizer, scheduler_lambda_fn, last_epoch)


def get_lr_scheduler_accelerate(
    optimizer: torch.optim.Optimizer,
    scheduler_lambda_fn: Type[torch.optim.lr_scheduler.LRScheduler],
    last_epoch: int = -1,
) -> torch.optim.lr_scheduler.LambdaLR:
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scheduler_lambda_fn, last_epoch)
    return scheduler


def get_lr_scheduler_ptd(
    optimizer: OptimizerWrapper, scheduler_lambda_fn: Type[torch.optim.lr_scheduler.LRScheduler], last_epoch: int = -1
) -> SchedulerWrapper:
    return SchedulerWrapper(optimizer.optimizers, scheduler_lambda_fn, last_epoch)


# ==============================
# Adapted from https://github.com/huggingface/diffusers/blob/196aef5a6f76e1ad6ba889184860c3633d166910/src/diffusers/optimization.py
# ==============================


def get_constant_schedule() -> Callable[[int], float]:
    r"""
    Create a schedule with a constant learning rate, using the learning rate set in optimizer.
    """

    def lr_lambda(current_step: int):
        return 1.0

    return lr_lambda


def get_constant_schedule_with_warmup(num_warmup_steps: int) -> Callable[[int], float]:
    r"""
    Create a schedule with a constant learning rate preceded by a warmup period during which the learning rate
    increases linearly between 0 and the initial lr set in the optimizer.

    Args:
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    return lr_lambda


def get_piecewise_constant_schedule(step_rules: str) -> Callable[[int], float]:
    r"""
    Create a schedule with a constant learning rate, using the learning rate set in optimizer.

    Args:
        step_rules (`string`):
            The rules for the learning rate. ex: rule_steps="1:10,0.1:20,0.01:30,0.005" it means that the learning rate
            if multiple 1 for the first 10 steps, multiple 0.1 for the next 20 steps, multiple 0.01 for the next 30
            steps and multiple 0.005 for the other steps.
    """

    rules_dict = {}
    rule_list = step_rules.split(",")
    for rule_str in rule_list[:-1]:
        value_str, steps_str = rule_str.split(":")
        steps = int(steps_str)
        value = float(value_str)
        rules_dict[steps] = value
    last_lr_multiple = float(rule_list[-1])

    def create_rules_function(rules_dict, last_lr_multiple):
        def rule_func(steps: int) -> float:
            sorted_steps = sorted(rules_dict.keys())
            for i, sorted_step in enumerate(sorted_steps):
                if steps < sorted_step:
                    return rules_dict[sorted_steps[i]]
            return last_lr_multiple

        return rule_func

    rules_func = create_rules_function(rules_dict, last_lr_multiple)
    return rules_func


def get_linear_schedule_with_warmup(num_warmup_steps: int, num_training_steps: int) -> Callable[[int], float]:
    r"""
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return lr_lambda


def get_cosine_schedule_with_warmup(
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
) -> Callable[[int], float]:
    r"""
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_periods (`float`, *optional*, defaults to 0.5):
            The number of periods of the cosine function in a schedule (the default is to just decrease from the max
            value to 0 following a half-cosine).
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return lr_lambda


def get_cosine_with_hard_restarts_schedule_with_warmup(
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int = 1,
) -> Callable[[int], float]:
    r"""
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, with several hard restarts, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer.

    Args:
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`int`, *optional*, defaults to 1):
            The number of hard restarts to use.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))

    return lr_lambda


def get_polynomial_decay_schedule_with_warmup(
    num_warmup_steps: int,
    num_training_steps: int,
    lr_init: float,
    lr_end: float = 1e-7,
    power: float = 1.0,
) -> Callable[[int], float]:
    r"""
    Create a schedule with a learning rate that decreases as a polynomial decay from the initial lr set in the
    optimizer to end lr defined by *lr_end*, after a warmup period during which it increases linearly from 0 to the
    initial lr set in the optimizer.

    Args:
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        lr_end (`float`, *optional*, defaults to 1e-7):
            The end LR.
        power (`float`, *optional*, defaults to 1.0):
            Power factor.

    Note: *power* defaults to 1.0 as in the fairseq implementation, which in turn is based on the original BERT implementation at
    https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/optimization.py#L37
    """

    if not (lr_init > lr_end):
        raise ValueError(f"lr_end ({lr_end}) must be smaller than initial lr ({lr_init})")

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step > num_training_steps:
            return lr_end / lr_init  # as LambdaLR multiplies by lr_init
        else:
            lr_range = lr_init - lr_end
            decay_steps = num_training_steps - num_warmup_steps
            pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
            decay = lr_range * pct_remaining**power + lr_end
            return decay / lr_init  # as LambdaLR multiplies by lr_init

    return lr_lambda


def _raise_errors_if_packages_not_available(name: str) -> None:
    name_split = name.split("-")
    if len(name_split) < 2:
        return
    package_name = name_split[1]
    if package_name == "bnb":
        if not is_bitsandbytes_available():
            raise ImportError(
                f"Please install bitsandbytes by running `pip install bitsandbytes` to use the {name} optimizer."
            )
