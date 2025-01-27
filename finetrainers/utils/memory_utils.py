import gc
from typing import Any, Dict, Union

import torch
from accelerate.logging import get_logger


logger = get_logger("finetrainers")


def get_memory_statistics(precision: int = 3) -> Dict[str, Any]:
    memory_stats = {
        "memory_allocated": None,
        "memory_reserved": None,
        "max_memory_allocated": None,
        "max_memory_reserved": None,
    }

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        memory_stats.update(
            {
                "memory_allocated": torch.cuda.memory_allocated(device),
                "memory_reserved": torch.cuda.memory_reserved(device),
                "max_memory_allocated": torch.cuda.max_memory_allocated(device),
                "max_memory_reserved": torch.cuda.max_memory_reserved(device),
            }
        )

    elif torch.backends.mps.is_available():
        memory_stats["memory_allocated"] = torch.mps.current_allocated_memory()

    else:
        logger.warning("No CUDA, MPS, or ROCm device found. Memory statistics are not available.")

    return {
        key: (round(bytes_to_gigabytes(value), ndigits=precision) if value else None)
        for key, value in memory_stats.items()
    }


def bytes_to_gigabytes(x: int) -> float:
    if x is not None:
        return x / 1024**3


def free_memory() -> None:
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # TODO(aryan): handle non-cuda devices


def reset_memory_stats(device: torch.device):
    # TODO: handle for non-cuda devices
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    else:
        logger.warning("No CUDA, device found. Nothing to reset memory of.")


def synchronize_device(device: torch.device):
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    else:
        logger.warning("No CUDA, device found. Nothing to synchronize.")


def make_contiguous(x: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    if isinstance(x, torch.Tensor):
        return x.contiguous()
    elif isinstance(x, dict):
        return {k: make_contiguous(v) for k, v in x.items()}
    else:
        return x
