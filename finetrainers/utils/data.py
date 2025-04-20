from pathlib import Path
from typing import Any, Union

import torch

from finetrainers.constants import PRECOMPUTED_CONDITIONS_DIR_NAME, PRECOMPUTED_LATENTS_DIR_NAME
from finetrainers.logging import get_logger


logger = get_logger()


def should_perform_precomputation(precomputation_dir: Union[str, Path]) -> bool:
    if isinstance(precomputation_dir, str):
        precomputation_dir = Path(precomputation_dir)
    conditions_dir = precomputation_dir / PRECOMPUTED_CONDITIONS_DIR_NAME
    latents_dir = precomputation_dir / PRECOMPUTED_LATENTS_DIR_NAME
    if conditions_dir.exists() and latents_dir.exists():
        num_files_conditions = len(list(conditions_dir.glob("*.pt")))
        num_files_latents = len(list(latents_dir.glob("*.pt")))
        if num_files_conditions != num_files_latents:
            logger.warning(
                f"Number of precomputed conditions ({num_files_conditions}) does not match number of precomputed latents ({num_files_latents})."
                f"Cleaning up precomputed directories and re-running precomputation."
            )
            # clean up precomputed directories
            for file in conditions_dir.glob("*.pt"):
                file.unlink()
            for file in latents_dir.glob("*.pt"):
                file.unlink()
            return True
        if num_files_conditions > 0:
            logger.info(f"Found {num_files_conditions} precomputed conditions and latents.")
            return False
    logger.info("Precomputed data not found. Running precomputation.")
    return True


def determine_batch_size(x: Any) -> int:
    if isinstance(x, list):
        return len(x)
    if isinstance(x, torch.Tensor):
        return x.size(0)
    if isinstance(x, dict):
        for key in x:
            try:
                return determine_batch_size(x[key])
            except ValueError:
                pass
        return 1
    raise ValueError("Could not determine batch size from input.")
