from pathlib import Path
from typing import Union

from accelerate.logging import get_logger

from ..constants import PRECOMPUTED_DIR_NAME, PRECOMPUTED_CONDITIONS_DIR_NAME, PRECOMPUTED_LATENTS_DIR_NAME


logger = get_logger("finetrainers")


def should_perform_precomputation(data_root: Union[str, Path]) -> bool:
    if isinstance(data_root, str):
        data_root = Path(data_root)
    conditions_dir = data_root / PRECOMPUTED_DIR_NAME / PRECOMPUTED_CONDITIONS_DIR_NAME
    latents_dir = data_root / PRECOMPUTED_DIR_NAME / PRECOMPUTED_LATENTS_DIR_NAME
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
