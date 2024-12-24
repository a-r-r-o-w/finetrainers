import os
from typing import Tuple
from accelerate.logging import get_logger
from ..constants import FINETRAINERS_LOG_LEVEL
from ..utils.file_utils import find_files, delete_files

logger = get_logger("finetrainers")
logger.setLevel(FINETRAINERS_LOG_LEVEL)

def get_latest_ckpt_path_to_resume_from(
    resume_from_checkpoint: str, num_update_steps_per_epoch: int, output_dir: str
) -> Tuple[str, int, int, int]:
    if not resume_from_checkpoint:
        initial_global_step = 0
        global_step = 0
        first_epoch = 0
        resume_from_checkpoint_path = None
    else:
        if resume_from_checkpoint != "latest":
            path = os.path.basename(resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            logger.info(
                f"Checkpoint '{resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            resume_from_checkpoint = None
            initial_global_step = 0
            resume_from_checkpoint_path = None
        else:
            logger.info(f"Resuming from checkpoint {path}")
            resume_from_checkpoint_path = os.path.join(output_dir, path)
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    return resume_from_checkpoint_path, initial_global_step, global_step, first_epoch


def get_intermediate_ckpt_path(checkpointing_limit: int, step: int, output_dir: str) -> str:
    # before saving state, check if this save would set us over the `checkpointing_limit`
    if checkpointing_limit is not None:
        checkpoints = find_files(output_dir, prefix="checkpoint")

        # before we save the new checkpoint, we need to have at_most `checkpoints_total_limit - 1` checkpoints
        if len(checkpoints) >= checkpointing_limit:
            num_to_remove = len(checkpoints) - checkpointing_limit + 1
            checkpoints_to_remove = checkpoints[0:num_to_remove]
            delete_files(checkpoints_to_remove)

    logger.info(f"Checkpointing at step {step}")
    save_path = os.path.join(output_dir, f"checkpoint-{step}")
    logger.info(f"Saving state to {save_path}")
    return save_path