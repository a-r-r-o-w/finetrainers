import argparse
import inspect
import os
import re

import torch
from accelerate.logging import get_logger
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video, load_image, load_video

from ..constants import FINETRAINERS_LOG_LEVEL
from ..utils.file_utils import string_to_filename


logger = get_logger("finetrainers")
logger.setLevel(FINETRAINERS_LOG_LEVEL)


def generate_artifacts(
    model_config: dict,
    pipeline: DiffusionPipeline,
    args: argparse.Namespace,
    generator: torch.Generator,
    step: int,
    num_processes: int,
    process_index: int,
    trackers: list,
    final_validation: bool = False,
) -> list:
    wandb_tracking = any("wandb" in tracker.name for tracker in trackers)
    if wandb_tracking:
        import wandb

    all_processes_artifacts = []
    num_validation_samples = len(args.validation_prompts)
    prompts_to_filenames = {}
    for i in range(num_validation_samples):
        # Skip current validation on all processes but one
        if i % num_processes != process_index:
            continue

        prompt = args.validation_prompts[i]
        image = args.validation_images[i]
        video = args.validation_videos[i]
        height = args.validation_heights[i]
        width = args.validation_widths[i]
        num_frames = args.validation_num_frames[i]

        if image is not None:
            image = load_image(image)
        if video is not None:
            video = load_video(video)

        logger.debug(
            f"Validating sample {i + 1}/{num_validation_samples} on process {process_index}. Prompt: {prompt}",
            main_process_only=False,
        )
        has_fps_in_call = any("fps" in p for p in inspect.signature(model_config["pipeline_cls"].__call__).parameters)
        validation_kwargs = {
            "pipeline": pipeline,
            "prompt": prompt,
            "image": image,
            "video": video,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "num_videos_per_prompt": args.num_validation_videos_per_prompt,
            "generator": generator,
        }
        if has_fps_in_call:
            validation_kwargs.update({"fps": args.fps})
        validation_artifacts = model_config["validation"](**validation_kwargs)

        prompt_filename = string_to_filename(prompt)[:25]
        artifacts = {
            "image": {"type": "image", "value": image},
            "video": {"type": "video", "value": video},
        }
        for i, (artifact_type, artifact_value) in enumerate(validation_artifacts):
            artifacts.update({f"artifact_{i}": {"type": artifact_type, "value": artifact_value}})
        logger.debug(
            f"Validation artifacts on process {process_index}: {list(artifacts.keys())}",
            main_process_only=False,
        )

        for key, value in list(artifacts.items()):
            artifact_type = value["type"]
            artifact_value = value["value"]
            if artifact_type not in ["image", "video"] or artifact_value is None:
                continue

            extension = "png" if artifact_type == "image" else "mp4"
            filename = "validation-" if not final_validation else "final-"
            filename += f"{step}-{process_index}-{prompt_filename}.{extension}"
            filename = os.path.join(args.output_dir, filename)

            if artifact_type == "image":
                logger.debug(f"Saving image to {filename}")
                artifact_value.save(filename)
                if wandb_tracking:
                    artifact_value = wandb.Image(filename)
            elif artifact_type == "video":
                logger.debug(f"Saving video to {filename}")
                export_to_video(artifact_value, filename, fps=args.fps)
                if wandb_tracking:
                    artifact_value = wandb.Video(filename, caption=prompt, fps=args.fps)

            all_processes_artifacts.append(artifact_value)
            # limit to first process only as this will go into the model card.
            if process_index == 0 and final_validation:
                prompts_to_filenames[prompt] = filename

    return all_processes_artifacts, prompts_to_filenames


def log_artifacts(artifacts: list, trackers: list, tracker_key: str, step: int) -> None:
    wandb_tracking = any("wandb" in tracker.name for tracker in trackers)
    if wandb_tracking:
        import wandb

    for tracker in trackers:
        if tracker.name == "wandb":
            image_artifacts = [artifact for artifact in artifacts if isinstance(artifact, wandb.Image)]
            video_artifacts = [artifact for artifact in artifacts if isinstance(artifact, wandb.Video)]
            tracker.log(
                {
                    tracker_key: {"images": image_artifacts, "videos": video_artifacts},
                },
                step=step,
            )
        else:
            logger.warning("No supported tracker found for which logging is available.")

    return


def get_latest_step_files(files):
    # Regex to extract step
    pattern = re.compile(r"validation-(\d+)-\d+-.+?\.mp4")

    latest_step = -1
    latest_files = []

    for file in files:
        match = pattern.match(file)
        if match:
            step = int(match.group(1))

            # Update the latest step and reset the list if a higher step is found
            if step > latest_step:
                latest_step = step
                latest_files = [file]
            elif step == latest_step:
                latest_files.append(file)

    return latest_files
