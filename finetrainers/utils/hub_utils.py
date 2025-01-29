import os
from typing import List, Union

import numpy as np
import wandb
from diffusers.utils import export_to_video
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from PIL import Image


def save_model_card(
    args,
    repo_id: str,
    videos: Union[List[str], Union[List[Image.Image], List[np.ndarray]]],
    validation_prompts: List[str],
    fps: int = 30,
) -> None:
    widget_dict = []
    output_dir = str(args.output_dir)
    if videos is not None and len(videos) > 0:
        for i, (video, validation_prompt) in enumerate(zip(videos, validation_prompts)):
            if not isinstance(video, str):
                export_to_video(video, os.path.join(output_dir, f"final_video_{i}.mp4"), fps=fps)
            widget_dict.append(
                {
                    "text": validation_prompt if validation_prompt else " ",
                    "output": {"url": video if isinstance(video, str) else f"final_video_{i}.mp4"},
                }
            )

    training_type = "Full" if args.training_type == "full-finetune" else "LoRA"
    model_description = f"""
# {training_type} Finetune

<Gallery />

## Model description

This is a {training_type.lower()} finetune of model: `{args.pretrained_model_name_or_path}`.

The model was trained using [`finetrainers`](https://github.com/a-r-r-o-w/finetrainers).

`id_token` used: {args.id_token} (if it's not `None`, it should be used in the prompts.)

## Download model

[Download LoRA]({repo_id}/tree/main) in the Files & Versions tab.

## Usage

Requires the [🧨 Diffusers library](https://github.com/huggingface/diffusers) installed.

```py
TODO
```

For more details, including weighting, merging and fusing LoRAs, check the [documentation](https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading_adapters) on loading LoRAs in diffusers.
"""
    if wandb.run and wandb.run.url:
        model_description += f"""
Find out the wandb run URL and training configurations [here]({wandb.run.url}).
"""

    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        base_model=args.pretrained_model_name_or_path,
        model_description=model_description,
        widget=widget_dict,
    )
    tags = [
        "text-to-video",
        "diffusers-training",
        "diffusers",
        "finetrainers",
        "template:sd-lora",
    ]
    if training_type == "Full":
        tags.append("full-finetune")
    else:
        tags.append("lora")

    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(args.output_dir, "README.md"))
