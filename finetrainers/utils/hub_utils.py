from diffusers.utils import export_to_video
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
import os
import wandb

def save_model_card(
    args,
    repo_id,
    videos,
    validation_prompts,
    fps=30,
):
    widget_dict = []
    output_dir = str(args.output_dir)
    if videos is not None and len(videos) > 0:
        for i, (video, validation_prompt) in enumerate(zip(videos, validation_prompts)):
            export_to_video(video, os.path.join(output_dir, f"final_video_{i}.mp4"), fps=fps)
            widget_dict.append(
                {
                    "text": validation_prompt if validation_prompt else " ",
                    "output": {"url": f"final_video_{i}.mp4"},
                }
            )

    model_description = f"""
# LoRA Finetune

<Gallery />

## Model description

This is a lora finetune of model: `{args.pretrained_model_name_or_path}`.

The model was trained using [`finetrainers`](https://github.com/a-r-r-o-w/finetrainers).

## Download model

[Download LoRA]({args.repo_id}/tree/main) in the Files & Versions tab.

## Usage

Requires the [🧨 Diffusers library](https://github.com/huggingface/diffusers) installed.

```py
TODO
```

For more details, including weighting, merging and fusing LoRAs, check the [documentation](https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading_adapters) on loading LoRAs in diffusers.
"""
    if wandb.run.url:
        model_description += f"""
Find out the wandb run URL and training configurations [here]({wandb.run.url}).
"""

    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="apache-2.0",
        base_model=args.pretrained_model_name_or_path,
        model_description=model_description,
        widget=widget_dict,
    )
    tags = [
        "text-to-video",
        "diffusers-training",
        "diffusers",
        "lora",
        "template:sd-lora",
    ]

    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(args.output_dir, "README.md"))