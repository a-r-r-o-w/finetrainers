import os
from typing import List, Union, Callable

import numpy as np
import wandb
from diffusers.utils import export_to_video
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from PIL import Image


# Define inference examples as template functions to allow customization
def get_ltx_inference(args, repo_id, validation_prompt):
    return f"""
import torch
from diffusers import LTXPipeline
from diffusers.utils import export_to_video

pipe = LTXPipeline.from_pretrained(
    "{args.pretrained_model_name_or_path}", torch_dtype=torch.bfloat16
).to("cuda")
pipe.load_lora_weights("{repo_id}", adapter_name="ltxv-lora")
pipe.set_adapters(["ltxv-lora"], [0.75])

video = pipe("{validation_prompt}")  # Custom prompt used here
export_to_video(video, "output.mp4", fps=8)
"""


def get_hunyuan_inference(args, repo_id, validation_prompt):
    return f"""
import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video

model_id = "hunyuanvideo-community/HunyuanVideo"
transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    model_id, subfolder="transformer", torch_dtype=torch.bfloat16
)
pipe = HunyuanVideoPipeline.from_pretrained("{args.pretrained_model_name_or_path}", transformer=transformer, torch_dtype=torch.float16)
pipe.load_lora_weights("{repo_id}", adapter_name="hunyuanvideo-lora")
pipe.set_adapters(["hunyuanvideo-lora"], [0.6])
pipe.vae.enable_tiling()
pipe.to("cuda")

output = pipe(
    prompt="{validation_prompt}",  # Custom prompt used here
    height=320,
    width=512,
    num_frames=61,
    num_inference_steps=30,
).frames[0]
export_to_video(output, "output.mp4", fps=15)
"""


def get_wan_inference(args, repo_id, validation_prompt):
    return f"""
import torch
from diffusers import WanPipeline
from diffusers.utils import export_to_video

pipe = WanPipeline.from_pretrained(
    "{args.pretrained_model_name_or_path}", torch_dtype=torch.bfloat16
).to("cuda")
pipe.load_lora_weights("{repo_id}", adapter_name="wan-lora")
pipe.set_adapters(["wan-lora"], [0.75])

video = pipe("{validation_prompt}")  # Custom prompt used here
export_to_video(video, "output.mp4", fps=8)
"""


def get_cog_video_inference(args, repo_id, validation_prompt):
    return f"""
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16
).to("cuda")
pipe.load_lora_weights("{repo_id}", adapter_name="cogvideox-lora")
pipe.set_adapters(["cogvideox-lora"], [0.75])

video = pipe("{validation_prompt}")  # Custom prompt used here
export_to_video(video, "output.mp4")
"""


def get_cogview4_inference(args, repo_id, validation_prompt):
    return f"""
import torch
from diffusers import CogView4Pipeline
from diffusers.utils import export_to_video

pipe = CogView4Pipeline.from_pretrained(
    "THUDM/CogView4-6B", torch_dtype=torch.bfloat16
).to("cuda")
pipe.load_lora_weights("{repo_id}", adapter_name="cogview4-lora")
pipe.set_adapters(["cogview4-lora"], [0.9])

video = pipe("{validation_prompt}")  # Custom prompt used here
export_to_video(video, "output.mp4")
"""


# Model path to inference example generator function mapping
MODEL_INFERENCE_MAP = {
    "Lightricks/LTX-Video": get_ltx_inference,
    "hunyuanvideo-community/HunyuanVideo": get_hunyuan_inference,
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers": get_wan_inference,
    "THUDM/CogVideoX-5b": get_cog_video_inference,
    "THUDM/CogView4-6B": get_cogview4_inference,
}


# get the appropriate inference example based on the model path
def get_inference_example(model_path, args, repo_id, validation_prompt):

    if model_path in MODEL_INFERENCE_MAP:
        return MODEL_INFERENCE_MAP[model_path](args, repo_id, validation_prompt)

    return ""


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

    # get the appropriate inference example based on the model path and parameters
    validation_prompt = validation_prompts[0] if validation_prompts else "my-awesome-prompt"
    inference_example = get_inference_example(
        args.pretrained_model_name_or_path,
        args,
        repo_id,
        validation_prompt
    )

    model_description = f"""
# LoRA Finetune

<Gallery />

## Model description

This is a lora finetune of model: `{args.pretrained_model_name_or_path}`.

The model was trained using [`finetrainers`](https://github.com/a-r-r-o-w/finetrainers).

## Download model

[Download LoRA]({repo_id}/tree/main) in the Files & Versions tab.

## Usage

Requires the [🧨 Diffusers library](https://github.com/huggingface/diffusers) installed.

```py
{inference_example}
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