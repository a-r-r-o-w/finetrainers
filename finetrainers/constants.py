import os


DEFAULT_HEIGHT_BUCKETS = [256, 320, 384, 480, 512, 576, 720, 768, 960, 1024, 1280, 1536]
DEFAULT_WIDTH_BUCKETS = [256, 320, 384, 480, 512, 576, 720, 768, 960, 1024, 1280, 1536]
DEFAULT_FRAME_BUCKETS = [49]

DEFAULT_IMAGE_RESOLUTION_BUCKETS = []
for height in DEFAULT_HEIGHT_BUCKETS:
    for width in DEFAULT_WIDTH_BUCKETS:
        DEFAULT_IMAGE_RESOLUTION_BUCKETS.append((height, width))

DEFAULT_VIDEO_RESOLUTION_BUCKETS = []
for frames in DEFAULT_FRAME_BUCKETS:
    for height in DEFAULT_HEIGHT_BUCKETS:
        for width in DEFAULT_WIDTH_BUCKETS:
            DEFAULT_VIDEO_RESOLUTION_BUCKETS.append((frames, height, width))


FINETRAINERS_LOG_LEVEL = os.environ.get("FINETRAINERS_LOG_LEVEL", "INFO")

PRECOMPUTED_DIR_NAME = "precomputed"
PRECOMPUTED_CONDITIONS_DIR_NAME = "conditions"
PRECOMPUTED_LATENTS_DIR_NAME = "latents"

MODEL_DESCRIPTION = r"""
\# {model_id} {training_type} finetune

<Gallery />

\#\# Model Description

This model is a {training_type} of the `{model_id}` model.

This model was trained using the `fine-video-trainers` library - a repository containing memory-optimized scripts for training video models with [Diffusers](https://github.com/huggingface/diffusers).

\#\# Download model

[Download LoRA]({repo_id}/tree/main) in the Files & Versions tab.

\#\# Usage

Requires [🧨 Diffusers](https://github.com/huggingface/diffusers) installed.

```python
{model_example}
```

For more details, including weighting, merging and fusing LoRAs, check the [documentation](https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading_adapters) on loading LoRAs in diffusers.

\#\# License

Please adhere to the license of the base model.
""".strip()
