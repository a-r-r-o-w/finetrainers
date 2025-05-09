import os


ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}

FINETRAINERS_LOG_LEVEL = os.environ.get("FINETRAINERS_LOG_LEVEL", "INFO")
FINETRAINERS_ATTN_PROVIDER = os.environ.get("FINETRAINERS_ATTN_PROVIDER", "native")
FINETRAINERS_ATTN_CHECKS = os.getenv("FINETRAINERS_ATTN_CHECKS", "0") in ENV_VARS_TRUE_VALUES
FINETRAINERS_ENABLE_TIMING = os.getenv("FINETRAINERS_ENABLE_TIMING", "1") in ENV_VARS_TRUE_VALUES

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

_COMMON_BEGINNING_PHRASES = (
    "This video",
    "The video",
    "This clip",
    "The clip",
    "The animation",
    "This image",
    "The image",
    "This picture",
    "The picture",
)
_COMMON_CONTINUATION_WORDS = ("shows", "depicts", "features", "captures", "highlights", "introduces", "presents")

COMMON_LLM_START_PHRASES = (
    "In the video,",
    "In this video,",
    "In this video clip,",
    "In the clip,",
    "Caption:",
    *(
        f"{beginning} {continuation}"
        for beginning in _COMMON_BEGINNING_PHRASES
        for continuation in _COMMON_CONTINUATION_WORDS
    ),
)

SUPPORTED_IMAGE_FILE_EXTENSIONS = ("jpg", "jpeg", "png")
SUPPORTED_VIDEO_FILE_EXTENSIONS = ("mp4", "mov")
