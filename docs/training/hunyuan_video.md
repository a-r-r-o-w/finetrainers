# HunyuanVideo

## Training

For LoRA training, specify `--training_type lora`. For full finetuning, specify `--training_type full-finetune`.

```bash
#!/bin/bash

export WANDB_MODE="offline"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export FINETRAINERS_LOG_LEVEL=DEBUG

GPU_IDS="0,1"

DATA_ROOT="/path/to/dataset"
CAPTION_COLUMN="prompts.txt"
VIDEO_COLUMN="videos.txt"
OUTPUT_DIR="/path/to/models/hunyuan-video/"

ID_TOKEN="afkx"

# Model arguments
model_cmd="--model_name hunyuan_video \
  --pretrained_model_name_or_path hunyuanvideo-community/HunyuanVideo"

# Dataset arguments
dataset_cmd="--data_root $DATA_ROOT \
  --video_column $VIDEO_COLUMN \
  --caption_column $CAPTION_COLUMN \
  --id_token $ID_TOKEN \
  --video_resolution_buckets 17x512x768 49x512x768 61x512x768 \
  --caption_dropout_p 0.05"

# Dataloader arguments
dataloader_cmd="--dataloader_num_workers 0"

# Diffusion arguments
diffusion_cmd=""

# Training arguments
training_cmd="--training_type lora \
  --seed 42 \
  --mixed_precision bf16 \
  --batch_size 1 \
  --train_steps 500 \
  --rank 128 \
  --lora_alpha 128 \
  --target_modules to_q to_k to_v to_out.0 \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing \
  --checkpointing_steps 500 \
  --checkpointing_limit 2 \
  --enable_slicing \
  --enable_tiling"

# Optimizer arguments
optimizer_cmd="--optimizer adamw \
  --lr 2e-5 \
  --lr_scheduler constant_with_warmup \
  --lr_warmup_steps 100 \
  --lr_num_cycles 1 \
  --beta1 0.9 \
  --beta2 0.95 \
  --weight_decay 1e-4 \
  --epsilon 1e-8 \
  --max_grad_norm 1.0"

# Miscellaneous arguments
miscellaneous_cmd="--tracker_name finetrainers-hunyuan-video \
  --output_dir $OUTPUT_DIR \
  --nccl_timeout 1800 \
  --report_to wandb"

cmd="accelerate launch --config_file accelerate_configs/uncompiled_8.yaml --gpu_ids $GPU_IDS train.py \
  $model_cmd \
  $dataset_cmd \
  $dataloader_cmd \
  $diffusion_cmd \
  $training_cmd \
  $optimizer_cmd \
  $miscellaneous_cmd"

echo "Running command: $cmd"
eval $cmd
echo -ne "-------------------- Finished executing script --------------------\n\n"
```

## Memory Usage

### LoRA

LoRA with rank 128, batch size 1, gradient checkpointing, optimizer adamw, `49x512x768` resolutions, **without precomputation**:

```
Training configuration: {
    "trainable parameters": 163577856,
    "total samples": 69,
    "train epochs": 1,
    "train steps": 10,
    "batches per device": 1,
    "total batches observed per epoch": 69,
    "train batch size": 1,
    "gradient accumulation steps": 1
}
```

| stage                   | memory_allocated | max_memory_reserved |
|:-----------------------:|:----------------:|:-------------------:|
| before training start   | 38.889           | 39.020              |
| before validation start | 39.747           | 56.266              |
| after validation end    | 39.748           | 58.385              |
| after epoch 1           | 39.748           | 40.910              |
| after training end      | 25.288           | 40.910              |

Note: requires about `59` GB of VRAM when validation is performed.

LoRA with rank 128, batch size 1, gradient checkpointing, optimizer adamw, `49x512x768` resolutions, **with precomputation**:

```
Training configuration: {
    "trainable parameters": 163577856,
    "total samples": 1,
    "train epochs": 10,
    "train steps": 10,
    "batches per device": 1,
    "total batches observed per epoch": 1,
    "train batch size": 1,
    "gradient accumulation steps": 1
}
```

| stage                         | memory_allocated | max_memory_reserved |
|:-----------------------------:|:----------------:|:-------------------:|
| after precomputing conditions | 14.232           | 14.461              |
| after precomputing latents    | 14.717           | 17.244              |
| before training start         | 24.195           | 26.039              |
| after epoch 1                 | 24.83            | 42.387              |
| before validation start       | 24.842           | 42.387              |
| after validation end          | 39.558           | 46.947              |
| after training end            | 24.842           | 41.039              |

Note: requires about `47` GB of VRAM with validation. If validation is not performed, the memory usage is reduced to about `42` GB.

### Full finetuning

Current, full finetuning is not supported for HunyuanVideo. It goes out of memory (OOM) for `49x512x768` resolutions.

## Inference

Assuming your LoRA is saved and pushed to the HF Hub, and named `my-awesome-name/my-awesome-lora`, we can now use the finetuned model for inference:

```py
import torch
from diffusers import HunyuanVideoPipeline

import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video

model_id = "hunyuanvideo-community/HunyuanVideo"
transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    model_id, subfolder="transformer", torch_dtype=torch.bfloat16
)
pipe = HunyuanVideoPipeline.from_pretrained(model_id, transformer=transformer, torch_dtype=torch.float16)
pipe.load_lora_weights("my-awesome-name/my-awesome-lora", adapter_name="hunyuanvideo-lora")
pipe.set_adapters(["hunyuanvideo-lora"], [0.6])
pipe.vae.enable_tiling()
pipe.to("cuda")

output = pipe(
    prompt="A cat walks on the grass, realistic",
    height=320,
    width=512,
    num_frames=61,
    num_inference_steps=30,
).frames[0]
export_to_video(output, "output.mp4", fps=15)
```

You can refer to the following guides to know more about the model pipeline and performing LoRA inference in `diffusers`:

* [Hunyuan-Video in Diffusers](https://huggingface.co/docs/diffusers/main/api/pipelines/hunyuan_video)
* [Load LoRAs for inference](https://huggingface.co/docs/diffusers/main/en/tutorials/using_peft_for_inference)
* [Merge LoRAs](https://huggingface.co/docs/diffusers/main/en/using-diffusers/merge_loras)