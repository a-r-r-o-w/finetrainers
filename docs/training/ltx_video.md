# LTX-Video

## Training

Provided you have a dataset:

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
OUTPUT_DIR="/path/to/models/ltx-video/"

ID_TOKEN="BW_STYLE"

# Model arguments
model_cmd="--model_name ltx_video \
  --pretrained_model_name_or_path Lightricks/LTX-Video"

# Dataset arguments
dataset_cmd="--data_root $DATA_ROOT \
  --video_column $VIDEO_COLUMN \
  --caption_column $CAPTION_COLUMN \
  --id_token $ID_TOKEN \
  --video_resolution_buckets 49x512x768 \
  --caption_dropout_p 0.05"

# Dataloader arguments
dataloader_cmd="--dataloader_num_workers 0"

# Diffusion arguments
diffusion_cmd="--flow_resolution_shifting"

# Training arguments
training_cmd="--training_type lora \
  --seed 42 \
  --mixed_precision bf16 \
  --batch_size 1 \
  --train_steps 1200 \
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
  --lr 3e-5 \
  --lr_scheduler constant_with_warmup \
  --lr_warmup_steps 100 \
  --lr_num_cycles 1 \
  --beta1 0.9 \
  --beta2 0.95 \
  --weight_decay 1e-4 \
  --epsilon 1e-8 \
  --max_grad_norm 1.0"

# Miscellaneous arguments
miscellaneous_cmd="--tracker_name finetrainers-ltxv \
  --output_dir $OUTPUT_DIR \
  --nccl_timeout 1800 \
  --report_to wandb"

cmd="accelerate launch --config_file accelerate_configs/uncompiled_2.yaml --gpu_ids $GPU_IDS train.py \
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

LoRA with rank 128, batch size 1, gradient checkpointing, optimizer adamw, `49x512x768` resolution, **without precomputation**:

```
Training configuration: {
    "trainable parameters": 117440512,
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
| before training start   | 13.486           | 13.879              |
| before validation start | 14.146           | 17.623              |
| after validation end    | 14.146           | 17.623              |
| after epoch 1           | 14.146           | 17.623              |
| after training end      | 4.461            | 17.623              |

Note: requires about `18` GB of VRAM without precomputation.

LoRA with rank 128, batch size 1, gradient checkpointing, optimizer adamw, `49x512x768` resolution, **with precomputation**:

```
Training configuration: {
    "trainable parameters": 117440512,
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
| after precomputing conditions | 8.88             | 8.920               |
| after precomputing latents    | 9.684            | 11.613              |
| before training start         | 3.809            | 10.010              |
| after epoch 1                 | 4.26             | 10.916              |
| before validation start       | 4.26             | 10.916              |
| after validation end          | 13.924           | 17.262              |
| after training end            | 4.26             | 14.314              |

Note: requires about `17.5` GB of VRAM with precomputation. If validation is not performed, the memory usage is reduced to `11` GB.

## Inference

Assuming your LoRA is saved and pushed to the HF Hub, and named `my-awesome-name/my-awesome-lora`, we can now use the finetuned model for inference:

```diff
import torch
from diffusers import LTXPipeline
from diffusers.utils import export_to_video

pipe = LTXPipeline.from_pretrained(
    "Lightricks/LTX-Video", torch_dtype=torch.bfloat16
).to("cuda")
+ pipe.load_lora_weights("my-awesome-name/my-awesome-lora", adapter_name="ltxv-lora")
+ pipe.set_adapters(["ltxv-lora"], [0.75])

video = pipe("<my-awesome-prompt>").frames[0]
export_to_video(video, "output.mp4", fps=8)
```

You can refer to the following guides to know more about performing LoRA inference in `diffusers`:

* [Load LoRAs for inference](https://huggingface.co/docs/diffusers/main/en/tutorials/using_peft_for_inference)
* [Merge LoRAs](https://huggingface.co/docs/diffusers/main/en/using-diffusers/merge_loras)