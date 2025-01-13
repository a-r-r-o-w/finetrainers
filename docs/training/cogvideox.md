# CogVideoX

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
CAPTION_COLUMN="prompt.txt"
VIDEO_COLUMN="videos.txt"
OUTPUT_DIR="/path/to/models/cog/"
ID_TOKEN="BW_STYLE"

# Model arguments
model_cmd="--model_name cogvideox \
  --pretrained_model_name_or_path THUDM/CogVideoX-5b"

# Dataset arguments
dataset_cmd="--data_root $DATA_ROOT \
  --video_column $VIDEO_COLUMN \
  --caption_column $CAPTION_COLUMN \
  --id_token $ID_TOKEN \
  --video_resolution_buckets 49x480x720 \
  --caption_dropout_p 0.05"

# Dataloader arguments
dataloader_cmd="--dataloader_num_workers 4"

# Training arguments
training_cmd="--training_type lora \
  --seed 42 \
  --batch_size 1 \
  --precompute_conditions \
  --train_steps 1000 \
  --rank 128 \
  --lora_alpha 128 \
  --target_modules to_q to_k to_v to_out.0 \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing \
  --checkpointing_steps 200 \
  --checkpointing_limit 2 \
  --resume_from_checkpoint=latest \
  --enable_slicing \
  --enable_tiling"

# Optimizer arguments
optimizer_cmd="--optimizer adamw \
  --use_8bit_bnb \
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
miscellaneous_cmd="--tracker_name finetrainers-cog \
  --output_dir $OUTPUT_DIR \
  --nccl_timeout 1800 \
  --report_to wandb"

cmd="accelerate launch --config_file accelerate_configs/deepspeed.yaml --gpu_ids $GPU_IDS train.py \
  $model_cmd \
  $dataset_cmd \
  $dataloader_cmd \
  $training_cmd \
  $optimizer_cmd \
  $miscellaneous_cmd"

echo "Running command: $cmd"
eval $cmd
echo -ne "-------------------- Finished executing script --------------------\n\n"
```

## Memory Usage

### LoRA

<!-- TODO(aryan): Update these numbers for 49x512x768 -->

> [!NOTE]
>
> The below measurements are done in `torch.bfloat16` precision. Memory usage can further be reduce by passing `--layerwise_upcasting_modules transformer` to the training script. This will cast the model weights to `torch.float8_e4m3fn` or `torch.float8_e5m2`, which halves the memory requirement for model weights. Computation is performed in the dtype set by `--transformer_dtype` (which defaults to `bf16`).

LoRA with rank 128, batch size 1, gradient checkpointing, optimizer adamw, `49x480x720` resolutions, **with precomputation**:

```
Training configuration: {
    "trainable parameters": 132120576,
    "total samples": 69,
    "train epochs": 1,
    "train steps": 10,
    "batches per device": 1,
    "total batches observed per epoch": 69,
    "train batch size": 1,
    "gradient accumulation steps": 1
}
```

| stage                         | memory_allocated  | max_memory_reserved |
|:-----------------------------:|:-----------------:|:-------------------:|
| after precomputing conditions |  8.880            | 8.941               |
| after precomputing latents    |  9.300            | 12.441              |
| before training start         | 10.622            | 20.701              |
| after epoch 1                 | 11.145            | 20.701              |
| before validation start       | 11.145            | 20.702              |
| after validation end          | 11.145            | 28.324              |
| after training end            | 11.144            | 11.592              |

### Full finetuning

```
Training configuration: {
    "trainable parameters": 5570283072,
    "total samples": 1,
    "train epochs": 2,
    "train steps": 2,
    "batches per device": 1,
    "total batches observed per epoch": 1,
    "train batch size": 1,
    "gradient accumulation steps": 1
}
```

| stage                         | memory_allocated  | max_memory_reserved |
|:-----------------------------:|:-----------------:|:-------------------:|
| after precomputing conditions |  8.880            | 8.941               |
| after precomputing latents    |  9.300            | 12.441              |
| before training start         | 10.376            | 10.387              |
| after epoch 1                 | 31.160            | 52.939              |
| before validation start       | 31.161            | 52.939              |
| after validation end          | 31.161            | 52.939              |
| after training end            | 31.160            | 34.295              |

## Supported checkpoints

CogVideoX has multiple checkpoints as one can note [here](https://huggingface.co/collections/THUDM/cogvideo-66c08e62f1685a3ade464cce). The following checkpoints were tested with `finetrainers` and are known to be working:

* [THUDM/CogVideoX-2b](https://huggingface.co/THUDM/CogVideoX-2b)
* [THUDM/CogVideoX-5B](https://huggingface.co/THUDM/CogVideoX-5B)
* [THUDM/CogVideoX1.5-5B](https://huggingface.co/THUDM/CogVideoX1.5-5B)

## Inference

Assuming your LoRA is saved and pushed to the HF Hub, and named `my-awesome-name/my-awesome-lora`, we can now use the finetuned model for inference:

```diff
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16
).to("cuda")
+ pipe.load_lora_weights("my-awesome-name/my-awesome-lora", adapter_name="cogvideox-lora")
+ pipe.set_adapters(["cogvideox-lora"], [0.75])

video = pipe("<my-awesome-prompt>").frames[0]
export_to_video(video, "output.mp4")
```

You can refer to the following guides to know more about the model pipeline and performing LoRA inference in `diffusers`:

* [CogVideoX in Diffusers](https://huggingface.co/docs/diffusers/main/en/api/pipelines/cogvideox)
* [Load LoRAs for inference](https://huggingface.co/docs/diffusers/main/en/tutorials/using_peft_for_inference)
* [Merge LoRAs](https://huggingface.co/docs/diffusers/main/en/using-diffusers/merge_loras)