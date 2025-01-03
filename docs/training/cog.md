## Training

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
  --pretrained_model_name_or_path THUDM/CogVideoX-2b"

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
  --mixed_precision fp16 \
  --transformer_dtype fp16 \
  --text_encoder_dtype fp16 \
  --vae_dtype fp16 \
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

# Validation arguments
validation_cmd="--validation_prompts \"$ID_TOKEN A black and white animated scene unfolds with an anthropomorphic goat surrounded by musical notes and symbols, suggesting a playful environment. Mickey Mouse appears, leaning forward in curiosity as the goat remains still. The goat then engages with Mickey, who bends down to converse or react. The dynamics shift as Mickey grabs the goat, potentially in surprise or playfulness, amidst a minimalistic background. The scene captures the evolving relationship between the two characters in a whimsical, animated setting, emphasizing their interactions and emotions.@@@49x512x768:::A woman with long brown hair and light skin smiles at another woman with long blonde hair. The woman with brown hair wears a black jacket and has a small, barely noticeable mole on her right cheek. The camera angle is a close-up, focused on the woman with brown hair's face. The lighting is warm and natural, likely from the setting sun, casting a soft glow on the scene. The scene appears to be real-life footage@@@49x512x768\" \
  --num_validation_videos 1 \
  --validation_steps 100"

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
  $validation_cmd \
  $miscellaneous_cmd"

echo "Running command: $cmd"
eval $cmd
echo -ne "-------------------- Finished executing script --------------------\n\n"
```

## Inference

Assuming your LoRA is saved and pushed to the HF Hub, and named `my-awesome-name/my-awesome-lora`, we can now use the finetuned model for inference:

```diff
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

pipe = DiffusionPipeline.from_pretrained(
    "THUDM/CogVideoX-2b", torch_dtype=torch.float16
).to("cuda")
+ pipe.load_lora_weights("my-awesome-name/my-awesome-lora", adapter_name="ltxv-lora")
+ pipe.set_adapters(["ltxv-lora"], [0.75])

video = pipe("<my-awesome-prompt>").frames[0]
export_to_video(video, "output.mp4")
```

You can refer to the following guides to know more about performing LoRA inference in `diffusers`:

* [Load LoRAs for inference](https://huggingface.co/docs/diffusers/main/en/tutorials/using_peft_for_inference)
* [Merge LoRAs](https://huggingface.co/docs/diffusers/main/en/using-diffusers/merge_loras)