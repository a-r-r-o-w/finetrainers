#!/bin/bash
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0

GPU_IDS="0"

DATA_ROOT="videos_prepared"
MODEL="genmo/mochi-1-preview"
OUTPUT_PATH="mochi-lora"

cmd="CUDA_VISIBLE_DEVICES=$GPU_IDS python text_to_video_lora.py \
  --pretrained_model_name_or_path $MODEL \
  --cast_dit \
  --data_root $DATA_ROOT \
  --seed 42 \
  --output_dir $OUTPUT_PATH \
  --train_batch_size 1 \
  --dataloader_num_workers 4 \
  --pin_memory \
  --caption_dropout 0.1 \
  --max_train_steps 2000 \
  --gradient_checkpointing \
  --enable_slicing \
  --enable_tiling \
  --enable_model_cpu_offload \
  --optimizer adamw \
  --validation_prompt \"A black and white animated scene unfolds with an anthropomorphic goat surrounded by musical notes and symbols, suggesting a playful environment. Mickey Mouse appears, leaning forward in curiosity as the goat remains still. The goat then engages with Mickey, who bends down to converse or react. The dynamics shift as Mickey grabs the goat, potentially in surprise or playfulness, amidst a minimalistic background. The scene captures the evolving relationship between the two characters in a whimsical, animated setting, emphasizing their interactions and emotions\" \
  --validation_prompt_separator ::: \
  --num_validation_videos 1 \
  --validation_epochs 1 \
  --allow_tf32 \
  --report_to wandb \
  --push_to_hub"

echo "Running command: $cmd"
eval $cmd
echo -ne "-------------------- Finished executing script --------------------\n\n"