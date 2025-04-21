#!/bin/bash

set -e -x

# export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
# export TORCHDYNAMO_VERBOSE=1
export WANDB_MODE="offline"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export FINETRAINERS_LOG_LEVEL="DEBUG"

# Download the validation dataset
if [ ! -d "examples/training/sft/wan_i2v/3dgs_dissolve/validation_dataset" ]; then
  echo "Downloading validation dataset..."
  huggingface-cli download --repo-type dataset finetrainers/OpenVid-1k-split-validation --local-dir examples/training/sft/wan_i2v/3dgs_dissolve/validation_dataset
else
  echo "Validation dataset already exists. Skipping download."
fi

# Finetrainers supports multiple backends for distributed training. Select your favourite and benchmark the differences!
# BACKEND="accelerate"
BACKEND="ptd"

# In this setting, I'm using 1 GPU on a 4-GPU node for training
NUM_GPUS=1
CUDA_VISIBLE_DEVICES="3"

# Check the JSON files for the expected JSON format
TRAINING_DATASET_CONFIG="examples/training/sft/wan_i2v/3dgs_dissolve/training.json"
VALIDATION_DATASET_FILE="examples/training/sft/wan_i2v/3dgs_dissolve/validation.json"

# Depending on how many GPUs you have available, choose your degree of parallelism and technique!
DDP_1="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 1 --dp_shards 1 --cp_degree 1 --tp_degree 1"
DDP_2="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 2 --dp_shards 1 --cp_degree 1 --tp_degree 1"
DDP_4="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 4 --dp_shards 1 --cp_degree 1 --tp_degree 1"
FSDP_2="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 1 --dp_shards 2 --cp_degree 1 --tp_degree 1"
FSDP_4="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 1 --dp_shards 4 --cp_degree 1 --tp_degree 1"
HSDP_2_2="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 2 --dp_shards 2 --cp_degree 1 --tp_degree 1"

# Parallel arguments
parallel_cmd=(
  $DDP_1
)

# Model arguments
model_cmd=(
  --model_name "wan"
  --pretrained_model_name_or_path "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
  --compile_modules text_encoder image_encoder transformer vae
  --compile_scopes regional
)

# Dataset arguments
dataset_cmd=(
  --dataset_config $TRAINING_DATASET_CONFIG
  --dataset_shuffle_buffer_size 32
  --enable_precomputation
  --precomputation_items 50
  --precomputation_once
)

# Dataloader arguments
dataloader_cmd=(
  --dataloader_num_workers 0
)

# Diffusion arguments
diffusion_cmd=(
  --flow_weighting_scheme "logit_normal"
)

# Training arguments
# We target just the attention projections layers for LoRA training here.
# You can modify as you please and target any layer (regex is supported)
training_cmd=(
  --training_type "lora"
  --seed 42
  --batch_size 1
  --train_steps 1000
  --rank 16
  --lora_alpha 16
  --target_modules "blocks.*(to_q|to_k|to_v|to_out.0)"
  --gradient_accumulation_steps 1
  --gradient_checkpointing
  --checkpointing_steps 501
  --checkpointing_limit 2
  # --resume_from_checkpoint 3000
  --enable_slicing
  --enable_tiling
)

# Optimizer arguments
optimizer_cmd=(
  --optimizer "adamw"
  --lr 1e-4
  --lr_scheduler "constant_with_warmup"
  --lr_warmup_steps 100
  --lr_num_cycles 1
  --beta1 0.9
  --beta2 0.99
  --weight_decay 1e-4
  --epsilon 1e-8
  --max_grad_norm 1.0
)

# Validation arguments
validation_cmd=(
  --validation_dataset_file "$VALIDATION_DATASET_FILE"
  --validation_steps 101
)

# Miscellaneous arguments
miscellaneous_cmd=(
  --tracker_name "finetrainers-wan-i2v"
  --output_dir "/raid/aryan/wan-i2v"
  --init_timeout 600
  --nccl_timeout 600
  --report_to "wandb"
)

# Torch config arguments
torch_config_cmd=(
  --allow_tf32
  --float32_matmul_precision high
)

# Execute the training script
if [ "$BACKEND" == "accelerate" ]; then

  ACCELERATE_CONFIG_FILE=""
  if [ "$NUM_GPUS" == 1 ]; then
    ACCELERATE_CONFIG_FILE="accelerate_configs/uncompiled_1.yaml"
  elif [ "$NUM_GPUS" == 2 ]; then
    ACCELERATE_CONFIG_FILE="accelerate_configs/uncompiled_2.yaml"
  elif [ "$NUM_GPUS" == 4 ]; then
    ACCELERATE_CONFIG_FILE="accelerate_configs/uncompiled_4.yaml"
  elif [ "$NUM_GPUS" == 8 ]; then
    ACCELERATE_CONFIG_FILE="accelerate_configs/uncompiled_8.yaml"
  fi
  
  accelerate launch --config_file "$ACCELERATE_CONFIG_FILE" --gpu_ids $CUDA_VISIBLE_DEVICES train.py \
    "${parallel_cmd[@]}" \
    "${model_cmd[@]}" \
    "${dataset_cmd[@]}" \
    "${dataloader_cmd[@]}" \
    "${diffusion_cmd[@]}" \
    "${training_cmd[@]}" \
    "${optimizer_cmd[@]}" \
    "${validation_cmd[@]}" \
    "${miscellaneous_cmd[@]}" \
    "${torch_config_cmd[@]}"

elif [ "$BACKEND" == "ptd" ]; then

  export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
  
  torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    --rdzv_backend c10d \
    --rdzv_endpoint="localhost:0" \
    train.py \
      "${parallel_cmd[@]}" \
      "${model_cmd[@]}" \
      "${dataset_cmd[@]}" \
      "${dataloader_cmd[@]}" \
      "${diffusion_cmd[@]}" \
      "${training_cmd[@]}" \
      "${optimizer_cmd[@]}" \
      "${validation_cmd[@]}" \
      "${miscellaneous_cmd[@]}" \
      "${torch_config_cmd[@]}"
fi

echo -ne "-------------------- Finished executing script --------------------\n\n"
