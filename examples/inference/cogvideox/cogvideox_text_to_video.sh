#!/bin/bash

set -e -x

# export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
# export TORCHDYNAMO_VERBOSE=1
# export WANDB_MODE="offline"
export WANDB_MODE="disabled"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export FINETRAINERS_LOG_LEVEL="DEBUG"

# Download the validation dataset
if [ ! -d "examples/inference/datasets/openvid-1k-split-validation" ]; then
  echo "Downloading validation dataset..."
  huggingface-cli download --repo-type dataset finetrainers/OpenVid-1k-split-validation --local-dir examples/inference/datasets/openvid-1k-split-validation
else
  echo "Validation dataset already exists. Skipping download."
fi

BACKEND="ptd"

NUM_GPUS=2
CUDA_VISIBLE_DEVICES="2,3"

# Check the JSON files for the expected JSON format
DATASET_FILE="examples/inference/cogvideox/dummy_text_to_video.json"

# Depending on how many GPUs you have available, choose your degree of parallelism and technique!
DDP_1="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 1 --dp_shards 1 --cp_degree 1 --tp_degree 1"
DDP_2="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 2 --dp_shards 1 --cp_degree 1 --tp_degree 1"
DDP_4="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 4 --dp_shards 1 --cp_degree 1 --tp_degree 1"
DDP_8="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 8 --dp_shards 1 --cp_degree 1 --tp_degree 1"
CP_2="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 1 --dp_shards 1 --cp_degree 2 --tp_degree 1"
CP_4="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 1 --dp_shards 1 --cp_degree 4 --tp_degree 1"
# FSDP_2="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 1 --dp_shards 2 --cp_degree 1 --tp_degree 1"
# FSDP_4="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 1 --dp_shards 4 --cp_degree 1 --tp_degree 1"
# HSDP_2_2="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 2 --dp_shards 2 --cp_degree 1 --tp_degree 1"

# Parallel arguments
parallel_cmd=(
  $CP_2
)

# Model arguments
model_cmd=(
  --model_name cogvideox
  --pretrained_model_name_or_path "THUDM/CogVideoX-5B"
  --enable_slicing
  --enable_tiling
)

# Inference arguments
inference_cmd=(
  --inference_type text_to_video
  --dataset_file "$DATASET_FILE"
)

# Attention provider arguments
attn_provider_cmd=(
  --attn_provider sage
)

# Torch config arguments
torch_config_cmd=(
  --allow_tf32
  --float32_matmul_precision high
)

# Miscellaneous arguments
miscellaneous_cmd=(
  --seed 31337
  --tracker_name "finetrainers-inference"
  --output_dir "/raid/aryan/cogvideox-inference"
  --init_timeout 600
  --nccl_timeout 600
  --report_to "wandb"
)

# Execute the inference script
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=$NUM_GPUS \
  --rdzv_backend c10d \
  --rdzv_endpoint="localhost:19242" \
  examples/inference/inference.py \
    "${parallel_cmd[@]}" \
    "${model_cmd[@]}" \
    "${inference_cmd[@]}" \
    "${attn_provider_cmd[@]}" \
    "${torch_config_cmd[@]}" \
    "${miscellaneous_cmd[@]}"

echo -ne "-------------------- Finished executing script --------------------\n\n"
