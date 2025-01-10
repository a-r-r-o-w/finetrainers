#!/bin/bash

# This shell script is for the maintainers and contributors to QUICKLY check
# if the major changes they're introducing still work with the rest of the models supported
# in `finetrainers`. It DOES NOT give a sense of implementation correctness as that requires
# much longer training runs but it DOES ensure basic functionalities work in the large training
# setup.

# It should be run as so from the root of `finetrainers`: `bash tests/test_model_runs_minimally_lora.sh`

######################################################
# Set common variables.
######################################################

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export ROOT_DIR
export WANDB_MODE="offline"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export FINETRAINERS_LOG_LEVEL=DEBUG

echo "Using $ROOT_DIR as rootdir."

######################################################
# Download Disney dataset.
######################################################

# Ensure dataset is downloaded
DATA_ROOT="$ROOT_DIR/video-dataset-disney"
if [ ! -d "$DATA_ROOT" ]; then
    echo "Downloading Disney dataset to $DATA_ROOT..."
    huggingface-cli download \
        --repo-type dataset Wild-Heart/Disney-VideoGeneration-Dataset \
        --local-dir "$DATA_ROOT"
else
    echo "Dataset already exists at $DATA_ROOT. Skipping download."
fi

######################################################
# Run models
######################################################

Define models to test
models=("dummy_ltx_video_lora" "dummy_cogvideox_lora" "dummy_hunyuanvideo_lora")
for model_script in "${models[@]}"; do
    echo "Running $model_script test..."
    bash $ROOT_DIR/tests/scripts/$model_script.sh
done