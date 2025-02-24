#!/bin/bash

MODEL_ID="THUDM/CogVideoX-2b"

NUM_GPUS=8

# For more details on the expected data format, please refer to the README.
DATA_ROOT="/path/to/my/datasets/video-dataset"  # This needs to be the path to the base directory where your videos are located.
CAPTION_COLUMN="prompt.txt"
VIDEO_COLUMN="videos.txt"
OUTPUT_DIR="/path/to/my/datasets/preprocessed-dataset"
HEIGHT_BUCKETS="480 720"
WIDTH_BUCKETS="720 960"
FRAME_BUCKETS="49"
MAX_NUM_FRAMES="49"
MAX_SEQUENCE_LENGTH=226
TARGET_FPS=8
BATCH_SIZE=1
DTYPE=fp32

# To create a folder-style dataset structure without pre-encoding videos and captions
# For Image-to-Video finetuning, make sure to pass `--save_image_latents`
CMD_WITHOUT_PRE_ENCODING="\
  torchrun --nproc_per_node=$NUM_GPUS \
    training/prepare_dataset.py \
      --model_id $MODEL_ID \
      --data_root $DATA_ROOT \
      --caption_column $CAPTION_COLUMN \
      --video_column $VIDEO_COLUMN \
      --output_dir $OUTPUT_DIR \
      --height_buckets $HEIGHT_BUCKETS \
      --width_buckets $WIDTH_BUCKETS \
      --frame_buckets $FRAME_BUCKETS \
      --max_num_frames $MAX_NUM_FRAMES \
      --max_sequence_length $MAX_SEQUENCE_LENGTH \
      --target_fps $TARGET_FPS \
      --batch_size $BATCH_SIZE \
      --dtype $DTYPE
"

CMD_WITH_PRE_ENCODING="$CMD_WITHOUT_PRE_ENCODING --save_latents_and_embeddings"

# Select which you'd like to run
CMD=$CMD_WITH_PRE_ENCODING

echo "===== Running \`$CMD\` ====="
eval $CMD
echo -ne "===== Finished running script =====\n"
