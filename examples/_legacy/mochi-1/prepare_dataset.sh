#!/bin/bash

GPU_ID=0
VIDEO_DIR=video-dataset-disney-organized
OUTPUT_DIR=videos_prepared
NUM_FRAMES=37
RESOLUTION=480x848

# Extract width and height from RESOLUTION
WIDTH=$(echo $RESOLUTION | cut -dx -f1)
HEIGHT=$(echo $RESOLUTION | cut -dx -f2)

python trim_and_crop_videos.py $VIDEO_DIR $OUTPUT_DIR --num_frames=$NUM_FRAMES --resolution=$RESOLUTION --force_upsample

CUDA_VISIBLE_DEVICES=$GPU_ID python embed.py $OUTPUT_DIR --shape=${NUM_FRAMES}x${WIDTH}x${HEIGHT}
