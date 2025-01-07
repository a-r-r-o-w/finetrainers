# finetrainers 🧪

`cogvideox-factory` was renamed to `finetrainers`. If you're looking to train CogVideoX or Mochi with the legacy training scripts, please refer to [this](./training/README.md) README instead. Everything in the `training/` directory will be eventually moved and supported under `finetrainers`.

FineTrainers is a work-in-progress library to support (accessible) training of video models. Our first priority is to support LoRA training for all popular video models in [Diffusers](https://github.com/huggingface/diffusers), and eventually other methods like controlnets, control-loras, distillation, etc.

<table align="center">
<tr>
  <td align="center"><video src="https://github.com/user-attachments/assets/aad07161-87cb-4784-9e6b-16d06581e3e5">Your browser does not support the video tag.</video></td>
</tr>
</table>

## News

- 🔥 **2024-12-20**: Support for T2V LoRA finetuning of [CogVideoX](https://huggingface.co/docs/diffusers/main/api/pipelines/cogvideox) added! 
- 🔥 **2024-12-20**: Support for T2V LoRA finetuning of [Hunyuan Video](https://huggingface.co/docs/diffusers/main/api/pipelines/hunyuan_video) added! We would like to thank @SHYuanBest for his work on a training script [here](https://github.com/huggingface/diffusers/pull/10254).
- 🔥 **2024-12-18**: Support for T2V LoRA finetuning of [LTX Video](https://huggingface.co/docs/diffusers/main/api/pipelines/ltx_video) added!

## Table of Contents

* [Quickstart](#quickstart)
* [Support Matrix](#support-matrix)
* [Acknowledgements](#acknowledgements)

## Quickstart

Clone the repository and make sure the requirements are installed: `pip install -r requirements.txt` and install `diffusers` from source by `pip install git+https://github.com/huggingface/diffusers`.

Then download a dataset:

```bash
# install `huggingface_hub`
huggingface-cli download \
  --repo-type dataset Wild-Heart/Disney-VideoGeneration-Dataset \
  --local-dir video-dataset-disney
```

Then launch LoRA fine-tuning. Below we provide an example for LTX-Video. We refer the users to [`docs/training`](./docs/training/) to learn more details.

> [!IMPORTANT] 
> It is recommended to use Pytorch 2.5.1 or above for training. Previous versions can lead to completely black videos, OOM errors, or other issues and are not tested.

<details>
<summary>Training command</summary>

```bash
#!/bin/bash
export WANDB_MODE="offline"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export FINETRAINERS_LOG_LEVEL=DEBUG

GPU_IDS="0,1"

DATA_ROOT="/path/to/video-dataset-disney"
CAPTION_COLUMN="prompts.txt"
VIDEO_COLUMN="videos.txt"
OUTPUT_DIR="/path/to/output/directory/ltx-video/ltxv_disney"

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

</details>

Here we are using two GPUs. But one can do single-GPU training by setting `GPU_IDS=0`. By default, we are using some simple optimizations to reduce memory consumption (such as gradient checkpointing). Please refer to [docs/training/optimizations](./docs/training/optimization.md) to learn about the memory optimizations currently supported.

For inference, refer [here](./docs/training/ltx_video.md#inference). For docs related to the other supported model, refer [here](./docs/training/).

## Support Matrix

<div align="center">

| **Model Name** | **Tasks** | **Min. GPU VRAM** |
|:---:|:---:|:---:|
| [LTX-Video](./docs/training/ltx_video.md) | Text-to-Video | 11 GB |
| [HunyuanVideo](./docs/training/hunyuan_video.md) | Text-to-Video | 42 GB |
| [CogVideoX](./docs/training/cogvideox.md) | Text-to-Video | 12GB<sup>*</sup> |

</div>

<sub><sup>*</sup>Noted for the 5B variant.</sub>

Note that the memory consumption in the table is reported with most of the options, discussed in [docs/training/optimizations](./docs/training/optimization.md), enabled.

If you would like to use a custom dataset, refer to the dataset preparation guide [here](./docs/dataset/README.md).

## Acknowledgements

* `finetrainers` builds on top of a body of great open-source libraries: `transformers`, `accelerate`, `peft`, `diffusers`, `bitsandbytes`, `torchao`, `deepspeed` -- to name a few.
* Some of the design choices of `finetrainers` were inspired by [`SimpleTuner`](https://github.com/bghira/SimpleTuner).
