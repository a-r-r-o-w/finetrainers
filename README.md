# finetrainers 🧪

`cogvideox-factory` was renamed to `finetrainers`. If you're looking to train CogVideoX or Mochi with the legacy training scripts, please refer to [this](./training/README.md) README instead. Everything in the `training/` directory will be eventually moved and supported under `finetrainers`.

FineTrainers is a work-in-progress library to support (accessible) training of video models. Our first priority is to support LoRA training for all popular video models in [Diffusers](https://github.com/huggingface/diffusers), and eventually other methods like controlnets, control-loras, distillation, etc.

<table align="center">
<tr>
  <td align="center"><video src="https://github.com/user-attachments/assets/aad07161-87cb-4784-9e6b-16d06581e3e5">Your browser does not support the video tag.</video></td>
</tr>
</table>

## News

- 🔥 **2025-02-12**: We have shipped a set of tooling to curate small and high-quality video datasets for fine-tuning. See [datasets](./docs/dataset/README.md) documentation page for details!
- 🔥 **2025-02-12**: Check out [eisneim/ltx_lora_training_i2v_t2v](https://github.com/eisneim/ltx_lora_training_i2v_t2v/)! It builds off of `finetrainers` to support image to video training for LTX-Video and STG guidance for inference.
- 🔥 **2025-01-15**: Support for naive FP8 weight-casting training added! This allows training HunyuanVideo in under 24 GB upto specific resolutions.
- 🔥 **2025-01-13**: Support for T2V full-finetuning added! Thanks to [@ArEnSc](https://github.com/ArEnSc) for taking up the initiative!
- 🔥 **2025-01-03**: Support for T2V LoRA finetuning of [CogVideoX](https://huggingface.co/docs/diffusers/main/api/pipelines/cogvideox) added! 
- 🔥 **2024-12-20**: Support for T2V LoRA finetuning of [Hunyuan Video](https://huggingface.co/docs/diffusers/main/api/pipelines/hunyuan_video) added! We would like to thank @SHYuanBest for his work on a training script [here](https://github.com/huggingface/diffusers/pull/10254).
- 🔥 **2024-12-18**: Support for T2V LoRA finetuning of [LTX Video](https://huggingface.co/docs/diffusers/main/api/pipelines/ltx_video) added!

## Table of Contents

* [Quickstart](#quickstart)
* [Support Matrix](#support-matrix)
* [Acknowledgements](#acknowledgements)

## Quickstart

Clone the repository and make sure the requirements are installed: `pip install -r requirements.txt` and install `diffusers` from source by `pip install git+https://github.com/huggingface/diffusers`. The requirements specify `diffusers>=0.32.1`, but it is always recommended to use the `main` branch for the latest features and bugfixes.

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

TODO: LTX does not do too well with the disney dataset. We will update this to use a better example soon.

```bash
#!/bin/bash
export WANDB_MODE="offline"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export FINETRAINERS_LOG_LEVEL=DEBUG

GPU_IDS="0,1"

DATA_ROOT="/path/to/video-dataset-disney"
CAPTION_COLUMN="prompt.txt"
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
diffusion_cmd="--flow_weighting_scheme logit_normal"

# Training arguments
training_cmd="--training_type lora \
  --seed 42 \
  --batch_size 1 \
  --train_steps 3000 \
  --rank 128 \
  --lora_alpha 128 \
  --target_modules to_q to_k to_v to_out.0 \
  --gradient_accumulation_steps 4 \
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

| **Model Name**                                   | **Tasks**     | **Min. LoRA VRAM<sup>*</sup>**     | **Min. Full Finetuning VRAM<sup>^</sup>**     |
|:------------------------------------------------:|:-------------:|:----------------------------------:|:---------------------------------------------:|
| [LTX-Video](./docs/training/ltx_video.md)        | Text-to-Video | 5 GB                               | 21 GB                                         |
| [HunyuanVideo](./docs/training/hunyuan_video.md) | Text-to-Video | 32 GB                              | OOM                                           |
| [CogVideoX-5b](./docs/training/cogvideox.md)     | Text-to-Video | 18 GB                              | 53 GB                                         |

</div>

<sub><sup>*</sup>Noted for training-only, no validation, at resolution `49x512x768`, rank 128, with pre-computation, using **FP8** weights & gradient checkpointing. Pre-computation of conditions and latents may require higher limits (but typically under 16 GB).</sub><br/>
<sub><sup>^</sup>Noted for training-only, no validation, at resolution `49x512x768`, with pre-computation, using **BF16** weights & gradient checkpointing.</sub>

If you would like to use a custom dataset, refer to the dataset preparation guide [here](./docs/dataset/README.md).

## Acknowledgements

* `finetrainers` builds on top of a body of great open-source libraries: `transformers`, `accelerate`, `peft`, `diffusers`, `bitsandbytes`, `torchao`, `deepspeed` -- to name a few.
* Some of the design choices of `finetrainers` were inspired by [`SimpleTuner`](https://github.com/bghira/SimpleTuner).
