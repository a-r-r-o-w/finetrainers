# CogVideoX Factory 🧪

[中文阅读](./README_zh.md)

Fine-tune Cog family of video models for custom video generation under 24GB of GPU memory ⚡️📼

<table align="center">
<tr>
  <td align="center"><video src="https://github.com/user-attachments/assets/aad07161-87cb-4784-9e6b-16d06581e3e5">Your browser does not support the video tag.</video></td>
</tr>
</table>

**Update 29 Nov 2024**: We have added an experimental memory-efficient trainer for Mochi-1. Check it out [here](https://github.com/a-r-r-o-w/cogvideox-factory/blob/main/training/mochi-1/)!

## Quickstart

Clone the repository and make sure the requirements are installed: `pip install -r requirements.txt` and install diffusers from source by `pip install git+https://github.com/huggingface/diffusers`.

Then download a dataset:

```bash
# install `huggingface_hub`
huggingface-cli download \
  --repo-type dataset Wild-Heart/Disney-VideoGeneration-Dataset \
  --local-dir video-dataset-disney
```

Then launch LoRA fine-tuning for text-to-video (modify the different hyperparameters, dataset root, and other configuration options as per your choice):

```bash
# For LoRA finetuning of the text-to-video CogVideoX models
./train_text_to_video_lora.sh

# For full finetuning of the text-to-video CogVideoX models
./train_text_to_video_sft.sh

# For LoRA finetuning of the image-to-video CogVideoX models
./train_image_to_video_lora.sh
```

Assuming your LoRA is saved and pushed to the HF Hub, and named `my-awesome-name/my-awesome-lora`, we can now use the finetuned model for inference:

```diff
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16
).to("cuda")
+ pipe.load_lora_weights("my-awesome-name/my-awesome-lora", adapter_name="cogvideox-lora")
+ pipe.set_adapters(["cogvideox-lora"], [1.0])

video = pipe("<my-awesome-prompt>").frames[0]
export_to_video(video, "output.mp4", fps=8)
```

For Image-to-Video LoRAs trained with multiresolution videos, one must also add the following lines (see [this](https://github.com/a-r-r-o-w/cogvideox-factory/issues/26) Issue for more details):

```python
from diffusers import CogVideoXImageToVideoPipeline

pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    "THUDM/CogVideoX-5b-I2V", torch_dtype=torch.bfloat16
).to("cuda")

# ...

del pipe.transformer.patch_embed.pos_embedding
pipe.transformer.patch_embed.use_learned_positional_embeddings = False
pipe.transformer.config.use_learned_positional_embeddings = False
```

You can also check if your LoRA is correctly mounted [here](tests/test_lora_inference.py).

Below we provide additional sections detailing on more options explored in this repository. They all attempt to make fine-tuning for video models as accessible as possible by reducing memory requirements as much as possible.

## Prepare Dataset and Training

Before starting the training, please check whether the dataset has been prepared according to the [dataset specifications](assets/dataset.md). We provide training scripts suitable for text-to-video and image-to-video generation, compatible with the [CogVideoX model family](https://huggingface.co/collections/THUDM/cogvideo-66c08e62f1685a3ade464cce). Training can be started using the `train*.sh` scripts, depending on the task you want to train. Let's take LoRA fine-tuning for text-to-video as an example.

- Configure environment variables as per your choice:

  ```bash
  export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
  export TORCHDYNAMO_VERBOSE=1
  export WANDB_MODE="offline"
  export NCCL_P2P_DISABLE=1
  export TORCH_NCCL_ENABLE_MONITORING=0
  ```

- Configure which GPUs to use for training: `GPU_IDS="0,1"`

- Choose hyperparameters for training. Let's try to do a sweep on learning rate and optimizer type as an example:

  ```bash
  LEARNING_RATES=("1e-4" "1e-3")
  LR_SCHEDULES=("cosine_with_restarts")
  OPTIMIZERS=("adamw" "adam")
  MAX_TRAIN_STEPS=("3000")
  ```

- Select which Accelerate configuration you would like to train with: `ACCELERATE_CONFIG_FILE="accelerate_configs/uncompiled_1.yaml"`. We provide some default configurations in the `accelerate_configs/` directory - single GPU uncompiled/compiled, 2x GPU DDP, DeepSpeed, etc. You can create your own config files with custom settings using `accelerate config --config_file my_config.yaml`.

- Specify the absolute paths and columns/files for captions and videos.

  ```bash
  DATA_ROOT="/path/to/my/datasets/video-dataset-disney"
  CAPTION_COLUMN="prompt.txt"
  VIDEO_COLUMN="videos.txt"
  ```

- Launch experiments sweeping different hyperparameters:
  ```
  for learning_rate in "${LEARNING_RATES[@]}"; do
    for lr_schedule in "${LR_SCHEDULES[@]}"; do
      for optimizer in "${OPTIMIZERS[@]}"; do
        for steps in "${MAX_TRAIN_STEPS[@]}"; do
          output_dir="/path/to/my/models/cogvideox-lora__optimizer_${optimizer}__steps_${steps}__lr-schedule_${lr_schedule}__learning-rate_${learning_rate}/"

          cmd="accelerate launch --config_file $ACCELERATE_CONFIG_FILE --gpu_ids $GPU_IDS training/cogvideox/cogvideox_text_to_video_lora.py \
            --pretrained_model_name_or_path THUDM/CogVideoX-5b \
            --data_root $DATA_ROOT \
            --caption_column $CAPTION_COLUMN \
            --video_column $VIDEO_COLUMN \
            --id_token BW_STYLE \
            --height_buckets 480 \
            --width_buckets 720 \
            --frame_buckets 49 \
            --dataloader_num_workers 8 \
            --pin_memory \
            --validation_prompt \"BW_STYLE A black and white animated scene unfolds with an anthropomorphic goat surrounded by musical notes and symbols, suggesting a playful environment. Mickey Mouse appears, leaning forward in curiosity as the goat remains still. The goat then engages with Mickey, who bends down to converse or react. The dynamics shift as Mickey grabs the goat, potentially in surprise or playfulness, amidst a minimalistic background. The scene captures the evolving relationship between the two characters in a whimsical, animated setting, emphasizing their interactions and emotions:::BW_STYLE A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance\" \
            --validation_prompt_separator ::: \
            --num_validation_videos 1 \
            --validation_epochs 10 \
            --seed 42 \
            --rank 128 \
            --lora_alpha 128 \
            --mixed_precision bf16 \
            --output_dir $output_dir \
            --max_num_frames 49 \
            --train_batch_size 1 \
            --max_train_steps $steps \
            --checkpointing_steps 1000 \
            --gradient_accumulation_steps 1 \
            --gradient_checkpointing \
            --learning_rate $learning_rate \
            --lr_scheduler $lr_schedule \
            --lr_warmup_steps 400 \
            --lr_num_cycles 1 \
            --enable_slicing \
            --enable_tiling \
            --optimizer $optimizer \
            --beta1 0.9 \
            --beta2 0.95 \
            --weight_decay 0.001 \
            --max_grad_norm 1.0 \
            --allow_tf32 \
            --report_to wandb \
            --nccl_timeout 1800"
          
          echo "Running command: $cmd"
          eval $cmd
          echo -ne "-------------------- Finished executing script --------------------\n\n"
        done
      done
    done
  done
  ```

  To understand what the different parameters mean, you could either take a look at the [args](./training/args.py) file or run the training script with `--help`.

Note: Training scripts are untested on MPS, so performance and memory requirements can differ widely compared to the CUDA reports below.

## Memory requirements

<table align="center">
<tr>
  <td align="center" colspan="2"><b>CogVideoX LoRA Finetuning</b></td>
</tr>
<tr>
  <td align="center"><a href="https://huggingface.co/THUDM/CogVideoX-2b">THUDM/CogVideoX-2b</a></td>
  <td align="center"><a href="https://huggingface.co/THUDM/CogVideoX-5b">THUDM/CogVideoX-5b</a></td>
</tr>
<tr>
  <td align="center"><img src="../assets/lora_2b.png" /></td>
  <td align="center"><img src="../assets/lora_5b.png" /></td>
</tr>

<tr>
  <td align="center" colspan="2"><b>CogVideoX Full Finetuning</b></td>
</tr>
<tr>
  <td align="center"><a href="https://huggingface.co/THUDM/CogVideoX-2b">THUDM/CogVideoX-2b</a></td>
  <td align="center"><a href="https://huggingface.co/THUDM/CogVideoX-5b">THUDM/CogVideoX-5b</a></td>
</tr>
<tr>
  <td align="center"><img src="../assets/sft_2b.png" /></td>
  <td align="center"><img src="../assets/sft_5b.png" /></td>
</tr>
</table>

Supported and verified memory optimizations for training include:

- `CPUOffloadOptimizer` from [`torchao`](https://github.com/pytorch/ao). You can read about its capabilities and limitations [here](https://github.com/pytorch/ao/tree/main/torchao/prototype/low_bit_optim#optimizer-cpu-offload). In short, it allows you to use the CPU for storing trainable parameters and gradients. This results in the optimizer step happening on the CPU, which requires a fast CPU optimizer, such as `torch.optim.AdamW(fused=True)` or applying `torch.compile` on the optimizer step. Additionally, it is recommended not to `torch.compile` your model for training. Gradient clipping and accumulation is not supported yet either.
- Low-bit optimizers from [`bitsandbytes`](https://huggingface.co/docs/bitsandbytes/optimizers). TODO: to test and make [`torchao`](https://github.com/pytorch/ao/tree/main/torchao/prototype/low_bit_optim) ones work
- DeepSpeed Zero2: Since we rely on `accelerate`, follow [this guide](https://huggingface.co/docs/accelerate/en/usage_guides/deepspeed) to configure your `accelerate` installation to enable training with DeepSpeed Zero2 optimizations. 

> [!IMPORTANT]
> The memory requirements are reported after running the `training/prepare_dataset.py`, which converts the videos and captions to latents and embeddings. During training, we directly load the latents and embeddings, and do not require the VAE or the T5 text encoder. However, if you perform validation/testing, these must be loaded and increase the amount of required memory. Not performing validation/testing saves a significant amount of memory, which can be used to focus solely on training if you're on smaller VRAM GPUs.
>
> If you choose to run validation/testing, you can save some memory on lower VRAM GPUs by specifying `--enable_model_cpu_offload`.

### LoRA finetuning

> [!NOTE]
> The memory requirements for image-to-video lora finetuning are similar to that of text-to-video on `THUDM/CogVideoX-5b`, so it hasn't been reported explicitly.
>
> Additionally, to prepare test images for I2V finetuning, you could either generate them on-the-fly by modifying the script, or extract some frames from your training data using:
> `ffmpeg -i input.mp4 -frames:v 1 frame.png`,
> or provide a URL to a valid and accessible image.

<details>
<summary> AdamW </summary>

**Note:** Trying to run CogVideoX-5b without gradient checkpointing OOMs even on an A100 (80 GB), so the memory measurements have not been specified.

With `train_batch_size = 1`:

|       model        | lora rank | gradient_checkpointing | memory_before_training | memory_before_validation | memory_after_validation | memory_after_testing |
|:------------------:|:---------:|:----------------------:|:----------------------:|:------------------------:|:-----------------------:|:--------------------:|
| THUDM/CogVideoX-2b |    16     |          False         |         12.945         |          43.764          |         46.918          |       24.234         |
| THUDM/CogVideoX-2b |    16     |          True          |         12.945         |          12.945          |         21.121          |       24.234         |
| THUDM/CogVideoX-2b |    64     |          False         |         13.035         |          44.314          |         47.469          |       24.469         |
| THUDM/CogVideoX-2b |    64     |          True          |         13.036         |          13.035          |         21.564          |       24.500         |
| THUDM/CogVideoX-2b |    256    |          False         |         13.095         |          45.826          |         48.990          |       25.543         |
| THUDM/CogVideoX-2b |    256    |          True          |         13.094         |          13.095          |         22.344          |       25.537         |
| THUDM/CogVideoX-5b |    16     |          True          |         19.742         |          19.742          |         28.746          |       38.123         |
| THUDM/CogVideoX-5b |    64     |          True          |         20.006         |          20.818          |         30.338          |       38.738         |
| THUDM/CogVideoX-5b |    256    |          True          |         20.771         |          22.119          |         31.939          |       41.537         |

With `train_batch_size = 4`:

|       model        | lora rank | gradient_checkpointing | memory_before_training | memory_before_validation | memory_after_validation | memory_after_testing |
|:------------------:|:---------:|:----------------------:|:----------------------:|:------------------------:|:-----------------------:|:--------------------:|
| THUDM/CogVideoX-2b |    16     |          True          |         12.945         |          21.803          |         21.814          |       24.322         |
| THUDM/CogVideoX-2b |    64     |          True          |         13.035         |          22.254          |         22.254          |       24.572         |
| THUDM/CogVideoX-2b |    256    |          True          |         13.094         |          22.020          |         22.033          |       25.574         |
| THUDM/CogVideoX-5b |    16     |          True          |         19.742         |          46.492          |         46.492          |       38.197         |
| THUDM/CogVideoX-5b |    64     |          True          |         20.006         |          47.805          |         47.805          |       39.365         |
| THUDM/CogVideoX-5b |    256    |          True          |         20.771         |          47.268          |         47.332          |       41.008         |

</details>

<details>
<summary> AdamW (8-bit bitsandbytes) </summary>

**Note:** Trying to run CogVideoX-5b without gradient checkpointing OOMs even on an A100 (80 GB), so the memory measurements have not been specified.

With `train_batch_size = 1`:

|       model        | lora rank | gradient_checkpointing | memory_before_training | memory_before_validation | memory_after_validation | memory_after_testing |
|:------------------:|:---------:|:----------------------:|:----------------------:|:------------------------:|:-----------------------:|:--------------------:|
| THUDM/CogVideoX-2b |    16     |          False         |         12.945         |          43.732          |         46.887          |        24.195        |
| THUDM/CogVideoX-2b |    16     |          True          |         12.945         |          12.945          |         21.430          |        24.195        |
| THUDM/CogVideoX-2b |    64     |          False         |         13.035         |          44.004          |         47.158          |        24.369        |
| THUDM/CogVideoX-2b |    64     |          True          |         13.035         |          13.035          |         21.297          |        24.357        |
| THUDM/CogVideoX-2b |    256    |          False         |         13.035         |          45.291          |         48.455          |        24.836        |
| THUDM/CogVideoX-2b |    256    |          True          |         13.035         |          13.035          |         21.625          |        24.869        |
| THUDM/CogVideoX-5b |    16     |          True          |         19.742         |          19.742          |         28.602          |        38.049        |
| THUDM/CogVideoX-5b |    64     |          True          |         20.006         |          20.818          |         29.359          |        38.520        |
| THUDM/CogVideoX-5b |    256    |          True          |         20.771         |          21.352          |         30.727          |        39.596        |

With `train_batch_size = 4`:

|       model        | lora rank | gradient_checkpointing | memory_before_training | memory_before_validation | memory_after_validation | memory_after_testing |
|:------------------:|:---------:|:----------------------:|:----------------------:|:------------------------:|:-----------------------:|:--------------------:|
| THUDM/CogVideoX-2b |    16     |          True          |         12.945         |          21.734          |         21.775          |       24.281         |
| THUDM/CogVideoX-2b |    64     |          True          |         13.036         |          21.941          |         21.941          |       24.445         |
| THUDM/CogVideoX-2b |    256    |          True          |         13.094         |          22.020          |         22.266          |       24.943         |
| THUDM/CogVideoX-5b |    16     |          True          |         19.742         |          46.320          |         46.326          |       38.104         |
| THUDM/CogVideoX-5b |    64     |          True          |         20.006         |          46.820          |         46.820          |       38.588         |
| THUDM/CogVideoX-5b |    256    |          True          |         20.771         |          47.920          |         47.980          |       40.002         |

</details>

<details>
<summary> AdamW + CPUOffloadOptimizer (with gradient offloading) </summary>

**Note:** Trying to run CogVideoX-5b without gradient checkpointing OOMs even on an A100 (80 GB), so the memory measurements have not been specified.

With `train_batch_size = 1`:

|       model        | lora rank | gradient_checkpointing | memory_before_training | memory_before_validation | memory_after_validation | memory_after_testing |
|:------------------:|:---------:|:----------------------:|:----------------------:|:------------------------:|:-----------------------:|:--------------------:|
| THUDM/CogVideoX-2b |    16     |          False         |         12.945         |          43.705          |         46.859          |       24.180         |
| THUDM/CogVideoX-2b |    16     |          True          |         12.945         |          12.945          |         21.395          |       24.180         |
| THUDM/CogVideoX-2b |    64     |          False         |         13.035         |          43.916          |         47.070          |       24.234         |
| THUDM/CogVideoX-2b |    64     |          True          |         13.035         |          13.035          |         20.887          |       24.266         |
| THUDM/CogVideoX-2b |    256    |          False         |         13.095         |          44.947          |         48.111          |       24.607         |
| THUDM/CogVideoX-2b |    256    |          True          |         13.095         |          13.095          |         21.391          |       24.635         |
| THUDM/CogVideoX-5b |    16     |          True          |         19.742         |          19.742          |         28.533          |       38.002         |
| THUDM/CogVideoX-5b |    64     |          True          |         20.006         |          20.006          |         29.107          |       38.785         |
| THUDM/CogVideoX-5b |    256    |          True          |         20.771         |          20.771          |         30.078          |       39.559         |

With `train_batch_size = 4`:

|       model        | lora rank | gradient_checkpointing | memory_before_training | memory_before_validation | memory_after_validation | memory_after_testing |
|:------------------:|:---------:|:----------------------:|:----------------------:|:------------------------:|:-----------------------:|:--------------------:|
| THUDM/CogVideoX-2b |    16     |          True          |         12.945         |          21.709          |         21.762          |       24.254         |
| THUDM/CogVideoX-2b |    64     |          True          |         13.035         |          21.844          |         21.855          |       24.338         |
| THUDM/CogVideoX-2b |    256    |          True          |         13.094         |          22.020          |         22.031          |       24.709         |
| THUDM/CogVideoX-5b |    16     |          True          |         19.742         |          46.262          |         46.297          |       38.400         |
| THUDM/CogVideoX-5b |    64     |          True          |         20.006         |          46.561          |         46.574          |       38.840         |
| THUDM/CogVideoX-5b |    256    |          True          |         20.771         |          47.268          |         47.332          |       39.623         |

</details>

<details>
<summary> DeepSpeed (AdamW + CPU/Parameter offloading) </summary>

**Note:** Results are reported with `gradient_checkpointing` enabled, running on a 2x A100.

With `train_batch_size = 1`:

|       model        | memory_before_training | memory_before_validation | memory_after_validation | memory_after_testing |
|:------------------:|:----------------------:|:------------------------:|:-----------------------:|:--------------------:|
| THUDM/CogVideoX-2b |         13.141         |          13.141          |         21.070          |       24.602         |
| THUDM/CogVideoX-5b |         20.170         |          20.170          |         28.662          |       38.957         |

With `train_batch_size = 4`:

|       model        | memory_before_training | memory_before_validation | memory_after_validation | memory_after_testing |
|:------------------:|:----------------------:|:------------------------:|:-----------------------:|:--------------------:|
| THUDM/CogVideoX-2b |         13.141         |          19.854          |         20.836          |       24.709         |
| THUDM/CogVideoX-5b |         20.170         |          40.635          |         40.699          |       39.027         |

</details>

### Full finetuning

> [!NOTE]
> The memory requirements for image-to-video full finetuning are similar to that of text-to-video on `THUDM/CogVideoX-5b`, so it hasn't been reported explicitly.
>
> Additionally, to prepare test images for I2V finetuning, you could either generate them on-the-fly by modifying the script, or extract some frames from your training data using:
> `ffmpeg -i input.mp4 -frames:v 1 frame.png`,
> or provide a URL to a valid and accessible image.

> [!NOTE]
> Trying to run full finetuning without gradient checkpointing OOMs even on an A100 (80 GB), so the memory measurements have not been specified.

<details>
<summary> AdamW </summary>

With `train_batch_size = 1`:

|       model        | gradient_checkpointing | memory_before_training | memory_before_validation | memory_after_validation | memory_after_testing |
|:------------------:|:----------------------:|:----------------------:|:------------------------:|:-----------------------:|:--------------------:|
| THUDM/CogVideoX-2b |          True          |         16.396         |          33.934          |         43.848          |       37.520         |
| THUDM/CogVideoX-5b |          True          |         30.061         |          OOM             |         OOM             |       OOM            |

With `train_batch_size = 4`:

|       model        | gradient_checkpointing | memory_before_training | memory_before_validation | memory_after_validation | memory_after_testing |
|:------------------:|:----------------------:|:----------------------:|:------------------------:|:-----------------------:|:--------------------:|
| THUDM/CogVideoX-2b |          True          |         16.396         |          38.281          |         48.341          |       37.544         |
| THUDM/CogVideoX-5b |          True          |         30.061         |          OOM             |         OOM             |       OOM            |

</details>

<details>
<summary> AdamW (8-bit bitsandbytes) </summary>

With `train_batch_size = 1`:

|       model        | gradient_checkpointing | memory_before_training | memory_before_validation | memory_after_validation | memory_after_testing |
|:------------------:|:----------------------:|:----------------------:|:------------------------:|:-----------------------:|:--------------------:|
| THUDM/CogVideoX-2b |          True          |         16.396         |          16.447          |         27.555          |       27.156         |
| THUDM/CogVideoX-5b |          True          |         30.061         |          52.826          |         58.570          |       49.541         |

With `train_batch_size = 4`:

|       model        | gradient_checkpointing | memory_before_training | memory_before_validation | memory_after_validation | memory_after_testing |
|:------------------:|:----------------------:|:----------------------:|:------------------------:|:-----------------------:|:--------------------:|
| THUDM/CogVideoX-2b |          True          |         16.396         |          27.930          |         27.990          |       27.326         |
| THUDM/CogVideoX-5b |          True          |         16.396         |          66.648          |         66.705          |       48.828         |

</details>

<details>
<summary> AdamW + CPUOffloadOptimizer (with gradient offloading) </summary>

With `train_batch_size = 1`:

|       model        | gradient_checkpointing | memory_before_training | memory_before_validation | memory_after_validation | memory_after_testing |
|:------------------:|:----------------------:|:----------------------:|:------------------------:|:-----------------------:|:--------------------:|
| THUDM/CogVideoX-2b |          True          |         16.396         |          16.396          |         26.100          |       23.832         |
| THUDM/CogVideoX-5b |          True          |         30.061         |          39.359          |         48.307          |       37.947         |

With `train_batch_size = 4`:

|       model        | gradient_checkpointing | memory_before_training | memory_before_validation | memory_after_validation | memory_after_testing |
|:------------------:|:----------------------:|:----------------------:|:------------------------:|:-----------------------:|:--------------------:|
| THUDM/CogVideoX-2b |          True          |         16.396         |          27.916          |         27.975          |       23.936         |
| THUDM/CogVideoX-5b |          True          |         30.061         |          66.607          |         66.668          |       38.061         |

</details>

<details>
<summary> DeepSpeed (AdamW + CPU/Parameter offloading) </summary>

**Note:** Results are reported with `gradient_checkpointing` enabled, running on a 2x A100.

With `train_batch_size = 1`:

|       model        | memory_before_training | memory_before_validation | memory_after_validation | memory_after_testing |
|:------------------:|:----------------------:|:------------------------:|:-----------------------:|:--------------------:|
| THUDM/CogVideoX-2b |         13.111         |          13.111          |         20.328          |       23.867         |
| THUDM/CogVideoX-5b |         19.762         |          19.998          |         27.697          |       38.018         |

With `train_batch_size = 4`:

|       model        | memory_before_training | memory_before_validation | memory_after_validation | memory_after_testing |
|:------------------:|:----------------------:|:------------------------:|:-----------------------:|:--------------------:|
| THUDM/CogVideoX-2b |         13.111         |          21.188          |         21.254          |       23.869         |
| THUDM/CogVideoX-5b |         19.762         |          43.465          |         43.531          |       38.082         |

</details>

> [!NOTE]
> - `memory_after_validation` is indicative of the peak memory required for training. This is because apart from the activations, parameters and gradients stored for training, you also need to load the vae and text encoder in memory and spend some memory to perform inference. In order to reduce total memory required to perform training, one can choose not to perform validation/testing as part of the training script.
>
> - `memory_before_validation` is the true indicator of the peak memory required for training if you choose to not perform validation/testing.

<table align="center">
<tr>
  <td align="center"><a href="https://www.youtube.com/watch?v=UvRl4ansfCg"> Slaying OOMs with PyTorch</a></td>
</tr>
<tr>
  <td align="center"><img src="assets/slaying-ooms.png" style="width: 480px; height: 480px;"></td>
</tr>
</table>

## TODOs

- [x] Make scripts compatible with DDP
- [ ] Make scripts compatible with FSDP
- [x] Make scripts compatible with DeepSpeed
- [ ] vLLM-powered captioning script
- [x] Multi-resolution/frame support in `prepare_dataset.py`
- [ ] Analyzing traces for potential speedups and removing as many syncs as possible
- [x] Test scripts with memory-efficient optimizer from bitsandbytes
- [x] Test scripts with CPUOffloadOptimizer, etc.
- [ ] Test scripts with torchao quantization, and low bit memory optimizers (Currently errors with AdamW (8/4-bit torchao))
- [ ] Test scripts with AdamW (8-bit bitsandbytes) + CPUOffloadOptimizer (with gradient offloading) (Currently errors out)
- [ ] [Sage Attention](https://github.com/thu-ml/SageAttention) (work with the authors to support backward pass, and optimize for A100)

> [!IMPORTANT]
> Since our goal is to make the scripts as memory-friendly as possible we don't guarantee multi-GPU training.
