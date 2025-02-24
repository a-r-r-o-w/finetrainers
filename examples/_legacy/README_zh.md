# CogVideoX Factory 🧪

[Read in English](./README.md)

在 24GB GPU 内存下对 Cog 系列视频模型进行微调以实现自定义视频生成，支持多分辨率 ⚡️📼

<table align="center">
<tr>
  <td align="center"><video src="https://github.com/user-attachments/assets/aad07161-87cb-4784-9e6b-16d06581e3e5">您的浏览器不支持视频标签。</video></td>
</tr>
</table>

## 快速开始

克隆此仓库并确保安装了相关依赖：`pip install -r requirements.txt`。

接着下载数据集：

```
# 安装 `huggingface_hub`
huggingface-cli download   --repo-type dataset Wild-Heart/Disney-VideoGeneration-Dataset   --local-dir video-dataset-disney
```

然后启动 LoRA 微调进行文本到视频的生成（根据您的选择修改不同的超参数、数据集根目录以及其他配置选项）：

```
# 对 CogVideoX 模型进行文本到视频的 LoRA 微调
./train_text_to_video_lora.sh

# 对 CogVideoX 模型进行文本到视频的完整微调
./train_text_to_video_sft.sh

# 对 CogVideoX 模型进行图像到视频的 LoRA 微调
./train_image_to_video_lora.sh
```

假设您的 LoRA 已保存并推送到 HF Hub，并命名为 `my-awesome-name/my-awesome-lora`，现在我们可以使用微调模型进行推理：

```
import torch
from diffusers import CogVideoXPipeline
from diffusers import export_to_video

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16
).to("cuda")
+ pipe.load_lora_weights("my-awesome-name/my-awesome-lora", adapter_name=["cogvideox-lora"])
+ pipe.set_adapters(["cogvideox-lora"], [1.0])

video = pipe("<my-awesome-prompt>").frames[0]
export_to_video(video, "output.mp4", fps=8)
```

你也可以在[这里](tests/test_lora_inference.py)来检查你的Lora是否正常挂载。

**注意：** 对于图像到视频的微调，您必须从 [这个分支](https://github.com/huggingface/diffusers/pull/9482) 安装
diffusers（该分支为 CogVideoX 的图像到视频添加了 LoRA 加载支持）直到它被合并。

以下我们提供了更多探索此仓库选项的额外部分。所有这些都旨在尽可能降低内存需求，使视频模型的微调变得更易于访问。

## 训练

在开始训练之前，请你检查是否按照[数据集规范](assets/dataset_zh.md)准备好了数据集。 我们提供了适用于文本到视频 (text-to-video) 和图像到视频 (image-to-video) 生成的训练脚本，兼容 [CogVideoX 模型家族](https://huggingface.co/collections/THUDM/cogvideo-66c08e62f1685a3ade464cce)。训练可以通过 `train*.sh` 脚本启动，具体取决于你想要训练的任务。让我们以文本到视频的 LoRA 微调为例。

- 根据你的需求配置环境变量：

  ```
  export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
  export TORCHDYNAMO_VERBOSE=1
  export WANDB_MODE="offline"
  export NCCL_P2P_DISABLE=1
  export TORCH_NCCL_ENABLE_MONITORING=0
  ```

- 配置用于训练的 GPU：`GPU_IDS="0,1"`

- 选择训练的超参数。让我们以学习率和优化器类型的超参数遍历为例：

  ```
  LEARNING_RATES=("1e-4" "1e-3")
  LR_SCHEDULES=("cosine_with_restarts")
  OPTIMIZERS=("adamw" "adam")
  MAX_TRAIN_STEPS=("3000")
  ```

- 选择用于训练的 Accelerate 配置文件：`ACCELERATE_CONFIG_FILE="accelerate_configs/uncompiled_1.yaml"`
  。我们在 `accelerate_configs/` 目录中提供了一些默认配置 - 单 GPU 编译/未编译、2x GPU DDP、DeepSpeed
  等。你也可以使用 `accelerate config --config_file my_config.yaml` 自定义配置文件。

- 指定字幕和视频的绝对路径以及列/文件。

  ```
  DATA_ROOT="/path/to/my/datasets/video-dataset-disney"
  CAPTION_COLUMN="prompt.txt"
  VIDEO_COLUMN="videos.txt"
  ```

- 运行实验，遍历不同的超参数：
    ```
  for learning_rate in "${LEARNING_RATES[@]}"; do
    for lr_schedule in "${LR_SCHEDULES[@]}"; do
      for optimizer in "${OPTIMIZERS[@]}"; do
        for steps in "${MAX_TRAIN_STEPS[@]}"; do
          output_dir="/path/to/my/models/cogvideox-lora__optimizer_${optimizer}__steps_${steps}__lr-schedule_${lr_schedule}__learning-rate_${learning_rate}/"

          cmd="accelerate launch --config_file $ACCELERATE_CONFIG_FILE --gpu_ids $GPU_IDS training/cogvideox_text_to_video_lora.py \
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

要了解不同参数的含义，你可以查看 [args](./training/args.py) 文件，或者使用 `--help` 运行训练脚本。

注意：训练脚本尚未在 MPS 上测试，因此性能和内存要求可能与下面的 CUDA 报告差异很大。

## 内存需求

<table align="center">
<tr>
  <td align="center" colspan="2"><b>CogVideoX LoRA 微调</b></td>
</tr>
<tr>
  <td align="center"><a href="https://huggingface.co/THUDM/CogVideoX-2b">THUDM/CogVideoX-2b</a></td>
  <td align="center"><a href="https://huggingface.co/THUDM/CogVideoX-5b">THUDM/CogVideoX-5b</a></td>
</tr>
<tr>
  <td align="center"><img src="assets/lora_2b.png" /></td>
  <td align="center"><img src="assets/lora_5b.png" /></td>
</tr>

<tr>
  <td align="center" colspan="2"><b>CogVideoX 全量微调</b></td>
</tr>
<tr>
  <td align="center"><a href="https://huggingface.co/THUDM/CogVideoX-2b">THUDM/CogVideoX-2b</a></td>
  <td align="center"><a href="https://huggingface.co/THUDM/CogVideoX-5b">THUDM/CogVideoX-5b</a></td>
</tr>
<tr>
  <td align="center"><img src="assets/sft_2b.png" /></td>
  <td align="center"><img src="assets/sft_5b.png" /></td>
</tr>
</table>

支持和验证的训练内存优化包括：

- `CPUOffloadOptimizer` 来自 [`torchao`](https://github.com/pytorch/ao)
  。你可以在[这里](https://github.com/pytorch/ao/tree/main/torchao/prototype/low_bit_optim#optimizer-cpu-offload)
  阅读它的能力和局限性。简而言之，它允许你将可训练参数和梯度存储在 CPU 中，从而在 CPU 上进行优化步骤。这需要快速的 CPU
  优化器，如 `torch.optim.AdamW(fused=True)`，或者在优化步骤中应用 `torch.compile`
  。此外，建议不要在训练时对模型应用 `torch.compile`。梯度裁剪和累积目前还不支持。
- 来自 [`bitsandbytes`](https://huggingface.co/docs/bitsandbytes/optimizers)
  的低位优化器。TODO：测试并使 [`torchao`](https://github.com/pytorch/ao/tree/main/torchao/prototype/low_bit_optim) 能正常工作。
- DeepSpeed Zero2：由于我们依赖 `accelerate`
  ，请按照[此指南](https://huggingface.co/docs/accelerate/en/usage_guides/deepspeed) 配置 `accelerate` 以启用 DeepSpeed
  Zero2 优化训练。

> [!重要提示]
> 内存需求是运行 `training/prepare_dataset.py`
>
后报告的，该脚本将视频和字幕转换为潜在向量和嵌入。在训练期间，我们直接加载这些潜在向量和嵌入，不需要VAE或T5文本编码器。然而，如果执行验证/测试，则必须加载这些模块，并且会增加所需内存的数量。不进行验证/测试可以节省大量内存，这些内存可以用于较小显存的GPU上专注于训练。
>
> 如果选择运行验证/测试，可以通过指定 `--enable_model_cpu_offload` 来为较低显存的GPU节省一些内存。

### LoRA微调

> [!重要提示]
> 图像到视频的LoRA微调的内存需求与文本到视频上的 `THUDM/CogVideoX-5b` 类似，因此没有明确报告。
>
> 此外，为了准备I2V微调的测试图像，可以通过修改脚本实时生成它们，或使用以下命令从训练数据中提取一些帧：
> `ffmpeg -i input.mp4 -frames:v 1 frame.png`，
> 或提供一个有效且可访问的图像URL。

<details>
<summary> AdamW </summary>

**注意：** 尝试在没有梯度检查点的情况下运行 CogVideoX-5b 即使在 A100（80 GB）上也会导致 OOM（内存不足）错误，因此内存需求尚未列出。

当 `train_batch_size = 1` 时:

|       model        | lora rank | gradient_checkpointing | memory_before_training | memory_before_validation | memory_after_validation | memory_after_testing |
|:------------------:|:---------:|:----------------------:|:----------------------:|:------------------------:|:-----------------------:|:--------------------:|
| THUDM/CogVideoX-2b |    16     |         False          |         12.945         |          43.764          |         46.918          |        24.234        |
| THUDM/CogVideoX-2b |    16     |          True          |         12.945         |          12.945          |         21.121          |        24.234        |
| THUDM/CogVideoX-2b |    64     |         False          |         13.035         |          44.314          |         47.469          |        24.469        |
| THUDM/CogVideoX-2b |    64     |          True          |         13.036         |          13.035          |         21.564          |        24.500        |
| THUDM/CogVideoX-2b |    256    |         False          |         13.095         |          45.826          |         48.990          |        25.543        |
| THUDM/CogVideoX-2b |    256    |          True          |         13.094         |          13.095          |         22.344          |        25.537        |
| THUDM/CogVideoX-5b |    16     |          True          |         19.742         |          19.742          |         28.746          |        38.123        |
| THUDM/CogVideoX-5b |    64     |          True          |         20.006         |          20.818          |         30.338          |        38.738        |
| THUDM/CogVideoX-5b |    256    |          True          |         20.771         |          22.119          |         31.939          |        41.537        |

当 `train_batch_size = 4` 时:

|       model        | lora rank | gradient_checkpointing | memory_before_training | memory_before_validation | memory_after_validation | memory_after_testing |
|:------------------:|:---------:|:----------------------:|:----------------------:|:------------------------:|:-----------------------:|:--------------------:|
| THUDM/CogVideoX-2b |    16     |          True          |         12.945         |          21.803          |         21.814          |        24.322        |
| THUDM/CogVideoX-2b |    64     |          True          |         13.035         |          22.254          |         22.254          |        24.572        |
| THUDM/CogVideoX-2b |    256    |          True          |         13.094         |          22.020          |         22.033          |        25.574        |
| THUDM/CogVideoX-5b |    16     |          True          |         19.742         |          46.492          |         46.492          |        38.197        |
| THUDM/CogVideoX-5b |    64     |          True          |         20.006         |          47.805          |         47.805          |        39.365        |
| THUDM/CogVideoX-5b |    256    |          True          |         20.771         |          47.268          |         47.332          |        41.008        |

</details>

<details>
<summary> AdamW (8-bit bitsandbytes) </summary>

**注意：** 在没有启用梯度检查点的情况下，尝试运行 CogVideoX-5b 模型即使在 A100（80 GB）上也会导致 OOM（内存不足），因此未列出内存测量数据。

当 `train_batch_size = 1` 时：

|       model        | lora rank | gradient_checkpointing | memory_before_training | memory_before_validation | memory_after_validation | memory_after_testing |
|:------------------:|:---------:|:----------------------:|:----------------------:|:------------------------:|:-----------------------:|:--------------------:|
| THUDM/CogVideoX-2b |    16     |         False          |         12.945         |          43.732          |         46.887          |        24.195        |
| THUDM/CogVideoX-2b |    16     |          True          |         12.945         |          12.945          |         21.430          |        24.195        |
| THUDM/CogVideoX-2b |    64     |         False          |         13.035         |          44.004          |         47.158          |        24.369        |
| THUDM/CogVideoX-2b |    64     |          True          |         13.035         |          13.035          |         21.297          |        24.357        |
| THUDM/CogVideoX-2b |    256    |         False          |         13.035         |          45.291          |         48.455          |        24.836        |
| THUDM/CogVideoX-2b |    256    |          True          |         13.035         |          13.035          |         21.625          |        24.869        |
| THUDM/CogVideoX-5b |    16     |          True          |         19.742         |          19.742          |         28.602          |        38.049        |
| THUDM/CogVideoX-5b |    64     |          True          |         20.006         |          20.818          |         29.359          |        38.520        |
| THUDM/CogVideoX-5b |    256    |          True          |         20.771         |          21.352          |         30.727          |        39.596        |

当 `train_batch_size = 4` 时:

|       model        | lora rank | gradient_checkpointing | memory_before_training | memory_before_validation | memory_after_validation | memory_after_testing |
|:------------------:|:---------:|:----------------------:|:----------------------:|:------------------------:|:-----------------------:|:--------------------:|
| THUDM/CogVideoX-2b |    16     |          True          |         12.945         |          21.734          |         21.775          |        24.281        |
| THUDM/CogVideoX-2b |    64     |          True          |         13.036         |          21.941          |         21.941          |        24.445        |
| THUDM/CogVideoX-2b |    256    |          True          |         13.094         |          22.020          |         22.266          |        24.943        |
| THUDM/CogVideoX-5b |    16     |          True          |         19.742         |          46.320          |         46.326          |        38.104        |
| THUDM/CogVideoX-5b |    64     |          True          |         20.006         |          46.820          |         46.820          |        38.588        |
| THUDM/CogVideoX-5b |    256    |          True          |         20.771         |          47.920          |         47.980          |        40.002        |

</details>

<details>
<summary> AdamW + CPUOffloadOptimizer (with gradient offloading) </summary>

**注意：** 在没有启用梯度检查点的情况下，尝试运行 CogVideoX-5b 模型即使在 A100（80 GB）上也会导致 OOM（内存不足），因此未列出内存测量数据。

当 `train_batch_size = 1` 时：

|       model        | lora rank | gradient_checkpointing | memory_before_training | memory_before_validation | memory_after_validation | memory_after_testing |
|:------------------:|:---------:|:----------------------:|:----------------------:|:------------------------:|:-----------------------:|:--------------------:|
| THUDM/CogVideoX-2b |    16     |         False          |         12.945         |          43.705          |         46.859          |        24.180        |
| THUDM/CogVideoX-2b |    16     |          True          |         12.945         |          12.945          |         21.395          |        24.180        |
| THUDM/CogVideoX-2b |    64     |         False          |         13.035         |          43.916          |         47.070          |        24.234        |
| THUDM/CogVideoX-2b |    64     |          True          |         13.035         |          13.035          |         20.887          |        24.266        |
| THUDM/CogVideoX-2b |    256    |         False          |         13.095         |          44.947          |         48.111          |        24.607        |
| THUDM/CogVideoX-2b |    256    |          True          |         13.095         |          13.095          |         21.391          |        24.635        |
| THUDM/CogVideoX-5b |    16     |          True          |         19.742         |          19.742          |         28.533          |        38.002        |
| THUDM/CogVideoX-5b |    64     |          True          |         20.006         |          20.006          |         29.107          |        38.785        |
| THUDM/CogVideoX-5b |    256    |          True          |         20.771         |          20.771          |         30.078          |        39.559        |

当 `train_batch_size = 4` 时:

|       model        | lora rank | gradient_checkpointing | memory_before_training | memory_before_validation | memory_after_validation | memory_after_testing |
|:------------------:|:---------:|:----------------------:|:----------------------:|:------------------------:|:-----------------------:|:--------------------:|
| THUDM/CogVideoX-2b |    16     |          True          |         12.945         |          21.709          |         21.762          |        24.254        |
| THUDM/CogVideoX-2b |    64     |          True          |         13.035         |          21.844          |         21.855          |        24.338        |
| THUDM/CogVideoX-2b |    256    |          True          |         13.094         |          22.020          |         22.031          |        24.709        |
| THUDM/CogVideoX-5b |    16     |          True          |         19.742         |          46.262          |         46.297          |        38.400        |
| THUDM/CogVideoX-5b |    64     |          True          |         20.006         |          46.561          |         46.574          |        38.840        |
| THUDM/CogVideoX-5b |    256    |          True          |         20.771         |          47.268          |         47.332          |        39.623        |

</details>

<details>
<summary> DeepSpeed (AdamW + CPU/Parameter offloading) </summary>

**注意：** 结果是在启用梯度检查点的情况下，使用 2x A100 运行时记录的。

当 `train_batch_size = 1` 时：

|       model        | memory_before_training | memory_before_validation | memory_after_validation | memory_after_testing |
|:------------------:|:----------------------:|:------------------------:|:-----------------------:|:--------------------:|
| THUDM/CogVideoX-2b |         13.141         |          13.141          |         21.070          |        24.602        |
| THUDM/CogVideoX-5b |         20.170         |          20.170          |         28.662          |        38.957        |

当 `train_batch_size = 4` 时:

|       model        | memory_before_training | memory_before_validation | memory_after_validation | memory_after_testing |
|:------------------:|:----------------------:|:------------------------:|:-----------------------:|:--------------------:|
| THUDM/CogVideoX-2b |         13.141         |          19.854          |         20.836          |        24.709        |
| THUDM/CogVideoX-5b |         20.170         |          40.635          |         40.699          |        39.027        |

</details>

### Full finetuning

> [!注意]  
> 图像到视频的完整微调内存需求与 `THUDM/CogVideoX-5b` 的文本到视频微调相似，因此没有单独列出。
>
> 此外，要准备用于 I2V 微调的测试图像，你可以通过修改脚本实时生成图像，或者从你的训练数据中提取一些帧：  
> `ffmpeg -i input.mp4 -frames:v 1 frame.png`，  
> 或提供一个有效且可访问的图像 URL。

> [!注意]  
> 在没有使用梯度检查点的情况下运行完整微调，即使是在 A100（80GB）上，也会出现 OOM（内存不足）错误，因此未列出内存需求。

<details>
<summary> AdamW </summary>

当 `train_batch_size = 1` 时：

|       model        | gradient_checkpointing | memory_before_training | memory_before_validation | memory_after_validation | memory_after_testing |
|:------------------:|:----------------------:|:----------------------:|:------------------------:|:-----------------------:|:--------------------:|
| THUDM/CogVideoX-2b |          True          |         16.396         |          33.934          |         43.848          |        37.520        |
| THUDM/CogVideoX-5b |          True          |         30.061         |           OOM            |           OOM           |         OOM          |

当 `train_batch_size = 4` 时:

|       model        | gradient_checkpointing | memory_before_training | memory_before_validation | memory_after_validation | memory_after_testing |
|:------------------:|:----------------------:|:----------------------:|:------------------------:|:-----------------------:|:--------------------:|
| THUDM/CogVideoX-2b |          True          |         16.396         |          38.281          |         48.341          |        37.544        |
| THUDM/CogVideoX-5b |          True          |         30.061         |           OOM            |           OOM           |         OOM          |

</details>

<details>
<summary> AdamW (8-bit 量化) </summary>

当 `train_batch_size = 1` 时：

|       model        | gradient_checkpointing | memory_before_training | memory_before_validation | memory_after_validation | memory_after_testing |
|:------------------:|:----------------------:|:----------------------:|:------------------------:|:-----------------------:|:--------------------:|
| THUDM/CogVideoX-2b |          True          |         16.396         |          16.447          |         27.555          |        27.156        |
| THUDM/CogVideoX-5b |          True          |         30.061         |          52.826          |         58.570          |        49.541        |

当 `train_batch_size = 4` 时:

|       model        | gradient_checkpointing | memory_before_training | memory_before_validation | memory_after_validation | memory_after_testing |
|:------------------:|:----------------------:|:----------------------:|:------------------------:|:-----------------------:|:--------------------:|
| THUDM/CogVideoX-2b |          True          |         16.396         |          27.930          |         27.990          |        27.326        |
| THUDM/CogVideoX-5b |          True          |         16.396         |          66.648          |         66.705          |        48.828        |

</details>

<details>
<summary> AdamW + CPUOffloadOptimizer（带有梯度卸载）</summary>

当 `train_batch_size = 1` 时：

|       model        | gradient_checkpointing | memory_before_training | memory_before_validation | memory_after_validation | memory_after_testing |
|:------------------:|:----------------------:|:----------------------:|:------------------------:|:-----------------------:|:--------------------:|
| THUDM/CogVideoX-2b |          True          |         16.396         |          16.396          |         26.100          |        23.832        |
| THUDM/CogVideoX-5b |          True          |         30.061         |          39.359          |         48.307          |        37.947        |

当 `train_batch_size = 4` 时:

|       model        | gradient_checkpointing | memory_before_training | memory_before_validation | memory_after_validation | memory_after_testing |
|:------------------:|:----------------------:|:----------------------:|:------------------------:|:-----------------------:|:--------------------:|
| THUDM/CogVideoX-2b |          True          |         16.396         |          27.916          |         27.975          |        23.936        |
| THUDM/CogVideoX-5b |          True          |         30.061         |          66.607          |         66.668          |        38.061        |

</details>

<details>
<summary> DeepSpeed（AdamW + CPU/参数卸载） </summary>

**注意:** 结果是在启用 `gradient_checkpointing`（梯度检查点）功能，并在 2 台 A100 显卡上运行时报告的。

当 `train_batch_size = 1` 时：

|       model        | memory_before_training | memory_before_validation | memory_after_validation | memory_after_testing |
|:------------------:|:----------------------:|:------------------------:|:-----------------------:|:--------------------:|
| THUDM/CogVideoX-2b |         13.111         |          13.111          |         20.328          |        23.867        |
| THUDM/CogVideoX-5b |         19.762         |          19.998          |         27.697          |        38.018        |

当 `train_batch_size = 4` 时:

|       model        | memory_before_training | memory_before_validation | memory_after_validation | memory_after_testing |
|:------------------:|:----------------------:|:------------------------:|:-----------------------:|:--------------------:|
| THUDM/CogVideoX-2b |         13.111         |          21.188          |         21.254          |        23.869        |
| THUDM/CogVideoX-5b |         19.762         |          43.465          |         43.531          |        38.082        |

</details>

> [!注意]
> - `memory_after_validation`（验证后内存） 表示训练所需的峰值内存。这是因为除了存储训练过程中需要的激活、参数和梯度之外，还需要加载
    VAE 和文本编码器到内存中，并且执行推理操作也会消耗一定内存。为了减少训练所需的总内存，您可以选择在训练脚本中不执行验证/测试。
>
> - 如果选择不进行验证/测试，`memory_before_validation`（验证前内存） 才是训练所需内存的真实指示器。

<table align="center">
<tr>
  <td align="center"><a href="https://www.youtube.com/watch?v=UvRl4ansfCg"> Slaying OOMs with PyTorch</a></td>
</tr>
<tr>
  <td align="center"><img src="assets/slaying-ooms.png" style="width: 480px; height: 480px;"></td>
</tr>
</table>

## 待办事项

- [x] 使脚本兼容 DDP
- [ ] 使脚本兼容 FSDP
- [x] 使脚本兼容 DeepSpeed
- [ ] 基于 vLLM 的字幕脚本
- [x] 在 `prepare_dataset.py` 中支持多分辨率/帧数
- [ ] 分析性能瓶颈并尽可能减少同步操作
- [ ] 支持 QLoRA（优先），以及其他高使用率的 LoRA 方法
- [x] 使用 bitsandbytes 的节省内存优化器测试脚本
- [x] 使用 CPUOffloadOptimizer 等测试脚本
- [ ] 使用 torchao 量化和低位内存优化器测试脚本（目前在 AdamW（8/4-bit torchao）上报错）
- [ ] 使用 AdamW（8-bit bitsandbytes）+ CPUOffloadOptimizer（带有梯度卸载）的测试脚本（目前报错）
- [ ] [Sage Attention](https://github.com/thu-ml/SageAttention) （与作者合作支持反向传播，并针对 A100 进行优化）

> [!重要]
> 由于我们的目标是使脚本尽可能节省内存，因此我们不保证支持多 GPU 训练。