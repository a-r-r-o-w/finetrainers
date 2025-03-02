# finetrainers 🧪

`cogvideox-factory` was renamed to `finetrainers`. If you're looking to train CogVideoX or Mochi with the legacy training scripts, please refer to [this](./training/README.md) README instead. Everything in the `training/` directory will be eventually moved and supported under `finetrainers`.

FineTrainers is a work-in-progress library to support (accessible) training of video models. Our first priority is to support LoRA training for all popular video models in [Diffusers](https://github.com/huggingface/diffusers), and eventually other methods like controlnets, control-loras, distillation, etc.

<table align="center">
<tr>
  <td align="center"><video src="https://github.com/user-attachments/assets/aad07161-87cb-4784-9e6b-16d06581e3e5">Your browser does not support the video tag.</video></td>
</tr>
</table>

## News

- 🔥 **2024-01-15**: Support for naive FP8 weight-casting training added! This allows training HunyuanVideo in under 24 GB upto specific resolutions.
- 🔥 **2024-01-13**: Support for T2V full-finetuning added! Thanks to @ArEnSc for taking up the initiative!
- 🔥 **2024-01-03**: Support for T2V LoRA finetuning of [CogVideoX](https://huggingface.co/docs/diffusers/main/api/pipelines/cogvideox) added! 
- 🔥 **2024-12-20**: Support for T2V LoRA finetuning of [Hunyuan Video](https://huggingface.co/docs/diffusers/main/api/pipelines/hunyuan_video) added! We would like to thank @SHYuanBest for his work on a training script [here](https://github.com/huggingface/diffusers/pull/10254).
- 🔥 **2024-12-18**: Support for T2V LoRA finetuning of [LTX Video](https://huggingface.co/docs/diffusers/main/api/pipelines/ltx_video) added!

## Table of Contents

* [Quickstart](#quickstart)
* [Support Matrix](#support-matrix)
* [Acknowledgements](#acknowledgements)

## Quickstart

Clone the repository and make sure the requirements are installed: `pip install -r requirements.txt` and install `diffusers` from source by `pip install git+https://github.com/huggingface/diffusers`. The requirements specify `diffusers>=0.32.1`, but it is always recommended to use the `main` branch of Diffusers for the latest features and bugfixes. Note that the `main` branch for `finetrainers` is also the development branch, and stable support should be expected from the release tags.

Checkout to the latest release tag:

```bash
git fetch --all --tags
git checkout tags/v0.0.1
```

Follow the instructions mentioned in the [README](https://github.com/a-r-r-o-w/finetrainers/tree/v0.0.1) for the release tag.

To get started quickly with example training scripts on the main development branch, refer to the following:
- [LTX-Video Pika Effects Crush](./examples/training/ltx_video/)

The following are some simple datasets/HF orgs with good datasets to test training with quickly:
- [Disney Video Generation Dataset](https://huggingface.co/datasets/Wild-Heart/Disney-VideoGeneration-Dataset)
- [bigdatapw's Video Dataset Collection](https://huggingface.co/bigdata-pw)
- [Finetrainers HF Dataset Collection](https://huggingface.co/finetrainers)

Please checkout [`docs/models`](./docs/models/) and [`examples/training`](./examples/training/) to learn more about supported models for training & example reproducible training launch scripts.

> [!IMPORTANT] 
> It is recommended to use Pytorch 2.5.1 or above for training. Previous versions can lead to completely black videos, OOM errors, or other issues and are not tested.

## Support Matrix

> [!NOTE]
> The following numbers were obtained from the [release branch](https://github.com/a-r-r-o-w/finetrainers/tree/v0.0.1). The `main` branch is unstable at the moment and may use higher memory.

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

* `finetrainers` builds on top of & takes inspiration from great open-source libraries - `transformers`, `accelerate`, `torchtune`, `torchtitan`, `peft`, `diffusers`, `bitsandbytes`, `torchao` and `deepspeed` - to name a few.
* Some of the design choices of `finetrainers` were inspired by [`SimpleTuner`](https://github.com/bghira/SimpleTuner).
