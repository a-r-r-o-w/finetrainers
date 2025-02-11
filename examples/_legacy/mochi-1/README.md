# Simple Mochi-1 finetuner 

<table align=center>
<tr>
<th align=center> Dataset Sample </th>
<th align=center> Test Sample </th>
</tr>
<tr>
  <td align=center><video src="https://github.com/user-attachments/assets/6f906a32-b169-493f-a713-07679e87cd91"> Your browser does not support the video tag. </video></td>
  <td align=center><video src="https://github.com/user-attachments/assets/d356e70f-ccf4-47f7-be1d-8d21108d8a84"> Your browser does not support the video tag. </video></td>
</tr>
</table>

Now you can make Mochi-1 your own with `diffusers`, too 🤗 🧨

We provide a minimal and faithful reimplementation of the [Mochi-1 original fine-tuner](https://github.com/genmoai/mochi/tree/aba74c1b5e0755b1fa3343d9e4bd22e89de77ab1/demos/fine_tuner). As usual, we leverage `peft` for things LoRA in our implementation. 

**Updates**

December 1 2024: Support for checkpoint saving and loading.

## Getting started

Install the dependencies: `pip install -r requirements.txt`. Also make sure your `diffusers` installation is from the current `main`. 

Download a demo dataset:

```bash
huggingface-cli download \
  --repo-type dataset sayakpaul/video-dataset-disney-organized \
  --local-dir video-dataset-disney-organized
```

The dataset follows the directory structure expected by the subsequent scripts. In particular, it follows what's prescribed [here](https://github.com/genmoai/mochi/tree/main/demos/fine_tuner#1-collect-your-videos-and-captions):

```bash
video_1.mp4
video_1.txt -- One-paragraph description of video_1
video_2.mp4
video_2.txt -- One-paragraph description of video_2
...
```

Then run (be sure to check the paths accordingly):

```bash
bash prepare_dataset.sh
```

We can adjust `num_frames` and `resolution`. By default, in `prepare_dataset.sh`, we use `--force_upsample`. This means if the original video resolution is smaller than the requested resolution, we will upsample the video.

> [!IMPORTANT]  
> It's important to have a resolution of at least 480x848 to satisy Mochi-1's requirements.

Now, we're ready to fine-tune. To launch, run:

```bash
bash train.sh
```

You can disable intermediate validation by:

```diff
- --validation_prompt "..." \
- --validation_prompt_separator ::: \
- --num_validation_videos 1 \
- --validation_epochs 1 \
```

We haven't rigorously tested but without validation enabled, this script should run under 40GBs of GPU VRAM.

To use the LoRA checkpoint:

```py
from diffusers import MochiPipeline
from diffusers.utils import export_to_video
import torch 

pipe = MochiPipeline.from_pretrained("genmo/mochi-1-preview")
pipe.load_lora_weights("path-to-lora")
pipe.enable_model_cpu_offload()

pipeline_args = {
    "prompt": "A black and white animated scene unfolds with an anthropomorphic goat surrounded by musical notes and symbols, suggesting a playful environment. Mickey Mouse appears, leaning forward in curiosity as the goat remains still. The goat then engages with Mickey, who bends down to converse or react. The dynamics shift as Mickey grabs the goat, potentially in surprise or playfulness, amidst a minimalistic background. The scene captures the evolving relationship between the two characters in a whimsical, animated setting, emphasizing their interactions and emotions",
    "guidance_scale": 6.0,
    "num_inference_steps": 64,
    "height": 480,
    "width": 848,
    "max_sequence_length": 256,
    "output_type": "np",
}

with torch.autocast("cuda", torch.bfloat16)
    video = pipe(**pipeline_args).frames[0]
export_to_video(video)
```

## Known limitations

(Contributions are welcome 🤗)

Our script currently doesn't leverage `accelerate` and some of its consequences are detailed below:

* No support for distributed training. 
* `train_batch_size > 1` are supported but can potentially lead to OOMs because we currently don't have gradient accumulation support.
* No support for 8bit optimizers (but should be relatively easy to add).

**Misc**: 

* We're aware of the quality issues in the `diffusers` implementation of Mochi-1. This is being fixed in [this PR](https://github.com/huggingface/diffusers/pull/10033). 
* `embed.py` script is non-batched. 
