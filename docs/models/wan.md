# Wan

## Training

For LoRA training, specify `--training_type lora`. For full finetuning, specify `--training_type full-finetune`.

Examples available:
- [PIKA crush effect](../../examples/training/sft/wan/crush_smol_lora/)
- [3DGS dissolve](../../examples/training/sft/wan/3dgs_dissolve/)
- [I2V conditioning](../../examples/training/control/wan/image_condition/)

To run an example, run the following from the root directory of the repository (assuming you have installed the requirements and are using Linux/WSL):

```bash
chmod +x ./examples/training/sft/wan/crush_smol_lora/train.sh
./examples/training/sft/wan/crush_smol_lora/train.sh
```

On Windows, you will have to modify the script to a compatible format to run it. [TODO(aryan): improve instructions for Windows]

## Inference

Assuming your LoRA is saved and pushed to the HF Hub, and named `my-awesome-name/my-awesome-lora`, we can now use the finetuned model for inference:

```diff
import torch
from diffusers import WanPipeline
from diffusers.utils import export_to_video

pipe = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", torch_dtype=torch.bfloat16
).to("cuda")
+ pipe.load_lora_weights("my-awesome-name/my-awesome-lora", adapter_name="wan-lora")
+ pipe.set_adapters(["wan-lora"], [0.75])

video = pipe("<my-awesome-prompt>").frames[0]
export_to_video(video, "output.mp4", fps=8)
```

To use trained Control LoRAs, the following can be used for inference (ideally, you should raise a support request in Diffusers):

<details>
<summary> Control Lora inference </summary>

```python
import numpy as np
import torch
from diffusers import WanPipeline
from diffusers.utils import export_to_video, load_video
from finetrainers.trainer.control_trainer.data import apply_frame_conditioning_on_latents
from finetrainers.models.utils import _expand_conv3d_with_zeroed_weights
from finetrainers.patches import load_lora_weights
from finetrainers.patches.dependencies.diffusers.control import control_channel_concat

dtype = torch.bfloat16
device = torch.device("cuda")
generator = torch.Generator().manual_seed(0)

pipe = WanPipeline.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", torch_dtype=dtype).to(device)

in_channels = pipe.transformer.config.in_channels
patch_channels = pipe.transformer.patch_embedding.in_channels
pipe.transformer.patch_embedding = _expand_conv3d_with_zeroed_weights(pipe.transformer.patch_embedding, new_in_channels=2 * patch_channels)

load_lora_weights(pipe, "/raid/aryan/wan-control-image-condition", "wan-lora")
pipe.to(device)

prompt = "The video shows a vibrant green Mustang GT parked in a parking lot. The car is positioned at an angle, showcasing its sleek design and black rims. The car's hood is black, contrasting with the green body. The Mustang GT logo is visible on the side of the car. The parking lot appears to be empty, with the car being the main focus of the video. The car's position and the absence of other vehicles suggest that the video might be a promotional or showcase video for the Mustang GT. The overall style of the video is simple and straightforward, focusing on the car and its design."
control_video = load_video("examples/training/control/wan/image_condition/validation_dataset/0.mp4")
height, width, num_frames = 480, 704, 49

# Take evenly space `num_frames` frames from the control video
indices = np.linspace(0, len(control_video) - 1, num_frames).astype(int)
control_video = [control_video[i] for i in indices]

with torch.no_grad():
    latents = pipe.prepare_latents(1, in_channels, height, width, num_frames, dtype, device, generator)
    latents_mean = torch.tensor(pipe.vae.config.latents_mean).view(1, -1, 1, 1, 1).to(latents)
    latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, -1, 1, 1, 1).to(latents)
    control_video = pipe.video_processor.preprocess_video(control_video, height=height, width=width)
    control_video = control_video.to(device=device, dtype=dtype)
    control_latents = pipe.vae.encode(control_video).latent_dist.sample(generator=generator)
    control_latents = ((control_latents.float() - latents_mean) * latents_std).to(dtype)
    control_latents = apply_frame_conditioning_on_latents(
        control_latents,
        expected_num_frames=latents.size(2),
        channel_dim=1,
        frame_dim=2,
        frame_conditioning_type="index",
        frame_conditioning_index=0,
        concatenate_mask=False,
    )

with control_channel_concat(pipe.transformer, ["hidden_states"], [control_latents], dims=[1]):
    video = pipe(prompt, latents=latents, num_inference_steps=30, generator=generator).frames[0]

export_to_video(video, "output.mp4", fps=16)
```
</details>

You can refer to the following guides to know more about the model pipeline and performing LoRA inference in `diffusers`:

- [Wan in Diffusers](https://huggingface.co/docs/diffusers/main/en/api/pipelines/wan)
- [Load LoRAs for inference](https://huggingface.co/docs/diffusers/main/en/tutorials/using_peft_for_inference)
- [Merge LoRAs](https://huggingface.co/docs/diffusers/main/en/using-diffusers/merge_loras)
