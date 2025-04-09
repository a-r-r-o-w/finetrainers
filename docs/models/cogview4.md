# CogView4

## Training

For LoRA training, specify `--training_type lora`. For full finetuning, specify `--training_type full-finetune`.

Examples available:
- [Raider White Tarot cards style](../../examples/training/sft/cogview4/raider_white_tarot/)
- [Omni Edit Control LoRA](../../examples/training/control/cogview4/omni_edit/)
- [Canny Control LoRA](../../examples/training/control/cogview4/canny/)

To run an example, run the following from the root directory of the repository (assuming you have installed the requirements and are using Linux/WSL):

```bash
chmod +x ./examples/training/sft/cogview4/raider_white_tarot/train.sh
./examples/training/sft/cogview4/raider_white_tarot/train.sh
```

On Windows, you will have to modify the script to a compatible format to run it. [TODO(aryan): improve instructions for Windows]

## Supported checkpoints

The following checkpoints were tested with `finetrainers` and are known to be working:

- [THUDM/CogView4-6B](https://huggingface.co/THUDM/CogView4-6B)

## Inference

Assuming your LoRA is saved and pushed to the HF Hub, and named `my-awesome-name/my-awesome-lora`, we can now use the finetuned model for inference:

```diff
import torch
from diffusers import CogView4Pipeline
from diffusers.utils import export_to_video

pipe = CogView4Pipeline.from_pretrained(
    "THUDM/CogView4-6B", torch_dtype=torch.bfloat16
).to("cuda")
+ pipe.load_lora_weights("my-awesome-name/my-awesome-lora", adapter_name="cogview4-lora")
+ pipe.set_adapters(["cogview4-lora"], [0.9])

video = pipe("<my-awesome-prompt>").frames[0]
export_to_video(video, "output.mp4")
```

To use trained Control LoRAs, the following can be used for inference (ideally, you should raise a support request in Diffusers):

<details>
<summary> Control Lora inference </summary>

```python
import torch
from diffusers import CogView4Pipeline
from diffusers.utils import load_image
from finetrainers.models.utils import _expand_linear_with_zeroed_weights
from finetrainers.patches import load_lora_weights
from finetrainers.patches.dependencies.diffusers.control import control_channel_concat

dtype = torch.bfloat16
device = torch.device("cuda")
generator = torch.Generator().manual_seed(0)

pipe = CogView4Pipeline.from_pretrained("THUDM/CogView4-6B", torch_dtype=dtype)

in_channels = pipe.transformer.config.in_channels
patch_channels = pipe.transformer.patch_embed.proj.in_features
pipe.transformer.patch_embed.proj = _expand_linear_with_zeroed_weights(pipe.transformer.patch_embed.proj, new_in_features=2 * patch_channels)

load_lora_weights(pipe, "/raid/aryan/cogview4-control-lora", "cogview4-lora")
pipe.to(device)

prompt = "Make the image look like it's from an ancient Egyptian mural."
control_image = load_image("examples/training/control/cogview4/omni_edit/validation_dataset/0.png")
height, width = 1024, 1024

with torch.no_grad():
    latents = pipe.prepare_latents(1, in_channels, height, width, dtype, device, generator)
    control_image = pipe.image_processor.preprocess(control_image, height=height, width=width)
    control_image = control_image.to(device=device, dtype=dtype)
    control_latents = pipe.vae.encode(control_image).latent_dist.sample(generator=generator)
    control_latents = (control_latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor

with control_channel_concat(pipe.transformer, ["hidden_states"], [control_latents], dims=[1]):
    image = pipe(prompt, latents=latents, num_inference_steps=30, generator=generator).images[0]

image.save("output.png")
```
</details>

You can refer to the following guides to know more about the model pipeline and performing LoRA inference in `diffusers`:

- [CogView4 in Diffusers](https://huggingface.co/docs/diffusers/main/en/api/pipelines/cogview4)
- [Load LoRAs for inference](https://huggingface.co/docs/diffusers/main/en/tutorials/using_peft_for_inference)
- [Merge LoRAs](https://huggingface.co/docs/diffusers/main/en/using-diffusers/merge_loras)
