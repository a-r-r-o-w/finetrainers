# Wan

## Training

For LoRA training, specify `--training_type lora`. For full finetuning, specify `--training_type full-finetune`.

Examples available:
- [PIKA crush effect](../../examples/training/sft/wan/crush_smol_lora/)
- [3DGS dissolve](../../examples/training/sft/wan/3dgs_dissolve/)

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

You can refer to the following guides to know more about the model pipeline and performing LoRA inference in `diffusers`:

* [Wan in Diffusers](https://huggingface.co/docs/diffusers/main/en/api/pipelines/wan)
* [Load LoRAs for inference](https://huggingface.co/docs/diffusers/main/en/tutorials/using_peft_for_inference)
* [Merge LoRAs](https://huggingface.co/docs/diffusers/main/en/using-diffusers/merge_loras)
