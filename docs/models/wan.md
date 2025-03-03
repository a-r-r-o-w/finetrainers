# Wan

## Training

For LoRA training, specify `--training_type lora`. For full finetuning, specify `--training_type full-finetune`.

See [this](../../examples/training/sft/wan/crush_smol_lora/) example training script for training Wan with Pika Effects Crush.

## Memory Usage

TODO

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
