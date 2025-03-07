# CogView4

## Training

For LoRA training, specify `--training_type lora`. For full finetuning, specify `--training_type full-finetune`.

Examples available:
- [Raider White Tarot cards style](../../examples/training/sft/cogview4/raider_white_tarot/)

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

You can refer to the following guides to know more about the model pipeline and performing LoRA inference in `diffusers`:

- [CogView4 in Diffusers](https://huggingface.co/docs/diffusers/main/en/api/pipelines/cogview4)
- [Load LoRAs for inference](https://huggingface.co/docs/diffusers/main/en/tutorials/using_peft_for_inference)
- [Merge LoRAs](https://huggingface.co/docs/diffusers/main/en/using-diffusers/merge_loras)
