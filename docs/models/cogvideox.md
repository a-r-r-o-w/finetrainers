# CogVideoX

## Training

For LoRA training, specify `--training_type lora`. For full finetuning, specify `--training_type full-finetune`.

Examples available:
- [PIKA crush effect](../../examples/training/sft/cogvideox/crush_smol_lora/)

To run an example, run the following from the root directory of the repository (assuming you have installed the requirements and are using Linux/WSL):

```bash
chmod +x ./examples/training/sft/cogvideox/crush_smol_lora/train.sh
./examples/training/sft/cogvideox/crush_smol_lora/train.sh
```

On Windows, you will have to modify the script to a compatible format to run it. [TODO(aryan): improve instructions for Windows]

## Supported checkpoints

CogVideoX has multiple checkpoints as one can note [here](https://huggingface.co/collections/THUDM/cogvideo-66c08e62f1685a3ade464cce). The following checkpoints were tested with `finetrainers` and are known to be working:

* [THUDM/CogVideoX-2b](https://huggingface.co/THUDM/CogVideoX-2b)
* [THUDM/CogVideoX-5B](https://huggingface.co/THUDM/CogVideoX-5B)
* [THUDM/CogVideoX1.5-5B](https://huggingface.co/THUDM/CogVideoX1.5-5B)

## Inference

Assuming your LoRA is saved and pushed to the HF Hub, and named `my-awesome-name/my-awesome-lora`, we can now use the finetuned model for inference:

```diff
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16
).to("cuda")
+ pipe.load_lora_weights("my-awesome-name/my-awesome-lora", adapter_name="cogvideox-lora")
+ pipe.set_adapters(["cogvideox-lora"], [0.75])

video = pipe("<my-awesome-prompt>").frames[0]
export_to_video(video, "output.mp4")
```

You can refer to the following guides to know more about the model pipeline and performing LoRA inference in `diffusers`:

* [CogVideoX in Diffusers](https://huggingface.co/docs/diffusers/main/en/api/pipelines/cogvideox)
* [Load LoRAs for inference](https://huggingface.co/docs/diffusers/main/en/tutorials/using_peft_for_inference)
* [Merge LoRAs](https://huggingface.co/docs/diffusers/main/en/using-diffusers/merge_loras)