# HunyuanVideo

## Training

For LoRA training, specify `--training_type lora`. For full finetuning, specify `--training_type full-finetune`.

Examples available:
- [PIKA Dissolve effect](../../examples/training/sft/hunyuan_video/modal_labs_dissolve/)

To run an example, run the following from the root directory of the repository (assuming you have installed the requirements and are using Linux/WSL):

```bash
chmod +x ./examples/training/sft/hunyuan_video/modal_labs_dissolve/train.sh
./examples/training/sft/hunyuan_video/modal_labs_dissolve/train.sh
```

On Windows, you will have to modify the script to a compatible format to run it. [TODO(aryan): improve instructions for Windows]

## Inference

Assuming your LoRA is saved and pushed to the HF Hub, and named `my-awesome-name/my-awesome-lora`, we can now use the finetuned model for inference:

```py
import torch
from diffusers import HunyuanVideoPipeline

import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video

model_id = "hunyuanvideo-community/HunyuanVideo"
transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    model_id, subfolder="transformer", torch_dtype=torch.bfloat16
)
pipe = HunyuanVideoPipeline.from_pretrained(model_id, transformer=transformer, torch_dtype=torch.float16)
pipe.load_lora_weights("my-awesome-name/my-awesome-lora", adapter_name="hunyuanvideo-lora")
pipe.set_adapters(["hunyuanvideo-lora"], [0.6])
pipe.vae.enable_tiling()
pipe.to("cuda")

output = pipe(
    prompt="A cat walks on the grass, realistic",
    height=320,
    width=512,
    num_frames=61,
    num_inference_steps=30,
).frames[0]
export_to_video(output, "output.mp4", fps=15)
```

You can refer to the following guides to know more about the model pipeline and performing LoRA inference in `diffusers`:

- [Hunyuan-Video in Diffusers](https://huggingface.co/docs/diffusers/main/api/pipelines/hunyuan_video)
- [Load LoRAs for inference](https://huggingface.co/docs/diffusers/main/en/tutorials/using_peft_for_inference)
- [Merge LoRAs](https://huggingface.co/docs/diffusers/main/en/using-diffusers/merge_loras)