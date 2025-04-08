# Flux

## Training

For LoRA training, specify `--training_type lora`. For full finetuning, specify `--training_type full-finetune`.

Examples available:
- [Raider White Tarot cards style](../../examples/training/sft/flux_dev/raider_white_tarot/)

To run an example, run the following from the root directory of the repository (assuming you have installed the requirements and are using Linux/WSL):

```bash
chmod +x ./examples/training/sft/flux_dev/raider_white_tarot/train.sh
./examples/training/sft/flux_dev/raider_white_tarot/train.sh
```

On Windows, you will have to modify the script to a compatible format to run it. [TODO(aryan): improve instructions for Windows]

> [!NOTE]
> Currently, only FLUX.1-dev is supported. It is a guidance-distilled model which directly predicts the outputs of its teacher model when the teacher is run with CFG. To match the output distribution of the distilled model with that of the teacher model, a guidance scale of 1.0 is hardcoded into the codebase. However, other values may work too but it is experimental.
> FLUX.1-schnell is not supported for training yet. It is a timestep-distilled model. Matching its output distribution for training is significantly more difficult.

## Supported checkpoints

The following checkpoints were tested with `finetrainers` and are known to be working:

- [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- [black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell)

## Inference

Assuming your LoRA is saved and pushed to the HF Hub, and named `my-awesome-name/my-awesome-lora`, we can now use the finetuned model for inference:

```diff
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
).to("cuda")
+ pipe.load_lora_weights("my-awesome-name/my-awesome-lora", adapter_name="flux-lora")
+ pipe.set_adapters(["flux-lora"], [0.9])

# Make sure to set guidance_scale to 0.0 when inferencing with FLUX.1-schnell or derivative models
image = pipe("<my-awesome-prompt>").images[0]
image.save("output.png")
```

You can refer to the following guides to know more about the model pipeline and performing LoRA inference in `diffusers`:

- [Flux in Diffusers](https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux)
- [Load LoRAs for inference](https://huggingface.co/docs/diffusers/main/en/tutorials/using_peft_for_inference)
- [Merge LoRAs](https://huggingface.co/docs/diffusers/main/en/using-diffusers/merge_loras)
