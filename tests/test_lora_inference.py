"""
Run this test in Lora adpater checking:

```shell
python3 test_lora_inference.py --prompt "A girl is ridding a bike." --model_path "THUDM/CogVideoX-5B" --lora_path "path/to/lora" --lora_name "lora_adapter" --output_file "output.mp4" --fps 8
```

"""

import argparse

import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video


def generate_video(model_path, prompt, lora_path, lora_name, output_file, fps):
    pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16).to("cuda")
    pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name=lora_name)
    pipe.set_adapters([lora_name], [1.0])
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    video = pipe(prompt=prompt).frames[0]
    export_to_video(video, output_file, fps=fps)


def main():
    parser = argparse.ArgumentParser(description="Generate video using CogVideoX and LoRA weights")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for the video generation")
    parser.add_argument("--model_path", type=str, default="THUDM/CogVideoX-5B", help="Base Model path or HF ID")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to the LoRA weights")
    parser.add_argument("--lora_name", type=str, default="lora_adapter", help="Name of the LoRA adapter")
    parser.add_argument("--output_file", type=str, default="output.mp4", help="Output video file name")
    parser.add_argument("--fps", type=int, default=8, help="Frames per second for the output video")

    args = parser.parse_args()

    generate_video(args.prompt, args.lora_path, args.lora_name, args.output_file, args.fps)


if __name__ == "__main__":
    main()
