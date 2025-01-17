import torch
from diffusers import LTXPipeline
from diffusers import LTXVideoTransformer3DModel
from diffusers.utils import export_to_video
from diffusers.utils import testing_utils
testing_utils.enable_full_determinism()

transformer = LTXVideoTransformer3DModel.from_pretrained("/home/ubuntu/upstream/finetrainers/ltx-video/ltxv_strip/checkpoint-38000/transformer/",torch_dtype=torch.bfloat16).to("cuda")
pipe = LTXPipeline.from_pretrained(
    "Lightricks/LTX-Video", torch_dtype=torch.bfloat16,
    # transformer=transformer
).to("cuda")

prompt = "A woman semi nude in red lingerie with black straps and garter belts walks through a modern, minimalist interior featuring a white sofa, black chairs, and a glass partition. Her confident and alluring demeanor is highlighted by the soft lighting and the room's neutral tones. As she strides forward, her bare feet touch the carpet, and her blonde hair cascades over her shoulders. The setting includes a sleek black dining table and a bookshelf filled with books and decorative items, creating an atmosphere of sophistication and intimacy."

g = torch.Generator(device="cuda")
g.manual_seed(42)

for seconds in range(3, 8):
    num_frames = seconds * 24

    result = pipe(
        prompt=prompt,
        width=768,
        height=512,
        num_frames=num_frames,
        num_inference_steps=50,
        generator=g
    )
    video = result.frames[0]
    export_to_video(video, f"out_{seconds}.mp4", fps=24)


# pipe.load_lora_weights("/home/ubuntu/finetrainers/ltx-video/ltxv_strip/checkpoint-10000/pytorch_lora_weights.safetensors", adapter_name="ltxv-lora")
# pipe.set_adapters(["ltxv-lora"], [1.0])
