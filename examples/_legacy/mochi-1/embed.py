"""
Adapted from:
https://github.com/genmoai/mochi/blob/main/demos/fine_tuner/encode_videos.py
https://github.com/genmoai/mochi/blob/main/demos/fine_tuner/embed_captions.py
"""

from pathlib import Path

import click
import torch
import torchvision
from diffusers import AutoencoderKLMochi, MochiPipeline
from tqdm.auto import tqdm
from transformers import T5EncoderModel, T5Tokenizer


def encode_videos(model: torch.nn.Module, vid_path: Path, shape: str):
    T, H, W = [int(s) for s in shape.split("x")]
    assert (T - 1) % 6 == 0, "Expected T to be 1 mod 6"
    video, _, metadata = torchvision.io.read_video(str(vid_path), output_format="THWC", pts_unit="secs")
    fps = metadata["video_fps"]
    video = video.permute(3, 0, 1, 2)
    og_shape = video.shape
    assert video.shape[2] == H, f"Expected {vid_path} to have height {H}, got {video.shape}"
    assert video.shape[3] == W, f"Expected {vid_path} to have width {W}, got {video.shape}"
    assert video.shape[1] >= T, f"Expected {vid_path} to have at least {T} frames, got {video.shape}"
    if video.shape[1] > T:
        video = video[:, :T]
        print(f"Trimmed video from {og_shape[1]} to first {T} frames")
    video = video.unsqueeze(0)
    video = video.float() / 127.5 - 1.0
    video = video.to(model.device)

    assert video.ndim == 5

    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ldist = model._encode(video)

        torch.save(dict(ldist=ldist), vid_path.with_suffix(".latent.pt"))


@click.command()
@click.argument("output_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path))
@click.option(
    "--model_id",
    type=str,
    help="Repo id. Should be genmo/mochi-1-preview",
    default="genmo/mochi-1-preview",
)
@click.option("--shape", default="163x480x848", help="Shape of the video to encode")
@click.option("--overwrite", "-ow", is_flag=True, help="Overwrite existing latents and caption embeddings.")
def batch_process(output_dir: Path, model_id: Path, shape: str, overwrite: bool) -> None:
    """Process all videos and captions in a directory using a single GPU."""
    # comment out when running on unsupported hardware
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Get all video paths
    video_paths = list(output_dir.glob("**/*.mp4"))
    if not video_paths:
        print(f"No MP4 files found in {output_dir}")
        return

    text_paths = list(output_dir.glob("**/*.txt"))
    if not text_paths:
        print(f"No text files found in {output_dir}")
        return

    # load the models
    vae = AutoencoderKLMochi.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32).to("cuda")
    text_encoder = T5EncoderModel.from_pretrained(model_id, subfolder="text_encoder")
    tokenizer = T5Tokenizer.from_pretrained(model_id, subfolder="tokenizer")
    pipeline = MochiPipeline.from_pretrained(
        model_id, text_encoder=text_encoder, tokenizer=tokenizer, transformer=None, vae=None
    ).to("cuda")

    for idx, video_path in tqdm(enumerate(sorted(video_paths))):
        print(f"Processing {video_path}")
        try:
            if video_path.with_suffix(".latent.pt").exists() and not overwrite:
                print(f"Skipping {video_path}")
                continue

            # encode videos.
            encode_videos(vae, vid_path=video_path, shape=shape)

            # embed captions.
            prompt_path = Path("/".join(str(video_path).split(".")[:-1]) + ".txt")
            embed_path = prompt_path.with_suffix(".embed.pt")

            if embed_path.exists() and not overwrite:
                print(f"Skipping {prompt_path} - embeddings already exist")
                continue

            with open(prompt_path) as f:
                text = f.read().strip()
            with torch.inference_mode():
                conditioning = pipeline.encode_prompt(prompt=[text])

            conditioning = {"prompt_embeds": conditioning[0], "prompt_attention_mask": conditioning[1]}
            torch.save(conditioning, embed_path)

        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"Error processing {video_path}: {str(e)}")


if __name__ == "__main__":
    batch_process()
