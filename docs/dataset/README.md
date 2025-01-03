## Dataset Format

### Prompt Dataset Requirements

Create a `prompt.txt` file, which should contain prompts separated by lines. Please note that the prompts must be in English, and it is recommended to use the [prompt refinement script](https://github.com/THUDM/CogVideo/blob/main/inference/convert_demo.py) for better prompts. Alternatively, you can use [CogVideo-caption](https://huggingface.co/THUDM/cogvlm2-llama3-caption) for data annotation:

```
A black and white animated sequence featuring a rabbit, named Rabbity Ribfried, and an anthropomorphic goat in a musical, playful environment, showcasing their evolving interaction.
A black and white animated sequence on a ship’s deck features a bulldog character, named Bully Bulldoger, showcasing exaggerated facial expressions and body language...
...
```

### Video Dataset Requirements

The framework supports resolutions and frame counts that meet the following conditions:

- **Supported Resolutions (Width * Height)**:
    - Any resolution as long as it is divisible by 32. For example, `720 * 480`, `1920 * 1020`, etc.

- **Supported Frame Counts (Frames)**:
    - Must be `4 * k` or `4 * k + 1` (example: 16, 32, 49, 81)

It is recommended to place all videos in a single folder.

Next, create a `videos.txt` file. The `videos.txt` file should contain the video file paths, separated by lines. Please note that the paths must be relative to the `--data_root` directory. The format is as follows:

```
videos/00000.mp4
videos/00001.mp4
...
```

For developers interested in more details, you can refer to the relevant `BucketSampler` code.

### Dataset Structure

Your dataset structure should look like this. Running the `tree` command, you should see:

```
dataset
├── prompt.txt
├── videos.txt
├── videos
    ├── videos/00000.mp4
    ├── videos/00001.mp4
    ├── ...
```

### Using the Dataset

When using this format, the `--caption_column` should be set to `prompt.txt`, and the `--video_column` should be set to `videos.txt`. If your data is stored in a CSV file, you can also specify `--dataset_file` as the path to the CSV file, with `--caption_column` and `--video_column` set to the actual column names in the CSV. Please refer to the [test_dataset](../tests/test_dataset.py) file for some simple examples.

For instance, you can fine-tune using [this](https://huggingface.co/datasets/Wild-Heart/Disney-VideoGeneration-Dataset) Disney dataset. The download can be done via the 🤗 Hugging Face CLI:

```
huggingface-cli download --repo-type dataset Wild-Heart/Disney-VideoGeneration-Dataset --local-dir video-dataset-disney
```

This dataset has been prepared in the expected format and can be used directly. However, directly using the video dataset may cause Out of Memory (OOM) issues on GPUs with smaller VRAM because it requires loading the [VAE](https://huggingface.co/THUDM/CogVideoX-5b/tree/main/vae) (which encodes videos into latent space) and the large [T5-XXL](https://huggingface.co/google/t5-v1_1-xxl/) text encoder. To reduce memory usage, you can use the `training/prepare_dataset.py` script to precompute latents and embeddings.

Fill or modify the parameters in `prepare_dataset.sh` and execute it to get precomputed latents and embeddings (make sure to specify `--save_latents_and_embeddings` to save the precomputed artifacts). If preparing for image-to-video training, make sure to pass `--save_image_latents`, which encodes and saves image latents along with videos. When using these artifacts during training, ensure that you specify the `--load_tensors` flag, or else the videos will be used directly, requiring the text encoder and VAE to be loaded. The script also supports PyTorch DDP so that large datasets can be encoded in parallel across multiple GPUs (modify the `NUM_GPUS` parameter).
