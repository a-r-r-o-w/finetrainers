# Dataset

## Dataset preparation

Please refer to [video-dataset-scripts](https://github.com/huggingface/video-dataset-scripts) for a collection of scripts to prepare datasets for training. The scripts are designed to work with the HF datasets library and can be used to prepare datasets for training with `finetrainers`.

## Training Dataset Format

Dataset loading format support is very limited at the moment. This will be improved in the future. For now, we support the following formats:

#### Two file format

> [!NOTE]
> Relevant classes to look for implementation:
> - ImageFileCaptionFileListDataset
> - VideoFileCaptionFileListDataset
>
> Supports loading directly from the HF Hub.

Your dataset structure should look like this. Running the `tree` command, you should see something similar to:

```
dataset
├── prompt.txt
├── videos.txt
├── videos
    ├── 00000.mp4
    ├── 00001.mp4
    ├── ...
```

- Make sure that the paths in `videos.txt` is relative to the `dataset` directory. The `prompt.txt` should contain the captions for the videos in the same order as the videos in `videos.txt`.
- Supported names for caption file: `captions.txt`, `caption.txt`, `prompt.txt`, `prompts.txt` (feel free to send PRs to add more common names).
- Supported names for video file: `videos.txt`, `video.txt`, (feel free to send PRs to add more common names).

#### Caption-Data filename pair format

> [!NOTE]
> Relevant classes to look for implementation:
> - ImageCaptionFilePairDataset
> - VideoCaptionFilePairDataset
>
> Does not support loading directly from the HF Hub.

Your dataset structure should look like this. Running the `tree` command, you should see something similar to:

```
dataset
├── a.txt
├── a.mp4
├── bkjlaskdjg.txt
├── bkjlaskdjg.mp4
├── ...
```

- Each caption file should have a corresponding image/video file with the same name.

#### CSV/JSON/JSONL format

> [!NOTE]
> Relevant classes to look for implementation:
> - ImageFolderDataset
> - VideoFolderDataset

Any dataset loadable via the [🤗 HF datasets] directly should work (not widely tested at the moment):
- https://huggingface.co/docs/datasets/v3.3.2/en/image_load#webdataset
- https://huggingface.co/docs/datasets/v3.3.2/en/video_load#webdataset

#### Webdataset format

> [!NOTE]
> Relevant classes to look for implementation:
> - ImageWebDataset
> - VideoWebDataset

Any dataset loadable via the [🤗 HF datasets] directly should work (not widely tested at the moment). We support the [`webdataset`](https://huggingface.co/docs/datasets/v3.3.2/en/image_dataset#webdataset) and [`webdataset`](https://huggingface.co/docs/datasets/v3.3.2/en/video_dataset#webdataset) formats.

## Validation Dataset Format

Arguments related to validation are:
- `--validation_dataset_file`: Path to the validation dataset file. Supported formats are CSV, JSON, JSONL, PARQUET, and ARROW. Note: PARQUET and ARROW have not been tested after a major refactor, but should most likely work. (TODO(aryan): look into this)
- `--validation_steps`: Interval of training steps after which validation should be performed.
- `--enable_model_cpu_offload`: If set, CPU offloading will be enabled during validation. Note that this has not been tested for FSDP, TP, or DDP after a major refactor, but should most likely work for single GPU training,

> [!IMPORTANT]
>
> When using `dp_shards > 1` or `tp_degree > 1`, you must make sure that the number of data samples contained is a multiple of `dp_shards * tp_degree`. If this is not the case, the training will fail due to a NCCL timeout. This will be improved/fixed in the future.

- Must contain "caption" as a column. If an image must be provided for validation (for example, image-to-video inference), then the "image_path" field must be provided. If a video must be provided for validation (for example, video-to-video inference), then the "video_path" field must be provided. Other fields like "num_inference_steps", "height", "width", "num_frames", and "frame_rate" can be provided too but are optional.

#### CSV Example

<details>
<summary>Click to expand</summary>

```csv
caption,image_path,video_path,num_inference_steps,height,width,num_frames,frame_rate
"A black and white animated scene unfolds with an anthropomorphic goat surrounded by musical notes and symbols, suggesting a playful environment. Mickey Mouse appears, leaning forward in curiosity as the goat remains still. The goat then engages with Mickey, who bends down to converse or react. The dynamics shift as Mickey grabs the goat, potentially in surprise or playfulness, amidst a minimalistic background. The scene captures the evolving relationship between the two characters in a whimsical, animated setting, emphasizing their interactions and emotions.",,"/raid/aryan/finetrainers-dummy-dataset-disney/a3c275fc2eb0a67168a7c58a6a9adb14.mp4",50,480,768,49,25
"<SECOND_CAPTION>",,"/path/to/second.mp4",50,512,704,161,25
```

</details>

#### JSON Example

Must contain "data" field, which should be a list of dictionaries. Each dictionary corresponds to one validation video that will be generated with the selected configuration of generation parameters.

<details>
<summary>Click to expand</summary>

```json
{
  "data": [
    {
      "caption": "A black and white animated scene unfolds with an anthropomorphic goat surrounded by musical notes and symbols, suggesting a playful environment. Mickey Mouse appears, leaning forward in curiosity as the goat remains still. The goat then engages with Mickey, who bends down to converse or react. The dynamics shift as Mickey grabs the goat, potentially in surprise or playfulness, amidst a minimalistic background. The scene captures the evolving relationship between the two characters in a whimsical, animated setting, emphasizing their interactions and emotions.",
      "image_path": "",
      "video_path": "/raid/aryan/finetrainers-dummy-dataset-disney/a3c275fc2eb0a67168a7c58a6a9adb14.mp4",
      "num_inference_steps": 50,
      "height": 480,
      "width": 768,
      "num_frames": 49,
      "frame_rate": 25
    },
    {
      "caption": "<SECOND_CAPTION>",
      "image_path": "",
      "video_path": "/path/to/second.mp4",
      "num_inference_steps": 50,
      "height": 512,
      "width": 704,
      "num_frames": 161,
      "frame_rate": 25
    }
  ]
}
```

</details>

## Understanding how datasets are loaded

For memory efficient training, it is important to precompute conditional and latent embeddings. If this is not done, we will need to keep the conditioning models in memory, which can be memory intensive. To avoid this, we implement some abstractions that allow us to do the following efficiently:
- Loading datasets
- Chaining multiple datasets together
- Splitting datasets across data replicas
- Preprocessing datasets to user-configured resolution buckets
- Precomputing embeddings without exhaustively using too much disk space

The following is a high-level overview of how datasets are loaded and preprocessed:

- Initially, the dataset is lazy loaded using the HF `datasets` library. Every dataset is loaded in streaming and infinite mode. This means that the dataset will be loaded indefinitely until some end conditions (e.g. user-configured training steps is completed). Users can chain together multiple datasets too! For example, if you only have high resolution data available, but want to perform multi-resolution training at certain lower resolutions too, you would have to perform the resizing manually and chain the data together. Finetrainers makes this easier by allowing you to specify multiple different, or same, datasets with different resolutions.
- The dataset is split across data replicas (GPUs groups that perform data parallelism). Each data replica will have a non-overlapping subset of the overall dataset.
- If multiple datasets have been provided, they will be chained together. Shuffling can also be done to ensure better dataset regularization. This is done by shuffling the iterable datasets in a buffer of user-configured `--dataset_shuffle_buffer_size`. For small datasets, it is recommended to not shuffle and use the default value of `1`. For larger datasets, there is a significant overhead the higher this value is set to, so it is recommended to keep it low (< 1000) [this is because we store the data in memory in a not-so-clever way yet].
- The dataset is preprocessed to the user-configured resolution buckets. This is done by resizing the images/videos to the specified resolution buckets. This is also necessary for collation when using batch_size > 1.
- The dataset is precomputed for embeddings and stored to disk. This is done in batches of user-configured `--precompute_batch_size`. This is done to avoid exhausting disk space. The smaller this value, the more number of times conditioning models will be loaded upon precomputation exhaustion. The larger this value, the more disk space will be used.
- When data points are required for training, they are loaded from disk on the main process and dispatched to data replicas. [TODO: this needs some improvements to speedup training eventually]

## Understanding how datasets are precomputed

There are 3 arguments related to precomputation:
- `--precomputation_items`: The number of data points to precompute and store to disk at a time. This is useful for performing memory-efficient training without exhausting disk space by precomputing embeddings of the entire dataset(s) at once. We default to `512` data points, but configure this to a lower value for smaller datasets. As training progresses, the precomputed data will be read from disk and dispatched to data replicas. Once all precomputed data has been used, the next batch of data points will be precomputed and stored to disk in a rolling fashion.
- `--precomputation_dir`: The directory where precomputed data will be stored. This is useful for resuming training from a checkpoint, as the precomputed data will be loaded from this directory. If this directory is not provided, the precomputed data will be stored in the `--output_dir/precomputed`.
- `--precomputation_once`: If you're working with small datasets and want to precompute all embeddings at once, set this flag. This will allow you to train without having to compute embeddings every time the precomputed data is exhausted. Currently, `webdataset` format loading does not support this feature, and it is also disabled for `> 1024` data points due to hard coded logic (can be removed manually by users for now).

Batching is not yet supported for precomputation. This will be added in the future.
