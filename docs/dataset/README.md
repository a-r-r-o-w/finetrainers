# Dataset

TODO(aryan): requires rewrite. Also need to handle `caption_column` correctly. Need ot mention that `file_name` is a must for CSV/JSON/JSONL formats for `datasets` to be able to map correctly.

## Training Dataset Format

Dataset loading format support is very limited at the moment. This will be improved in the future. For now, we support the following formats:

#### Two file format

Your dataset structure should look like this. Running the `tree` command, you should see:

```
dataset
├── prompt.txt
├── videos.txt
├── videos
    ├── 00000.mp4
    ├── 00001.mp4
    ├── ...
```

For this format, you would specify arguments as follows:

```
--data_root /path/to/dataset --caption_column prompt.txt --video_column videos.txt
```

#### CSV format

```
dataset
├── dataset.csv
├── videos
    ├── 00000.mp4
    ├── 00001.mp4
    ├── ...
```

The CSV can contain any number of columns, but due to limited support at the moment, we only make use of prompt and video columns. The CSV should look like this:

```
caption,video_file,other_column1,other_column2
A black and white animated sequence featuring a rabbit, named Rabbity Ribfried, and an anthropomorphic goat in a musical, playful environment, showcasing their evolving interaction.,videos/00000.mp4,...,...
```

For this format, you would specify arguments as follows:

```
--data_root /path/to/dataset --caption_column caption --video_column video_file
```

### JSON format

```
dataset
├── dataset.json
├── videos
    ├── 00000.mp4
    ├── 00001.mp4
    ├── ...
```

The JSON can contain any number of attributes, but due to limited support at the moment, we only make use of prompt and video columns. The JSON should look like this:

```json
[
    {
        "short_prompt": "A black and white animated sequence featuring a rabbit, named Rabbity Ribfried, and an anthropomorphic goat in a musical, playful environment, showcasing their evolving interaction.",
        "filename": "videos/00000.mp4"
    }
]
```

For this format, you would specify arguments as follows:

```
--data_root /path/to/dataset --caption_column short_prompt --video_column filename
```

### JSONL format

```
dataset
├── dataset.jsonl
├── videos
    ├── 00000.mp4
    ├── 00001.mp4
    ├── ...
```

The JSONL can contain any number of attributes, but due to limited support at the moment, we only make use of prompt and video columns. The JSONL should look like this:

```json
{"llm_prompt": "A black and white animated sequence featuring a rabbit, named Rabbity Ribfried, and an anthropomorphic goat in a musical, playful environment, showcasing their evolving interaction.", "filename": "videos/00000.mp4"}
{"llm_prompt": "A black and white animated sequence on a ship’s deck features a bulldog character, named Bully Bulldoger, showcasing exaggerated facial expressions and body language.", "filename": "videos/00001.mp4"}
...
```

For this format, you would specify arguments as follows:

```
--data_root /path/to/dataset --caption_column llm_prompt --video_column filename
```

> ![NOTE]
> Using images for finetuning is also supported. The dataset format remains the same as above. Find an example [here](https://huggingface.co/datasets/a-r-r-o-w/flux-retrostyle-dataset-mini).
>
> For example, to finetune with `512x512` resolution images, one must specify `--video_resolution_buckets 1x512x512` and point to the image files correctly.

If you are using LLM-captioned videos, it is common to see many unwanted starting phrases like "In this video, ...", "This video features ...", etc. To remove a simple subset of these phrases, you can specify `--remove_common_llm_caption_prefixes` when starting training.

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
