## Dataset Format

For a more elaborate set of tooling related curating small and high-quality video datasets for fine-tuning, refer to [this blog post](https://huggingface.co/blog/vid_ds_scripts) and [repository](https://github.com/huggingface/video-dataset-scripts).

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
