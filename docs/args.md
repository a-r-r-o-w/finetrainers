# Arguments

This document lists all the arguments that can be passed to the `train.py` script. For more information, please take a look at the `finetrainers/args.py` file.

## Table of contents

- [General arguments](#general)
- [SFT training arguments](#sft-training)
- [Control training arguments](#control-training)

## General

<!-- TODO(aryan): write a github workflow that automatically updates this page -->

```
PARALLEL ARGUMENTS
------------------
parallel_backend (`str`, defaults to `accelerate`):
    The parallel backend to use for training. Choose between ['accelerate', 'ptd'].
pp_degree (`int`, defaults to `1`):
    The degree of pipeline parallelism.
dp_degree (`int`, defaults to `1`):
    The degree of data parallelism (number of model replicas).
dp_shards (`int`, defaults to `-1`):
    The number of data parallel shards (number of model partitions).
cp_degree (`int`, defaults to `1`):
    The degree of context parallelism.

MODEL ARGUMENTS
---------------
model_name (`str`):
    Name of model to train. To get a list of models, run `python train.py --list_models`.
pretrained_model_name_or_path (`str`):
    Path to pretrained model or model identifier from https://huggingface.co/models. The model should be
    loadable based on specified `model_name`.
revision (`str`, defaults to `None`):
    If provided, the model will be loaded from a specific branch of the model repository.
variant (`str`, defaults to `None`):
    Variant of model weights to use. Some models provide weight variants, such as `fp16`, to reduce disk
    storage requirements.
cache_dir (`str`, defaults to `None`):
    The directory where the downloaded models and datasets will be stored, or loaded from.
tokenizer_id (`str`, defaults to `None`):
    Identifier for the tokenizer model. This is useful when using a different tokenizer than the default from `pretrained_model_name_or_path`.
tokenizer_2_id (`str`, defaults to `None`):
    Identifier for the second tokenizer model. This is useful when using a different tokenizer than the default from `pretrained_model_name_or_path`.
tokenizer_3_id (`str`, defaults to `None`):
    Identifier for the third tokenizer model. This is useful when using a different tokenizer than the default from `pretrained_model_name_or_path`.
text_encoder_id (`str`, defaults to `None`):
    Identifier for the text encoder model. This is useful when using a different text encoder than the default from `pretrained_model_name_or_path`.
text_encoder_2_id (`str`, defaults to `None`):
    Identifier for the second text encoder model. This is useful when using a different text encoder than the default from `pretrained_model_name_or_path`.
text_encoder_3_id (`str`, defaults to `None`):
    Identifier for the third text encoder model. This is useful when using a different text encoder than the default from `pretrained_model_name_or_path`.
transformer_id (`str`, defaults to `None`):
    Identifier for the transformer model. This is useful when using a different transformer model than the default from `pretrained_model_name_or_path`.
vae_id (`str`, defaults to `None`):
    Identifier for the VAE model. This is useful when using a different VAE model than the default from `pretrained_model_name_or_path`.
text_encoder_dtype (`torch.dtype`, defaults to `torch.bfloat16`):
    Data type for the text encoder when generating text embeddings.
text_encoder_2_dtype (`torch.dtype`, defaults to `torch.bfloat16`):
    Data type for the text encoder 2 when generating text embeddings.
text_encoder_3_dtype (`torch.dtype`, defaults to `torch.bfloat16`):
    Data type for the text encoder 3 when generating text embeddings.
transformer_dtype (`torch.dtype`, defaults to `torch.bfloat16`):
    Data type for the transformer model.
vae_dtype (`torch.dtype`, defaults to `torch.bfloat16`):
    Data type for the VAE model.
layerwise_upcasting_modules (`List[str]`, defaults to `[]`):
    Modules that should have fp8 storage weights but higher precision computation. Choose between ['transformer'].
layerwise_upcasting_storage_dtype (`torch.dtype`, defaults to `float8_e4m3fn`):
    Data type for the layerwise upcasting storage. Choose between ['float8_e4m3fn', 'float8_e5m2'].
layerwise_upcasting_skip_modules_pattern (`List[str]`, defaults to `["patch_embed", "pos_embed", "x_embedder", "context_embedder", "^proj_in$", "^proj_out$", "norm"]`):
    Modules to skip for layerwise upcasting. Layers such as normalization and modulation, when casted to fp8 precision
    naively (as done in layerwise upcasting), can lead to poorer training and inference quality. We skip these layers
    by default, and recommend adding more layers to the default list based on the model architecture.
compile_modules (`List[str]`, defaults to `[]`):
    Modules that should be regionally compiled with `torch.compile`.
compile_scopes (`str`, defaults to `None`):
    The scope of compilation for each `--compile_modules`. Choose between ['regional', 'full']. Must have the same length as
    `--compile_modules`. If `None`, will default to `regional` for all modules.

DATASET ARGUMENTS
-----------------
dataset_config (`str`):
    File to a dataset file containing information about training data. This file can contain information about one or
    more datasets in JSON format. The file must have a key called "datasets", which is a list of dictionaries. Each
    dictionary must contain the following keys:
        - "data_root": (`str`)
            The root directory containing the dataset. This parameter must be provided if `dataset_file` is not provided.
        - "dataset_file": (`str`)
            Path to a CSV/JSON/JSONL/PARQUET/ARROW/HF_HUB_DATASET file containing metadata for training. This parameter
            must be provided if `data_root` is not provided.
        - "dataset_type": (`str`)
            Type of dataset. Choose between ['image', 'video'].
        - "id_token": (`str`)
            Identifier token appended to the start of each prompt if provided. This is useful for LoRA-type training
            for single subject/concept/style training, but is not necessary.
        - "image_resolution_buckets": (`List[Tuple[int, int]]`)
            Resolution buckets for image. This should be a list of tuples containing 2 values, where each tuple
            represents the resolution (height, width). All images will be resized to the nearest bucket resolution.
            This parameter must be provided if `dataset_type` is 'image'.
        - "video_resolution_buckets": (`List[Tuple[int, int, int]]`)
            Resolution buckets for video. This should be a list of tuples containing 3 values, where each tuple
            represents the resolution (num_frames, height, width). All videos will be resized to the nearest bucket
            resolution. This parameter must be provided if `dataset_type` is 'video'.
        - "reshape_mode": (`str`)
            All input images/videos are reshaped using this mode. Choose between the following:
            ["center_crop", "random_crop", "bicubic"].
        - "remove_common_llm_caption_prefixes": (`boolean`)
            Whether or not to remove common LLM caption prefixes. See `~constants.py` for the list of common prefixes.
dataset_shuffle_buffer_size (`int`, defaults to `1`):
    The buffer size for shuffling the dataset. This is useful for shuffling the dataset before training. The default
    value of `1` means that the dataset will not be shuffled.
precomputation_items (`int`, defaults to `512`):
    Number of data samples to precompute at once for memory-efficient training. The higher this value,
    the more disk memory will be used to save the precomputed samples (conditions and latents).
precomputation_dir (`str`, defaults to `None`):
    The directory where the precomputed samples will be stored. If not provided, the precomputed samples
    will be stored in a temporary directory of the output directory.
precomputation_once (`bool`, defaults to `False`):
    Precompute embeddings from all datasets at once before training. This is useful to save time during training
    with smaller datasets. If set to `False`, will save disk space by precomputing embeddings on-the-fly during
    training when required. Make sure to set `precomputation_items` to a reasonable value in line with the size
    of your dataset(s).

DATALOADER_ARGUMENTS
--------------------
See https://pytorch.org/docs/stable/data.html for more information.

dataloader_num_workers (`int`, defaults to `0`):
    Number of subprocesses to use for data loading. `0` means that the data will be loaded in a blocking manner
    on the main process.
pin_memory (`bool`, defaults to `False`):
    Whether or not to use the pinned memory setting in PyTorch dataloader. This is useful for faster data loading.

DIFFUSION ARGUMENTS
-------------------
flow_resolution_shifting (`bool`, defaults to `False`):
    Resolution-dependent shifting of timestep schedules.
    [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206).
    TODO(aryan): We don't support this yet.
flow_base_seq_len (`int`, defaults to `256`):
    Base number of tokens for images/video when applying resolution-dependent shifting.
flow_max_seq_len (`int`, defaults to `4096`):
    Maximum number of tokens for images/video when applying resolution-dependent shifting.
flow_base_shift (`float`, defaults to `0.5`):
    Base shift for timestep schedules when applying resolution-dependent shifting.
flow_max_shift (`float`, defaults to `1.15`):
    Maximum shift for timestep schedules when applying resolution-dependent shifting.
flow_shift (`float`, defaults to `1.0`):
    Instead of training with uniform/logit-normal sigmas, shift them as (shift * sigma) / (1 + (shift - 1) * sigma).
    Setting it higher is helpful when trying to train models for high-resolution generation or to produce better
    samples in lower number of inference steps.
flow_weighting_scheme (`str`, defaults to `none`):
    We default to the "none" weighting scheme for uniform sampling and uniform loss.
    Choose between ['sigma_sqrt', 'logit_normal', 'mode', 'cosmap', 'none'].
flow_logit_mean (`float`, defaults to `0.0`):
    Mean to use when using the `'logit_normal'` weighting scheme.
flow_logit_std (`float`, defaults to `1.0`):
    Standard deviation to use when using the `'logit_normal'` weighting scheme.
flow_mode_scale (`float`, defaults to `1.29`):
    Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.

TRAINING ARGUMENTS
------------------
training_type (`str`, defaults to `None`):
    Type of training to perform. Choose between ['lora'].
seed (`int`, defaults to `42`):
    A seed for reproducible training.
batch_size (`int`, defaults to `1`):
    Per-device batch size.
train_steps (`int`, defaults to `1000`):
    Total number of training steps to perform.
max_data_samples (`int`, defaults to `2**64`):
    Maximum number of data samples observed during training training. If lesser than that required by `train_steps`,
    the training will stop early.
gradient_accumulation_steps (`int`, defaults to `1`):
    Number of gradients steps to accumulate before performing an optimizer step.
gradient_checkpointing (`bool`, defaults to `False`):
    Whether or not to use gradient/activation checkpointing to save memory at the expense of slower
    backward pass.
checkpointing_steps (`int`, defaults to `500`):
    Save a checkpoint of the training state every X training steps. These checkpoints can be used both
    as final checkpoints in case they are better than the last checkpoint, and are also suitable for
    resuming training using `resume_from_checkpoint`.
checkpointing_limit (`int`, defaults to `None`):
    Max number of checkpoints to store.
resume_from_checkpoint (`str`, defaults to `None`):
    Whether training should be resumed from a previous checkpoint. Use a path saved by `checkpointing_steps`,
    or `"latest"` to automatically select the last available checkpoint.

OPTIMIZER ARGUMENTS
-------------------
optimizer (`str`, defaults to `adamw`):
    The optimizer type to use. Choose between the following:
        - Torch optimizers: ["adam", "adamw"]
        - Bitsandbytes optimizers: ["adam-bnb", "adamw-bnb", "adam-bnb-8bit", "adamw-bnb-8bit"]
lr (`float`, defaults to `1e-4`):
    Initial learning rate (after the potential warmup period) to use.
lr_scheduler (`str`, defaults to `cosine_with_restarts`):
    The scheduler type to use. Choose between ['linear', 'cosine', 'cosine_with_restarts', 'polynomial',
    'constant', 'constant_with_warmup'].
lr_warmup_steps (`int`, defaults to `500`):
    Number of steps for the warmup in the lr scheduler.
lr_num_cycles (`int`, defaults to `1`):
    Number of hard resets of the lr in cosine_with_restarts scheduler.
lr_power (`float`, defaults to `1.0`):
    Power factor of the polynomial scheduler.
beta1 (`float`, defaults to `0.9`):
beta2 (`float`, defaults to `0.95`):
beta3 (`float`, defaults to `0.999`):
weight_decay (`float`, defaults to `0.0001`):
    Penalty for large weights in the model.
epsilon (`float`, defaults to `1e-8`):
    Small value to avoid division by zero in the optimizer.
max_grad_norm (`float`, defaults to `1.0`):
    Maximum gradient norm to clip the gradients.

VALIDATION ARGUMENTS
--------------------
validation_dataset_file (`str`, defaults to `None`):
    Path to a CSV/JSON/PARQUET/ARROW file containing information for validation. The file must contain atleast the
    "caption" column. Other columns such as "image_path" and "video_path" can be provided too. If provided, "image_path"
    will be used to load a PIL.Image.Image and set the "image" key in the sample dictionary. Similarly, "video_path"
    will be used to load a List[PIL.Image.Image] and set the "video" key in the sample dictionary.
    The validation dataset file may contain other attributes specific to inference/validation such as:
        - "height" and "width" and "num_frames": Resolution
        - "num_inference_steps": Number of inference steps
        - "guidance_scale": Classifier-free Guidance Scale
        - ... (any number of additional attributes can be provided. The ModelSpecification::validate method will be
          invoked with the sample dictionary to validate the sample.)
validation_steps (`int`, defaults to `500`):
    Number of training steps after which a validation step is performed.
enable_model_cpu_offload (`bool`, defaults to `False`):
    Whether or not to offload different modeling components to CPU during validation.

MISCELLANEOUS ARGUMENTS
-----------------------
tracker_name (`str`, defaults to `finetrainers`):
    Name of the tracker/project to use for logging training metrics.
push_to_hub (`bool`, defaults to `False`):
    Whether or not to push the model to the Hugging Face Hub.
hub_token (`str`, defaults to `None`):
    The API token to use for pushing the model to the Hugging Face Hub.
hub_model_id (`str`, defaults to `None`):
    The model identifier to use for pushing the model to the Hugging Face Hub.
output_dir (`str`, defaults to `None`):
    The directory where the model checkpoints and logs will be stored.
logging_dir (`str`, defaults to `logs`):
    The directory where the logs will be stored.
logging_steps (`int`, defaults to `1`):
    Training logs will be tracked every `logging_steps` steps.
nccl_timeout (`int`, defaults to `1800`):
    Timeout for the NCCL communication.
report_to (`str`, defaults to `wandb`):
    The name of the logger to use for logging training metrics. Choose between ['wandb'].
verbose (`int`, defaults to `1`):
    Whether or not to print verbose logs.
        - 0: Diffusers/Transformers warning logging on local main process only
        - 1: Diffusers/Transformers info logging on local main process only
        - 2: Diffusers/Transformers debug logging on local main process only
        - 3: Diffusers/Transformers debug logging on all processes

TORCH CONFIG ARGUMENTS
----------------------
allow_tf32 (`bool`, defaults to `False`):
    Whether or not to allow the use of TF32 matmul on compatible hardware.
float32_matmul_precision (`str`, defaults to `highest`):
    The precision to use for float32 matmul. Choose between ['highest', 'high', 'medium'].
```

### Attention Provider

These arguments are relevant to setting attention provider for different modeling components. The attention providers may be set differently for training and validation/inference.

```
attn_provider_training (`str`, defaults to "native"):
    The attention provider to use for training. Choose between
    [
        'flash', 'flash_varlen', 'flex', 'native', '_native_cudnn', '_native_efficient', '_native_flash',
        '_native_math'
    ]
attn_provider_inference (`str`, defaults to "native"):
    The attention provider to use for validation. Choose between
    [
        'flash', 'flash_varlen', 'flex', 'native', '_native_cudnn', '_native_efficient', '_native_flash',
        '_native_math', 'sage', 'sage_varlen', '_sage_qk_int8_pv_fp8_cuda', '_sage_qk_int8_pv_fp8_cuda_sm90',
        '_sage_qk_int8_pv_fp16_cuda', '_sage_qk_int8_pv_fp16_triton', 'xformers'
    ]
```

## SFT training

If using `--training_type lora`, these arguments can be specified.

```
rank (int):
    Rank of the low rank approximation.
lora_alpha (int):
    The lora_alpha parameter to compute scaling factor (lora_alpha / rank) for low-rank matrices.
target_modules (`str` or `List[str]`):
    Target modules for the low rank approximation. Can be a regex string or a list of regex strings.
```

No additional arguments are required for `--training_type full-finetune`.

## Control training

If using `--training_type control-lora`, these arguments can be specified.

```
control_type (`str`, defaults to `"canny"`):
    Control type for the low rank approximation matrices. Can be "canny", "custom".
rank (int, defaults to `64`):
    Rank of the low rank approximation matrix.
lora_alpha (int, defaults to `64`):
    The lora_alpha parameter to compute scaling factor (lora_alpha / rank) for low-rank matrices.
target_modules (`str` or `List[str]`, defaults to `"(transformer_blocks|single_transformer_blocks).*(to_q|to_k|to_v|to_out.0|ff.net.0.proj|ff.net.2)"`):
    Target modules for the low rank approximation matrices. Can be a regex string or a list of regex strings.
train_qk_norm (`bool`, defaults to `False`):
    Whether to train the QK normalization layers.
frame_conditioning_type (`str`, defaults to `"full"`):
    Type of frame conditioning. Can be "index", "prefix", "random", "first_and_last", or "full".
frame_conditioning_index (int, defaults to `0`):
    Index of the frame conditioning. Only used if `frame_conditioning_type` is "index".
frame_conditioning_concatenate_mask (`bool`, defaults to `False`):
    Whether to concatenate the frame mask with the latents across channel dim.
```

If using `--training_type control-full-finetune`, these arguments can be specified.

```
control_type (`str`, defaults to `"canny"`):
    Control type for the low rank approximation matrices. Can be "canny", "custom".
train_qk_norm (`bool`, defaults to `False`):
    Whether to train the QK normalization layers.
frame_conditioning_type (`str`, defaults to `"index"`):
    Type of frame conditioning. Can be "index", "prefix", "random", "first_and_last", or "full".
frame_conditioning_index (int, defaults to `0`):
    Index of the frame conditioning. Only used if `frame_conditioning_type` is "index".
frame_conditioning_concatenate_mask (`bool`, defaults to `False`):
    Whether to concatenate the frame mask with the latents across channel dim.
```
