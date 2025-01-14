import argparse
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch

from .constants import DEFAULT_IMAGE_RESOLUTION_BUCKETS, DEFAULT_VIDEO_RESOLUTION_BUCKETS
from .models import SUPPORTED_MODEL_CONFIGS


class Args:
    r"""
    The arguments for the finetrainers training script.

    For helpful information about arguments, run `python train.py --help`.

    TODO(aryan): add `python train.py --recommend_configs --model_name <model_name>` to recommend
    good training configs for a model after extensive testing.
    TODO(aryan): add `python train.py --memory_requirements --model_name <model_name>` to show
    memory requirements per model, per training type with sensible training settings.

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

    DATASET ARGUMENTS
    -----------------
    data_root (`str`):
        A folder containing the training data.
    dataset_file (`str`, defaults to `None`):
        Path to a CSV/JSON/JSONL file containing metadata for training. This should be provided if you're not using
        a directory dataset format containing a simple `prompts.txt` and `videos.txt`/`images.txt` for example.
    video_column (`str`):
        The column of the dataset containing videos. Or, the name of the file in `data_root` folder containing the
        line-separated path to video data.
    caption_column (`str`):
        The column of the dataset containing the instance prompt for each video. Or, the name of the file in
        `data_root` folder containing the line-separated instance prompts.
    id_token (`str`, defaults to `None`):
        Identifier token appended to the start of each prompt if provided. This is useful for LoRA-type training.
    image_resolution_buckets (`List[Tuple[int, int]]`, defaults to `None`):
        Resolution buckets for images. This should be a list of integer tuples, where each tuple represents the
        resolution (height, width) of the image. All images will be resized to the nearest bucket resolution.
    video_resolution_buckets (`List[Tuple[int, int, int]]`, defaults to `None`):
        Resolution buckets for videos. This should be a list of integer tuples, where each tuple represents the
        resolution (num_frames, height, width) of the video. All videos will be resized to the nearest bucket
        resolution.
    video_reshape_mode (`str`, defaults to `None`):
        All input videos are reshaped to this mode. Choose between ['center', 'random', 'none'].
        TODO(aryan): We don't support this.
    caption_dropout_p (`float`, defaults to `0.00`):
        Probability of dropout for the caption tokens. This is useful to improve the unconditional generation
        quality of the model.
    caption_dropout_technique (`str`, defaults to `empty`):
        Technique to use for caption dropout. Choose between ['empty', 'zero']. Some models apply caption dropout
        by setting the prompt condition to an empty string, while others zero-out the text embedding tensors.
    precompute_conditions (`bool`, defaults to `False`):
        Whether or not to precompute the conditionings for the model. This is useful for faster training, and
        reduces the memory requirements.
    remove_common_llm_caption_prefixes (`bool`, defaults to `False`):
        Whether or not to remove common LLM caption prefixes. This is useful for improving the quality of the
        generated text.

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
    train_epochs (`int`, defaults to `1`):
        Number of training epochs.
    train_steps (`int`, defaults to `None`):
        Total number of training steps to perform. If provided, overrides `train_epochs`.
    rank (`int`, defaults to `128`):
        The rank for LoRA matrices.
    lora_alpha (`float`, defaults to `64`):
        The lora_alpha to compute scaling factor (lora_alpha / rank) for LoRA matrices.
    target_modules (`List[str]`, defaults to `["to_k", "to_q", "to_v", "to_out.0"]`):
        The target modules for LoRA. Make sure to modify this based on the model.
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
        The optimizer type to use. Choose between ['adam', 'adamw'].
    use_8bit_bnb (`bool`, defaults to `False`):
        Whether to use 8bit variant of the `optimizer` using `bitsandbytes`.
    lr (`float`, defaults to `1e-4`):
        Initial learning rate (after the potential warmup period) to use.
    scale_lr (`bool`, defaults to `False`):
        Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.
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
    validation_prompts (`List[str]`, defaults to `None`):
        List of prompts to use for validation. If not provided, a random prompt will be selected from the training
        dataset.
    validation_images (`List[str]`, defaults to `None`):
        List of image paths to use for validation.
    validation_videos (`List[str]`, defaults to `None`):
        List of video paths to use for validation.
    validation_heights (`List[int]`, defaults to `None`):
        List of heights for the validation videos.
    validation_widths (`List[int]`, defaults to `None`):
        List of widths for the validation videos.
    validation_num_frames (`List[int]`, defaults to `None`):
        List of number of frames for the validation videos.
    num_validation_videos_per_prompt (`int`, defaults to `1`):
        Number of videos to use for validation per prompt.
    validation_every_n_epochs (`int`, defaults to `None`):
        Perform validation every `n` training epochs.
    validation_every_n_steps (`int`, defaults to `None`):
        Perform validation every `n` training steps.
    enable_model_cpu_offload (`bool`, defaults to `False`):
        Whether or not to offload different modeling components to CPU during validation.
    validation_frame_rate (`int`, defaults to `25`):
        Frame rate to use for the validation videos. This value is defaulted to 25, as used in LTX Video pipeline.

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
    allow_tf32 (`bool`, defaults to `False`):
        Whether or not to allow the use of TF32 matmul on compatible hardware.
    nccl_timeout (`int`, defaults to `1800`):
        Timeout for the NCCL communication.
    report_to (`str`, defaults to `wandb`):
        The name of the logger to use for logging training metrics. Choose between ['wandb'].
    """

    # Model arguments
    model_name: str = None
    pretrained_model_name_or_path: str = None
    revision: Optional[str] = None
    variant: Optional[str] = None
    cache_dir: Optional[str] = None
    text_encoder_dtype: torch.dtype = torch.bfloat16
    text_encoder_2_dtype: torch.dtype = torch.bfloat16
    text_encoder_3_dtype: torch.dtype = torch.bfloat16
    transformer_dtype: torch.dtype = torch.bfloat16
    vae_dtype: torch.dtype = torch.bfloat16
    layerwise_upcasting_modules: List[str] = []
    layerwise_upcasting_storage_dtype: torch.dtype = torch.float8_e4m3fn
    layerwise_upcasting_skip_modules_pattern: List[str] = [
        "patch_embed",
        "pos_embed",
        "x_embedder",
        "context_embedder",
        "time_embed",
        "^proj_in$",
        "^proj_out$",
        "norm",
    ]

    # Dataset arguments
    data_root: str = None
    dataset_file: Optional[str] = None
    video_column: str = None
    caption_column: str = None
    id_token: Optional[str] = None
    image_resolution_buckets: List[Tuple[int, int]] = None
    video_resolution_buckets: List[Tuple[int, int, int]] = None
    video_reshape_mode: Optional[str] = None
    caption_dropout_p: float = 0.00
    caption_dropout_technique: str = "empty"
    precompute_conditions: bool = False
    remove_common_llm_caption_prefixes: bool = False

    # Dataloader arguments
    dataloader_num_workers: int = 0
    pin_memory: bool = False

    # Diffusion arguments
    flow_resolution_shifting: bool = False
    flow_base_seq_len: int = 256
    flow_max_seq_len: int = 4096
    flow_base_shift: float = 0.5
    flow_max_shift: float = 1.15
    flow_shift: float = 1.0
    flow_weighting_scheme: str = "none"
    flow_logit_mean: float = 0.0
    flow_logit_std: float = 1.0
    flow_mode_scale: float = 1.29

    # Training arguments
    training_type: str = None
    seed: int = 42
    batch_size: int = 1
    train_epochs: int = 1
    train_steps: int = None
    rank: int = 128
    lora_alpha: float = 64
    target_modules: List[str] = ["to_k", "to_q", "to_v", "to_out.0"]
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    checkpointing_steps: int = 500
    checkpointing_limit: Optional[int] = None
    resume_from_checkpoint: Optional[str] = None
    enable_slicing: bool = False
    enable_tiling: bool = False

    # Optimizer arguments
    optimizer: str = "adamw"
    use_8bit_bnb: bool = False
    lr: float = 1e-4
    scale_lr: bool = False
    lr_scheduler: str = "cosine_with_restarts"
    lr_warmup_steps: int = 0
    lr_num_cycles: int = 1
    lr_power: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    beta3: float = 0.999
    weight_decay: float = 0.0001
    epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Validation arguments
    validation_prompts: List[str] = None
    validation_images: List[str] = None
    validation_videos: List[str] = None
    validation_heights: List[int] = None
    validation_widths: List[int] = None
    validation_num_frames: List[int] = None
    num_validation_videos_per_prompt: int = 1
    validation_every_n_epochs: Optional[int] = None
    validation_every_n_steps: Optional[int] = None
    enable_model_cpu_offload: bool = False
    validation_frame_rate: int = 25

    # Miscellaneous arguments
    tracker_name: str = "finetrainers"
    push_to_hub: bool = False
    hub_token: Optional[str] = None
    hub_model_id: Optional[str] = None
    output_dir: str = None
    logging_dir: Optional[str] = "logs"
    allow_tf32: bool = False
    nccl_timeout: int = 1800  # 30 minutes
    report_to: str = "wandb"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_arguments": {
                "model_name": self.model_name,
                "pretrained_model_name_or_path": self.pretrained_model_name_or_path,
                "revision": self.revision,
                "variant": self.variant,
                "cache_dir": self.cache_dir,
                "text_encoder_dtype": self.text_encoder_dtype,
                "text_encoder_2_dtype": self.text_encoder_2_dtype,
                "text_encoder_3_dtype": self.text_encoder_3_dtype,
                "transformer_dtype": self.transformer_dtype,
                "vae_dtype": self.vae_dtype,
                "layerwise_upcasting_modules": self.layerwise_upcasting_modules,
                "layerwise_upcasting_storage_dtype": self.layerwise_upcasting_storage_dtype,
                "layerwise_upcasting_skip_modules_pattern": self.layerwise_upcasting_skip_modules_pattern,
            },
            "dataset_arguments": {
                "data_root": self.data_root,
                "dataset_file": self.dataset_file,
                "video_column": self.video_column,
                "caption_column": self.caption_column,
                "id_token": self.id_token,
                "image_resolution_buckets": self.image_resolution_buckets,
                "video_resolution_buckets": self.video_resolution_buckets,
                "video_reshape_mode": self.video_reshape_mode,
                "caption_dropout_p": self.caption_dropout_p,
                "caption_dropout_technique": self.caption_dropout_technique,
                "precompute_conditions": self.precompute_conditions,
                "remove_common_llm_caption_prefixes": self.remove_common_llm_caption_prefixes,
            },
            "dataloader_arguments": {
                "dataloader_num_workers": self.dataloader_num_workers,
                "pin_memory": self.pin_memory,
            },
            "diffusion_arguments": {
                "flow_resolution_shifting": self.flow_resolution_shifting,
                "flow_base_seq_len": self.flow_base_seq_len,
                "flow_max_seq_len": self.flow_max_seq_len,
                "flow_base_shift": self.flow_base_shift,
                "flow_max_shift": self.flow_max_shift,
                "flow_shift": self.flow_shift,
                "flow_weighting_scheme": self.flow_weighting_scheme,
                "flow_logit_mean": self.flow_logit_mean,
                "flow_logit_std": self.flow_logit_std,
                "flow_mode_scale": self.flow_mode_scale,
            },
            "training_arguments": {
                "training_type": self.training_type,
                "seed": self.seed,
                "batch_size": self.batch_size,
                "train_epochs": self.train_epochs,
                "train_steps": self.train_steps,
                "rank": self.rank,
                "lora_alpha": self.lora_alpha,
                "target_modules": self.target_modules,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "gradient_checkpointing": self.gradient_checkpointing,
                "checkpointing_steps": self.checkpointing_steps,
                "checkpointing_limit": self.checkpointing_limit,
                "resume_from_checkpoint": self.resume_from_checkpoint,
                "enable_slicing": self.enable_slicing,
                "enable_tiling": self.enable_tiling,
            },
            "optimizer_arguments": {
                "optimizer": self.optimizer,
                "use_8bit_bnb": self.use_8bit_bnb,
                "lr": self.lr,
                "scale_lr": self.scale_lr,
                "lr_scheduler": self.lr_scheduler,
                "lr_warmup_steps": self.lr_warmup_steps,
                "lr_num_cycles": self.lr_num_cycles,
                "lr_power": self.lr_power,
                "beta1": self.beta1,
                "beta2": self.beta2,
                "beta3": self.beta3,
                "weight_decay": self.weight_decay,
                "epsilon": self.epsilon,
                "max_grad_norm": self.max_grad_norm,
            },
            "validation_arguments": {
                "validation_prompts": self.validation_prompts,
                "validation_images": self.validation_images,
                "validation_videos": self.validation_videos,
                "num_validation_videos_per_prompt": self.num_validation_videos_per_prompt,
                "validation_every_n_epochs": self.validation_every_n_epochs,
                "validation_every_n_steps": self.validation_every_n_steps,
                "enable_model_cpu_offload": self.enable_model_cpu_offload,
                "validation_frame_rate": self.validation_frame_rate,
            },
            "miscellaneous_arguments": {
                "tracker_name": self.tracker_name,
                "push_to_hub": self.push_to_hub,
                "hub_token": self.hub_token,
                "hub_model_id": self.hub_model_id,
                "output_dir": self.output_dir,
                "logging_dir": self.logging_dir,
                "allow_tf32": self.allow_tf32,
                "nccl_timeout": self.nccl_timeout,
                "report_to": self.report_to,
            },
        }


# TODO(aryan): handle more informative messages
_IS_ARGUMENTS_REQUIRED = "--list_models" not in sys.argv


def parse_arguments() -> Args:
    parser = argparse.ArgumentParser()

    if _IS_ARGUMENTS_REQUIRED:
        _add_model_arguments(parser)
        _add_dataset_arguments(parser)
        _add_dataloader_arguments(parser)
        _add_diffusion_arguments(parser)
        _add_training_arguments(parser)
        _add_optimizer_arguments(parser)
        _add_validation_arguments(parser)
        _add_miscellaneous_arguments(parser)

        args = parser.parse_args()
        return _map_to_args_type(args)
    else:
        _add_helper_arguments(parser)

        args = parser.parse_args()
        _display_helper_messages(args)
        sys.exit(0)


def validate_args(args: Args):
    _validated_model_args(args)
    _validate_training_args(args)
    _validate_validation_args(args)


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=list(SUPPORTED_MODEL_CONFIGS.keys()),
        help="Name of model to train.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--text_encoder_dtype", type=str, default="bf16", help="Data type for the text encoder.")
    parser.add_argument("--text_encoder_2_dtype", type=str, default="bf16", help="Data type for the text encoder 2.")
    parser.add_argument("--text_encoder_3_dtype", type=str, default="bf16", help="Data type for the text encoder 3.")
    parser.add_argument("--transformer_dtype", type=str, default="bf16", help="Data type for the transformer model.")
    parser.add_argument("--vae_dtype", type=str, default="bf16", help="Data type for the VAE model.")
    parser.add_argument(
        "--layerwise_upcasting_modules",
        type=str,
        default=[],
        nargs="+",
        choices=["transformer"],
        help="Modules that should have fp8 storage weights but higher precision computation.",
    )
    parser.add_argument(
        "--layerwise_upcasting_storage_dtype",
        type=str,
        default="float8_e4m3fn",
        choices=["float8_e4m3fn", "float8_e5m2"],
        help="Data type for the layerwise upcasting storage.",
    )
    parser.add_argument(
        "--layerwise_upcasting_skip_modules_pattern",
        type=str,
        default=["patch_embed", "pos_embed", "x_embedder", "context_embedder", "^proj_in$", "^proj_out$", "norm"],
        nargs="+",
        help="Modules to skip for layerwise upcasting.",
    )


def _add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    def parse_resolution_bucket(resolution_bucket: str) -> Tuple[int, ...]:
        return tuple(map(int, resolution_bucket.split("x")))

    def parse_image_resolution_bucket(resolution_bucket: str) -> Tuple[int, int]:
        resolution_bucket = parse_resolution_bucket(resolution_bucket)
        assert (
            len(resolution_bucket) == 2
        ), f"Expected 2D resolution bucket, got {len(resolution_bucket)}D resolution bucket"
        return resolution_bucket

    def parse_video_resolution_bucket(resolution_bucket: str) -> Tuple[int, int, int]:
        resolution_bucket = parse_resolution_bucket(resolution_bucket)
        assert (
            len(resolution_bucket) == 3
        ), f"Expected 3D resolution bucket, got {len(resolution_bucket)}D resolution bucket"
        return resolution_bucket

    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help=("A folder containing the training data."),
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        default=None,
        help=("Path to a CSV file if loading prompts/video paths using this format."),
    )
    parser.add_argument(
        "--video_column",
        type=str,
        default="video",
        help="The column of the dataset containing videos. Or, the name of the file in `--data_root` folder containing the line-separated path to video data.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing the instance prompt for each video. Or, the name of the file in `--data_root` folder containing the line-separated instance prompts.",
    )
    parser.add_argument(
        "--id_token",
        type=str,
        default=None,
        help="Identifier token appended to the start of each prompt if provided.",
    )
    parser.add_argument(
        "--image_resolution_buckets",
        type=parse_image_resolution_bucket,
        default=None,
        nargs="+",
        help="Resolution buckets for images.",
    )
    parser.add_argument(
        "--video_resolution_buckets",
        type=parse_video_resolution_bucket,
        default=None,
        nargs="+",
        help="Resolution buckets for videos.",
    )
    parser.add_argument(
        "--video_reshape_mode",
        type=str,
        default=None,
        help="All input videos are reshaped to this mode. Choose between ['center', 'random', 'none']",
    )
    parser.add_argument(
        "--caption_dropout_p",
        type=float,
        default=0.00,
        help="Probability of dropout for the caption tokens.",
    )
    parser.add_argument(
        "--caption_dropout_technique",
        type=str,
        default="empty",
        choices=["empty", "zero"],
        help="Technique to use for caption dropout.",
    )
    parser.add_argument(
        "--precompute_conditions",
        action="store_true",
        help="Whether or not to precompute the conditionings for the model.",
    )
    parser.add_argument(
        "--remove_common_llm_caption_prefixes",
        action="store_true",
        help="Whether or not to remove common LLM caption prefixes.",
    )


def _add_dataloader_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Whether or not to use the pinned memory setting in pytorch dataloader.",
    )


def _add_diffusion_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--flow_resolution_shifting",
        action="store_true",
        help="Resolution-dependent shifting of timestep schedules.",
    )
    parser.add_argument(
        "--flow_base_seq_len",
        type=int,
        default=256,
        help="Base image/video sequence length for the diffusion model.",
    )
    parser.add_argument(
        "--flow_max_seq_len",
        type=int,
        default=4096,
        help="Maximum image/video sequence length for the diffusion model.",
    )
    parser.add_argument(
        "--flow_base_shift",
        type=float,
        default=0.5,
        help="Base shift as described in [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206)",
    )
    parser.add_argument(
        "--flow_max_shift",
        type=float,
        default=1.15,
        help="Maximum shift as described in [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206)",
    )
    parser.add_argument(
        "--flow_shift",
        type=float,
        default=1.0,
        help="Shift value to use for the flow matching timestep schedule.",
    )
    parser.add_argument(
        "--flow_weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help='We default to the "none" weighting scheme for uniform sampling and uniform loss',
    )
    parser.add_argument(
        "--flow_logit_mean",
        type=float,
        default=0.0,
        help="Mean to use when using the `'logit_normal'` weighting scheme.",
    )
    parser.add_argument(
        "--flow_logit_std",
        type=float,
        default=1.0,
        help="Standard deviation to use when using the `'logit_normal'` weighting scheme.",
    )
    parser.add_argument(
        "--flow_mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    # TODO: support full finetuning and other kinds
    parser.add_argument(
        "--training_type",
        type=str,
        choices=["lora", "full-finetune"],
        required=True,
        help="Type of training to perform. Choose between ['lora', 'full-finetune']",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--train_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument(
        "--train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides `--num_train_epochs`.",
    )
    parser.add_argument("--rank", type=int, default=64, help="The rank for LoRA matrices.")
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=64,
        help="The lora_alpha to compute scaling factor (lora_alpha / rank) for LoRA matrices.",
    )
    parser.add_argument(
        "--target_modules",
        type=str,
        default=["to_k", "to_q", "to_v", "to_out.0"],
        nargs="+",
        help="The target modules for LoRA.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpointing_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_slicing",
        action="store_true",
        help="Whether or not to use VAE slicing for saving memory.",
    )
    parser.add_argument(
        "--enable_tiling",
        action="store_true",
        help="Whether or not to use VAE tiling for saving memory.",
    )


def _add_optimizer_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--optimizer",
        type=lambda s: s.lower(),
        default="adam",
        choices=["adam", "adamw"],
        help=("The optimizer type to use."),
    )
    parser.add_argument(
        "--use_8bit_bnb",
        action="store_true",
        help=("Whether to use 8bit variant of the `--optimizer` using `bitsandbytes`."),
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam and Prodigy optimizers.",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.95,
        help="The beta2 parameter for the Adam and Prodigy optimizers.",
    )
    parser.add_argument(
        "--beta3",
        type=float,
        default=None,
        help="Coefficients for computing the Prodigy optimizer's stepsize using running averages. If set to None, uses the value of square root of beta2.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-04,
        help="Weight decay to use for optimizer.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-8,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")


def _add_validation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        help="One or more prompt(s) that is used during validation to verify that the model is learning. Multiple validation prompts should be separated by the '--validation_prompt_seperator' string.",
    )
    parser.add_argument(
        "--validation_images",
        type=str,
        default=None,
        help="One or more image path(s)/URLs that is used during validation to verify that the model is learning. Multiple validation paths should be separated by the '--validation_prompt_seperator' string. These should correspond to the order of the validation prompts.",
    )
    parser.add_argument(
        "--validation_videos",
        type=str,
        default=None,
        help="One or more video path(s)/URLs that is used during validation to verify that the model is learning. Multiple validation paths should be separated by the '--validation_prompt_seperator' string. These should correspond to the order of the validation prompts.",
    )
    parser.add_argument(
        "--validation_separator",
        type=str,
        default=":::",
        help="String that separates multiple validation prompts",
    )
    parser.add_argument(
        "--num_validation_videos",
        type=int,
        default=1,
        help="Number of videos that should be generated during validation per `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=None,
        help="Run validation every X training epochs. Validation consists of running the validation prompt `args.num_validation_videos` times.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=None,
        help="Run validation every X training steps. Validation consists of running the validation prompt `args.num_validation_videos` times.",
    )
    parser.add_argument(
        "--validation_frame_rate",
        type=int,
        default=25,
        help="Frame rate to use for the validation videos.",
    )
    parser.add_argument(
        "--enable_model_cpu_offload",
        action="store_true",
        help="Whether or not to enable model-wise CPU offloading when performing validation/testing to save memory.",
    )


def _add_miscellaneous_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tracker_name", type=str, default="finetrainers", help="Project tracker name")
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="finetrainers-training",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Directory where logs are stored.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--nccl_timeout",
        type=int,
        default=600,
        help="Maximum timeout duration before which allgather, or related, operations fail in multi-GPU/multi-node training settings.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="none",
        choices=["none", "wandb"],
        help="The integration to report the results and logs to.",
    )


def _add_helper_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--list_models",
        action="store_true",
        help="List all the supported models.",
    )


_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
}


def _map_to_args_type(args: Dict[str, Any]) -> Args:
    result_args = Args()

    # Model arguments
    result_args.model_name = args.model_name
    result_args.pretrained_model_name_or_path = args.pretrained_model_name_or_path
    result_args.revision = args.revision
    result_args.variant = args.variant
    result_args.cache_dir = args.cache_dir
    result_args.text_encoder_dtype = _DTYPE_MAP[args.text_encoder_dtype]
    result_args.text_encoder_2_dtype = _DTYPE_MAP[args.text_encoder_2_dtype]
    result_args.text_encoder_3_dtype = _DTYPE_MAP[args.text_encoder_3_dtype]
    result_args.transformer_dtype = _DTYPE_MAP[args.transformer_dtype]
    result_args.vae_dtype = _DTYPE_MAP[args.vae_dtype]
    result_args.layerwise_upcasting_modules = args.layerwise_upcasting_modules
    result_args.layerwise_upcasting_storage_dtype = _DTYPE_MAP[args.layerwise_upcasting_storage_dtype]
    result_args.layerwise_upcasting_skip_modules_pattern = args.layerwise_upcasting_skip_modules_pattern

    # Dataset arguments
    if args.data_root is None and args.dataset_file is None:
        raise ValueError("At least one of `data_root` or `dataset_file` should be provided.")

    result_args.data_root = args.data_root
    result_args.dataset_file = args.dataset_file
    result_args.video_column = args.video_column
    result_args.caption_column = args.caption_column
    result_args.id_token = args.id_token
    result_args.image_resolution_buckets = args.image_resolution_buckets or DEFAULT_IMAGE_RESOLUTION_BUCKETS
    result_args.video_resolution_buckets = args.video_resolution_buckets or DEFAULT_VIDEO_RESOLUTION_BUCKETS
    result_args.video_reshape_mode = args.video_reshape_mode
    result_args.caption_dropout_p = args.caption_dropout_p
    result_args.caption_dropout_technique = args.caption_dropout_technique
    result_args.precompute_conditions = args.precompute_conditions
    result_args.remove_common_llm_caption_prefixes = args.remove_common_llm_caption_prefixes

    # Dataloader arguments
    result_args.dataloader_num_workers = args.dataloader_num_workers
    result_args.pin_memory = args.pin_memory

    # Diffusion arguments
    result_args.flow_resolution_shifting = args.flow_resolution_shifting
    result_args.flow_base_seq_len = args.flow_base_seq_len
    result_args.flow_max_seq_len = args.flow_max_seq_len
    result_args.flow_base_shift = args.flow_base_shift
    result_args.flow_max_shift = args.flow_max_shift
    result_args.flow_shift = args.flow_shift
    result_args.flow_weighting_scheme = args.flow_weighting_scheme
    result_args.flow_logit_mean = args.flow_logit_mean
    result_args.flow_logit_std = args.flow_logit_std
    result_args.flow_mode_scale = args.flow_mode_scale

    # Training arguments
    result_args.training_type = args.training_type
    result_args.seed = args.seed
    result_args.batch_size = args.batch_size
    result_args.train_epochs = args.train_epochs
    result_args.train_steps = args.train_steps
    result_args.rank = args.rank
    result_args.lora_alpha = args.lora_alpha
    result_args.target_modules = args.target_modules
    result_args.gradient_accumulation_steps = args.gradient_accumulation_steps
    result_args.gradient_checkpointing = args.gradient_checkpointing
    result_args.checkpointing_steps = args.checkpointing_steps
    result_args.checkpointing_limit = args.checkpointing_limit
    result_args.resume_from_checkpoint = args.resume_from_checkpoint
    result_args.enable_slicing = args.enable_slicing
    result_args.enable_tiling = args.enable_tiling

    # Optimizer arguments
    result_args.optimizer = args.optimizer or "adamw"
    result_args.use_8bit_bnb = args.use_8bit_bnb
    result_args.lr = args.lr or 1e-4
    result_args.scale_lr = args.scale_lr
    result_args.lr_scheduler = args.lr_scheduler
    result_args.lr_warmup_steps = args.lr_warmup_steps
    result_args.lr_num_cycles = args.lr_num_cycles
    result_args.lr_power = args.lr_power
    result_args.beta1 = args.beta1
    result_args.beta2 = args.beta2
    result_args.beta3 = args.beta3
    result_args.weight_decay = args.weight_decay
    result_args.epsilon = args.epsilon
    result_args.max_grad_norm = args.max_grad_norm

    # Validation arguments
    validation_prompts = args.validation_prompts.split(args.validation_separator) if args.validation_prompts else []
    validation_images = args.validation_images.split(args.validation_separator) if args.validation_images else None
    validation_videos = args.validation_videos.split(args.validation_separator) if args.validation_videos else None
    stripped_validation_prompts = []
    validation_heights = []
    validation_widths = []
    validation_num_frames = []
    for prompt in validation_prompts:
        prompt: str
        prompt = prompt.strip()
        actual_prompt, separator, resolution = prompt.rpartition("@@@")
        stripped_validation_prompts.append(actual_prompt)
        num_frames, height, width = None, None, None
        if len(resolution) > 0:
            num_frames, height, width = map(int, resolution.split("x"))
        validation_num_frames.append(num_frames)
        validation_heights.append(height)
        validation_widths.append(width)

    if validation_images is None:
        validation_images = [None] * len(validation_prompts)
    if validation_videos is None:
        validation_videos = [None] * len(validation_prompts)

    result_args.validation_prompts = stripped_validation_prompts
    result_args.validation_heights = validation_heights
    result_args.validation_widths = validation_widths
    result_args.validation_num_frames = validation_num_frames
    result_args.validation_images = validation_images
    result_args.validation_videos = validation_videos

    result_args.num_validation_videos_per_prompt = args.num_validation_videos
    result_args.validation_every_n_epochs = args.validation_epochs
    result_args.validation_every_n_steps = args.validation_steps
    result_args.enable_model_cpu_offload = args.enable_model_cpu_offload
    result_args.validation_frame_rate = args.validation_frame_rate

    # Miscellaneous arguments
    result_args.tracker_name = args.tracker_name
    result_args.push_to_hub = args.push_to_hub
    result_args.hub_token = args.hub_token
    result_args.hub_model_id = args.hub_model_id
    result_args.output_dir = args.output_dir
    result_args.logging_dir = args.logging_dir
    result_args.allow_tf32 = args.allow_tf32
    result_args.nccl_timeout = args.nccl_timeout
    result_args.report_to = args.report_to

    return result_args


def _validated_model_args(args: Args):
    if args.training_type == "full-finetune":
        assert (
            "transformer" not in args.layerwise_upcasting_modules
        ), "Layerwise upcasting is not supported for full-finetune training"


def _validate_training_args(args: Args):
    if args.training_type == "lora":
        assert args.rank is not None, "Rank is required for LoRA training"
        assert args.lora_alpha is not None, "LoRA alpha is required for LoRA training"
        assert (
            args.target_modules is not None and len(args.target_modules) > 0
        ), "Target modules are required for LoRA training"


def _validate_validation_args(args: Args):
    assert args.validation_prompts is not None, "Validation prompts are required for validation"
    if args.validation_images is not None:
        assert len(args.validation_images) == len(
            args.validation_prompts
        ), "Validation images and prompts should be of same length"
    if args.validation_videos is not None:
        assert len(args.validation_videos) == len(
            args.validation_prompts
        ), "Validation videos and prompts should be of same length"
    assert len(args.validation_prompts) == len(
        args.validation_heights
    ), "Validation prompts and heights should be of same length"
    assert len(args.validation_prompts) == len(
        args.validation_widths
    ), "Validation prompts and widths should be of same length"


def _display_helper_messages(args: argparse.Namespace):
    if args.list_models:
        print("Supported models:")
        for index, model_name in enumerate(SUPPORTED_MODEL_CONFIGS.keys()):
            print(f"  {index + 1}. {model_name}")
