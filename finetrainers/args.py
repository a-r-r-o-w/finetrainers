import argparse
from typing import Any, Dict, List, Optional, Tuple

import torch

from .constants import DEFAULT_IMAGE_RESOLUTION_BUCKETS, DEFAULT_VIDEO_RESOLUTION_BUCKETS
from .models import SUPPORTED_MODEL_CONFIGS


class Args:
    r"""
    The arguments for the finetrainers training script.

    Args:
        flow_resolution_shifting (`bool`, defaults to `False`):
            Resolution-dependant shifting of timestep schedules.
            [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206)
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
    mixed_precision: str = (
        None  # TODO: consider removing later https://github.com/a-r-r-o-w/finetrainers/pull/139#discussion_r1897438414
    )
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
                "mixed_precision": self.mixed_precision,
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


def parse_arguments() -> Args:
    parser = argparse.ArgumentParser()

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


def validate_args(args: Args):
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
        default=None,
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
        default=None,
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
        help="Resolution-dependant shifting of timestep schedules.",
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
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--flow_logit_mean",
        type=float,
        default=0.0,
        help="mean to use when using the `'logit_normal'` weighting scheme.",
    )
    parser.add_argument(
        "--flow_logit_std",
        type=float,
        default=1.0,
        help="std to use when using the `'logit_normal'` weighting scheme.",
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
        default=None,
        help="Type of training to perform. Choose between ['lora']",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp8", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Defaults to the value of accelerate config of the current system or the "
            "flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--train_epochs", type=int, default=1)
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
        "--target_modules", type=str, default="to_k to_q to_v to_out.0", nargs="+", help="The target modules for LoRA."
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
        default=False,
        help="Whether or not to use VAE slicing for saving memory.",
    )
    parser.add_argument(
        "--enable_tiling",
        action="store_true",
        default=False,
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
        default=False,
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
        "--enable_model_cpu_offload",
        action="store_true",
        default=False,
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
        default="finetrainer-training",
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
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )


_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}
_INVERSE_DTYPE_MAP = {v: k for k, v in _DTYPE_MAP.items()}


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
    result_args.mixed_precision = args.mixed_precision
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
