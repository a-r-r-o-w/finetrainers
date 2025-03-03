"""
Default values taken from 
https://github.com/genmoai/mochi/blob/aba74c1b5e0755b1fa3343d9e4bd22e89de77ab1/demos/fine_tuner/configs/lora.yaml
when applicable.
"""

import argparse


def _get_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
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
    parser.add_argument(
        "--cast_dit",
        action="store_true",
        help="If we should cast DiT params to a lower precision.",
    )
    parser.add_argument(
        "--compile_dit",
        action="store_true",
        help="If we should compile the DiT.",
    )


def _get_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help=("A folder containing the training data."),
    )
    parser.add_argument(
        "--caption_dropout",
        type=float,
        default=None,
        help=("Probability to drop out captions randomly."),
    )

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


def _get_validation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--validation_prompt",
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
        "--validation_prompt_separator",
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
        default=50,
        help="Run validation every X training steps. Validation consists of running the validation prompt `args.num_validation_videos` times.",
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
    parser.add_argument(
        "--enable_model_cpu_offload",
        action="store_true",
        default=False,
        help="Whether or not to enable model-wise CPU offloading when performing validation/testing to save memory.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="FPS to use when serializing the output videos.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=848,
    )


def _get_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--rank", type=int, default=16, help="The rank for LoRA matrices.")
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="The lora_alpha to compute scaling factor (lora_alpha / rank) for LoRA matrices.",
    )
    parser.add_argument(
        "--target_modules",
        nargs="+",
        type=str,
        default=["to_k", "to_q", "to_v", "to_out.0"],
        help="Target modules to train LoRA for.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="mochi-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides `--num_train_epochs`.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=200,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
    )


def _get_optimizer_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--optimizer",
        type=lambda s: s.lower(),
        default="adam",
        choices=["adam", "adamw"],
        help=("The optimizer type to use."),
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay to use for optimizer.",
    )


def _get_configuration_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tracker_name", type=str, default=None, help="Project tracker name")
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
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--report_to", type=str, default=None, help="If logging to wandb.")


def get_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script for Mochi-1.")

    _get_model_args(parser)
    _get_dataset_args(parser)
    _get_training_args(parser)
    _get_validation_args(parser)
    _get_optimizer_args(parser)
    _get_configuration_args(parser)

    return parser.parse_args()
