import argparse
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Union

from finetrainers.utils import ArgsConfigMixin


if TYPE_CHECKING:
    from finetrainers.args import BaseArgs


class ControlType(str, Enum):
    r"""
    Enum class for the control types.
    """

    CANNY = "canny"
    CUSTOM = "custom"
    NONE = "none"


class FrameConditioningType(str, Enum):
    r"""
    Enum class for the frame conditioning types.
    """

    INDEX = "index"
    PREFIX = "prefix"
    RANDOM = "random"
    FIRST_AND_LAST = "first_and_last"
    FULL = "full"


class ControlLowRankConfig(ArgsConfigMixin):
    r"""
    Configuration class for SFT channel-concatenated Control low rank training.

    Args:
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
    """

    control_type: str = ControlType.CANNY
    rank: int = 64
    lora_alpha: int = 64
    target_modules: Union[str, List[str]] = (
        "(transformer_blocks|single_transformer_blocks).*(to_q|to_k|to_v|to_out.0|ff.net.0.proj|ff.net.2)"
    )
    train_qk_norm: bool = False

    # Specific to video models
    frame_conditioning_type: str = FrameConditioningType.FULL
    frame_conditioning_index: int = 0
    frame_conditioning_concatenate_mask: bool = False

    def add_args(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--control_type",
            type=str,
            default=ControlType.CANNY.value,
            choices=[x.value for x in ControlType.__members__.values()],
        )
        parser.add_argument("--rank", type=int, default=64)
        parser.add_argument("--lora_alpha", type=int, default=64)
        parser.add_argument(
            "--target_modules",
            type=str,
            nargs="+",
            default=[
                "(transformer_blocks|single_transformer_blocks).*(to_q|to_k|to_v|to_out.0|ff.net.0.proj|ff.net.2)"
            ],
        )
        parser.add_argument("--train_qk_norm", action="store_true")
        parser.add_argument(
            "--frame_conditioning_type",
            type=str,
            default=FrameConditioningType.INDEX.value,
            choices=[x.value for x in FrameConditioningType.__members__.values()],
        )
        parser.add_argument("--frame_conditioning_index", type=int, default=0)
        parser.add_argument("--frame_conditioning_concatenate_mask", action="store_true")

    def validate_args(self, args: "BaseArgs"):
        assert self.rank > 0, "Rank must be a positive integer."
        assert self.lora_alpha > 0, "lora_alpha must be a positive integer."

    def map_args(self, argparse_args: argparse.Namespace, mapped_args: "BaseArgs"):
        mapped_args.control_type = argparse_args.control_type
        mapped_args.rank = argparse_args.rank
        mapped_args.lora_alpha = argparse_args.lora_alpha
        mapped_args.target_modules = (
            argparse_args.target_modules[0] if len(argparse_args.target_modules) == 1 else argparse_args.target_modules
        )
        mapped_args.train_qk_norm = argparse_args.train_qk_norm
        mapped_args.frame_conditioning_type = argparse_args.frame_conditioning_type
        mapped_args.frame_conditioning_index = argparse_args.frame_conditioning_index
        mapped_args.frame_conditioning_concatenate_mask = argparse_args.frame_conditioning_concatenate_mask

    def to_dict(self) -> Dict[str, Any]:
        return {
            "control_type": self.control_type,
            "rank": self.rank,
            "lora_alpha": self.lora_alpha,
            "target_modules": self.target_modules,
            "train_qk_norm": self.train_qk_norm,
            "frame_conditioning_type": self.frame_conditioning_type,
            "frame_conditioning_index": self.frame_conditioning_index,
            "frame_conditioning_concatenate_mask": self.frame_conditioning_concatenate_mask,
        }


class ControlFullRankConfig(ArgsConfigMixin):
    r"""
    Configuration class for SFT channel-concatenated Control full rank training.

    Args:
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
    """

    control_type: str = ControlType.CANNY
    train_qk_norm: bool = False

    # Specific to video models
    frame_conditioning_type: str = FrameConditioningType.INDEX
    frame_conditioning_index: int = 0
    frame_conditioning_concatenate_mask: bool = False

    def add_args(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--control_type",
            type=str,
            default=ControlType.CANNY.value,
            choices=[x.value for x in ControlType.__members__.values()],
        )
        parser.add_argument("--train_qk_norm", action="store_true")
        parser.add_argument(
            "--frame_conditioning_type",
            type=str,
            default=FrameConditioningType.INDEX.value,
            choices=[x.value for x in FrameConditioningType.__members__.values()],
        )
        parser.add_argument("--frame_conditioning_index", type=int, default=0)
        parser.add_argument("--frame_conditioning_concatenate_mask", action="store_true")

    def validate_args(self, args: "BaseArgs"):
        pass

    def map_args(self, argparse_args: argparse.Namespace, mapped_args: "BaseArgs"):
        mapped_args.control_type = argparse_args.control_type
        mapped_args.train_qk_norm = argparse_args.train_qk_norm
        mapped_args.frame_conditioning_type = argparse_args.frame_conditioning_type
        mapped_args.frame_conditioning_index = argparse_args.frame_conditioning_index
        mapped_args.frame_conditioning_concatenate_mask = argparse_args.frame_conditioning_concatenate_mask

    def to_dict(self) -> Dict[str, Any]:
        return {
            "control_type": self.control_type,
            "train_qk_norm": self.train_qk_norm,
            "frame_conditioning_type": self.frame_conditioning_type,
            "frame_conditioning_index": self.frame_conditioning_index,
            "frame_conditioning_concatenate_mask": self.frame_conditioning_concatenate_mask,
        }
