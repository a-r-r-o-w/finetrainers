import argparse
from enum import Enum
from typing import TYPE_CHECKING, List, Union

from ..config_utils import ConfigMixin


if TYPE_CHECKING:
    from ...args import BaseArgs


class ControlType(str, Enum):
    r"""
    Enum class for the control types.
    """

    CANNY = "canny"
    CUSTOM = "custom"


class ControlLowRankConfig(ConfigMixin):
    r"""
    Configuration class for SFT channel-concatenated Control low rank training.

    Args:
        control_type (`str`):
            Control type for the low rank approximation matrices. Can be "canny".
        rank (int):
            Rank of the low rank approximation matrix.
        lora_alpha (int):
            The lora_alpha parameter to compute scaling factor (lora_alpha / rank) for low-rank matrices.
        target_modules (`str` or `List[str]`):
            Target modules for the low rank approximation matrices. Can be a regex string or a list of regex strings.
        train_qk_norm (`bool`):
            Whether to train the QK normalization layers.
    """

    control_type: str = ControlType.CANNY
    rank: int = 64
    lora_alpha: int = 64
    target_modules: Union[
        str, List[str]
    ] = "(transformer_blocks|single_transformer_blocks).*(to_q|to_k|to_v|to_out.0|ff.net.0.proj|ff.net.2)"
    train_qk_norm: bool = False

    def add_args(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--control_type", type=str, default=ControlType.CANNY, choices=list(ControlType.__members__.keys())
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


class ControlFullRankConfig(ConfigMixin):
    r"""
    Configuration class for SFT channel-concatenated Control full rank training.
    """

    control_type: str = ControlType.CANNY
    train_qk_norm: bool = False

    def add_args(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--control_type", type=str, default=ControlType.CANNY, choices=list(ControlType.__members__.keys())
        )
        parser.add_argument("--train_qk_norm", action="store_true")

    def validate_args(self, args: "BaseArgs"):
        pass

    def map_args(self, argparse_args: argparse.Namespace, mapped_args: "BaseArgs"):
        mapped_args.control_type = argparse_args.control_type
        mapped_args.train_qk_norm = argparse_args.train_qk_norm
