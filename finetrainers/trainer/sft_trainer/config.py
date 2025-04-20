import argparse
from typing import TYPE_CHECKING, Any, Dict, List, Union

from finetrainers.utils import ArgsConfigMixin


if TYPE_CHECKING:
    from finetrainers.args import BaseArgs


class SFTLowRankConfig(ArgsConfigMixin):
    r"""
    Configuration class for SFT low rank training.

    Args:
        rank (int):
            Rank of the low rank approximation matrix.
        lora_alpha (int):
            The lora_alpha parameter to compute scaling factor (lora_alpha / rank) for low-rank matrices.
        target_modules (`str` or `List[str]`):
            Target modules for the low rank approximation matrices. Can be a regex string or a list of regex strings.
    """

    rank: int = 64
    lora_alpha: int = 64
    target_modules: Union[str, List[str]] = "(transformer_blocks|single_transformer_blocks).*(to_q|to_k|to_v|to_out.0)"

    def add_args(self, parser: argparse.ArgumentParser):
        parser.add_argument("--rank", type=int, default=64)
        parser.add_argument("--lora_alpha", type=int, default=64)
        parser.add_argument(
            "--target_modules",
            type=str,
            nargs="+",
            default=["(transformer_blocks|single_transformer_blocks).*(to_q|to_k|to_v|to_out.0)"],
        )

    def validate_args(self, args: "BaseArgs"):
        assert self.rank > 0, "Rank must be a positive integer."
        assert self.lora_alpha > 0, "lora_alpha must be a positive integer."

    def map_args(self, argparse_args: argparse.Namespace, mapped_args: "BaseArgs"):
        mapped_args.rank = argparse_args.rank
        mapped_args.lora_alpha = argparse_args.lora_alpha
        mapped_args.target_modules = (
            argparse_args.target_modules[0] if len(argparse_args.target_modules) == 1 else argparse_args.target_modules
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"rank": self.rank, "lora_alpha": self.lora_alpha, "target_modules": self.target_modules}


class SFTFullRankConfig(ArgsConfigMixin):
    r"""
    Configuration class for SFT full rank training.
    """

    def add_args(self, parser: argparse.ArgumentParser):
        pass

    def validate_args(self, args: "BaseArgs"):
        pass

    def map_args(self, argparse_args: argparse.Namespace, mapped_args: "BaseArgs"):
        pass
