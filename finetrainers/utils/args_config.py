import argparse
from typing import TYPE_CHECKING, Any, Dict


if TYPE_CHECKING:
    from finetrainers.args import BaseArgs


class ArgsConfigMixin:
    def add_args(self, parser: argparse.ArgumentParser):
        raise NotImplementedError("ArgsConfigMixin::add_args should be implemented by subclasses.")

    def map_args(self, argparse_args: argparse.Namespace, mapped_args: "BaseArgs"):
        raise NotImplementedError("ArgsConfigMixin::map_args should be implemented by subclasses.")

    def validate_args(self, args: "BaseArgs"):
        raise NotImplementedError("ArgsConfigMixin::validate_args should be implemented by subclasses.")

    def to_dict(self) -> Dict[str, Any]:
        return {}
