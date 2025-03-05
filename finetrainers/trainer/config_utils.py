import argparse
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ..args import BaseArgs


class ConfigMixin:
    def add_args(self, parser: argparse.ArgumentParser):
        raise NotImplementedError("ConfigMixin::add_args should be implemented by subclasses.")

    def validate_args(self, args: "BaseArgs"):
        raise NotImplementedError("ConfigMixin::map_args should be implemented by subclasses.")

    def map_args(self, argparse_args: argparse.Namespace, mapped_args: "BaseArgs"):
        raise NotImplementedError("ConfigMixin::validate_args should be implemented by subclasses.")
