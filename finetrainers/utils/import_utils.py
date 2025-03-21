import importlib
import operator as op
from typing import Union

import importlib_metadata
from packaging.version import Version, parse

from finetrainers.logging import get_logger


logger = get_logger()

STR_OPERATION_TO_FUNC = {">": op.gt, ">=": op.ge, "==": op.eq, "!=": op.ne, "<=": op.le, "<": op.lt}


# This function was copied from: https://github.com/huggingface/accelerate/blob/874c4967d94badd24f893064cc3bef45f57cadf7/src/accelerate/utils/versions.py#L319
def compare_versions(library_or_version: Union[str, Version], operation: str, requirement_version: str):
    """
    Compares a library version to some requirement using a given operation.

    Args:
        library_or_version (`str` or `packaging.version.Version`):
            A library name or a version to check.
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`.
        requirement_version (`str`):
            The version to compare the library version against
    """
    if operation not in STR_OPERATION_TO_FUNC.keys():
        raise ValueError(f"`operation` must be one of {list(STR_OPERATION_TO_FUNC.keys())}, received {operation}")
    operation = STR_OPERATION_TO_FUNC[operation]
    if isinstance(library_or_version, str):
        library_or_version = parse(importlib_metadata.version(library_or_version))
    return operation(library_or_version, parse(requirement_version))


_bitsandbytes_available = importlib.util.find_spec("bitsandbytes") is not None
try:
    _bitsandbytes_version = importlib_metadata.version("bitsandbytes")
    logger.debug(f"Successfully imported bitsandbytes version {_bitsandbytes_version}")
except importlib_metadata.PackageNotFoundError:
    _bitsandbytes_available = False


_datasets_available = importlib.util.find_spec("datasets") is not None
try:
    _datasets_version = importlib_metadata.version("datasets")
    logger.debug(f"Successfully imported datasets version {_datasets_version}")
except importlib_metadata.PackageNotFoundError:
    _datasets_available = False


def is_bitsandbytes_available():
    return _bitsandbytes_available


def is_datasets_available():
    return _datasets_available


def is_datasets_version(operation: str, version: str):
    """
    Compares the current datasets version to a given reference with an operation.

    Args:
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`
        version (`str`):
            A version string
    """
    if not _datasets_available:
        return False
    return compare_versions(parse(_datasets_version), operation, version)
