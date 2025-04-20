import importlib
import importlib.util
import operator as op
from typing import Union

import importlib_metadata
from packaging.version import Version, parse

from finetrainers.logging import get_logger


logger = get_logger()

STR_OPERATION_TO_FUNC = {">": op.gt, ">=": op.ge, "==": op.eq, "!=": op.ne, "<=": op.le, "<": op.lt}


# This function was copied from: https://github.com/huggingface/diffusers/blob/5873377a660dac60a6bd86ef9b4fdfc385305977/src/diffusers/utils/import_utils.py#L57
def _is_package_available(pkg_name: str):
    pkg_exists = importlib.util.find_spec(pkg_name) is not None
    pkg_version = "N/A"

    if pkg_exists:
        try:
            pkg_version = importlib_metadata.version(pkg_name)
            logger.debug(f"Successfully imported {pkg_name} version {pkg_version}")
        except (ImportError, importlib_metadata.PackageNotFoundError):
            pkg_exists = False

    return pkg_exists, pkg_version


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


_bitsandbytes_available, _bitsandbytes_version = _is_package_available("bitsandbytes")
_datasets_available, _datasets_version = _is_package_available("datasets")
_flash_attn_available, _flash_attn_version = _is_package_available("flash_attn")
_kornia_available, _kornia_version = _is_package_available("kornia")
_sageattention_available, _sageattention_version = _is_package_available("sageattention")
_torch_available, _torch_version = _is_package_available("torch")
_xformers_available, _xformers_version = _is_package_available("xformers")


def is_bitsandbytes_available():
    return _bitsandbytes_available


def is_datasets_available():
    return _datasets_available


def is_flash_attn_available():
    return _flash_attn_available


def is_kornia_available():
    return _kornia_available


def is_sageattention_available():
    return _sageattention_available


def is_torch_available():
    return _torch_available


def is_xformers_available():
    return _xformers_available


def is_bitsandbytes_version(operation: str, version: str):
    if not _bitsandbytes_available:
        return False
    return compare_versions(parse(_bitsandbytes_version), operation, version)


def is_datasets_version(operation: str, version: str):
    if not _datasets_available:
        return False
    return compare_versions(parse(_datasets_version), operation, version)


def is_flash_attn_version(operation: str, version: str):
    if not _flash_attn_available:
        return False
    return compare_versions(parse(_flash_attn_version), operation, version)


def is_kornia_version(operation: str, version: str):
    if not _kornia_available:
        return False
    return compare_versions(parse(_kornia_version), operation, version)


def is_sageattention_version(operation: str, version: str):
    if not _sageattention_available:
        return False
    return compare_versions(parse(_sageattention_version), operation, version)


def is_torch_version(operation: str, version: str):
    if not _torch_available:
        return False
    return compare_versions(parse(_torch_version), operation, version)


def is_xformers_version(operation: str, version: str):
    if not _xformers_available:
        return False
    return compare_versions(parse(_xformers_version), operation, version)
