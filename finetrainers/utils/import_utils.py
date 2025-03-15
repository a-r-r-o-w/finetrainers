import importlib

import importlib_metadata

from ..logging import get_logger


logger = get_logger()


_bitsandbytes_available = importlib.util.find_spec("bitsandbytes") is not None
try:
    _bitsandbytes_version = importlib_metadata.version("bitsandbytes")
    logger.debug(f"Successfully imported bitsandbytes version {_bitsandbytes_version}")
except importlib_metadata.PackageNotFoundError:
    _bitsandbytes_available = False


_kornia_available = importlib.util.find_spec("kornia") is not None
try:
    _kornia_version = importlib_metadata.version("kornia")
    logger.debug(f"Successfully imported kornia version {_kornia_version}")
except importlib_metadata.PackageNotFoundError:
    _kornia_available = False


def is_bitsandbytes_available():
    return _bitsandbytes_available


def is_kornia_available():
    return _kornia_available
