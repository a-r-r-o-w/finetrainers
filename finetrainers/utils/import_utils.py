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


def is_bitsandbytes_available():
    return _bitsandbytes_available
