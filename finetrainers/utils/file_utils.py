import logging
import os
import shutil
from pathlib import Path
from typing import List, Union


logger = logging.getLogger("finetrainers")
logger.setLevel(os.environ.get("FINETRAINERS_LOG_LEVEL", "INFO"))


def find_files(dir: Union[str, Path], prefix: str = "checkpoint") -> List[str]:
    if not isinstance(dir, Path):
        dir = Path(dir)
    if not dir.exists():
        return []
    checkpoints = os.listdir(dir.as_posix())
    checkpoints = [c for c in checkpoints if c.startswith(prefix)]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
    return checkpoints


def delete_files(dirs: Union[str, List[str], Path, List[Path]]) -> None:
    if not isinstance(dirs, list):
        dirs = [dirs]
    dirs = [Path(d) if isinstance(d, str) else d for d in dirs]
    logger.info(f"Deleting files: {dirs}")
    for dir in dirs:
        if not dir.exists():
            continue
        shutil.rmtree(dir, ignore_errors=True)


def string_to_filename(s: str) -> str:
    return (
        s.replace(" ", "-")
        .replace("/", "-")
        .replace(":", "-")
        .replace(".", "-")
        .replace(",", "-")
        .replace(";", "-")
        .replace("!", "-")
        .replace("?", "-")
    )
