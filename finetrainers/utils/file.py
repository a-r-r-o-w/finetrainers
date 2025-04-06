import shutil
from pathlib import Path
from typing import List, Union

from finetrainers.logging import get_logger


logger = get_logger()


def find_files(dir: Union[str, Path], prefix: str = None, suffix: str = None) -> List[str]:
    if not isinstance(dir, Path):
        dir = Path(dir)
    if not dir.is_dir():
        return []
    files = []
    for file in dir.rglob("*"):
        if file.is_file():
            if prefix is not None and not file.name.startswith(prefix):
                continue
            if suffix is not None and not file.name.endswith(suffix):
                continue
            files.append(file.as_posix())
    return files


def delete_files(dirs: Union[str, List[str], Path, List[Path]]) -> None:
    if not isinstance(dirs, list):
        dirs = [dirs]
    dirs = [Path(d) if isinstance(d, str) else d for d in dirs]
    logger.debug(f"Deleting files: {dirs}")
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
