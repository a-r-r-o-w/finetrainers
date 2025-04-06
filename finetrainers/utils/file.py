import pathlib
import shutil
from pathlib import Path
from typing import List, Union

from finetrainers.logging import get_logger


logger = get_logger()


def find_files(root: str, pattern: str, depth: int = 0) -> List[str]:
    root_path = pathlib.Path(root)
    result_files = []

    def within_depth(path: pathlib.Path) -> bool:
        return len(path.relative_to(root_path).parts) <= depth

    if depth == 0:
        result_files.extend([str(file) for file in root_path.glob(pattern)])
    else:
        for file in root_path.rglob(pattern):
            if not file.is_file() or not within_depth(file.parent):
                continue
            result_files.append(str(file))

    return result_files


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
