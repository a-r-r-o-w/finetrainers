import pathlib
from typing import List


def find_files(root: str, pattern: str, depth: int = 0) -> List[str]:
    root_path = pathlib.Path(root)
    result_files = []

    def within_depth(path: pathlib.Path) -> bool:
        return len(path.relative_to(root_path).parts) <= depth

    if depth == 0:
        result_files.extend([str(file) for file in root_path.glob(pattern)])
    else:
        # rglob matches all levels, but we filter by depth
        for file in root_path.rglob(pattern):
            if file.is_file() and within_depth(file.parent):
                result_files.append(str(file))

    return result_files
