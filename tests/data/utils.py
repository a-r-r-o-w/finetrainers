import pathlib
from typing import List

from diffusers.utils import export_to_video
from PIL import Image

from finetrainers.data.dataset import COMMON_CAPTION_FILES, COMMON_IMAGE_FILES, COMMON_VIDEO_FILES  # noqa


def create_dummy_directory_structure(
    directory_structure: List[str], tmpdir, num_data_files: int, caption: str, metadata_extension: str
):
    for item in directory_structure:
        # TODO(aryan): this should be improved
        if item in COMMON_CAPTION_FILES:
            data_file = pathlib.Path(tmpdir.name) / item
            with open(data_file.as_posix(), "w") as f:
                for _ in range(num_data_files):
                    f.write(f"{caption}\n")
        elif item in COMMON_IMAGE_FILES:
            data_file = pathlib.Path(tmpdir.name) / item
            with open(data_file.as_posix(), "w") as f:
                for i in range(num_data_files):
                    f.write(f"images/{i}.jpg\n")
        elif item in COMMON_VIDEO_FILES:
            data_file = pathlib.Path(tmpdir.name) / item
            with open(data_file.as_posix(), "w") as f:
                for i in range(num_data_files):
                    f.write(f"videos/{i}.mp4\n")
        elif item == "metadata.csv":
            data_file = pathlib.Path(tmpdir.name) / item
            with open(data_file.as_posix(), "w") as f:
                f.write("file_name,caption\n")
                for i in range(num_data_files):
                    f.write(f"{i}.{metadata_extension},{caption}\n")
        elif item == "metadata.jsonl":
            data_file = pathlib.Path(tmpdir.name) / item
            with open(data_file.as_posix(), "w") as f:
                for i in range(num_data_files):
                    f.write(f'{{"file_name": "{i}.{metadata_extension}", "caption": "{caption}"}}\n')
        elif item.endswith(".txt"):
            data_file = pathlib.Path(tmpdir.name) / item
            with open(data_file.as_posix(), "w") as f:
                f.write(caption)
        elif item.endswith(".jpg") or item.endswith(".png"):
            data_file = pathlib.Path(tmpdir.name) / item
            Image.new("RGB", (64, 64)).save(data_file.as_posix())
        elif item.endswith(".mp4"):
            data_file = pathlib.Path(tmpdir.name) / item
            export_to_video([Image.new("RGB", (64, 64))] * 4, data_file.as_posix(), fps=2)
        else:
            data_file = pathlib.Path(tmpdir.name, item)
            data_file.mkdir(exist_ok=True, parents=True)
