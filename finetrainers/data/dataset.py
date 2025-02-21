import pathlib

import datasets
import datasets.data_files
import datasets.distributed
import datasets.exceptions
import numpy as np
import PIL.Image
import torch
import torch.distributed.checkpoint.stateful
from diffusers.utils import load_image, load_video

from .. import constants
from ..logging import get_logger
from . import utils


import decord  # isort:skip

decord.bridge.set_bridge("torch")

logger = get_logger()


class ImageCaptionFileDataset(torch.utils.data.IterableDataset, torch.distributed.checkpoint.stateful.Stateful):
    def __init__(self, root: str) -> None:
        super().__init__()

        self.root = pathlib.Path(root)

        data = []
        caption_files = sorted(utils.find_files(self.root.as_posix(), "*.txt", depth=0))
        for caption_file in caption_files:
            data_file = self._find_data_file(caption_file)
            if data_file:
                data.append(
                    {
                        "caption": (self.root / caption_file).as_posix(),
                        "image": (self.root / data_file).as_posix(),
                    }
                )

        data = datasets.Dataset.from_list(data)
        data = data.cast_column("image", datasets.Image(mode="RGB"))

        self._data = data.to_iterable_dataset()
        self._sample_index = 0

    def _get_data_iter(self):
        if self._sample_index == 0:
            return iter(self._data)

        if isinstance(self._data, datasets.Dataset) and self._sample_index >= len(self._data):
            return iter([])

        return iter(self._data.skip(self._sample_index))

    def __iter__(self):
        for sample in self._get_data_iter():
            self._sample_index += 1
            sample["caption"] = _read_caption_from_file(sample["caption"])
            sample["image"] = _preprocess_image(sample["image"])
            yield sample

    def load_state_dict(self, state_dict):
        self._sample_index = state_dict["sample_index"]

    def state_dict(self):
        return {"sample_index": self._sample_index}

    def _find_data_file(self, caption_file: str) -> str:
        caption_file = pathlib.Path(caption_file)
        data_file = None
        found_data = 0

        for extension in constants.SUPPORTED_IMAGE_FILE_EXTENSIONS:
            image_filename = caption_file.with_suffix(f".{extension}")
            if image_filename.exists():
                found_data += 1
                data_file = image_filename

        if found_data == 0:
            return False
        elif found_data > 1:
            raise ValueError(
                f"Multiple data files found for caption file {caption_file}. Please ensure there is only one data "
                f"file per caption file. The following extensions are supported:\n"
                f"  - Images: {constants.SUPPORTED_IMAGE_FILE_EXTENSIONS}\n"
            )

        return data_file.as_posix()


class VideoCaptionFileDataset(torch.utils.data.IterableDataset, torch.distributed.checkpoint.stateful.Stateful):
    def __init__(self, root: str) -> None:
        super().__init__()

        self.root = pathlib.Path(root)

        data = []
        caption_files = sorted(utils.find_files(self.root.as_posix(), "*.txt", depth=0))
        for caption_file in caption_files:
            data_file = self._find_data_file(caption_file)
            if data_file:
                data.append(
                    {
                        "caption": (self.root / caption_file).as_posix(),
                        "video": (self.root / data_file).as_posix(),
                    }
                )

        data = datasets.Dataset.from_list(data)
        data = data.cast_column("video", datasets.Video())

        self._data = data.to_iterable_dataset()
        self._sample_index = 0

    def _get_data_iter(self):
        if self._sample_index == 0:
            return iter(self._data)

        if isinstance(self._data, datasets.Dataset) and self._sample_index >= len(self._data):
            return iter([])

        return iter(self._data.skip(self._sample_index))

    def __iter__(self):
        for sample in self._get_data_iter():
            self._sample_index += 1
            sample["caption"] = _read_caption_from_file(sample["caption"])
            sample["video"] = _preprocess_video(sample["video"])
            yield sample

    def load_state_dict(self, state_dict):
        self._sample_index = state_dict["sample_index"]

    def state_dict(self):
        return {"sample_index": self._sample_index}

    def _find_data_file(self, caption_file: str) -> str:
        caption_file = pathlib.Path(caption_file)
        data_file = None
        found_data = 0

        for extension in constants.SUPPORTED_VIDEO_FILE_EXTENSIONS:
            video_filename = caption_file.with_suffix(f".{extension}")
            if video_filename.exists():
                found_data += 1
                data_file = video_filename

        if found_data == 0:
            return False
        elif found_data > 1:
            raise ValueError(
                f"Multiple data files found for caption file {caption_file}. Please ensure there is only one data "
                f"file per caption file. The following extensions are supported:\n"
                f"  - Videos: {constants.SUPPORTED_VIDEO_FILE_EXTENSIONS}\n"
            )

        return data_file.as_posix()


class ImageFolderDataset(torch.utils.data.IterableDataset, torch.distributed.checkpoint.stateful.Stateful):
    def __init__(self, root: str) -> None:
        super().__init__()

        self.root = pathlib.Path(root)

        data = datasets.load_dataset("imagefolder", data_dir=self.root.as_posix(), split="train")

        self._data = data.to_iterable_dataset()
        self._sample_index = 0

    def _get_data_iter(self):
        if self._sample_index == 0:
            return iter(self._data)

        if isinstance(self._data, datasets.Dataset) and self._sample_index >= len(self._data):
            return iter([])

        return iter(self._data.skip(self._sample_index))

    def __iter__(self):
        for sample in self._get_data_iter():
            self._sample_index += 1
            sample["image"] = _preprocess_image(sample["image"])
            yield sample

    def load_state_dict(self, state_dict):
        self._sample_index = state_dict["sample_index"]

    def state_dict(self):
        return {"sample_index": self._sample_index}


class VideoFolderDataset(torch.utils.data.IterableDataset, torch.distributed.checkpoint.stateful.Stateful):
    def __init__(self, root: str, infinite: bool = True) -> None:
        super().__init__()

        self.root = pathlib.Path(root)
        self.infinite = infinite

        data = datasets.load_dataset("videofolder", data_dir=self.root.as_posix(), split="train")

        self._data = data.to_iterable_dataset()
        self._sample_index = 0

    def _get_data_iter(self):
        if self._sample_index == 0:
            return iter(self._data)
        return iter(self._data.skip(self._sample_index))

    def __iter__(self):
        while True:
            for sample in self._get_data_iter():
                self._sample_index += 1
                sample["video"] = _preprocess_video(sample["video"])
                yield sample

            if not self.infinite:
                logger.warning(f"Dataset ({self.__class__.__name__}={self.root}) has run out of data")
                break
            else:
                self._sample_index = 0

    def load_state_dict(self, state_dict):
        self._sample_index = state_dict["sample_index"]

    def state_dict(self):
        return {"sample_index": self._sample_index}


class ImageWebDataset(torch.utils.data.IterableDataset, torch.distributed.checkpoint.stateful.Stateful):
    def __init__(self, dataset_name: str, streaming: bool = True, infinite: bool = False) -> None:
        super().__init__()

        self.dataset_name = dataset_name
        self.streaming = streaming
        self.infinite = infinite

        data = datasets.load_dataset(dataset_name, split="train", streaming=streaming)
        data = data.rename_column("txt", "caption")
        for column_name in constants.SUPPORTED_IMAGE_FILE_EXTENSIONS:
            if column_name in data.column_names:
                data = data.cast_column(column_name, datasets.Image(mode="RGB"))
                data = data.rename_column(column_name, "image")

        self._data = data
        self._sample_index = 0

    def _get_data_iter(self):
        if self._sample_index == 0:
            return iter(self._data)

        if isinstance(self._data, datasets.Dataset) and self._sample_index >= len(self._data):
            return iter([])

        return iter(self._data.skip(self._sample_index))

    def __iter__(self):
        while True:
            for sample in self._get_data_iter():
                self._sample_index += 1
                yield sample

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                # Reset offset for the next iteration
                self._sample_index = 0
                logger.warning(f"Dataset {self.dataset_name} is being re-looped")

    def load_state_dict(self, state_dict):
        self._sample_index = state_dict["sample_index"]

    def state_dict(self):
        return {"sample_index": self._sample_index}


class VideoWebDataset(torch.utils.data.IterableDataset, torch.distributed.checkpoint.stateful.Stateful):
    def __init__(self, dataset_name: str, infinite: bool = True) -> None:
        super().__init__()

        self.dataset_name = dataset_name
        self.infinite = infinite

        data = datasets.load_dataset(dataset_name, split="train", streaming=True)
        data = data.rename_column("txt", "caption")
        for column_name in constants.SUPPORTED_VIDEO_FILE_EXTENSIONS:
            if column_name in data.column_names:
                data = data.cast_column(column_name, datasets.Video())
                data = data.rename_column(column_name, "video")

        self._data = data
        self._sample_index = 0

    def _get_data_iter(self):
        if self._sample_index == 0:
            return iter(self._data)

        if isinstance(self._data, datasets.Dataset) and self._sample_index >= len(self._data):
            return iter([])

        return iter(self._data.skip(self._sample_index))

    def __iter__(self):
        while True:
            for sample in self._get_data_iter():
                self._sample_index += 1
                yield sample

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                # Reset offset for the next iteration
                self._sample_index = 0
                logger.warning(f"Dataset {self.dataset_name} is being re-looped")

    def load_state_dict(self, state_dict):
        self._sample_index = state_dict["sample_index"]

    def state_dict(self):
        return {"sample_index": self._sample_index}


class ValidationDataset(torch.utils.data.IterableDataset):
    def __init__(self, filename: str):
        super().__init__()

        self.filename = pathlib.Path(filename)

        if not self.filename.exists():
            raise FileNotFoundError(f"File {self.filename.as_posix()} does not exist")

        if self.filename.suffix == ".csv":
            data = datasets.load_dataset("csv", data_files=self.filename.as_posix(), split="train")
        elif self.filename.suffix == ".json":
            data = datasets.load_dataset("json", data_files=self.filename.as_posix(), split="train", field="data")
        elif self.filename.suffix == ".parquet":
            data = datasets.load_dataset("parquet", data_files=self.filename.as_posix(), split="train")
        elif self.filename.suffix == ".arrow":
            data = datasets.load_dataset("arrow", data_files=self.filename.as_posix(), split="train")
        else:
            _SUPPORTED_FILE_FORMATS = [".csv", ".json", ".parquet", ".arrow"]
            raise ValueError(
                f"Unsupported file format {self.filename.suffix} for validation dataset. Supported formats are: {_SUPPORTED_FILE_FORMATS}"
            )

        self._data = data.to_iterable_dataset()

    def __iter__(self):
        for sample in self._data:
            # For consistency reasons, we mandate that "caption" is always present in the validation dataset.
            # However, since the model specifications use "prompt", we create an alias here.
            sample["prompt"] = sample["caption"]

            # Load image or video if the path is provided
            # TODO(aryan): need to handle custom columns here for control conditions
            sample["image"] = None
            sample["video"] = None

            if sample.get("image_path", None) is not None:
                image_path = pathlib.Path(sample["image_path"])
                if not image_path.is_file():
                    logger.warning(f"Image file {image_path.as_posix()} does not exist.")
                else:
                    sample["image"] = load_image(sample["image_path"])

            if sample.get("video_path", None) is not None:
                video_path = pathlib.Path(sample["video_path"])
                if not video_path.is_file():
                    logger.warning(f"Video file {video_path.as_posix()} does not exist.")
                else:
                    sample["video"] = load_video(sample["video_path"])

            sample = {k: v for k, v in sample.items() if v is not None}
            yield sample


# TODO(aryan): maybe write a test for this
def initialize_dataset(
    dataset_name_or_root: str, dataset_type: str = "video", streaming: bool = True, infinite: bool = False
) -> torch.utils.data.IterableDataset:
    # 1. If there is a metadata.json or metadata.jsonl or metadata.csv file, we use the
    #    ImageFolderDataset or VideoFolderDataset class respectively.
    # 2. If there is a list of .txt files sharing the same name with image extensions, we
    #    use the ImageCaptionFileDataset class.
    # 3. If there is a list of .txt files sharing the same name with video extensions, we
    #    use the VideoCaptionFileDataset class.
    # 4. If there is a dataset name, we use the ImageWebDataset or VideoWebDataset class.
    assert dataset_type in ["image", "video"]

    root = pathlib.Path(dataset_name_or_root)
    if not root.is_dir():
        return _initialize_webdataset(dataset_name_or_root, streaming, infinite, dataset_type)

    supported_metadata_files = ["metadata.json", "metadata.jsonl", "metadata.csv"]
    metadata_files = [root / metadata_file for metadata_file in supported_metadata_files]
    metadata_files = [metadata_file for metadata_file in metadata_files if metadata_file.exists()]

    dataset = None
    if len(metadata_files) == 0:
        raise ValueError(
            f"No metadata file found. Please ensure there is a metadata file named one of: {supported_metadata_files}."
        )
    elif len(metadata_files) > 1:
        raise ValueError("Found multiple metadata files. Please ensure there is only one metadata file.")
    else:
        if dataset_type == "image":
            dataset = ImageFolderDataset(root.as_posix())
        else:
            dataset = VideoFolderDataset(root.as_posix(), infinite=infinite)

    if dataset is None:
        if dataset_type == "image":
            dataset = ImageCaptionFileDataset(root.as_posix())
        else:
            dataset = VideoCaptionFileDataset(root.as_posix())

    return dataset


def _initialize_webdataset(
    dataset_name: str, dataset_type: str, infinite: bool = False
) -> torch.utils.data.IterableDataset:
    if dataset_type == "image":
        return ImageWebDataset(dataset_name, infinite=infinite)
    else:
        return VideoWebDataset(dataset_name, infinite=infinite)


def _read_caption_from_file(filename: str) -> str:
    with open(filename, "r") as f:
        return f.read().strip()


def _preprocess_image(image: PIL.Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    image = np.array(image).astype(np.float32)
    image = torch.from_numpy(image)
    image = image.permute(2, 0, 1).contiguous() / 127.5 - 1.0
    return image


def _preprocess_video(video: decord.VideoReader) -> torch.Tensor:
    video = video.get_batch(list(range(len(video))))
    video = video.permute(0, 3, 1, 2).contiguous()
    video = video.float() / 127.5 - 1.0
    return video
