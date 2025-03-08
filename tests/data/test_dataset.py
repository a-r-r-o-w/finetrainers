import pathlib
import tempfile
import unittest

import torch
from PIL import Image

from finetrainers.data import (
    ImageCaptionFilePairDataset,
    ImageFileCaptionFileListDataset,
    ImageFolderDataset,
    ValidationDataset,
    VideoCaptionFilePairDataset,
    VideoFileCaptionFileListDataset,
    VideoFolderDataset,
    VideoWebDataset,
    initialize_dataset,
)
from finetrainers.data.utils import find_files

from .utils import create_dummy_directory_structure


import decord  # isort: skip


class DatasetTesterMixin:
    num_data_files = None
    directory_structure = None
    caption = "A cat ruling the world"
    metadata_extension = None

    def setUp(self):
        if self.num_data_files is None:
            raise ValueError("num_data_files is not defined")
        if self.directory_structure is None:
            raise ValueError("dataset_structure is not defined")

        self.tmpdir = tempfile.TemporaryDirectory()
        create_dummy_directory_structure(
            self.directory_structure, self.tmpdir, self.num_data_files, self.caption, self.metadata_extension
        )

    def tearDown(self):
        self.tmpdir.cleanup()


class ImageDatasetTesterMixin(DatasetTesterMixin):
    metadata_extension = "jpg"


class VideoDatasetTesterMixin(DatasetTesterMixin):
    metadata_extension = "mp4"


class ImageCaptionFilePairDatasetFastTests(ImageDatasetTesterMixin, unittest.TestCase):
    num_data_files = 3
    directory_structure = [
        "0.jpg",
        "1.jpg",
        "2.jpg",
        "0.txt",
        "1.txt",
        "2.txt",
    ]

    def setUp(self):
        super().setUp()
        self.dataset = ImageCaptionFilePairDataset(self.tmpdir.name, infinite=False)

    def test_getitem(self):
        iterator = iter(self.dataset)
        for _ in range(self.num_data_files):
            item = next(iterator)
            self.assertEqual(item["caption"], self.caption)
            self.assertTrue(torch.is_tensor(item["image"]))
            self.assertEqual(item["image"].shape, (3, 64, 64))

    def test_initialize_dataset(self):
        dataset = initialize_dataset(self.tmpdir.name, "image", infinite=False)
        self.assertIsInstance(dataset, ImageCaptionFilePairDataset)


class ImageFileCaptionFileListDatasetFastTests(ImageDatasetTesterMixin, unittest.TestCase):
    num_data_files = 3
    directory_structure = [
        "prompts.txt",
        "images.txt",
        "images/",
        "images/0.jpg",
        "images/1.jpg",
        "images/2.jpg",
    ]

    def setUp(self):
        super().setUp()
        self.dataset = ImageFileCaptionFileListDataset(self.tmpdir.name, infinite=False)

    def test_getitem(self):
        iterator = iter(self.dataset)
        for i in range(3):
            item = next(iterator)
            self.assertEqual(item["caption"], self.caption)
            self.assertTrue(torch.is_tensor(item["image"]))
            self.assertEqual(item["image"].shape, (3, 64, 64))

    def test_initialize_dataset(self):
        dataset = initialize_dataset(self.tmpdir.name, "image", infinite=False)
        self.assertIsInstance(dataset, ImageFileCaptionFileListDataset)


class ImageFolderDatasetFastTests___CSV(ImageDatasetTesterMixin, unittest.TestCase):
    num_data_files = 3
    directory_structure = [
        "metadata.csv",
        "0.jpg",
        "1.jpg",
        "2.jpg",
    ]

    def setUp(self):
        super().setUp()
        self.dataset = ImageFolderDataset(self.tmpdir.name, infinite=False)

    def test_getitem(self):
        iterator = iter(self.dataset)
        for _ in range(3):
            item = next(iterator)
            self.assertIn("caption", item)
            self.assertEqual(item["caption"], self.caption)
            self.assertTrue(torch.is_tensor(item["image"]))

    def test_initialize_dataset(self):
        dataset = initialize_dataset(self.tmpdir.name, "image", infinite=False)
        self.assertIsInstance(dataset, ImageFolderDataset)


class ImageFolderDatasetFastTests___JSONL(ImageDatasetTesterMixin, unittest.TestCase):
    num_data_files = 3
    directory_structure = [
        "metadata.jsonl",
        "0.jpg",
        "1.jpg",
        "2.jpg",
    ]

    def setUp(self):
        super().setUp()
        self.dataset = ImageFolderDataset(self.tmpdir.name, infinite=False)

    def test_getitem(self):
        iterator = iter(self.dataset)
        for _ in range(3):
            item = next(iterator)
            self.assertIn("caption", item)
            self.assertEqual(item["caption"], self.caption)
            self.assertTrue(torch.is_tensor(item["image"]))

    def test_initialize_dataset(self):
        dataset = initialize_dataset(self.tmpdir.name, "image", infinite=False)
        self.assertIsInstance(dataset, ImageFolderDataset)


class VideoCaptionFilePairDatasetFastTests(VideoDatasetTesterMixin, unittest.TestCase):
    num_data_files = 3
    directory_structure = [
        "0.mp4",
        "1.mp4",
        "2.mp4",
        "0.txt",
        "1.txt",
        "2.txt",
    ]

    def setUp(self):
        super().setUp()
        self.dataset = VideoCaptionFilePairDataset(self.tmpdir.name, infinite=False)

    def test_getitem(self):
        iterator = iter(self.dataset)
        for _ in range(self.num_data_files):
            item = next(iterator)
            self.assertEqual(item["caption"], self.caption)
            self.assertTrue(torch.is_tensor(item["video"]))
            self.assertEqual(len(item["video"]), 4)
            self.assertEqual(item["video"][0].shape, (3, 64, 64))

    def test_initialize_dataset(self):
        dataset = initialize_dataset(self.tmpdir.name, "video", infinite=False)
        self.assertIsInstance(dataset, VideoCaptionFilePairDataset)


class VideoFileCaptionFileListDatasetFastTests(VideoDatasetTesterMixin, unittest.TestCase):
    num_data_files = 3
    directory_structure = [
        "prompts.txt",
        "videos.txt",
        "videos/",
        "videos/0.mp4",
        "videos/1.mp4",
        "videos/2.mp4",
    ]

    def setUp(self):
        super().setUp()
        self.dataset = VideoFileCaptionFileListDataset(self.tmpdir.name, infinite=False)

    def test_getitem(self):
        iterator = iter(self.dataset)
        for _ in range(3):
            item = next(iterator)
            self.assertEqual(item["caption"], self.caption)
            self.assertTrue(torch.is_tensor(item["video"]))
            self.assertEqual(len(item["video"]), 4)
            self.assertEqual(item["video"][0].shape, (3, 64, 64))

    def test_initialize_dataset(self):
        dataset = initialize_dataset(self.tmpdir.name, "video", infinite=False)
        self.assertIsInstance(dataset, VideoFileCaptionFileListDataset)


class VideoFolderDatasetFastTests___CSV(VideoDatasetTesterMixin, unittest.TestCase):
    num_data_files = 3
    directory_structure = [
        "metadata.csv",
        "0.mp4",
        "1.mp4",
        "2.mp4",
    ]

    def setUp(self):
        super().setUp()
        self.dataset = VideoFolderDataset(self.tmpdir.name, infinite=False)

    def test_getitem(self):
        iterator = iter(self.dataset)
        for _ in range(3):
            item = next(iterator)
            self.assertIn("caption", item)
            self.assertEqual(item["caption"], self.caption)
            self.assertTrue(torch.is_tensor(item["video"]))
            self.assertEqual(len(item["video"]), 4)
            self.assertEqual(item["video"][0].shape, (3, 64, 64))

    def test_initialize_dataset(self):
        dataset = initialize_dataset(self.tmpdir.name, "video", infinite=False)
        self.assertIsInstance(dataset, VideoFolderDataset)


class VideoFolderDatasetFastTests___JSONL(VideoDatasetTesterMixin, unittest.TestCase):
    num_data_files = 3
    directory_structure = [
        "metadata.jsonl",
        "0.mp4",
        "1.mp4",
        "2.mp4",
    ]

    def setUp(self):
        super().setUp()
        self.dataset = VideoFolderDataset(self.tmpdir.name, infinite=False)

    def test_getitem(self):
        iterator = iter(self.dataset)
        for _ in range(3):
            item = next(iterator)
            self.assertIn("caption", item)
            self.assertEqual(item["caption"], self.caption)
            self.assertTrue(torch.is_tensor(item["video"]))
            self.assertEqual(len(item["video"]), 4)
            self.assertEqual(item["video"][0].shape, (3, 64, 64))

    def test_initialize_dataset(self):
        dataset = initialize_dataset(self.tmpdir.name, "video", infinite=False)
        self.assertIsInstance(dataset, VideoFolderDataset)


class ImageWebDatasetFastTests(unittest.TestCase):
    # TODO(aryan): setup a dummy dataset
    pass


class VideoWebDatasetFastTests(unittest.TestCase):
    def setUp(self):
        self.num_data_files = 15
        self.dataset = VideoWebDataset("finetrainers/dummy-squish-wds", infinite=False)

    def test_getitem(self):
        for index, item in enumerate(self.dataset):
            if index > 2:
                break
            self.assertIsInstance(item["video"], decord.VideoReader)
            self.assertEqual(len(item["video"].get_batch([0, 1, 2, 3])), 4)

    def test_initialize_dataset(self):
        dataset = initialize_dataset("finetrainers/dummy-squish-wds", "video", infinite=False)
        self.assertIsInstance(dataset, VideoWebDataset)


class DatasetUtilsFastTests(unittest.TestCase):
    def test_find_files_depth_0(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = tempfile.NamedTemporaryFile(dir=tmpdir, suffix=".txt", delete=False)
            file2 = tempfile.NamedTemporaryFile(dir=tmpdir, suffix=".txt", delete=False)
            file3 = tempfile.NamedTemporaryFile(dir=tmpdir, suffix=".txt", delete=False)

            files = find_files(tmpdir, "*.txt")
            self.assertEqual(len(files), 3)
            self.assertIn(file1.name, files)
            self.assertIn(file2.name, files)
            self.assertIn(file3.name, files)

    def test_find_files_depth_n(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dir1 = tempfile.TemporaryDirectory(dir=tmpdir)
            dir2 = tempfile.TemporaryDirectory(dir=dir1.name)
            file1 = tempfile.NamedTemporaryFile(dir=dir1.name, suffix=".txt", delete=False)
            file2 = tempfile.NamedTemporaryFile(dir=dir2.name, suffix=".txt", delete=False)

            files = find_files(tmpdir, "*.txt", depth=1)
            self.assertEqual(len(files), 1)
            self.assertIn(file1.name, files)
            self.assertNotIn(file2.name, files)

            files = find_files(tmpdir, "*.txt", depth=2)
            self.assertEqual(len(files), 2)
            self.assertIn(file1.name, files)
            self.assertIn(file2.name, files)
            self.assertNotIn(dir1.name, files)
            self.assertNotIn(dir2.name, files)


class ValidationDatasetFastTests(unittest.TestCase):
    def setUp(self):
        num_data_files = 3

        self.tmpdir = tempfile.TemporaryDirectory()
        metadata_filename = pathlib.Path(self.tmpdir.name) / "metadata.csv"

        with open(metadata_filename, "w") as f:
            f.write("caption,image_path,video_path\n")
            for i in range(num_data_files):
                Image.new("RGB", (64, 64)).save((pathlib.Path(self.tmpdir.name) / f"{i}.jpg").as_posix())
                f.write(f"test caption,{self.tmpdir.name}/{i}.jpg,\n")

        self.dataset = ValidationDataset(metadata_filename.as_posix())

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_getitem(self):
        for i, data in enumerate(self.dataset):
            self.assertEqual(data["image_path"], f"{self.tmpdir.name}/{i}.jpg")
            self.assertIsInstance(data["image"], Image.Image)
            self.assertEqual(data["image"].size, (64, 64))
