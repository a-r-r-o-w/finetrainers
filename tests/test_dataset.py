import pathlib
import sys
import tempfile
import unittest

import torch
from diffusers.utils import export_to_video
from PIL import Image, JpegImagePlugin


project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import decord  # noqa

from finetrainers.data import (  # noqa
    ImageCaptionFileDataset,
    ImageFolderDataset,
    VideoCaptionFileDataset,
    VideoFolderDataset,
    VideoWebDataset,
    ValidationDataset,
)
from finetrainers.data.utils import find_files  # noqa


class ImageCaptionFileDatasetFastTests(unittest.TestCase):
    def setUp(self):
        num_data_files = 3

        self.tmpdir = tempfile.TemporaryDirectory()
        self.caption_files = []
        self.data_files = []
        for _ in range(num_data_files):
            caption_file = tempfile.NamedTemporaryFile(dir=self.tmpdir.name, suffix=".txt", delete=False)
            self.caption_files.append(caption_file.name)
            data_file = pathlib.Path(caption_file.name).with_suffix(".jpg")
            Image.new("RGB", (64, 64)).save(data_file.as_posix())
            self.data_files.append((pathlib.Path(self.tmpdir.name) / data_file).as_posix())

        self.dataset = ImageCaptionFileDataset(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_getitem(self):
        self.assertEqual(len(self.dataset), 3)
        for index in range(3):
            item = self.dataset[index]
            self.assertEqual(item["caption"], "")
            self.assertIsInstance(item["image"], JpegImagePlugin.JpegImageFile)

    def test_direct_access(self):
        self.assertEqual(self.dataset._data["caption"], [""] * 3)
        self.assertEqual(len(self.dataset._data["image"]), 3)


class ImageFolderDatasetFastTests(unittest.TestCase):
    def setUp(self):
        num_data_files = 3

        self.num_data_files = num_data_files
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data_files = []
        for i in range(num_data_files):
            data_file = pathlib.Path(self.tmpdir.name) / f"{i}.jpg"
            Image.new("RGB", (64, 64)).save(data_file.as_posix())
            self.data_files.append(data_file.as_posix())

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_getitem_csv(self):
        csv_filename = pathlib.Path(self.tmpdir.name) / "metadata.csv"
        with open(csv_filename.as_posix(), "w") as f:
            f.write("file_name,label\n")
            for i in range(self.num_data_files):
                f.write(f"{i}.jpg,{i}\n")

        dataset = ImageFolderDataset(self.tmpdir.name)

        self.assertEqual(len(dataset), 3)
        for index in range(3):
            item = dataset[index]
            self.assertIsInstance(item["image"], JpegImagePlugin.JpegImageFile)

    def test_getitem_jsonl(self):
        jsonl_filename = pathlib.Path(self.tmpdir.name) / "metadata.jsonl"
        with open(jsonl_filename.as_posix(), "w") as f:
            for i in range(self.num_data_files):
                f.write(f'{{"file_name": "{i}.jpg", "label": {i}}}\n')

        dataset = ImageFolderDataset(self.tmpdir.name)

        self.assertEqual(len(dataset), 3)
        for index in range(3):
            item = dataset[index]
            self.assertIsInstance(item["image"], JpegImagePlugin.JpegImageFile)


class VideoCaptionFileDatasetFastTests(unittest.TestCase):
    def setUp(self):
        num_data_files = 3

        self.tmpdir = tempfile.TemporaryDirectory()
        self.caption_files = []
        self.data_files = []
        for _ in range(num_data_files):
            caption_file = tempfile.NamedTemporaryFile(dir=self.tmpdir.name, suffix=".txt", delete=False)
            self.caption_files.append(caption_file.name)
            data_file = pathlib.Path(caption_file.name).with_suffix(".mp4")
            export_to_video([Image.new("RGB", (64, 64))] * 4, data_file.as_posix(), fps=2)
            self.data_files.append((pathlib.Path(self.tmpdir.name) / data_file).as_posix())

        self.dataset = VideoCaptionFileDataset(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_getitem(self):
        self.assertEqual(len(self.dataset), 3)
        for index in range(3):
            item = self.dataset[index]
            self.assertEqual(item["caption"], "")
            self.assertEqual(len(item["video"]), 4)
            self.assertEqual(item["video"].get_batch([-1]).shape, (1, 64, 64, 3))
            self.assertTrue(torch.is_tensor(item["video"].get_batch(list(range(2)))))

    def test_direct_access(self):
        self.assertEqual(self.dataset._data["caption"], [""] * 3)
        self.assertEqual(len(self.dataset._data["video"]), 3)
        self.assertEqual(len(self.dataset._data["video"][-1]), 4)
        self.assertEqual(self.dataset._data["video"][-1].get_batch([-1]).shape, (1, 64, 64, 3))
        self.assertTrue(torch.is_tensor(self.dataset._data["video"][-1].get_batch(list(range(2)))))


class VideoFolderDatasetFastTests(unittest.TestCase):
    def setUp(self):
        num_data_files = 3

        self.num_data_files = num_data_files
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data_files = []
        for i in range(num_data_files):
            data_file = pathlib.Path(self.tmpdir.name) / f"{i}.mp4"
            export_to_video([Image.new("RGB", (64, 64))] * 4, data_file.as_posix(), fps=2)
            self.data_files.append(data_file.as_posix())

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_getitem_csv(self):
        csv_filename = pathlib.Path(self.tmpdir.name) / "metadata.csv"
        with open(csv_filename.as_posix(), "w") as f:
            f.write("file_name,label\n")
            for i in range(self.num_data_files):
                f.write(f"{i}.mp4,{i}\n")

        dataset = VideoFolderDataset(self.tmpdir.name)

        self.assertEqual(len(dataset), 3)
        for index in range(3):
            item = dataset[index]
            self.assertEqual(len(item["video"]), 4)
            self.assertEqual(item["video"].get_batch([-1]).shape, (1, 64, 64, 3))
            self.assertTrue(torch.is_tensor(item["video"].get_batch(list(range(2)))))

    def test_getitem_jsonl(self):
        jsonl_filename = pathlib.Path(self.tmpdir.name) / "metadata.jsonl"
        with open(jsonl_filename.as_posix(), "w") as f:
            for i in range(self.num_data_files):
                f.write(f'{{"file_name": "{i}.mp4", "label": {i}}}\n')

        dataset = VideoFolderDataset(self.tmpdir.name)

        self.assertEqual(len(dataset), 3)
        for index in range(3):
            item = dataset[index]
            self.assertEqual(len(item["video"]), 4)
            self.assertEqual(item["video"].get_batch([-1]).shape, (1, 64, 64, 3))
            self.assertTrue(torch.is_tensor(item["video"].get_batch(list(range(2)))))


class VideoWebDatasetFastTests(unittest.TestCase):
    def setUp(self):
        self.num_data_files = 15
        # Should be converted into an iterable dataset internally
        self.map_dataset = VideoWebDataset("finetrainers/dummy-squish-wds", streaming=False, infinite=False)
        self.iterable_dataset = VideoWebDataset("finetrainers/dummy-squish-wds", streaming=True, infinite=False)

    def test_getitem(self):
        self.assertTrue(isinstance(self.map_dataset._data, torch.utils.data.IterableDataset))
        for index, item in enumerate(self.map_dataset):
            if index > 2:
                break
            self.assertIsInstance(item["video"], decord.VideoReader)
            self.assertEqual(len(item["video"].get_batch([0, 1, 2, 3])), 4)
        for index, item in enumerate(self.iterable_dataset):
            if index > 2:
                break
            self.assertIsInstance(item["video"], decord.VideoReader)
            self.assertEqual(len(item["video"].get_batch([0, 1, 2, 3])), 4)


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
