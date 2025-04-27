import os
import tempfile
import unittest

from finetrainers.data import (
    InMemoryDistributedDataPreprocessor,
    PrecomputedDistributedDataPreprocessor,
    VideoCaptionFilePairDataset,
    initialize_preprocessor,
    wrap_iterable_dataset_for_preprocessing,
)
from finetrainers.data.precomputation import PRECOMPUTED_DATA_DIR
from finetrainers.utils import find_files

from .utils import create_dummy_directory_structure


class PreprocessorFastTests(unittest.TestCase):
    def setUp(self):
        self.rank = 0
        self.world_size = 1
        self.num_items = 3
        self.processor_fn = {
            "latent": self._latent_processor_fn,
            "condition": self._condition_processor_fn,
        }
        self.save_dir = tempfile.TemporaryDirectory()

        directory_structure = [
            "0.mp4",
            "1.mp4",
            "2.mp4",
            "0.txt",
            "1.txt",
            "2.txt",
        ]
        create_dummy_directory_structure(
            directory_structure, self.save_dir, self.num_items, "a cat ruling the world", "mp4"
        )

        dataset = VideoCaptionFilePairDataset(self.save_dir.name, infinite=True)
        dataset = wrap_iterable_dataset_for_preprocessing(
            dataset,
            dataset_type="video",
            config={
                "video_resolution_buckets": [[2, 32, 32]],
                "reshape_mode": "bicubic",
            },
        )
        self.dataset = dataset

    def tearDown(self):
        self.save_dir.cleanup()

    @staticmethod
    def _latent_processor_fn(**data):
        video = data["video"]
        video = video[:, :, :16, :16]
        data["video"] = video
        return data

    @staticmethod
    def _condition_processor_fn(**data):
        caption = data["caption"]
        caption = caption + " surrounded by mystical aura"
        data["caption"] = caption
        return data

    def test_initialize_preprocessor(self):
        preprocessor = initialize_preprocessor(
            self.rank,
            self.world_size,
            self.num_items,
            self.processor_fn,
            self.save_dir.name,
            enable_precomputation=False,
        )
        self.assertIsInstance(preprocessor, InMemoryDistributedDataPreprocessor)

        preprocessor = initialize_preprocessor(
            self.rank,
            self.world_size,
            self.num_items,
            self.processor_fn,
            self.save_dir.name,
            enable_precomputation=True,
        )
        self.assertIsInstance(preprocessor, PrecomputedDistributedDataPreprocessor)

    def test_in_memory_preprocessor_consume(self):
        data_iterator = iter(self.dataset)
        preprocessor = initialize_preprocessor(
            self.rank,
            self.world_size,
            self.num_items,
            self.processor_fn,
            self.save_dir.name,
            enable_precomputation=False,
        )

        condition_iterator = preprocessor.consume(
            "condition", components={}, data_iterator=data_iterator, cache_samples=True
        )
        latent_iterator = preprocessor.consume(
            "latent", components={}, data_iterator=data_iterator, use_cached_samples=True, drop_samples=True
        )

        self.assertFalse(preprocessor.requires_data)
        for _ in range(self.num_items):
            condition_item = next(condition_iterator)
            latent_item = next(latent_iterator)
            self.assertIn("caption", condition_item)
            self.assertIn("video", latent_item)
            self.assertEqual(condition_item["caption"], "a cat ruling the world surrounded by mystical aura")
            self.assertEqual(latent_item["video"].shape[-2:], (16, 16))
        self.assertTrue(preprocessor.requires_data)

    def test_in_memory_preprocessor_consume_once(self):
        data_iterator = iter(self.dataset)
        preprocessor = initialize_preprocessor(
            self.rank,
            self.world_size,
            self.num_items,
            self.processor_fn,
            self.save_dir.name,
            enable_precomputation=False,
        )

        condition_iterator = preprocessor.consume_once(
            "condition", components={}, data_iterator=data_iterator, cache_samples=True
        )
        latent_iterator = preprocessor.consume_once(
            "latent", components={}, data_iterator=data_iterator, use_cached_samples=True, drop_samples=True
        )

        self.assertFalse(preprocessor.requires_data)
        for _ in range(self.num_items):
            condition_item = next(condition_iterator)
            latent_item = next(latent_iterator)
            self.assertIn("caption", condition_item)
            self.assertIn("video", latent_item)
            self.assertEqual(condition_item["caption"], "a cat ruling the world surrounded by mystical aura")
            self.assertEqual(latent_item["video"].shape[-2:], (16, 16))
        self.assertFalse(preprocessor.requires_data)

    def test_precomputed_preprocessor_consume(self):
        data_iterator = iter(self.dataset)
        preprocessor = initialize_preprocessor(
            self.rank,
            self.world_size,
            self.num_items,
            self.processor_fn,
            self.save_dir.name,
            enable_precomputation=True,
        )

        condition_iterator = preprocessor.consume(
            "condition", components={}, data_iterator=data_iterator, cache_samples=True
        )
        latent_iterator = preprocessor.consume(
            "latent", components={}, data_iterator=data_iterator, use_cached_samples=True, drop_samples=True
        )

        precomputed_data_dir = os.path.join(self.save_dir.name, PRECOMPUTED_DATA_DIR)
        condition_file_list = find_files(precomputed_data_dir, "condition-*")
        latent_file_list = find_files(precomputed_data_dir, "latent-*")
        self.assertEqual(len(condition_file_list), 3)
        self.assertEqual(len(latent_file_list), 3)

        self.assertFalse(preprocessor.requires_data)
        for _ in range(self.num_items):
            condition_item = next(condition_iterator)
            latent_item = next(latent_iterator)
            self.assertIn("caption", condition_item)
            self.assertIn("video", latent_item)
            self.assertEqual(condition_item["caption"], "a cat ruling the world surrounded by mystical aura")
            self.assertEqual(latent_item["video"].shape[-2:], (16, 16))
        self.assertTrue(preprocessor.requires_data)

    def test_precomputed_preprocessor_consume_once(self):
        data_iterator = iter(self.dataset)
        preprocessor = initialize_preprocessor(
            self.rank,
            self.world_size,
            self.num_items,
            self.processor_fn,
            self.save_dir.name,
            enable_precomputation=True,
        )

        condition_iterator = preprocessor.consume_once(
            "condition", components={}, data_iterator=data_iterator, cache_samples=True
        )
        latent_iterator = preprocessor.consume_once(
            "latent", components={}, data_iterator=data_iterator, use_cached_samples=True, drop_samples=True
        )

        precomputed_data_dir = os.path.join(self.save_dir.name, PRECOMPUTED_DATA_DIR)
        condition_file_list = find_files(precomputed_data_dir, "condition-*")
        latent_file_list = find_files(precomputed_data_dir, "latent-*")
        self.assertEqual(len(condition_file_list), 3)
        self.assertEqual(len(latent_file_list), 3)

        self.assertFalse(preprocessor.requires_data)
        for _ in range(self.num_items):
            condition_item = next(condition_iterator)
            latent_item = next(latent_iterator)
            self.assertIn("caption", condition_item)
            self.assertIn("video", latent_item)
            self.assertEqual(condition_item["caption"], "a cat ruling the world surrounded by mystical aura")
            self.assertEqual(latent_item["video"].shape[-2:], (16, 16))
        self.assertFalse(preprocessor.requires_data)
