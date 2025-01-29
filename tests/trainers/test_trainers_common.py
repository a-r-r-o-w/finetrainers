import sys
import tempfile
from pathlib import Path
from typing import Tuple

import torch
from huggingface_hub import snapshot_download


current_file = Path(__file__).resolve()
root_dir = current_file.parents[2]
sys.path.append(str(root_dir))

from finetrainers import Trainer  # noqa
from finetrainers.constants import (  # noqa
    PRECOMPUTED_CONDITIONS_DIR_NAME,
    PRECOMPUTED_DIR_NAME,
    PRECOMPUTED_LATENTS_DIR_NAME,
)
from finetrainers.utils.file_utils import string_to_filename  # noqa


def parse_resolution_bucket(resolution_bucket: str) -> Tuple[int, ...]:
    """Parse a resolution like '512x512' into a tuple of ints (512, 512)."""
    return tuple(map(int, resolution_bucket.split("x")))


class TrainerTestMixin:
    MODEL_NAME = None
    EXPECTED_PRECOMPUTATION_LATENT_KEYS = set()
    EXPECTED_LATENT_SHAPES = {}
    EXPECTED_PRECOMPUTATION_CONDITION_KEYS = set()
    EXPECTED_CONDITION_SHAPES = {}

    def get_training_args(self):
        raise NotImplementedError

    @property
    def latent_output_shape(self):
        raise NotImplementedError

    @property
    def condition_output_shape(self):
        raise NotImplementedError

    def populate_shapes(self):
        raise NotImplementedError

    def download_dataset_txt_format(self, cache_dir):
        return snapshot_download(repo_id="finetrainers/dummy-disney-dataset", repo_type="dataset", cache_dir=cache_dir)

    def get_precomputation_dir(self, training_args):
        """Return the path of the precomputation directory based on the training args."""
        cleaned_model_id = string_to_filename(training_args.pretrained_model_name_or_path)
        return Path(training_args.data_root) / f"{training_args.model_name}_{cleaned_model_id}_{PRECOMPUTED_DIR_NAME}"

    def tearDown(self):
        super().tearDown()
        self.EXPECTED_LATENT_SHAPES.clear()
        self.EXPECTED_CONDITION_SHAPES.clear()

    def _verify_precomputed_files(self, video_paths, all_conditions, all_latents):
        """Check that the correct number of precomputed files exist and have the right keys."""
        assert len(video_paths) == len(all_conditions), "Mismatch in conditions file count"
        assert len(video_paths) == len(all_latents), "Mismatch in latents file count"

        for latent, condition in zip(all_latents, all_conditions):
            latent_keys = sorted(set(torch.load(latent, weights_only=True).keys()))
            condition_keys = sorted(set(torch.load(condition, weights_only=True).keys()))
            assert latent_keys == sorted(
                self.EXPECTED_PRECOMPUTATION_LATENT_KEYS
            ), f"Unexpected latent keys: {latent_keys}"
            assert condition_keys == sorted(
                self.EXPECTED_PRECOMPUTATION_CONDITION_KEYS
            ), f"Unexpected condition keys: {condition_keys}"

    def _verify_shapes(self, latent_files, condition_files):
        """Check that the shapes of latents and conditions match expected shapes."""
        self.populate_shapes()
        for l_path, c_path in zip(latent_files, condition_files):
            latent = torch.load(l_path, weights_only=True, map_location="cpu")
            condition = torch.load(c_path, weights_only=True, map_location="cpu")

            for key in self.EXPECTED_PRECOMPUTATION_LATENT_KEYS:
                if not torch.is_tensor(latent[key]):
                    continue
                expected = self.EXPECTED_LATENT_SHAPES[key]
                original = tuple(latent[key].shape[1:])
                assert (
                    original == expected
                ), f"Latent shape mismatch for key: {key}. expected={expected}, got={original}"

            for key in self.EXPECTED_PRECOMPUTATION_CONDITION_KEYS:
                if not torch.is_tensor(condition[key]):
                    continue
                expected = self.EXPECTED_CONDITION_SHAPES[key]
                original = tuple(condition[key].shape[1:])
                assert (
                    original == expected
                ), f"Condition shape mismatch for key: {key}. expected={expected}, got={original}"

    def _setup_trainer(self, tmpdir):
        """
        Helper method to reduce duplication across tests.
        Creates and returns a trainer, along with updated training args.
        """
        training_args = self.get_training_args()
        training_args.data_root = Path(self.download_dataset_txt_format(cache_dir=tmpdir))
        training_args.video_column = "videos.txt"
        training_args.caption_column = "prompt.txt"
        training_args.output_dir = tmpdir

        trainer = Trainer(training_args)
        # Trainer may update the training_args internally, so refresh the reference
        training_args = trainer.args

        trainer.prepare_dataset()
        trainer.prepare_models()
        return trainer, training_args

    def test_precomputation_txt_format_creates_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer, training_args = self._setup_trainer(tmpdir)

            # Load video paths (only needed in this test)
            with open(training_args.data_root / training_args.video_column, "r", encoding="utf-8") as file:
                video_paths = [training_args.data_root / line.strip() for line in file if line.strip()]

            trainer.prepare_precomputations()

            precomputation_dir = self.get_precomputation_dir(training_args)
            conditions_dir = precomputation_dir / PRECOMPUTED_CONDITIONS_DIR_NAME
            latents_dir = precomputation_dir / PRECOMPUTED_LATENTS_DIR_NAME

            assert precomputation_dir.exists(), f"Precomputed dir not found: {precomputation_dir}"
            assert conditions_dir.exists(), f"Conditions dir not found: {conditions_dir}"
            assert latents_dir.exists(), f"Latents dir not found: {latents_dir}"

            all_conditions = list(conditions_dir.glob("*.pt"))
            all_latents = list(latents_dir.glob("*.pt"))

            self._verify_precomputed_files(video_paths, all_conditions, all_latents)

    def test_precomputation_txt_format_matches_shapes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer, training_args = self._setup_trainer(tmpdir)

            with self.assertLogs(level="INFO") as captured:
                trainer.prepare_precomputations()
            assert any(
                "Precomputed data not found. Running precomputation." in msg for msg in captured.output
            ), "Expected info log about missing precomputed data."

            precomputation_dir = self.get_precomputation_dir(training_args)
            conditions_dir = precomputation_dir / PRECOMPUTED_CONDITIONS_DIR_NAME
            latents_dir = precomputation_dir / PRECOMPUTED_LATENTS_DIR_NAME

            latent_files = list(latents_dir.glob("*.pt"))
            condition_files = list(conditions_dir.glob("*.pt"))

            self._verify_shapes(latent_files, condition_files)

    def test_precomputation_txt_format_no_redo(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer, _ = self._setup_trainer(tmpdir)

            # should create new precomputations
            trainer.prepare_precomputations()

            # should detect existing precomputations and not redo
            with self.assertLogs(level="INFO") as captured:
                trainer.prepare_precomputations()

            assert any(
                "Precomputed conditions and latents found. Loading precomputed data" in msg for msg in captured.output
            ), "Expected info log about found precomputations."
