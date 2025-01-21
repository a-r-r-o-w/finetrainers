import sys
from pathlib import Path


current_file = Path(__file__).resolve()
root_dir = current_file.parents[1]
sys.path.append(str(root_dir))


import os  # noqa
import tempfile  # noqa

from huggingface_hub import snapshot_download  # noqa

from finetrainers import Trainer  # noqa
from finetrainers.constants import PRECOMPUTED_CONDITIONS_DIR_NAME, PRECOMPUTED_DIR_NAME, PRECOMPUTED_LATENTS_DIR_NAME  # noqa
from finetrainers.utils.file_utils import string_to_filename  # noqa


class TrainerTestMixin:
    model_name = None

    def get_training_args(self):
        raise NotImplementedError

    def download_dataset_txt_format(self, cache_dir):
        path = snapshot_download(repo_id="finetrainers/dummy-disney-dataset", repo_type="dataset", cache_dir=cache_dir)
        return path

    def test_precomputation_txt_format(self):
        # Here we assume the dataset is formatted like:
        # https://huggingface.co/datasets/Wild-Heart/Disney-VideoGeneration-Dataset/tree/main
        training_args = self.get_training_args()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Prepare remaining args.
            training_args.data_root = Path(self.download_dataset_txt_format(cache_dir=tmpdir))

            training_args.video_column = "videos.txt"
            training_args.caption_column = "prompt.txt"
            with open(f"{training_args.data_root}/{training_args.video_column}", "r", encoding="utf-8") as file:
                video_paths = [
                    training_args.data_root.joinpath(line.strip())
                    for line in file.readlines()
                    if len(line.strip()) > 0
                ]

            # Initialize trainer.
            training_args.output_dir = tmpdir
            trainer = Trainer(training_args)
            training_args = trainer.args

            # Perform precomputations.
            trainer.prepare_dataset()
            trainer.prepare_models()
            trainer.prepare_precomputations()

            cleaned_model_id = string_to_filename(training_args.pretrained_model_name_or_path)
            precomputation_dir = (
                Path(training_args.data_root) / f"{training_args.model_name}_{cleaned_model_id}_{PRECOMPUTED_DIR_NAME}"
            )

            # Checks.
            conditions_dir = precomputation_dir / PRECOMPUTED_CONDITIONS_DIR_NAME
            latents_dir = precomputation_dir / PRECOMPUTED_LATENTS_DIR_NAME
            assert os.path.exists(
                precomputation_dir
            ), f"Precomputation wasn't successful. Couldn't find the precomputed dir: {os.listdir(training_args.data_root)=}\n"
            assert os.path.exists(conditions_dir), f"conditions dir ({str(conditions_dir)}) doesn't exist"
            assert os.path.exists(latents_dir), f"latents dir ({str(latents_dir)}) doesn't exist"
            assert len(video_paths) == len([p for p in conditions_dir.glob("*.pt")])  # noqa
            assert len(video_paths) == len([p for p in latents_dir.glob("*.pt")])  # noqa
