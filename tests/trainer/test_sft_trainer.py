# torchrun --nnodes=1 --nproc_per_node=1 -m pytest -s tests/trainer/test_sft_trainer.py

import os
import pathlib
import sys
import tempfile
import unittest

from diffusers.utils import export_to_video
from PIL import Image


os.environ["WANDB_MODE"] = "disabled"
os.environ["FINETRAINERS_LOG_LEVEL"] = "DEBUG"

project_root = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from finetrainers import Args, SFTTrainer, get_logger  # noqa

from ..models.dummy.base_specification import DummyLTXVideoModelSpecification  # noqa


logger = get_logger()


class SFTTrainerFastTests(unittest.TestCase):
    num_data_files = 3

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data_files = []
        for i in range(self.num_data_files):
            data_file = pathlib.Path(self.tmpdir.name) / f"{i}.mp4"
            export_to_video([Image.new("RGB", (64, 64))] * 4, data_file.as_posix(), fps=2)
            self.data_files.append(data_file.as_posix())

        csv_filename = pathlib.Path(self.tmpdir.name) / "metadata.csv"
        with open(csv_filename.as_posix(), "w") as f:
            f.write("file_name,caption\n")
            for i in range(self.num_data_files):
                prompt = f"A cat ruling the world - {i}"
                f.write(f'{i}.mp4,"{prompt}"\n')

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_train_lora(self):
        args = Args()
        args.parallel_backend = "ptd"
        args.data_root = self.tmpdir.name
        args.training_type = "lora"
        args.train_steps = 5
        args.rank = 4
        args.gradient_checkpointing = True
        args.output_dir = self.tmpdir.name
        args.precomputation_items = self.num_data_files - 1
        args.precomputation_dir = os.path.join(self.tmpdir.name, "precomputed")

        model_specification = DummyLTXVideoModelSpecification()
        trainer = SFTTrainer(args, model_specification)

        trainer.run()
