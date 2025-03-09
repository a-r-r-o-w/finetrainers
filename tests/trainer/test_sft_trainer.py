# torchrun --nnodes=1 --nproc_per_node=1 -m pytest -s tests/trainer/test_sft_trainer.py

import json
import os
import pathlib
import tempfile
import time
import unittest

import pytest
from diffusers.utils import export_to_video
from parameterized import parameterized
from PIL import Image

from finetrainers import BaseArgs, SFTTrainer, TrainingType, get_logger


os.environ["WANDB_MODE"] = "disabled"
os.environ["FINETRAINERS_LOG_LEVEL"] = "INFO"

from ..models.cogvideox.base_specification import DummyCogVideoXModelSpecification  # noqa
from ..models.cogview4.base_specification import DummyCogView4ModelSpecification  # noqa
from ..models.hunyuan_video.base_specification import DummyHunyuanVideoModelSpecification  # noqa
from ..models.ltx_video.base_specification import DummyLTXVideoModelSpecification  # noqa
from ..models.wan.base_specification import DummyWanModelSpecification  # noqa


logger = get_logger()


@pytest.fixture(autouse=True)
def slow_down_tests():
    yield
    # Sleep between each test so that process groups are cleaned and resources are released.
    # Not doing so seems to randomly trigger some test failures, which wouldn't fail if run individually.
    # !!!Look into this in future!!!
    time.sleep(3)


class SFTTrainerFastTestsMixin:
    model_specification_cls = None
    num_data_files = 4
    num_frames = 4
    height = 64
    width = 64

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data_files = []
        for i in range(self.num_data_files):
            data_file = pathlib.Path(self.tmpdir.name) / f"{i}.mp4"
            export_to_video(
                [Image.new("RGB", (self.width, self.height))] * self.num_frames, data_file.as_posix(), fps=2
            )
            self.data_files.append(data_file.as_posix())

        csv_filename = pathlib.Path(self.tmpdir.name) / "metadata.csv"
        with open(csv_filename.as_posix(), "w") as f:
            f.write("file_name,caption\n")
            for i in range(self.num_data_files):
                prompt = f"A cat ruling the world - {i}"
                f.write(f'{i}.mp4,"{prompt}"\n')

        dataset_config = {
            "datasets": [
                {
                    "data_root": self.tmpdir.name,
                    "dataset_type": "video",
                    "id_token": "TEST",
                    "video_resolution_buckets": [[self.num_frames, self.height, self.width]],
                    "reshape_mode": "bicubic",
                }
            ]
        }

        self.dataset_config_filename = pathlib.Path(self.tmpdir.name) / "dataset_config.json"
        with open(self.dataset_config_filename.as_posix(), "w") as f:
            json.dump(dataset_config, f)

    def tearDown(self):
        self.tmpdir.cleanup()

    def get_base_args(self) -> BaseArgs:
        args = BaseArgs()
        args.dataset_config = self.dataset_config_filename.as_posix()
        args.train_steps = 10
        args.max_data_samples = 25
        args.batch_size = 1
        args.gradient_checkpointing = True
        args.output_dir = self.tmpdir.name
        args.checkpointing_steps = 6
        args.enable_precomputation = False
        args.precomputation_items = self.num_data_files
        args.precomputation_dir = os.path.join(self.tmpdir.name, "precomputed")
        return args

    def get_args(self) -> BaseArgs:
        raise NotImplementedError("`get_args` must be implemented in the subclass.")

    def _test_training(self, args: BaseArgs):
        model_specification = self.model_specification_cls()
        trainer = SFTTrainer(args, model_specification)
        trainer.run()


class SFTTrainerLoRATestsMixin___PTD(SFTTrainerFastTestsMixin):
    def get_args(self) -> BaseArgs:
        args = self.get_base_args()
        args.parallel_backend = "ptd"
        args.training_type = TrainingType.LORA
        args.rank = 4
        args.lora_alpha = 4
        args.target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
        return args

    @parameterized.expand([(False,), (True,)])
    def test___dp_degree_1___batch_size_1(self, enable_precomputation: bool):
        args = self.get_args()
        args.dp_degree = 1
        args.batch_size = 1
        args.enable_precomputation = enable_precomputation
        self._test_training(args)

    @parameterized.expand([(False,), (True,)])
    def test___dp_degree_1___batch_size_2(self, enable_precomputation: bool):
        args = self.get_args()
        args.dp_degree = 1
        args.batch_size = 2
        args.enable_precomputation = enable_precomputation
        self._test_training(args)

    @parameterized.expand([(False,), (True,)])
    def test___dp_degree_2___batch_size_1(self, enable_precomputation: bool):
        args = self.get_args()
        args.dp_degree = 2
        args.batch_size = 1
        args.enable_precomputation = enable_precomputation
        self._test_training(args)

    @parameterized.expand([(False,), (True,)])
    def test___dp_degree_2___batch_size_2(self, enable_precomputation: bool):
        args = self.get_args()
        args.dp_degree = 2
        args.batch_size = 2
        args.enable_precomputation = enable_precomputation
        self._test_training(args)

    @parameterized.expand([(False,), (True,)])
    def test___dp_shards_2___batch_size_1(self, enable_precomputation: bool):
        args = self.get_args()
        args.dp_shards = 2
        args.batch_size = 1
        args.enable_precomputation = enable_precomputation
        self._test_training(args)

    @parameterized.expand([(False,), (True,)])
    def test___dp_shards_2___batch_size_2(self, enable_precomputation: bool):
        args = self.get_args()
        args.dp_shards = 2
        args.batch_size = 1
        args.enable_precomputation = enable_precomputation
        self._test_training(args)

    @parameterized.expand([(False,), (True,)])
    def test___dp_degree_2___dp_shards_2___batch_size_1(self, enable_precomputation: bool):
        args = self.get_args()
        args.dp_degree = 2
        args.dp_shards = 2
        args.batch_size = 1
        args.enable_precomputation = enable_precomputation
        self._test_training(args)

    @parameterized.expand([(False,), (True,)])
    def test___tp_degree_2___batch_size_2(self, enable_precomputation: bool):
        args = self.get_args()
        args.tp_degree = 2
        args.batch_size = 1
        args.enable_precomputation = enable_precomputation
        self._test_training(args)


class SFTTrainerFullFinetuneTestsMixin___PTD(SFTTrainerFastTestsMixin):
    def get_args(self) -> BaseArgs:
        args = self.get_base_args()
        args.parallel_backend = "ptd"
        args.training_type = TrainingType.FULL_FINETUNE
        return args

    @parameterized.expand([(False,), (True,)])
    def test___dp_degree_1___batch_size_1(self, enable_precomputation: bool):
        args = self.get_args()
        args.dp_degree = 1
        args.batch_size = 1
        args.enable_precomputation = enable_precomputation
        self._test_training(args)

    @parameterized.expand([(False,), (True,)])
    def test___dp_degree_1___batch_size_2(self, enable_precomputation: bool):
        args = self.get_args()
        args.dp_degree = 1
        args.batch_size = 2
        args.enable_precomputation = enable_precomputation
        self._test_training(args)

    @parameterized.expand([(False,), (True,)])
    def test___dp_degree_2___batch_size_1(self, enable_precomputation: bool):
        args = self.get_args()
        args.dp_degree = 2
        args.batch_size = 1
        args.enable_precomputation = enable_precomputation
        self._test_training(args)

    @parameterized.expand([(False,), (True,)])
    def test___dp_degree_2___batch_size_2(self, enable_precomputation: bool):
        args = self.get_args()
        args.dp_degree = 2
        args.batch_size = 2
        args.enable_precomputation = enable_precomputation
        self._test_training(args)

    @parameterized.expand([(False,), (True,)])
    def test___dp_shards_2___batch_size_1(self, enable_precomputation: bool):
        args = self.get_args()
        args.dp_shards = 2
        args.batch_size = 1
        args.enable_precomputation = enable_precomputation
        self._test_training(args)

    @parameterized.expand([(False,), (True,)])
    def test___dp_shards_2___batch_size_2(self, enable_precomputation: bool):
        args = self.get_args()
        args.dp_shards = 2
        args.batch_size = 1
        args.enable_precomputation = enable_precomputation
        self._test_training(args)

    @parameterized.expand([(False,), (True,)])
    def test___dp_degree_2___dp_shards_2___batch_size_1(self, enable_precomputation: bool):
        args = self.get_args()
        args.dp_degree = 2
        args.dp_shards = 2
        args.batch_size = 1
        args.enable_precomputation = enable_precomputation
        self._test_training(args)

    @parameterized.expand([(False,), (True,)])
    def test___tp_degree_2___batch_size_2(self, enable_precomputation: bool):
        args = self.get_args()
        args.tp_degree = 2
        args.batch_size = 1
        args.enable_precomputation = enable_precomputation
        self._test_training(args)


class SFTTrainerCogVideoXLoRATests___PTD(SFTTrainerLoRATestsMixin___PTD, unittest.TestCase):
    model_specification_cls = DummyCogVideoXModelSpecification


class SFTTrainerCogVideoXFullFinetuneTests___PTD(SFTTrainerFullFinetuneTestsMixin___PTD, unittest.TestCase):
    model_specification_cls = DummyCogVideoXModelSpecification


class SFTTrainerCogView4LoRATests___PTD(SFTTrainerLoRATestsMixin___PTD, unittest.TestCase):
    model_specification_cls = DummyCogView4ModelSpecification


class SFTTrainerCogView4FullFinetuneTests___PTD(SFTTrainerFullFinetuneTestsMixin___PTD, unittest.TestCase):
    model_specification_cls = DummyCogView4ModelSpecification


class SFTTrainerHunyuanVideoLoRATests___PTD(SFTTrainerLoRATestsMixin___PTD, unittest.TestCase):
    model_specification_cls = DummyHunyuanVideoModelSpecification


class SFTTrainerHunyuanVideoFullFinetuneTests___PTD(SFTTrainerFullFinetuneTestsMixin___PTD, unittest.TestCase):
    model_specification_cls = DummyHunyuanVideoModelSpecification


class SFTTrainerLTXVideoLoRATests___PTD(SFTTrainerLoRATestsMixin___PTD, unittest.TestCase):
    model_specification_cls = DummyLTXVideoModelSpecification


class SFTTrainerLTXVideoFullFinetuneTests___PTD(SFTTrainerFullFinetuneTestsMixin___PTD, unittest.TestCase):
    model_specification_cls = DummyLTXVideoModelSpecification


class SFTTrainerWanLoRATests___PTD(SFTTrainerLoRATestsMixin___PTD, unittest.TestCase):
    model_specification_cls = DummyWanModelSpecification


class SFTTrainerWanFullFinetuneTests___PTD(SFTTrainerFullFinetuneTestsMixin___PTD, unittest.TestCase):
    model_specification_cls = DummyWanModelSpecification
