import sys
from pathlib import Path


current_file = Path(__file__).resolve()
root_dir = current_file.parents[3]
sys.path.append(str(root_dir))

import unittest  # noqa
from typing import Tuple  # noqa

from finetrainers import Args  # noqa

from ..test_trainers_common import TrainerTestMixin  # noqa


# Copied for now.
def parse_resolution_bucket(resolution_bucket: str) -> Tuple[int, ...]:
    return tuple(map(int, resolution_bucket.split("x")))


class CogVideoXTester(unittest.TestCase, TrainerTestMixin):
    model_name = "cogvideox"

    def get_training_args(self):
        args = Args()
        args.model_name = self.model_name
        args.training_type = "lora"
        args.pretrained_model_name_or_path = "finetrainers/dummy-cogvideox"
        args.data_root = ""  # will be set from the tester method.
        args.video_resolution_buckets = [parse_resolution_bucket("9x16x16")]
        args.precompute_conditions = True
        args.validation_prompts = []
        args.validation_heights = []
        args.validation_widths = []
        return args
