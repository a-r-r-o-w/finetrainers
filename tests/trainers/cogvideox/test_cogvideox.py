import sys 
from pathlib import Path

current_file = Path(__file__).resolve()
root_dir = current_file.parents[3]
sys.path.append(str(root_dir))

from ..test_trainers_common import TrainerTestMixin
from finetrainers import parse_arguments
import unittest


class CogVideoXTester(unittest.TestCase, TrainerTestMixin):
    def get_training_args(self):
        args = parse_arguments()
        args.training_type = "lora"
        args.pretrained_model_name_or_path = "finetrainers/dummy-cogvideox"
        args.video_resolution_buckets = "9x16x16"
        return args