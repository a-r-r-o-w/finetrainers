import sys
import unittest
from pathlib import Path


current_file = Path(__file__).resolve()
root_dir = current_file.parents[3]
sys.path.append(str(root_dir))

from finetrainers import Args  # noqa
from ..test_trainers_common import TrainerTestMixin, parse_resolution_bucket  # noqa


class HunyuanVideoTester(unittest.TestCase, TrainerTestMixin):
    MODEL_NAME = "hunyuan_video"
    EXPECTED_PRECOMPUTATION_LATENT_KEYS = {"latents"}
    EXPECTED_PRECOMPUTATION_CONDITION_KEYS = {
        "guidance",
        "pooled_prompt_embeds",
        "prompt_attention_mask",
        "prompt_embeds",
    }

    def get_training_args(self):
        args = Args()
        args.model_name = self.MODEL_NAME
        args.training_type = "lora"
        args.pretrained_model_name_or_path = "finetrainers/dummy-hunyaunvideo"
        args.data_root = ""  # will be set from the tester method.
        args.video_resolution_buckets = [parse_resolution_bucket("9x16x16")]
        args.precompute_conditions = True
        args.validation_prompts = []
        args.validation_heights = []
        args.validation_widths = []
        return args

    @property
    def latent_output_shape(self):
        # only tensor object shapes
        return (8, 3, 2, 2)

    @property
    def condition_output_shape(self):
        # only tensor object shapes
        return (), (8,), (256,), (256, 16)

    def populate_shapes(self):
        for i, k in enumerate(sorted(self.EXPECTED_PRECOMPUTATION_LATENT_KEYS)):
            self.EXPECTED_LATENT_SHAPES[k] = self.latent_output_shape
        for i, k in enumerate(sorted(self.EXPECTED_PRECOMPUTATION_CONDITION_KEYS)):
            self.EXPECTED_CONDITION_SHAPES[k] = self.condition_output_shape[i]
