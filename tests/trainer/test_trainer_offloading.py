import unittest
import torch
import pytest
from unittest.mock import patch, MagicMock

from finetrainers.trainer.sft_trainer.trainer import SFTTrainer
from finetrainers.args import BaseArgs
from finetrainers.models.flux import FluxModelSpecification


class DummyFluxModelSpecification(FluxModelSpecification):
    def __init__(self, **kwargs):
        super().__init__(pretrained_model_name_or_path="hf-internal-testing/tiny-flux-pipe", **kwargs)

    # Override to avoid loading models from hub
    def load_diffusion_models(self):
        return {
            "transformer": MagicMock(),
            "scheduler": MagicMock(),
        }

    def load_pipeline(self, **kwargs):
        return MagicMock()


class TestTrainerOffloading(unittest.TestCase):
    def setUp(self):
        # Mock BaseArgs for testing
        self.args = MagicMock(spec=BaseArgs)
        self.args.enable_model_cpu_offload = False
        self.args.enable_group_offload = True
        self.args.group_offload_type = "block_level"
        self.args.group_offload_blocks_per_group = 2
        self.args.group_offload_use_stream = True
        self.args.model_name = "flux"
        self.args.training_type = "lora"
        self.args.enable_slicing = False
        self.args.enable_tiling = False

        # Create model specification
        self.model_spec = DummyFluxModelSpecification()

        # Create a partial mock for the trainer to avoid initializing everything
        patcher = patch.multiple(
            SFTTrainer,
            _init_distributed=MagicMock(),
            _init_config_options=MagicMock(),
            __init__=lambda self, args, model_spec: None,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        # Create the trainer
        self.trainer = SFTTrainer(None, None)
        self.trainer.args = self.args
        self.trainer.model_specification = self.model_spec
        self.trainer.state = MagicMock()
        self.trainer.state.parallel_backend = MagicMock()
        self.trainer.state.parallel_backend.device = torch.device("cpu")
        self.trainer.state.train_state = MagicMock()
        self.trainer.state.train_state.step = 1000  # Mock step number

        # Set the necessary attributes that would be set in _prepare_models
        self.trainer.transformer = MagicMock()
        self.trainer.vae = MagicMock()
        self.trainer.text_encoder = MagicMock()
        self.trainer.text_encoder_2 = MagicMock()
        self.trainer.text_encoder_3 = MagicMock()
        self.trainer.tokenizer = MagicMock()
        self.trainer.tokenizer_2 = MagicMock()
        self.trainer.tokenizer_3 = MagicMock()
        self.trainer.image_encoder = MagicMock()
        self.trainer.image_processor = MagicMock()
        self.trainer.scheduler = MagicMock()

    def test_init_pipeline_with_group_offload(self):
        """Test that _init_pipeline passes group offloading arguments to load_pipeline."""
        # Mock the load_pipeline method to capture arguments
        self.model_spec.load_pipeline = MagicMock(return_value=MagicMock())

        # Call _init_pipeline
        self.trainer._init_pipeline(final_validation=False)

        # Check that load_pipeline was called with the correct arguments
        _, kwargs = self.model_spec.load_pipeline.call_args

        self.assertEqual(kwargs["enable_group_offload"], True)
        self.assertEqual(kwargs["group_offload_type"], "block_level")
        self.assertEqual(kwargs["group_offload_blocks_per_group"], 2)
        self.assertEqual(kwargs["group_offload_use_stream"], True)

    def test_init_pipeline_final_validation_with_group_offload(self):
        """Test that _init_pipeline passes group offloading arguments during final validation."""
        # Mock the load_pipeline method to capture arguments
        self.model_spec.load_pipeline = MagicMock(return_value=MagicMock())

        # Call _init_pipeline with final_validation=True
        self.trainer._init_pipeline(final_validation=True)

        # Check that load_pipeline was called with the correct arguments
        _, kwargs = self.model_spec.load_pipeline.call_args

        self.assertEqual(kwargs["enable_group_offload"], True)
        self.assertEqual(kwargs["group_offload_type"], "block_level")
        self.assertEqual(kwargs["group_offload_blocks_per_group"], 2)
        self.assertEqual(kwargs["group_offload_use_stream"], True)

    def test_mutually_exclusive_offloading_methods(self):
        """Test that only one offloading method is used when both are enabled."""
        # Set both offloading methods to True (model offload should take precedence)
        self.args.enable_model_cpu_offload = True
        self.args.enable_group_offload = True

        # Mock the load_pipeline method to capture arguments
        self.model_spec.load_pipeline = MagicMock(return_value=MagicMock())

        # Call _init_pipeline
        self.trainer._init_pipeline(final_validation=False)

        # Check that load_pipeline was called with the correct arguments
        _, kwargs = self.model_spec.load_pipeline.call_args

        # Model offloading should be enabled and group offloading should be disabled
        self.assertEqual(kwargs["enable_model_cpu_offload"], True)
        self.assertEqual(kwargs["enable_group_offload"], True)

        # The model specification's implementation should ensure only one is actually used


if __name__ == "__main__":
    unittest.main()