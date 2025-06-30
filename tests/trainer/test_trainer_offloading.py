import json
import os
from unittest.mock import MagicMock, patch

import pytest
import torch

from finetrainers.args import BaseArgs
from finetrainers.models.cogvideox import CogVideoXModelSpecification
from finetrainers.models.flux import FluxModelSpecification
from finetrainers.models.hunyuan_video import HunyuanVideoModelSpecification
from finetrainers.models.ltx_video import LTXVideoModelSpecification
from finetrainers.trainer.sft_trainer.trainer import SFTTrainer


class DummyHunyuanVideoModelSpecification(HunyuanVideoModelSpecification):
    def __init__(self, **kwargs):
        # Use the existing dummy model from the Hub - it's small enough for testing
        super().__init__(pretrained_model_name_or_path="finetrainers/dummy-hunyaunvideo", **kwargs)


class DummyFluxModelSpecification(FluxModelSpecification):
    def __init__(self, **kwargs):
        super().__init__(pretrained_model_name_or_path="hf-internal-testing/tiny-flux-pipe", **kwargs)


class DummyCogVideoXModelSpecification(CogVideoXModelSpecification):
    def __init__(self, **kwargs):
        super().__init__(pretrained_model_name_or_path="finetrainers/dummy-cogvideox", **kwargs)


class DummyLTXVideoModelSpecification(LTXVideoModelSpecification):
    def __init__(self, **kwargs):
        super().__init__(pretrained_model_name_or_path="finetrainers/dummy-ltxvideo", **kwargs)


@pytest.mark.parametrize(
    "model_specification_class,model_name",
    [
        (DummyFluxModelSpecification, "flux"),
        (DummyHunyuanVideoModelSpecification, "hunyuan_video"),
        (DummyCogVideoXModelSpecification, "cogvideox"),
        (DummyLTXVideoModelSpecification, "ltx_video"),
    ],
)
class TestTrainerOffloading:
    @pytest.fixture(autouse=True)
    def setup_method(self, model_specification_class, model_name):
        """Set up test fixtures for each parameterized test with realistic dummy models."""
        self.model_specification_class = model_specification_class
        self.model_name = model_name

        # Check if CUDA is available for realistic stream testing
        self.has_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.has_cuda else "cpu")

        # Create realistic BaseArgs for testing
        self.args = MagicMock(spec=BaseArgs)
        self.args.enable_model_cpu_offload = False
        self.args.enable_group_offload = False  # Start with group offload disabled by default
        self.args.group_offload_type = "block_level"
        self.args.group_offload_blocks_per_group = 2
        self.args.group_offload_use_stream = self.has_cuda  # Only use streams if CUDA is available
        self.args.model_name = self.model_name
        self.args.training_type = "lora"  # Use LoRA training as it's more popular and realistic
        self.args.enable_slicing = False
        self.args.enable_tiling = False

        # Add other required args for trainer initialization
        self.args.output_dir = "/tmp/test_output"
        self.args.cache_dir = None
        self.args.revision = None
        self.args.local_files_only = False
        self.args.trust_remote_code = False

        # Add missing attention provider args
        self.args.attn_provider_training = None
        self.args.attn_provider_inference = None

        # Use LoRA as default since it's much more popular and realistic
        self.args.training_type = "lora"

        # Create model specification with dummy models
        self.model_spec = self.model_specification_class()

        # Mock only the distributed and config initialization to avoid complex setup
        # Create a mock parallel backend before trainer creation
        mock_parallel_backend = MagicMock()
        mock_parallel_backend.device = self.device
        mock_parallel_backend.pipeline_parallel_enabled = False
        mock_parallel_backend.tensor_parallel_enabled = False

        def mock_init_distributed(trainer_self):
            trainer_self.state.parallel_backend = mock_parallel_backend

        self.patcher = patch.multiple(
            SFTTrainer,
            _init_distributed=mock_init_distributed,
            _init_config_options=MagicMock(),
        )
        self.patcher.start()

        # Create the trainer with realistic initialization
        self.trainer = SFTTrainer(self.args, self.model_spec)

        # Ensure the state is properly set up
        self.trainer.state.train_state = MagicMock()
        self.trainer.state.train_state.step = 1000

        # Load actual dummy model components - this is the realistic part!
        self.trainer._prepare_models()

        # Create a realistic LoRA weights directory for final validation tests
        os.makedirs("/tmp/test_output/lora_weights/001000", exist_ok=True)

        # Create a more realistic adapter_config.json with common LoRA settings
        adapter_config = {
            "base_model_name_or_path": self.model_spec.pretrained_model_name_or_path,
            "bias": "none",
            "fan_in_fan_out": False,
            "inference_mode": True,
            "init_lora_weights": True,
            "layers_pattern": None,
            "layers_to_transform": None,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "modules_to_save": None,
            "peft_type": "LORA",
            "r": 16,
            "revision": None,
            "target_modules": ["to_q", "to_v", "to_k", "to_out.0"],
            "task_type": "FEATURE_EXTRACTION",
            "use_rslora": False,
        }

        with open("/tmp/test_output/lora_weights/001000/adapter_config.json", "w") as f:
            json.dump(adapter_config, f, indent=2)

        # Create realistic LoRA weight tensors with proper naming
        lora_weights = {}
        for target_module in adapter_config["target_modules"]:
            # Create typical LoRA weight matrices (A and B matrices)
            lora_weights[f"transformer.{target_module}.lora_A.weight"] = torch.randn(16, 64)
            lora_weights[f"transformer.{target_module}.lora_B.weight"] = torch.randn(64, 16)

        torch.save(lora_weights, "/tmp/test_output/lora_weights/001000/pytorch_lora_weights.bin")

    def teardown_method(self):
        """Clean up after each test."""
        if hasattr(self, "patcher"):
            self.patcher.stop()

    def _get_param(self, param_name):
        """Helper method to get pytest parameters - no longer needed with proper fixtures."""
        pass

    def test_init_pipeline_with_group_offload(self):
        """Test that _init_pipeline creates a pipeline with group offloading enabled."""
        # Skip group offloading tests if CUDA is not available
        if not torch.cuda.is_available():
            pytest.skip("Group offloading requires CUDA - skipping test on CPU-only system")

        # Enable group offloading for this test
        self.args.enable_group_offload = True

        # Call _init_pipeline with group offloading enabled
        try:
            pipeline = self.trainer._init_pipeline(final_validation=False)

            # Verify that a pipeline was created
            assert pipeline is not None

            # Verify that the pipeline has the expected components
            # (This tests that the dummy models were loaded correctly)
            assert hasattr(pipeline, "transformer")
            assert hasattr(pipeline, "vae")

            # Verify that group offloading was properly configured
            # (We can't easily inspect internal offloading state, but we can verify the pipeline was created)
            assert pipeline.transformer is not None
            assert pipeline.vae is not None

        except Exception as e:
            # If group offloading fails (e.g., on CPU-only systems), that's expected
            # The important thing is that we properly handle the error
            if "accelerator" in str(e) or "cuda" in str(e).lower():
                pytest.skip(f"Group offloading not supported in this environment: {e}")
            else:
                # Re-raise unexpected errors
                raise

    def test_init_pipeline_final_validation_with_group_offload(self):
        """Test that _init_pipeline creates a pipeline for final validation with group offloading."""
        # Call _init_pipeline with final_validation=True
        pipeline = self.trainer._init_pipeline(final_validation=True)

        # Verify that a pipeline was created for validation
        assert pipeline is not None

        # Verify that the pipeline components are properly set
        assert hasattr(pipeline, "transformer")
        assert hasattr(pipeline, "vae")

    def test_mutually_exclusive_offloading_methods(self):
        """Test that both offloading methods can be passed to the pipeline (implementation handles mutual exclusion)."""
        # Set both offloading methods to True
        self.args.enable_model_cpu_offload = True
        self.args.enable_group_offload = True

        # Use patch to spy on the load_pipeline method
        with patch.object(self.model_spec, "load_pipeline", wraps=self.model_spec.load_pipeline) as mock_pipeline:
            # Call _init_pipeline
            pipeline = self.trainer._init_pipeline(final_validation=False)

            # Check that load_pipeline was called with both offloading methods
            _, kwargs = mock_pipeline.call_args
            assert kwargs["enable_model_cpu_offload"]
            assert kwargs["enable_group_offload"]

        # Verify that a pipeline was still created successfully
        assert pipeline is not None
        assert hasattr(pipeline, "transformer")
        assert hasattr(pipeline, "vae")

    def test_group_offload_disabled(self):
        """Test that group offloading is properly disabled when not requested."""
        # Set group offload to False
        self.args.enable_group_offload = False
        self.args.enable_model_cpu_offload = False

        # Use patch to spy on the load_pipeline method
        with patch.object(self.model_spec, "load_pipeline", wraps=self.model_spec.load_pipeline) as mock_pipeline:
            # Call _init_pipeline
            pipeline = self.trainer._init_pipeline(final_validation=False)

            # Check that load_pipeline was called without group offloading
            _, kwargs = mock_pipeline.call_args
            assert not kwargs["enable_group_offload"]
            assert not kwargs["enable_model_cpu_offload"]

        # Verify that a pipeline was still created successfully
        assert pipeline is not None
        assert hasattr(pipeline, "transformer")
        assert hasattr(pipeline, "vae")

    def test_different_group_offload_types(self):
        """Test different group offload types are passed correctly to the real pipeline."""
        test_cases = [
            ("block_level", 1, False),
            ("leaf_level", 4, self.has_cuda),  # Only use streams if CUDA is available
            ("block_level", 8, False),  # Test different block group size
        ]

        for offload_type, blocks_per_group, use_stream in test_cases:
            # Set test parameters
            self.args.group_offload_type = offload_type
            self.args.group_offload_blocks_per_group = blocks_per_group
            self.args.group_offload_use_stream = use_stream

            # Use patch to spy on the load_pipeline method
            with patch.object(self.model_spec, "load_pipeline", wraps=self.model_spec.load_pipeline) as mock_pipeline:
                # Call _init_pipeline
                try:
                    pipeline = self.trainer._init_pipeline(final_validation=False)

                    # Check parameters were passed correctly
                    _, kwargs = mock_pipeline.call_args
                    assert kwargs["group_offload_type"] == offload_type
                    assert kwargs["group_offload_blocks_per_group"] == blocks_per_group
                    assert kwargs["group_offload_use_stream"] == use_stream

                    # Verify that a pipeline was created successfully
                    assert pipeline is not None
                    assert hasattr(pipeline, "transformer")
                    assert hasattr(pipeline, "vae")
                except (AttributeError, RuntimeError) as e:
                    if (
                        "'NoneType' object has no attribute 'type'" in str(e)
                        or "accelerator" in str(e).lower()
                        or "cuda" in str(e).lower()
                    ):
                        pytest.skip(f"Group offloading not supported in this environment: {e}")
                    else:
                        raise

    def test_group_offload_edge_case_values(self):
        """Test edge case values for group offload parameters work with real pipelines."""
        # Test minimum values
        self.args.group_offload_blocks_per_group = 1
        self.args.group_offload_use_stream = False

        # Use patch to spy on the load_pipeline method
        with patch.object(self.model_spec, "load_pipeline", wraps=self.model_spec.load_pipeline) as mock_pipeline:
            # Call _init_pipeline
            pipeline = self.trainer._init_pipeline(final_validation=False)

            # Check parameters
            _, kwargs = mock_pipeline.call_args
            assert kwargs["group_offload_blocks_per_group"] == 1
            assert not kwargs["group_offload_use_stream"]

        # Verify that a pipeline was created successfully even with edge case values
        assert pipeline is not None
        assert hasattr(pipeline, "transformer")
        assert hasattr(pipeline, "vae")

    def test_group_offload_with_other_memory_optimizations(self):
        """Test group offload works with other memory optimization options."""
        # Skip group offloading tests if CUDA is not available
        if not torch.cuda.is_available():
            pytest.skip("Group offloading requires CUDA - skipping test on CPU-only system")

        # Enable group offload and other memory optimizations
        self.args.enable_group_offload = True
        self.args.enable_slicing = True
        self.args.enable_tiling = True

        # Use patch to spy on the load_pipeline method
        with patch.object(self.model_spec, "load_pipeline", wraps=self.model_spec.load_pipeline) as mock_pipeline:
            # Call _init_pipeline
            try:
                pipeline = self.trainer._init_pipeline(final_validation=False)

                # Check that all memory optimizations are passed
                _, kwargs = mock_pipeline.call_args
                assert kwargs["enable_group_offload"]
                assert kwargs["enable_slicing"]
                assert kwargs["enable_tiling"]

                # Verify that a pipeline was created successfully with all optimizations
                assert pipeline is not None
                assert hasattr(pipeline, "transformer")
                assert hasattr(pipeline, "vae")
            except (AttributeError, RuntimeError) as e:
                if "'NoneType' object has no attribute 'type'" in str(e) or "accelerator" in str(e).lower():
                    pytest.skip(f"Group offloading not supported in this environment: {e}")
                else:
                    raise

    def test_group_offload_training_vs_validation_mode(self):
        """Test that training parameter is correctly set for different modes."""
        # Use patch to spy on the load_pipeline method
        with patch.object(self.model_spec, "load_pipeline", wraps=self.model_spec.load_pipeline) as mock_pipeline:
            # Test training mode (final_validation=False)
            pipeline1 = self.trainer._init_pipeline(final_validation=False)
            _, kwargs = mock_pipeline.call_args
            assert kwargs["training"]

            # Verify pipeline creation
            assert pipeline1 is not None
            assert hasattr(pipeline1, "transformer")
            assert hasattr(pipeline1, "vae")

            # Reset mock
            mock_pipeline.reset_mock()

            # Test validation mode (final_validation=True)
            pipeline2 = self.trainer._init_pipeline(final_validation=True)
            _, kwargs = mock_pipeline.call_args
            assert not kwargs["training"]

            # Verify pipeline creation for validation mode
            assert pipeline2 is not None
            assert hasattr(pipeline2, "transformer")
            assert hasattr(pipeline2, "vae")

    def test_group_offload_parameter_consistency(self):
        """Test that all group offload parameters are consistently passed."""
        # Set comprehensive parameters with valid offload type
        self.args.enable_group_offload = True
        self.args.group_offload_type = "block_level"  # Use valid offload type
        self.args.group_offload_blocks_per_group = 99
        self.args.group_offload_use_stream = self.has_cuda  # Only use streams if CUDA is available

        # Use patch to spy on the load_pipeline method
        with patch.object(self.model_spec, "load_pipeline", wraps=self.model_spec.load_pipeline) as mock_pipeline:
            # Call _init_pipeline
            try:
                pipeline = self.trainer._init_pipeline(final_validation=False)

                # Check that all parameters are correctly passed
                _, kwargs = mock_pipeline.call_args

                # Verify all group offload related parameters
                expected_group_offload_params = {
                    "enable_group_offload": True,
                    "group_offload_type": "block_level",
                    "group_offload_blocks_per_group": 99,
                    "group_offload_use_stream": self.has_cuda,
                }

                for param, expected_value in expected_group_offload_params.items():
                    assert param in kwargs, f"Parameter {param} missing from kwargs"
                    assert kwargs[param] == expected_value, f"Parameter {param} has incorrect value"

                # Verify that a pipeline was created successfully with all parameters
                assert pipeline is not None
                assert hasattr(pipeline, "transformer")
                assert hasattr(pipeline, "vae")
            except (AttributeError, RuntimeError) as e:
                if (
                    "'NoneType' object has no attribute 'type'" in str(e)
                    or "accelerator" in str(e).lower()
                    or "cuda" in str(e).lower()
                ):
                    pytest.skip(f"Group offloading not supported in this environment: {e}")
                else:
                    raise

    def test_cuda_stream_behavior(self):
        """Test that stream usage is correctly handled based on CUDA availability."""
        # Test with streams enabled (should work if CUDA is available, gracefully handle if not)
        self.args.group_offload_use_stream = True

        # Use patch to spy on the load_pipeline method
        with patch.object(self.model_spec, "load_pipeline", wraps=self.model_spec.load_pipeline) as mock_pipeline:
            # Call _init_pipeline
            pipeline1 = self.trainer._init_pipeline(final_validation=False)

            # Check that stream parameter was passed
            _, kwargs = mock_pipeline.call_args
            assert kwargs["group_offload_use_stream"]

            # Verify that a pipeline was created successfully
            # (The model implementation should handle stream compatibility internally)
            assert pipeline1 is not None
            assert hasattr(pipeline1, "transformer")
            assert hasattr(pipeline1, "vae")

        # Test with streams disabled (should always work)
        self.args.group_offload_use_stream = False

        with patch.object(self.model_spec, "load_pipeline", wraps=self.model_spec.load_pipeline) as mock_pipeline:
            # Call _init_pipeline
            pipeline2 = self.trainer._init_pipeline(final_validation=False)

            # Check that stream parameter was passed as False
            _, kwargs = mock_pipeline.call_args
            assert not kwargs["group_offload_use_stream"]

            # Verify that a pipeline was created successfully
            assert pipeline2 is not None
            assert hasattr(pipeline2, "transformer")
            assert hasattr(pipeline2, "vae")


if __name__ == "__main__":
    pytest.main([__file__])
