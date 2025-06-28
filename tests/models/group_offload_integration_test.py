import unittest
from unittest.mock import patch

import pytest
import torch

# Import the proper test model specifications that use hf-internal-testing models
from tests.models.flux.base_specification import DummyFluxModelSpecification


# Skip tests if CUDA is not available
has_cuda = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not has_cuda, reason="Test requires CUDA")


# Test with real HuggingFace dummy models that work completely
@pytest.mark.parametrize(
    "model_specification_class",
    [
        DummyFluxModelSpecification,  # Uses hf-internal-testing/tiny-flux-pipe - complete tiny model ✅
        # DummyCogView4ModelSpecification,  # Uses hf-internal-testing/tiny-random-cogview4 - WORKS but needs trust_remote_code=True fix
        # Skip models that need dummy checkpoints uploaded (have TODO comments):
        # DummyCogVideoXModelSpecification,  # Creates components from scratch - needs upload
        # DummyLTXVideoModelSpecification,  # Creates components from scratch - needs upload
        # DummyHunyuanVideoModelSpecification,  # Creates components from scratch - needs upload
        # DummyWanModelSpecification,  # Creates components from scratch - needs upload
    ],
)
class TestGroupOffloadingIntegration:
    @patch("finetrainers.utils.offloading.enable_group_offload_on_components")
    def test_load_pipeline_with_group_offload(self, mock_enable_group_offload, model_specification_class):
        """Test that group offloading is properly enabled when loading the pipeline."""

        # Create model specification
        model_spec = model_specification_class()

        # Call load_pipeline with group offloading enabled
        # Disable streams on non-CUDA systems to avoid errors
        use_stream = torch.cuda.is_available()
        model_spec.load_pipeline(
            enable_group_offload=True,
            group_offload_type="block_level",
            group_offload_blocks_per_group=4,
            group_offload_use_stream=use_stream,
        )

        # Assert that enable_group_offload_on_components was called with the correct arguments
        mock_enable_group_offload.assert_called_once()

        # Check the call arguments - they are passed as keyword arguments
        call_kwargs = mock_enable_group_offload.call_args.kwargs

        assert "components" in call_kwargs
        assert "device" in call_kwargs
        assert isinstance(call_kwargs["components"], dict)
        assert isinstance(call_kwargs["device"], torch.device)
        assert call_kwargs["offload_type"] == "block_level"
        assert call_kwargs["num_blocks_per_group"] == 4
        assert call_kwargs["use_stream"] == use_stream

    @patch("finetrainers.utils.offloading.enable_group_offload_on_components")
    def test_mutually_exclusive_offload_methods(self, mock_enable_group_offload, model_specification_class):
        """Test that only one offloading method is used when both are enabled."""
        # Skip this test on CPU-only systems since model_cpu_offload requires accelerator
        if not torch.cuda.is_available():
            pytest.skip("enable_model_cpu_offload requires accelerator")

        # Create model specification
        model_spec = model_specification_class()

        # Call load_pipeline with both offloading methods enabled (model offload should take precedence)
        model_spec.load_pipeline(
            enable_model_cpu_offload=True,
            enable_group_offload=True,
        )

        # Assert that group_offload was not called when model_cpu_offload is also enabled
        mock_enable_group_offload.assert_not_called()

    @patch("finetrainers.utils.offloading.enable_group_offload_on_components")
    def test_import_error_handling(self, mock_enable_group_offload, model_specification_class):
        """Test that ImportError is handled gracefully when diffusers version is too old."""
        # Simulate an ImportError when trying to use group offloading
        mock_enable_group_offload.side_effect = ImportError("Module not found")

        # Mock the logger at the module level where it's used
        with patch("finetrainers.models.flux.base_specification.logger") as mock_logger:
            # Create model specification
            model_spec = model_specification_class()

            # Call load_pipeline with group offloading enabled
            model_spec.load_pipeline(
                enable_group_offload=True,
            )

            # Assert that a warning was logged
            mock_logger.warning.assert_called_once()
            warning_msg = mock_logger.warning.call_args[0][0]
            assert "Failed to enable group offloading" in warning_msg
            assert "Using standard pipeline without offloading" in warning_msg


if __name__ == "__main__":
    unittest.main()
