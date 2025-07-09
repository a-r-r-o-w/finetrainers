import unittest
from unittest.mock import patch

import pytest
import torch

from finetrainers.models.cogvideox import CogVideoXModelSpecification
from finetrainers.models.hunyuan_video import HunyuanVideoModelSpecification
from finetrainers.models.ltx_video import LTXVideoModelSpecification
from tests.models.flux.base_specification import DummyFluxModelSpecification


class DummyHunyuanVideoModelSpecification(HunyuanVideoModelSpecification):
    def __init__(self, **kwargs):
        super().__init__(pretrained_model_name_or_path="finetrainers/dummy-hunyaunvideo", **kwargs)


class DummyCogVideoXModelSpecification(CogVideoXModelSpecification):
    def __init__(self, **kwargs):
        super().__init__(pretrained_model_name_or_path="finetrainers/dummy-cogvideox", **kwargs)


class DummyLTXVideoModelSpecification(LTXVideoModelSpecification):
    def __init__(self, **kwargs):
        super().__init__(pretrained_model_name_or_path="finetrainers/dummy-ltxvideo", **kwargs)


# Skip tests if CUDA is not available
has_cuda = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not has_cuda, reason="Test requires CUDA")


@pytest.mark.parametrize(
    "model_specification_class",
    [
        DummyFluxModelSpecification,
        # DummyCogView4ModelSpecification,  # Uses hf-internal-testing/tiny-random-cogview4 - WORKS but needs trust_remote_code=True fix
        DummyHunyuanVideoModelSpecification,
        DummyCogVideoXModelSpecification,
        DummyLTXVideoModelSpecification,
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
    def test_load_pipeline_with_disk_offload(self, mock_enable_group_offload, model_specification_class):
        """Test that disk offloading is properly enabled when loading the pipeline."""

        # Create model specification
        model_spec = model_specification_class()

        # Call load_pipeline with disk offloading enabled
        model_spec.load_pipeline(
            enable_group_offload=True,
            group_offload_to_disk_path="/tmp/offload_dir",
        )

        # Assert that enable_group_offload_on_components was called with the correct arguments
        mock_enable_group_offload.assert_called_once()

        # Check the call arguments - they are passed as keyword arguments
        call_kwargs = mock_enable_group_offload.call_args.kwargs

        assert "components" in call_kwargs
        assert "device" in call_kwargs
        assert isinstance(call_kwargs["components"], dict)
        assert isinstance(call_kwargs["device"], torch.device)
        assert call_kwargs["offload_to_disk_path"] == "/tmp/offload_dir"

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

        # Determine the correct logger path based on the model specification class
        # Check the base class to determine which model type this is
        base_classes = [cls.__name__ for cls in model_specification_class.__mro__]

        if "FluxModelSpecification" in base_classes:
            logger_path = "finetrainers.models.flux.base_specification.logger"
        elif "HunyuanVideoModelSpecification" in base_classes:
            logger_path = "finetrainers.models.hunyuan_video.base_specification.logger"
        elif "CogVideoXModelSpecification" in base_classes:
            logger_path = "finetrainers.models.cogvideox.base_specification.logger"
        elif "LTXVideoModelSpecification" in base_classes:
            logger_path = "finetrainers.models.ltx_video.base_specification.logger"
        elif "WanModelSpecification" in base_classes:
            logger_path = "finetrainers.models.wan.base_specification.logger"
        else:
            # Default fallback
            logger_path = "finetrainers.models.flux.base_specification.logger"

        # Mock the logger at the module level where it's used
        with patch(logger_path) as mock_logger:
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
