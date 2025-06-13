import unittest
import torch
import pytest
from unittest.mock import patch, MagicMock

from finetrainers.models.flux import FluxModelSpecification
from finetrainers.models.cogview4 import CogView4ModelSpecification
from finetrainers.models.cogvideox import CogVideoXModelSpecification
from finetrainers.models.ltx_video import LTXVideoModelSpecification
from finetrainers.models.hunyuan_video import HunyuanVideoModelSpecification
from finetrainers.models.wan import WanModelSpecification


# Skip tests if CUDA is not available
has_cuda = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not has_cuda, reason="Test requires CUDA")


class DummyFluxModelSpecification(FluxModelSpecification):
    def __init__(self, **kwargs):
        super().__init__(pretrained_model_name_or_path="hf-internal-testing/tiny-flux-pipe", **kwargs)


class DummyCogVideoXModelSpecification(CogVideoXModelSpecification):
    def __init__(self, **kwargs):
        super().__init__(pretrained_model_name_or_path="hf-internal-testing/tiny-cogvideox", **kwargs)


class DummyLTXVideoModelSpecification(LTXVideoModelSpecification):
    def __init__(self, **kwargs):
        super().__init__(pretrained_model_name_or_path="hf-internal-testing/tiny-ltx-video", **kwargs)


class DummyHunyuanVideoModelSpecification(HunyuanVideoModelSpecification):
    def __init__(self, **kwargs):
        super().__init__(pretrained_model_name_or_path="hf-internal-testing/tiny-hunyuan-video", **kwargs)


class DummyCogView4ModelSpecification(CogView4ModelSpecification):
    def __init__(self, **kwargs):
        super().__init__(pretrained_model_name_or_path="hf-internal-testing/tiny-cogview4", **kwargs)


class DummyWanModelSpecification(WanModelSpecification):
    def __init__(self, **kwargs):
        super().__init__(pretrained_model_name_or_path="hf-internal-testing/tiny-wan", **kwargs)


@pytest.mark.parametrize(
    "model_specification_class",
    [
        DummyFluxModelSpecification,
        DummyCogVideoXModelSpecification,
        DummyLTXVideoModelSpecification,
        DummyHunyuanVideoModelSpecification,
        DummyCogView4ModelSpecification,
        DummyWanModelSpecification,
    ],
)
class TestGroupOffloadingIntegration:
    @patch("diffusers.pipelines.pipeline_utils.DiffusionPipeline.from_pretrained")
    @patch("finetrainers.utils.offloading.enable_group_offload_on_components")
    def test_load_pipeline_with_group_offload(
        self, mock_enable_group_offload, mock_from_pretrained, model_specification_class
    ):
        """Test that group offloading is properly enabled when loading the pipeline."""
        # Mock the pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.device = torch.device("cuda:0") if has_cuda else torch.device("cpu")
        mock_pipeline.components = {"transformer": MagicMock(), "vae": MagicMock()}
        mock_from_pretrained.return_value = mock_pipeline

        # Create model specification
        model_spec = model_specification_class()

        # Call load_pipeline with group offloading enabled
        pipeline = model_spec.load_pipeline(
            enable_group_offload=True,
            group_offload_type="block_level",
            group_offload_blocks_per_group=4,
            group_offload_use_stream=True,
        )

        # Assert that enable_group_offload_on_components was called with the correct arguments
        mock_enable_group_offload.assert_called_once()

        args = mock_enable_group_offload.call_args[0]
        kwargs = mock_enable_group_offload.call_args[1]

        self.assertEqual(args[0], mock_pipeline.components)
        self.assertEqual(args[1], mock_pipeline.device)
        self.assertEqual(kwargs["offload_type"], "block_level")
        self.assertEqual(kwargs["num_blocks_per_group"], 4)
        self.assertEqual(kwargs["use_stream"], True)

    @patch("diffusers.pipelines.pipeline_utils.DiffusionPipeline.from_pretrained")
    @patch("diffusers.pipelines.pipeline_utils.DiffusionPipeline.enable_model_cpu_offload")
    @patch("finetrainers.utils.offloading.enable_group_offload_on_components")
    def test_mutually_exclusive_offload_methods(
        self, mock_enable_group_offload, mock_enable_model_cpu_offload, mock_from_pretrained, model_specification_class
    ):
        """Test that only one offloading method is used when both are enabled."""
        # Mock the pipeline
        mock_pipeline = MagicMock()
        mock_from_pretrained.return_value = mock_pipeline

        # Create model specification
        model_spec = model_specification_class()

        # Call load_pipeline with both offloading methods enabled (model offload should take precedence)
        pipeline = model_spec.load_pipeline(
            enable_model_cpu_offload=True,
            enable_group_offload=True,
        )

        # Assert that model_cpu_offload was called and group_offload was not
        mock_enable_model_cpu_offload.assert_called_once()
        mock_enable_group_offload.assert_not_called()

    @patch("diffusers.pipelines.pipeline_utils.DiffusionPipeline.from_pretrained")
    @patch("finetrainers.utils.offloading.enable_group_offload_on_components")
    def test_import_error_handling(
        self, mock_enable_group_offload, mock_from_pretrained, model_specification_class
    ):
        """Test that ImportError is handled gracefully when diffusers version is too old."""
        # Mock the pipeline
        mock_pipeline = MagicMock()
        mock_from_pretrained.return_value = mock_pipeline

        # Simulate an ImportError when trying to use group offloading
        mock_enable_group_offload.side_effect = ImportError("Module not found")

        # Mock the logger to check for warning message
        with patch("finetrainers.logging.get_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            # Create model specification
            model_spec = model_specification_class()

            # Call load_pipeline with group offloading enabled
            pipeline = model_spec.load_pipeline(
                enable_group_offload=True,
            )

            # Assert that a warning was logged
            mock_logger.warning.assert_called_once()
            warning_msg = mock_logger.warning.call_args[0][0]
            self.assertIn("Failed to enable group offloading", warning_msg)
            self.assertIn("Using standard pipeline without offloading", warning_msg)


if __name__ == "__main__":
    unittest.main()