import unittest
from unittest.mock import MagicMock, patch

import torch

from finetrainers.utils.offloading import enable_group_offload_on_components


class TestGroupOffloading(unittest.TestCase):
    def setUp(self):
        # Create mock components for testing - inherit from torch.nn.Module
        self.mock_component1 = MagicMock(spec=torch.nn.Module)
        self.mock_component1.enable_group_offload = MagicMock()
        self.mock_component1.__class__.__name__ = "MockComponent1"

        self.mock_component2 = MagicMock(spec=torch.nn.Module)
        self.mock_component2.enable_group_offload = MagicMock()
        self.mock_component2.__class__.__name__ = "MockComponent2"

        self.components = {
            "component1": self.mock_component1,
            "component2": self.mock_component2,
        }

        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    @patch("finetrainers.utils.offloading._is_group_offload_enabled")
    def test_enable_group_offload_components_with_interface(self, mock_is_enabled):
        """Test that components with the enable_group_offload interface are handled correctly."""
        mock_is_enabled.return_value = False

        enable_group_offload_on_components(
            self.components,
            self.device,
            offload_type="block_level",
            num_blocks_per_group=2,
            use_stream=True,
        )

        # Check that enable_group_offload was called on both components
        self.mock_component1.enable_group_offload.assert_called_once()
        self.mock_component2.enable_group_offload.assert_called_once()

        # Verify the arguments
        args1 = self.mock_component1.enable_group_offload.call_args[1]
        self.assertEqual(args1["offload_type"], "block_level")
        self.assertEqual(args1["num_blocks_per_group"], 2)
        self.assertEqual(args1["use_stream"], True)

        args2 = self.mock_component2.enable_group_offload.call_args[1]
        self.assertEqual(args2["offload_type"], "block_level")
        self.assertEqual(args2["num_blocks_per_group"], 2)
        self.assertEqual(args2["use_stream"], True)

    @patch("finetrainers.utils.offloading._is_group_offload_enabled")
    @patch("finetrainers.utils.offloading.apply_group_offloading")
    def test_enable_group_offload_components_without_interface(self, mock_apply, mock_is_enabled):
        """Test that components without the enable_group_offload interface are handled correctly."""
        mock_is_enabled.return_value = False

        # Remove the enable_group_offload method to simulate components without the interface
        del self.mock_component1.enable_group_offload
        del self.mock_component2.enable_group_offload

        enable_group_offload_on_components(
            self.components,
            self.device,
            offload_type="leaf_level",
            use_stream=False,
        )

        # Check that apply_group_offloading was called for both components
        self.assertEqual(mock_apply.call_count, 2)

        # Verify the arguments for each call
        for call in mock_apply.call_args_list:
            kwargs = call[1]
            self.assertEqual(kwargs["offload_type"], "leaf_level")
            self.assertEqual(kwargs["use_stream"], False)
            self.assertFalse("num_blocks_per_group" in kwargs)

    @patch("finetrainers.utils.offloading._is_group_offload_enabled")
    def test_skip_already_offloaded_components(self, mock_is_enabled):
        """Test that components with group offloading already enabled are skipped."""
        # Component1 already has group offloading enabled
        mock_is_enabled.side_effect = lambda x: x == self.mock_component1

        enable_group_offload_on_components(
            self.components,
            self.device,
        )

        # Component1 should be skipped, Component2 should be processed
        self.mock_component1.enable_group_offload.assert_not_called()
        self.mock_component2.enable_group_offload.assert_called_once()

    @patch("finetrainers.utils.offloading._is_group_offload_enabled")
    def test_exclude_components(self, mock_is_enabled):
        """Test that excluded components are skipped."""
        mock_is_enabled.return_value = False

        enable_group_offload_on_components(
            self.components,
            self.device,
            excluded_components=["component1"],
        )

        # Component1 should be excluded, Component2 should be processed
        self.mock_component1.enable_group_offload.assert_not_called()
        self.mock_component2.enable_group_offload.assert_called_once()

    @patch("finetrainers.utils.offloading._DIFFUSERS_AVAILABLE", False)
    def test_import_error_handling(self):
        """Test that ImportError is handled correctly."""
        with self.assertRaises(ImportError) as context:
            enable_group_offload_on_components(
                self.components,
                self.device,
                required_import_error_message="Custom error message",
            )

        # Verify the custom error message
        self.assertEqual(str(context.exception), "Custom error message")


if __name__ == "__main__":
    unittest.main()
