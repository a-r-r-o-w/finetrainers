import unittest

from finetrainers.args import BaseArgs, _validate_validation_args


class TestOffloadingArgsValidation(unittest.TestCase):
    def setUp(self):
        self.args = BaseArgs()
        self.args.enable_model_cpu_offload = False
        self.args.enable_group_offload = False
        self.args.group_offload_type = "block_level"
        self.args.group_offload_blocks_per_group = 1
        self.args.pp_degree = 1
        self.args.dp_degree = 1
        self.args.dp_shards = 1
        self.args.cp_degree = 1
        self.args.tp_degree = 1

    def test_mutually_exclusive_offloading_methods(self):
        """Test that enabling both offloading methods raises a ValueError."""
        self.args.enable_model_cpu_offload = True
        self.args.enable_group_offload = True

        with self.assertRaises(ValueError) as context:
            _validate_validation_args(self.args)

        self.assertIn("Model CPU offload and group offload cannot be enabled at the same time", str(context.exception))

    def test_model_cpu_offload_multi_gpu_restriction(self):
        """Test that model CPU offload with multi-GPU setup raises a ValueError."""
        self.args.enable_model_cpu_offload = True
        self.args.dp_degree = 2  # Set multi-GPU configuration

        with self.assertRaises(ValueError) as context:
            _validate_validation_args(self.args)

        self.assertIn("Model CPU offload is not supported on multi-GPU", str(context.exception))

    def test_group_offload_blocks_validation(self):
        """Test that group offload with invalid blocks_per_group raises a ValueError."""
        self.args.enable_group_offload = True
        self.args.group_offload_type = "block_level"
        self.args.group_offload_blocks_per_group = 0  # Invalid value

        with self.assertRaises(ValueError) as context:
            _validate_validation_args(self.args)

        self.assertIn("blocks_per_group must be at least 1", str(context.exception))

    def test_valid_group_offload_args(self):
        """Test that valid group offload arguments pass validation."""
        self.args.enable_group_offload = True
        self.args.group_offload_type = "block_level"
        self.args.group_offload_blocks_per_group = 2

        try:
            _validate_validation_args(self.args)
        except ValueError:
            self.fail("_validate_validation_args() raised ValueError unexpectedly!")

    def test_leaf_level_offload_blocks_ignored(self):
        """Test that blocks_per_group is ignored for leaf_level offloading."""
        self.args.enable_group_offload = True
        self.args.group_offload_type = "leaf_level"
        self.args.group_offload_blocks_per_group = 0  # Would be invalid for block_level

        try:
            _validate_validation_args(self.args)
        except ValueError:
            self.fail("_validate_validation_args() raised ValueError unexpectedly!")


if __name__ == "__main__":
    unittest.main()
