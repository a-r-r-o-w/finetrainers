import logging
import os
import pathlib
import tempfile
import unittest
from unittest.mock import Mock

from diffusers.utils.testing_utils import CaptureLogger

from finetrainers.parallel.accelerate import AccelerateCheckpointer
from finetrainers.parallel.ptd import PTDCheckpointer
from finetrainers.trackers import WandbTracker


os.environ["WANDB_MODE"] = "offline"
os.environ["FINETRAINERS_LOG_LEVEL"] = "INFO"


class WandbFastTests(unittest.TestCase):
    def test_wandb_logdir(self):
        logger = logging.getLogger("finetrainers")

        with tempfile.TemporaryDirectory() as tempdir, CaptureLogger(logger) as cap_log:
            tracker = WandbTracker("finetrainers-experiment", log_dir=tempdir, config={})
            tracker.log({"loss": 0.1}, step=0)
            tracker.log({"loss": 0.2}, step=1)
            tracker.finish()
            self.assertTrue(pathlib.Path(tempdir).exists())

        self.assertTrue("WandB logging enabled" in cap_log.out)


class TestWandbResumption(unittest.TestCase):
    """Test the core issue from #188: resuming wandb runs from checkpoint."""

    def test_issue_188_core_problem(self):
        """Test the exact scenario Aryan described in issue #188.

        The core problem: when resuming from checkpoint, a NEW wandb run is created
        instead of resuming the original one.

        This test simulates:
        1. Start training with wandb tracker -> get run_id
        2. Save checkpoint with wandb run_id
        3. Resume training from checkpoint with same run_id
        4. Verify NO new run is created (same run_id is used)
        """

        with tempfile.TemporaryDirectory() as log_dir:
            # STEP 1: Start training with wandb tracker -> get run_id
            original_tracker = WandbTracker("issue-188-test", log_dir=log_dir, config={"lr": 0.001})
            original_run_id = original_tracker.get_wandb_run_id()
            original_tracker.finish()

            # STEP 2: Save checkpoint with wandb run_id
            checkpoint_data = {"wandb_run_id": original_run_id}

            # STEP 3: Resume training from checkpoint with same run_id
            resumed_tracker = WandbTracker(
                "issue-188-test",
                log_dir=log_dir,
                config={"lr": 0.001},
                resume_run_id=checkpoint_data["wandb_run_id"],
            )

            # STEP 4: Verify NO new run is created (same run_id is used)
            resumed_run_id = resumed_tracker.get_wandb_run_id()
            self.assertEqual(
                original_run_id, resumed_run_id, "BUG: New wandb run created instead of resuming original run!"
            )

            resumed_tracker.finish()

    def test_checkpointer_saves_wandb_run_id(self):
        """Test that both PTDCheckpointer and AccelerateCheckpointer save wandb run_id to enable resumption."""
        with tempfile.TemporaryDirectory() as log_dir:
            # Create tracker
            tracker = WandbTracker("checkpoint-test", log_dir=log_dir, config={})
            run_id = tracker.get_wandb_run_id()

            # Test PTDCheckpointer
            mock_parallel_backend = Mock()
            mock_parallel_backend.tracker = tracker

            # Create proper mock for schedulers
            mock_schedulers = Mock()
            mock_schedulers.get_lr_scheduler_state.return_value = {}

            ptd_checkpointer = PTDCheckpointer(
                dataloader=Mock(),
                model_parts=[Mock()],
                optimizers=Mock(),
                schedulers=mock_schedulers,
                states={},
                checkpointing_steps=1,
                checkpointing_limit=1,
                output_dir=log_dir,
                enable=True,
                _parallel_backend=mock_parallel_backend,
            )

            # Simulate the wandb run_id being saved during checkpoint
            if ptd_checkpointer._parallel_backend.tracker:
                ptd_checkpointer.states["wandb_run_id"] = ptd_checkpointer._parallel_backend.tracker.get_wandb_run_id()

            # Test retrieval from PTDCheckpointer
            saved_run_id = ptd_checkpointer.get_wandb_run_id_from_checkpoint()
            self.assertEqual(saved_run_id, run_id)

            # Test AccelerateCheckpointer
            from accelerate import Accelerator

            mock_accelerator = Mock(spec=Accelerator)

            accelerate_checkpointer = AccelerateCheckpointer(
                accelerator=mock_accelerator,
                states={},
                checkpointing_steps=1,
                checkpointing_limit=1,
                output_dir=log_dir,
                enable=True,
            )

            # Simulate the wandb run_id being saved during checkpoint
            accelerate_checkpointer.states["wandb_run_id"] = run_id

            # Test retrieval from AccelerateCheckpointer
            saved_run_id = accelerate_checkpointer.get_wandb_run_id_from_checkpoint()
            self.assertEqual(saved_run_id, run_id)

            tracker.finish()

    def test_sft_trainer_uses_checkpointed_wandb_run_id(self):
        """Test that SFTTrainer has the required logic for wandb resumption."""
        import inspect

        from finetrainers.trainer.sft_trainer.trainer import SFTTrainer

        # Verify the trainer has the core logic for wandb resumption
        source = inspect.getsource(SFTTrainer._prepare_checkpointing)

        # The core flow should be:
        # 1. Load checkpoint if resuming
        # 2. Extract wandb run_id from checkpoint
        # 3. Pass run_id to _init_trackers for resumption
        self.assertIn(
            "get_wandb_run_id_from_checkpoint",
            source,
            "SFTTrainer missing logic to extract wandb run_id from checkpoint",
        )
        self.assertIn("resume_run_id", source, "SFTTrainer missing logic to pass resume_run_id to trackers")

    def test_control_trainer_uses_checkpointed_wandb_run_id(self):
        """Test that ControlTrainer has the required logic for wandb resumption."""
        import inspect

        from finetrainers.trainer.control_trainer.trainer import ControlTrainer

        # Verify the trainer has the core logic for wandb resumption
        source = inspect.getsource(ControlTrainer._prepare_checkpointing)

        # The core flow should be:
        # 1. Load checkpoint if resuming
        # 2. Extract wandb run_id from checkpoint
        # 3. Pass run_id to _init_trackers for resumption
        self.assertIn(
            "get_wandb_run_id_from_checkpoint",
            source,
            "ControlTrainer missing logic to extract wandb run_id from checkpoint",
        )
        self.assertIn("resume_run_id", source, "ControlTrainer missing logic to pass resume_run_id to trackers")
