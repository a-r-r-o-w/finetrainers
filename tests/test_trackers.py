import logging
import os
import pathlib
import tempfile
import unittest

from diffusers.utils.testing_utils import CaptureLogger
from .models.cogview4.base_specification import DummyCogView4ModelSpecification  # noqa
from tests.trainer import SFTTrainerFastTestsMixin
from finetrainers import BaseArgs, SFTTrainer, TrainingType, get_logger

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


class SFTTrainerLoRAWandbResumeTests(SFTTrainerFastTestsMixin, unittest.TestCase):
    model_specification_cls = DummyCogView4ModelSpecification

    def get_args(self) -> BaseArgs:
        args = self.get_base_args()
        args.checkpointing_steps = 5
        args.parallel_backend = "accelerate"
        args.training_type = TrainingType.LORA
        args.rank = 4
        args.lora_alpha = 4
        args.target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
        return args

    def test_wandb_session_resumption_with_checkpoint(self):
        """
        Test the core issue: wandb session should be continued when resuming from checkpoint.

        Steps:
        1. Start training for 6 steps (with checkpointing every 5 steps)
        2. Verify checkpoint is created at step 5
        3. Resume training from checkpoint at step 5 for additional steps
        4. Verify that the same wandb session ID is maintained
        """
        logger = logging.getLogger("finetrainers")

        with CaptureLogger(logger) as cap_log:
            # Phase 1: Initial training run (6 steps, checkpoint at step 5)
            args_phase1 = self.get_args()
            args_phase1.train_steps = 6  # Train for 6 steps (will checkpoint at step 5)

            model_specification_1 = self.model_specification_cls()
            trainer_phase1 = SFTTrainer(args_phase1, model_specification_1)
            trainer_phase1.run()

            # Verify checkpoint was created at step 5
            checkpoint_dir = pathlib.Path(self.tmpdir.name) / "finetrainers_step_5"
            self.assertTrue(checkpoint_dir.exists(), f"Checkpoint should exist at {checkpoint_dir}")

            # Extract the wandb run ID from the first training run
            # This should be stored in the checkpoint
            original_wandb_run_id = trainer_phase1.checkpointer.get_wandb_run_id_from_checkpoint()
            self.assertIsNotNone(original_wandb_run_id, "WandB run ID should be saved in checkpoint")

            # Clean up the first trainer
            del trainer_phase1

            # Phase 2: Resume training from checkpoint
            args_phase2 = self.get_args()
            args_phase2.resume_from_checkpoint = "finetrainers_step_5"  # Resume from step 5 checkpoint

            model_specification_2 = self.model_specification_cls()
            trainer_phase2 = SFTTrainer(args_phase2, model_specification_2)
            trainer_phase2.run()

            # Verify that the resumed training uses the same wandb run ID
            resumed_wandb_run_id = None
            if hasattr(trainer_phase2.state.parallel_backend, 'tracker') and trainer_phase2.state.parallel_backend.tracker:
                resumed_wandb_run_id = trainer_phase2.state.parallel_backend.tracker.get_wandb_run_id()

            self.assertIsNotNone(resumed_wandb_run_id, "Resumed training should have a wandb run ID")
            self.assertEqual(
                original_wandb_run_id,
                resumed_wandb_run_id,
                f"WandB run ID should be the same after resumption. "
                f"Original: {original_wandb_run_id}, Resumed: {resumed_wandb_run_id}"
            )

            # Verify that training continued from the correct step
            final_step = trainer_phase2.state.train_state.step
            self.assertGreaterEqual(final_step, 10, "Training should have reached at least step 10")

            # Clean up the second trainer
            del trainer_phase2

        # Verify logging contains resumption messages
        log_output = cap_log.out
        self.assertIn("WandB logging enabled", log_output)

        # If resumption happened correctly, we should see a resumption message
        # The exact message depends on the WandB implementation, but we can check for key indicators
        if "Resuming WandB run with ID:" in log_output:
            self.assertIn(f"Resuming WandB run with ID: {original_wandb_run_id}", log_output)
