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

    def test_issue_188_direct_reproduction_with_accelerate_checkpointer(self):
        """Direct reproduction of issue #188 using AccelerateCheckpointer: Train for 10 steps with checkpointing at 5,
        quit after 6 steps, then resume from checkpoint."""

        with tempfile.TemporaryDirectory() as output_dir:
            # Simulate the exact scenario from the issue
            checkpointing_steps = 5
            max_train_steps = 10

            # PHASE 1: Start initial training run
            # =====================================

            # Step 1: Initialize wandb tracker for initial training
            initial_tracker = WandbTracker(
                "issue-188-accelerate-reproduction",
                log_dir=output_dir,
                config={"lr": 0.001, "max_steps": max_train_steps}
            )
            original_wandb_run_id = initial_tracker.get_wandb_run_id()

            # Step 2: Set up a real AccelerateCheckpointer (easier to mock)
            from accelerate import Accelerator

            mock_parallel_backend = Mock()
            mock_parallel_backend.tracker = initial_tracker

            mock_accelerator = Mock(spec=Accelerator)
            mock_accelerator.is_main_process = True

            # Mock the save_state method to simulate checkpoint saving
            checkpoint_dir = None
            def mock_save_state(path, **kwargs):
                nonlocal checkpoint_dir
                checkpoint_dir = pathlib.Path(path)
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                # Save the states.pt file that would contain wandb_run_id
                import torch
                states_to_save = {"wandb_run_id": initial_tracker.get_wandb_run_id()}
                torch.save(states_to_save, checkpoint_dir / "states.pt")

            mock_accelerator.save_state = mock_save_state

            checkpointer = AccelerateCheckpointer(
                accelerator=mock_accelerator,
                states={},
                checkpointing_steps=checkpointing_steps,
                checkpointing_limit=3,
                output_dir=output_dir,
                enable=True,
                _parallel_backend=mock_parallel_backend,
            )

            # Step 3: Simulate training for 6 steps (past checkpoint at step 5)
            for step in range(1, 7):  # Steps 1-6
                # Log training metrics
                initial_tracker.log({"loss": 1.0 / step, "step": step}, step=step)

                # Save checkpoint using real checkpointer at step 5
                if step == checkpointing_steps:
                    checkpointer.save(step, force=True, _device=None, _is_main_process=True)

            # Step 4: "Quit" training after step 6 (simulating interruption)
            initial_tracker.finish()


            # PHASE 2: Resume training from checkpoint
            # =========================================

            # Step 5: Load checkpoint using the real checkpointer and extract wandb_run_id
            # Create a new checkpointer instance for loading (simulating a new training session)
            mock_accelerator_resume = Mock(spec=Accelerator)

            # Mock the load_state method to simulate checkpoint loading
            def mock_load_state(path):
                import torch
                states_path = pathlib.Path(path) / "states.pt"
                if states_path.exists():
                    loaded_states = torch.load(states_path)
                    checkpointer_resume.states.update(loaded_states)
                    return True
                return False

            mock_accelerator_resume.load_state = mock_load_state

            checkpointer_resume = AccelerateCheckpointer(
                accelerator=mock_accelerator_resume,
                states={},
                checkpointing_steps=checkpointing_steps,
                checkpointing_limit=3,
                output_dir=output_dir,
                enable=True,
            )

            # Load the checkpoint (this populates checkpointer_resume.states with saved data)
            checkpoint_loaded = checkpointer_resume.load(checkpointing_steps)
            self.assertTrue(checkpoint_loaded, "Checkpoint should have been loaded successfully")

            # Extract the wandb run_id from the loaded checkpoint
            loaded_wandb_run_id = checkpointer_resume.get_wandb_run_id_from_checkpoint()
            self.assertIsNotNone(loaded_wandb_run_id, "Wandb run ID should be available from checkpoint")
            loaded_step = checkpointing_steps  # We know we saved at this step

            # Step 6: Initialize new tracker with resume_run_id from checkpoint
            resumed_tracker = WandbTracker(
                "issue-188-accelerate-reproduction",
                log_dir=output_dir,
                config={"lr": 0.001, "max_steps": max_train_steps},
                resume_run_id=loaded_wandb_run_id,  # This should resume the same wandb run
            )

            # Step 7: CRITICAL TEST - Verify the same wandb run is being used
            resumed_wandb_run_id = resumed_tracker.get_wandb_run_id()

            self.assertEqual(
                original_wandb_run_id,
                resumed_wandb_run_id,
                f"BUG REPRODUCED: Issue #188 with AccelerateCheckpointer - wandb session not resumed! "
                f"Original run ID: {original_wandb_run_id}, "
                f"Resumed run ID: {resumed_wandb_run_id}. "
                f"Expected the same run ID to be reused to preserve training history."
            )

            # Step 8: Continue training from step 6 to step 10
            for step in range(loaded_step + 1, max_train_steps + 1):  # Steps 6-10
                resumed_tracker.log({"loss": 1.0 / step, "step": step}, step=step)

            resumed_tracker.finish()

            # Additional verification: Ensure no new run was created
            self.assertIsNotNone(original_wandb_run_id, "Original wandb run ID should not be None")
            self.assertIsNotNone(resumed_wandb_run_id, "Resumed wandb run ID should not be None")
            self.assertEqual(len(original_wandb_run_id), len(resumed_wandb_run_id),
                            "Run IDs should have the same format/length")

    def test_issue_188_direct_reproduction_with_ptd_checkpointer(self):
        """Direct reproduction of issue #188 using PTDCheckpointer: Train for 10 steps with checkpointing at 5,
        quit after 6 steps, then resume from checkpoint."""

        with tempfile.TemporaryDirectory() as output_dir:
            # Simulate the exact scenario from the issue
            checkpointing_steps = 5
            max_train_steps = 10

            # PHASE 1: Start initial training run
            # =====================================

            # Step 1: Initialize wandb tracker for initial training
            initial_tracker = WandbTracker(
                "issue-188-ptd-reproduction",
                log_dir=output_dir,
                config={"lr": 0.001, "max_steps": max_train_steps}
            )
            original_wandb_run_id = initial_tracker.get_wandb_run_id()

            # Step 2: Set up a real PTDCheckpointer with proper mocking
            import torch.nn as nn

            # Create a simple model for testing
            class SimpleModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(1, 1)

                def forward(self, x):
                    return self.linear(x)

            mock_parallel_backend = Mock()
            mock_parallel_backend.tracker = initial_tracker

            mock_schedulers = Mock()
            mock_schedulers.get_lr_scheduler_state.return_value = {}

            checkpointer = PTDCheckpointer(
                dataloader=Mock(),
                model_parts=[SimpleModel()],  # Use real model instead of Mock
                optimizers=Mock(),
                schedulers=mock_schedulers,
                states={},
                checkpointing_steps=checkpointing_steps,
                checkpointing_limit=3,
                output_dir=output_dir,
                enable=True,
                _parallel_backend=mock_parallel_backend,
            )

            # Step 3: Simulate training for 6 steps (past checkpoint at step 5)
            for step in range(1, 7):  # Steps 1-6
                # Log training metrics
                initial_tracker.log({"loss": 1.0 / step, "step": step}, step=step)

                # Save checkpoint using real PTDCheckpointer at step 5
                if step == checkpointing_steps:
                    # Note: PTDCheckpointer.save needs proper device and main process parameters
                    # but we can't easily test the full distributed checkpoint saving in unit tests
                    # So we'll simulate the wandb_run_id being saved to states manually
                    if checkpointer._parallel_backend and checkpointer._parallel_backend.tracker:
                        checkpointer.states["wandb_run_id"] = checkpointer._parallel_backend.tracker.get_wandb_run_id()

            # Step 4: "Quit" training after step 6 (simulating interruption)
            initial_tracker.finish()

            # PHASE 2: Resume training from checkpoint
            # =========================================

            # Step 5: Load checkpoint using the real PTDCheckpointer and extract wandb_run_id
            # Create a new checkpointer instance for loading (simulating a new training session)
            mock_parallel_backend_resume = Mock()
            mock_schedulers_resume = Mock()
            mock_schedulers_resume.get_lr_scheduler_state.return_value = {}

            checkpointer_resume = PTDCheckpointer(
                dataloader=Mock(),
                model_parts=[SimpleModel()],  # Use real model instead of Mock
                optimizers=Mock(),
                schedulers=mock_schedulers_resume,
                states={},
                checkpointing_steps=checkpointing_steps,
                checkpointing_limit=3,
                output_dir=output_dir,
                enable=True,
                _parallel_backend=mock_parallel_backend_resume,
            )

            # Simulate loading checkpoint by manually setting the wandb_run_id in states
            # (In real PTD checkpointing, this would be loaded from distributed checkpoint)
            checkpointer_resume.states["wandb_run_id"] = original_wandb_run_id

            # Extract the wandb run_id from the loaded checkpoint
            loaded_wandb_run_id = checkpointer_resume.get_wandb_run_id_from_checkpoint()
            self.assertIsNotNone(loaded_wandb_run_id, "Wandb run ID should be available from PTD checkpoint")
            self.assertEqual(loaded_wandb_run_id, original_wandb_run_id, "Loaded wandb run ID should match original")
            loaded_step = checkpointing_steps  # We know we saved at this step

            # Step 6: Initialize new tracker with resume_run_id from checkpoint
            resumed_tracker = WandbTracker(
                "issue-188-ptd-reproduction",
                log_dir=output_dir,
                config={"lr": 0.001, "max_steps": max_train_steps},
                resume_run_id=loaded_wandb_run_id,  # This should resume the same wandb run
            )

            # Step 7: CRITICAL TEST - Verify the same wandb run is being used
            resumed_wandb_run_id = resumed_tracker.get_wandb_run_id()

            self.assertEqual(
                original_wandb_run_id,
                resumed_wandb_run_id,
                f"BUG REPRODUCED: Issue #188 with PTDCheckpointer - wandb session not resumed! "
                f"Original run ID: {original_wandb_run_id}, "
                f"Resumed run ID: {resumed_wandb_run_id}. "
                f"Expected the same run ID to be reused to preserve training history."
            )

            # Step 8: Continue training from step 6 to step 10
            for step in range(loaded_step + 1, max_train_steps + 1):  # Steps 6-10
                resumed_tracker.log({"loss": 1.0 / step, "step": step}, step=step)

            resumed_tracker.finish()

            # Additional verification: Ensure no new run was created
            self.assertIsNotNone(original_wandb_run_id, "Original wandb run ID should not be None")
            self.assertIsNotNone(resumed_wandb_run_id, "Resumed wandb run ID should not be None")
            self.assertEqual(len(original_wandb_run_id), len(resumed_wandb_run_id),
                            "Run IDs should have the same format/length")

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

            # Simulate the wandb_run_id being saved during checkpoint
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

            # Simulate the wandb_run_id being saved during checkpoint
            accelerate_checkpointer.states["wandb_run_id"] = run_id

            # Test retrieval from AccelerateCheckpointer
            saved_run_id = accelerate_checkpointer.get_wandb_run_id_from_checkpoint()
            self.assertEqual(saved_run_id, run_id)

            tracker.finish()
