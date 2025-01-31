import logging
import os
import pathlib
import sys
import tempfile
import unittest

from diffusers.utils.testing_utils import CaptureLogger


os.environ["WANDB_MODE"] = "offline"
project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))


from finetrainers.trackers import WandbTracker  # noqa


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
