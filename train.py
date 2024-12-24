import logging
import traceback

from finetrainers import Trainer, parse_arguments
from finetrainers.constants import FINETRAINERS_LOG_LEVEL


logger = logging.getLogger("finetrainers")
logger.setLevel(FINETRAINERS_LOG_LEVEL)


def main():
    try:
        import multiprocessing

        multiprocessing.set_start_method("fork")
    except Exception as e:
        logger.error(
            f'Failed to set multiprocessing start method to "fork". This can lead to poor performance, high memory usage, or crashes. '
            f"See: https://pytorch.org/docs/stable/notes/multiprocessing.html\n"
            f"Error: {e}"
        )

    try:
        args = parse_arguments()
        trainer = Trainer(args)

        trainer.prepare_dataset()
        trainer.prepare_models()
        trainer.prepare_precomputations()
        trainer.prepare_trainable_parameters()
        trainer.prepare_optimizer()
        trainer.prepare_for_training()
        trainer.prepare_trackers()
        trainer.train()
        # trainer.evaluate()

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Exiting...")
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
