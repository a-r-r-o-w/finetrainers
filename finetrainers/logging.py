import logging

from .constants import FINETRAINERS_LOG_LEVEL


logger = logging.getLogger("finetrainers")
logger.setLevel(FINETRAINERS_LOG_LEVEL)
console_handler = logging.StreamHandler()
console_handler.setLevel(FINETRAINERS_LOG_LEVEL)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
