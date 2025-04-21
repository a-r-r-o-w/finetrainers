import logging
import os
from typing import TYPE_CHECKING, Union

import diffusers
import transformers

from .constants import FINETRAINERS_LOG_LEVEL


if TYPE_CHECKING:
    from .parallel import ParallelBackendType


class FinetrainersLoggerAdapter(logging.LoggerAdapter):
    def __init__(self, logger: logging.Logger, parallel_backend: "ParallelBackendType" = None) -> None:
        super().__init__(logger, {})
        self.parallel_backend = parallel_backend
        self._log_freq = {}
        self._log_freq_counter = {}

    def log(
        self,
        level,
        msg,
        *args,
        main_process_only: bool = False,
        local_main_process_only: bool = True,
        in_order: bool = False,
        **kwargs,
    ):
        # set `stacklevel` to exclude ourself in `Logger.findCaller()` while respecting user's choice
        kwargs.setdefault("stacklevel", 2)

        if not self.isEnabledFor(level):
            return

        if self.parallel_backend is None:
            if int(os.environ.get("RANK", 0)) == 0:
                msg, kwargs = self.process(msg, kwargs)
                self.logger.log(level, msg, *args, **kwargs)
            return

        if (main_process_only or local_main_process_only) and in_order:
            raise ValueError(
                "Cannot set `main_process_only` or `local_main_process_only` to True while `in_order` is True."
            )

        if (main_process_only and self.parallel_backend.is_main_process) or (
            local_main_process_only and self.parallel_backend.is_local_main_process
        ):
            msg, kwargs = self.process(msg, kwargs)
            self.logger.log(level, msg, *args, **kwargs)
            return

        if in_order:
            for i in range(self.parallel_backend.world_size):
                if self.rank == i:
                    msg, kwargs = self.process(msg, kwargs)
                    self.logger.log(level, msg, *args, **kwargs)
                self.parallel_backend.wait_for_everyone()
            return

        if not main_process_only and not local_main_process_only:
            msg, kwargs = self.process(msg, kwargs)
            self.logger.log(level, msg, *args, **kwargs)
            return

    def log_freq(
        self,
        level: str,
        name: str,
        msg: str,
        frequency: int,
        *,
        main_process_only: bool = False,
        local_main_process_only: bool = True,
        in_order: bool = False,
        **kwargs,
    ) -> None:
        if frequency <= 0:
            return
        if name not in self._log_freq_counter:
            self._log_freq[name] = frequency
            self._log_freq_counter[name] = 0
        if self._log_freq_counter[name] % self._log_freq[name] == 0:
            self.log(
                level,
                msg,
                main_process_only=main_process_only,
                local_main_process_only=local_main_process_only,
                in_order=in_order,
                **kwargs,
            )
        self._log_freq_counter[name] += 1


def get_logger() -> Union[logging.Logger, FinetrainersLoggerAdapter]:
    global _logger
    return _logger


def _set_parallel_backend(parallel_backend: "ParallelBackendType") -> FinetrainersLoggerAdapter:
    _logger.parallel_backend = parallel_backend


_logger = logging.getLogger("finetrainers")
_logger.setLevel(FINETRAINERS_LOG_LEVEL)
_console_handler = logging.StreamHandler()
_console_handler.setLevel(FINETRAINERS_LOG_LEVEL)
_formatter = logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
_console_handler.setFormatter(_formatter)
_logger.addHandler(_console_handler)
_logger.propagate = False
_logger = FinetrainersLoggerAdapter(_logger)


def set_dependency_log_level(verbose: int = 0, is_local_main_process: bool = False) -> None:
    transformers_log_level = transformers.utils.logging.set_verbosity_error
    diffusers_log_level = diffusers.utils.logging.set_verbosity_error

    if verbose == 0:
        if is_local_main_process:
            transformers_log_level = transformers.utils.logging.set_verbosity_warning
            diffusers_log_level = diffusers.utils.logging.set_verbosity_warning
    elif verbose == 1:
        if is_local_main_process:
            transformers_log_level = transformers.utils.logging.set_verbosity_info
            diffusers_log_level = diffusers.utils.logging.set_verbosity_info
    elif verbose == 2:
        if is_local_main_process:
            transformers_log_level = transformers.utils.logging.set_verbosity_debug
            diffusers_log_level = diffusers.utils.logging.set_verbosity_debug
    else:
        transformers_log_level = transformers.utils.logging.set_verbosity_debug
        diffusers_log_level = diffusers.utils.logging.set_verbosity_debug

    transformers_log_level()
    diffusers_log_level()
