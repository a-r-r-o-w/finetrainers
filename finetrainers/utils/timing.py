import time
from dataclasses import dataclass
from enum import Enum

import torch

from finetrainers.constants import FINETRAINERS_ENABLE_TIMING
from finetrainers.logging import get_logger


logger = get_logger()


class TimerDevice(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"


@dataclass
class TimerData:
    name: str
    device: TimerDevice
    start_time: float = 0.0
    end_time: float = 0.0


class Timer:
    def __init__(self, name: str, device: TimerDevice, device_sync: bool = False):
        self.data = TimerData(name=name, device=device)

        self._device_sync = device_sync
        self._start_event = None
        self._end_event = None
        self._active = False
        self._enabled = FINETRAINERS_ENABLE_TIMING

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end()
        return False

    def start(self):
        if self._active:
            logger.warning(f"Timer {self.data.name} is already running. Please stop it before starting again.")
            return
        self._active = True
        if not self._enabled:
            return
        if self.data.device == TimerDevice.CUDA and torch.cuda.is_available():
            self._start_cuda()
        else:
            self._start_cpu()
            if not self.data.device == TimerDevice.CPU:
                logger.warning(
                    f"Timer device {self.data.device} is either not supported or incorrect device selected. Falling back to CPU."
                )

    def end(self):
        if not self._active:
            logger.warning(f"Timer {self.data.name} is not running. Please start it before stopping.")
            return
        self._active = False
        if not self._enabled:
            return
        if self.data.device == TimerDevice.CUDA and torch.cuda.is_available():
            self._end_cuda()
        else:
            self._end_cpu()
            if not self.data.device == TimerDevice.CPU:
                logger.warning(
                    f"Timer device {self.data.device} is either not supported or incorrect device selected. Falling back to CPU."
                )

    @property
    def elapsed_time(self) -> float:
        if self._active:
            if self.data.device == TimerDevice.CUDA and torch.cuda.is_available():
                premature_end_event = torch.cuda.Event(enable_timing=True)
                premature_end_event.record()
                premature_end_event.synchronize()
                return self._start_event.elapsed_time(premature_end_event) / 1000.0
            else:
                return time.time() - self.data.start_time
        else:
            if self.data.device == TimerDevice.CUDA and torch.cuda.is_available():
                return self._start_event.elapsed_time(self._end_event) / 1000.0
            else:
                return self.data.end_time - self.data.start_time

    def _start_cpu(self):
        self.data.start_time = time.time()

    def _start_cuda(self):
        torch.cuda.synchronize()
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._end_event = torch.cuda.Event(enable_timing=True)
        self._start_event.record()

    def _end_cpu(self):
        self.data.end_time = time.time()

    def _end_cuda(self):
        if self._device_sync:
            torch.cuda.synchronize()
        self._end_event.record()
