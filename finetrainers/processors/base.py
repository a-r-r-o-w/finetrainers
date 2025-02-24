import inspect
from typing import Any


class ProcessorMixin:
    def __init__(self) -> None:
        self._forward_parameter_names = inspect.signature(self.forward).parameters.keys()

    def __call__(self, *args, **kwargs) -> Any:
        acceptable_kwargs = {k: v for k, v in kwargs.items() if k in self._forward_parameter_names}
        return self.forward(*args, **acceptable_kwargs)

    def forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError("ProcessorMixin::forward method should be implemented by the subclass.")
