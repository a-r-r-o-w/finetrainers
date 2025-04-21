import inspect
from typing import Any, Dict, List


class ProcessorMixin:
    def __init__(self) -> None:
        self._forward_parameter_names = inspect.signature(self.forward).parameters.keys()
        self.output_names: List[str] = None
        self.input_names: Dict[str, Any] = None

    def __call__(self, *args, **kwargs) -> Any:
        shallow_copy_kwargs = dict(kwargs.items())
        if self.input_names is not None:
            for k, v in self.input_names.items():
                if k in shallow_copy_kwargs:
                    shallow_copy_kwargs[v] = shallow_copy_kwargs.pop(k)
        acceptable_kwargs = {k: v for k, v in shallow_copy_kwargs.items() if k in self._forward_parameter_names}
        output = self.forward(*args, **acceptable_kwargs)
        if "__drop__" in output:
            output.pop("__drop__")
        return output

    def forward(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("ProcessorMixin::forward method should be implemented by the subclass.")
