from typing import Any, Dict, Set


class Processor:
    _input_names = []
    _output_names = []

    def __init__(self) -> None:
        pass

    def __call__(self) -> None:
        raise NotImplementedError(f"Processor::__call__ is not implemented for {self.__class__.__name__}")


def get_processor_parameters_from_dict(accepted_parameters: Set[str], parameters: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in parameters.items() if k in accepted_parameters}
