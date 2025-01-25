from typing import Any, Dict, Set


class Condition:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def __call__(self, *args, **kwargs) -> None:
        raise NotImplementedError(f"Condition::__call__ is not implemented for {self.__class__.__name__}")


def get_condition_parameters_from_dict(accepted_parameters: Set[str], parameters: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in parameters.items() if k in accepted_parameters}
