import functools

from peft.tuners.tuners_utils import BaseTunerLayer

from finetrainers.patches.utils import DisableTensorToDtype


def patch_peft_move_adapter_to_device_of_base_layer() -> None:
    _perform_patch_move_adapter_to_device_of_base_layer()


def _perform_patch_move_adapter_to_device_of_base_layer() -> None:
    BaseTunerLayer._move_adapter_to_device_of_base_layer = _patched_move_adapter_to_device_of_base_layer(
        BaseTunerLayer._move_adapter_to_device_of_base_layer
    )


def _patched_move_adapter_to_device_of_base_layer(func) -> None:
    # TODO(aryan): This is really unsafe probably and may break things. It works for now, but revisit and refactor.
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        with DisableTensorToDtype():
            return func(self, *args, **kwargs)

    return wrapper
