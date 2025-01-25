import inspect
from typing import Any, Dict, Optional, Set

from .data_utils import determine_batch_size, should_perform_precomputation
from .diffusion_utils import (
    default_flow_shift,
    get_scheduler_alphas,
    get_scheduler_sigmas,
    prepare_loss_weights,
    prepare_sigmas,
    prepare_target,
    resolution_dependent_timestep_flow_shift,
)
from .file_utils import delete_files, find_files
from .memory_utils import bytes_to_gigabytes, free_memory, get_memory_statistics, make_contiguous
from .model_utils import resolve_component_cls
from .optimizer_utils import get_optimizer, gradient_norm, max_gradient
from .torch_utils import unwrap_model


def get_parameter_names(obj: Any, method_name: Optional[str] = None) -> Set[str]:
    if method_name is not None:
        obj = getattr(obj, method_name)
    return {name for name, _ in inspect.signature(obj).parameters.items()}


def get_non_null_items(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}
