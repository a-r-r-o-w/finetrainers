import inspect
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ._rename_this_file_checkpointing import get_intermediate_ckpt_path, get_latest_ckpt_path_to_resume_from
from .checkpoint_utils import apply_activation_checkpointing
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
from .file_utils import delete_files, find_files, string_to_filename
from .hub_utils import save_model_card
from .memory_utils import bytes_to_gigabytes, free_memory, get_memory_statistics, make_contiguous
from .model_utils import resolve_component_cls
from .optimizer_utils import get_optimizer, gradient_norm, max_gradient
from .torch_utils import (
    align_device_and_dtype,
    clip_grad_norm_,
    enable_determinism,
    expand_tensor_dims,
    get_device_info,
    synchronize_device,
    unwrap_model,
)


apply_gradient_checkpointing = apply_activation_checkpointing


def get_parameter_names(obj: Any, method_name: Optional[str] = None) -> Set[str]:
    if method_name is not None:
        obj = getattr(obj, method_name)
    return {name for name, _ in inspect.signature(obj).parameters.items()}


def get_non_null_items(
    x: Union[List[Any], Tuple[Any], Dict[str, Any]]
) -> Union[List[Any], Tuple[Any], Dict[str, Any]]:
    if isinstance(x, dict):
        return {k: v for k, v in x.items() if v is not None}
    if isinstance(x, (list, tuple)):
        return type(x)(v for v in x if v is not None)
