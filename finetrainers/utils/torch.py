import math
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.backends
import torch.distributed as dist
import torch.distributed.tensor

from finetrainers.logging import get_logger


logger = get_logger()

_STRING_TO_DTYPE = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

_DTYPE_TO_STRING = {v: k for k, v in _STRING_TO_DTYPE.items()}

_HAS_ERRORED_CLIP_GRAD_NORM_WHILE_HANDLING_FAILING_DTENSOR_CASES = False


def align_device_and_dtype(
    x: Union[torch.Tensor, Dict[str, torch.Tensor]],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    if isinstance(x, torch.Tensor):
        if device is not None:
            x = x.to(device)
        if dtype is not None:
            x = x.to(dtype)
    elif isinstance(x, dict):
        if device is not None:
            x = {k: align_device_and_dtype(v, device, dtype) for k, v in x.items()}
        if dtype is not None:
            x = {k: align_device_and_dtype(v, device, dtype) for k, v in x.items()}
    return x


def apply_compile(model: torch.nn.Module, compile_scope: str) -> torch.nn.Module:
    r"""Apply torch.compile to a model or its submodules if not already compiled."""
    if getattr(model, "_torch_compiled", False):
        return model  # Already compiled

    if compile_scope == "full":
        model = torch.compile(model)
        setattr(model, "_torch_compiled", True)
    elif compile_scope == "regional":
        if isinstance(model, torch.nn.ModuleList):
            for name, module in model.named_children():
                if not getattr(module, "_torch_compiled", False):
                    compiled_module = torch.compile(module)
                    setattr(compiled_module, "_torch_compiled", True)
                    model.register_module(name, compiled_module)
        else:
            for name, module in model.named_children():
                apply_compile(module, compile_scope)
    else:
        raise ValueError(f"Unknown compile mode: {compile_scope}. Use 'full' or 'regional'.")

    return model


def _clip_grad_norm_while_handling_failing_dtensor_cases(
    parameters: Union[torch.Tensor, List[torch.Tensor]],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: Optional[bool] = None,
    pp_mesh: Optional[torch.distributed.device_mesh.DeviceMesh] = None,
) -> Optional[torch.Tensor]:
    global _HAS_ERRORED_CLIP_GRAD_NORM_WHILE_HANDLING_FAILING_DTENSOR_CASES

    if not _HAS_ERRORED_CLIP_GRAD_NORM_WHILE_HANDLING_FAILING_DTENSOR_CASES:
        try:
            return clip_grad_norm_(parameters, max_norm, norm_type, error_if_nonfinite, foreach, pp_mesh)
        except NotImplementedError as e:
            if "DTensor does not support cross-mesh operation" in str(e):
                # https://github.com/pytorch/pytorch/issues/134212
                logger.warning(
                    "DTensor does not support cross-mesh operation. If you haven't fully tensor-parallelized your "
                    "model, while combining other parallelisms such as FSDP, it could be the reason for this error. "
                    "Gradient clipping will be skipped and gradient norm will not be logged."
                )
        except Exception as e:
            logger.warning(
                f"An error occurred while clipping gradients: {e}. Gradient clipping will be skipped and gradient "
                f"norm will not be logged."
            )
            _HAS_ERRORED_CLIP_GRAD_NORM_WHILE_HANDLING_FAILING_DTENSOR_CASES = True
    return None


# Copied from https://github.com/pytorch/torchtitan/blob/4a169701555ab9bd6ca3769f9650ae3386b84c6e/torchtitan/utils.py#L362
@torch.no_grad()
def clip_grad_norm_(
    parameters: Union[torch.Tensor, List[torch.Tensor]],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: Optional[bool] = None,
    pp_mesh: Optional[torch.distributed.device_mesh.DeviceMesh] = None,
) -> torch.Tensor:
    r"""
    Clip the gradient norm of parameters.

    Gradient norm clipping requires computing the gradient norm over the entire model.
    `torch.nn.utils.clip_grad_norm_` only computes gradient norm along DP/FSDP/TP dimensions.
    We need to manually reduce the gradient norm across PP stages.
    See https://github.com/pytorch/torchtitan/issues/596 for details.

    Args:
        parameters (`torch.Tensor` or `List[torch.Tensor]`):
            Tensors that will have gradients normalized.
        max_norm (`float`):
            Maximum norm of the gradients after clipping.
        norm_type (`float`, defaults to `2.0`):
            Type of p-norm to use. Can be `inf` for infinity norm.
        error_if_nonfinite (`bool`, defaults to `False`):
            If `True`, an error is thrown if the total norm of the gradients from `parameters` is `nan`, `inf`, or `-inf`.
        foreach (`bool`, defaults to `None`):
            Use the faster foreach-based implementation. If `None`, use the foreach implementation for CUDA and CPU native tensors
            and silently fall back to the slow implementation for other device types.
        pp_mesh (`torch.distributed.device_mesh.DeviceMesh`, defaults to `None`):
            Pipeline parallel device mesh. If not `None`, will reduce gradient norm across PP stages.

    Returns:
        `torch.Tensor`:
            Total norm of the gradients
    """
    grads = [p.grad for p in parameters if p.grad is not None]

    # TODO(aryan): Wait for next Pytorch release to use `torch.nn.utils.get_total_norm`
    # total_norm = torch.nn.utils.get_total_norm(grads, norm_type, error_if_nonfinite, foreach)
    total_norm = _get_total_norm(grads, norm_type, error_if_nonfinite, foreach)

    # If total_norm is a DTensor, the placements must be `torch.distributed._tensor.ops.math_ops._NormPartial`.
    # We can simply reduce the DTensor to get the total norm in this tensor's process group
    # and then convert it to a local tensor.
    # It has two purposes:
    #   1. to make sure the total norm is computed correctly when PP is used (see below)
    #   2. to return a reduced total_norm tensor whose .item() would return the correct value
    if isinstance(total_norm, torch.distributed.tensor.DTensor):
        # Will reach here if any non-PP parallelism is used.
        # If only using PP, total_norm will be a local tensor.
        total_norm = total_norm.full_tensor()

    if pp_mesh is not None:
        if math.isinf(norm_type):
            dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=pp_mesh.get_group())
        else:
            total_norm **= norm_type
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=pp_mesh.get_group())
            total_norm **= 1.0 / norm_type

    _clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
    return total_norm


def enable_determinism(
    seed: int,
    world_mesh: Optional[torch.distributed.DeviceMesh] = None,
    deterministic: bool = False,
) -> None:
    r"""
    For all ranks within the same DTensor SPMD group, the same seed will be set.
    For PP groups, different seeds will be set.
    """
    if deterministic:
        logger.info("Deterministic algorithms are enabled (expect performance degradation).")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    if not world_mesh:
        if seed is not None:
            torch.manual_seed(seed)
            os.environ["PYTHONHASHSEED"] = str(seed % 2**32)
            logger.debug(f"Single-process job using seed: {seed}")
        return

    # For PP + SPMD cases, we want to separate the world into the SPMD mesh and the PP mesh,
    # and choose a unique seed for each rank on the PP mesh.
    if torch.distributed.distributed_c10d.get_world_size() > 1 and "pp" in world_mesh.mesh_dim_names:
        pp_mesh = world_mesh["pp"]
        seed += pp_mesh.get_local_rank()
        seed %= 2**64

        info = {
            "pp_rank": pp_mesh.get_local_rank(),
            "global_rank": torch.distributed.distributed_c10d.get_rank(),
            "seed": seed,
        }
        logger.debug(f"Enabling determinism: {info}")
        spmd_mesh_dims = list(filter(lambda name: name != "pp", world_mesh.mesh_dim_names))
        spmd_mesh = world_mesh[spmd_mesh_dims] if len(spmd_mesh_dims) else None
    else:
        spmd_mesh = world_mesh
        info = {"global_rank": torch.distributed.distributed_c10d.get_rank(), "seed": seed}
        logger.debug(f"Enabling determinism: {info}")

    # The native RNGs and python RNG may not be important, except for the 1-D PP case, but we seed them for consistency
    torch.manual_seed(seed)
    # PYTHONHASHSEED can be a decimal number in the range [0, 2**32 - 1]
    os.environ["PYTHONHASHSEED"] = str(seed % 2**32)

    # As long as we are not in the 1-D (PP-only) case, we will have a seed to use for all ranks of the SPMD mesh.
    # IF PP is also used, this seed is unique per PP rank.
    if spmd_mesh and spmd_mesh.get_coordinate() is not None:
        torch.distributed.tensor._random.manual_seed(seed, spmd_mesh)


def expand_tensor_dims(tensor: torch.Tensor, ndim: int) -> torch.Tensor:
    assert len(tensor.shape) <= ndim
    return tensor.reshape(tensor.shape + (1,) * (ndim - len(tensor.shape)))


def get_device_info():
    from torch._utils import _get_available_device_type, _get_device_module

    device_type = _get_available_device_type()
    if device_type is None:
        device_type = "cuda"
    device_module = _get_device_module(device_type)
    return device_type, device_module


def get_dtype_from_string(dtype: str):
    return _STRING_TO_DTYPE[dtype]


def get_string_from_dtype(dtype: torch.dtype):
    return _DTYPE_TO_STRING[dtype]


def get_submodule_by_name(model: torch.nn.Module, name: str) -> Union[torch.nn.Module, List[torch.nn.Module]]:
    assert name.count("*") <= 1, "Wildcard '*' can only be used once in the name"
    return _find_submodule_by_name(model, name)


def get_unwrapped_model_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # Remove _orig_mod occurrences from the state dict keys
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


def is_compiled_module(module) -> bool:
    return isinstance(module, torch._dynamo.eval_frame.OptimizedModule)


def set_requires_grad(models: Union[torch.nn.Module, List[torch.nn.Module]], value: bool) -> None:
    if isinstance(models, torch.nn.Module):
        models = [models]
    for model in models:
        if model is not None:
            model.requires_grad_(value)


def synchronize_device() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()


def unwrap_module(module):
    """Unwraps a module if it was compiled with torch.compile()"""
    return module._orig_mod if is_compiled_module(module) else module


def _find_submodule_by_name(model: torch.nn.Module, name: str) -> Union[torch.nn.Module, List[torch.nn.Module]]:
    if name == "":
        return model
    first_atom, remaining_name = name.split(".", 1) if "." in name else (name, "")
    if first_atom == "*":
        # Wildcard '*' can only be used once in the name
        assert isinstance(model, torch.nn.ModuleList), "Wildcard '*' can only be used with ModuleList"
        submodules = []
        for submodule in model:
            subsubmodules = _find_submodule_by_name(submodule, remaining_name)
            if not isinstance(subsubmodules, list):
                subsubmodules = [subsubmodules]
            submodules.extend(subsubmodules)
        return submodules
    else:
        if hasattr(model, first_atom):
            submodule = getattr(model, first_atom)
            return _find_submodule_by_name(submodule, remaining_name)
        else:
            raise ValueError(f"'{first_atom}' is not a submodule of '{model.__class__.__name__}'")


# TODO(aryan): remove everything below this after next torch release
def _get_total_norm(
    tensors: Union[torch.Tensor, List[torch.Tensor]],
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: Optional[bool] = None,
) -> torch.Tensor:
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    else:
        tensors = list(tensors)
    norm_type = float(norm_type)
    if len(tensors) == 0:
        return torch.tensor(0.0)
    first_device = tensors[0].device
    grouped_tensors: dict[tuple[torch.device, torch.dtype], tuple[list[list[torch.Tensor]], list[int]]] = (
        _group_tensors_by_device_and_dtype(
            [tensors]  # type: ignore[list-item]
        )
    )  # type: ignore[assignment]

    norms: List[torch.Tensor] = []
    for (device, _), ([device_tensors], _) in grouped_tensors.items():
        if (foreach is None and _has_foreach_support(device_tensors, device)) or (
            foreach and _device_has_foreach_support(device)
        ):
            norms.extend(torch._foreach_norm(device_tensors, norm_type))
        elif foreach:
            raise RuntimeError(f"foreach=True was passed, but can't use the foreach API on {device.type} tensors")
        else:
            norms.extend([torch.linalg.vector_norm(g, norm_type) for g in device_tensors])

    total_norm = torch.linalg.vector_norm(torch.stack([norm.to(first_device) for norm in norms]), norm_type)

    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from "
            "`parameters` is non-finite, so it cannot be clipped. To disable "
            "this error and scale the gradients by the non-finite norm anyway, "
            "set `error_if_nonfinite=False`"
        )
    return total_norm


@torch.no_grad()
def _clip_grads_with_norm_(
    parameters: Union[torch.Tensor, List[torch.Tensor]],
    max_norm: float,
    total_norm: torch.Tensor,
    foreach: Optional[bool] = None,
) -> None:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    if len(grads) == 0:
        return
    grouped_grads: dict[Tuple[torch.device, torch.dtype], Tuple[List[List[torch.Tensor]], List[int]]] = (
        _group_tensors_by_device_and_dtype([grads])
    )  # type: ignore[assignment]

    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for (device, _), ([device_grads], _) in grouped_grads.items():
        if (foreach is None and _has_foreach_support(device_grads, device)) or (
            foreach and _device_has_foreach_support(device)
        ):
            torch._foreach_mul_(device_grads, clip_coef_clamped.to(device))
        elif foreach:
            raise RuntimeError(f"foreach=True was passed, but can't use the foreach API on {device.type} tensors")
        else:
            clip_coef_clamped_device = clip_coef_clamped.to(device)
            for g in device_grads:
                g.mul_(clip_coef_clamped_device)


def _get_foreach_kernels_supported_devices() -> list[str]:
    r"""Return the device type list that supports foreach kernels."""
    return ["cuda", "xpu", torch._C._get_privateuse1_backend_name()]


@torch.no_grad()
def _group_tensors_by_device_and_dtype(
    tensorlistlist: List[List[Optional[torch.Tensor]]],
    with_indices: bool = False,
) -> dict[tuple[torch.device, torch.dtype], tuple[List[List[Optional[torch.Tensor]]], List[int]]]:
    return torch._C._group_tensors_by_device_and_dtype(tensorlistlist, with_indices)


def _device_has_foreach_support(device: torch.device) -> bool:
    return device.type in (_get_foreach_kernels_supported_devices() + ["cpu"]) and not torch.jit.is_scripting()


def _has_foreach_support(tensors: List[torch.Tensor], device: torch.device) -> bool:
    return _device_has_foreach_support(device) and all(t is None or type(t) in [torch.Tensor] for t in tensors)
