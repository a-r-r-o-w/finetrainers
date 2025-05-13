import contextlib
import functools
import inspect
from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch

# Since we will be patching the `scaled_dot_product_attention` function with `attention_dispatch` to take
# control for dispatching to different attention providers, we need to import the original function
# to be able to use it and not go into infinite recursion when the dispatcher calls `scaled_dot_product_attention`.
import torch.autograd
from diffusers.utils.import_utils import OptionalDependencyNotAvailable
from torch.nn.functional import scaled_dot_product_attention as native_sdpa

from finetrainers.constants import FINETRAINERS_ATTN_CHECKS, FINETRAINERS_ATTN_PROVIDER
from finetrainers.logging import get_logger
from finetrainers.utils.import_utils import (
    is_flash_attn_available,
    is_flash_attn_version,
    is_sageattention_available,
    is_sageattention_version,
    is_torch_version,
    is_xformers_available,
    is_xformers_version,
)


if is_flash_attn_available():
    if is_flash_attn_version("<", "2.6.3"):
        raise OptionalDependencyNotAvailable(
            "The `flash-attn` library version is too old. Please update it to at least 2.6.3."
        )

    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.flash_attn_interface import _flash_attn_backward, _flash_attn_forward
else:
    flash_attn_func = None
    flash_attn_varlen_func = None
    _flash_attn_forward = None
    _flash_attn_backward = None


if is_sageattention_available():
    if is_sageattention_version("<", "2.1.1"):
        raise OptionalDependencyNotAvailable(
            "The `sageattention` library version is too old. Please update it to at least 2.1.1."
        )

    from sageattention import (
        sageattn,
        sageattn_qk_int8_pv_fp8_cuda,
        sageattn_qk_int8_pv_fp8_cuda_sm90,
        sageattn_qk_int8_pv_fp16_cuda,
        sageattn_qk_int8_pv_fp16_triton,
        sageattn_varlen,
    )
else:
    sageattn = None
    sageattn_qk_int8_pv_fp16_cuda = None
    sageattn_qk_int8_pv_fp16_triton = None
    sageattn_qk_int8_pv_fp8_cuda = None
    sageattn_qk_int8_pv_fp8_cuda_sm90 = None
    sageattn_varlen = None


if is_torch_version(">=", "2.5.0"):
    import torch.nn.attention.flex_attention as flex_attention


if is_torch_version(">=", "2.6.0"):
    from torch.distributed.tensor.experimental._attention import (
        _AttentionOp,
        _cp_options,
        _templated_ring_attention,
        _templated_ring_attention_backward,
        set_rotate_method,
    )
else:
    _cp_options = None
    _templated_ring_attention = None
    set_rotate_method = None

    class _AttentionOp:
        def __init__(self, *args, **kwargs):
            raise OptionalDependencyNotAvailable(
                "The `torch.distributed.tensor.experimental._attention` module is not available. Please update PyTorch to at least 2.6.0."
            )


if is_xformers_available():
    if is_xformers_version("<", "0.0.29"):
        raise OptionalDependencyNotAvailable(
            "The `xformers` library version is too old. Please update it to at least 0.0.29."
        )

    import xformers.ops as xops
else:
    xops = None


logger = get_logger()

_SAGE_ATTENTION_PV_ACCUM_DTYPE = Literal["fp32", "fp32+fp32"]
_SAGE_ATTENTION_QK_QUANT_GRAN = Literal["per_thread", "per_warp"]
_SAGE_ATTENTION_QUANTIZATION_BACKEND = Literal["cuda", "triton"]


# ===== Custom operator implementations/wrappers =====


def _finetrainers_scaled_dot_product_efficient_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Optional[torch.Tensor] = None,
    compute_log_sumexp: bool = False,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Wrapper for https://github.com/pytorch/pytorch/blob/8904ba638726f8c9a5aff5977c4aa76c9d2edfa6/aten/src/ATen/native/native_functions.yaml#L14946
    # See: https://github.com/pytorch/pytorch/issues/152942
    seqlen_q = query.shape[-2]
    out, lse, philox_seed, philox_offset = torch.ops.aten._scaled_dot_product_efficient_attention(
        query=query,
        key=key,
        value=value,
        attn_bias=attn_bias,
        compute_log_sumexp=compute_log_sumexp,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )

    # LSE is aligned to the next nearest multiple of 32. This is a workaround to return the lse without alignment so that pytorch
    # ring attention does not error out with shape mismatch
    if compute_log_sumexp:
        assert lse.ndim == 3
        lse = lse[:, :, :seqlen_q]  # .contiguous()

    return out, lse, philox_seed, philox_offset


# aten::_scaled_dot_product_efficient_attention_backward(Tensor grad_out_, Tensor query, Tensor key, Tensor value, Tensor attn_bias, Tensor out, Tensor logsumexp, Tensor philox_seed, Tensor philox_offset, float dropout_p, bool[4] grad_input_mask, bool is_causal=False, *, float? scale=None) -> (Tensor, Tensor, Tensor, Tensor)
def _finetrainers_scaled_dot_product_efficient_attention_backward(
    grad_out_: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    philox_seed: torch.Tensor,
    philox_offset: torch.Tensor,
    dropout_p: float,
    grad_input_mask: List[bool],
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert len(grad_input_mask) == 4
    # https://github.com/pytorch/pytorch/blob/bb9fbb294af385057a72e5b1386cf40f86aadbec/aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h#L113
    kAlignLSE = 32

    logsumexp = torch.nn.functional.pad(
        logsumexp, (0, kAlignLSE - (logsumexp.shape[-1] % kAlignLSE)), value=float("inf")
    )

    grad_query, grad_key, grad_value, grad_attn_bias = torch.ops.aten._scaled_dot_product_efficient_attention_backward(
        grad_out_=grad_out_,
        query=query,
        key=key,
        value=value,
        attn_bias=attn_bias,
        out=out,
        logsumexp=logsumexp,
        philox_seed=philox_seed,
        philox_offset=philox_offset,
        dropout_p=dropout_p,
        grad_input_mask=grad_input_mask,
        is_causal=is_causal,
        scale=scale,
    )

    return grad_query, grad_key, grad_value, grad_attn_bias


# This function wraps the actual _flash_attn_forward call to return LSE at index 1 to be compatible with pytorch's native ring attention
def _finetrainers_flash_attn_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float = 0.0,
    scale: Optional[float] = None,
    is_causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    return_softmax: bool = False,
):
    query, key, value = (
        x.permute(0, 2, 1, 3).contiguous() for x in (query, key, value)
    )  # [B, N, S, D] -> [B, S, N, D]
    out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_forward(
        query, key, value, dropout_p, scale, is_causal, window_size, softcap, alibi_slopes, return_softmax
    )
    out = out.permute(0, 2, 1, 3).contiguous()  # [B, S, N, D] -> [B, N, S, D]
    return out, softmax_lse, q, k, v, out_padded, S_dmask, rng_state


# This function wraps the actual _flash_attn_backward call as the counterpart of the _finetrainers_flash_attn_forward function
def _finetrainers_flash_attn_backward(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,  # Needs a different names than the one used in flash-attn because _templated_ring_attention_backward assumes name is logsumexp
    dropout_p: float,
    scale: Optional[float] = None,
    is_causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    rng_state: Optional[torch.Tensor] = None,
    _permute_outputs: bool = True,
):
    dq, dk, dv = torch.empty_like(query), torch.empty_like(key), torch.empty_like(value)
    grad_out = grad_out.permute(0, 2, 1, 3).contiguous()  # [B, N, S, D] -> [B, S, N, D]

    dq, dk, dv, softmax_d = _flash_attn_backward(
        grad_out,
        query,
        key,
        value,
        out,
        logsumexp,
        dq,
        dk,
        dv,
        dropout_p,
        scale,
        is_causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        rng_state,
    )

    # Head dimension may have been padded
    dq = dq[..., : grad_out.shape[-1]]
    dk = dk[..., : grad_out.shape[-1]]
    dv = dv[..., : grad_out.shape[-1]]

    if _permute_outputs:
        dq, dk, dv = (x.permute(0, 2, 1, 3).contiguous() for x in (dq, dk, dv))  # [B, S, N, D] -> [B, N, S, D]
    return dq, dk, dv


# ===== Attention provider =====


class AttentionProvider(str, Enum):
    # EAGER = "eager"

    # `flash-attn`
    FLASH = "flash"
    FLASH_VARLEN = "flash_varlen"

    # PyTorch native
    FLEX = "flex"
    NATIVE = "native"
    _NATIVE_CUDNN = "_native_cudnn"
    _NATIVE_EFFICIENT = "_native_efficient"
    _NATIVE_FLASH = "_native_flash"
    _NATIVE_MATH = "_native_math"

    # `sageattention`
    SAGE = "sage"
    SAGE_VARLEN = "sage_varlen"
    _SAGE_QK_INT8_PV_FP8_CUDA = "_sage_qk_int8_pv_fp8_cuda"
    _SAGE_QK_INT8_PV_FP8_CUDA_SM90 = "_sage_qk_int8_pv_fp8_cuda_sm90"
    _SAGE_QK_INT8_PV_FP16_CUDA = "_sage_qk_int8_pv_fp16_cuda"
    _SAGE_QK_INT8_PV_FP16_TRITON = "_sage_qk_int8_pv_fp16_triton"
    # TODO: let's not add support for Sparge Attention now because it requires tuning per model
    # We can look into supporting something "autotune"-ing in the future
    # SPARGE = "sparge"

    # `xformers`
    XFORMERS = "xformers"


class _AttentionProviderRegistry:
    _providers = {}
    _constraints = {}
    _supports_cp = {}
    _supported_arg_names = {}

    _active_provider = AttentionProvider(FINETRAINERS_ATTN_PROVIDER)
    _checks_enabled = FINETRAINERS_ATTN_CHECKS

    # Context parallel attributes
    _mesh: torch.distributed.device_mesh.DeviceMesh = None
    _convert_to_fp32: bool = None
    _rotate_method: Literal["allgather", "alltoall"] = None

    @classmethod
    def register(
        cls, provider: AttentionProvider, constraints: Optional[List[Callable]] = None, supports_cp: bool = False
    ):
        logger.debug(f"Registering attention provider: {provider}")

        def decorator(func):
            cls._providers[provider] = func
            cls._constraints[provider] = constraints or []
            cls._supports_cp[provider] = supports_cp
            cls._supported_arg_names[provider] = set(inspect.signature(func).parameters.keys())
            return func

        return decorator

    @classmethod
    def get_active_provider(cls):
        return cls._active_provider, cls._providers[cls._active_provider]

    @classmethod
    def list_providers(cls):
        return list(cls._providers.keys())

    @classmethod
    def supports_context_parallel(cls, provider: AttentionProvider):
        if provider not in cls._providers:
            raise ValueError(f"Provider {provider} is not registered.")
        return cls._supports_cp.get(provider, False)

    @classmethod
    def context_parallel_enabled(cls):
        return cls._mesh is not None

    @classmethod
    def _set_context_parallel(
        cls,
        mesh: torch.distributed.device_mesh.DeviceMesh = None,
        convert_to_fp32: bool = None,
        rotate_method: str = None,
        *,
        reset: bool = False,
    ):
        if reset:
            mesh = convert_to_fp32 = rotate_method = None
        cls._mesh = mesh
        cls._convert_to_fp32 = convert_to_fp32
        cls._rotate_method = rotate_method

    @classmethod
    def _raise_cp_error_if_mesh_not_set(cls):
        if cls._mesh is None:
            raise ValueError(
                "`_AttentionProviderRegistry._mesh` is None. It must be set before calling context parallel attention methods."
            )


@contextlib.contextmanager
def attention_provider(
    provider: AttentionProvider = AttentionProvider.NATIVE,
    *,
    mesh: Optional[torch.distributed.device_mesh.DeviceMesh] = None,
    convert_to_fp32: bool = True,
    rotate_method: str = "allgather",
):
    """Context manager to set the active attention provider and possibly enable context parallelism."""

    if provider not in _AttentionProviderRegistry._providers:
        raise ValueError(f"Provider {provider} is not registered.")
    if mesh is not None and not _AttentionProviderRegistry.supports_context_parallel(provider):
        raise ValueError(f"Provider {provider} does not support context parallelism.")

    old_provider = _AttentionProviderRegistry._active_provider
    _AttentionProviderRegistry._active_provider = provider

    _AttentionProviderRegistry._mesh = mesh
    _AttentionProviderRegistry._convert_to_fp32 = convert_to_fp32
    _AttentionProviderRegistry._rotate_method = rotate_method
    if mesh is not None:
        _convert_to_f32 = _cp_options.convert_to_f32
        _enable_load_balance = _cp_options.enable_load_balance
        _rotate_method = _cp_options.rotate_method

    try:
        yield
    finally:
        _AttentionProviderRegistry._active_provider = old_provider

        _AttentionProviderRegistry._mesh = None
        _AttentionProviderRegistry._convert_to_fp32 = None
        _AttentionProviderRegistry._rotate_method = None
        if mesh is not None:
            _cp_options.convert_to_f32 = _convert_to_f32
            _cp_options.enable_load_balance = _enable_load_balance
            _cp_options.rotate_method = _rotate_method


def attention_dispatch(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    attention_kwargs: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    attention_kwargs = attention_kwargs or {}
    provider_name, provider_fn = _AttentionProviderRegistry.get_active_provider()
    kwargs = {
        "query": query,
        "key": key,
        "value": value,
        "attn_mask": attn_mask,
        "dropout_p": dropout_p,
        "is_causal": is_causal,
        "scale": scale,
        "enable_gqa": enable_gqa,
        **attention_kwargs,
    }

    if _AttentionProviderRegistry._checks_enabled:
        removed_kwargs = set(kwargs) - set(_AttentionProviderRegistry._supported_arg_names[provider_name])
        if removed_kwargs:
            log_freq = 512
            msg = (
                f"Removing unsupported arguments for attention provider {provider_name}: {removed_kwargs}. This "
                f"message will be logged every {log_freq} calls."
            )
            logger.log_freq("WARNING", "REMOVING_ATTN_UNSUPPORTED_KWARGS", msg, log_freq)
        for check in _AttentionProviderRegistry._constraints.get(provider_name):
            check(**kwargs)

    kwargs = {k: v for k, v in kwargs.items() if k in _AttentionProviderRegistry._supported_arg_names[provider_name]}

    if _AttentionProviderRegistry.context_parallel_enabled():
        _set_context_parallel_options(**kwargs)

    return provider_fn(**kwargs)


# ===== Helper functions =====


# @torch.compiler.assume_constant_result
def _set_context_parallel_options(is_causal: bool, **kwargs):
    _cp_options.enable_load_balance = is_causal
    _cp_options.convert_to_f32 = _AttentionProviderRegistry._convert_to_fp32
    set_rotate_method(_AttentionProviderRegistry._rotate_method)


def _check_attn_mask_is_none(attn_mask: Optional[torch.Tensor], **kwargs) -> None:
    if attn_mask is not None:
        raise ValueError("Attention mask must be None for this provider.")


def _check_attn_mask_or_causal(attn_mask: Optional[torch.Tensor], is_causal: bool, **kwargs) -> None:
    if attn_mask is not None and is_causal:
        raise ValueError("`is_causal` cannot be True when `attn_mask` is not None.")


def _check_device(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs) -> None:
    if query.device != key.device or query.device != value.device:
        raise ValueError("Query, key, and value must be on the same device.")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError("Query, key, and value must have the same dtype.")


def _check_device_cuda(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs) -> None:
    _check_device(query, key, value)
    if query.device.type != "cuda":
        raise ValueError("Query, key, and value must be on a CUDA device.")


def _check_device_cuda_atleast_smXY(major: int, minor: int) -> Callable:
    def check_device_cuda(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs) -> None:
        _check_device_cuda(query, key, value)
        if torch.cuda.get_device_capability(query.device) < (major, minor):
            raise ValueError(
                f"Query, key, and value must be on a CUDA device with compute capability >= {major}.{minor}."
            )

    return check_device_cuda


def _check_qkv_dtype_match(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs) -> None:
    if query.dtype != key.dtype:
        raise ValueError("Query and key must have the same dtype.")
    if query.dtype != value.dtype:
        raise ValueError("Query and value must have the same dtype.")


def _check_qkv_dtype_bf16_or_fp16(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs) -> None:
    _check_qkv_dtype_match(query, key, value)
    if query.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("Query, key, and value must be either bfloat16 or float16.")


def _check_shape(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> None:
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("Query and key must have the same last dimension.")
    if query.shape[-2] != value.shape[-2]:
        raise ValueError("Query and value must have the same second to last dimension.")
    if attn_mask is not None and attn_mask.shape[-1] != key.shape[-2]:
        raise ValueError("Attention mask must match the key's second to last dimension.")


def _prepare_for_flash_attn_or_sage_varlen(
    batch_size: int,
    seq_len_q: int,
    seq_len_kv: int,
    attn_mask: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> None:
    seqlens_q = torch.full((batch_size,), seq_len_q, dtype=torch.int32, device=device)
    if attn_mask is None:
        seqlens_k = torch.full((batch_size,), seq_len_kv, dtype=torch.int32, device=device)
    else:
        seqlens_k = attn_mask.sum(dim=1, dtype=torch.int32)
    cu_seqlens_q = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu_seqlens_q[1:] = torch.cumsum(seqlens_q, dim=0)
    cu_seqlens_k[1:] = torch.cumsum(seqlens_k, dim=0)
    max_seqlen_q = seqlens_q.max().item()
    max_seqlen_k = seqlens_k.max().item()
    return (seqlens_q, seqlens_k), (cu_seqlens_q, cu_seqlens_k), (max_seqlen_q, max_seqlen_k)


def _normalize_attn_mask(attn_mask: torch.Tensor, batch_size: int, seq_len_k: int) -> torch.Tensor:
    """
    Normalize an attention mask to shape [batch_size, seq_len_k] (bool) suitable for inferring seqlens_k in
    FlashAttention/Sage varlen.

    Supports 1D to 4D shapes and common broadcasting patterns.
    """
    if attn_mask.dtype != torch.bool:
        raise ValueError(f"Attention mask must be of type bool, got {attn_mask.dtype}.")

    if attn_mask.ndim == 1:
        # [seq_len_k] -> broadcast across batch
        attn_mask = attn_mask.unsqueeze(0).expand(batch_size, seq_len_k)

    elif attn_mask.ndim == 2:
        # [batch_size, seq_len_k]. Maybe broadcast across batch
        if attn_mask.size(0) not in [1, batch_size]:
            raise ValueError(
                f"attn_mask.shape[0] ({attn_mask.shape[0]}) must be 1 or {batch_size} for 2D attention mask."
            )
        attn_mask = attn_mask.expand(batch_size, seq_len_k)

    elif attn_mask.ndim == 3:
        # [batch_size, seq_len_q, seq_len_k] -> reduce over query dimension
        if attn_mask.size(0) not in [1, batch_size]:
            raise ValueError(
                f"attn_mask.shape[0] ({attn_mask.shape[0]}) must be 1 or {batch_size} for 3D attention mask."
            )
        attn_mask = attn_mask.any(dim=1)
        attn_mask = attn_mask.expand(batch_size, seq_len_k)

    elif attn_mask.ndim == 4:
        # [batch_size, num_heads, seq_len_q, seq_len_k] or broadcastable versions
        if attn_mask.size(0) not in [1, batch_size]:
            raise ValueError(
                f"attn_mask.shape[0] ({attn_mask.shape[0]}) must be 1 or {batch_size} for 4D attention mask."
            )
        attn_mask = attn_mask.expand(batch_size, -1, -1, seq_len_k)  # [B, H, Q, K]
        attn_mask = attn_mask.any(dim=(1, 2))  # [B, K]

    else:
        raise ValueError(f"Unsupported attention mask shape: {attn_mask.shape}")

    if attn_mask.shape != (batch_size, seq_len_k):
        raise ValueError(
            f"Normalized attention mask shape mismatch: got {attn_mask.shape}, expected ({batch_size}, {seq_len_k})"
        )

    return attn_mask


def _flex_attention_causal_mask_mod(batch_idx, head_idx, q_idx, kv_idx):
    return q_idx >= kv_idx


# ===== Attention provider implementations =====


# Adapted from: https://github.com/Dao-AILab/flash-attention/blob/fd2fc9d85c8e54e5c20436465bca709bc1a6c5a1/flash_attn/flash_attn_interface.py#L807
class _flash_attn_flash_attention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dropout_p: float = 0.0,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Tuple[int, int] = (-1, -1),
        softcap: float = 0.0,
        alibi_slopes: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        return_softmax: bool = False,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic

        out, lse, q, k, v, out_padded, S_dmask, rng_state = _finetrainers_flash_attn_forward(
            query=q,
            key=k,
            value=v,
            dropout_p=dropout_p,
            scale=softmax_scale,
            is_causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax,
        )

        ctx.save_for_backward(q, k, v, out_padded, lse, rng_state)

        return (out, lse) if return_softmax else out

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_out: torch.Tensor,
        *args: torch.Tensor,
    ):
        q, k, v, out, lse, rng_state = ctx.saved_tensors

        grad_query, grad_key, grad_value = _finetrainers_flash_attn_backward(
            grad_out=grad_out,
            query=q,
            key=k,
            value=v,
            out=out,
            logsumexp=lse,
            dropout_p=ctx.dropout_p,
            scale=ctx.softmax_scale,
            is_causal=ctx.causal,
            window_size=ctx.window_size,
            softcap=ctx.softcap,
            alibi_slopes=ctx.alibi_slopes,
            deterministic=ctx.deterministic,
            rng_state=rng_state,
        )

        return grad_query, grad_key, grad_value, None, None, None, None, None, None, None, None


# Adapted from: https://github.com/Dao-AILab/flash-attention/blob/fd2fc9d85c8e54e5c20436465bca709bc1a6c5a1/flash_attn/flash_attn_interface.py#L807
class _native_ring_flash_attn_flash_attention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dropout_p: float = 0.0,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Tuple[int, int] = (-1, -1),
        softcap: float = 0.0,
        alibi_slopes: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        return_softmax: bool = False,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        # For ring flash attention using the flash-attn repo, we want the LSE but flash-attn only supports it if dropout_p > 0
        dropout_p = dropout_p if dropout_p > 0 else 1e-30

        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic

        out, lse, q, k, v, out_padded, S_dmask, rng_state = _templated_ring_attention(
            mesh=_AttentionProviderRegistry._mesh,
            seq_dim=2,
            op=_finetrainers_flash_attn_forward,
            query=q,
            key=k,
            value=v,
            dropout_p=dropout_p,
            scale=softmax_scale,
            is_causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=True,
        )

        ctx.save_for_backward(q, k, v, out_padded, lse, rng_state)

        return (out, lse) if return_softmax else out

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_out: torch.Tensor,
        *args: torch.Tensor,
    ):
        q, k, v, out, lse, rng_state = ctx.saved_tensors
        lse = lse.permute(0, 2, 1).contiguous()  # [B, N, S] -> [B, S, N]

        grad_query, grad_key, grad_value = _templated_ring_attention_backward(
            mesh=_AttentionProviderRegistry._mesh,
            # This needs to be 1 because q, k, v, out_padded returned from forward are BSND instead of BNSD
            # The grad_out permutation is handled in _finetrainers_flash_attn_backward, and the outputs from that are expected to have
            # shape BSND instead of BNSD (requirement of _templated_ring_attention_backward), so we need to set seq_dim=1 and permute the
            # returned outputs
            seq_dim=1,
            op=functools.partial(_finetrainers_flash_attn_backward, _permute_outputs=False),
            grad_out=grad_out,
            grad_out_name="grad_out",
            query=q,
            key=k,
            value=v,
            out=out,
            logsumexp=lse,
            dropout_p=ctx.dropout_p,
            scale=ctx.softmax_scale,
            is_causal=ctx.causal,
            window_size=ctx.window_size,
            softcap=ctx.softcap,
            alibi_slopes=ctx.alibi_slopes,
            deterministic=ctx.deterministic,
            rng_state=rng_state,
        )
        grad_query, grad_key, grad_value = (
            x.permute(0, 2, 1, 3).contiguous() for x in (grad_query, grad_key, grad_value)
        )

        return grad_query, grad_key, grad_value, None, None, None, None, None, None, None, None


@_AttentionProviderRegistry.register(
    AttentionProvider.FLASH,
    constraints=[_check_attn_mask_is_none, _check_device, _check_qkv_dtype_bf16_or_fp16, _check_shape],
    supports_cp=True,
)
def flash_attn_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float = 0.0,
    scale: Optional[float] = None,
    is_causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_lse: bool = False,
) -> torch.Tensor:
    dispatch_fn = (
        _native_ring_flash_attn_flash_attention
        if _AttentionProviderRegistry.context_parallel_enabled()
        else _flash_attn_flash_attention
    )
    return dispatch_fn.apply(
        query, key, value, dropout_p, scale, is_causal, window_size, softcap, alibi_slopes, deterministic, return_lse
    )


@_AttentionProviderRegistry.register(
    AttentionProvider.FLASH_VARLEN,
    constraints=[_check_device, _check_qkv_dtype_bf16_or_fp16, _check_shape],
    supports_cp=False,
)
def _flash_varlen_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    dropout_p: float = 0.0,
    scale: Optional[float] = None,
    is_causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    batch_size, _, seq_len_q, _ = query.shape
    _, _, seq_len_kv, _ = key.shape

    if attn_mask is not None:
        attn_mask = _normalize_attn_mask(attn_mask, batch_size, seq_len_kv)

    if any(x is None for x in (cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)):
        (_, seqlens_k), (cu_seqlens_q, cu_seqlens_k), (max_seqlen_q, max_seqlen_k) = (
            _prepare_for_flash_attn_or_sage_varlen(
                batch_size, seq_len_q, seq_len_kv, attn_mask=attn_mask, device=query.device
            )
        )
    else:
        seqlens_k = torch.full((batch_size,), max_seqlen_k, dtype=torch.int32, device=query.device)
        cu_seqlens_q = cu_seqlens_q.to(dtype=torch.int32, device=query.device)
        cu_seqlens_k = cu_seqlens_k.to(dtype=torch.int32, device=query.device)

    query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))

    key_valid, value_valid = [], []
    for b in range(batch_size):
        valid_len = seqlens_k[b]
        key_valid.append(key[b, :valid_len])
        value_valid.append(value[b, :valid_len])

    query_packed = query.flatten(0, 1)
    key_packed = torch.cat(key_valid, dim=0)
    value_packed = torch.cat(value_valid, dim=0)

    if _AttentionProviderRegistry.context_parallel_enabled():
        return_attn_probs = True

    out = flash_attn_varlen_func(
        q=query_packed,
        k=key_packed,
        v=value_packed,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=dropout_p,
        softmax_scale=scale,
        causal=is_causal,
        window_size=window_size,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
    )

    rest = None
    if return_attn_probs:
        out, *rest = out
    out = out.unflatten(0, (batch_size, -1)).permute(0, 2, 1, 3)  # .contiguous()
    if return_attn_probs:
        return out, *rest[:1]
    return out


@_AttentionProviderRegistry.register(
    AttentionProvider.FLEX,
    constraints=[_check_attn_mask_or_causal, _check_device, _check_shape],
    supports_cp=False,
)
def _native_flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[Union[torch.Tensor, "flex_attention.BlockMask"]] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
    kernel_options: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    # TODO: should we LRU cache the block mask creation?
    score_mod = None
    block_mask = None
    batch_size, num_heads, seq_len_q, _ = query.shape
    _, _, seq_len_kv, _ = key.shape

    if attn_mask is None or isinstance(attn_mask, flex_attention.BlockMask):
        block_mask = attn_mask
    elif is_causal:
        block_mask = flex_attention.create_block_mask(
            _flex_attention_causal_mask_mod, None, None, seq_len_q, seq_len_kv, query.device
        )
    elif torch.is_tensor(attn_mask):
        if attn_mask.ndim == 2:
            attn_mask = attn_mask.view(attn_mask.size(0), 1, attn_mask.size(1), 1)

        attn_mask = attn_mask.expand(batch_size, num_heads, seq_len_q, seq_len_kv)

        if attn_mask.dtype == torch.bool:
            # TODO: this probably does not work but verify!
            def mask_mod(batch_idx, head_idx, q_idx, kv_idx):
                return attn_mask[batch_idx, head_idx, q_idx, kv_idx]

            block_mask = flex_attention.create_block_mask(
                mask_mod, batch_size, None, seq_len_q, seq_len_kv, query.device
            )
        else:

            def score_mod(score, batch_idx, head_idx, q_idx, kv_idx):
                return score + attn_mask[batch_idx, head_idx, q_idx, kv_idx]
    else:
        raise ValueError("Attention mask must be either None, a BlockMask, or a 2D/4D tensor.")

    return flex_attention.flex_attention(
        query=query,
        key=key,
        value=value,
        score_mod=score_mod,
        block_mask=block_mask,
        scale=scale,
        enable_gqa=enable_gqa,
        return_lse=return_lse,
        kernel_options=None,
    )


@_AttentionProviderRegistry.register(
    AttentionProvider.NATIVE,
    constraints=[_check_device, _check_shape],
    supports_cp=False,
)
def _native_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    return native_sdpa(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )


class _native_cudnn_attention(torch.autograd.Function):
    # https://github.com/pytorch/pytorch/blob/8904ba638726f8c9a5aff5977c4aa76c9d2edfa6/aten/src/ATen/native/native_functions.yaml#L14958
    # forward declaration:
    #   aten::_scaled_dot_product_cudnn_attention(Tensor query, Tensor key, Tensor value, Tensor? attn_bias, bool compute_log_sumexp, float dropout_p=0., bool is_causal=False, bool return_debug_mask=False, *, float? scale=None) -> (Tensor output, Tensor logsumexp, Tensor cum_seq_q, Tensor cum_seq_k, SymInt max_q, SymInt max_k, Tensor philox_seed, Tensor philox_offset, Tensor debug_attn_mask)
    # backward declaration:
    #   aten::_scaled_dot_product_cudnn_attention_backward(Tensor grad_out, Tensor query, Tensor key, Tensor value, Tensor out, Tensor logsumexp, Tensor philox_seed, Tensor philox_offset, Tensor attn_bias, Tensor cum_seq_q, Tensor cum_seq_k, SymInt max_q, SymInt max_k, float dropout_p, bool is_causal, *, float? scale=None) -> (Tensor, Tensor, Tensor)

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None,
        return_lse: bool = False,
    ):
        ctx.dropout_p = dropout_p
        ctx.is_causal = is_causal
        ctx.scale = scale
        ctx.attn_mask = attn_mask

        out, lse, cum_seq_q, cum_seq_k, max_q, max_k, philox_seed, philox_offset, debug_attn_mask = (
            torch.ops.aten._scaled_dot_product_cudnn_attention(
                query=query,
                key=key,
                value=value,
                attn_bias=attn_mask,
                compute_log_sumexp=True,
                dropout_p=dropout_p,
                is_causal=is_causal,
                return_debug_mask=False,
                scale=scale,
            )
        )

        ctx.max_q = max_q
        ctx.max_k = max_k
        ctx.save_for_backward(query, key, value, out, lse, cum_seq_q, cum_seq_k, philox_seed, philox_offset)

        return (out, lse) if return_lse else out

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_out: torch.Tensor,
        *args: torch.Tensor,
    ):
        query, key, value, out, lse, cum_seq_q, cum_seq_k, philox_seed, philox_offset = ctx.saved_tensors

        grad_query, grad_key, grad_value = torch.ops.aten._scaled_dot_product_cudnn_attention_backward(
            grad_out=grad_out,
            query=query,
            key=key,
            value=value,
            out=out,
            logsumexp=lse,
            philox_seed=philox_seed,
            philox_offset=philox_offset,
            attn_bias=ctx.attn_mask,
            cum_seq_q=cum_seq_q,
            cum_seq_k=cum_seq_k,
            max_q=ctx.max_q,
            max_k=ctx.max_k,
            dropout_p=ctx.dropout_p,
            is_causal=ctx.is_causal,
            scale=ctx.scale,
        )

        return grad_query, grad_key, grad_value, None, None, None, None, None


class _native_ring_native_cudnn_attention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None,
        return_lse: bool = False,
    ):
        _AttentionProviderRegistry._raise_cp_error_if_mesh_not_set()
        ctx.dropout_p = dropout_p
        ctx.is_causal = is_causal
        ctx.scale = scale
        ctx.attn_mask = attn_mask

        out, lse, cum_seq_q, cum_seq_k, max_q, max_k, philox_seed, philox_offset, debug_attn_mask = (
            _templated_ring_attention(
                mesh=_AttentionProviderRegistry._mesh,
                seq_dim=2,
                op=torch.ops.aten._scaled_dot_product_cudnn_attention,
                query=query,
                key=key,
                value=value,
                attn_bias=attn_mask,
                compute_log_sumexp=True,
                dropout_p=dropout_p,
                is_causal=is_causal,
                return_debug_mask=False,
                scale=scale,
            )
        )

        ctx.max_q = max_q
        ctx.max_k = max_k
        ctx.save_for_backward(query, key, value, out, lse, cum_seq_q, cum_seq_k, philox_seed, philox_offset)

        return (out, lse) if return_lse else out

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_out: torch.Tensor,
        *args: torch.Tensor,
    ):
        _AttentionProviderRegistry._raise_cp_error_if_mesh_not_set()
        query, key, value, out, lse, cum_seq_q, cum_seq_k, philox_seed, philox_offset = ctx.saved_tensors

        grad_query, grad_key, grad_value = _templated_ring_attention_backward(
            mesh=_AttentionProviderRegistry._mesh,
            seq_dim=2,
            op=torch.ops.aten._scaled_dot_product_cudnn_attention_backward,
            grad_out=grad_out,
            grad_out_name="grad_out",
            query=query,
            key=key,
            value=value,
            out=out,
            logsumexp=lse,
            philox_seed=philox_seed,
            philox_offset=philox_offset,
            attn_bias=ctx.attn_mask,
            cum_seq_q=cum_seq_q,
            cum_seq_k=cum_seq_k,
            max_q=ctx.max_q,
            max_k=ctx.max_k,
            dropout_p=ctx.dropout_p,
            is_causal=ctx.is_causal,
            scale=ctx.scale,
        )

        return grad_query, grad_key, grad_value, None, None, None, None, None


@_AttentionProviderRegistry.register(
    AttentionProvider._NATIVE_CUDNN,
    constraints=[_check_device, _check_qkv_dtype_bf16_or_fp16, _check_shape],
    supports_cp=True,
)
def native_cudnn_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    return_lse: bool = False,
) -> torch.Tensor:
    dispatch_fn = (
        _native_ring_native_cudnn_attention
        if _AttentionProviderRegistry.context_parallel_enabled()
        else _native_cudnn_attention
    )
    return dispatch_fn.apply(query, key, value, attn_mask, dropout_p, is_causal, scale, return_lse)


class _native_efficient_attention(torch.autograd.Function):
    # https://github.com/pytorch/pytorch/blob/8904ba638726f8c9a5aff5977c4aa76c9d2edfa6/aten/src/ATen/native/native_functions.yaml#L14946
    # forward declaration:
    #   aten::_scaled_dot_product_efficient_attention(Tensor query, Tensor key, Tensor value, Tensor? attn_bias, bool compute_log_sumexp, float dropout_p=0., bool is_causal=False, *, float? scale=None) -> (Tensor output, Tensor log_sumexp, Tensor philox_seed, Tensor philox_offset)
    # backward declaration:
    #   aten::_scaled_dot_product_efficient_attention_backward(Tensor grad_out_, Tensor query, Tensor key, Tensor value, Tensor attn_bias, Tensor out, Tensor logsumexp, Tensor philox_seed, Tensor philox_offset, float dropout_p, bool[4] grad_input_mask, bool is_causal=False, *, float? scale=None) -> (Tensor, Tensor, Tensor, Tensor)

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None,
        return_lse: bool = False,
    ):
        ctx.dropout_p = dropout_p
        ctx.is_causal = is_causal
        ctx.scale = scale
        ctx.attn_mask = attn_mask

        # NOTE: Uses finetrainers registered op because of LSE alignment issue. See the op registration for more details.
        out, lse, philox_seed, philox_offset = _finetrainers_scaled_dot_product_efficient_attention_forward(
            query=query,
            key=key,
            value=value,
            attn_bias=attn_mask,
            compute_log_sumexp=True,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )

        ctx.save_for_backward(query, key, value, out, lse, philox_seed, philox_offset)

        return (out, lse) if return_lse else out

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_out: torch.Tensor,
        *args: torch.Tensor,
    ):
        query, key, value, out, lse, philox_seed, philox_offset = ctx.saved_tensors

        # NOTE: Uses finetrainers registered op because of LSE alignment issue. See the op registration for more details.
        grad_query, grad_key, grad_value, grad_attn_bias = (
            _finetrainers_scaled_dot_product_efficient_attention_backward(
                grad_out_=grad_out,
                query=query,
                key=key,
                value=value,
                attn_bias=ctx.attn_mask,
                out=out,
                logsumexp=lse,
                philox_seed=philox_seed,
                philox_offset=philox_offset,
                dropout_p=ctx.dropout_p,
                grad_input_mask=[True, True, True, False],
                is_causal=ctx.is_causal,
                scale=ctx.scale,
            )
        )

        return grad_query, grad_key, grad_value, None, None, None, None, None


class _native_ring_native_efficient_attention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None,
        return_lse: bool = False,
    ):
        _AttentionProviderRegistry._raise_cp_error_if_mesh_not_set()
        ctx.dropout_p = dropout_p
        ctx.is_causal = is_causal
        ctx.scale = scale
        ctx.attn_mask = attn_mask

        # NOTE: Uses finetrainers registered op because of LSE alignment issue. See the op registration for more details.
        out, lse, philox_seed, philox_offset = _templated_ring_attention(
            mesh=_AttentionProviderRegistry._mesh,
            seq_dim=2,
            op=_finetrainers_scaled_dot_product_efficient_attention_forward,
            query=query,
            key=key,
            value=value,
            attn_bias=attn_mask,
            compute_log_sumexp=True,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )

        ctx.save_for_backward(query, key, value, out, lse, philox_seed, philox_offset)

        return (out, lse) if return_lse else out

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_out: torch.Tensor,
        *args: torch.Tensor,
    ):
        _AttentionProviderRegistry._raise_cp_error_if_mesh_not_set()
        query, key, value, out, lse, philox_seed, philox_offset = ctx.saved_tensors

        # NOTE: Uses finetrainers registered op because of LSE alignment issue. See the op registration for more details.
        grad_query, grad_key, grad_value, grad_attn_bias = _templated_ring_attention_backward(
            mesh=_AttentionProviderRegistry._mesh,
            seq_dim=2,
            op=_finetrainers_scaled_dot_product_efficient_attention_backward,
            grad_out=grad_out,
            grad_out_name="grad_out_",
            query=query,
            key=key,
            value=value,
            attn_bias=ctx.attn_mask,
            out=out,
            logsumexp=lse,
            philox_seed=philox_seed,
            philox_offset=philox_offset,
            dropout_p=ctx.dropout_p,
            grad_input_mask=[True, True, True, False],
            is_causal=ctx.is_causal,
            scale=ctx.scale,
        )

        return grad_query, grad_key, grad_value, None, None, None, None, None


@_AttentionProviderRegistry.register(
    AttentionProvider._NATIVE_EFFICIENT,
    constraints=[_check_device, _check_shape],
    supports_cp=True,
)
def native_efficient_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    dispatch_fn = (
        _native_ring_native_efficient_attention
        if _AttentionProviderRegistry.context_parallel_enabled()
        else _native_efficient_attention
    )
    return dispatch_fn.apply(query, key, value, attn_mask, dropout_p, is_causal, scale)


class _native_flash_attention(torch.autograd.Function):
    # https://github.com/pytorch/pytorch/blob/8904ba638726f8c9a5aff5977c4aa76c9d2edfa6/aten/src/ATen/native/native_functions.yaml#L14910
    # forward declaration:
    #   aten::_scaled_dot_product_flash_attention(Tensor query, Tensor key, Tensor value, float dropout_p=0., bool is_causal=False, bool return_debug_mask=False, *, float? scale=None) -> (Tensor output, Tensor logsumexp, Tensor cum_seq_q, Tensor cum_seq_k, SymInt max_q, SymInt max_k, Tensor philox_seed, Tensor philox_offset, Tensor debug_attn_mask)
    # backward declaration:
    #   aten::_scaled_dot_product_flash_attention_backward(Tensor grad_out, Tensor query, Tensor key, Tensor value, Tensor out, Tensor logsumexp, Tensor cum_seq_q, Tensor cum_seq_k, SymInt max_q, SymInt max_k, float dropout_p, bool is_causal, Tensor philox_seed, Tensor philox_offset, *, float? scale=None) -> (Tensor grad_query, Tensor grad_key, Tensor grad_value)

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None,
        return_lse: bool = False,
    ):
        ctx.dropout_p = dropout_p
        ctx.is_causal = is_causal
        ctx.scale = scale

        out, lse, cum_seq_q, cum_seq_k, max_q, max_k, philox_seed, philox_offset, debug_attn_mask = (
            torch.ops.aten._scaled_dot_product_flash_attention(
                query=query,
                key=key,
                value=value,
                dropout_p=dropout_p,
                is_causal=is_causal,
                return_debug_mask=False,
                scale=scale,
            )
        )

        ctx.max_q = max_q
        ctx.max_k = max_k
        ctx.save_for_backward(query, key, value, out, lse, cum_seq_q, cum_seq_k, philox_seed, philox_offset)

        return (out, lse) if return_lse else out

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_out: torch.Tensor,
        *args: torch.Tensor,
    ):
        query, key, value, out, lse, cum_seq_q, cum_seq_k, philox_seed, philox_offset = ctx.saved_tensors

        grad_query, grad_key, grad_value = torch.ops.aten._scaled_dot_product_flash_attention_backward(
            grad_out=grad_out,
            query=query,
            key=key,
            value=value,
            out=out,
            logsumexp=lse,
            cum_seq_q=cum_seq_q,
            cum_seq_k=cum_seq_k,
            max_q=ctx.max_q,
            max_k=ctx.max_k,
            dropout_p=ctx.dropout_p,
            is_causal=ctx.is_causal,
            philox_seed=philox_seed,
            philox_offset=philox_offset,
            scale=ctx.scale,
        )

        return grad_query, grad_key, grad_value, None, None, None, None


class _native_ring_native_flash_attention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None,
        return_lse: bool = False,
    ):
        _AttentionProviderRegistry._raise_cp_error_if_mesh_not_set()
        ctx.dropout_p = dropout_p
        ctx.is_causal = is_causal
        ctx.scale = scale

        out, lse, cum_seq_q, cum_seq_k, max_q, max_k, philox_seed, philox_offset, debug_attn_mask = (
            _templated_ring_attention(
                mesh=_AttentionProviderRegistry._mesh,
                seq_dim=2,
                op=torch.ops.aten._scaled_dot_product_flash_attention,
                query=query,
                key=key,
                value=value,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
            )
        )

        ctx.max_q = max_q
        ctx.max_k = max_k
        ctx.save_for_backward(query, key, value, out, lse, cum_seq_q, cum_seq_k, philox_seed, philox_offset)

        return (out, lse) if return_lse else out

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_out: torch.Tensor,
        *args: torch.Tensor,
    ):
        _AttentionProviderRegistry._raise_cp_error_if_mesh_not_set()
        query, key, value, out, lse, cum_seq_q, cum_seq_k, philox_seed, philox_offset = ctx.saved_tensors

        grad_query, grad_key, grad_value, *_ = _templated_ring_attention_backward(
            mesh=_AttentionProviderRegistry._mesh,
            seq_dim=2,
            op=torch.ops.aten._scaled_dot_product_flash_attention_backward,
            grad_out=grad_out,
            grad_out_name="grad_out",
            query=query,
            key=key,
            value=value,
            out=out,
            logsumexp=lse,
            dropout_p=ctx.dropout_p,
            is_causal=ctx.is_causal,
            scale=ctx.scale,
            cum_seq_q=cum_seq_q,
            cum_seq_k=cum_seq_k,
            max_q=ctx.max_q,
            max_k=ctx.max_k,
            philox_seed=philox_seed,
            philox_offset=philox_offset,
        )

        return grad_query, grad_key, grad_value, None, None, None, None


@_AttentionProviderRegistry.register(
    AttentionProvider._NATIVE_FLASH,
    constraints=[_check_device, _check_qkv_dtype_bf16_or_fp16, _check_shape],
    supports_cp=True,
)
def native_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    return_lse: bool = False,
) -> torch.Tensor:
    dispatch_fn = (
        _native_ring_native_flash_attention
        if _AttentionProviderRegistry.context_parallel_enabled()
        else _native_flash_attention
    )
    return dispatch_fn.apply(query, key, value, dropout_p, is_causal, scale, return_lse)


# class _native_math_attention(torch.autograd.Function):
#     # https://github.com/pytorch/pytorch/blob/8904ba638726f8c9a5aff5977c4aa76c9d2edfa6/aten/src/ATen/native/native_functions.yaml#L14901
#     # forward declaration:
#     #   aten::_scaled_dot_product_attention_math(Tensor query, Tensor key, Tensor value, Tensor? attn_mask=None, float dropout_p=0., bool is_causal=False, Tensor? dropout_mask=None, *, float? scale=None, bool enable_gqa=False) -> (Tensor, Tensor)
#     # backward declaration:
#     #   does not exist
#     @staticmethod
#     def forward(
#         ctx: torch.autograd.function.FunctionCtx,
#         query: torch.Tensor,
#         key: torch.Tensor,
#         value: torch.Tensor,
#         attn_mask: Optional[torch.Tensor] = None,
#         dropout_p: float = 0.0,
#         is_causal: bool = False,
#         dropout_mask: Optional[torch.Tensor] = None,
#         scale: Optional[float] = None,
#         enable_gqa: bool = False,
#         return_scores: bool = False,
#     ):
#         ctx.dropout_p = dropout_p
#         ctx.is_causal = is_causal
#         ctx.scale = scale
#         ctx.enable_gqa = enable_gqa

#         print(f"query.shape: {query.shape}")
#         with torch.enable_grad():
#             out, scores = torch.ops.aten._scaled_dot_product_attention_math(
#                 query=query,
#                 key=key,
#                 value=value,
#                 attn_mask=attn_mask,
#                 dropout_p=dropout_p,
#                 is_causal=is_causal,
#                 dropout_mask=dropout_mask,
#                 scale=scale,
#                 enable_gqa=enable_gqa,
#             )

#         ctx.save_for_backward(query, key, value, out)

#         return (out, scores) if return_scores else out

#     @staticmethod
#     def backward(
#         ctx: torch.autograd.function.FunctionCtx,
#         grad_out: torch.Tensor,
#     ):
#         raise NotImplementedError("Backward pass for native math attention is not implemented.")


@_AttentionProviderRegistry.register(
    AttentionProvider._NATIVE_MATH,
    constraints=[_check_device, _check_shape],
    supports_cp=False,
)
def native_math_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        return native_sdpa(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )


@_AttentionProviderRegistry.register(
    AttentionProvider.SAGE,
    constraints=[_check_device_cuda, _check_qkv_dtype_bf16_or_fp16, _check_shape],
    supports_cp=False,
)
def _sage_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
    scale: Optional[float] = None,
    return_lse: bool = False,
) -> torch.Tensor:
    if _AttentionProviderRegistry.context_parallel_enabled():
        return_lse = True

    kwargs = {
        "q": query,
        "k": key,
        "v": value,
        "tensor_layout": "HND",
        "is_causal": is_causal,
        "sm_scale": scale,
        "return_lse": return_lse,
    }
    out = sageattn(**kwargs)

    rest = None
    if return_lse:
        out, *rest = out
    if return_lse:
        return out, *rest[:1]
    return out


@_AttentionProviderRegistry.register(
    AttentionProvider.SAGE_VARLEN,
    constraints=[_check_device_cuda, _check_qkv_dtype_bf16_or_fp16, _check_shape],
)
def _sage_varlen_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
    smooth_k: bool = True,
    attn_mask: Optional[torch.Tensor] = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    batch_size, _, seq_len_q, _ = query.shape
    _, _, seq_len_kv, _ = key.shape

    if attn_mask is not None:
        attn_mask = _normalize_attn_mask(attn_mask, batch_size, seq_len_kv)

    if enable_gqa:
        # TODO
        pass

    if any(x is None for x in (cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)):
        (_, seqlens_k), (cu_seqlens_q, cu_seqlens_k), (max_seqlen_q, max_seqlen_k) = (
            _prepare_for_flash_attn_or_sage_varlen(
                batch_size, seq_len_q, seq_len_kv, attn_mask=attn_mask, device=query.device
            )
        )
    else:
        seqlens_k = torch.full((batch_size,), max_seqlen_k, dtype=torch.int32, device=query.device)
        cu_seqlens_q = cu_seqlens_q.to(dtype=torch.int32, device=query.device)
        cu_seqlens_k = cu_seqlens_k.to(dtype=torch.int32, device=query.device)

    query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))

    key_valid, value_valid = [], []
    for b in range(batch_size):
        valid_len = seqlens_k[b]
        key_valid.append(key[b, :valid_len])
        value_valid.append(value[b, :valid_len])

    query_packed = query.flatten(0, 1)
    key_packed = torch.cat(key_valid, dim=0)
    value_packed = torch.cat(value_valid, dim=0)

    out = sageattn_varlen(
        q=query_packed,
        k=key_packed,
        v=value_packed,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        is_causal=is_causal,
        sm_scale=scale,
        smooth_k=smooth_k,
    )
    out = out.unflatten(0, (batch_size, -1)).permute(0, 2, 1, 3)  # .contiguous()

    return out


@_AttentionProviderRegistry.register(
    AttentionProvider._SAGE_QK_INT8_PV_FP8_CUDA,
    constraints=[_check_device_cuda_atleast_smXY(9, 0), _check_shape],
    supports_cp=False,
)
def _sage_qk_int8_pv_fp8_cuda_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
    scale: Optional[float] = None,
    qk_quant_gran: _SAGE_ATTENTION_QK_QUANT_GRAN = "per_thread",
    pv_accum_dtype: _SAGE_ATTENTION_PV_ACCUM_DTYPE = "fp32+fp32",
    smooth_k: bool = True,
    smooth_v: bool = False,
    return_lse: bool = False,
) -> torch.Tensor:
    return sageattn_qk_int8_pv_fp8_cuda(
        q=query,
        k=key,
        v=value,
        tensor_layout="HND",
        is_causal=is_causal,
        qk_quant_gran=qk_quant_gran,
        sm_scale=scale,
        pv_accum_dtype=pv_accum_dtype,
        smooth_k=smooth_k,
        smooth_v=smooth_v,
        return_lse=return_lse,
    )


@_AttentionProviderRegistry.register(
    AttentionProvider._SAGE_QK_INT8_PV_FP8_CUDA_SM90,
    constraints=[_check_device_cuda_atleast_smXY(9, 0), _check_shape],
    supports_cp=False,
)
def _sage_qk_int8_pv_fp8_cuda_sm90_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
    scale: Optional[float] = None,
    qk_quant_gran: _SAGE_ATTENTION_QK_QUANT_GRAN = "per_thread",
    pv_accum_dtype: _SAGE_ATTENTION_PV_ACCUM_DTYPE = "fp32+fp32",
    smooth_k: bool = True,
    return_lse: bool = False,
) -> torch.Tensor:
    return sageattn_qk_int8_pv_fp8_cuda_sm90(
        q=query,
        k=key,
        v=value,
        tensor_layout="HND",
        is_causal=is_causal,
        qk_quant_gran=qk_quant_gran,
        sm_scale=scale,
        pv_accum_dtype=pv_accum_dtype,
        smooth_k=smooth_k,
        return_lse=return_lse,
    )


@_AttentionProviderRegistry.register(
    AttentionProvider._SAGE_QK_INT8_PV_FP16_CUDA,
    constraints=[_check_device_cuda_atleast_smXY(8, 0), _check_shape],
    supports_cp=False,
)
def _sage_qk_int8_pv_fp16_cuda_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
    scale: Optional[float] = None,
    qk_quant_gran: _SAGE_ATTENTION_QK_QUANT_GRAN = "per_thread",
    pv_accum_dtype: _SAGE_ATTENTION_PV_ACCUM_DTYPE = "fp32+fp32",
    smooth_k: bool = True,
    smooth_v: bool = False,
    return_lse: bool = False,
) -> torch.Tensor:
    return sageattn_qk_int8_pv_fp16_cuda(
        q=query,
        k=key,
        v=value,
        tensor_layout="HND",
        is_causal=is_causal,
        qk_quant_gran=qk_quant_gran,
        sm_scale=scale,
        pv_accum_dtype=pv_accum_dtype,
        smooth_k=smooth_k,
        smooth_v=smooth_v,
        return_lse=return_lse,
    )


@_AttentionProviderRegistry.register(
    AttentionProvider._SAGE_QK_INT8_PV_FP16_TRITON,
    constraints=[_check_device_cuda_atleast_smXY(8, 0), _check_shape],
    supports_cp=False,
)
def _sage_qk_int8_pv_fp16_triton_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
    scale: Optional[float] = None,
    quantization_backend: _SAGE_ATTENTION_QUANTIZATION_BACKEND = "triton",
    smooth_k: bool = True,
    return_lse: bool = False,
) -> torch.Tensor:
    return sageattn_qk_int8_pv_fp16_triton(
        q=query,
        k=key,
        v=value,
        tensor_layout="HND",
        quantization_backend=quantization_backend,
        is_causal=is_causal,
        sm_scale=scale,
        smooth_k=smooth_k,
        return_lse=return_lse,
    )


@_AttentionProviderRegistry.register(
    AttentionProvider.XFORMERS,
    constraints=[_check_attn_mask_or_causal, _check_device, _check_shape],
)
def _xformers_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    batch_size, num_heads_q, seq_len_q, _ = query.shape
    _, num_heads_kv, seq_len_kv, _ = key.shape

    # TODO: check if `contiguous` is really needed since it may cause unnecessary slowdowns
    if is_causal:
        attn_mask = xops.LowerTriangularMask()
    elif attn_mask is not None:
        if attn_mask.ndim == 2:
            attn_mask = attn_mask.view(attn_mask.size(0), 1, attn_mask.size(1), 1)
        elif attn_mask.ndim != 4:
            raise ValueError("Only 2D and 4D attention masks are supported for xformers attention.")
        attn_mask = attn_mask.expand(batch_size, num_heads_q, seq_len_q, seq_len_kv).type_as(query)

    # QKV need to be in [batch, seq_len, num_heads, head_dim] format for xformers
    # query, key, value = (x.permute(0, 2, 1, 3).contiguous() for x in (query, key, value))
    query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))

    if enable_gqa:
        if num_heads_q % num_heads_kv != 0:
            raise ValueError("Number of heads in query must be divisible by number of heads in key/value.")
        num_heads_per_group = num_heads_q // num_heads_kv
        query = query.unflatten(2, (num_heads_kv, -1))
        key = key.unflatten(2, (num_heads_kv, -1)).expand(-1, -1, -1, num_heads_per_group, -1)
        value = value.unflatten(2, (num_heads_kv, -1)).expand(-1, -1, -1, num_heads_per_group, -1)

    out = xops.memory_efficient_attention(query, key, value, attn_mask, dropout_p, scale)
    if enable_gqa:
        out = out.flatten(2, 3)

    out = out.permute(0, 2, 1, 3)  # .contiguous()
    return out
