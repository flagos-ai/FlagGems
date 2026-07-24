# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validation and canonical input preparation for Hopper TLE FA3."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from functools import lru_cache

import torch
import triton
import triton.language as tl

from flag_gems.utils import has_triton_tle

tle = None
if has_triton_tle(3, 6, 0):
    try:
        import triton.experimental.tle.language as tle
    except Exception:
        # Keep FA2 importable when a partially installed compiler advertises
        # TLE but cannot load it. Explicit FA3 requests report this below.
        tle = None


# Cap the public Split-KV hint so callers cannot create unbounded compile-time
# specializations or workspaces.
MAX_SPLIT_KV = 32

_TMA_ALLOCATOR_REGISTERED = False


@lru_cache(maxsize=None)
def _parameter_names(function) -> frozenset[str]:
    """Cache stable TLE primitive signatures used on every FA3 invocation."""

    try:
        return frozenset(inspect.signature(function).parameters)
    except (TypeError, ValueError):
        return frozenset()


@dataclass(frozen=True)
class NormalizedWindow:
    """Canonical causal/local-window state consumed by routing and kernels."""

    causal: bool
    local: bool
    left: int
    right: int


@dataclass(frozen=True)
class PreparedFA3Inputs:
    """Validated, canonical inputs shared by the scheduler and launcher.

    Optional public arguments needed only for API compatibility are rejected or
    consumed before this object is created.  Kernel ABI placeholders are
    materialized here so the launcher never has to reinterpret the public
    contract.
    """

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    out: torch.Tensor | None
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    seqused_k: torch.Tensor | None
    page_table: torch.Tensor
    alibi_slopes: torch.Tensor | None
    s_aux: torch.Tensor | None
    max_seqlen_q: int
    max_seqlen_k: int
    window: NormalizedWindow
    is_softcap: bool
    adjusted_softcap: float
    adjusted_scale_softmax: float
    adjusted_scale_softmax_log2e: float
    total_q: int
    batch_size: int
    num_heads: int
    num_heads_k: int
    head_dim: int
    block_size: int
    num_pages: int
    is_paged: bool
    has_cache_kv: bool
    return_softmax_lse: bool
    arch: int
    num_sms: int
    qo_tma_aligned: bool
    kv_tma_aligned: bool
    max_num_splits: int = 0


def normalize_window(
    *,
    max_seqlen_q: int,
    max_seqlen_k: int,
    head_dim: int,
    is_paged: bool,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
) -> NormalizedWindow:
    """Normalize mask arguments exactly like vLLM's Hopper FA3 wrapper."""

    left = int(window_size_left)
    right = int(window_size_right)
    causal = bool(causal)

    if left >= max_seqlen_k - 1:
        left = -1
    if right >= max_seqlen_q - 1:
        right = -1

    if max_seqlen_q == 1 and left == -1 and right == -1:
        if (head_dim <= 64 or head_dim > 128) or not is_paged:
            causal = False

    if causal:
        right = 0

    normalized_causal = left < 0 and right == 0
    local = (left >= 0 or right >= 0) and not normalized_causal
    if left < 0 and right >= 0:
        left = max_seqlen_k - 1
    if left >= 0 and right < 0:
        right = max_seqlen_q - 1
    return NormalizedWindow(normalized_causal, local, left, right)


def _validate_python_arguments(
    *,
    dropout_p,
    deterministic,
    softmax_scale,
    softcap,
    return_attn_probs,
    return_softmax_lse,
    max_seqlen_q,
    max_seqlen_k,
    causal,
    window_size,
    cp_world_size,
    cp_rank,
    num_splits,
) -> tuple[int, int]:
    """Validate host-side scalar/container types and ranges."""

    if not isinstance(dropout_p, (int, float)):
        raise RuntimeError("dropout_p must be a Python number")
    if type(deterministic) is not bool:
        raise RuntimeError("deterministic must be a Python bool")
    if type(cp_world_size) is not int or type(cp_rank) is not int:
        raise RuntimeError("cp_world_size and cp_rank must be Python integers")
    if type(num_splits) is not int or num_splits < 0:
        raise RuntimeError("num_splits must be a non-negative Python integer")
    if num_splits > MAX_SPLIT_KV:
        raise RuntimeError(
            f"num_splits must be between 0 and {MAX_SPLIT_KV}, got {num_splits}"
        )
    if softmax_scale is not None and not isinstance(softmax_scale, (int, float)):
        raise RuntimeError("softmax_scale must be a Python number or None")
    if not isinstance(softcap, (int, float)):
        raise RuntimeError("softcap must be a Python number")
    if type(return_attn_probs) is not bool:
        raise RuntimeError("return_attn_probs must be a Python bool")
    if type(return_softmax_lse) is not bool:
        raise RuntimeError("return_softmax_lse must be a Python bool")
    if type(max_seqlen_q) is not int or type(max_seqlen_k) is not int:
        raise RuntimeError("max_seqlen_q and max_seqlen_k must be Python integers")
    if max_seqlen_q <= 0 or max_seqlen_k <= 0:
        raise RuntimeError("max_seqlen_q and max_seqlen_k must be positive")
    if type(causal) is not bool:
        raise RuntimeError("causal must be a Python bool")
    if window_size is not None and (
        not isinstance(window_size, (list, tuple)) or len(window_size) != 2
    ):
        raise RuntimeError("window_size must contain exactly two integers")

    requested_window = (-1, -1) if window_size is None else tuple(window_size)
    if any(type(bound) is not int or bound < -1 for bound in requested_window):
        raise RuntimeError("window_size values must be Python integers >= -1")
    return requested_window


def _reject_unsupported_features(
    *,
    q,
    k,
    v,
    q_v,
    dropout_p,
    return_attn_probs,
    q_descale,
    k_descale,
    v_descale,
    cp_world_size,
    cp_rank,
    cp_tot_seqused_k,
) -> None:
    """Reject public features that the current TLE FA3 kernels do not implement."""

    if dropout_p != 0:
        raise NotImplementedError("TLE FA3 does not support dropout")
    if q_v is not None:
        # TODO: implement distinct QK and QV head dimensions.
        raise NotImplementedError("TLE FA3 does not support q_v yet")
    if cp_world_size != 1 or cp_rank != 0 or cp_tot_seqused_k is not None:
        raise NotImplementedError("TLE FA3 does not support context parallelism")
    if return_attn_probs:
        raise NotImplementedError("TLE FA3 does not return attention probabilities")

    descales = (q_descale, k_descale, v_descale)
    if any(scale is not None for scale in descales) and any(
        tensor.dtype not in (torch.float16, torch.bfloat16) for tensor in (q, k, v)
    ):
        # TODO: consume FP8 query/cache descales in TLE.
        raise NotImplementedError("TLE FA3 does not support FP8 descales")


def _validate_tensor_contracts(
    *,
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_k,
    block_table,
    alibi_slopes,
    s_aux,
    out,
    q_descale,
    k_descale,
    v_descale,
) -> tuple[bool, int]:
    """Validate each tensor's own type, layout metadata, dtype, and device."""

    if not all(isinstance(tensor, torch.Tensor) for tensor in (q, k, v)):
        raise RuntimeError("q, k, and v must be torch.Tensor instances")
    if q.ndim != 3:
        raise RuntimeError(f"q must have shape (total_q, h, d), got {q.shape}")
    if block_table is not None and (
        not isinstance(block_table, torch.Tensor)
        or block_table.ndim != 2
        or block_table.dtype != torch.int32
        or block_table.device != q.device
        or block_table.stride(-1) != 1
    ):
        raise RuntimeError(
            "block_table must be a row-contiguous int32 2D Q-device tensor"
        )

    is_paged = block_table is not None
    if k.ndim != (4 if is_paged else 3) or v.ndim != k.ndim:
        layout = "(pages, block, hk, d)" if is_paged else "(total_k, hk, d)"
        raise RuntimeError(f"k and v must use the {layout} layout")
    if (
        not isinstance(cu_seqlens_q, torch.Tensor)
        or cu_seqlens_q.ndim != 1
        or cu_seqlens_q.numel() < 2
        or cu_seqlens_q.dtype != torch.int32
        or cu_seqlens_q.device != q.device
        or not cu_seqlens_q.is_contiguous()
    ):
        raise RuntimeError(
            "cu_seqlens_q must be a non-empty contiguous int32 Q-device vector"
        )

    if cu_seqlens_k is not None and (
        not isinstance(cu_seqlens_k, torch.Tensor)
        or cu_seqlens_k.dtype != torch.int32
        or cu_seqlens_k.device != q.device
        or not cu_seqlens_k.is_contiguous()
    ):
        raise RuntimeError("cu_seqlens_k must match cu_seqlens_q's batch layout")
    if seqused_k is not None and (
        not isinstance(seqused_k, torch.Tensor)
        or seqused_k.dtype != torch.int32
        or seqused_k.device != q.device
        or not seqused_k.is_contiguous()
    ):
        raise RuntimeError("seqused_k must be a contiguous int32 Q-device batch vector")
    if alibi_slopes is not None and (
        not isinstance(alibi_slopes, torch.Tensor)
        or alibi_slopes.ndim not in (1, 2)
        or alibi_slopes.dtype != torch.float32
        or alibi_slopes.device != q.device
        or alibi_slopes.stride(-1) != 1
    ):
        raise RuntimeError(
            "alibi_slopes must be last-dimension-contiguous fp32 with shape "
            "(num_heads,) or (batch_size, num_heads) on the Q device"
        )
    if s_aux is not None and (
        not isinstance(s_aux, torch.Tensor)
        or s_aux.shape != (q.size(1),)
        or s_aux.dtype != torch.bfloat16
        or s_aux.device != q.device
        or not s_aux.is_contiguous()
        or q.size(1) > 64
    ):
        raise RuntimeError(
            "s_aux must be a contiguous bf16 Q-device vector with shape "
            "(num_heads,), and num_heads must not exceed 64"
        )
    if out is not None and (
        not isinstance(out, torch.Tensor)
        or out.shape != q.shape
        or out.dtype != q.dtype
        or out.device != q.device
        or out.stride(-1) != 1
    ):
        raise RuntimeError("out must match q's shape, dtype, device, and last stride")

    for name, scale in zip(("q", "k", "v"), (q_descale, k_descale, v_descale)):
        if scale is not None and (
            not isinstance(scale, torch.Tensor)
            or scale.dtype != torch.float32
            or scale.device != q.device
        ):
            raise RuntimeError(
                f"{name}_descale must be a float32 tensor on the Q device"
            )
    return is_paged, cu_seqlens_q.numel() - 1


def _validate_argument_relationships(
    *,
    q,
    k,
    v,
    cu_seqlens_k,
    seqused_k,
    block_table,
    alibi_slopes,
    max_seqlen_q,
    causal,
    requested_window,
    is_paged,
    batch_size,
) -> tuple[int, int]:
    """Validate relationships between otherwise well-formed arguments."""

    if causal and 0 < requested_window[1] < max_seqlen_q - 1:
        raise RuntimeError(
            "causal=True conflicts with a positive right attention window"
        )
    if k.shape != v.shape or k.size(-1) != q.size(-1):
        raise RuntimeError("k/v shapes must match and use q's head dimension")

    num_heads = q.size(1)
    num_heads_k = k.size(-2)
    if num_heads <= 0 or num_heads_k <= 0 or num_heads % num_heads_k != 0:
        raise RuntimeError("num_heads_k must be positive and divide num_heads")
    if alibi_slopes is not None and alibi_slopes.shape not in (
        (num_heads,),
        (batch_size, num_heads),
    ):
        raise RuntimeError(
            "alibi_slopes must be last-dimension-contiguous fp32 with shape "
            "(num_heads,) or (batch_size, num_heads) on the Q device"
        )
    if (cu_seqlens_k is None) == (seqused_k is None):
        raise RuntimeError("exactly one of cu_seqlens_k and seqused_k is required")
    if cu_seqlens_k is not None and cu_seqlens_k.shape != (batch_size + 1,):
        raise RuntimeError("cu_seqlens_k must match cu_seqlens_q's batch layout")
    if seqused_k is not None and seqused_k.shape != (batch_size,):
        raise RuntimeError("seqused_k must be a contiguous int32 Q-device batch vector")
    if is_paged and (seqused_k is None or block_table.size(0) != batch_size):
        raise RuntimeError(
            "paged KV requires seqused_k and one block-table row per sequence"
        )
    if is_paged and k.size(1) <= 0:
        raise RuntimeError("paged KV requires a positive page size")
    return num_heads, num_heads_k


def _validate_kernel_constraints(q, k, v, is_paged: bool) -> None:
    """Validate constraints imposed by the current Hopper TLE kernels."""

    if q.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(
            "TLE FA3 currently supports torch.float16 and torch.bfloat16 inputs only."
        )
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise RuntimeError("TLE FA3 requires q, k, and v to have the same dtype.")
    if k.device != q.device or v.device != q.device:
        raise RuntimeError("TLE FA3 requires q, k, and v on the same device.")
    if any(tensor.stride(-1) != 1 for tensor in (q, k, v)):
        raise RuntimeError(
            "TLE FA3 requires q, k, and v to be contiguous in head_dim; "
            "implicit layout copies are not permitted"
        )

    head_size = q.size(-1)
    if head_size < 32 or head_size > 256 or head_size % 8 != 0:
        raise RuntimeError(
            "TLE FA3 requires 32 <= head_dim <= 256 and head_dim % 8 == 0."
        )
    if k.stride() != v.stride():
        raise RuntimeError("TLE FA3 requires k and v to have matching strides")
    if is_paged and k.stride(0) % k.stride(1):
        raise RuntimeError("paged K/V page strides must be multiples of token strides")


def ensure_tma_allocator() -> None:
    global _TMA_ALLOCATOR_REGISTERED
    if _TMA_ALLOCATOR_REGISTERED:
        return

    def _alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(_alloc_fn)
    _TMA_ALLOCATOR_REGISTERED = True


@lru_cache(maxsize=1)
def _missing_fa3_primitives() -> tuple[str, ...]:
    """Return the required FlagTree Hopper extensions absent at runtime."""

    if tle is None:
        return ("TLE >= 3.6",)

    missing = []
    gpu = getattr(tle, "gpu", None)
    copy = getattr(gpu, "copy", None)
    alloc_barriers = getattr(gpu, "alloc_barriers", None)
    buffered_tensor = getattr(gpu, "buffered_tensor", None)

    copy_params = _parameter_names(copy)
    if not {"mask", "other"}.issubset(copy_params):
        missing.append("masked tle.gpu.copy(mask=..., other=...)")

    barrier_params = _parameter_names(alloc_barriers)
    if "arrival_mode" not in barrier_params:
        missing.append("tle.gpu.alloc_barriers(arrival_mode=...)")

    if buffered_tensor is None or not hasattr(buffered_tensor, "reshape"):
        missing.append("tle.gpu.buffered_tensor.reshape")

    if not hasattr(tl, "make_tensor_descriptor"):
        missing.append("triton.language.make_tensor_descriptor")
    if not hasattr(triton, "set_allocator"):
        missing.append("triton.set_allocator")
    return tuple(missing)


@lru_cache(maxsize=128)
def _tma_stride_signature_is_aligned(
    element_size: int, strides: tuple[int, ...]
) -> bool:
    return all((stride * element_size) % 16 == 0 for stride in strides[:-1])


def _tma_strides_are_aligned(tensor: torch.Tensor) -> bool:
    return _tma_stride_signature_is_aligned(tensor.element_size(), tensor.stride())


def _fa3_runtime_error(device=None) -> str | None:
    missing = _missing_fa3_primitives()
    if missing:
        return (
            "fa_version=3 is missing required FlagTree Hopper TLE primitives: "
            + ", ".join(missing)
        )
    if (
        not torch.cuda.is_available()
        or (device is not None and torch.device(device).type != "cuda")
        or torch.cuda.get_device_capability(device)[0] != 9
    ):
        return "fa_version=3 requires an available NVIDIA Hopper (SM90) device"
    return None


def is_fa3_supported(device=None) -> bool:
    return _fa3_runtime_error(device) is None


def _require_fa3_runtime(device) -> None:
    """Require the CUDA, Hopper, TMA, and TLE runtime used by this backend."""

    error = _fa3_runtime_error(device)
    if error is not None:
        raise RuntimeError(error)


def validate_fa3_plan(inputs: PreparedFA3Inputs, plan) -> None:
    """Validate the one internal boundary not covered by public input checks."""

    from .scheduling import KernelFamily

    if plan.kernel not in (KernelFamily.DIRECT, KernelFamily.LONG):
        raise RuntimeError("FA3 launcher requires a concrete kernel plan")

    pack_factor = plan.pack_factor
    if (
        plan.pack_gqa
        and (pack_factor != inputs.num_heads // inputs.num_heads_k or pack_factor <= 1)
    ) or (not plan.pack_gqa and pack_factor != 1):
        raise RuntimeError(
            "internal FA3 packed shape does not match the execution plan"
        )

    if plan.requires_tma_alignment:
        aligned = inputs.qo_tma_aligned
        if not inputs.is_paged or not plan.paged_kv_non_tma:
            aligned = aligned and inputs.kv_tma_aligned
        if not aligned:
            raise RuntimeError(
                "TLE FA3 TMA-backed path requires 16-byte aligned Q/K/V/O strides"
            )


def prepare_fa3_inputs(
    *,
    q,
    k,
    v,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    cu_seqlens_k,
    seqused_k,
    q_v,
    dropout_p,
    softmax_scale,
    causal,
    window_size,
    softcap,
    alibi_slopes,
    deterministic,
    return_attn_probs,
    return_softmax_lse,
    block_table,
    out,
    q_descale,
    k_descale,
    v_descale,
    num_splits,
    s_aux,
    cp_world_size,
    cp_rank,
    cp_tot_seqused_k,
) -> PreparedFA3Inputs:
    """Validate and canonicalize one public FA3 invocation.

    This is the only public-contract boundary for the TLE implementation.  A
    successful return guarantees that scheduling can inspect shapes without
    optional-layout branches and that the launcher receives kernel-compatible
    tensors and ABI placeholders.
    """

    # Foreign scheduler metadata remains outside the native plan.  num_splits
    # is a public upper bound: 0 selects the native auto policy, 1 requests a
    # one-pass kernel, and larger values cap native Split-KV parallelism.
    requested_window = _validate_python_arguments(
        dropout_p=dropout_p,
        deterministic=deterministic,
        softmax_scale=softmax_scale,
        softcap=softcap,
        return_attn_probs=return_attn_probs,
        return_softmax_lse=return_softmax_lse,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=causal,
        window_size=window_size,
        cp_world_size=cp_world_size,
        cp_rank=cp_rank,
        num_splits=num_splits,
    )
    is_paged, batch_size = _validate_tensor_contracts(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_k=seqused_k,
        block_table=block_table,
        alibi_slopes=alibi_slopes,
        s_aux=s_aux,
        out=out,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    _reject_unsupported_features(
        q=q,
        k=k,
        v=v,
        q_v=q_v,
        dropout_p=dropout_p,
        return_attn_probs=return_attn_probs,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        cp_world_size=cp_world_size,
        cp_rank=cp_rank,
        cp_tot_seqused_k=cp_tot_seqused_k,
    )
    num_heads, num_heads_k = _validate_argument_relationships(
        q=q,
        k=k,
        v=v,
        cu_seqlens_k=cu_seqlens_k,
        seqused_k=seqused_k,
        block_table=block_table,
        alibi_slopes=alibi_slopes,
        max_seqlen_q=max_seqlen_q,
        causal=causal,
        requested_window=requested_window,
        is_paged=is_paged,
        batch_size=batch_size,
    )
    _require_fa3_runtime(q.device)

    _validate_kernel_constraints(q, k, v, is_paged)

    # The seqused-K specialization never reads cu_seqlens_k.  Reuse the always
    # valid Q prefix tensor instead of allocating a device-side ABI placeholder.
    cu_seqlens_k_abi = cu_seqlens_q if cu_seqlens_k is None else cu_seqlens_k
    window = normalize_window(
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        head_dim=q.size(-1),
        is_paged=is_paged,
        causal=causal,
        window_size_left=requested_window[0],
        window_size_right=requested_window[1],
    )

    total_q, _, head_dim = q.shape
    block_size = k.size(1) if is_paged else 1
    num_pages = k.size(0) if is_paged else 0
    page_table = block_table
    if page_table is None:
        page_table = torch.empty((0, 0), device=q.device, dtype=torch.int32)

    scale = head_dim**-0.5 if softmax_scale is None else float(softmax_scale)
    softcap = float(softcap)
    if softcap > 0.0:
        is_softcap = True
        adjusted_softcap = scale / softcap
        adjusted_scale_softmax = softcap
    else:
        is_softcap = False
        adjusted_softcap = 0.0
        adjusted_scale_softmax = scale

    device_properties = torch.cuda.get_device_properties(q.device)
    arch = device_properties.major * 10 + device_properties.minor
    qo_tma_aligned = _tma_strides_are_aligned(q) and (
        out is None or _tma_strides_are_aligned(out)
    )
    kv_tma_aligned = _tma_strides_are_aligned(k) and _tma_strides_are_aligned(v)

    # Registration mutates global Triton runtime state, so do it only after all
    # argument validation and canonical tensor preparation have succeeded.
    ensure_tma_allocator()

    return PreparedFA3Inputs(
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k_abi,
        seqused_k=seqused_k,
        page_table=page_table,
        alibi_slopes=alibi_slopes,
        s_aux=s_aux,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        window=window,
        is_softcap=is_softcap,
        adjusted_softcap=adjusted_softcap,
        adjusted_scale_softmax=adjusted_scale_softmax,
        adjusted_scale_softmax_log2e=adjusted_scale_softmax * 1.4426950408889634,
        total_q=total_q,
        batch_size=batch_size,
        num_heads=num_heads,
        num_heads_k=num_heads_k,
        head_dim=head_dim,
        block_size=block_size,
        num_pages=num_pages,
        is_paged=is_paged,
        has_cache_kv=seqused_k is not None,
        return_softmax_lse=return_softmax_lse,
        arch=arch,
        num_sms=device_properties.multi_processor_count,
        qo_tma_aligned=qo_tma_aligned,
        kv_tma_aligned=kv_tma_aligned,
        max_num_splits=num_splits,
    )


__all__ = [
    "NormalizedWindow",
    "PreparedFA3Inputs",
    "is_fa3_supported",
    "normalize_window",
    "prepare_fa3_inputs",
    "validate_fa3_plan",
]
