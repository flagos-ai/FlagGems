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

import logging
import os
import warnings
from typing import Optional

import torch
import triton
import triton.language as tl
from triton.backends.tsingmicro.precision import (
    restore_precision_mode,
    set_precision_mode,
)

from ..utils.codegen_config_utils import CodeGenConfig
from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)

_FLOAT8_E8M0FNU = getattr(torch, "float8_e8m0fnu", None)

# TSM_FORCE_CUDAGRAPH: when set, copy_ runs the on-device-only code path
# (no CPU round-trips / ATen redispatch for same-device strided tensors), so
# the surrounding op graph can be captured by CUDA Graph.
_FORCE_CUDAGRAPH = os.environ.get("TSM_FORCE_CUDAGRAPH") is not None

# Generated pointwise kernels compute task ids in 32-bit arithmetic, so a
# single launch cannot address more than INT32_MAX elements.
_INT32_MAX = torch.iinfo(torch.int32).max

_COPY_RANK3_CODEGEN_CONFIG = CodeGenConfig(
    max_tile_size=1024,
    max_grid_size=(32, 1, 1),
    max_num_warps_per_cta=1,
    prefer_block_pointer=True,
    prefer_1d_tile=False,
)

# TX Triton may fail lowering int64 pointwise loads when a non-unit dim uses a
# large stride (GetStructuredStateOp rewrite → llvm.cond_br with `index`).
_LARGE_STRIDE_THRESHOLD = 2048

# f32 represents integers up to 2**24 exactly. Under precision-mode 0/1 the
# rank-N tile decode (%/div) and the per-dim address offsets of a strided copy
# are vectorized i32 ops demoted to f32, so intermediates >= 2**24 lose
# precision and elements land at wrong offsets. Contiguous copies collapse to
# a flat task whose offsets stay in the scalar domain and are exempt, so only
# strided layouts (or views whose max offset crosses the bound) are gated.
# Force pm=2 for exact integer arithmetic (same rationale as cat/flip).
_F32_PRECISION_THRESHOLD = 2**24


def _needs_i64_precision(dst: torch.Tensor, src: torch.Tensor) -> bool:
    if dst.is_contiguous() and src.is_contiguous():
        return False
    for t in (dst, src):
        if t.numel() > _F32_PRECISION_THRESHOLD:
            return True
        max_offset = sum(
            (size - 1) * abs(stride) for size, stride in zip(t.shape, t.stride())
        )
        if max_offset > _F32_PRECISION_THRESHOLD:
            return True
    return False


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def _copy_kernel(src):
    return src


@pointwise_dynamic(
    is_tensor=[True],
    promotion_methods=[(0, "DEFAULT")],
    config=_COPY_RANK3_CODEGEN_CONFIG,
)
@triton.jit
def _copy_rank3_kernel(src):
    return src


@triton.jit
def _copy_contiguous_kernel(src, dst, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(dst + offsets, tl.load(src + offsets, mask=mask), mask=mask)


def _can_use_contiguous_kernel(dst: torch.Tensor, src: torch.Tensor) -> bool:
    return (
        dst.numel() <= 128
        and dst.is_contiguous()
        and src.is_contiguous()
        and dst.shape == src.shape
    )


def _can_use_rank3_block_pointer(dst: torch.Tensor, src: torch.Tensor) -> bool:
    return (
        dst.ndim == 3
        and src.ndim == 3
        and dst.shape == src.shape
        and dst.shape[-1] <= 128
        and dst.stride(-1) == 1
        and src.stride(-1) == 1
        and not (dst.is_contiguous() and src.is_contiguous())
    )


def _redispatch_copy_(dst, src, non_blocking: bool):
    return torch.ops.aten.copy_.default.redispatch(
        _FALLBACK_KEYSET, dst, src, non_blocking
    )


def _triton_copy_(dst: torch.Tensor, src: torch.Tensor, non_blocking: bool):
    """Launch the triton copy kernel; dst/src are same-device strided tensors."""
    expanded_src = src if src.shape == dst.shape else src.expand(dst.shape)
    try:
        # Pointwise preparation collapses contiguous input/output layouts to a
        # flat task, avoiding expensive per-element rank-N index reconstruction.
        # Keep direct instantiation for zero-stride destinations (written-to
        # by some workloads even though ATen refuses them), which general
        # pointwise output validation treats as internally overlapping.
        if any(
            size > 1 and stride == 0
            for size, stride in zip(dst.shape, dst.stride())
        ):
            overload = _copy_kernel.instantiate(expanded_src.ndim)
            overload(expanded_src, out0=dst)
        else:
            _copy_kernel(expanded_src, out0=dst)
    except Exception as exc:
        # Emergency net for unenumerated TX Triton compiler failures; the
        # shapes/dtypes themselves are supported, so this should stay quiet.
        logger.warning(
            "GEMS_TSINGMICRO COPY_ triton path failed (%s); redispatch to ATen",
            exc,
        )
        return _redispatch_copy_(dst, src, non_blocking)
    return dst


def _copy_in_chunks_(dst: torch.Tensor, src: torch.Tensor, non_blocking: bool):
    """Split a >INT32_MAX copy into 32-bit-addressable slices."""
    axis = None
    for i in range(dst.ndim):
        if dst.shape[i] > 1 and dst.numel() // dst.shape[i] <= _INT32_MAX:
            axis = i
            break
    if axis is None:
        # Every complementary product already exceeds int32; peel the first
        # non-trivial dim and let the recursion split further.
        axis = next(i for i in range(dst.ndim) if dst.shape[i] > 1)
    per_slice = dst.numel() // dst.shape[axis]
    step = max(1, _INT32_MAX // per_slice)
    expanded = src if src.shape == dst.shape else src.expand(dst.shape)
    for begin in range(0, dst.shape[axis], step):
        length = min(step, dst.shape[axis] - begin)
        copy_(
            dst.narrow(axis, begin, length),
            expanded.narrow(axis, begin, length),
            non_blocking,
        )
    return dst


def _copy_complex_(dst: torch.Tensor, src: torch.Tensor, non_blocking: bool):
    """Complex copy via real/imag decomposition (triton has no complex dtype).

    The real()/imag() views read through conj/neg flags, so lazy conjugate or
    negative views of src are handled for free.
    """
    if src.is_complex():
        if dst.is_complex():
            copy_(dst.real, src.real, non_blocking)
            copy_(dst.imag, src.imag, non_blocking)
            return dst
        warnings.warn(
            "Casting complex values to real discards the imaginary part.",
            UserWarning,
            stacklevel=3,
        )
        return copy_(dst, src.real, non_blocking)
    # Real source into a complex destination: fill real part, zero imag part.
    copy_(dst.real, src, non_blocking)
    copy_(dst.imag, torch.zeros((), dtype=dst.imag.dtype, device=dst.device))
    return dst


def _has_large_noncontig_stride(t: torch.Tensor) -> bool:
    if t.numel() == 0 or t.is_contiguous():
        return False
    for size, stride in zip(t.shape, t.stride()):
        if size > 1 and abs(stride) >= _LARGE_STRIDE_THRESHOLD:
            return True
    return False


def _int64_large_strided_unsafe(dst: torch.Tensor, src: torch.Tensor) -> bool:
    """TX backend cannot reliably compile int64 copy with large non-contig strides."""
    if dst.dtype != torch.int64 and src.dtype != torch.int64:
        return False
    return _has_large_noncontig_stride(dst) or _has_large_noncontig_stride(src)


def _can_use_triton(dst: torch.Tensor, src: torch.Tensor) -> bool:
    if dst.layout != torch.strided or src.layout != torch.strided:
        return False
    if dst.device != src.device:
        return False
    if dst.is_quantized or src.is_quantized:
        return False
    if src.is_complex() or dst.is_complex():
        # Preserve PyTorch's behaviour of warning when casting complex to real
        # by forcing the redispatch path, which issues the warning internally.
        return False
    if _FLOAT8_E8M0FNU is not None and (
        src.dtype == _FLOAT8_E8M0FNU or dst.dtype == _FLOAT8_E8M0FNU
    ):
        # Triton does not support float8 yet, so defer to PyTorch which has a reference implementation.
        return False
    if _int64_large_strided_unsafe(dst, src):
        return False
    return True


def _expand_like(src: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    if src.shape == target_shape:
        return src
    return src.expand(target_shape)


def copy(
    template: torch.Tensor, src: torch.Tensor, *, non_blocking: Optional[bool] = False
):
    logger.debug("GEMS_TSINGMICRO COPY (functional)")
    out = torch.empty_strided(
        template.size(), template.stride(), dtype=template.dtype, device=template.device
    )
    copy_(out, src, non_blocking=bool(non_blocking))
    return out


def _cudagraph_copy_(dst: torch.Tensor, src: torch.Tensor, non_blocking: bool):
    """On-device-only copy_ path (TSM_FORCE_CUDAGRAPH).

    No CPU round-trips: every case a triton kernel cannot address is either
    re-expressed on device (chunked copies, complex decomposition, fp8 via
    uint8 reinterpretation) or explicitly left to ATen for the cases where no
    device kernel exists at all (cross-device DMA, sparse / quantized layouts).
    """
    if isinstance(src, (int, float, bool, complex)):
        src = torch.tensor(src, device=dst.device)
    elif not isinstance(src, torch.Tensor):
        raise TypeError("unsupport src type for copy_: ", type(src))

    # this is the same as PyTorch's check
    if dst._is_zerotensor():
        raise RuntimeError("ZeroTensors are immutable. Call clone() before copy_.")
    if src._is_zerotensor():
        return dst.zero_()

    # No triton path exists for these; ATen owns cross-device DMA, sparse and
    # quantized layouts.
    if (
        dst.device != src.device
        or dst.layout != torch.strided
        or src.layout != torch.strided
        or dst.is_quantized
        or src.is_quantized
    ):
        return _redispatch_copy_(dst, src, non_blocking)

    # Triton has no fp8 dtype yet. Same-dtype fp8 copies are pure bit moves, so
    # reinterpret through uint8; cross-dtype fp8 casts need ATen's reference.
    if _FLOAT8_E8M0FNU is not None and (
        src.dtype == _FLOAT8_E8M0FNU or dst.dtype == _FLOAT8_E8M0FNU
    ):
        if src.dtype == dst.dtype:
            return copy_(
                dst.view(torch.uint8), src.view(torch.uint8), non_blocking
            )
        return _redispatch_copy_(dst, src, non_blocking)

    if src.is_complex() or dst.is_complex():
        return _copy_complex_(dst, src, non_blocking)

    # Materialize lazy neg/conj views so the kernel reads the visible values.
    if src.is_neg():
        src = src.resolve_neg()
    if src.is_conj():
        src = src.resolve_conj()
    if dst.is_neg():
        # Writes must land inverted so reads through the neg view agree.
        src = -src

    if torch._C._is_alias_of(dst, src):
        # Align with PyTorch: if metadata fully matches, this is a no-op.
        if (
            dst.storage_offset() == src.storage_offset()
            and dst.stride() == src.stride()
            and dst.size() == src.size()
            and dst.dtype == src.dtype
            and dst.device == src.device
            and dst.is_conj() == src.is_conj()
            and dst.is_neg() == src.is_neg()
        ):
            return dst
        # Partially overlapping views: snapshot src so reads cannot be
        # corrupted by the writes into dst.
        src = src.clone()

    # Broadcast validation, also for empty tensors (matches PyTorch).
    try:
        broadcast_shape = torch.broadcast_shapes(dst.shape, src.shape)
    except RuntimeError as exc:
        raise RuntimeError(str(exc)) from exc
    if torch.Size(broadcast_shape) != dst.shape:
        raise RuntimeError(
            f"The broadcast shape {broadcast_shape} does not match destination shape {tuple(dst.shape)}"
        )

    if dst.numel() == 0:
        return dst

    if dst.numel() > _INT32_MAX:
        return _copy_in_chunks_(dst, src, non_blocking)

    return _triton_copy_(dst, src, non_blocking)


def copy_(dst: torch.Tensor, src: torch.Tensor, non_blocking: bool = False):
    logger.debug("GEMS_TSINGMICRO COPY_")
    if _FORCE_CUDAGRAPH:
        return _cudagraph_copy_(dst, src, non_blocking)
    if isinstance(src, (int, float, bool)):
        src = torch.tensor(src, device=dst.device)
    elif not isinstance(src, torch.Tensor):
        raise TypeError("unsupport src type for copy_: ", type(src))

    # this is the same as PyTorch's check
    if dst._is_zerotensor():
        raise RuntimeError("ZeroTensors are immutable. Call clone() before copy_.")
    if src._is_zerotensor():
        return dst.zero_()

    if torch._C._is_alias_of(dst, src):
        # Align with PyTorch: if metadata fully matches, this is a no-op.
        if (
            dst.storage_offset() == src.storage_offset()
            and dst.stride() == src.stride()
            and dst.size() == src.size()
            and dst.dtype == src.dtype
            and dst.device == src.device
            and dst.is_conj() == src.is_conj()
            and dst.is_neg() == src.is_neg()
        ):
            return dst
        # Otherwise defer to PyTorch for well-defined semantics on overlapping writes.
        return _redispatch_copy_(dst, src, non_blocking)

    if _FLOAT8_E8M0FNU is not None and (
        src.dtype == _FLOAT8_E8M0FNU or dst.dtype == _FLOAT8_E8M0FNU
    ):
        return _redispatch_copy_(dst, src, non_blocking)

    if src.numel() > 2**31 - 1 or dst.numel() > 2**31 - 1:
        return _redispatch_copy_(dst, src, non_blocking)

    if not _can_use_triton(dst, src):
        return _redispatch_copy_(dst, src, non_blocking)

    if dst.numel() == 0:
        # Respect PyTorch behaviour: empty tensors should still validate broadcast.
        return _redispatch_copy_(dst, src, non_blocking)

    try:
        broadcast_shape = torch.broadcast_shapes(dst.shape, src.shape)
    except RuntimeError as exc:
        raise RuntimeError(str(exc)) from exc

    if torch.Size(broadcast_shape) != dst.shape:
        raise RuntimeError(
            f"The broadcast shape {broadcast_shape} does not match destination shape {tuple(dst.shape)}"
        )

    expanded_src = _expand_like(src, dst.shape)

    if _can_use_contiguous_kernel(dst, expanded_src):
        try:
            grid = lambda meta: (triton.cdiv(dst.numel(), meta["BLOCK_SIZE"]),)
            _copy_contiguous_kernel[grid](
                expanded_src, dst, dst.numel(), BLOCK_SIZE=1024
            )
            return dst
        except Exception as exc:
            logger.warning(
                "GEMS_TSINGMICRO contiguous copy kernel failed (%s); "
                "falling back to pointwise copy",
                exc,
            )

    if _can_use_rank3_block_pointer(dst, expanded_src):
        try:
            _copy_rank3_kernel(expanded_src, out0=dst)
            return dst
        except Exception as exc:
            logger.warning(
                "GEMS_TSINGMICRO rank3 block-pointer copy failed (%s); "
                "falling back to pointwise copy",
                exc,
            )

    precision_token = (
        set_precision_mode("2") if _needs_i64_precision(expanded_src, dst) else None
    )
    try:
        # Pointwise preparation collapses contiguous input/output layouts to a
        # flat task, avoiding expensive per-element rank-N index reconstruction.
        # Keep direct instantiation for zero-stride destinations, which general
        # pointwise output validation treats as internally overlapping.
        if any(
            size > 1 and stride == 0 for size, stride in zip(dst.shape, dst.stride())
        ):
            overload = _copy_kernel.instantiate(expanded_src.ndim)
            overload(expanded_src, out0=dst)
        else:
            _copy_kernel(expanded_src, out0=dst)
    except Exception as exc:
        # TX Triton may still fail on unenumerated int64 / strided combos.
        logger.warning(
            "GEMS_TSINGMICRO COPY_ triton path failed (%s); redispatch to ATen",
            exc,
        )
        return _redispatch_copy_(dst, src, non_blocking)
    finally:
        if precision_token is not None:
            restore_precision_mode(precision_token)
    return dst


def contiguous_on_device(t: torch.Tensor) -> torch.Tensor:
    """``t.contiguous()`` that always routes through the FlagGems copy kernel.

    The native TXDA backend mishandles some permuted int64 copies (e.g. a 2-D
    transpose of int64 data), so op implementations must not rely on aten
    ``contiguous()``/``copy_`` when the FlagGems dispatcher may be inactive
    (direct ``flag_gems.<op>`` function calls).
    """
    if t.is_contiguous():
        return t
    out = torch.empty(t.shape, dtype=t.dtype, device=t.device)
    copy_(out, t, False)
    return out
