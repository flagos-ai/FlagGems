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
from typing import Optional

import torch
import triton
from triton.backends.tsingmicro.precision import (
    restore_precision_mode,
    set_precision_mode,
)

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)

_FLOAT8_E8M0FNU = getattr(torch, "float8_e8m0fnu", None)

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


def _redispatch_copy_(dst, src, non_blocking: bool):
    return torch.ops.aten.copy_.default.redispatch(
        _FALLBACK_KEYSET, dst, src, non_blocking
    )


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


def copy_(dst: torch.Tensor, src: torch.Tensor, non_blocking: bool = False):
    logger.debug("GEMS_TSINGMICRO COPY_")
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
