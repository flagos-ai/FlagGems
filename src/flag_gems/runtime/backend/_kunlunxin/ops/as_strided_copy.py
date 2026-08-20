import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.as_strided_copy import (
    _can_use_byte_triton,
    _can_use_triton,
    _fallback_as_strided_copy,
    _fallback_as_strided_copy_out,
    _launch_as_strided_copy,
    _launch_byte_as_strided_copy,
    _make_as_strided_view,
)
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.shape_utils import MemOverlap, has_internal_overlapping

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))

# Only take the block-copy fast path when the contiguous inner run is long
# enough for a block DMA to beat launch overhead; otherwise defer to the
# generic strided kernels.
_MIN_BLOCK_RUN = 16
_FLAT_BLOCK = 8192
_MAX_BLOCK_R = 4096
# For non-unit inner strides we bake the stride into the kernel as a constexpr
# so OffsetAnalysis can prove the (regular) access pattern instead of treating
# a runtime stride as a discrete gather. Small strides waste bounded bandwidth
# (~S x) so are always allowed; larger strides only for small tensors, where
# launch latency dominates and the wasted traffic is negligible. Big + large-
# stride views (e.g. a full-matrix transpose) fall back to generic.
_MAX_SMALL_STRIDE = 4
_SMALL_NUMEL = 65536
_STRIDE_BLOCK = 4096


@libentry()
@triton.jit
def _flat_copy_kernel(inp, out, n_elements, BLOCK: tl.constexpr):
    # Pure contiguous copy: both sides linear -> block DMA.
    pid = ext.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    vals = tl.load(inp + offs, mask=mask, other=0)
    tl.store(out + offs, vals, mask=mask)


@libentry()
@triton.jit
def _block_copy_kernel(inp, out, base_ptr, R, BLOCK_R: tl.constexpr):
    # One program per output row. Each row copies R contiguous elements from
    # a per-row scalar base offset into the (contiguous) output. The inner run
    # is stride-1 with a scalar base, so OffsetAnalysis proves it -> block DMA,
    # completely avoiding the tiny-tile strided gather of the generic kernel.
    pid = ext.program_id(axis=0)
    in_base = tl.load(base_ptr + pid)
    out_base = pid.to(tl.int64) * R
    for c in range(0, R, BLOCK_R):
        cols = c + tl.arange(0, BLOCK_R)
        mask = cols < R
        vals = tl.load(inp + in_base + cols, mask=mask, other=0)
        tl.store(out + out_base + cols, vals, mask=mask)


@libentry()
@triton.jit
def _block_copy_1d_kernel(inp, out, row_stride, R, BLOCK_R: tl.constexpr):
    # Specialization for a single (regularly strided) outer dim: base is
    # computed in-kernel as pid * row_stride, avoiding the host-side base
    # array (and its extra kernel launches).
    pid = ext.program_id(axis=0)
    in_base = pid.to(tl.int64) * row_stride
    out_base = pid.to(tl.int64) * R
    for c in range(0, R, BLOCK_R):
        cols = c + tl.arange(0, BLOCK_R)
        mask = cols < R
        vals = tl.load(inp + in_base + cols, mask=mask, other=0)
        tl.store(out + out_base + cols, vals, mask=mask)


@libentry()
@triton.jit
def _const_stride_1d_kernel(inp, out, n_out, S: tl.constexpr, BLOCK: tl.constexpr):
    # Single regular-strided run. S is a constexpr so `offs * S` is a
    # compile-time-known regular pattern -> OffsetAnalysis proves it and issues
    # a strided/superset DMA instead of the runtime-stride discrete gather.
    pid = ext.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_out
    vals = tl.load(inp + offs * S, mask=mask, other=0)
    tl.store(out + offs, vals, mask=mask)


@libentry()
@triton.jit
def _const_stride_row_kernel(
    inp, out, base_ptr, R, S: tl.constexpr, BLOCK_R: tl.constexpr
):
    # One program per output row; each row reads R elements at constexpr inner
    # stride S from a per-row scalar base. Same idea as _block_copy_kernel but
    # for a regular (non-unit) inner stride baked in as a constexpr.
    pid = ext.program_id(axis=0)
    in_base = tl.load(base_ptr + pid)
    out_base = pid.to(tl.int64) * R
    for c in range(0, R, BLOCK_R):
        cols = c + tl.arange(0, BLOCK_R)
        mask = cols < R
        vals = tl.load(inp + in_base + cols * S, mask=mask, other=0)
        tl.store(out + out_base + cols, vals, mask=mask)


@libentry()
@triton.jit
def _const_stride_row1d_kernel(
    inp, out, row_stride, R, S: tl.constexpr, BLOCK_R: tl.constexpr
):
    # Single (regularly strided) outer dim + constexpr inner stride S: base is
    # computed in-kernel as pid * row_stride, avoiding the host-side base array
    # (and its several extra small kernel launches, which otherwise dominate).
    pid = ext.program_id(axis=0)
    in_base = pid.to(tl.int64) * row_stride
    out_base = pid.to(tl.int64) * R
    for c in range(0, R, BLOCK_R):
        cols = c + tl.arange(0, BLOCK_R)
        mask = cols < R
        vals = tl.load(inp + in_base + cols * S, mask=mask, other=0)
        tl.store(out + out_base + cols, vals, mask=mask)


def _merge_contiguous(sizes, strides):
    # Drop size-1 dims (any stride, no effect on layout), then merge adjacent
    # dims from the inside out when stride[outer] == stride[inner]*size[inner].
    dims = [(sz, st) for sz, st in zip(sizes, strides) if sz != 1]
    if not dims:
        return [1], [1]
    merged_s = []
    merged_t = []
    for sz, st in reversed(dims):
        if merged_s and st == merged_t[-1] * merged_s[-1]:
            merged_s[-1] *= sz  # inner stride stays the smallest one
        else:
            merged_s.append(sz)
            merged_t.append(st)
    merged_s.reverse()
    merged_t.reverse()
    return merged_s, merged_t


def _try_fast_copy(view: torch.Tensor, out: torch.Tensor) -> bool:
    # Returns True if a fast path handled the copy, else False.
    if not _can_use_triton(view, out):
        return False

    n = out.numel()
    if view.is_contiguous() and out.is_contiguous():
        flat_in = view.reshape(-1)
        flat_out = out.reshape(-1)
        grid = (triton.cdiv(n, _FLAT_BLOCK),)
        _flat_copy_kernel[grid](flat_in, flat_out, n, BLOCK=_FLAT_BLOCK)
        return True

    if not out.is_contiguous():
        return False

    merged_s, merged_t = _merge_contiguous(list(view.shape), list(view.stride()))
    R = merged_s[-1]
    S = merged_t[-1]
    if R < _MIN_BLOCK_RUN:
        return False

    outer_s = merged_s[:-1]
    outer_t = merged_t[:-1]

    if S == 1:
        if not outer_s:
            # Contiguous run == whole tensor but view not flagged contiguous
            # (e.g. leading size-1 dims). Treat as flat copy.
            flat_out = out.reshape(-1)
            grid = (triton.cdiv(n, _FLAT_BLOCK),)
            _flat_copy_kernel[grid](view, flat_out, n, BLOCK=_FLAT_BLOCK)
            return True

        block_r = min(triton.next_power_of_2(R), _MAX_BLOCK_R)
        grid = (math.prod(outer_s),)

        if len(outer_s) == 1:
            _block_copy_1d_kernel[grid](
                view, out.reshape(-1), outer_t[0], R, BLOCK_R=block_r
            )
            return True

        n_rows = math.prod(outer_s)
        rem = torch.arange(n_rows, device=view.device, dtype=torch.int64)
        base = torch.zeros(n_rows, device=view.device, dtype=torch.int64)
        for sz, st in zip(reversed(outer_s), reversed(outer_t)):
            base += (rem % sz) * st
            rem = rem // sz

        _block_copy_kernel[grid](view, out.reshape(-1), base, R, BLOCK_R=block_r)
        return True

    # Non-unit inner stride: bake it as a constexpr so the (regular) access
    # pattern is provable. Guard against pathological bandwidth waste on large
    # tensors with a large stride (e.g. a full-matrix transpose).
    if S > _MAX_SMALL_STRIDE and n > _SMALL_NUMEL:
        return False

    if not outer_s:
        grid = (triton.cdiv(R, _STRIDE_BLOCK),)
        _const_stride_1d_kernel[grid](
            view, out.reshape(-1), R, S=S, BLOCK=_STRIDE_BLOCK
        )
        return True

    block_r = min(triton.next_power_of_2(R), _MAX_BLOCK_R)
    grid = (math.prod(outer_s),)

    if len(outer_s) == 1:
        _const_stride_row1d_kernel[grid](
            view, out.reshape(-1), outer_t[0], R, S=S, BLOCK_R=block_r
        )
        return True

    n_rows = math.prod(outer_s)
    rem = torch.arange(n_rows, device=view.device, dtype=torch.int64)
    base = torch.zeros(n_rows, device=view.device, dtype=torch.int64)
    for sz, st in zip(reversed(outer_s), reversed(outer_t)):
        base += (rem % sz) * st
        rem = rem // sz

    _const_stride_row_kernel[grid](view, out.reshape(-1), base, R, S=S, BLOCK_R=block_r)
    return True


def as_strided_copy(input, size, stride, storage_offset=None):
    logger.debug("GEMS_KUNLUNXIN AS_STRIDED_COPY")
    if input.device.type != "cuda":
        view = _make_as_strided_view(input, size, stride, storage_offset)
        return view.clone(memory_format=torch.contiguous_format)

    out = torch.empty(size, dtype=input.dtype, device=input.device)
    if out.numel() == 0:
        _make_as_strided_view(input, size, stride, storage_offset)
        return out

    view = _make_as_strided_view(input, size, stride, storage_offset)
    if _try_fast_copy(view, out):
        return out
    if _can_use_triton(view, out):
        return _launch_as_strided_copy(view, out)
    if _can_use_byte_triton(view, out):
        return _launch_byte_as_strided_copy(view, out)
    return _fallback_as_strided_copy(input, size, stride, storage_offset)


def as_strided_copy_out(input, size, stride, storage_offset=None, *, out):
    logger.debug("GEMS_KUNLUNXIN AS_STRIDED_COPY_OUT")
    if out.dtype != input.dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {input.dtype}, but got {out.dtype} instead"
        )

    target_size = tuple(size)
    if tuple(out.shape) != target_size:
        out.resize_(target_size)

    if out.numel() == 0:
        _make_as_strided_view(input, size, stride, storage_offset)
        return out

    if input.device.type != "cuda":
        view = _make_as_strided_view(input, size, stride, storage_offset)
        if (
            torch._C._is_alias_of(input, out)
            or has_internal_overlapping(out) != MemOverlap.No
        ):
            view = view.clone(memory_format=torch.contiguous_format)
        out.copy_(view)
        return out

    if (
        torch._C._is_alias_of(input, out)
        or has_internal_overlapping(out) != MemOverlap.No
    ):
        return _fallback_as_strided_copy_out(
            input, size, stride, storage_offset, out=out
        )

    view = _make_as_strided_view(input, size, stride, storage_offset)
    if _try_fast_copy(view, out):
        return out
    if _can_use_triton(view, out):
        return _launch_as_strided_copy(view, out)
    if _can_use_byte_triton(view, out):
        return _launch_byte_as_strided_copy(view, out)
    return _fallback_as_strided_copy_out(input, size, stride, storage_offset, out=out)
