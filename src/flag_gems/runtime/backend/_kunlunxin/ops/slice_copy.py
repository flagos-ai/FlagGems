# Copyright 2026, The FlagOS Contributors.
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

"""``slice_copy`` on XPU3: route the strided gather onto the DMA paths.

The generic kernel walks the flat output index and derives the source address
with two or three **runtime** integer divisions per lane.  On this backend a
runtime div/mod inside a lane address scalarises the access: the benchmark cell
``(4096, 4096)`` sliced along dim 0 measured 614 us against 25 us for native
``slice_copy`` (0.04x), and the trailing-dim cell 1100 us against 328 us
(0.30x).  A dtype-equal speedup of 0.59x is what that costs.

Two shapes cover every slice this op performs; both are pure data movement, so
they are routed onto the engine instead of per-lane arithmetic:

* ``inner > 1`` -- the selected elements of every outer index form a
  ``(slice_len, inner)`` window whose innermost axis is unit-stride on both
  sides.  That is exactly a 2-D DMA with independent source/destination row
  strides, so the window goes through :func:`tle_copy` (``tle.dsa``).  Measured
  on the benchmark matrix: dim-0 fp16 18.6 us vs 15.9 us (0.84x), fp32 34.1 us
  vs 25.1 us (0.73x), and a ``(4096, 4096, 8)`` dim-1 slice 0.48 ms vs 4.85 ms
  (10.1x).
* ``inner == 1`` and ``step == 2`` -- every source element sits in the low half
  of a wider integer word.  Loading the word keeps the read unit-stride
  (contiguous 8-byte lanes) instead of a stride-2 element gather, and the
  half-word is stored into the contiguous output.  fp32 takes one output per
  lane (int64 load -> int32 store); 16-bit dtypes take two (int64 load ->
  packed int32 store).  Measured 268.7 us vs 453 us (1.69x) and 261.5 us vs
  328 us (1.26x).  The plain element gather with the same tiling measures the
  same for fp32, but a 1024-lane bf16 element tile was silently wrong, which is
  why the word form is used for both widths.

Everything else -- small copies that are launch-bound, every other step, odd
alignments -- stays on the generic kernels, which are the better path there
(a (64, 64) trailing-dim slice is 5.5 us generic against 8.6 us for the tiled
form) and keep the operator's full contract.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.slice_copy import (
    _INT32_MAX,
    _dtype_name,
    _native_copy_,
    _reinterpret_for_triton,
    _resolve_slice_bounds,
    _slice_view,
    slice_copy_kernel,
    slice_copy_kernel_inner1,
    slice_copy_kernel_step1,
)
from flag_gems.runtime.backend._kunlunxin.utils.tle_copy import tle_copy
from flag_gems.utils.shape_utils import MemOverlap, has_internal_overlapping

logger = logging.getLogger(__name__)

# The generic flat kernels already win below this size: the whole copy is one
# launch, so a tiled kernel only adds programs (measured 5.5 us vs 8.6 us on
# (64, 64)).  The DMA/word paths take over once the copy is bandwidth-bound.
_SMALL_NUMEL = 1 << 16

# Lanes per program for the word kernels.  Both widths peak here (a 256-lane
# tile costs 4-6x: the run, not the grid, is what the engine pays for).
_WORD_BLOCK = 1024

# The word kernel pays one program-scoped run per row, so it needs a row long
# enough to amortise that run: at >= 2 KB of output per row it wins (fp16
# (4096,4096) 261 us vs 618 us for the DSA path), below it the 2-D DMA wins
# (fp16 (1024,1024) 47.7 us vs 184 us).
_PAIR_MIN_ROW_BYTES = 2048

# A 2-D grid axis is capped; beyond it the flat generic kernel stays in charge.
_MAX_GRID_Y = 65535

_BLOCK_SIZE = 512  # generic kernel tile, mirrors flag_gems.ops.slice_copy

_I64 = torch.int64
_I32 = torch.int32


# ---------------------------------------------------------------------------
# Kernels: trailing-dim slice with step 2, read through an 8-byte word
# ---------------------------------------------------------------------------
@triton.jit
def _slice_copy_step2_word32(
    src64,
    dst32,
    sbase,
    srow,
    L,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """fp32 (and other 4-byte lanes): one output per lane.

    ``src64`` is the input viewed as int64, so lane ``i`` of row ``r`` reads
    the aligned word holding ``inp[r, start + 2*i]`` and
    ``inp[r, start + 2*i + 1]``; only the low 32 bits are stored.
    """
    row = tl.program_id(1).to(tl.int64)
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    base = sbase + row * srow
    if NEED_MASK:
        m = off < L
        w = tl.load(src64 + base + off, mask=m)
        tl.store(dst32 + row * L + off, w.to(tl.int32), mask=m)
    else:
        w = tl.load(src64 + base + off)
        tl.store(dst32 + row * L + off, w.to(tl.int32))


@triton.jit
def _slice_copy_step2_word16(
    src64,
    dst32,
    sbase,
    srow,
    L2,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """16-bit dtypes: two outputs per lane, packed into one int32 store.

    Lane ``i`` of row ``r`` reads one int64 word, which holds four 16-bit
    elements ``inp[r, start + 4*i : start + 4*i + 4]``; the two selected ones
    (positions 0 and 2) are packed into the two output halves.
    """
    row = tl.program_id(1).to(tl.int64)
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    base = sbase + row * srow
    if NEED_MASK:
        m = off < L2
        w = tl.load(src64 + base + off, mask=m)
        lo = (w & 0xFFFF).to(tl.int32)
        hi = ((w >> 32) & 0xFFFF).to(tl.int32)
        tl.store(dst32 + row * L2 + off, lo | (hi << 16), mask=m)
    else:
        w = tl.load(src64 + base + off)
        lo = (w & 0xFFFF).to(tl.int32)
        hi = ((w >> 32) & 0xFFFF).to(tl.int32)
        tl.store(dst32 + row * L2 + off, lo | (hi << 16))


def _largest_pow2_le(n: int) -> int:
    return 1 << (n.bit_length() - 1)


def _pair_word_plan(inp, out, dim_size, slice_len, start, outer):
    """Return the word-kernel arguments for a step-2 trailing-dim slice.

    ``None`` when the layout cannot be expressed as one 8-byte word per lane
    (odd base, odd row stride, unaligned pointer) or the copy is launch-bound.
    """
    if outer > _MAX_GRID_Y:
        return None
    esize = inp.element_size()
    if esize == 4:
        # one 4-byte element per int64 word
        if start % 2 or dim_size % 2:
            return None
        try:
            src64 = inp.view(_I64)
            dst32 = out.view(_I32)
        except RuntimeError:
            return None
        if src64.data_ptr() % 8 or dst32.data_ptr() % 4:
            return None
        return (
            _slice_copy_step2_word32,
            src64,
            dst32,
            start // 2,
            dim_size // 2,
            slice_len,
        )
    if esize == 2:
        # two 2-byte elements per int32 store word
        if start % 4 or dim_size % 4 or slice_len % 2:
            return None
        try:
            src64 = inp.view(_I64)
            dst32 = out.view(_I32)
        except RuntimeError:
            return None
        if src64.data_ptr() % 8 or dst32.data_ptr() % 4:
            return None
        return (
            _slice_copy_step2_word16,
            src64,
            dst32,
            start // 4,
            dim_size // 4,
            slice_len // 2,
        )
    return None


def _launch_pair_word(plan, outer):
    kernel, src64, dst32, sbase, srow, L_words = plan
    block = min(_largest_pow2_le(L_words), _WORD_BLOCK)
    need_mask = L_words % block != 0
    grid = (-(-L_words // block), outer)
    kernel[grid](
        src64,
        dst32,
        sbase,
        srow,
        L_words,
        BLOCK=block,
        NEED_MASK=need_mask,
        num_warps=8,
    )


# ---------------------------------------------------------------------------
# Host: same contract as the generic op, different launch
# ---------------------------------------------------------------------------
def _generic_launch(
    inp, out, inner, dim_size, start, slice_len, step, numel, int64_index
):
    """The generic kernels, used when a routing predicate does not hold."""
    grid = (-(-numel // _BLOCK_SIZE),)
    if inner == 1:
        slice_copy_kernel_inner1[grid](
            inp,
            out,
            numel,
            slice_len,
            dim_size,
            start,
            step,
            BLOCK_SIZE=_BLOCK_SIZE,
            INT64_INDEX=int64_index,
        )
    elif step == 1:
        slice_copy_kernel_step1[grid](
            inp,
            out,
            numel,
            inner,
            slice_len,
            dim_size,
            start,
            BLOCK_SIZE=_BLOCK_SIZE,
            INT64_INDEX=int64_index,
        )
    else:
        slice_copy_kernel[grid](
            inp,
            out,
            numel,
            inner,
            slice_len,
            dim_size,
            start,
            step,
            BLOCK_SIZE=_BLOCK_SIZE,
            INT64_INDEX=int64_index,
        )


def _launch_dma(inp, out, outer, slice_len, inner, dim_size, start, step):
    """Copy the ``(outer, slice_len, inner)`` window through the DSA row DMA.

    Both sides are unit-stride along ``inner``, which is what the 2-D DMA
    needs; ``step`` rides the source row stride.  Returns False when tle cannot
    express the copy, so the caller keeps the generic kernels.
    """
    src = inp.as_strided(
        (outer, slice_len, inner),
        (dim_size * inner, step * inner, 1),
        inp.storage_offset() + start * inner,
    )
    dst = out.as_strided(
        (outer, slice_len, inner),
        (slice_len * inner, inner, 1),
        out.storage_offset(),
    )
    return tle_copy(src, dst)


def _launch_slice_copy(inp, out, dim, start, slice_len, step):
    """Gather ``inp[..., start:start + slice_len * step:step, ...]`` into ``out``.

    ``out`` must be contiguous, of the sliced shape, and must not alias ``inp``.
    """
    # Same prep as the generic launcher: resolve lazy bits, then work on a
    # contiguous input so the flat index mapping holds.
    inp = inp.resolve_conj().resolve_neg()
    inp = inp.contiguous()

    outer = 1
    for d in range(dim):
        outer *= inp.size(d)
    inner = 1
    for d in range(dim + 1, inp.ndim):
        inner *= inp.size(d)
    dim_size = inp.size(dim)

    inp, out, inner = _reinterpret_for_triton(inp, out, inner)

    numel = out.numel()
    # Flat offsets are computed from the *input* extent, which can exceed the
    # int32 range even when the slice itself is small.  The DMA/word paths take
    # int32 scalars, so those tensors stay on the generic int64-index kernels.
    int64_index = max(numel, inp.numel()) > _INT32_MAX

    # inner > 1: any step is one 2-D DMA (the inner axis is unit-stride on both
    # sides), and it wins even on launch-bound sizes, so it is tried first.
    if inner > 1 and not int64_index:
        if _launch_dma(inp, out, outer, slice_len, inner, dim_size, start, step):
            return
        _generic_launch(
            inp, out, inner, dim_size, start, slice_len, step, numel, int64_index
        )
        return

    # inner == 1: below the crossover the whole copy is one launch and the
    # generic flat kernel is the cheaper program.
    if numel >= _SMALL_NUMEL and not int64_index:
        row_bytes = slice_len * inp.element_size()
        if step == 2 and row_bytes >= _PAIR_MIN_ROW_BYTES:
            plan = _pair_word_plan(inp, out, dim_size, slice_len, start, outer)
            if plan is not None:
                _launch_pair_word(plan, outer)
                return
        # Every other step (and short step-2 rows) is a strided innermost run,
        # which the DSA path reads per element: still far ahead of the generic
        # flat kernel's runtime div/mod gather.
        if _launch_dma(inp, out, outer, slice_len, 1, dim_size, start, step):
            return

    _generic_launch(
        inp, out, inner, dim_size, start, slice_len, step, numel, int64_index
    )


def _slice_copy_impl(inp, dim, start, end, step, out):
    ndim = inp.ndim
    # Validation order matches ATen: the shape/step diagnostics fire before the
    # out= dtype and device checks (measured on aten::slice_copy.Tensor_out).
    if ndim == 0:
        raise IndexError("slice() cannot be applied to a 0-dim tensor.")
    if dim < -ndim or dim >= ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-ndim}, {ndim - 1}], but got {dim})"
        )
    if step <= 0:
        # PyTorch only supports positive step for slice_copy.
        raise RuntimeError("slice step must be positive")

    dim = dim % ndim
    dim_size = inp.size(dim)
    start, end = _resolve_slice_bounds(dim_size, start, end)
    slice_len = max(0, (end - start + step - 1) // step)

    target_shape = list(inp.shape)
    target_shape[dim] = slice_len

    if out is None:
        out = torch.empty(target_shape, dtype=inp.dtype, device=inp.device)
    else:
        # ATen resizes a mismatched out before reporting a dtype/device
        # mismatch, so keep the resize first.
        if list(out.shape) != target_shape:
            out.resize_(target_shape)
        if out.dtype != inp.dtype:
            raise RuntimeError(
                f"Expected out tensor to have dtype {_dtype_name(inp.dtype)}, "
                f"but got {_dtype_name(out.dtype)} instead"
            )
        if out.device != inp.device:
            raise RuntimeError(
                f"Expected out tensor to have device {inp.device}, "
                f"but got {out.device} instead"
            )

    if out.numel() == 0:
        return out

    # ``out`` may be a strided view, may alias ``inp``, or may address the same
    # element twice; all three invalidate the flat-offset store, so the copy
    # goes through a contiguous temporary and the native strided copy_ places
    # it (an internally overlapping out then raises ATen's diagnostic).
    needs_staging = (
        not out.is_contiguous()
        or torch._C._is_alias_of(inp, out)
        or has_internal_overlapping(out) != MemOverlap.No
    )

    if inp.device.type != "cuda":
        view = _slice_view(inp, dim, start, slice_len, step)
        if needs_staging:
            view = view.clone(memory_format=torch.contiguous_format)
        _native_copy_(out, view)
        return out

    dst = (
        torch.empty(target_shape, dtype=out.dtype, device=out.device)
        if needs_staging
        else out
    )
    _launch_slice_copy(inp, dst, dim, start, slice_len, step)
    if needs_staging:
        _native_copy_(out, dst)
    return out


def slice_copy(inp, dim=0, start=None, end=None, step=1):
    logger.debug("GEMS_KUNLUNXIN SLICE_COPY")
    return _slice_copy_impl(inp, dim, start, end, step, out=None)


def slice_copy_out(inp, dim=0, start=None, end=None, step=1, *, out):
    logger.debug("GEMS_KUNLUNXIN SLICE_COPY_OUT")
    return _slice_copy_impl(inp, dim, start, end, step, out=out)
