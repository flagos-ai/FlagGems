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

"""Kunlunxin (XPU) index_fill / index_fill_ override.

The generic implementation in ``flag_gems.ops.index_fill`` decomposes the fill
into a flat index ``m = outer * index_len + k`` and derives ``(outer, k)`` with
a per-lane ``//``/``%`` pair; on XPU those software-emulated divisions put the
row-fill (``dim < ndim-1``, inner_size > 1) path at ~2 GB/s.

This override specializes three cases:

1. ``inner_size == 1`` (``dim`` is the last dim): a 2-D grid scatter kernel
   whose two axes map to (index position, outer position); no division at all
   (kept from the original vendor implementation).
2. ``inner_size > 1`` (row/column fill, e.g. ``dim=0`` on a 2-D tensor): the
   output rows are contiguous runs, so each program handles ONE index value
   and writes ``BLOCK`` contiguous elements at a runtime base pointer
   (``out + norm*INNER + ib*BLOCK + arange``).  The pointer operand keeps the
   ``arange`` as the only vector term, which is what the XPU store path
   vectorizes (a ``runtime*INNER + arange`` folded vector degenerates to
   per-lane discrete stores, ~3 GB/s).  Measured 210-570 GB/s vs ~2 GB/s for
   the generic kernel on the benchmark matrix.
3. Full-permutation index (``index.numel() == dim_size``, all values in range
   and distinct): every position is filled exactly once, so the fill is a
   dense ``fill_`` -- the vendor engine fills 16M+ elements in ~0.1 ms while
   the Triton scatter takes 20 ms.

A host-side bounds check (two elementwise compares + one all-reduce) routes a
(rare) out-of-range index to the generic Triton implementation, which skips
such lanes silently (matching the historic behavior).  No native/CPU fallback
is used.
"""

import torch
import triton
import triton.language as tl

from flag_gems.ops.index_fill import (
    _FALLBACK_KEYSET,
    index_fill as _generic_index_fill,
    index_fill_ as _generic_index_fill_,
    _native_clone,
    _native_copy_,
    _prepare_index,
    _prepare_tensor_value,
)
from flag_gems.utils import libentry

_SCATTER_BLOCK_K = 128
_SCATTER_BLOCK_O = 8
# Fast row-fill tile: each program stores BLOCK contiguous elements
# (BLOCK * 4 bytes of a single row). 8192 lanes == 32 KB is the sweet spot
# measured on the XPU target; larger tiles start to hit uni_sram limits.
_ROW_BLOCK = 8192
# Burst-fill (inner <= 8, outer > 1): unroll BO rows per program.
_BURST_BO = 16
# The dense-fill shortcut needs an exact permutation check (~0.35 ms of
# scatter/reduce), which only pays off for large outputs; smaller ones keep
# the (already fast, ~parity) Triton scatter/row paths.
_DENSE_FILL_MIN_ELEMS = 1024 * 1024


@libentry()
@triton.jit
def index_fill_scatter_kernel(
    out,
    index,
    value,
    index_len,
    dim_size,
    outer_size,
    VALUE_IS_TENSOR: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_O: tl.constexpr,
):
    # Pure-scatter path (dim is the last dim, inner_size == 1): fill
    # out[o, index[k]] = value for all (o, k).  The two grid axes map
    # directly to (index position, outer position), so no integer division
    # is needed; BLOCK_K x BLOCK_O gives 1024 outstanding stores per program.
    pid_k = tl.program_id(0)
    pid_o = tl.program_id(1)
    k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    o = pid_o * BLOCK_O + tl.arange(0, BLOCK_O)
    k_mask = k < index_len
    o_mask = o < outer_size
    # Clamp the load address into the index buffer: XPU's masked load ignores
    # `other=` on out-of-bounds lanes (reads adjacent memory instead), so a
    # raw `index + k` would leak garbage into `raw_index` for k >= index_len.
    # The store mask below still discards those lanes; the clamp only keeps
    # the read itself in bounds.
    load_k = tl.where(k_mask, k, 0)
    raw_index = tl.load(index + load_k, mask=k_mask, other=0).to(tl.int64)
    valid_index = (raw_index >= -dim_size) & (raw_index < dim_size)
    normalized_index = tl.where(raw_index < 0, raw_index + dim_size, raw_index)
    offsets = o[:, None].to(tl.int64) * dim_size + normalized_index[None, :]
    store_mask = o_mask[:, None] & (k_mask[None, :] & valid_index[None, :])
    if VALUE_IS_TENSOR:
        fill_value = tl.load(value)
    else:
        fill_value = value
    # Out-of-range index entries are skipped silently by the store mask.
    # (PyTorch reports them as an error, but tl.device_assert fails to
    # compile on non-CUDA FlagGems backends, so the check is omitted.)
    tl.store(out + offsets, fill_value, mask=store_mask)


@libentry()
@triton.jit
def index_fill_row_kernel(
    out,
    index,
    value,
    dim_size,
    inner,
    INNER: tl.constexpr,
    NB: tl.constexpr,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
    VALUE_IS_TENSOR: tl.constexpr,
):
    # Row/column fill (inner_size > 1, outer_size == 1): one program per
    # (index position, inner block).  out.reshape(1, dim_size, INNER) is
    # filled at [0, index[k], ib*BLOCK:(ib+1)*BLOCK]; the store address is
    # a runtime scalar base plus `arange(BLOCK)` so the XPU store path can
    # vectorize the contiguous run.  The caller guarantees that every index
    # is in [-dim_size, dim_size), so no per-lane validity mask is needed
    # (a `& splat(valid)` mask would de-vectorize the store).
    pid = tl.program_id(0)
    k = pid // NB
    ib = pid % NB
    raw = tl.load(index + k).to(tl.int64)
    norm = tl.where(raw < 0, raw + dim_size, raw)
    if VALUE_IS_TENSOR:
        val = tl.load(value)
    else:
        val = value
    p = out + norm * INNER + ib * BLOCK
    if NEED_MASK:
        # Tail block only (inner % BLOCK != 0); full blocks keep the mask
        # all-true which the hardware folds away.
        tl.store(p + tl.arange(0, BLOCK), val, mask=tl.arange(0, BLOCK) < inner - ib * BLOCK)
    else:
        tl.store(p + tl.arange(0, BLOCK), val)


@libentry()
@triton.jit
def index_fill_burst_kernel(
    out,
    index,
    value,
    dim_size,
    inner,
    INNER: tl.constexpr,
    OSTRIDE: tl.constexpr,
    OBLK: tl.constexpr,
    BO: tl.constexpr,
    IPAD: tl.constexpr,
    VALUE_IS_TENSOR: tl.constexpr,
):
    # Small-inner burst fill (inner_size <= 8, outer_size > 1): fill
    # out[o, index[k], :] for a block of BO outer rows per program.  The
    # whole outer dimension is covered by OBLK blocks on grid axis 1
    # (decomposed from a 1-D grid); OSTRIDE = dim_size*inner is constexpr so
    # the per-row base stays a runtime*constexpr form.
    pid = tl.program_id(0)
    k = pid // OBLK
    ob = pid % OBLK
    raw = tl.load(index + k).to(tl.int64)
    norm = tl.where(raw < 0, raw + dim_size, raw)
    if VALUE_IS_TENSOR:
        val = tl.load(value)
    else:
        val = value
    base = out + norm * INNER + (ob * BO) * OSTRIDE
    for b in tl.static_range(BO):
        tl.store(
            base + b * OSTRIDE + tl.arange(0, IPAD),
            val,
            mask=tl.arange(0, IPAD) < inner,
        )


def _index_fill_scatter_launch(out, index, value, value_is_tensor, dim_size, outer_size):
    index_len = index.numel()
    grid = (
        triton.cdiv(index_len, _SCATTER_BLOCK_K),
        triton.cdiv(outer_size, _SCATTER_BLOCK_O),
    )
    index_fill_scatter_kernel[grid](
        out,
        index,
        value,
        index_len,
        dim_size,
        outer_size,
        VALUE_IS_TENSOR=value_is_tensor,
        BLOCK_K=_SCATTER_BLOCK_K,
        BLOCK_O=_SCATTER_BLOCK_O,
    )


def _index_fill_row_launch(out, index, value, value_is_tensor, dim_size, inner):
    index_len = index.numel()
    # The XPU store path widens an unmasked store to at least 64 lanes (the
    # warp width), so BLOCK must be >= 64 and the tail must stay masked.
    # 16-bit dtypes always take the masked store: the vectorized unmasked
    # store mangles the splat value on this backend (bf16 lanes get re-encoded
    # with a different magnitude), while the masked store is exact.
    block = min(_ROW_BLOCK, max(64, triton.next_power_of_2(inner)))
    nb = triton.cdiv(inner, block)
    grid = (index_len * nb,)
    index_fill_row_kernel[grid](
        out,
        index,
        value,
        dim_size,
        inner,
        INNER=inner,
        NB=nb,
        BLOCK=block,
        NEED_MASK=(inner % block != 0) or (out.dtype.itemsize < 4),
        VALUE_IS_TENSOR=value_is_tensor,
    )


def _index_fill_burst_launch(out, index, value, value_is_tensor, dim_size, outer, inner):
    index_len = index.numel()
    oblk = triton.cdiv(outer, _BURST_BO)
    grid = (index_len * oblk,)
    index_fill_burst_kernel[grid](
        out,
        index,
        value,
        dim_size,
        inner,
        INNER=inner,
        OSTRIDE=dim_size * inner,
        OBLK=oblk,
        BO=_BURST_BO,
        IPAD=triton.next_power_of_2(inner),
        VALUE_IS_TENSOR=value_is_tensor,
    )


def _native_fill_(out, value):
    # Plain fill_ (dispatches to the registered FlagGems fill` under
    # use_gems/enable, or to the vendor engine otherwise); either is a
    # correct dense fill and neither recurses back into index_fill.
    return out.fill_(value)


def _is_full_permutation(index, dim_size, in_range):
    # All index values are in [0, dim_size) (`in_range` with lo >= 0) and the
    # count equals dim_size; verify that every position is hit exactly once.
    # A dense fill_ is then semantically identical to index_fill.
    if index.numel() != dim_size or not in_range:
        return False
    seen = torch.zeros(dim_size, dtype=torch.bool, device=index.device)
    seen[index] = True
    return bool(seen.all())


def _index_fill_bounds_ok(index, dim_size, nonneg=False):
    # One round-trip check (two elementwise compares + one all-reduce)
    # instead of two separate min/max reductions with two syncs.
    lower = 0 if nonneg else -dim_size
    return bool(((index >= lower) & (index < dim_size)).all())


def _try_dense_fill(out, dim, index, value, value_is_tensor):
    """If `index` hits every position in the `dim` axis exactly once, fill all
    of `out` with `value` via the native fill engine and return True."""
    dim_size = out.size(dim)
    if index.numel() != dim_size or out.numel() < _DENSE_FILL_MIN_ELEMS:
        return False
    if not _index_fill_bounds_ok(index, dim_size, nonneg=True):
        return False
    seen = torch.zeros(dim_size, dtype=torch.bool, device=index.device)
    seen[index] = True
    if not bool(seen.all()):
        return False
    _native_fill_(out, value)
    return True


def _index_fill_impl(out, dim, index, value, value_is_tensor, check_dense=True, is_inplace=False):
    """Fill `out` in place. `out` is either the input (in-place op) or a
    fresh empty_like copy (functional op)."""
    if out.numel() == 0 or index.numel() == 0:
        return out
    if not out.is_contiguous():
        # All fast paths below assume a row-major layout; strided outputs go
        # through the generic (rank-aware) Triton implementation.
        if is_inplace:
            return _generic_index_fill_(out, dim, index, value)
        return _generic_index_fill(out, dim, index, value)

    dim_size = out.size(dim)
    # Full permutation: every position is filled exactly once -> dense fill.
    # (Must run before the dim-shape dispatch: it also applies to the last-dim
    # scatter case, where a full-permutation index makes the whole tensor a
    # dense fill -- the scatter kernel would be 20-40 ms vs ~0.4 ms.)
    if check_dense and _try_dense_fill(out, dim, index, value, value_is_tensor):
        return out

    inner = 1
    for i in range(dim + 1, out.ndim):
        inner *= out.shape[i]
    if inner == 1:
        # dim is the last dim: pure scatter (two-grid-axis kernel).
        outer = out.numel() // dim_size
        _index_fill_scatter_launch(out, index, value, value_is_tensor, dim_size, outer)
        return out

    # Host-side bounds check: only indices in [-dim_size, dim_size) are
    # accepted by the fast paths; anything else keeps the historic
    # "silently skip out-of-range lanes" semantics via the generic kernel.
    if not _index_fill_bounds_ok(index, dim_size):
        if is_inplace:
            return _generic_index_fill_(out, dim, index, value)
        return _generic_index_fill(out, dim, index, value)

    outer = out.numel() // (dim_size * inner)
    if outer == 1:
        _index_fill_row_launch(out, index, value, value_is_tensor, dim_size, inner)
    else:
        _index_fill_burst_launch(out, index, value, value_is_tensor, dim_size, outer, inner)
    return out


def index_fill(inp, dim, index, value):
    # Entry for both `index_fill.int_Scalar` and `index_fill.int_Tensor`: the
    # dispatcher routes by value type, so a 0-dimensional tensor value arrives
    # as a Tensor and a plain number as a Python scalar.
    dim, index = _prepare_index(inp, dim, index)
    if isinstance(value, torch.Tensor):
        value_is_tensor, value = _prepare_tensor_value(inp, value)
    else:
        value_is_tensor = False
    if inp.numel() == 0 or index.numel() == 0:
        return _native_clone(inp)
    if not inp.is_contiguous():
        # Strided input: the generic implementation clones, fills the clone
        # (which preserves the strided layout) and returns it.
        return _generic_index_fill(inp, dim, index, value)
    # Dense-fill (full permutation) must decide before the copy: the fill
    # overwrites everything, so the copy would be wasted work.
    out = torch.empty_like(inp)
    if _try_dense_fill(out, dim, index, value, value_is_tensor):
        return out
    _native_copy_(out, inp)
    _index_fill_impl(out, dim, index, value, value_is_tensor, check_dense=False)
    return out


def index_fill_(inp, dim, index, value):
    # Entry for both `index_fill_.int_Scalar` and `index_fill_.int_Tensor`.
    dim, index = _prepare_index(inp, dim, index)
    if isinstance(value, torch.Tensor):
        value_is_tensor, value = _prepare_tensor_value(inp, value)
    else:
        value_is_tensor = False
    if inp.numel() == 0 or index.numel() == 0:
        return inp
    _index_fill_impl(inp, dim, index, value, value_is_tensor, is_inplace=True)
    return inp