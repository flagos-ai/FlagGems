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
"""Kunlunxin (XPU) vendor override of cumulative_trapezoid.

`cumulative_trapezoid(y)` is the composite

    out[j] = sum_{k<=j} (y[k] + y[k+1]) * (dx/2),   j in [0, D-1)

whose generic implementation degrades to a *fully serial* per-(outer, inner)
loop on this backend (``_SERIAL_SCAN`` in ``ops/cumulative_trapezoid.py``):
the generic kernel feeds ``tl.cumsum`` a tile that is itself an elementwise
combination of two shifted loads, which this backend cannot legalize
(``triton_xpu.unpack ... without a bufPtr``).  Measured cost of that serial
kernel here: 5.8 ms for (64, 64), 2.2 s for (4096, 4096) fp16, and >20 s for
(10000, 65536) -- the last one trips the device launch timeout, so the old
`benchmark` baseline could not even be recorded.

Why the obvious rewrite is not enough
-------------------------------------
The output holds ``K = D-1`` elements per row while the input holds ``D``, so
the two row grids can never line up.  On XPU3 *any* address that is not
``scalar base + arange(BLOCK)`` is a per-lane gather/scatter and runs at
~2-4 ns/element instead of ~0.007 ns/element, which swamps everything else
(measured: the flat ``(i // K) * D + (i % K)`` form 5.7 ms vs a flat copy
0.12 ms on (4096, 4096) fp32, with ``i // K`` strength-reduced via a constexpr
``K`` or not).  Per-row programs keep every access affine, but each contiguous
run costs a fixed ~0.1 us, so a per-row kernel is ~10x off the copy floor as
soon as the row is narrower than the widest tile.

Implemented decomposition (three passes, all affine)
----------------------------------------------------
1. ``build`` -- one *flat* pass over the contiguous scan source:
   ``build[i] = (dx/2) * (y[i] + y[i+1])`` for ``i < N - 1``.  Both addresses are
   flat affine (the second is a +1 shift), so this runs at the block-DMA rate.
   ``build`` is a (rows, D) fp32 buffer; column ``D-1`` of every row is left
   untouched because the scan is causal and only columns ``[0, D-1)`` are
   consumed.
2. ``scan`` -- the backend's own Triton ``cumsum`` over the last axis of
   ``build``, into an fp32 (or fp64) accumulator buffer.  Scanning *pair terms*
   rather than ``y`` removes the per-row ``-y[0]`` correction, which would
   otherwise need a per-row scalar broadcast inside a flat kernel (a
   loaded-scalar broadcast miscompiles on this backend when it shares a kernel
   with a 2-D load).
3. ``compress`` -- ``s[..., :K]`` into ``out``.  The source is row-strided
   (``D``) and the destination row length is ``K``, i.e. exactly the
   "N contiguous runs into a strided destination" shape that
   ``tle_copy``'s 2-D SDNN descriptor moves at ~1.2 TB/s.  Small tensors (where
   tle's 12-40 us host setup dominates) and layouts tle cannot express fall
   back to a per-row Triton copy that clamps the tile start per *program* and
   lets the last tile of a row rewrite the elements the previous tile already
   wrote (idempotent, so unmasked and in-bounds).

Tensors small enough that the flat passes cannot fill a 4096-lane tile use a
two-pass per-row variant instead (scan ``y``, then one pair kernel per row), so
the launch count stays at two.

Vendor-private: no cross-backend impact.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from ..utils.tle_copy import tle_copy
from .cumsum import _scan_rows_into

logger = logging.getLogger(__name__)

_MAX_RANK = 8
# Widest per-row 1D tile: the widest tile the shipped scan uses, and enough for
# every per-row path here.
_BLOCK_MAX = 4096
# Widest tile for the *flat* passes.  They have no reduction and no per-row
# address, so they stay on the block-DMA path; 8192 lanes is the measured
# optimum (4096 -> 8192 halves the per-program dispatch cost, wider is flat).
_FLAT_BLOCK = 8192
# Below this many output elements the two-pass variant (scan, then one *flat*
# pair kernel over the output) is cheaper: the three-pass chain pays one extra
# launch + host setup, and the flat pair kernel's per-lane source address is
# only ~0.3 ns/element (measured crossover 130-150k outputs).
_FLAT_MIN_OUT = 1 << 17


@triton.jit
def _row_offsets(
    r,
    S0,
    SI0,
    SO0,
    S1,
    SI1,
    SO1,
    S2,
    SI2,
    SO2,
    S3,
    SI3,
    SO3,
    S4,
    SI4,
    SO4,
    S5,
    SI5,
    SO5,
    S6,
    SI6,
    SO6,
    S7,
    SI7,
    SO7,
):
    """Flat row id -> (source base, destination base) element offsets.

    Mixed-radix decomposition: `S0` varies fastest, and both bases come from
    the same digit tuple, so every logical row is visited exactly once with
    matching source/destination addresses (which ordering is irrelevant, only
    that the source and destination digits agree).
    """
    rem = r
    j = rem % S0
    rem = rem // S0
    b_in = j * SI0
    b_out = j * SO0
    j = rem % S1
    rem = rem // S1
    b_in += j * SI1
    b_out += j * SO1
    j = rem % S2
    rem = rem // S2
    b_in += j * SI2
    b_out += j * SO2
    j = rem % S3
    rem = rem // S3
    b_in += j * SI3
    b_out += j * SO3
    j = rem % S4
    rem = rem // S4
    b_in += j * SI4
    b_out += j * SO4
    j = rem % S5
    rem = rem // S5
    b_in += j * SI5
    b_out += j * SO5
    j = rem % S6
    rem = rem // S6
    b_in += j * SI6
    b_out += j * SO6
    j = rem % S7
    b_in += j * SI7
    b_out += j * SO7
    return b_in, b_out


@libentry()
@triton.jit
def _pair_flat_kernel(
    y_ptr,
    t_ptr,
    NE,
    scale,
    BLOCK: tl.constexpr,
):
    """t[i] = scale * (y[i] + y[i+1]) for the flat index range [0, NE).

    `NE` is the *last* element index (``rows * D - 1``), so ``i + 1`` never
    leaves the buffer.  Both the load pair and the store are flat affine: the
    tile start is clamped per program (a scalar) to keep the last tile inside
    the range, and it overlaps the previous tile, re-writing the same values.
    """
    pid = tl.program_id(0)
    i = tl.minimum(pid * BLOCK, NE - BLOCK) + tl.arange(0, BLOCK)
    lo = tl.load(y_ptr + i).to(tl.float32)
    hi = tl.load(y_ptr + i + 1).to(tl.float32)
    tl.store(t_ptr + i, (lo + hi) * scale)


@libentry()
@triton.jit
def _pair_flat_tiny_kernel(
    s_ptr,
    out_ptr,
    NO,
    scale,
    K: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Small-tensor pair pass written over the *flat output*.

    ``out[i] = (s[so] + s[so+1] - s[row*D]) * scale`` with ``row = i // K`` and
    ``so = row*D + i - row*K``.  ``K``/``D`` are ``tl.constexpr`` so the divide
    strength-reduces to a multiply-shift; with a runtime row width the address
    stream is a per-lane gather and runs ~100x slower.  Only used while the
    output fits a couple of tiles -- the per-lane row base load is exact but
    does not survive the gather wall at scale (the large path scans the pair
    terms instead, which needs no per-row base at all).
    """
    pid = tl.program_id(0)
    i = tl.minimum(pid * BLOCK, NO - BLOCK) + tl.arange(0, BLOCK)
    row = i // K
    so = row * D + (i - row * K)
    base = tl.load(s_ptr + row * D).to(tl.float32)
    lo = tl.load(s_ptr + so).to(tl.float32)
    hi = tl.load(s_ptr + so + 1).to(tl.float32)
    tl.store(out_ptr + i, (lo + hi - base) * scale)


@libentry()
@triton.jit
def _pair_shift_kernel(
    s_ptr,
    out_ptr,
    K,
    STEP_S,
    STEP_O,
    scale,
    S0,
    SI0,
    SO0,
    S1,
    SI1,
    SO1,
    S2,
    SI2,
    SO2,
    S3,
    SI3,
    SO3,
    S4,
    SI4,
    SO4,
    S5,
    SI5,
    SO5,
    S6,
    SI6,
    SO6,
    S7,
    SI7,
    SO7,
    BLOCK: tl.constexpr,
    OVERLAP: tl.constexpr,
):
    """out[row, j] = (s[row, j] + s[row, j+1] - s[row, 0]) * scale.

    Two-pass small-tensor path: with ``s = cumsum(y)`` and ``h = dx/2``,
    ``out[j] = h * sum_{k<=j}(y[k] + y[k+1]) = h * (s[j] + s[j+1] - s[0])``.
    Per-row tiles keep the accesses affine.  ``K = D-1`` is never a multiple of
    a power-of-two tile, so ``OVERLAP`` clamps the tile start per *program* to
    ``K - BLOCK`` and lets the last tile rewrite what the previous one wrote;
    the per-lane clamp needed when ``K < BLOCK`` (``OVERLAP=False``) turns the
    store into a data-dependent scatter (~8x slower per element, but this path
    only ever sees a few thousand elements).
    """
    row = tl.program_id(0)
    blk = tl.program_id(1)
    b_s, b_o = _row_offsets(
        row,
        S0,
        SI0,
        SO0,
        S1,
        SI1,
        SO1,
        S2,
        SI2,
        SO2,
        S3,
        SI3,
        SO3,
        S4,
        SI4,
        SO4,
        S5,
        SI5,
        SO5,
        S6,
        SI6,
        SO6,
        S7,
        SI7,
        SO7,
    )
    cols = tl.arange(0, BLOCK)
    if OVERLAP:
        start = tl.minimum(blk * BLOCK, K - BLOCK)
        j = start + cols
    else:
        j = tl.minimum(blk * BLOCK + cols, K - 1)
    base = tl.load(s_ptr + b_s).to(tl.float32)
    lo = tl.load(s_ptr + b_s + j * STEP_S).to(tl.float32)
    hi = tl.load(s_ptr + b_s + (j + 1) * STEP_S).to(tl.float32)
    tl.store(out_ptr + b_o + j * STEP_O, (lo + hi - base) * scale)


@libentry()
@triton.jit
def _copy_rows_kernel(
    s_ptr,
    out_ptr,
    K,
    STEP_S,
    STEP_O,
    S0,
    SI0,
    SO0,
    S1,
    SI1,
    SO1,
    S2,
    SI2,
    SO2,
    S3,
    SI3,
    SO3,
    S4,
    SI4,
    SO4,
    S5,
    SI5,
    SO5,
    S6,
    SI6,
    SO6,
    S7,
    SI7,
    SO7,
    BLOCK: tl.constexpr,
    OVERLAP: tl.constexpr,
):
    """out[row, :K] = s[row, :K] for the strided-output layouts tle_copy skips."""
    row = tl.program_id(0)
    blk = tl.program_id(1)
    b_s, b_o = _row_offsets(
        row,
        S0,
        SI0,
        SO0,
        S1,
        SI1,
        SO1,
        S2,
        SI2,
        SO2,
        S3,
        SI3,
        SO3,
        S4,
        SI4,
        SO4,
        S5,
        SI5,
        SO5,
        S6,
        SI6,
        SO6,
        S7,
        SI7,
        SO7,
    )
    cols = tl.arange(0, BLOCK)
    if OVERLAP:
        start = tl.minimum(blk * BLOCK, K - BLOCK)
        j = start + cols
    else:
        j = tl.minimum(blk * BLOCK + cols, K - 1)
    tl.store(out_ptr + b_o + j * STEP_O, tl.load(s_ptr + b_s + j * STEP_S))


def _merge_rows(shape, dims, s_strides, o_strides):
    """Merge adjacent non-integration dims that are contiguous in both buffers.

    Keeps the triad count (and therefore the kernel's scalar prologue) small
    for normal layouts.  Correctness does not depend on which groups are
    merged as long as ``s`` and ``out`` are merged together: a merged group's
    address is the same mixed-radix sum (i_hi * (size * stride) + i_lo * stride).
    """
    meta = []
    for k in dims:
        if (
            meta
            and meta[-1][1] == s_strides[k] * shape[k]
            and meta[-1][2] == o_strides[k] * shape[k]
        ):
            size = meta[-1][0]
            meta[-1] = (size * shape[k], s_strides[k], o_strides[k])
        else:
            meta.append((shape[k], s_strides[k], o_strides[k]))
    return meta


def _row_tile(K):
    """(BLOCK, OVERLAP) for the per-row kernels."""
    if K >= _BLOCK_MAX:
        return _BLOCK_MAX, 1
    if K >= 64:
        return 1 << (K.bit_length() - 1), 1
    return 64, 0


def _row_meta(shape, out_shape, d, D):
    if d == len(shape) - 1:
        # Fast path: every non-integration dim is contiguous in both the scan
        # buffer and the output, so they collapse into one triad.  Built
        # directly because this host-side work is on the critical path for the
        # launch-bound small shapes.
        n = 1
        for k, s in enumerate(shape):
            if k != d:
                n *= s
        return [n, D, out_shape[d]] + [1, 0, 0] * (_MAX_RANK - 1), 1
    dims = [k for k in range(len(shape)) if k != d]
    s_strides = {}
    acc = D
    for k in reversed(dims):
        s_strides[k] = acc
        acc *= shape[k]
    o_strides = {}
    acc = 1
    for k in reversed(range(len(shape))):
        o_strides[k] = acc
        acc *= out_shape[k]
    meta = _merge_rows(shape, dims, s_strides, o_strides)
    while len(meta) < _MAX_RANK:
        meta.append((1, 0, 0))
    flat = []
    for size, s_stride, o_stride in meta:
        flat.extend((size, s_stride, o_stride))
    return flat, o_strides[d]


_PLANS = {}


def _plan(shape, d):
    """Per-(shape, dim) constants; the wrapper is host-bound on small shapes."""
    key = (shape, d)
    p = _PLANS.get(key)
    if p is None:
        D = shape[d]
        K = D - 1 if D > 1 else 0
        out_shape = list(shape)
        out_shape[d] = K
        n = 1
        for k, s in enumerate(shape):
            if k != d:
                n *= s
        flat, step_out = _row_meta(shape, out_shape, d, D) if K else (None, 1)
        tile, overlap = _row_tile(K) if K else (64, 0)
        p = (D, K, out_shape, n, flat, step_out, tile, overlap)
        _PLANS[key] = p
    return p


def _scan_rows_last_dim(inp, acc_dtype):
    """Row-wise inclusive scan over the last (already contiguous) dimension.

    Equivalent to ``_cumsum(inp, dim=-1, dtype=acc_dtype)`` for a contiguous
    ``(rows..., D)`` tensor (so ``K == 1``).  ``allow_mm=False`` keeps the
    scan on the group path: the matmul path's multi-group (``N > G``) case used
    to be unbuildable (``TritonSDNNCombineBefore`` abort, ``out of resource:
    uni_sram``); it now compiles as a scan-then-fan split but arranges its
    segment-sum / prefix passes as one tiny program per (row, group), which is
    measured 5-24x slower than the group scan (see ``cumsum._MM_ALLOW_MULTIGROUP``).
    batch 17 already shipped this op on the group path at 0.379 dtype-equal, so
    this keeps that (validated) formulation rather than the slower matmul one.
    """
    out = torch.empty_like(inp, dtype=acc_dtype)
    n = inp.shape[-1]
    m = inp.numel() // n
    with torch_device_fn.device(inp.device):
        _scan_rows_into(inp, out, m, n, allow_mm=False)
    return out


def cumulative_trapezoid(y, *, dx=1, dim=-1):
    logger.debug("GEMS_KUNLUNXIN CUMULATIVE_TRAPEZOID")

    ndim = y.dim()
    if ndim == 0:
        raise IndexError("cumulative_trapezoid: dim out of range for a 0-dim tensor")
    d = dim + ndim if dim < 0 else dim
    if d < 0 or d >= ndim:
        raise IndexError(
            "Dimension out of range (expected to be in range of "
            "[{}, {}], but got {})".format(-ndim, ndim - 1, dim)
        )

    D, K, out_shape, nrows, flat, step_out, tile, overlap = _plan(
        tuple(y.shape), d
    )
    out = torch.empty(out_shape, dtype=y.dtype, device=y.device)
    if K == 0:
        return out
    if nrows == 0:
        return out

    # The scan source: integration axis last and contiguous.
    if d == ndim - 1:
        src = y.contiguous()
    else:
        src = y.movedim(d, -1).contiguous()

    # The accumulator is always wide: scanning into the input dtype would round
    # every prefix to fp16/bf16 (a whole ulp of a partial sum), and the
    # reference is the *float64* result rounded once -- those intermediate
    # roundings alone break the fp16 tolerance at cancellation points (measured
    # max abs err 9.8e-4 against 1e-4 allowed).
    acc_dtype = torch.float64 if y.dtype == torch.float64 else torch.float32
    scale = 0.5 * float(dx)
    n_elems = nrows * D

    with torch_device_fn.device(y.device):
        if out.numel() >= _FLAT_MIN_OUT:
            # 1) flat pair build, 2) prefix scan, 3) D -> D-1 compression.
            block = _FLAT_BLOCK
            while block > n_elems - 1:
                block //= 2
            build = torch.empty(nrows * D, dtype=acc_dtype, device=y.device)
            _pair_flat_kernel[(triton.cdiv(n_elems - 1, block),)](
                src, build, n_elems - 1, scale, BLOCK=block, num_warps=4
            )
            scanned = _scan_rows_last_dim(build.view(src.shape), acc_dtype)
            cut = scanned.narrow(-1, 0, K)
            if d == ndim - 1 and tle_copy(cut, out):
                return out
            _copy_rows_kernel[(nrows, triton.cdiv(K, tile))](
                scanned,
                out,
                K,
                1,
                step_out,
                *flat,
                BLOCK=tile,
                OVERLAP=overlap,
                num_warps=4,
            )
            return out

        # Small tensors: two passes (scan, then a pair kernel).
        scanned = _scan_rows_last_dim(src, acc_dtype)
        if d == ndim - 1 and out.numel() >= 64:
            block = 1 << (out.numel().bit_length() - 1)
            block = min(block, _BLOCK_MAX)
            _pair_flat_tiny_kernel[(triton.cdiv(out.numel(), block),)](
                scanned,
                out,
                out.numel(),
                scale,
                K=K,
                D=D,
                BLOCK=block,
                num_warps=4,
            )
            return out
        _pair_shift_kernel[(nrows, triton.cdiv(K, tile))](
            scanned,
            out,
            K,
            1,
            step_out,
            scale,
            *flat,
            BLOCK=tile,
            OVERLAP=overlap,
            num_warps=4,
        )
    return out
