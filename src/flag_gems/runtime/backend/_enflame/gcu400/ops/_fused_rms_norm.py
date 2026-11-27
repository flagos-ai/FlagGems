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
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime  # noqa: F401
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

# gcu400: cap the per-row tile at 32768 elements. A row wider than this (e.g.
# REDUCTION_SHAPES (200, 40999, 3) -> N=40999 -> next_power_of_2 = 65536) makes
# the backward dx kernel exceed the DSM budget ("out of resource: dsm memory")
# at compile time, and large tiles can silently corrupt results on this
# hardware. Wide rows are therefore walked tile-by-tile, matching rms_norm.py.
MAX_BLOCK_SIZE = 32768


@libentry()
@triton.jit(do_not_specialize=["eps"])
def _fused_rms_norm_kernel(
    out_ptr,  # pointer to the output
    INV_RMS,  # pointer to inverse rms
    in_ptr,  # pointer to the input
    w_ptr,  # pointer to the weights
    y_stride_r: tl.constexpr,
    y_stride_c: tl.constexpr,
    x_stride_r: tl.constexpr,  # how much to increase the pointer when moving by 1 row
    x_stride_c: tl.constexpr,  # how much to increase the pointer when moving by 1 col
    N: tl.constexpr,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    out_ptr += pid * y_stride_r
    in_ptr += pid * x_stride_r

    cols = tl.arange(0, BLOCK_SIZE)

    # Wide rows are reduced tile-by-tile so the per-program DSM footprint stays
    # bounded (see MAX_BLOCK_SIZE). Masked (padding) lanes contribute zero, so
    # the accumulated variance matches the single-tile path. The variance is
    # always accumulated in float32 (matching the reference, which upcasts to
    # float32 before squaring), regardless of the input dtype.
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        c = off + cols
        mask = c < N
        x = tl.load(in_ptr + c * x_stride_c, mask, other=0.0).to(tl.float32)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rrms = 1 / tl.sqrt(var + eps)

    for off in range(0, N, BLOCK_SIZE):
        c = off + cols
        mask = c < N
        x = tl.load(in_ptr + c * x_stride_c, mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + c, mask=mask, other=0.0)
        y = (x * rrms).to(out_ptr.dtype.element_ty) * w
        tl.store(out_ptr + c * y_stride_c, y, mask=mask)
    tl.store(INV_RMS + pid, rrms)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def _fused_rms_norm_kernel_no_weight(
    out_ptr,  # pointer to the output
    INV_RMS,  # pointer to inverse rms
    in_ptr,  # pointer to the input
    y_stride_r: tl.constexpr,
    y_stride_c: tl.constexpr,
    x_stride_r: tl.constexpr,  # how much to increase the pointer when moving by 1 row
    x_stride_c: tl.constexpr,  # how much to increase the pointer when moving by 1 col
    N: tl.constexpr,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    out_ptr += pid * y_stride_r
    in_ptr += pid * x_stride_r

    cols = tl.arange(0, BLOCK_SIZE)

    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        c = off + cols
        mask = c < N
        x = tl.load(in_ptr + c * x_stride_c, mask, other=0.0).to(tl.float32)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rrms = 1 / tl.sqrt(var + eps)

    for off in range(0, N, BLOCK_SIZE):
        c = off + cols
        mask = c < N
        x = tl.load(in_ptr + c * x_stride_c, mask, other=0.0).to(tl.float32)
        y = (x * rrms).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + c * y_stride_c, y, mask=mask)
    tl.store(INV_RMS + pid, rrms)


# Narrow rows: one program per ROWS rows. A single row of 256 bf16 is only 512B,
# which is far below one DMA line, so the one-row-per-program launch measured
# 14 GB/s on (4096, 256). Folding ROWS rows into one [ROWS, BLOCK_N] tile costs
# the same DSM budget (ROWS * BLOCK_N <= MAX_BLOCK_SIZE fp32 accumulator) but
# gives every program a contiguous run of tens of KB.
_ROW_TILE_MAX_ELEMS = 32768
_ROW_TILE_MAX_ROWS = 128
# Above this row width the column loop already streams full DMA lines, so the
# single-row kernels (which also cover N > 4096) stay in charge.
_ROW_TILE_MAX_WIDTH = 4096


@libentry()
@triton.jit(do_not_specialize=["eps"])
def _fused_rms_norm_rows_kernel(
    out_ptr,  # pointer to the output
    INV_RMS,  # pointer to inverse rms
    in_ptr,  # pointer to the input
    w_ptr,  # pointer to the weights (unused when HAS_WEIGHT is False)
    M,  # number of rows
    y_stride_r: tl.constexpr,
    y_stride_c: tl.constexpr,
    x_stride_r: tl.constexpr,  # row stride of the input
    x_stride_c: tl.constexpr,  # column stride of the input
    N: tl.constexpr,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_N: tl.constexpr,
    ROWS: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
):
    pid = tle.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    row_mask = rows < M
    cols = tl.arange(0, BLOCK_N)
    col_mask = cols < N

    tile_mask = row_mask[:, None] & col_mask[None, :]
    x_base = in_ptr + rows[:, None] * x_stride_r + cols[None, :] * x_stride_c
    x = tl.load(x_base, tile_mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=1) / N
    rrms = 1 / tl.sqrt(var + eps)

    y = (x * rrms[:, None]).to(out_ptr.dtype.element_ty)
    if HAS_WEIGHT:
        w = tl.load(w_ptr + cols, mask=col_mask, other=0.0)
        y = y * w[None, :]
    tl.store(
        out_ptr + rows[:, None] * y_stride_r + cols[None, :] * y_stride_c,
        y,
        tile_mask,
    )
    tl.store(INV_RMS + rows, rrms, row_mask)


def _choose_fused_rms_norm_rows(n_cols):
    """Return ROWS (rows per program) for a narrow-row forward launch."""
    block_n = min(triton.next_power_of_2(n_cols), _ROW_TILE_MAX_WIDTH)
    rows = max(1, _ROW_TILE_MAX_ELEMS // block_n)
    return min(rows, _ROW_TILE_MAX_ROWS), block_n


def _fused_rms_norm_forward(x, normalized_shape, weight=None, eps=1e-5):
    """
    Fused RMS normalization forward pass.

    Args:
        x: Input tensor
        normalized_shape: Shape of the normalization dimensions (from the right)
        weight: Optional weight tensor for scaling
        eps: Epsilon for numerical stability

    Returns:
        tuple: (output, inv_rms) where inv_rms is 1/sqrt(mean(x^2) + eps)
    """
    logger.debug("GEMS_ENFLAME _FUSED_RMS_NORM_FORWARD")
    dim = x.ndim - len(normalized_shape)
    M = math.prod(x.shape[:dim])
    N = math.prod(normalized_shape)

    x = x.contiguous()
    y = torch.empty_like(x)
    inv_rms = torch.empty((M,), device=x.device, dtype=torch.float32)

    with torch_device_fn.device(x.device):
        if 0 < N <= _ROW_TILE_MAX_WIDTH and M > 1:
            rows, block_n = _choose_fused_rms_norm_rows(N)
            if weight is not None:
                weight = weight.contiguous()
            _fused_rms_norm_rows_kernel[(triton.cdiv(M, rows),)](
                y,
                inv_rms,
                x,
                weight if weight is not None else x,
                M,
                N,
                1,
                N,
                1,
                N,
                eps,
                BLOCK_N=block_n,
                ROWS=rows,
                HAS_WEIGHT=weight is not None,
            )
        else:
            BLOCK_SIZE = min(triton.next_power_of_2(N), MAX_BLOCK_SIZE)
            if weight is not None:
                weight = weight.contiguous()
                _fused_rms_norm_kernel[M,](
                    y, inv_rms, x, weight, N, 1, N, 1, N, eps, BLOCK_SIZE
                )
            else:
                _fused_rms_norm_kernel_no_weight[M,](
                    y, inv_rms, x, N, 1, N, 1, N, eps, BLOCK_SIZE
                )

    return y, inv_rms


# ---- backward launch banding (calibrated on GCU400, job j20261117175525) ----
# Persistent 1D grid (<= 48 CTAs) with a row-batch stride loop: immune to the
# grid.x>65535 / grid.y>255 (P19) overflow the old 2-D dw grid hit at
# N>=65536, and bounds the dw partial buffer to [grid, N] <= 48*N fp32.
_BWD_TARGET_CTAS = 48
# Element budget for the [R, BLOCK] row-batch tile, and the single-tile
# threshold: one program keeps ~24B/elem of fp32 live set (x, dy, products),
# so R*BLOCK <= 16384 holds ~393KB of DSM per program and keeps two programs
# per SP (48-way). Doubling the tile halves occupancy (R=8/N=4096 measured
# 2x slower than R=4). Rows wider than this walk tiles instead.
_BWD_TILE_ELEM_MAX = 16384
_BWD_R_CAP = 16
# Wide (multi-tile-row) path, measured on (1024, 65536): 2-byte dtypes want
# BLOCK=16384 with 2 warps on the flat 1-D kernel (1022us vs 1194us at
# BLOCK=32768/nw=4, and BLOCK=32768/nw<=2 exceeds the DSM budget); fp32 keeps
# the R=2 2-D variant with BLOCK=8192/nw=1.
_BWD_WIDE_BLOCK_2B = 16384
_BWD_WIDE_BLOCK_4B = 8192
_BWD_WIDE_R_4B = 2
# The single-tile row-batch kernel wins with one warp on every probed shape
# (do_bench w1000/r100 median, 3-round min); more warps only add DSM pressure.
_BWD_NW_SINGLE = 1
_BWD_NW_WIDE_2B = 2
_BWD_NW_WIDE_4B = 1

# Scratch buffers for the backward pass, keyed by (device, role, shape).
# Each ``torch.empty`` measured ~2.1us of host time in vector_norm's reduction
# path (~16us per allocation here on the GCU allocator), and the backward used
# to allocate up to three of them per call. The dw partials / row sums / row
# tile partials are pure intra-launch scratch consumed within the same call, so
# caching one buffer per (device, shape) is safe: consecutive launches on one
# stream are ordered. Both the entry count and the cached bytes are capped so
# unusual shape sweeps cannot pin device memory.
_BWD_SCRATCH = {}
_BWD_SCRATCH_LIMIT = 64
_BWD_SCRATCH_BYTES = 0
_BWD_SCRATCH_BYTES_MAX = 32 << 20


def _bwd_scratch(device, role, shape):
    """Reused fp32 scratch buffer for the fused RMSNorm backward launches."""
    global _BWD_SCRATCH_BYTES
    key = (device.type, device.index, role, shape)
    buf = _BWD_SCRATCH.get(key)
    if buf is not None:
        return buf
    buf = torch.empty(shape, dtype=torch.float32, device=device)
    nbytes = 4 * math.prod(shape)
    if (
        len(_BWD_SCRATCH) < _BWD_SCRATCH_LIMIT
        and _BWD_SCRATCH_BYTES + nbytes <= _BWD_SCRATCH_BYTES_MAX
    ):
        _BWD_SCRATCH[key] = buf
        _BWD_SCRATCH_BYTES += nbytes
    return buf


@libentry()
@triton.jit
def _fused_rms_norm_grad_kernel(
    X,
    DY,
    INV_RMS,
    DX,
    W,
    DW_PART,
    ROW_SUM,
    M,
    N,
    BLOCK: tl.constexpr,
    R: tl.constexpr,
    SINGLE_TILE: tl.constexpr,
    HAS_W: tl.constexpr,
    COMPUTE_DX: tl.constexpr,
    COMPUTE_DW: tl.constexpr,
    WIDE_IDX: tl.constexpr,
):
    # Fused dx + dw-partial kernel, one persistent 1-D grid.
    #   dx[m,n] = (dy[m,n]*w[n] - x[m,n]*inv_rms[m]/N * row_sum[m]) * inv_rms[m]
    #   dw[n]   = sum_m dy[m,n] * x[m,n] * inv_rms[m]   (per-program partial)
    # Single-tile rows are computed in ONE pass (the loaded tile stays live
    # across the reduction and the dx store); wide rows walk tiles twice via
    # a ROW_SUM scratch because row_sum must be complete before dx.
    pid = tle.program_id(0)
    nprog = tl.num_programs(0)
    num_batches = tl.cdiv(M, R)
    cols = tl.arange(0, BLOCK)
    inv_n = 1.0 / N

    if SINGLE_TILE:
        cmask = cols < N
        if HAS_W:
            w = tl.load(W + cols, mask=cmask, other=0.0).to(tl.float32)
        dw_acc = tl.zeros([BLOCK], dtype=tl.float32)
        for batch in range(pid, num_batches, nprog):
            rows = batch * R + tl.arange(0, R)
            rmask = rows < M
            m2 = rmask[:, None] & cmask[None, :]
            if WIDE_IDX:
                off = rows.to(tl.int64)[:, None] * N + cols[None, :]
            else:
                off = rows[:, None] * N + cols[None, :]
            x = tl.load(X + off, m2, other=0.0).to(tl.float32)
            dy = tl.load(DY + off, m2, other=0.0).to(tl.float32)
            inv_rms = tl.load(INV_RMS + rows, rmask, other=0.0).to(tl.float32)
            a = x * inv_rms[:, None]
            if HAS_W:
                b = dy * w[None, :]
            else:
                b = dy
            row_sum = tl.sum(a * b, axis=1)
            if COMPUTE_DX:
                dx = (b - a * (row_sum * inv_n)[:, None]) * inv_rms[:, None]
                tl.store(DX + off, dx.to(DX.dtype.element_ty), mask=m2)
            if COMPUTE_DW:
                dw_acc += tl.sum(dy * a, axis=0)
        if COMPUTE_DW:
            tl.store(DW_PART + pid * N + cols, dw_acc, mask=cmask)
    else:
        num_tiles = tl.cdiv(N, BLOCK)
        if COMPUTE_DX:
            for batch in range(pid, num_batches, nprog):
                rows = batch * R + tl.arange(0, R)
                rmask = rows < M
                inv_rms = tl.load(INV_RMS + rows, rmask, other=0.0).to(tl.float32)
                rs = tl.zeros([R], dtype=tl.float32)
                for t in range(0, num_tiles):
                    c = t * BLOCK + cols
                    cmask = c < N
                    m2 = rmask[:, None] & cmask[None, :]
                    if WIDE_IDX:
                        off = rows.to(tl.int64)[:, None] * N + c[None, :]
                    else:
                        off = rows[:, None] * N + c[None, :]
                    x = tl.load(X + off, m2, other=0.0).to(tl.float32)
                    dy = tl.load(DY + off, m2, other=0.0).to(tl.float32)
                    if HAS_W:
                        w = tl.load(W + c, mask=cmask, other=0.0).to(tl.float32)
                        b = dy * w[None, :]
                    else:
                        b = dy
                    rs += tl.sum((x * inv_rms[:, None]) * b, axis=1)
                tl.store(ROW_SUM + rows, rs, mask=rmask)
        for t in range(0, num_tiles):
            c = t * BLOCK + cols
            cmask = c < N
            if HAS_W:
                w = tl.load(W + c, mask=cmask, other=0.0).to(tl.float32)
            dw_acc = tl.zeros([BLOCK], dtype=tl.float32)
            for batch in range(pid, num_batches, nprog):
                rows = batch * R + tl.arange(0, R)
                rmask = rows < M
                m2 = rmask[:, None] & cmask[None, :]
                if WIDE_IDX:
                    off = rows.to(tl.int64)[:, None] * N + c[None, :]
                else:
                    off = rows[:, None] * N + c[None, :]
                x = tl.load(X + off, m2, other=0.0).to(tl.float32)
                dy = tl.load(DY + off, m2, other=0.0).to(tl.float32)
                inv_rms = tl.load(INV_RMS + rows, rmask, other=0.0).to(tl.float32)
                a = x * inv_rms[:, None]
                if HAS_W:
                    b = dy * w[None, :]
                else:
                    b = dy
                if COMPUTE_DX:
                    rs = tl.load(ROW_SUM + rows, rmask, other=0.0)
                    dx = (b - a * (rs * inv_n)[:, None]) * inv_rms[:, None]
                    tl.store(DX + off, dx.to(DX.dtype.element_ty), mask=m2)
                if COMPUTE_DW:
                    dw_acc += tl.sum(dy * a, axis=0)
            if COMPUTE_DW:
                tl.store(DW_PART + pid * N + c, dw_acc, mask=cmask)


@libentry()
@triton.jit
def _fused_rms_norm_grad_kernel_1d(
    X,
    DY,
    INV_RMS,
    DX,
    W,
    DW_PART,
    ROW_SUM,
    RS_PART,
    M,
    N,
    BLOCK: tl.constexpr,
    HAS_W: tl.constexpr,
    COMPUTE_DX: tl.constexpr,
    COMPUTE_DW: tl.constexpr,
    WIDE_IDX: tl.constexpr,
):
    # Flat 1-D-tile variant of _fused_rms_norm_grad_kernel for R == 1 (one row
    # per batch step). Two structural wins over the old degenerate [1, BLOCK]
    # 2-D tile (measured on (1024, 65536) fp16, job j20261119074414):
    #   1. flat tiles + scalar inv_rms:          1012.5us -> 579us
    #   2. pass1 tile-outer order (w read once
    #      per tile instead of per (row, tile)):  579us   -> 481us
    # Pass 1 stores per-row tile partials in RS_PART[T, M]; a tiny merge loop
    # folds them into ROW_SUM[M] keeping the tile accumulation order. The
    # scalar inv is applied after the tile sum (mathematically inv*sum(x*dy*w)
    # == sum((x*inv)*(dy*w)) with one fewer per-element multiply). Pass 2 is
    # the plain dx + dw-partial loop. Wide fp32 rows stay on the R=2 2-D
    # kernel (measured 798us vs 815us kernel-only on the same shape).
    pid = tle.program_id(0)
    nprog = tl.num_programs(0)
    cols = tl.arange(0, BLOCK)
    num_tiles = tl.cdiv(N, BLOCK)
    inv_n = 1.0 / N
    if COMPUTE_DX:
        for t in range(0, num_tiles):
            c = t * BLOCK + cols
            cmask = c < N
            if HAS_W:
                w = tl.load(W + c, cmask, other=0.0).to(tl.float32)
            for batch in range(pid, M, nprog):
                if WIDE_IDX:
                    base = batch.to(tl.int64) * N
                else:
                    base = batch * N
                x = tl.load(X + base + c, cmask, other=0.0).to(tl.float32)
                dy = tl.load(DY + base + c, cmask, other=0.0).to(tl.float32)
                inv = tl.load(INV_RMS + batch).to(tl.float32)
                if HAS_W:
                    p = tl.sum(x * (dy * w), axis=0) * inv
                else:
                    p = tl.sum(x * dy, axis=0) * inv
                tl.store(RS_PART + t * M + batch, p)
        for batch in range(pid, M, nprog):
            rs = tl.zeros([1], dtype=tl.float32)
            for t in range(0, num_tiles):
                rs += tl.load(RS_PART + t * M + batch)
            tl.store(ROW_SUM + batch, tl.sum(rs, axis=0))
    for t in range(0, num_tiles):
        c = t * BLOCK + cols
        cmask = c < N
        if HAS_W:
            w = tl.load(W + c, cmask, other=0.0).to(tl.float32)
        dw_acc = tl.zeros([BLOCK], dtype=tl.float32)
        for batch in range(pid, M, nprog):
            if WIDE_IDX:
                base = batch.to(tl.int64) * N
            else:
                base = batch * N
            x = tl.load(X + base + c, cmask, other=0.0).to(tl.float32)
            dy = tl.load(DY + base + c, cmask, other=0.0).to(tl.float32)
            inv = tl.load(INV_RMS + batch).to(tl.float32)
            a = x * inv
            if HAS_W:
                dyw = dy * w
            else:
                dyw = dy
            if COMPUTE_DX:
                rs = tl.load(ROW_SUM + batch)
                dx = (dyw - a * (rs * inv_n)) * inv
                tl.store(DX + base + c, dx.to(DX.dtype.element_ty), mask=cmask)
            if COMPUTE_DW:
                dw_acc += dy * a
        if COMPUTE_DW:
            tl.store(DW_PART + pid * N + c, dw_acc, mask=cmask)


@libentry()
@triton.jit
def _fused_rms_norm_dw_reduce_kernel(
    DW_PART,
    DW,
    P,
    N,
    BLOCK: tl.constexpr,
    PB: tl.constexpr,
):
    # Collapse the [P, N] per-program partials into dw[N] (fp32 accumulate,
    # cast on store). Replaces the host-side aten ``torch.sum`` + ``.to()``
    # pair the old implementation used (host-function rule violation).
    pid = tl.program_id(0)
    cols = pid * BLOCK + tl.arange(0, BLOCK)
    cmask = cols < N
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for p0 in range(0, P, PB):
        pr = p0 + tl.arange(0, PB)
        pm = pr < P
        part = tl.load(
            DW_PART + pr[:, None] * N + cols[None, :],
            pm[:, None] & cmask[None, :],
            other=0.0,
        )
        acc += tl.sum(part, axis=0)
    tl.store(DW + cols, acc.to(DW.dtype.element_ty), mask=cmask)


def _fused_rms_norm_backward(
    grad_out, input, normalized_shape, rstd, weight=None, output_mask=(True, True)
):
    """
    Fused RMS normalization backward pass.

    Matches the ``aten::_fused_rms_norm_backward`` schema
    ``(Tensor grad_out, Tensor input, int[] normalized_shape, Tensor rstd,
    Tensor? weight, bool[2] output_mask) -> (Tensor, Tensor)``.

    dx and the dw partials are produced by one persistent fused kernel; a
    small second kernel collapses the [grid, N] partials (the old version
    launched a 2-D-grid dw kernel that overflowed grid.y (P19) for N >=
    65536 and then called host-side ``torch.sum`` + ``.to()``). When the grid
    collapses to a single program that program's dw partial already equals dw,
    so it stores into dw directly and the second launch is skipped.

    Rows processed one per batch step (wide 2-byte rows, and single-tile rows
    wider than 8192) take the flat 1-D kernel: pass 1 computes per-row tile
    partials with the weight tile loaded once per column tile, a merge loop
    folds them into the row sums, then pass 2 emits dx and the dw partials.
    Other shapes run the row-batch 2-D kernel (single pass for single-tile
    rows, two passes for wide fp32 rows).

    Args:
        grad_out: Gradient of the output.
        input: Input tensor from the forward pass.
        normalized_shape: Shape of the normalization dimensions.
        rstd: Reciprocal RMS (inverse RMS) from the forward pass.
        weight: Optional weight tensor.
        output_mask: Two booleans selecting which of ``(dx, dw)`` to compute.

    Returns:
        tuple: (dx, dw) gradients for input and weight. Each entry is ``None``
        when not requested by ``output_mask`` (``dw`` is also ``None`` when
        ``weight`` is ``None``).
    """
    logger.debug("GEMS_ENFLAME _FUSED_RMS_NORM_BACKWARD")

    if len(output_mask) != 2:
        raise ValueError("output_mask must contain two booleans")
    if output_mask[1] and weight is None:
        raise RuntimeError("weight gradient requested without a weight tensor")

    dim = input.ndim - len(normalized_shape)
    M = math.prod(input.shape[:dim])
    N = math.prod(normalized_shape)

    want_dx = bool(output_mask[0])
    want_dw = bool(output_mask[1]) and weight is not None
    has_w = weight is not None

    if M == 0 or N == 0:
        dx = torch.empty_like(input) if want_dx else None
        dw = (
            torch.zeros(tuple(normalized_shape), dtype=input.dtype, device=input.device)
            if want_dw
            else None
        )
        return dx, dw

    x = input.contiguous()
    dy = grad_out.contiguous()
    rstd = rstd.contiguous()
    if has_w:
        weight = weight.contiguous()

    two_byte = x.dtype in (torch.float16, torch.bfloat16)
    np2_n = triton.next_power_of_2(N)
    single_tile = np2_n <= _BWD_TILE_ELEM_MAX

    if single_tile:
        block = np2_n
        r = max(1, min(_BWD_R_CAP, _BWD_TILE_ELEM_MAX // block))
        num_warps = _BWD_NW_SINGLE
    else:
        # Wide 2-byte rows always take the flat 1-D kernel (one row per step);
        # fp32 keeps the calibrated R=2 2-D variant (the 1-D form measured
        # 815us vs 798us kernel-only on (1024, 65536) fp32).
        block = _BWD_WIDE_BLOCK_2B if two_byte else _BWD_WIDE_BLOCK_4B
        r = 1 if two_byte else _BWD_WIDE_R_4B
        num_warps = _BWD_NW_WIDE_2B if two_byte else _BWD_NW_WIDE_4B

    num_batches = triton.cdiv(M, r)
    grid = max(1, min(num_batches, _BWD_TARGET_CTAS))
    wide_idx = M * N >= 2**31
    # One row per batch step: flat 1-D tiles beat the degenerate [1, BLOCK]
    # 2-D tile (1012.5us -> 481us on (1024, 65536) fp16). Covers the wide
    # 2-byte band and the R=1 single-tile band (N in (8192, 16384]).
    use_1d = r == 1
    num_tiles = triton.cdiv(N, block)

    dx = torch.empty_like(x) if want_dx else x

    # grid == 1: one program owns every row, so its dw partial column *is* the
    # final dw. Storing straight into dw drops both the [1, N] partial buffer
    # and the reduce launch -- the only second launch on the single-program
    # path.
    single_program = grid == 1
    dw = None
    if want_dw:
        dw = torch.empty(tuple(normalized_shape), dtype=x.dtype, device=x.device).view(
            -1
        )

    if not want_dw:
        dw_part = x
    elif single_program:
        dw_part = dw
    else:
        dw_part = _bwd_scratch(x.device, "dw_part", (grid, N))
    row_sum = (
        _bwd_scratch(x.device, "row_sum", (M,))
        if (want_dx and (use_1d or not single_tile))
        else x
    )
    # Per-row tile partials consumed by the 1-D kernel's merge loop. Allocated
    # whenever pass 1 runs: the kernel writes RS_PART unconditionally inside
    # COMPUTE_DX, so a dummy tensor must never be passed there.
    rs_part = (
        _bwd_scratch(x.device, "rs_part", (num_tiles, M)) if (want_dx and use_1d) else x
    )

    with torch_device_fn.device(x.device):
        if use_1d:
            _fused_rms_norm_grad_kernel_1d[grid,](
                x,
                dy,
                rstd,
                dx,
                weight if has_w else x,
                dw_part,
                row_sum,
                rs_part,
                M,
                N,
                BLOCK=block,
                HAS_W=has_w,
                COMPUTE_DX=want_dx,
                COMPUTE_DW=want_dw,
                WIDE_IDX=wide_idx,
                num_warps=num_warps,
            )
        else:
            _fused_rms_norm_grad_kernel[grid,](
                x,
                dy,
                rstd,
                dx,
                weight if has_w else x,
                dw_part,
                row_sum,
                M,
                N,
                BLOCK=block,
                R=r,
                SINGLE_TILE=single_tile,
                HAS_W=has_w,
                COMPUTE_DX=want_dx,
                COMPUTE_DW=want_dw,
                WIDE_IDX=wide_idx,
                num_warps=num_warps,
            )

    if want_dw:
        if not single_program:
            rb = max(256, min(1024, triton.next_power_of_2(max(16, N // 8))))
            pb = max(2, triton.next_power_of_2(min(grid, 8)))
            with torch_device_fn.device(x.device):
                _fused_rms_norm_dw_reduce_kernel[triton.cdiv(N, rb),](
                    dw_part,
                    dw,
                    grid,
                    N,
                    BLOCK=rb,
                    PB=pb,
                    num_warps=4,
                )
        dw = dw.view(tuple(normalized_shape))

    return (dx if want_dx else None), dw


class _FusedRmsNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, eps=1e-5):
        y, inv_rms = _fused_rms_norm_forward(x, normalized_shape, weight, eps)
        ctx.save_for_backward(x, inv_rms, weight)
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        return y, inv_rms

    @staticmethod
    def backward(ctx, dy, d_inv_rms):
        x, inv_rms, weight = ctx.saved_tensors
        normalized_shape = ctx.normalized_shape

        dx, dw = _fused_rms_norm_backward(
            dy, x, normalized_shape, inv_rms, weight, (True, weight is not None)
        )
        return dx, None, dw, None


def _fused_rms_norm(x, normalized_shape, weight=None, eps=1e-5):
    """
    Fused RMS normalization that returns both output and inverse RMS.

    This is useful for efficient backward computation as the inverse RMS
    is computed during forward pass and reused in backward.

    Args:
        x: Input tensor of shape (*, normalized_shape)
        normalized_shape: List of integers specifying the shape to normalize over
        weight: Optional learnable weight tensor of shape normalized_shape
        eps: Small constant for numerical stability (default: 1e-5)

    Returns:
        tuple: (output, inv_rms) where:
            - output: Normalized tensor of same shape as input
            - inv_rms: Inverse RMS tensor of shape (M,) where
              M = prod(x.shape[:-len(normalized_shape)])
    """
    logger.debug("GEMS_ENFLAME _FUSED_RMS_NORM")
    assert x.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ), f"_fused_rms_norm does not support dtype {x.dtype}"
    # The aten schema declares eps as ``float?``; the dispatcher may forward
    # ``None`` when the caller omits it, so fall back to the PyTorch default.
    if eps is None:
        eps = 1e-5
    return _FusedRmsNorm.apply(x, list(normalized_shape), weight, eps)


def _patch_ops_namespace():
    """Point ``flag_gems.ops._fused_rms_norm_backward`` at this vendor copy.

    ``SpecOpRegistrar`` only rebinds the ``flag_gems`` top-level names to the
    arch-specific implementations; the ``flag_gems.ops`` package namespace
    keeps the generic public import. The generic backward crashes on this
    backend (dx tile exceeds the DSM budget and the dw 2-D grid overflows
    grid.y, P19) for wide rows, and consumers addressing the op through the
    ops package (e.g. ``benchmark/test_fused_rms_norm.py``'s candidate leg)
    must get the GCU implementation, matching what the top-level name and
    the dispatcher already resolve to.
    """
    try:
        import flag_gems.ops as _ops_pkg

        _ops_pkg._fused_rms_norm_backward = _fused_rms_norm_backward
    except Exception as error:
        logger.warning(
            "_fused_rms_norm_backward: flag_gems.ops rebind unavailable: %s", error
        )


# Module-level handles keep the Library (and its kernels) alive for the
# process; the registrations themselves must never break the flag_gems import.
# _patch_ops_namespace()
