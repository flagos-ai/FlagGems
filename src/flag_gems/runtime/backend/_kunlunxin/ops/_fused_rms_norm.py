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

import builtins
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

from .rms_norm import (
    rms_norm_flat_kernel,
    rms_norm_kerne_tile,
    rms_norm_kernel,
    rms_norm_multirow_kernel,
    rms_norm_tile2d_kernel,
)

# Launch-bound / 2D-tile dispatch for the fused forward. Same structural fixes as
# rms_norm (constexpr-N unmasked [TILE_M, N] tile -> block DMA; flat kernel for
# N == 1; kerne_tile for N > 8192), kept local to this file so the shared
# rms_norm.py dispatch is untouched. Measured on XPU (2026-08-13, official
# do_bench): for tile2d the preferred TILE_M is 32, not 16 -- (1024,1024) tm32
# 63-74us vs tm16 71-80us, (4096,4096) tm32 520-693us vs tm16 558-732us for all
# three dtypes.
MULTIROW_N = 256
MULTIROW_M = 4096
TILE_BUDGET = 8192


def _fused_tile_m(N, M):
    """TILE_M for the unmasked 2D tile kernel, or None if not applicable."""
    if N <= 256:
        for cand in (32, 16):
            if M % cand == 0:
                return cand
        return None
    tm = 32  # measured faster than 16 on XPU for every M % 32 == 0 shape
    while tm * N > 65536 * 2:  # [TILE_M, N] fp32 tile SRAM budget (tm32 x N=4096 OK)
        tm //= 2
    while tm >= 2:
        if M % tm == 0:
            return tm
        tm //= 2
    return None


def _fused_rms_norm_forward(x, normalized_shape, weight, eps=1e-5):
    """Forward of fused rms_norm (returns y, inv_rms), Kunlunxin backend local.

    Dispatch mirrors rms_norm_forward but keeps the tile selection in this file
    (TILE_M=32 preference) without touching the shared rms_norm.py kernels.
    """
    dim = x.ndim - len(normalized_shape)
    M = math.prod(x.shape[:dim])
    N = math.prod(normalized_shape)

    BLOCK_SIZE = builtins.min(64 * 128, triton.next_power_of_2(N))

    x = x.contiguous()
    weight = weight.contiguous()
    # Native empty_strided: gems `empty` is intercepted and JIT-recompiles per
    # call on this XPU (~95-100ms/call, see rms_norm fix); empty_strided is not.
    y = torch.empty_strided(x.size(), x.stride(), dtype=x.dtype, device=x.device)
    inv_rms = torch.empty_strided((M,), (1,), dtype=torch.float32, device=x.device)

    with torch_device_fn.device(x.device):
        if N > 64 * 128:
            need_mask = (N % BLOCK_SIZE) != 0
            rms_norm_kerne_tile[M,](
                y,
                inv_rms,
                x,
                weight,
                N,
                1,
                N,
                1,
                M,
                N,
                eps,
                BLOCK_SIZE,
                need_mask,
            )
        elif N == 1:
            FLAT_BLOCK = 4096
            rms_norm_flat_kernel[(triton.cdiv(M, FLAT_BLOCK),)](
                y, inv_rms, x, weight, M, eps, FLAT_BLOCK
            )
        else:
            TILE_M = _fused_tile_m(N, M)
            if TILE_M is not None:
                rms_norm_tile2d_kernel[(M // TILE_M,)](
                    y, inv_rms, x, weight, eps, TILE_M, N
                )
            elif N <= MULTIROW_N and M >= MULTIROW_M:
                TILE_M = builtins.max(1, TILE_BUDGET // N)
                rms_norm_multirow_kernel[(triton.cdiv(M, TILE_M),)](
                    y, inv_rms, x, weight, M, eps, TILE_M, N
                )
            else:
                need_mask = (N % BLOCK_SIZE) != 0
                rms_norm_kernel[M,](
                    y,
                    inv_rms,
                    x,
                    weight,
                    N,
                    1,
                    N,
                    1,
                    M,
                    N,
                    eps,
                    BLOCK_SIZE,
                    need_mask,
                )
    return y, inv_rms


# --- Backward (aten::_fused_rms_norm_backward) --------------------------------
# The generic implementation is unusable on XPU3:
#   * dx kernel uses one program per row with BLOCK_SIZE = next_pow2(N); for
#     N >= 32768 the 1-D reduce is wrong (measured N=40999 -> abs err ~5e-3,
#     i.e. the row-sum constant is wrong), and BLOCK >= 32768 is a known
#     illegal-access regime.
#   * dw kernel does `tl.sum(x*dy*inv[:,None], axis=0)` on a computed 2-D tile,
#     which fails to legalize (`'arith.addf' op requires the same type ...`),
#     and then reduces the partial buffer with `torch.sum` (an ATen compute op /
#     forbidden fallback) over a (M, N) fp32 buffer (2.6 GB at 10000x65536).
# Backend-local replacement:
#   * dx: one program per row, chunked two-pass (chunk <= 8192 lanes, the proven
#     1-D reduce width). Pass 1 accumulates T = sum_j x*dy*w, pass 2 writes
#     dx = inv*dy*w - (inv^3*T/N)*x.
#   * dw: two-level column reduction via `tl.dot(ones, v)` (the verified XPU3
#     cross-lane reduction -- a 2-D axis-0 `tl.sum` is rejected, see
#     wiki/techniques/xpu3-dot-split-column-reduction.md). Each program owns a
#     row split + column block, accumulates [16, COL] with dot; a second kernel
#     folds the row splits with dot; a tiny cast kernel copies row 0 to dw.
#     A stride-0 masked store of the [16, COL] dot result is silently dropped
#     on XPU3, hence the explicit scratch + cast.
_DW_ROW = 64  # rows per dot chunk in the dw partial kernel
_DW_COL_MAX = 256  # column block width
_DW_TARGET_SPLITS = 256  # aim for this many programs across the column blocks


@triton.jit
def _fused_rms_norm_bwd_dx_kernel(
    X,
    DY,
    RSTD,
    W,
    DX,
    N: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    CHUNK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    row = tl.program_id(0)
    xp = X + row * N
    gp = DY + row * N
    dp = DX + row * N
    inv = tl.load(RSTD + row).to(tl.float32)

    # Pass 1: T = sum_j x * dy * w  (fp32, chunked over the normalized dim).
    acc = tl.zeros([CHUNK], dtype=tl.float32)
    for off in range(0, N, CHUNK):
        cols = off + tl.arange(0, CHUNK)
        if NEED_MASK:
            mask = cols < N
            x = tl.load(xp + cols, mask=mask, other=0.0).to(tl.float32)
            g = tl.load(gp + cols, mask=mask, other=0.0).to(tl.float32)
            if HAS_WEIGHT:
                w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
                g = g * w
        else:
            x = tl.load(xp + cols).to(tl.float32)
            g = tl.load(gp + cols).to(tl.float32)
            if HAS_WEIGHT:
                g = g * tl.load(W + cols).to(tl.float32)
        acc += x * g
    T = tl.sum(acc, axis=0)
    scale = inv * inv * inv * T / N

    # Pass 2: dx = inv*(dy*w) - scale*x.
    for off in range(0, N, CHUNK):
        cols = off + tl.arange(0, CHUNK)
        if NEED_MASK:
            mask = cols < N
            x = tl.load(xp + cols, mask=mask, other=0.0).to(tl.float32)
            g = tl.load(gp + cols, mask=mask, other=0.0).to(tl.float32)
            if HAS_WEIGHT:
                w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
                g = g * w
            dx = inv * g - scale * x
            tl.store(dp + cols, dx.to(DX.dtype.element_ty), mask=mask)
        else:
            x = tl.load(xp + cols).to(tl.float32)
            g = tl.load(gp + cols).to(tl.float32)
            if HAS_WEIGHT:
                g = g * tl.load(W + cols).to(tl.float32)
            dx = inv * g - scale * x
            tl.store(dp + cols, dx.to(DX.dtype.element_ty))


@triton.jit
def _fused_rms_norm_bwd_dx_multirow_kernel(
    X,
    DY,
    RSTD,
    W,
    DX,
    M,
    N: tl.constexpr,
    TILE_M: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    NEED_RMASK: tl.constexpr,
):
    # One-pass 2-D tile: [TILE_M, N] is loaded once, reduced along axis=1 and
    # written back from registers. Used only for launch-bound small-N/large-M
    # (a per-row kernel then launches M tiny programs -> 10-100x slower).
    pid = tl.program_id(0)
    n_off = tl.arange(0, N)
    w = (
        tl.load(W + n_off).to(tl.float32)
        if HAS_WEIGHT
        else tl.zeros([N], dtype=tl.float32)
    )
    m_off = pid * TILE_M + tl.arange(0, TILE_M)
    offs = m_off[:, None] * N + n_off[None, :]
    if NEED_RMASK:
        rmask = m_off < M
        x = tl.load(X + offs, mask=rmask[:, None], other=0.0).to(tl.float32)
        g = tl.load(DY + offs, mask=rmask[:, None], other=0.0).to(tl.float32)
        inv = tl.load(RSTD + m_off, mask=rmask, other=0.0).to(tl.float32)
    else:
        x = tl.load(X + offs).to(tl.float32)
        g = tl.load(DY + offs).to(tl.float32)
        inv = tl.load(RSTD + m_off).to(tl.float32)
    if HAS_WEIGHT:
        g = g * w[None, :]
    T = tl.sum(x * g, axis=1)
    scale = inv * inv * inv * T / N
    dxv = inv[:, None] * g - scale[:, None] * x
    if NEED_RMASK:
        tl.store(DX + offs, dxv.to(DX.dtype.element_ty), mask=rmask[:, None])
    else:
        tl.store(DX + offs, dxv.to(DX.dtype.element_ty))


@triton.jit
def _fused_rms_norm_bwd_dx_flat_kernel(
    X,
    DY,
    RSTD,
    W,
    DX,
    NUMEL,
    BLK: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
):
    # N == 1: rms-norm is elementwise; a flat [BLK] stream avoids M tiny programs.
    off = tl.program_id(0) * BLK + tl.arange(0, BLK)
    mask = off < NUMEL
    x = tl.load(X + off, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(DY + off, mask=mask, other=0.0).to(tl.float32)
    inv = tl.load(RSTD + off, mask=mask, other=0.0).to(tl.float32)
    if HAS_WEIGHT:
        g = g * tl.load(W).to(tl.float32)
    T = x * g
    dxv = inv * g - (inv * inv * inv * T) * x
    tl.store(DX + off, dxv.to(DX.dtype.element_ty), mask=mask)


@triton.jit
def _fused_rms_norm_bwd_dw_part_kernel(
    X,
    DY,
    RSTD,
    PART,
    M,
    N: tl.constexpr,
    BM: tl.constexpr,
    R: tl.constexpr,
    COL: tl.constexpr,
    NSTRIDE: tl.constexpr,
    NEED_CMASK: tl.constexpr,
):
    sp = tl.program_id(0)
    jb = tl.program_id(1)
    cols = jb * COL + tl.arange(0, COL)
    cmask = cols < N

    rbase = sp * BM
    nrow = tl.maximum(0, tl.minimum(BM, M - rbase))
    ones = tl.full((16, R), 1.0, dtype=tl.float32)
    acc = tl.zeros((16, COL), dtype=tl.float32)
    for r0 in range(0, nrow, R):
        rows = rbase + r0 + tl.arange(0, R)
        valid = rows < (rbase + nrow)
        if NEED_CMASK:
            mask = valid[:, None] & cmask[None, :]
        else:
            mask = valid[:, None]
        x = tl.load(X + rows[:, None] * N + cols[None, :], mask=mask, other=0.0).to(
            tl.float32
        )
        g = tl.load(DY + rows[:, None] * N + cols[None, :], mask=mask, other=0.0).to(
            tl.float32
        )
        inv = tl.load(RSTD + rows, mask=valid, other=0.0).to(tl.float32)
        acc += tl.dot(
            ones, x * g * inv[:, None], out_dtype=tl.float32, input_precision="ieee"
        )

    offs = (
        (sp * 16 + tl.arange(0, 16)[:, None]) * NSTRIDE
        + jb * COL
        + tl.arange(0, COL)[None, :]
    )
    if NEED_CMASK:
        tl.store(PART + offs, acc, mask=cmask[None, :])
    else:
        tl.store(PART + offs, acc)


@triton.jit
def _fused_rms_norm_bwd_dw_fin_kernel(
    PART,
    SCR,
    NSPLIT: tl.constexpr,
    NPAD: tl.constexpr,
    COL: tl.constexpr,
    NSTRIDE: tl.constexpr,
    NEED_CMASK: tl.constexpr,
    N: tl.constexpr,
):
    jb = tl.program_id(0)
    cols = jb * COL + tl.arange(0, COL)
    cs = tl.arange(0, NPAD)
    offs = (cs[:, None] * 16) * NSTRIDE + jb * COL + tl.arange(0, COL)[None, :]
    if NEED_CMASK:
        cmask = cols < N
        mask = (cs < NSPLIT)[:, None] & cmask[None, :]
    else:
        mask = (cs < NSPLIT)[:, None]
    p = tl.load(PART + offs, mask=mask, other=0.0)
    ones = tl.full((16, NPAD), 1.0, dtype=tl.float32)
    s = tl.dot(ones, p, out_dtype=tl.float32, input_precision="ieee")
    o2 = tl.arange(0, 16)[:, None] * NSTRIDE + jb * COL + tl.arange(0, COL)[None, :]
    if NEED_CMASK:
        tl.store(SCR + o2, s, mask=cmask[None, :])
    else:
        tl.store(SCR + o2, s)


@triton.jit
def _fused_rms_norm_bwd_dw_cast_kernel(SCR, DW, N, BLK: tl.constexpr):
    off = tl.program_id(0) * BLK + tl.arange(0, BLK)
    mask = off < N
    v = tl.load(SCR + off, mask=mask, other=0.0)
    tl.store(DW + off, v.to(DW.dtype.element_ty), mask=mask)


def _pick_dx_chunk(N):
    return builtins.max(64, builtins.min(8192, triton.next_power_of_2(N)))


def _pick_bwd_tile_m(M):
    """TILE_M for the one-pass multirow dx kernel.

    Measured (10000/4096 x {64,128,256} fp32): TILE_M=16 is best; TILE_M>=32
    regresses badly (2.3 ms vs 0.17 ms for 10000x256) and TILE_M==64 is an
    illegal-access regime, so never pick them here.
    """
    for cand in (16, 8):
        if M % cand == 0:
            return cand
    return None


def _pick_dw_plan(M, N):
    col = builtins.max(16, builtins.min(_DW_COL_MAX, triton.next_power_of_2(N)))
    ncol = triton.cdiv(N, col)
    nsplit = builtins.max(1, builtins.min(triton.cdiv(_DW_TARGET_SPLITS, ncol), 512, M))
    bm = triton.cdiv(M, nsplit)
    nsplit = triton.cdiv(M, bm)  # never launch an empty/rbase>=M split
    npad = builtins.max(16, triton.next_power_of_2(nsplit))
    return col, ncol, nsplit, bm, npad


def _fused_rms_norm_backward(
    grad_out, input, normalized_shape, rstd, weight=None, output_mask=(True, True)
):
    """Fused RMS normalization backward, Kunlunxin backend local.

    Matches ``aten::_fused_rms_norm_backward``:
    ``(grad_out, input, normalized_shape, rstd, weight?, output_mask) -> (dx, dw)``.
    """
    if len(output_mask) != 2:
        raise ValueError("output_mask must contain two booleans")
    if output_mask[1] and weight is None:
        raise RuntimeError("weight gradient requested without a weight tensor")

    dim = input.ndim - len(normalized_shape)
    M = math.prod(input.shape[:dim])
    N = math.prod(normalized_shape)

    x = input if input.is_contiguous() else input.contiguous()
    dy = grad_out if grad_out.is_contiguous() else grad_out.contiguous()
    rstd = rstd if rstd.is_contiguous() else rstd.contiguous()

    dx = None
    dw = None

    if M == 0 or N == 0:
        # Empty domain: dx is empty, sum over zero rows is exactly zero.
        if output_mask[0]:
            dx = torch.empty_like(x)
        if output_mask[1] and weight is not None:
            dw = torch.zeros(normalized_shape, dtype=x.dtype, device=x.device)
        return dx, dw

    has_weight = weight is not None
    if has_weight:
        weight = weight if weight.is_contiguous() else weight.contiguous()

    with torch_device_fn.device(x.device):
        if output_mask[0]:
            dx = torch.empty_like(x)
            wptr = weight if has_weight else x
            if N == 1:
                blk = 8192
                _fused_rms_norm_bwd_dx_flat_kernel[(triton.cdiv(M, blk),)](
                    x, dy, rstd, wptr, dx, M, BLK=blk, HAS_WEIGHT=has_weight
                )
            else:
                tile_m = (
                    _pick_bwd_tile_m(M)
                    if (N <= MULTIROW_N and M >= MULTIROW_M)
                    else None
                )
                if tile_m is not None:
                    _fused_rms_norm_bwd_dx_multirow_kernel[(M // tile_m,)](
                        x,
                        dy,
                        rstd,
                        wptr,
                        dx,
                        M,
                        N=N,
                        TILE_M=tile_m,
                        HAS_WEIGHT=has_weight,
                        NEED_RMASK=False,
                    )
                else:
                    chunk = _pick_dx_chunk(N)
                    need_mask = (N % chunk) != 0
                    _fused_rms_norm_bwd_dx_kernel[(M,)](
                        x,
                        dy,
                        rstd,
                        wptr,
                        dx,
                        N=N,
                        HAS_WEIGHT=has_weight,
                        CHUNK=chunk,
                        NEED_MASK=need_mask,
                    )

        if output_mask[1] and has_weight:
            col, ncol, nsplit, bm, npad = _pick_dw_plan(M, N)
            nstride = ncol * col
            need_cmask = (N % col) != 0
            part = torch.empty(
                (nsplit, 16, nstride), dtype=torch.float32, device=x.device
            )
            _fused_rms_norm_bwd_dw_part_kernel[(nsplit, ncol)](
                x,
                dy,
                rstd,
                part,
                M,
                N=N,
                BM=bm,
                R=_DW_ROW,
                COL=col,
                NSTRIDE=nstride,
                NEED_CMASK=need_cmask,
            )
            if nsplit == 1:
                # Row 0 of the single split is already the column sum.
                scr = part
            else:
                scr = torch.empty((16, nstride), dtype=torch.float32, device=x.device)
                _fused_rms_norm_bwd_dw_fin_kernel[(ncol,)](
                    part,
                    scr,
                    NSPLIT=nsplit,
                    NPAD=npad,
                    COL=col,
                    NSTRIDE=nstride,
                    NEED_CMASK=need_cmask,
                    N=N,
                )
            dw = torch.empty(N, dtype=x.dtype, device=x.device)
            blk = 4096
            _fused_rms_norm_bwd_dw_cast_kernel[(triton.cdiv(N, blk),)](
                scr, dw, N, BLK=blk
            )
            dw = dw.reshape(normalized_shape)

    return dx, dw


class _FusedRmsNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, eps):
        y, inv_rms = _fused_rms_norm_forward(x, normalized_shape, weight, eps)
        ctx.save_for_backward(x, inv_rms, weight)
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        return y, inv_rms

    @staticmethod
    def backward(ctx, dy, d_inv_rms):
        x, inv_rms, weight = ctx.saved_tensors
        dx, dw = _fused_rms_norm_backward(
            dy, x, ctx.normalized_shape, inv_rms, weight, (True, weight is not None)
        )
        return dx, None, dw, None


def _fused_rms_norm(x, normalized_shape, weight=None, eps=1e-5):
    if weight is not None:
        return _FusedRmsNorm.apply(x, normalized_shape, weight, eps)

    n = math.prod(normalized_shape)
    unit_weight = torch.ones(n, dtype=x.dtype, device=x.device)
    return _fused_rms_norm_forward(x, normalized_shape, unit_weight, eps)
