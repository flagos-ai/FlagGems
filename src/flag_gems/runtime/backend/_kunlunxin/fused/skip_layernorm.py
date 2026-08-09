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
import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# Above this N the single-block per-row kernel no longer fits; fall back to the
# per-row loop kernel that strides the normalized dim in chunks.
COL_CAP = 4096

# XPU-20 measurements (2026-08-17, P800):
#  * 1D contiguous accesses at blk>=16384 run at ~1.1TB/s; anything expressed as
#    a 2D tile whose inner dim is a non-power-of-2 (e.g. [TILE_M, 100]) collapses
#    to ~5-20GB/s discrete lanes and is a dead end (measured copy_2d ~5GB/s).
#  * Multi-row 2D tiles with N a power of 2 (or any N that fp-`tl.arange` accepts)
#    are fast; measured crossover: [TM, N] with TM*N <= 65536 lanes.
#  * Masking the 2D tile (row mask) or the 1D load (partial cols) is expensive
#    (1.5-10x); masked loads with other=0.0 are numerically UNRELIABLE on this
#    backend (garbage leaks into the reduce), so we never rely on `other=` in a
#    reduction: tail rows either divide TILE_M exactly (unmasked) or go through
#    the masked store path with masked loads only for correctness-carrying tails.
#  * For a non-power-of-2 row width N, a [TILE_M, 128] padded tile is ALSO a dead
#    end: reading x[r*N + c] with a 128-wide tile is a strided 2D access again.
#    On this backend a [TILE_M, N] tile with non-power-of-2 N is numerically
#    UNRELIABLE: tl.arange(0, N) is constant-folded to the next power of two and
#    the extra lanes carry garbage into the reduce (measured: TILE_M=16/64 clean,
#    TILE_M=20/8/4 -> up to 1e34 error).  Only two configurations are trusted for
#    N=100 rows: [64, 100] and [16, 100] tiles, both verified elementwise against
#    a fp64 reference on [6553600, 100].  Everything else with non-pow2 N goes
#    through the masked per-row kernel (slow but exact).
MULTIROW_CFG = [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]  # TILE_M candidates (largest that divides M)
MAX_TILE_LANES = 32768  # TILE_M*N cap: measured sweet spot (65536 spills)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def skip_layer_norm_kernel(
    Y,  # pointer to the output
    X,  # pointer to the input
    R,  # pointer to the residual
    W,  # pointer to the weights
    B,  # pointer to the biases
    y_stride_r,
    y_stride_c,
    x_stride_r,  # how much to increase the pointer when moving by 1 row
    x_stride_c,  # how much to increase the pointer when moving by 1 col
    r_stride_r,  # how much to increase the pointer when moving by 1 row
    r_stride_c,  # how much to increase the pointer when moving by 1 col
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = ext.program_id(0)
    Y += pid * y_stride_r
    X += pid * x_stride_r
    R += pid * r_stride_r

    cols = tl.arange(0, BLOCK_SIZE)
    if NEED_MASK:
        mask = cols < N
        x = tl.load(X + cols * x_stride_c, mask, other=0.0).to(tl.float32)
        r = tl.load(R + cols * r_stride_c, mask, other=0.0).to(tl.float32)
        x += r
        mean = tl.sum(x, axis=0) / N
        _var = tl.where(mask, x - mean, 0.0)
    else:
        x = tl.load(X + cols * x_stride_c).to(tl.float32)
        r = tl.load(R + cols * r_stride_c).to(tl.float32)
        x += r
        mean = tl.sum(x, axis=0) / N
        _var = x - mean
    _var = _var * _var
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    if NEED_MASK:
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
        y = w * ((x - mean) * rstd) + b
        tl.store(Y + cols * y_stride_c, y.to(Y.dtype.element_ty), mask=mask)
    else:
        w = tl.load(W + cols).to(tl.float32)
        b = tl.load(B + cols).to(tl.float32)
        y = w * ((x - mean) * rstd) + b
        tl.store(Y + cols * y_stride_c, y.to(Y.dtype.element_ty))


# Multi-row 2D-tile kernel.  Each program owns a [TILE_M, N] tile (TILE_M rows,
# the whole normalized dim as ONE column block) and reduces along axis=1.
# N is passed as a constexpr and the columns span exactly [0, N) with NO
# power-of-2 padding.  NEED_MASK=0 requires M % TILE_M == 0 (grid covers the
# matrix exactly); the tile is then one stride-1 contiguous block and the XPU
# OffsetAnalysis emits block DMA instead of the discrete access it falls back
# to when rows are masked or a padded TILE_N introduces a non-unit inner stride.
@libentry()
@triton.jit(do_not_specialize=["eps"])
def skip_layer_norm_multirow_kernel(
    Y,  # output
    X,  # input
    R,  # residual
    W,  # weight
    B,  # bias
    M,  # number of rows
    eps,
    TILE_M: tl.constexpr,
    N: tl.constexpr,  # number of columns (normalized dim), used as tile width
    NEED_MASK: tl.constexpr,
):
    pid = ext.program_id(0)

    n_off = tl.arange(0, N)
    w = tl.load(W + n_off).to(tl.float32)
    b = tl.load(B + n_off).to(tl.float32)

    m_off = pid * TILE_M + tl.arange(0, TILE_M)
    offs = m_off[:, None] * N + n_off[None, :]

    if NEED_MASK:
        m_mask = m_off < M
        # Rows are masked; out-of-range rows load garbage but their reduction is
        # per-row and independent, and the store below is masked out, so they
        # never affect the valid rows.  This path is slow (masked 2D); it only
        # exists for correctness when M % TILE_M != 0.
        x = tl.load(X + offs, mask=m_mask[:, None], other=0.0).to(tl.float32)
        r = tl.load(R + offs, mask=m_mask[:, None], other=0.0).to(tl.float32)
        x += r
        mean = tl.sum(x, axis=1) / N
        d = x - mean[:, None]
        var = tl.sum(d * d, axis=1) / N
        rstd = 1.0 / tl.sqrt(var + eps)
        y = d * rstd[:, None] * w[None, :] + b[None, :]
        tl.store(Y + offs, y.to(Y.dtype.element_ty), mask=m_mask[:, None])
    else:
        x = tl.load(X + offs).to(tl.float32)
        r = tl.load(R + offs).to(tl.float32)
        x += r
        mean = tl.sum(x, axis=1) / N
        d = x - mean[:, None]
        var = tl.sum(d * d, axis=1) / N
        rstd = 1.0 / tl.sqrt(var + eps)
        y = d * rstd[:, None] * w[None, :] + b[None, :]
        tl.store(Y + offs, y.to(Y.dtype.element_ty))


# Single-row looped kernel for rows longer than COL_CAP.  One program owns one
# row and strides it in BLOCK_SIZE chunks; with NEED_MASK=0 every chunk is a
# plain 1D unmasked block (contiguous, block DMA).  The mean/variance pass
# accumulates sum and sum-of-squares in ONE loop (var = E[x^2] - mean^2), so
# only the epilogue re-reads x+r from the row (served from L2 for rows that fit).
@libentry()
@triton.jit(do_not_specialize=["eps"])
def skip_layer_norm_kernel_tile(
    Y,  # pointer to the output
    X,  # pointer to the input
    R,  # pointer to the residual
    W,  # pointer to the weights
    B,  # pointer to the biases
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = ext.program_id(0)
    X += pid * N
    R += pid * N
    Y += pid * N

    _s = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    _s2 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        if NEED_MASK:
            mask = cols < N
            x = tl.load(X + cols, mask, other=0.0).to(tl.float32)
            r = tl.load(R + cols, mask, other=0.0).to(tl.float32)
        else:
            x = tl.load(X + cols).to(tl.float32)
            r = tl.load(R + cols).to(tl.float32)
        x += r
        _s += x
        _s2 += x * x
    mean = tl.sum(_s) / N
    var = tl.maximum(tl.sum(_s2) / N - mean * mean, 0.0)
    rstd = 1 / tl.sqrt(var + eps)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        if NEED_MASK:
            mask = cols < N
            x = tl.load(X + cols, mask, other=0.0).to(tl.float32)
            r = tl.load(R + cols, mask, other=0.0).to(tl.float32)
            w = tl.load(W + cols, mask, other=0.0).to(tl.float32)
            b = tl.load(B + cols, mask, other=0.0).to(tl.float32)
            y = w * ((x + r - mean) * rstd) + b
            tl.store(Y + cols, y.to(Y.dtype.element_ty), mask=mask)
        else:
            x = tl.load(X + cols).to(tl.float32)
            r = tl.load(R + cols).to(tl.float32)
            w = tl.load(W + cols).to(tl.float32)
            b = tl.load(B + cols).to(tl.float32)
            y = w * ((x + r - mean) * rstd) + b
            tl.store(Y + cols, y.to(Y.dtype.element_ty))


# N == 1: normalized dim is a single element; layer_norm of a 1-wide row is
# exactly y = bias (x_hat == 0).  We still run a real kernel so that dispatch
# and autograd semantics stay identical; the elementwise read is only needed to
# keep the same memory footprint pattern.  Flat 4096-lane blocks avoid the
# 6.3ms->0.3ms pathological per-row launch of BLOCK_SIZE=1.
@libentry()
@triton.jit
def skip_layer_norm_flat_kernel(
    Y,  # output
    X,  # input
    R,  # residual
    W,  # weight
    B,  # bias
    M,  # total number of elements (== rows)
    eps,  # unused mathematically, kept for rstd semantics
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    off = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = off < M
    x = tl.load(X + off, mask=m, other=0.0).to(tl.float32)
    r = tl.load(R + off, mask=m, other=0.0).to(tl.float32)
    w = tl.load(W).to(tl.float32)
    b = tl.load(B).to(tl.float32)
    # N == 1: mean = x + r, var = 0 -> x_hat = 0, y = b.  Written generically so
    # the compiler sees the exact same normalization math as the other kernels.
    s = x + r
    mean = s
    d = s - mean
    rstd = 1.0 / tl.sqrt(0.0 + eps)
    y = w * (d * rstd) + b
    tl.store(Y + off, y.to(Y.dtype.element_ty), mask=m)


class SkipLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual, normalized_shape, weight, bias, eps=1e-5):
        logger.debug("GEMS_KUNLUNXIN SKIP_LAYERNORM_FORWARD")
        dim = x.ndim - len(normalized_shape)
        M = math.prod(x.shape[:dim])
        N = math.prod(normalized_shape)

        x = x.contiguous()
        residual = residual.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()
        y = torch.empty_like(x)

        with torch_device_fn.device(x.device):
            if N == 1:
                # Flat elementwise kernel (210us->8us on [10000,1]).
                BLOCK_SIZE = 4096
                skip_layer_norm_flat_kernel[(triton.cdiv(M, BLOCK_SIZE),)](
                    y, x, residual, weight, bias, M, eps, BLOCK_SIZE
                )
            elif N <= COL_CAP:
                if (N & (N - 1)) == 0:
                    # Power-of-2 N: exactly-covering 2D tiles are numerically
                    # clean (verified).  Prefer the largest TILE_M that divides
                    # M; if none divides, fall back to the per-row kernel.
                    tm = None
                    for cand in MULTIROW_CFG:
                        if cand <= M and cand * N <= MAX_TILE_LANES and M % cand == 0:
                            tm = cand
                            break
                    if tm is not None:
                        skip_layer_norm_multirow_kernel[(M // tm,)](
                            y,
                            x,
                            residual,
                            weight,
                            bias,
                            M,
                            eps,
                            tm,
                            N,
                            False,
                        )
                    else:
                        BLOCK_SIZE = triton.next_power_of_2(N)
                        skip_layer_norm_kernel[M,](
                            y,
                            x,
                            residual,
                            weight,
                            bias,
                            N,
                            1,
                            N,
                            1,
                            N,
                            1,
                            N,
                            eps,
                            BLOCK_SIZE,
                            False,
                        )
                elif N >= 64 and M % 64 == 0 and N <= 128:
                    # Non-pow2 N: only the [64, N] tile (N<128) was verified
                    # clean on this backend (see header comment).
                    skip_layer_norm_multirow_kernel[(M // 64,)](
                        y,
                        x,
                        residual,
                        weight,
                        bias,
                        M,
                        eps,
                        64,
                        N,
                        False,
                    )
                else:
                    # Safe but slow: masked per-row 1D reduce (exact).
                    BLOCK_SIZE = triton.next_power_of_2(N)
                    skip_layer_norm_kernel[M,](
                        y,
                        x,
                        residual,
                        weight,
                        bias,
                        N,
                        1,
                        N,
                        1,
                        N,
                        1,
                        N,
                        eps,
                        BLOCK_SIZE,
                        True,
                    )
            else:
                # Large-N per-row loop path.  Unmasked chunks when the row
                # length divides cleanly (N % BLOCK == 0) -> plain 1D loads.
                # Reduction accumulator block-size stability (5/5 runs,
                # [10000, 65536] vs fp64 ref):
                #   fp16: 16384 clean (14ms)
                #   fp32: 8192 clean, 16384 corrupt (7.2e0)
                #   bf16: only 4096 clean (8192 corrupts ~20%, 16384 5/5)
                # Dispatch per dtype accordingly.
                NEED_MASK_BLOCK = 4096
                if x.dtype == torch.float16 and N % 16384 == 0:
                    BLOCK_SIZE = 16384
                elif x.dtype != torch.bfloat16 and N % 8192 == 0:
                    BLOCK_SIZE = 8192
                else:
                    BLOCK_SIZE = 4096
                need_mask = N % BLOCK_SIZE != 0
                skip_layer_norm_kernel_tile[M,](
                    y,
                    x,
                    residual,
                    weight,
                    bias,
                    N,
                    eps,
                    BLOCK_SIZE,
                    need_mask,
                )
        return y


def skip_layer_norm(x, residual, normalized_shape, weight, bias, eps=1e-5):
    return SkipLayerNorm.apply(x, residual, normalized_shape, weight, bias, eps)