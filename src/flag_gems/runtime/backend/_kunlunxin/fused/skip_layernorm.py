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
# per-row loop kernel that strides the normalized dim in BLOCK_SIZE chunks.
COL_CAP = 64 * 64  # 4096

# skip_layer_norm == layer_norm(x + residual). The default path launches one
# program per row (grid = (M,)), which is fast per row (1D reduce along a
# [BLOCK_SIZE] vector runs at the XPU memory ceiling). But launch latency
# (~0.6-0.9us/program) dominates when both N is small (each row moves few
# elements) and M is large (many programs): [100,65536,100] -> M=6.5M rows of
# N=100 spends ~5.9s almost entirely in launch. In that regime a 2D multi-row
# tile amortizes launch latency by having each program own TILE_M rows
# (grid = cdiv(M, TILE_M)). The 2D axis=1 reduce is slower per element, so it is
# a NET WIN only when BOTH: N is small enough that TILE_M is large (measured XPU
# crossover ~256; N<=256: 1.3-9x faster, N>=1024: 8-25x slower) AND M is large
# enough that per-row launch cost dominates (M<=~256 is not launch-bound, where
# the slower 2D reduce loses). Otherwise keep the per-row kernel.
MULTIROW_N = 256  # multi-row tile only pays off for normalized dims this small
MULTIROW_M = 4096  # ... and only once there are enough rows to be launch-bound
TILE_BUDGET = 8192  # rows * padded-cols per 2D tile; keeps the tile in SRAM


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
):
    pid = ext.program_id(0)
    Y += pid * y_stride_r
    X += pid * x_stride_r
    R += pid * r_stride_r

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + cols * x_stride_c, mask, other=0.0).to(tl.float32)
    r = tl.load(R + cols * r_stride_c, mask, other=0.0).to(tl.float32)

    x += r

    mean = tl.sum(x, axis=0) / N

    # Compute variance
    _var = tl.where(mask, x - mean, 0.0)
    _var = _var * _var
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    w = tl.load(W + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0).to(tl.float32)

    x_hat = (x - mean) * rstd
    y = w * x_hat + b
    y = y.to(Y.dtype.element_ty)
    tl.store(Y + cols * y_stride_c, y, mask=mask)


# --- Multi-row 2D-tile kernel (launch-bound huge-M / small-N only) ---------
# Each program owns a [TILE_M, N] tile (TILE_M consecutive rows, the whole
# normalized dim as ONE contiguous column block) and reduces along axis=1.
# grid = cdiv(M, TILE_M) so the launch count drops from M to M/TILE_M.
# N is passed as a constexpr and the columns span exactly [0, N) with NO
# power-of-2 padding: the whole tile is then one stride-1 contiguous block, so
# XPU OffsetAnalysis emits block DMA instead of the discrete access it falls
# back to when a padded TILE_N introduces a non-unit inner stride. Measured
# ~2-3x faster than the padded variant ([100,65536,100] 3301ms -> ~1130ms).
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
):
    pid = ext.program_id(0)

    n_off = tl.arange(0, N)
    w = tl.load(W + n_off).to(tl.float32)
    b = tl.load(B + n_off).to(tl.float32)

    m_off = pid * TILE_M + tl.arange(0, TILE_M)
    m_mask = m_off < M
    offs = m_off[:, None] * N + n_off[None, :]

    # Only rows are masked. Out-of-range rows load garbage (XPU ignores `other=`)
    # but their reduction is per-row (axis=1) and independent, and their store is
    # masked out below, so they never affect the valid rows or the output.
    x = tl.load(X + offs, mask=m_mask[:, None], other=0.0).to(tl.float32)
    r = tl.load(R + offs, mask=m_mask[:, None], other=0.0).to(tl.float32)
    x += r

    mean = tl.sum(x, axis=1) / N
    d = x - mean[:, None]
    var = tl.sum(d * d, axis=1) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    y = d * rstd[:, None] * w[None, :] + b[None, :]
    tl.store(Y + offs, y.to(Y.dtype.element_ty), mask=m_mask[:, None])


@libentry()
@triton.jit(do_not_specialize=["eps"])
def skip_layer_norm_kernel_tile(
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
    N: tl.constexpr,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    Y += pid * y_stride_r
    X += pid * x_stride_r
    R += pid * r_stride_r

    _sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        r = tl.load(R + cols, mask, other=0.0).to(tl.float32)
        x += r
        _sum += x

    mean = tl.sum(_sum) / N

    _var_base = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        r = tl.load(R + cols, mask, other=0.0).to(tl.float32)
        x += r
        _var = tl.where(mask, x - mean, 0.0)
        _var = _var * _var
        _var_base += _var

    var = tl.sum(_var_base, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        r = tl.load(R + cols, mask, other=0.0).to(tl.float32)
        x += r
        w = tl.load(W + cols, mask, other=0.0).to(tl.float32)
        b = tl.load(B + cols, mask, other=0.0).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = w * x_hat + b
        y = y.to(Y.dtype.element_ty)
        tl.store(Y + cols * y_stride_c, y, mask=mask)


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
            if N > COL_CAP:
                # Large-N per-row loop path.
                BLOCK_SIZE = builtins.min(COL_CAP, triton.next_power_of_2(N))
                skip_layer_norm_kernel_tile[M,](
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
                    isCloseUnrollControl=True,
                )
            elif N <= MULTIROW_N and M >= MULTIROW_M:
                # Small N + many rows: the per-row kernel is launch-bound, so
                # batch TILE_M rows per program to cut the grid from M to
                # cdiv(M, TILE_M). Columns span exactly N (no padding) so the
                # tile is one contiguous block -> block DMA.
                TILE_M = builtins.max(1, TILE_BUDGET // N)
                grid = (triton.cdiv(M, TILE_M),)
                skip_layer_norm_multirow_kernel[grid](
                    y, x, residual, weight, bias, M, eps, TILE_M, N
                )
            else:
                # Default fast path: one program per row, 1D reduce.
                BLOCK_SIZE = triton.next_power_of_2(N)
                skip_layer_norm_kernel[M,](
                    y, x, residual, weight, bias, N, 1, N, 1, N, 1, N, eps, BLOCK_SIZE
                )
        return y


def skip_layer_norm(x, residual, normalized_shape, weight, bias, eps=1e-5):
    return SkipLayerNorm.apply(x, residual, normalized_shape, weight, bias, eps)
