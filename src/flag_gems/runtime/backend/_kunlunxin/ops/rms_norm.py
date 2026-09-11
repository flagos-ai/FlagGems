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

# Launch-bound fix (same pattern as skip_layer_norm). The default forward kernel
# launches one program per row (grid=(M,)); each row's 1D reduce runs at the XPU
# memory ceiling, so per-row is optimal when N is large. BUT when N is small (each
# row moves few elements) AND M is large (many programs), per-program launch latency
# (~0.6-0.9us) dominates: [10000,256] launches 10000 programs -> speedup 0.006.
# In that regime a 2D multi-row tile (each program owns TILE_M rows, reduces along
# axis=1) cuts the grid from M to cdiv(M, TILE_M) and amortizes launch cost. The 2D
# axis=1 reduce is slower per element, so gate it strictly to the launch-bound corner:
# N small (TILE_M can be large) AND M large (actually launch-bound). Otherwise keep
# the per-row kernel.
MULTIROW_N = 256  # multi-row tile only pays off for normalized dims this small
MULTIROW_M = 4096  # ... and only once there are enough rows to be launch-bound
TILE_BUDGET = 8192  # rows * padded-cols per 2D tile; keeps the tile in SRAM


@libentry()
@triton.jit
def rms_norm_kernel(
    Y,  # pointer to the output
    INV_RMS,  # pointer to inverse rms
    X,  # pointer to the input
    W,  # pointer to the weights
    y_stride_r,
    y_stride_c,
    x_stride_r,  # how much to increase the pointer when moving by 1 row
    x_stride_c,  # how much to increase the pointer when moving by 1 col
    M: tl.constexpr,  # number of rows in X
    N: tl.constexpr,  # number of columns in X
    eps: tl.constexpr,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    Y += pid * y_stride_r
    X += pid * x_stride_r

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + cols * x_stride_c, mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=0) / N
    rrms = 1 / tl.sqrt(var + eps)

    w = tl.load(W + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    y = (x * rrms).to(Y.dtype.element_ty) * w
    tl.store(Y + cols * y_stride_c, y, mask=mask)
    tl.store(INV_RMS + pid, rrms)


# --- Multi-row 2D-tile forward kernel (launch-bound huge-M / small-N only) -------
# Each program owns a [TILE_M, N] tile (TILE_M consecutive rows, the whole
# normalized dim as ONE contiguous column block) and reduces along axis=1, so the
# launch count drops from M to cdiv(M, TILE_M). Also stores per-row INV_RMS for
# the backward pass. N is passed as a constexpr and the columns span exactly
# [0, N) with NO power-of-2 padding: the whole tile is then one stride-1
# contiguous block, so XPU OffsetAnalysis emits block DMA. If N were a runtime
# arg the `m_off * N` index math cannot be proven stride-1 -> discrete access
# (~2x slower); if TILE_N were padded to next_pow2(N) the non-unit inner stride
# also forces discrete access. Both cost ~2x (measured [10000,256] 4.9->2.4ms).
@libentry()
@triton.jit
def rms_norm_multirow_kernel(
    Y,  # output
    INV_RMS,  # per-row inverse rms
    X,  # input
    W,  # weight
    M,  # number of rows
    eps: tl.constexpr,
    TILE_M: tl.constexpr,
    N: tl.constexpr,  # number of columns (normalized dim), used as tile width
):
    pid = ext.program_id(0)

    n_off = tl.arange(0, N)
    w = tl.load(W + n_off).to(tl.float32)

    m_off = pid * TILE_M + tl.arange(0, TILE_M)
    m_mask = m_off < M
    offs = m_off[:, None] * N + n_off[None, :]

    # Only rows are masked. Out-of-range rows load garbage (XPU ignores `other=`)
    # but their reduction is per-row (axis=1) and independent, and their store is
    # masked out below, so they never affect the valid rows or the output.
    x = tl.load(X + offs, mask=m_mask[:, None], other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=1) / N
    rrms = 1.0 / tl.sqrt(var + eps)

    y = (x * rrms[:, None]).to(Y.dtype.element_ty) * w[None, :]
    tl.store(Y + offs, y.to(Y.dtype.element_ty), mask=m_mask[:, None])
    tl.store(INV_RMS + m_off, rrms, mask=m_mask)


@libentry()
@triton.jit
def rms_norm_kerne_tile(
    Y,  # pointer to the output
    INV_RMS,  # pointer to inverse rms
    X,  # pointer to the input
    W,  # pointer to the weights
    y_stride_r,
    y_stride_c,
    x_stride_r,  # how much to increase the pointer when moving by 1 row
    x_stride_c,  # how much to increase the pointer when moving by 1 col
    M: tl.constexpr,  # number of rows in X
    N: tl.constexpr,  # number of columns in X
    eps: tl.constexpr,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,  # whether N is not a multiple of BLOCK_SIZE
):
    pid = tl.program_id(0)
    Y += pid * y_stride_r
    X += pid * x_stride_r

    # NOTE (kunlunxin/XPU): when N is a multiple of BLOCK_SIZE (the common
    # transformer case, e.g. N=65536, BLOCK_SIZE=8192) every `cols < N` mask is
    # trivially all-true, but keeping the masked tl.load/tl.store still forces the
    # XPU slow masked-memory path.  Measured on this XPU: masked vs unmasked is
    # ~1.06x for fp32 but ~1.85x (fp16) / ~2.43x (bf16) slower, with byte-for-byte
    # identical output.  So we take an unmasked fast path when NEED_MASK is False
    # and fall back to the masked path (correctness) when N is not divisible.
    _var_base = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        if NEED_MASK:
            mask = cols < N
            x = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        else:
            x = tl.load(X + cols).to(tl.float32)
        _var_base += x * x / N
    var = tl.sum(_var_base)
    rrms = 1 / tl.sqrt(var + eps)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        if NEED_MASK:
            mask = cols < N
            x = tl.load(X + cols, mask, other=0.0).to(tl.float32)
            w = tl.load(W + cols, mask, other=0.0)
            y = (x * rrms).to(Y.dtype.element_ty) * w
            tl.store(Y + cols * y_stride_c, y, mask=mask)
        else:
            x = tl.load(X + cols).to(tl.float32)
            w = tl.load(W + cols)
            y = (x * rrms).to(Y.dtype.element_ty) * w
            tl.store(Y + cols * y_stride_c, y)

    tl.store(INV_RMS + pid, rrms)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_grad_dx_kernel(
    X,  # pointer to the input
    DY,
    INV_RMS,  # pointer to inverse rms
    DX,  # pointer to the output
    W,  # pointer to the weights
    dx_stride_r,
    dx_stride_c,
    x_stride_r,  # how much to increase the pointer when moving by 1 row
    x_stride_c,  # how much to increase the pointer when moving by 1 col
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    DX += pid * dx_stride_r
    X += pid * x_stride_r
    DY += pid * x_stride_r
    INV_RMS += pid

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + cols * x_stride_c, mask, other=0.0).to(tl.float32)
    inv_rms = tl.load(INV_RMS).to(tl.float32)
    dy = tl.load(DY + cols * x_stride_c, mask, other=0.0).to(tl.float32)
    w = tl.load(W + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)

    dy = dy * w

    normalized_buf = x * inv_rms
    row_sum_stats = tl.sum(normalized_buf * dy, axis=0)

    norm_val = normalized_buf / N
    dx = (dy - norm_val * row_sum_stats) * inv_rms

    tl.store(DX + cols * dx_stride_c, dx, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_grad_dx_kernel_tile(
    X,  # pointer to the input
    DY,
    INV_RMS,  # pointer to inverse rms
    DX,  # pointer to the output
    W,  # pointer to the weights
    dx_stride_r,
    dx_stride_c,
    x_stride_r,  # how much to increase the pointer when moving by 1 row
    x_stride_c,  # how much to increase the pointer when moving by 1 col
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    DX += pid * dx_stride_r
    X += pid * x_stride_r
    DY += pid * x_stride_r
    INV_RMS += pid

    # mask = tl.arange(0, BLOCK_SIZE) < N
    # cols = tl.arange(0, BLOCK_SIZE)
    # x = tl.load(X + cols * x_stride_c, mask, other=0.0).to(tl.float32)
    inv_rms = tl.load(INV_RMS).to(tl.float32)
    # dy = tl.load(DY + cols * x_stride_c, mask, other=0.0).to(tl.float32)
    # w = tl.load(W + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)

    # dy = dy * w

    # normalized_buf = x * inv_rms
    # row_sum_stats = tl.sum(normalized_buf * dy, axis=0)

    row_sum_stats_base = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + cols, mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask, other=0.0).to(tl.float32)

        dy = dy * w

        normalized_buf = x * inv_rms

        row_sum_stats_base += normalized_buf * dy
    row_sum_stats = tl.sum(row_sum_stats_base)

    # norm_val = normalized_buf / N
    # dx = (dy - norm_val * row_sum_stats) * inv_rms

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + cols, mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask, other=0.0).to(tl.float32)

        dy = dy * w

        normalized_buf = x * inv_rms
        norm_val = normalized_buf / N
        dx = (dy - norm_val * row_sum_stats) * inv_rms

        tl.store(DX + cols * dx_stride_c, dx, mask=mask)


@libentry()
@triton.jit
def rms_norm_grad_dw_kernel(
    X,  # pointer to the input
    DY,
    INV_RMS,  # pointer to inverse rms
    DW,  # pointer to the output
    dx_stride_r,
    dx_stride_c,
    x_stride_r,  # how much to increase the pointer when moving by 1 row
    x_stride_c,  # how much to increase the pointer when moving by 1 col
    M,  # number of rows in X
    N,  # number of columns in X
    ROW_BLOCK_SIZE: tl.constexpr,
    COL_BLOCK_SIZE: tl.constexpr,
):
    row_pid = tl.program_id(0)
    col_pid = tl.program_id(1)

    row_start = row_pid * ROW_BLOCK_SIZE
    col_start = col_pid * COL_BLOCK_SIZE

    offset = row_start * x_stride_r + col_start * x_stride_c
    X += offset
    DY += offset
    INV_RMS += row_start

    rows = tl.arange(0, ROW_BLOCK_SIZE)
    cols = tl.arange(0, COL_BLOCK_SIZE)

    row_mask = (row_start + rows) < M
    col_mask = (col_start + cols) < N

    x = tl.load(
        X + rows[:, None] * x_stride_r + cols[None, :] * x_stride_c,
        row_mask[:, None] & col_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    inv_rms = tl.load(INV_RMS + rows, row_mask, other=0.0).to(tl.float32)
    dy = tl.load(
        DY + rows[:, None] * x_stride_r + cols[None, :] * x_stride_c,
        row_mask[:, None] & col_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    d_weight = x * dy * inv_rms[:, None]
    partial_dweight_sum = tl.sum(d_weight, axis=0)

    tl.store(
        DW + row_pid * N + col_start + cols,
        partial_dweight_sum,
        mask=col_mask,
    )


@libentry()
@triton.jit
def rms_norm_grad_kernel(
    X,
    DY,
    DX,
    W,
    INV_RMS,
    DW,
    M: tl.constexpr,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x_ptr = X + row_idx * N + cols
    dy_ptr = DY + row_idx * N + cols
    w_ptr = W + cols

    x = tl.load(x_ptr, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(w_ptr, mask=mask, other=0.0).to(tl.float32)
    inv_rms = tl.load(INV_RMS + row_idx).to(tl.float32)

    dy_w = dy * weight
    x_inv_rms = x * inv_rms
    m_grad = tl.sum(dy_w * x, axis=0)
    dx = inv_rms * (dy_w - x_inv_rms * (m_grad / N))
    dx_ptr = DX + row_idx * N + cols
    tl.store(dx_ptr, dx, mask=mask)
    dw_partial = dy * x_inv_rms
    dw_ptr = DW + cols
    tl.store(dw_ptr, dw_partial, mask=mask)


def rms_norm_forward(x, normalized_shape, weight, eps=1e-5):
    logger.debug("GEMS_KUNLUNXIN RMS_NORM")
    dim = x.ndim - len(normalized_shape)
    M = math.prod(x.shape[:dim])
    N = math.prod(normalized_shape)

    # BLOCK_SIZE = triton.next_power_of_2(N)
    BLOCK_SIZE = builtins.min(
        64 * 128, triton.next_power_of_2(N)
    )  # core_num * buffer_size_limit

    x = x.contiguous()
    weight = weight.contiguous()
    y = torch.empty_like(x)
    inv_rms = torch.empty((M,), device=x.device, dtype=torch.float32)

    with torch_device_fn.device(x.device):
        if N > 64 * 128:
            need_mask = (N % BLOCK_SIZE) != 0
            rms_norm_kerne_tile[M,](
                y, inv_rms, x, weight, N, 1, N, 1, M, N, eps, BLOCK_SIZE, need_mask
            )
        elif N <= MULTIROW_N and M >= MULTIROW_M:
            # Small N + many rows: the per-row kernel is launch-bound, so batch
            # TILE_M rows per program to cut the grid from M to cdiv(M, TILE_M).
            # Columns span exactly N (no padding) so the tile is one contiguous
            # block -> block DMA. N is a constexpr for the same reason.
            TILE_M = builtins.max(1, TILE_BUDGET // N)
            grid = (triton.cdiv(M, TILE_M),)
            rms_norm_multirow_kernel[grid](y, inv_rms, x, weight, M, eps, TILE_M, N)
        else:
            rms_norm_kernel[M,](
                y, inv_rms, x, weight, N, 1, N, 1, M, N, eps, BLOCK_SIZE
            )

    return y, inv_rms


def rms_norm_backward(dy, x, inv_rms, normalized_shape, weight, eps=1e-5):
    logger.debug("GEMS_KUNLUNXIN RMS_NORM_BACKWARD")

    dim = x.ndim - len(normalized_shape)
    M = math.prod(x.shape[:dim])
    N = math.prod(normalized_shape)

    BLOCK_SIZE = triton.next_power_of_2(N)
    x = x.contiguous()
    dy = dy.contiguous()
    weight = weight.contiguous()
    dx = torch.empty_like(x)

    with torch_device_fn.device(x.device):
        if N > 64 * 128:
            BLOCK_SIZE = 8192
            rms_norm_grad_dx_kernel_tile[M,](
                x,
                dy,
                inv_rms,
                dx,
                weight,
                N,
                1,
                N,
                1,
                N,
                eps,
                BLOCK_SIZE,
                isCloseUnrollControl=True,
                isCloseVectorization=True,
            )
        else:
            rms_norm_grad_dx_kernel[M,](
                x,
                dy,
                inv_rms,
                dx,
                weight,
                N,
                1,
                N,
                1,
                N,
                eps,
                BLOCK_SIZE,
                isCloseUnrollControl=True,
            )

    ROW_BLOCK_SIZE = 1
    COL_BLOCK_SIZE = 256
    row_block_num = triton.cdiv(M, ROW_BLOCK_SIZE)
    col_block_num = triton.cdiv(N, COL_BLOCK_SIZE)

    partial_buffer = torch.empty(
        (row_block_num, N), dtype=torch.float32, device=x.device
    )

    with torch_device_fn.device(x.device):
        rms_norm_grad_dw_kernel[row_block_num, col_block_num](
            x,
            dy,
            inv_rms,
            partial_buffer,
            N,
            1,
            N,
            1,
            M,
            N,
            ROW_BLOCK_SIZE,
            COL_BLOCK_SIZE,
            isCloseUnrollControl=True,
            isCloseCoreTiling=True,
        )
        dw = torch.sum(partial_buffer, dim=0, dtype=x.dtype).reshape(-1)
    return dx, dw


def rms_norm_backward_fusion(dy, x, inv_rms, normalized_shape, weight, eps=1e-5):
    logger.debug("GEMS_KUNLUNXIN RMS_NORM_BACKWARD")

    dim = x.ndim - len(normalized_shape)
    M = math.prod(x.shape[:dim])  # Batch dimension
    N = math.prod(normalized_shape)  # Feature dimension

    x = x.contiguous()
    dy = dy.contiguous()
    weight = weight.contiguous()

    dx = torch.empty_like(x)
    dw = torch.empty_like(weight)

    BLOCK_SIZE = 64

    with torch_device_fn.device(x.device):
        rms_norm_grad_kernel[(M,)](
            x,
            dy,
            dx,
            weight,
            inv_rms,
            dw,
            M,
            N,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    return dx, dw


class RmsNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, eps=1e-5):
        y, inv_rms = rms_norm_forward(x, normalized_shape, weight, eps)
        ctx.save_for_backward(x, inv_rms, weight)
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        x, inv_rms, weight = ctx.saved_tensors
        normalized_shape = ctx.normalized_shape
        eps = ctx.eps

        dx, dw = rms_norm_backward(dy, x, inv_rms, normalized_shape, weight, eps)
        return dx, None, dw, None


def rms_norm(x, normalized_shape, weight, eps=1e-5):
    return RmsNorm.apply(x, normalized_shape, weight, eps)
