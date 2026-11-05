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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# Ascend NPU coreDim (the launch grid) cannot exceed UINT16_MAX. Ops whose grid
# is derived from the row count must cap it and walk the remaining work inside
# the kernel, see triton-ascend docs migration_guide/migrate_from_gpu.md.
MAX_CORE_DIM = 65535


@libentry()
@triton.jit(do_not_specialize=["eps", "M"])
def rms_norm_kernel(
    Y,  # pointer to the output
    INV_RMS,  # pointer to inverse rms
    X,  # pointer to the input
    W,  # pointer to the weights
    y_stride_r,
    y_stride_c,
    x_stride_r,  # how much to increase the pointer when moving by 1 row
    x_stride_c,  # how much to increase the pointer when moving by 1 col
    M,  # number of rows in X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # The grid is capped at MAX_CORE_DIM, so each program handles several rows.
    for row in range(ext.program_id(0), M, ext.num_programs(0)):
        x_row = X + row * x_stride_r
        y_row = Y + row * y_stride_r

        var = 0.0
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(x_row + cols, mask, other=0.0).to(tl.float32)
            var += tl.sum(x * x / N)

        rrms = 1 / tl.sqrt(var + eps)

        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(x_row + cols, mask, other=0.0).to(tl.float32)
            w = tl.load(W + cols, mask, other=0.0)
            y = (x * rrms).to(Y.dtype.element_ty) * w
            tl.store(y_row + cols * y_stride_c, y, mask=mask)

        tl.store(INV_RMS + row, rrms)


@libentry()
@triton.jit(do_not_specialize=["eps", "M"])
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
    M,  # number of rows in X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # The grid is capped at MAX_CORE_DIM, so each program handles several rows.
    for row in range(ext.program_id(0), M, ext.num_programs(0)):
        dx_row = DX + row * dx_stride_r
        x_row = X + row * x_stride_r
        dy_row = DY + row * x_stride_r

        inv_rms = tl.load(INV_RMS + row).to(tl.float32)

        row_sum_stats = 0.0
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(x_row + cols, mask, other=0.0).to(tl.float32)
            dy = tl.load(dy_row + cols, mask, other=0.0).to(tl.float32)
            w = tl.load(W + cols, mask, other=0.0).to(tl.float32)
            dy = dy * w
            normalized_buf = x * inv_rms
            row_sum_stats += tl.sum(normalized_buf * dy)

        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(x_row + cols, mask, other=0.0).to(tl.float32)
            dy = tl.load(dy_row + cols, mask, other=0.0).to(tl.float32)
            w = tl.load(W + cols, mask, other=0.0).to(tl.float32)
            dy = dy * w
            normalized_buf = x * inv_rms
            norm_val = normalized_buf / N
            dx = (dy - norm_val * row_sum_stats) * inv_rms
            tl.store(dx_row + cols * dx_stride_c, dx, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["M", "N", "num_col_blocks"])
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
    num_col_blocks,  # number of col tiles per row tile
    ROW_BLOCK_SIZE: tl.constexpr,
    COL_BLOCK_SIZE: tl.constexpr,
):
    rows = tl.arange(0, ROW_BLOCK_SIZE)
    cols = tl.arange(0, COL_BLOCK_SIZE)

    # The (row tile, col tile) grid is flattened to 1D so that it can be capped
    # at MAX_CORE_DIM; each program then walks several tiles.
    num_tiles = tl.cdiv(M, ROW_BLOCK_SIZE) * num_col_blocks
    for tile in range(ext.program_id(0), num_tiles, ext.num_programs(0)):
        row_pid = tile // num_col_blocks
        col_pid = tile % num_col_blocks

        row_start = row_pid * ROW_BLOCK_SIZE
        col_start = col_pid * COL_BLOCK_SIZE

        # int64: the element offset of the last row tile easily exceeds int32.
        offset = row_start.to(tl.int64) * x_stride_r + col_start * x_stride_c

        row_mask = (row_start + rows) < M
        col_mask = (col_start + cols) < N

        x = tl.load(
            X + offset + rows[:, None] * x_stride_r + cols[None, :] * x_stride_c,
            row_mask[:, None] & col_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        inv_rms = tl.load(INV_RMS + row_start + rows, row_mask, other=0.0).to(
            tl.float32
        )
        dy = tl.load(
            DY + offset + rows[:, None] * x_stride_r + cols[None, :] * x_stride_c,
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


class RmsNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, eps=1e-5):
        logger.debug("GEMS_ASCEND LAYERNORM_FORWARD")
        y, inv_rms = rms_norm_forward(x, normalized_shape, weight, eps)

        ctx.save_for_backward(x, inv_rms, weight)
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        logger.debug("GEMS_ASCEND LAYERNORM_BACKWARD")
        x, inv_rms, weight = ctx.saved_tensors
        dx, dw = rms_norm_backward(
            dy, x, inv_rms, ctx.normalized_shape, weight, ctx.eps
        )
        return dx, None, dw, None


def rms_norm(x, normalized_shape, weight, eps=1e-5):
    return RmsNorm.apply(x, normalized_shape, weight, eps)


def rms_norm_forward(x, normalized_shape, weight, eps=1e-5):
    """rms_norm without autograd, returning (y, inv_rms).

    TransformerEngine's FlagOS backend calls this entry point directly, so it
    must carry the same capped launch as RmsNorm.forward or that path still
    overflows coreDim.
    """
    logger.debug("GEMS_ASCEND RMS_NORM_FORWARD")
    dim = x.ndim - len(normalized_shape)
    M = math.prod(x.shape[:dim])
    N = math.prod(normalized_shape)

    BLOCK_SIZE = min(triton.next_power_of_2(N), 12064)

    x = x.contiguous()
    weight = weight.contiguous()
    y = torch.empty_like(x)
    inv_rms = torch.empty((M,), device=x.device, dtype=torch.float32)

    with torch_device_fn.device(x.device):
        rms_norm_kernel[min(M, MAX_CORE_DIM),](
            y, inv_rms, x, weight, N, 1, N, 1, M, N, eps, BLOCK_SIZE
        )

    return y, inv_rms


def rms_norm_backward(dy, x, inv_rms, normalized_shape, weight, eps=1e-5):
    """Backward of rms_norm_forward, returning (dx, dw)."""
    logger.debug("GEMS_ASCEND RMS_NORM_BACKWARD")
    dim = x.ndim - len(normalized_shape)
    M = math.prod(x.shape[:dim])
    N = math.prod(normalized_shape)

    BLOCK_SIZE = min(triton.next_power_of_2(N), 6912)
    x = x.contiguous()
    dy = dy.contiguous()
    weight = weight.contiguous()
    dx = torch.empty_like(x)

    with torch_device_fn.device(x.device):
        rms_norm_grad_dx_kernel[min(M, MAX_CORE_DIM),](
            x, dy, inv_rms, dx, weight, N, 1, N, 1, M, N, eps, BLOCK_SIZE
        )

    ROW_BLOCK_SIZE = 16
    COL_BLOCK_SIZE = 256
    row_block_num = triton.cdiv(M, ROW_BLOCK_SIZE)
    col_block_num = triton.cdiv(N, COL_BLOCK_SIZE)

    partial_buffer = torch.empty(
        (row_block_num, N), dtype=torch.float32, device=x.device
    )

    num_tiles = row_block_num * col_block_num
    with torch_device_fn.device(x.device):
        rms_norm_grad_dw_kernel[min(num_tiles, MAX_CORE_DIM),](
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
            col_block_num,
            ROW_BLOCK_SIZE,
            COL_BLOCK_SIZE,
        )
        dw = torch.sum(partial_buffer, dim=0).to(x.dtype).reshape(-1)

    return dx, dw
