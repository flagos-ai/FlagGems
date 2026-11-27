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
from flag_gems.runtime.backend._ascend.tensorview import (
    HAS_TENSORVIEW,
    tensorview_call,
    tile_rows_for_ub,
)
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

from .sum import sum_dim

logger = logging.getLogger(__name__)

if HAS_TENSORVIEW:
    import triton.experimental.tle as tle_exp
    from triton.experimental.tle import tile_load, tile_store

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


# ---------------------------------------------------------------------------
# TensorView fast path (forward only)
#
# The generic kernel runs one program per row. Here a PartitionView hands a
# whole [BLOCK_M, WIDTH] tile to each program as one block DMA, so the row
# reduction happens over a tile instead of a single row, and the weight vector
# is fetched once per program through its own 1-D view.
#
# The backward pass is untouched: it needs per-row dx and a partial-sum
# reduction for dw that the existing kernels already express well.
# ---------------------------------------------------------------------------
TV_BLOCK_M = 32

if HAS_TENSORVIEW:

    @triton.jit(do_not_specialize=["eps"])
    def rms_norm_tv_kernel(
        y,
        inv_rms,
        x,
        weight,
        num_rows,
        eps,
        WIDTH: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_WIDTH: tl.constexpr,
    ):
        # The views do not depend on the tile index, so they are built once and
        # reused across the tiles this program walks.
        x_view = tle_exp.make_partition_view(
            x,
            shape=[num_rows, WIDTH],
            strides=[WIDTH, 1],
            tile=[BLOCK_M, BLOCK_WIDTH],
        )
        weight_view = tle_exp.make_partition_view(
            weight, shape=[WIDTH], strides=[1], tile=[BLOCK_WIDTH]
        )
        y_view = tle_exp.make_partition_view(
            y,
            shape=[num_rows, WIDTH],
            strides=[WIDTH, 1],
            tile=[BLOCK_M, BLOCK_WIDTH],
        )
        scale = tile_load(
            weight_view, index=(0,), dst_space=tle_exp.language.dsa.ascend.UB
        ).to(tl.float32)

        # The grid is capped at MAX_CORE_DIM like the generic kernel's, so each
        # program walks every BLOCK_M-row tile at its own stride. Without this
        # a row count above MAX_CORE_DIM * BLOCK_M overflows coreDim -- the same
        # failure the capped generic launch exists to avoid.
        num_blocks = tl.cdiv(num_rows, BLOCK_M)
        for row_block in range(tl.program_id(0), num_blocks, tl.num_programs(0)):
            rows = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
            tile = tile_load(
                x_view, index=(row_block, 0), dst_space=tle_exp.language.dsa.ascend.UB
            ).to(tl.float32)

            rrms = 1.0 / tl.sqrt(tl.sum(tile * tile, axis=1) / WIDTH + eps)
            normalized = tile * rrms[:, None] * scale[None, :]

            # The output rows are the same consecutive rows the input tile came
            # from, so a PartitionView addressed by the tile index covers them.
            # Rows past the end of the tensor fall outside the view and are
            # clipped by it, which handles the tail block without a mask.
            tile_store(
                y_view,
                value=normalized.to(y.dtype.element_ty),
                index=(row_block, 0),
                src_space=tle_exp.language.dsa.ascend.UB,
            )
            tl.store(inv_rms + rows, rrms, mask=rows < num_rows)

    def _tv_block_m(width):
        """Rows per tile: the tile and the normalized result are live as fp32."""
        return tile_rows_for_ub(triton.next_power_of_2(width) * 4, 2, TV_BLOCK_M)

    def _rms_norm_forward_tensorview(x, weight, M, N, eps, block_m=None):
        block_m = _tv_block_m(N) if block_m is None else block_m
        logger.debug("GEMS_ASCEND RMS_NORM_FORWARD(TENSORVIEW)")
        y = torch.empty_like(x)
        inv_rms = torch.empty((M,), device=x.device, dtype=torch.float32)
        grid = (min(triton.cdiv(M, block_m), MAX_CORE_DIM),)
        with torch_device_fn.device(x.device):
            rms_norm_tv_kernel[grid](
                y,
                inv_rms,
                x,
                weight,
                M,
                eps,
                WIDTH=N,
                BLOCK_M=block_m,
                BLOCK_WIDTH=triton.next_power_of_2(N),
            )
        return y, inv_rms

    def _rms_norm_tv_eligible(x, weight, normalized_shape, N):
        # The view is 2-D [rows, width], so only the last axis may be
        # normalized and the weight must cover exactly that width. A width that
        # is not a power of two makes the dense tile wider than the view, which
        # the padding path handles but costs more than it saves here.
        return (
            len(normalized_shape) == 1
            and x.numel() > 0
            and x.shape[-1] == N
            and weight.numel() == N
            and triton.next_power_of_2(N) == N
            # The win comes from batching rows into one tile. Once the row is
            # wide enough that only a single row fits in UB the fast path is
            # doing the same work as the per-row generic kernel -- measured at
            # 1.00x for width 8192 -- so there is nothing to gain.
            and _tv_block_m(N) > 1
        )


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

    x = x.contiguous()
    weight = weight.contiguous()
    y = torch.empty_like(x)
    inv_rms = torch.empty((M,), device=x.device, dtype=torch.float32)
    if M == 0 or N == 0:
        if N == 0:
            inv_rms = torch.full(
                (M,), float("nan"), device=x.device, dtype=torch.float32
            )
        return y, inv_rms

    BLOCK_SIZE = min(triton.next_power_of_2(N), 12064)

    def _generic_forward(x, weight, M, N, eps):
        with torch_device_fn.device(x.device):
            rms_norm_kernel[min(M, MAX_CORE_DIM),](
                y, inv_rms, x, weight, N, 1, N, 1, M, N, eps, BLOCK_SIZE
            )
        return y, inv_rms

    # The gate sits here rather than in RmsNorm.forward so the TransformerEngine
    # entry point described above takes the same path.
    if HAS_TENSORVIEW and _rms_norm_tv_eligible(x, weight, normalized_shape, N):
        return tensorview_call(
            "RMS_NORM_FORWARD",
            _rms_norm_forward_tensorview,
            _generic_forward,
            x,
            weight,
            M,
            N,
            eps,
        )
    return _generic_forward(x, weight, M, N, eps)


def rms_norm_backward(dy, x, inv_rms, normalized_shape, weight, eps=1e-5):
    """Backward of rms_norm_forward, returning (dx, dw)."""
    logger.debug("GEMS_ASCEND RMS_NORM_BACKWARD")
    dim = x.ndim - len(normalized_shape)
    M = math.prod(x.shape[:dim])
    N = math.prod(normalized_shape)

    x = x.contiguous()
    dy = dy.contiguous()
    weight = weight.contiguous()
    dx = torch.empty_like(x)
    if M == 0 or N == 0:
        return dx, torch.zeros_like(weight)

    BLOCK_SIZE = min(triton.next_power_of_2(N), 6912)

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
        dw = sum_dim(partial_buffer, dim=[0], dtype=weight.dtype).reshape(weight.shape)

    return dx, dw
