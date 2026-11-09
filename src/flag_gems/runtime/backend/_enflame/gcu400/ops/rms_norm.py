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

from ..utils.config_utils import MAX_GRID_DIM

logger = logging.getLogger(__name__)

MAX_BLOCK_SIZE = 32768
_FORWARD_WARPS = 4  # 前向 launch 的 warp 数，按 shape 扫描后固定


@triton.jit
def _rms_norm_rows(
    Y,
    INV_RMS,
    X,
    W,
    pid,
    y_stride_r: tl.constexpr,
    y_stride_c: tl.constexpr,
    x_stride_r: tl.constexpr,
    x_stride_c: tl.constexpr,
    M,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
    ROWS: tl.constexpr,
    SINGLE_BLOCK: tl.constexpr,
    SAVE_INV_RMS: tl.constexpr,
):
    """一个 program 处理 ROWS 行：单块路径下 w 只加载一次供所有行复用。

    每行仍用标量 tl.sum(axis=0) 归约（与 1D kernel 完全等价），避免 2D
    tl.sum(axis=1) 在 GCU 后端上的归约语义问题（实测 1024² 会产生整行错值）。
    """
    cols = tl.arange(0, BLOCK_SIZE)
    cmask = cols < N
    if SINGLE_BLOCK:
        w_shared = tl.load(W + cols, mask=cmask, other=0.0)
    else:
        w_shared = tl.full((BLOCK_SIZE,), 0.0, tl.float32)

    for r in range(ROWS):
        row = pid * ROWS + r
        if row < M:
            Y_cur = row * y_stride_r + Y
            X_cur = row * x_stride_r + X
            if SINGLE_BLOCK:
                x = tl.load(X_cur + cols * x_stride_c, mask=cmask, other=0.0).to(
                    tl.float32
                )
                var = tl.sum(x * x, axis=0) / N
                rrms = tl.math.rsqrt(var + eps)
                y = (x * rrms).to(Y_cur.dtype.element_ty) * w_shared
                tl.store(Y_cur + cols * y_stride_c, y, mask=cmask)
                if SAVE_INV_RMS:
                    tl.store(INV_RMS + row, rrms)
            else:
                # 行宽超过单 tile 上限：分块累加方差，再分块写结果
                _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
                for off in range(0, N, BLOCK_SIZE):
                    c = off + tl.arange(0, BLOCK_SIZE)
                    m = c < N
                    t = tl.load(X_cur + c * x_stride_c, mask=m, other=0.0).to(
                        tl.float32
                    )
                    _var += t * t
                var = tl.sum(_var, axis=0) / N
                rrms = tl.math.rsqrt(var + eps)
                for off in range(0, N, BLOCK_SIZE):
                    c = off + tl.arange(0, BLOCK_SIZE)
                    m = c < N
                    t = tl.load(X_cur + c * x_stride_c, mask=m, other=0.0).to(
                        tl.float32
                    )
                    w = tl.load(W + c, mask=m, other=0.0)
                    y = (t * rrms).to(Y_cur.dtype.element_ty) * w
                    tl.store(Y_cur + c * y_stride_c, y, mask=m)
                if SAVE_INV_RMS:
                    tl.store(INV_RMS + row, rrms)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_kernel(
    Y,
    INV_RMS,
    X,
    W,
    y_stride_r: tl.constexpr,
    y_stride_c: tl.constexpr,
    x_stride_r: tl.constexpr,
    x_stride_c: tl.constexpr,
    M,
    N,
    eps,
    total_programs,
    BLOCK_SIZE: tl.constexpr,
    ROWS: tl.constexpr,
    SINGLE_BLOCK: tl.constexpr,
    SAVE_INV_RMS: tl.constexpr,
):
    for pid in range(tl.program_id(0), total_programs, tl.num_programs(0)):
        _rms_norm_rows(
            Y,
            INV_RMS,
            X,
            W,
            pid,
            y_stride_r,
            y_stride_c,
            x_stride_r,
            x_stride_c,
            M,
            N,
            eps,
            BLOCK_SIZE,
            ROWS,
            SINGLE_BLOCK,
            SAVE_INV_RMS,
        )


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
    BLOCK_SIZE_M: tl.constexpr,
):
    for i in range(BLOCK_SIZE_M):
        pid = tl.program_id(0) * BLOCK_SIZE_M + i
        DX_cur = pid * dx_stride_r + DX
        X_cur = pid * x_stride_r + X
        DY_cur = pid * x_stride_r + DY
        INV_RMS_cur = pid + INV_RMS

        inv_rms = tl.load(INV_RMS_cur).to(tl.float32)

        _row_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(X_cur + cols * x_stride_c, mask, other=0.0).to(tl.float32)
            dy = tl.load(DY_cur + cols * x_stride_c, mask, other=0.0).to(tl.float32)
            w = tl.load(W + cols, mask=mask, other=0.0)
            dy_w = dy * w
            normalized_buf = x * inv_rms
            _row_sum += normalized_buf * dy_w
        row_sum_stats = tl.sum(_row_sum, axis=0)

        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(X_cur + cols * x_stride_c, mask, other=0.0).to(tl.float32)
            dy = tl.load(DY_cur + cols * x_stride_c, mask, other=0.0).to(tl.float32)
            w = tl.load(W + cols, mask=mask, other=0.0)
            dy_w = dy * w
            normalized_buf = x * inv_rms
            norm_val = normalized_buf / N
            dx = (dy_w - norm_val * row_sum_stats) * inv_rms
            tl.store(DX_cur + cols * dx_stride_c, dx, mask=mask)


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


class RmsNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, eps=1e-5):
        logger.debug("GEMS_ENFLAME RMSNORM_FORWARD")
        dim = x.ndim - len(normalized_shape)
        M = math.prod(x.shape[:dim])
        N = math.prod(normalized_shape)

        BLOCK_SIZE = min(triton.next_power_of_2(N), MAX_BLOCK_SIZE)
        single_block = triton.next_power_of_2(N) <= MAX_BLOCK_SIZE
        # 只有需要反向时才保存 inv_rms；benchmark / 推理路径不分配也不写
        save_inv_rms = x.requires_grad or weight.requires_grad
        x = x.contiguous()
        weight = weight.contiguous()
        y = torch.empty_like(x)
        inv_rms = (
            torch.empty((M,), device=x.device, dtype=torch.float32)
            if save_inv_rms
            else x
        )

        # 每个 program 处理多行：w 只加载一次、program 数按 M 减少
        if single_block:
            if M >= 512:
                rows = 8
            elif M >= 128:
                rows = 4
            else:
                rows = 2
        else:
            rows = 1
        total_programs = triton.cdiv(M, rows)
        grid_m = min(total_programs, MAX_GRID_DIM)

        with torch_device_fn.device(x.device):
            rms_norm_kernel[grid_m,](
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
                total_programs,
                BLOCK_SIZE,
                rows,
                single_block,
                save_inv_rms,
                num_warps=_FORWARD_WARPS,
            )

        ctx.save_for_backward(x, inv_rms, weight)
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        logger.debug("GEMS_ENFLAME RMSNORM_BACKWARD")
        x, inv_rms, weight = ctx.saved_tensors
        normalized_shape = ctx.normalized_shape
        eps = ctx.eps

        dim = x.ndim - len(normalized_shape)
        M = math.prod(x.shape[:dim])
        N = math.prod(normalized_shape)

        BLOCK_SIZE = min(triton.next_power_of_2(N), MAX_BLOCK_SIZE)
        x = x.contiguous()
        weight = weight.contiguous()
        dx = torch.empty_like(x)

        grid_m = M
        BLOCK_SIZE_M = 1
        if M > 65535:
            grid_m = MAX_GRID_DIM
            BLOCK_SIZE_M = (M + MAX_GRID_DIM - 1) // MAX_GRID_DIM

        with torch_device_fn.device(x.device):
            rms_norm_grad_dx_kernel[grid_m,](
                x, dy, inv_rms, dx, weight, N, 1, N, 1, N, eps, BLOCK_SIZE, BLOCK_SIZE_M
            )

        ROW_BLOCK_SIZE = 16
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
            )
            # TODO(haizhu.shao TBD) after support dim=0 in triton_gcu backend,
            # we can directly use flag_gems.sum(partial_buffer, dim=0, dtype=x.dtype)
            dw = torch.sum(partial_buffer, dim=0).to(x.dtype).reshape(-1)

        return dx, None, dw, None


def rms_norm(x, normalized_shape, weight, eps=1e-5):
    return RmsNorm.apply(x, normalized_shape, weight, eps)
