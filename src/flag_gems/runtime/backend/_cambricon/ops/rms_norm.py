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

from ..utils import MAX_GRID_SIZE_X

logger = logging.getLogger(__name__)
MAX_NRAM_C_FORWARD = 16384


def cfggen_rms_norm_c_split():
    return [
        triton.Config({"BLOCK_SIZE": block_size}, num_warps=1, num_stages=num_stages)
        for block_size in [1024, 2048, 4096, 8192, 16384]
        for num_stages in [1, 3]
    ]


def rms_norm_forward(x, normalized_shape, weight, eps=1e-5):
    logger.debug("GEMS_CAMBRICON RMS_NORM")
    dim = x.ndim - len(normalized_shape)
    M = math.prod(x.shape[:dim])
    N = math.prod(normalized_shape)

    BLOCK_SIZE = N  # triton.next_power_of_2(N)
    x = x.contiguous()
    weight = weight.contiguous()
    y = torch.empty_like(x)
    inv_rms = torch.empty((M,), device=x.device, dtype=torch.float32)
    grid = (min(M, MAX_GRID_SIZE_X // 4),)
    with torch_device_fn.device(x.device):
        if BLOCK_SIZE <= MAX_NRAM_C_FORWARD:
            logger.debug("GEMS_CAMBRICON RMS_NORM")
            rms_norm_kernel[grid](
                y, inv_rms, x, weight, N, 1, N, 1, N, eps, M, BLOCK_SIZE
            )
        else:
            logger.debug("GEMS_CAMBRICON RMS_NORM")
            rms_norm_kernel_C_split[grid](y, inv_rms, x, weight, N, 1, N, 1, N, eps, M)
    return y, inv_rms


def rms_norm_backward(dy, x, inv_rms, normalized_shape, weight, eps=1e-5):
    logger.debug("GEMS_CAMBRICON RMS_NORM_BACKWARD")
    dim = x.ndim - len(normalized_shape)
    M = math.prod(x.shape[:dim])
    N = math.prod(normalized_shape)

    # BLOCK_SIZE = triton.next_power_of_2(N)
    BLOCK_SIZE = N
    x = x.contiguous()
    weight = weight.contiguous()
    dx = torch.empty_like(x)
    grid = (min(M, MAX_GRID_SIZE_X // 4),)
    with torch_device_fn.device(x.device):
        if BLOCK_SIZE <= MAX_NRAM_C_FORWARD:
            logger.debug("GEMS_CAMBRICON RMS_NORM_BACKWARD")
            rms_norm_grad_dx_kernel[grid](
                x, dy, inv_rms, dx, weight, N, 1, N, 1, N, eps, M, BLOCK_SIZE
            )
        else:
            logger.debug("GEMS_CAMBRICON RMS_NORM_BACKWARD")
            rms_norm_grad_dx_kernel_C_split[grid](
                x, dy, inv_rms, dx, weight, N, 1, N, 1, N, eps, M
            )

    is_bfloat16 = x.dtype == torch.bfloat16
    ROW_BLOCK_SIZE = 1 if is_bfloat16 else 16
    COL_BLOCK_SIZE = 256
    row_block_num = triton.cdiv(M, ROW_BLOCK_SIZE)
    col_block_num = triton.cdiv(N, COL_BLOCK_SIZE)
    grid_dw = (
        min(row_block_num, MAX_GRID_SIZE_X),
        triton.cdiv(row_block_num, MAX_GRID_SIZE_X),
        col_block_num,
    )

    partial_dtype = x.dtype if is_bfloat16 else torch.float32
    partial_buffer = torch.empty(
        (row_block_num, N), dtype=partial_dtype, device=x.device
    )

    with torch_device_fn.device(x.device):
        rms_norm_grad_dw_kernel[grid_dw](
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
            row_block_num,
            ROW_BLOCK_SIZE,
            COL_BLOCK_SIZE,
        )
        sum_dtype = x.dtype if is_bfloat16 else torch.float32
        dw = torch.sum(partial_buffer, dim=0, dtype=sum_dtype).to(x.dtype).reshape(-1)

    return dx, dw


@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_kernel(
    Y,  # pointer to the output
    INV_RMS,  # pointer to inverse rms
    X,  # pointer to the input
    W,  # pointer to the weights
    y_stride_r,
    y_stride_c,
    x_stride_r,  # how much to increase the pointer when moving by 1 row
    x_stride_c,  # how much to increase the pointer when moving by 1 col
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    M,  # number of rows in X
    BLOCK_SIZE: tl.constexpr,
):
    prog_num = tl.num_programs(0).to(tl.uint64)
    task_num = M
    pid = tl.program_id(0).to(tl.uint64)
    while pid < task_num:
        Y_ = Y + pid * y_stride_r
        X_ = X + pid * x_stride_r

        mask = tl.arange(0, BLOCK_SIZE) < N
        cols = tl.arange(0, BLOCK_SIZE)
        x = tl.load(X_ + cols * x_stride_c, mask, other=0.0).to(tl.float32)

        var = tl.sum(x * x, axis=0) / N
        rrms = 1 / tl.sqrt(var + eps)

        w = tl.load(W + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
        # Cast x_normed back to input dtype before multiplying with weight
        # to align with vLLM native: x.to(weight.dtype) * weight
        x_normed = (x * rrms).to(X_.dtype.element_ty)
        y = x_normed * w
        tl.store(Y_ + cols * y_stride_c, y, mask=mask)
        tl.store(INV_RMS + pid, rrms)
        pid += prog_num


@triton.autotune(
    configs=cfggen_rms_norm_c_split(),
    key=["N"],
)
@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_kernel_C_split(
    Y,  # pointer to the output
    INV_RMS,  # pointer to inverse rms
    X,  # pointer to the input
    W,  # pointer to the weights
    y_stride_r,
    y_stride_c,
    x_stride_r,  # how much to increase the pointer when moving by 1 row
    x_stride_c,  # how much to increase the pointer when moving by 1 col
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    M,  # number of rows in X
    BLOCK_SIZE: tl.constexpr,
):
    prog_num = tl.num_programs(0).to(tl.uint64)
    task_num = M
    pid = tl.program_id(0).to(tl.uint64)
    while pid < task_num:
        Y_ = Y + pid * y_stride_r
        X_ = X + pid * x_stride_r

        var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for m_idx in range(0, N, BLOCK_SIZE):
            cols = m_idx + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(X_ + cols * x_stride_c, mask, other=0.0).to(tl.float32)
            var += x * x

        var = tl.sum(var, axis=0) / N
        rrms = 1 / tl.sqrt(var + eps)

        for m_idx in range(0, N, BLOCK_SIZE):
            cols = m_idx + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            w = tl.load(W + cols, mask=mask, other=0.0)
            x = tl.load(X_ + cols * x_stride_c, mask, other=0.0).to(tl.float32)
            # Cast x_normed back to input dtype before multiplying with weight
            x_normed = (x * rrms).to(X_.dtype.element_ty)
            y = x_normed * w
            tl.store(Y_ + cols * y_stride_c, y, mask=mask)
        tl.store(INV_RMS + pid, rrms)
        pid += prog_num


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
    M,  # number of rows in X
    BLOCK_SIZE: tl.constexpr,
):
    prog_num = tl.num_programs(0).to(tl.uint64)
    task_num = M
    pid = tl.program_id(0).to(tl.uint64)
    while pid < task_num:
        DX_ = DX + pid * dx_stride_r
        X_ = X + pid * x_stride_r
        DY_ = DY + pid * x_stride_r
        INV_RMS_ = INV_RMS + pid

        mask = tl.arange(0, BLOCK_SIZE) < N
        cols = tl.arange(0, BLOCK_SIZE)
        x = tl.load(X_ + cols * x_stride_c, mask, other=0.0).to(tl.float32)
        inv_rms = tl.load(INV_RMS_).to(tl.float32)
        dy = tl.load(DY_ + cols * x_stride_c, mask, other=0.0).to(tl.float32)
        w = tl.load(W + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0).to(tl.float32)

        dy = (dy * w).to(DY_.dtype.element_ty).to(tl.float32)

        normalized_buf = x * inv_rms
        row_sum_stats = tl.sum(normalized_buf * dy, axis=0)

        norm_val = normalized_buf / N
        dx = (dy - norm_val * row_sum_stats) * inv_rms

        tl.store(DX_ + cols * dx_stride_c, dx, mask=mask)
        pid += prog_num


@triton.autotune(
    configs=cfggen_rms_norm_c_split(),
    key=["N"],
)
@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_grad_dx_kernel_C_split(
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
    M,  # number of rows in X
    BLOCK_SIZE: tl.constexpr,
):
    prog_num = tl.num_programs(0).to(tl.uint64)
    task_num = M
    pid = tl.program_id(0).to(tl.uint64)
    while pid < task_num:
        DX_ = DX + pid * dx_stride_r
        X_ = X + pid * x_stride_r
        DY_ = DY + pid * x_stride_r
        INV_RMS_ = INV_RMS + pid
        inv_rms = tl.load(INV_RMS_).to(tl.float32)

        acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for m_idx in range(0, N, BLOCK_SIZE):
            cols = m_idx + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(X_ + cols * x_stride_c, mask=mask, other=0.0).to(tl.float32)
            inv_rms = tl.load(INV_RMS_).to(tl.float32)
            dy = tl.load(DY_ + cols * x_stride_c, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
            dy = (dy * w).to(DY_.dtype.element_ty).to(tl.float32)
            normalized = x * inv_rms
            acc += normalized * dy

        row_sum_stats = tl.sum(acc, axis=0)

        for m_idx in range(0, N, BLOCK_SIZE):
            cols = m_idx + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(X_ + cols * x_stride_c, mask=mask, other=0.0).to(tl.float32)
            inv_rms = tl.load(INV_RMS_).to(tl.float32)
            dy = tl.load(DY_ + cols * x_stride_c, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
            dy = (dy * w).to(DY_.dtype.element_ty).to(tl.float32)
            normalized = x * inv_rms
            norm_val = normalized / N
            dx = (dy - norm_val * row_sum_stats) * inv_rms
            tl.store(DX_ + cols * dx_stride_c, dx, mask=mask)
        pid += prog_num


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
    ROW_BLOCK_NUM,
    ROW_BLOCK_SIZE: tl.constexpr,
    COL_BLOCK_SIZE: tl.constexpr,
):
    row_pid = tl.program_id(0) + tl.program_id(1) * tl.num_programs(0)
    col_pid = tl.program_id(2)

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

    normalized = (x * inv_rms[:, None]).to(DY.dtype.element_ty).to(tl.float32)
    d_weight = normalized * dy
    partial_dweight_sum = tl.sum(d_weight, axis=0)

    tl.store(
        DW + row_pid * N + col_start + cols,
        partial_dweight_sum,
        mask=(row_pid < ROW_BLOCK_NUM) & col_mask,
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
        normalized_shape = ctx.normalized_shape
        eps = ctx.eps
        dx, dw = rms_norm_backward(dy, x, inv_rms, normalized_shape, weight, eps)
        return dx, None, dw, None


def rms_norm(x, normalized_shape, weight, eps=1e-5):
    return RmsNorm.apply(x, normalized_shape, weight, eps)
