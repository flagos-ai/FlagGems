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
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

TRITON_MAX_TENSOR_NUMEL = 1048576


def _prune_rmsnorm_fwd_configs(configs, nargs, **kwargs):
    """Filter out configs where 2D block (ROWS_PER_PROGRAM x BLOCK_N) exceeds max tensor numel."""
    N = nargs["N"]
    block_n = triton.next_power_of_2(N)
    return [
        c
        for c in configs
        if c.kwargs["ROWS_PER_PROGRAM"] * block_n <= TRITON_MAX_TENSOR_NUMEL
    ]


def _prune_rmsnorm_bwd_configs(configs, nargs, **kwargs):
    """Filter out backward configs where the grid size exceeds hardware limits."""
    M = nargs["M"]
    MAX_GRID = 65536
    return [
        c for c in configs if triton.cdiv(M, c.kwargs["ROW_BLOCK_SIZE"]) <= MAX_GRID
    ]


@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_kernel(
    out_ptr,  # pointer to the output
    INV_RMS,  # pointer to inverse rms
    in_ptr,  # pointer to the input
    w_ptr,  # pointer to the weights
    y_stride_r,
    y_stride_c,
    x_stride_r,  # how much to increase the pointer when moving by 1 row
    x_stride_c,  # how much to increase the pointer when moving by 1 col
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    if tl.constexpr(in_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        in_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = in_ptr.dtype.element_ty

    pid = tl.program_id(0)
    out_ptr += pid * y_stride_r
    in_ptr += pid * x_stride_r

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(in_ptr + cols * x_stride_c, mask, other=0.0).to(cdtype)

    var = tl.sum(x * x, axis=0) / N
    rrms = 1 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    y = (x * rrms * w).to(cdtype)
    tl.store(out_ptr + cols * y_stride_c, y, mask=mask)
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
    pid = tle.program_id(0)
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
    # Sum over rows (axis=0) - masked rows are 0 (from other=0.0 in load), so sum is correct
    # The mask ensures invalid rows contribute 0 to the sum
    partial_dweight_sum = tl.sum(d_weight, axis=0)

    tl.store(
        DW + row_pid * N + col_start + cols,
        partial_dweight_sum,
        mask=col_mask,
    )


@libentry()
@libtuner(
    configs=[
        triton.Config({"ROW_BLOCK_SIZE": 1}, num_stages=1),
        triton.Config({"ROW_BLOCK_SIZE": 4}, num_stages=1),
        triton.Config({"ROW_BLOCK_SIZE": 8}, num_stages=1),
        triton.Config({"ROW_BLOCK_SIZE": 16}, num_stages=1),
        triton.Config({"ROW_BLOCK_SIZE": 32}, num_stages=1),
        triton.Config({"ROW_BLOCK_SIZE": 64}, num_stages=1),
        triton.Config({"ROW_BLOCK_SIZE": 128}, num_stages=1),
        triton.Config({"ROW_BLOCK_SIZE": 256}, num_stages=1),
    ],
    key=["N"],
    prune_configs_by={"early_config_prune": _prune_rmsnorm_bwd_configs},
)
@triton.heuristics(
    values={"BLOCK_SIZE": lambda META: triton.next_power_of_2(META["N"])}
)
@triton.jit(do_not_specialize=["eps"])
def rms_norm_backward_fused_kernel(
    X,
    DY,
    INV_RMS,
    DX,
    DW_PARTIAL,
    W,
    dx_stride_r,
    dx_stride_c,
    x_stride_r,
    x_stride_c,
    M,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
    ROW_BLOCK_SIZE: tl.constexpr,
):
    """Fused dX + dW backward kernel — per-row 1D loads, axis=0 reduction.

    Processes ROW_BLOCK_SIZE rows per program via tl.static_range.
    Each row is loaded individually (1D) and reduced using tl.sum(axis=0)
    which is proven correct on this backend.
    dW is accumulated element-wise across rows.
    """
    cdtype = tl.float32

    row_pid = tl.program_id(0)
    row_start = row_pid * ROW_BLOCK_SIZE

    col_mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)

    w = tl.load(W + cols, mask=col_mask, other=0.0).to(cdtype)
    dw_acc = tl.zeros((BLOCK_SIZE,), dtype=cdtype)

    for r in tl.static_range(0, ROW_BLOCK_SIZE):
        row_idx = row_start + r
        row_valid = row_idx < M

        row_valid_idx = tl.where(row_valid, row_idx, 0)
        x_row = tl.load(
            X + row_valid_idx * x_stride_r + cols * x_stride_c,
            mask=col_mask,
            other=0.0,
        ).to(cdtype)

        inv_rms_r = tl.load(INV_RMS + row_valid_idx, mask=row_valid, other=0.0).to(
            cdtype
        )

        dy_row = tl.load(
            DY + row_valid_idx * x_stride_r + cols * x_stride_c,
            mask=col_mask,
            other=0.0,
        ).to(cdtype)

        # dW: accumulate BEFORE dX computation.
        # Use tl.where(row_valid, ...) to guard against garbage values.
        dw_acc = dw_acc + tl.where(row_valid, x_row * dy_row * inv_rms_r, 0.0)

        # dX: 1D reduction axis=0, proven correct
        dy_w = dy_row * w
        normalized_buf = x_row * inv_rms_r
        row_sum_stats = tl.sum(normalized_buf * dy_w, axis=0)
        norm_val = normalized_buf / N
        dx_row = (dy_w - norm_val * row_sum_stats) * inv_rms_r

        # Store dX only for valid rows (protect adjacent memory from out-of-bounds writes)
        tl.store(
            DX + row_valid_idx * dx_stride_r + cols * dx_stride_c,
            dx_row,
            mask=col_mask,
        )

    tl.store(
        DW_PARTIAL + row_pid * N + cols,
        dw_acc,
        mask=col_mask,
    )


@libentry()
@libtuner(
    configs=[
        triton.Config({"ROWS_PER_PROGRAM": 1}, num_stages=1),
        triton.Config({"ROWS_PER_PROGRAM": 4}, num_stages=1),
        triton.Config({"ROWS_PER_PROGRAM": 8}, num_stages=1),
        triton.Config({"ROWS_PER_PROGRAM": 16}, num_stages=1),
        triton.Config({"ROWS_PER_PROGRAM": 32}, num_stages=1),
        triton.Config({"ROWS_PER_PROGRAM": 64}, num_stages=1),
        # triton.Config({"ROWS_PER_PROGRAM": 128}),
        # triton.Config({"ROWS_PER_PROGRAM": 256}),
    ],
    key=["N"],
    prune_configs_by={"early_config_prune": _prune_rmsnorm_fwd_configs},
)
@triton.heuristics(values={"BLOCK_N": lambda META: triton.next_power_of_2(META["N"])})
@triton.jit(do_not_specialize=["eps"])
def rmsnorm_fwd_kernel_multirow(
    out_ptr,
    inv_rms_ptr,
    x_ptr,
    w_ptr,
    stride_row,
    N,
    num_rows,
    eps,
    BLOCK_N: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    pid = tl.program_id(0)

    # 该 program 负责的行号范围
    row_start = pid * ROWS_PER_PROGRAM
    row_offsets = row_start + tl.arange(0, ROWS_PER_PROGRAM)  # (ROWS_PER_PROGRAM,)
    row_mask = row_offsets < num_rows

    col_offsets = tl.arange(0, BLOCK_N)  # (BLOCK_N,)
    col_mask = col_offsets < N

    # 2D mask: (ROWS_PER_PROGRAM, BLOCK_N)
    mask = row_mask[:, None] & col_mask[None, :]

    # 2D 指针: ptr = base + row * stride_row + col
    ptrs = x_ptr + row_offsets[:, None] * stride_row + col_offsets[None, :]
    x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)

    # 沿列方向(axis=1)做 reduction，每行独立算 RMS
    sq_sum = tl.sum(x * x, axis=1)  # (ROWS_PER_PROGRAM,)
    rrms = 1.0 / tl.sqrt(sq_sum / N + eps)  # (ROWS_PER_PROGRAM,)

    # row_start is a scalar, rrms is a block — need block pointer for block store
    tl.store(
        inv_rms_ptr + row_start + tl.arange(0, ROWS_PER_PROGRAM), rrms, mask=row_mask
    )

    # weight 在所有行间共享，只 load 一次
    w = tl.load(w_ptr + col_offsets, mask=col_mask, other=0.0).to(tl.float32)

    y = x * rrms[:, None] * w[None, :]

    out_ptrs = out_ptr + row_offsets[:, None] * stride_row + col_offsets[None, :]
    tl.store(out_ptrs, y.to(x_ptr.dtype.element_ty), mask=mask)


def rms_norm_forward(x, normalized_shape, weight, eps=1e-5):
    logger.debug("GEMS_TSINGMICRO RMS_NORM_FORWARD")
    dim = x.ndim - len(normalized_shape)
    M = math.prod(x.shape[:dim])
    N = math.prod(normalized_shape)

    # Flatten batch dims → (M, N) so the kernel sees a 2D layout.
    x_2d = x.reshape(M, N)
    # Use empty_strided to preserve x's strides in the output tensor.
    # torch.empty_like does NOT preserve strides on all PyTorch versions,
    # and the kernel writes using x_2d.stride(0) (which may be > N for
    # non-contiguous inputs). The output must have matching strides so
    # that out_ptr + i*stride_row + j lands on the correct element.
    out = torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device=x.device)

    inv_rms = torch.empty((M,), device=x.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta["ROWS_PER_PROGRAM"]),)

    rmsnorm_fwd_kernel_multirow[grid](
        out,
        inv_rms,
        x_2d,
        weight,
        x_2d.stride(0),
        N,
        M,
        eps,
    )
    return out, inv_rms


def rms_norm_backward(dy, x, inv_rms, normalized_shape, weight, eps=1e-5):
    logger.debug("GEMS_TSINGMICRO RMS_NORM_BACKWARD")
    dim = x.ndim - len(normalized_shape)
    M = math.prod(x.shape[:dim])
    N = math.prod(normalized_shape)

    x = x.contiguous()
    dy = dy.contiguous()
    weight = weight.contiguous()
    dx = torch.empty_like(x)

    # Fused dX+dW: per-row 1D loads with axis=0 reduction.
    # The autotuner selects ROW_BLOCK_SIZE at runtime, so we cannot know the grid
    # size before the kernel launch.  Capture the actual grid from the lambda so
    # we can slice the partial-buffer correctly afterwards — unused rows are
    # uninitialised and would corrupt the dw sum otherwise.
    actual_grid_size = [None]

    def grid(meta):
        g = triton.cdiv(M, meta["ROW_BLOCK_SIZE"])
        actual_grid_size[0] = g
        return (g,)

    # Size the partial dW buffer for the maximum grid that can survive pruning.
    # The prune function filters out configs where grid > MAX_GRID, so the
    # smallest surviving ROW_BLOCK_SIZE determines the largest possible grid.
    MAX_GRID = 65536
    min_row_block = max(1, (M + MAX_GRID - 1) // MAX_GRID)
    safe_row_block = 1 << max(0, (min_row_block - 1).bit_length())
    row_block_num = triton.cdiv(M, safe_row_block)
    partial_buffer = torch.empty(
        (row_block_num, N), dtype=torch.float32, device=x.device
    )

    with torch_device_fn.device(x.device):
        rms_norm_backward_fused_kernel[grid](
            x, dy, inv_rms, dx, partial_buffer, weight, N, 1, N, 1, M, N, eps
        )
    # Sum only the rows that were actually written by the kernel.
    # unused rows (beyond actual_grid_size) contain uninitialised data.
    dw = (
        torch.sum(partial_buffer[: actual_grid_size[0]], dim=0, dtype=torch.float32)
        .to(x.dtype)
        .reshape(-1)
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
