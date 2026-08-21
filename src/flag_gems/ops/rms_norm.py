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

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.device_info import get_device_capability
from flag_gems.utils.triton_version_utils import HAS_TLE

logger = logging.getLogger(__name__)

if HAS_TLE:
    import triton.experimental.tle.language as tle
else:
    tle = None

_TLE_MIN_CAPABILITY = 9

_DW_ROW_BLOCK_SIZE = 16
_DW_COL_BLOCK_SIZE = 256
_DW_TLE_NUM_WARPS = 4


@triton.jit
def prev_multiple_of(a, b):
    return tl.cdiv(a, b) * b - b


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
    # Cast x_normed back to input dtype before multiplying with weight
    # to align with vLLM native: x.to(weight.dtype) * weight
    x_normed = (x * rrms).to(in_ptr.dtype.element_ty)
    y = x_normed * w
    tl.store(out_ptr + cols * y_stride_c, y, mask=mask)
    tl.store(INV_RMS + pid, rrms)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("rms_norm_loop"),
    key=["N"],
)
@triton.jit(do_not_specialize=["eps"])
def rms_norm_loop_kernel(
    out_ptr,
    INV_RMS,
    in_ptr,
    w_ptr,
    N,
    eps,
    TILE_N: tl.constexpr,
):
    if tl.constexpr(in_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        in_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = in_ptr.dtype.element_ty

    pid = ext.program_id(0)

    acc = tl.zeros((TILE_N,), dtype=tl.float32)
    num_steps = tl.cdiv(N, TILE_N)

    for step in range(0, num_steps - 1):
        start_n = step * TILE_N
        n_offsets = start_n + tl.arange(0, TILE_N)
        x = tl.load(in_ptr + pid * N + n_offsets).to(tl.float32)
        acc += x * x

    start_n = (num_steps - 1) * TILE_N
    n_offsets = start_n + tl.arange(0, TILE_N)
    mask = n_offsets < N
    x = tl.load(in_ptr + pid * N + n_offsets, mask=mask, other=0.0).to(tl.float32)
    acc += x * x

    var = tl.sum(acc) / N
    rrms = 1 / tl.sqrt(var + eps)
    tl.store(INV_RMS + pid, rrms)

    prev_multiple = prev_multiple_of(N, TILE_N)

    for start_n in range(0, TILE_N, TILE_N):
        n_offsets = (prev_multiple - start_n) + tl.arange(0, TILE_N)
        mask = n_offsets < N
        x = tl.load(
            in_ptr + pid * N + n_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(cdtype)
        w = tl.load(w_ptr + n_offsets, mask=mask, other=0.0)
        # Cast x_normed back to input dtype before multiplying with weight
        x_normed = (x * rrms).to(in_ptr.dtype.element_ty)
        y = x_normed * w
        tl.store(out_ptr + pid * N + n_offsets, y, mask=mask)

    for start_n in range(TILE_N, N, TILE_N):
        n_offsets = (prev_multiple - start_n) + tl.arange(0, TILE_N)
        x = tl.load(
            in_ptr + pid * N + n_offsets,
            eviction_policy="evict_first",
        ).to(cdtype)
        w = tl.load(w_ptr + n_offsets)
        # Cast x_normed back to input dtype before multiplying with weight
        x_normed = (x * rrms).to(in_ptr.dtype.element_ty)
        y = x_normed * w
        tl.store(out_ptr + pid * N + n_offsets, y)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_grad_dx_loop_kernel(
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

    inv_rms = tl.load(INV_RMS).to(tl.float32)

    # First pass: compute row_sum_stats = sum(x * inv_rms * dy * w)
    row_sum_stats = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for start_n in range(0, N, BLOCK_SIZE):
        cols = start_n + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols * x_stride_c, mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + cols * x_stride_c, mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0)
        dy = dy * w
        normalized_buf = x * inv_rms
        row_sum_stats += normalized_buf * dy

    row_sum_stats_scalar = tl.sum(row_sum_stats, axis=0)

    # Second pass: compute and store dx
    for start_n in range(0, N, BLOCK_SIZE):
        cols = start_n + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols * x_stride_c, mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + cols * x_stride_c, mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0)
        dy = dy * w
        normalized_buf = x * inv_rms
        norm_val = normalized_buf / N
        dx = (dy - norm_val * row_sum_stats_scalar) * inv_rms
        tl.store(DX + cols * dx_stride_c, dx, mask=mask)


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


if HAS_TLE:

    @triton.jit
    def rms_norm_grad_dw_kernel_tle(
        X,
        DY,
        INV_RMS,
        DW,
        dx_stride_r,
        dx_stride_c,
        x_stride_r,
        x_stride_c,
        M,
        N,
        ROW_BLOCK_SIZE: tl.constexpr,
        COL_BLOCK_SIZE: tl.constexpr,
        TARGET_LAYOUT: tl.constexpr,
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
        d_weight = tle.gpu.set_layout(d_weight, TARGET_LAYOUT)
        partial_dweight_sum = tl.sum(d_weight, axis=0)

        tl.store(
            DW + row_pid * N + col_start + cols,
            partial_dweight_sum,
            mask=col_mask,
        )

    def _build_dw_set_layout(row_block_size, col_block_size, num_warps):
        denom = 32 * num_warps
        assert col_block_size % denom == 0, (
            f"col_block_size={col_block_size} must be divisible by "
            f"32*num_warps={denom}"
        )
        size_per_thread_col = col_block_size // denom
        return tle.gpu.BlockEncoding(
            size_per_thread=[row_block_size, size_per_thread_col],
            threads_per_warp=[1, 32],
            warps_per_cta=[1, num_warps],
            order=[1, 0],
        )

    _DW_TARGET_LAYOUT = _build_dw_set_layout(
        _DW_ROW_BLOCK_SIZE, _DW_COL_BLOCK_SIZE, _DW_TLE_NUM_WARPS
    )

else:
    rms_norm_grad_dw_kernel_tle = None
    _DW_TARGET_LAYOUT = None


def _dw_tle_available(x: torch.Tensor) -> bool:
    if not HAS_TLE:
        return False
    if x.device.type != "cuda":
        return False
    return get_device_capability()[0] >= _TLE_MIN_CAPABILITY


def rms_norm_out(result, x, normalized_shape, weight, eps=1e-5):
    y, _ = rms_norm_forward(x, normalized_shape, weight, eps=eps)
    result.copy_(y)
    return result


def rms_norm_forward(x, normalized_shape, weight, eps=1e-5):
    logger.debug("GEMS RMS_NORM FORWARD")
    dim = x.ndim - len(normalized_shape)
    M = math.prod(x.shape[:dim])
    N = math.prod(normalized_shape)

    x = x.contiguous()
    weight = weight.contiguous()
    y = torch.empty_like(x)
    inv_rms = torch.empty((M,), device=x.device, dtype=torch.float32)

    with torch_device_fn.device(x.device):
        if N <= 4096:
            BLOCK_SIZE = triton.next_power_of_2(N)
            rms_norm_kernel[M,](y, inv_rms, x, weight, N, 1, N, 1, N, eps, BLOCK_SIZE)
        else:
            rms_norm_loop_kernel[M,](y, inv_rms, x, weight, N, eps)

    return y, inv_rms


def rms_norm_backward(dy, x, inv_rms, normalized_shape, weight, eps=1e-5):
    logger.debug("GEMS RMS_NORM BACKWARD")
    dim = x.ndim - len(normalized_shape)
    M = math.prod(x.shape[:dim])
    N = math.prod(normalized_shape)

    x = x.contiguous()
    dy = dy.contiguous()
    weight = weight.contiguous()
    dx = torch.empty_like(x)

    with torch_device_fn.device(x.device):
        if N <= 4096:
            BLOCK_SIZE = triton.next_power_of_2(N)
            rms_norm_grad_dx_kernel[M,](
                x, dy, inv_rms, dx, weight, N, 1, N, 1, N, eps, BLOCK_SIZE
            )
        else:
            BLOCK_SIZE = 1024
            rms_norm_grad_dx_loop_kernel[M,](
                x, dy, inv_rms, dx, weight, N, 1, N, 1, N, eps, BLOCK_SIZE
            )

    ROW_BLOCK_SIZE = _DW_ROW_BLOCK_SIZE
    COL_BLOCK_SIZE = _DW_COL_BLOCK_SIZE
    row_block_num = triton.cdiv(M, ROW_BLOCK_SIZE)
    col_block_num = triton.cdiv(N, COL_BLOCK_SIZE)

    partial_buffer = torch.empty(
        (row_block_num, N), dtype=torch.float32, device=x.device
    )

    with torch_device_fn.device(x.device):
        if _dw_tle_available(x):
            rms_norm_grad_dw_kernel_tle[row_block_num, col_block_num](
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
                TARGET_LAYOUT=_DW_TARGET_LAYOUT,
                num_warps=_DW_TLE_NUM_WARPS,
            )
        else:
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
        dw = (
            torch.sum(partial_buffer, dim=0, dtype=torch.float32)
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
