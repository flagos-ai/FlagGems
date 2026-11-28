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
from typing import Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.ops.layernorm import layer_norm_loop_kernel as _precise_loop
from flag_gems.ops.layernorm import layer_norm_persistent_kernel as _precise_persistent
from flag_gems.ops.layernorm import (
    layer_norm_persistent_kernel_multiline as _precise_multiline,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@triton.jit
def prev_multiple_of(a, b):
    # the largest x<a that x%b ==0
    return tl.cdiv(a, b) * b - b


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("layer_norm_persistent"),
    key=["M", "N"],
)
@triton.jit(do_not_specialize=["eps"])
def layer_norm_persistent_kernel(
    in_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    out_mean_ptr,  # pointer to the mean
    out_rstd_ptr,  # pointer to the 1/std
    M,
    N,
    eps,
    TILE_N: tl.constexpr,
):
    # using 1d tile makes code clean
    # Map the program id to the row of X and Y it should compute.
    pid = ext.program_id(0)

    n_offsets = tl.arange(0, TILE_N)
    mask = n_offsets < N

    x = tl.load(in_ptr + pid * N + n_offsets, mask, other=0.0).to(tl.float32)
    m = tl.sum(x) / N
    d = x - m  # deviation
    s = tl.where(mask, d * d, 0)
    sum_square = tl.sum(s)  # sum of square of deviation
    var = sum_square / N
    rstd = tl.math.rsqrt(var + eps)

    tl.store(out_mean_ptr + pid, m)
    tl.store(out_rstd_ptr + pid, rstd)

    if weight_ptr is None:
        w = 1
    else:
        w = tl.load(weight_ptr + n_offsets, mask=mask)
    if bias_ptr is None:
        b = 0
    else:
        b = tl.load(bias_ptr + n_offsets, mask=mask)
    out = (x - m) * rstd * w + b

    tl.store(out_ptr + pid * N + n_offsets, out, mask=mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("layer_norm_persistent"),
    key=["M", "N"],
)
@triton.jit(do_not_specialize=["eps"])
def layer_norm_persistent_kernel_multiline(
    in_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    out_mean_ptr,  # pointer to the mean
    out_rstd_ptr,  # pointer to the 1/std
    M,
    N,
    eps,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    pid = ext.program_id(0)
    m_offsets = pid * TILE_M + tl.arange(0, TILE_M)
    m_mask = m_offsets < M

    n_offsets = tl.arange(0, TILE_N)[None, :]
    n_mask = n_offsets < N
    mask = m_mask[:, None] & n_mask

    x = tl.load(in_ptr + m_offsets[:, None] * N + n_offsets, mask, other=0.0).to(
        tl.float32
    )
    m = tl.sum(x, axis=1) / N
    d = x - m[:, None]  # deviation
    s = tl.where(mask, d * d, 0)
    sum_square = tl.sum(s, axis=1)  # sum of square of deviation
    var = sum_square / N
    rstd = tl.math.rsqrt(var + eps)

    tl.store(out_mean_ptr + m_offsets, m, mask=m_mask)
    tl.store(out_rstd_ptr + m_offsets, rstd, mask=m_mask)

    if weight_ptr is None:
        w = 1
    else:
        w = tl.load(weight_ptr + n_offsets, mask=n_mask)
    if bias_ptr is None:
        b = 0
    else:
        b = tl.load(bias_ptr + n_offsets, mask=n_mask)
    out = (x - m[:, None]) * rstd[:, None] * w + b

    tl.store(out_ptr + m_offsets[:, None] * N + n_offsets, out, mask=mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("layer_norm_loop"),
    key=["M", "N"],
)
@triton.jit(do_not_specialize=["eps"])
def layer_norm_loop_kernel(
    in_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    out_mean_ptr,  # pointer to the mean
    out_rstd_ptr,  # pointer to the 1/std
    M,
    N,
    eps,
    TILE_N: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    pid = ext.program_id(0)

    # Compute mean
    m = tl.zeros((TILE_N,), dtype=tl.float32)  # mean
    s = tl.zeros((TILE_N,), dtype=tl.float32)  # sum((x - m)^2)
    cnt = tl.zeros((TILE_N,), dtype=tl.int32)
    num_steps = tl.cdiv(N, TILE_N)
    for step in range(0, num_steps - 1, 1):
        start_n = step * TILE_N
        n_offsets = start_n + tl.arange(0, TILE_N)
        x = tl.load(in_ptr + pid * N + n_offsets).to(tl.float32)
        new_m = m + (x - m) / (step + 1)
        new_s = s + (x - new_m) * (x - m)
        cnt += 1
        m = new_m
        s = new_s

    # the last step
    for step in range(num_steps - 1, num_steps, 1):
        start_n = step * TILE_N
        n_offsets = start_n + tl.arange(0, TILE_N)
        mask = n_offsets < N
        x = tl.load(in_ptr + pid * N + n_offsets, mask=mask).to(tl.float32)
        new_m = tl.where(mask, m + (x - m) / (step + 1), m)
        new_s = tl.where(mask, s + (x - new_m) * (x - m), s)
        cnt += mask.to(tl.int32)
        m = new_m
        s = new_s

    final_m = tl.sum(m * cnt) / N
    var = tl.sum(s + cnt * (m - final_m) * (m - final_m)) / N
    rstd = tl.math.rsqrt(var + eps)
    m = final_m
    # Write mean / rstd
    tl.store(out_mean_ptr + pid, m)
    tl.store(out_rstd_ptr + pid, rstd)

    # reverse the order of the second sweep
    # Normalize and apply linear transformation
    prev_multiple = prev_multiple_of(N, TILE_N)
    # the first step, masking is needed
    for start_n in range(0, TILE_N, TILE_N):
        n_offsets = (prev_multiple - start_n) + tl.arange(0, TILE_N)
        mask = n_offsets < N
        x = tl.load(
            in_ptr + pid * N + n_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        if weight_ptr is None:
            w = 1
        else:
            w = tl.load(weight_ptr + n_offsets, mask=mask)
        if bias_ptr is None:
            b = 0
        else:
            b = tl.load(bias_ptr + n_offsets, mask=mask)
        out = w * (x - m) * rstd + b
        tl.store(out_ptr + pid * N + n_offsets, out, mask=mask)

    for start_n in range(TILE_N, N, TILE_N):
        n_offsets = (prev_multiple - start_n) + tl.arange(0, TILE_N)
        x = tl.load(in_ptr + pid * N + n_offsets, eviction_policy="evict_first").to(
            tl.float32
        )
        if weight_ptr is None:
            w = 1
        else:
            w = tl.load(weight_ptr + n_offsets)
        if bias_ptr is None:
            b = 0
        else:
            b = tl.load(bias_ptr + n_offsets)
        out = w * (x - m) * rstd + b
        tl.store(out_ptr + pid * N + n_offsets, out)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("layer_norm_backward"),
    key=["M", "N"],
)
@triton.jit
def layer_norm_backward_kernel(
    dY,
    X,
    W,
    Mean,
    Rstd,
    dX,
    M,
    N,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    pid = ext.program_id(0) * BLOCK_ROW_SIZE + tl.arange(0, BLOCK_ROW_SIZE)[:, None]
    row_mask = pid < M
    dY += pid * N
    X += pid * N
    dX += pid * N
    Mean += pid
    Rstd += pid

    mean = tl.load(Mean, mask=row_mask, other=0).to(tl.float32)
    rstd = tl.load(Rstd, mask=row_mask, other=0).to(tl.float32)

    dx_part2 = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    dx_part3 = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)

    for off in range(0, N, BLOCK_COL_SIZE):
        cols = off + tl.arange(0, BLOCK_COL_SIZE)
        col_mask = cols[None, :] < N
        mask = row_mask & col_mask
        dy = tl.load(dY + cols[None, :], mask, other=0).to(tl.float32)
        x = tl.load(X + cols[None, :], mask, other=0).to(tl.float32)
        x = tl.where(mask, x - mean, 0.0)
        x_hat = x * rstd
        if W is None:
            w = 1
        else:
            w = tl.load(W + cols, mask=cols < N, other=0).to(tl.float32)
        dx_hat = dy * w
        dx_part2 += dx_hat
        dx_part3 += dx_hat * x_hat

    dx_2 = tl.sum(dx_part2, axis=1)[:, None]
    dx_3 = tl.sum(dx_part3, axis=1)[:, None]

    for off in range(0, N, BLOCK_COL_SIZE):
        cols = off + tl.arange(0, BLOCK_COL_SIZE)
        col_mask = cols[None, :] < N
        mask = row_mask & col_mask
        dy = tl.load(dY + cols[None, :], mask).to(tl.float32)
        x = tl.load(X + cols[None, :], mask).to(tl.float32)
        if W is None:
            w = 1
        else:
            w = tl.load(W + cols, mask=cols < N).to(tl.float32)
        x = tl.where(mask, x - mean, 0.0)
        x_hat = x * rstd
        dx_hat = dy * w
        dx = rstd * (dx_hat - (dx_2 + x_hat * dx_3) / N)
        tl.store(dX + cols, dx, mask=mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("weight_bias_backward"),
    key=["N"],
)
@triton.jit
def weight_bias_backward_kernel(
    dY,
    X,
    Mean,
    Rstd,
    dW,
    dB,
    M,
    N,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    pid = ext.program_id(0) * BLOCK_COL_SIZE + tl.arange(0, BLOCK_COL_SIZE)
    col_mask = pid < N
    dY += pid[None, :]
    X += pid[None, :]
    accW = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    accB = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    for off in range(0, M, BLOCK_ROW_SIZE):
        rows = off + tl.arange(0, BLOCK_ROW_SIZE)[:, None]
        row_mask = rows < M
        mask = row_mask & col_mask[None, :]
        dy = tl.load(dY + rows * N, mask, other=0).to(tl.float32)
        x = tl.load(X + rows * N, mask, other=0).to(tl.float32)
        mean = tl.load(Mean + rows, mask=row_mask, other=0).to(tl.float32)
        rstd = tl.load(Rstd + rows, mask=row_mask, other=0).to(tl.float32)
        x = tl.where(mask, x - mean, 0.0)
        accW += dy * x * rstd
        accB += dy
    if dW:
        dw = tl.sum(accW, axis=0)
        tl.store(dW + pid, dw, mask=col_mask)
    if dB:
        db = tl.sum(accB, axis=0)
        tl.store(dB + pid, db, mask=col_mask)


@libentry()
@triton.jit
def _ln_bwd_row_dx(
    dY,
    X,
    W,
    Mean,
    Rstd,
    dX,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    dy = tl.load(dY + row * N + cols, mask, other=0).to(tl.float32)
    x = tl.load(X + row * N + cols, mask, other=0).to(tl.float32)
    mean = tl.load(Mean + row).to(tl.float32)
    rstd = tl.load(Rstd + row).to(tl.float32)
    if W is None:
        w = 1.0
    else:
        w = tl.load(W + cols, mask, other=0).to(tl.float32)
    xhat = tl.where(mask, (x - mean) * rstd, 0.0)
    dyw = dy * w
    sum2 = tl.sum(dyw, 0)
    sum3 = tl.sum(dyw * xhat, 0)
    dx = rstd * (dyw - (sum2 + xhat * sum3) / N)
    tl.store(dX + row * N + cols, dx, mask)


@libentry()
@triton.jit
def _ln_bwd_row_sums(
    dY,
    X,
    W,
    Mean,
    Rstd,
    Sum2,
    Sum3,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    part = tl.program_id(1)
    cols = part * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    dy = tl.load(dY + row * N + cols, mask, other=0).to(tl.float32)
    x = tl.load(X + row * N + cols, mask, other=0).to(tl.float32)
    mean = tl.load(Mean + row).to(tl.float32)
    rstd = tl.load(Rstd + row).to(tl.float32)
    if W is None:
        w = 1.0
    else:
        w = tl.load(W + cols, mask, other=0).to(tl.float32)
    xhat = tl.where(mask, (x - mean) * rstd, 0.0)
    dyw = dy * w
    parts: tl.constexpr = triton.cdiv(N, BLOCK)
    tl.store(Sum2 + row * parts + part, tl.sum(dyw, 0))
    tl.store(Sum3 + row * parts + part, tl.sum(dyw * xhat, 0))


@libentry()
@triton.jit
def _ln_bwd_fused_columns(
    dY,
    X,
    W,
    Mean,
    Rstd,
    Sum2,
    Sum3,
    dX,
    dW,
    dB,
    M: tl.constexpr,
    N: tl.constexpr,
    PARTS: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_PART: tl.constexpr,
):
    rows = tl.arange(0, TILE_M)
    cols = tl.program_id(0) * TILE_N + tl.arange(0, TILE_N)
    row_mask = rows < M
    col_mask = cols < N
    mask = row_mask[:, None] & col_mask[None, :]
    mean = tl.load(Mean + rows, row_mask, other=0).to(tl.float32)
    rstd = tl.load(Rstd + rows, row_mask, other=0).to(tl.float32)
    dy = tl.load(dY + rows[:, None] * N + cols[None, :], mask, other=0).to(tl.float32)
    x = tl.load(X + rows[:, None] * N + cols[None, :], mask, other=0).to(tl.float32)
    xhat = (x - mean[:, None]) * rstd[:, None]
    if dW is not None:
        dw = tl.sum(dy * xhat, 0)
        tl.store(dW + cols, dw, col_mask)
    if dB is not None:
        db = tl.sum(dy, 0)
        tl.store(dB + cols, db, col_mask)
    if dX is not None:
        part = tl.arange(0, TILE_PART)
        part_mask = row_mask[:, None] & (part[None, :] < PARTS)
        sum2 = tl.sum(
            tl.load(Sum2 + rows[:, None] * PARTS + part[None, :], part_mask, other=0), 1
        )
        sum3 = tl.sum(
            tl.load(Sum3 + rows[:, None] * PARTS + part[None, :], part_mask, other=0), 1
        )
        if W is None:
            w = 1.0
        else:
            w = tl.load(W + cols, col_mask, other=0).to(tl.float32)[None, :]
        dx = rstd[:, None] * (dy * w - (sum2[:, None] + xhat * sum3[:, None]) / N)
        tl.store(dX + rows[:, None] * N + cols[None, :], dx, mask)


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    logger.debug("GEMS_HYGON LAYERNORM_FORWARD")

    normalized_shape = tuple(normalized_shape)
    if not normalized_shape:
        raise RuntimeError("Expected normalized_shape to be at least 1-dimensional")
    if (
        len(normalized_shape) > input.ndim
        or input.shape[-len(normalized_shape) :] != normalized_shape
    ):
        raise RuntimeError(
            "Expected input with trailing dimensions matching normalized_shape"
        )
    for name, affine in (("weight", weight), ("bias", bias)):
        if affine is not None and affine.shape != normalized_shape:
            raise RuntimeError(f"Expected {name} shape to match normalized_shape")

    N = math.prod(normalized_shape)
    M = input.numel() // N

    input = input.contiguous()
    weight = None if weight is None else weight.contiguous()
    bias = None if bias is None else bias.contiguous()
    y = torch.empty_like(input)

    # NOTE: when the input is half-precision(either float16 or bfloat16)
    # these statistical data saved for backward is in single precision
    mean = torch.empty(M, dtype=torch.float32, device=input.device)
    rstd = torch.empty_like(mean)

    multiline = layer_norm_persistent_kernel_multiline
    persistent = layer_norm_persistent_kernel
    loop = layer_norm_loop_kernel
    moments = {}
    if input.dtype == torch.float32:
        multiline, persistent, loop = (
            _precise_multiline,
            _precise_persistent,
            _precise_loop,
        )
        depth = max(1, (triton.cdiv(N // 8, 16) - 1).bit_length())
        moments = dict(
            USE_CASCADE=True,
            MOMENTS_N=N,
            MOMENTS_DEPTH=depth,
            MOMENTS_STACK=triton.next_power_of_2(depth),
            enable_fp_fusion=False,
        )
    with torch_device_fn.device(input.device):
        if N <= 128:
            TILE_N = triton.next_power_of_2(N)
            TILE_M = triton.cdiv(1024, TILE_N)
            grid = (triton.cdiv(M, TILE_M), 1, 1)
            multiline[grid](
                input,
                y,
                weight,
                bias,
                mean,
                rstd,
                M,
                N,
                eps,
                TILE_M,
                TILE_N,
                **moments,
            )
        elif N <= 4096:
            TILE_N = triton.next_power_of_2(N)
            grid = (M, 1, 1)
            persistent[grid](
                input,
                y,
                weight,
                bias,
                mean,
                rstd,
                M,
                N,
                eps,
                TILE_N,
                **moments,
            )
        else:
            grid = (M, 1, 1)
            loop[grid](
                input,
                y,
                weight,
                bias,
                mean,
                rstd,
                M,
                N,
                eps,
                **moments,
            )
    return y, mean, rstd


def layer_norm_backward(
    grad_out,
    input,
    normalized_shape,
    mean,
    rstd,
    weight=None,
    bias=None,
    output_mask=None,
):
    logger.debug("GEMS_HYGON LAYERNORM_BACKWARD")

    grad_out = grad_out.contiguous()
    input = input.contiguous()
    mean = mean.contiguous()
    rstd = rstd.contiguous()
    weight = None if weight is None else weight.contiguous()
    bias = None if bias is None else bias.contiguous()

    N = math.prod(normalized_shape)
    M = input.numel() // N

    # Small row counts benefit from parallel column tiles that write all gradients.
    if (
        0 < M <= 32
        and N > 0
        and input.dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        in_grad = torch.empty_like(input) if output_mask[0] else None
        weight_grad = torch.empty_like(weight) if output_mask[1] else None
        bias_grad = torch.empty_like(bias) if output_mask[2] else None
        if not any(output_mask):
            return in_grad, weight_grad, bias_grad

        if N <= 16384:
            block = triton.next_power_of_2(N)
            with torch_device_fn.device(input.device):
                if output_mask[0]:
                    _ln_bwd_row_dx[(M,)](
                        grad_out,
                        input,
                        weight,
                        mean,
                        rstd,
                        in_grad,
                        M,
                        N,
                        BLOCK=block,
                        num_warps=4 if block <= 4096 else 8,
                    )
                if output_mask[1] or output_mask[2]:
                    _ln_bwd_fused_columns[(triton.cdiv(N, 128),)](
                        grad_out,
                        input,
                        weight,
                        mean,
                        rstd,
                        None,
                        None,
                        None,
                        weight_grad,
                        bias_grad,
                        M,
                        N,
                        PARTS=1,
                        TILE_M=triton.next_power_of_2(M),
                        TILE_N=128,
                        TILE_PART=1,
                        num_warps=4,
                    )
            return in_grad, weight_grad, bias_grad

        block = min(4096, triton.next_power_of_2(N))
        parts = triton.cdiv(N, block)
        sum2 = (
            torch.empty((M, parts), dtype=torch.float32, device=input.device)
            if output_mask[0]
            else None
        )
        sum3 = torch.empty_like(sum2) if output_mask[0] else None
        with torch_device_fn.device(input.device):
            if output_mask[0]:
                _ln_bwd_row_sums[(M, parts)](
                    grad_out,
                    input,
                    weight,
                    mean,
                    rstd,
                    sum2,
                    sum3,
                    M,
                    N,
                    BLOCK=block,
                    num_warps=4,
                )
            _ln_bwd_fused_columns[(triton.cdiv(N, 128),)](
                grad_out,
                input,
                weight,
                mean,
                rstd,
                sum2,
                sum3,
                in_grad,
                weight_grad,
                bias_grad,
                M,
                N,
                PARTS=parts,
                TILE_M=triton.next_power_of_2(M),
                TILE_N=128,
                TILE_PART=triton.next_power_of_2(parts),
                num_warps=4,
            )
        return in_grad, weight_grad, bias_grad

    if output_mask[0]:
        in_grad = torch.empty_like(input)
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_ROW_SIZE"]), 1, 1)
        with torch_device_fn.device(input.device):
            layer_norm_backward_kernel[grid](
                grad_out, input, weight, mean, rstd, in_grad, M, N
            )
    else:
        in_grad = None

    if output_mask[1] is False and output_mask[2] is False:
        return in_grad, None, None

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_COL_SIZE"]), 1, 1)
    weight_grad = torch.empty_like(weight) if output_mask[1] else None
    bias_grad = torch.empty_like(bias) if output_mask[2] else None
    with torch_device_fn.device(input.device):
        weight_bias_backward_kernel[grid](
            grad_out, input, mean, rstd, weight_grad, bias_grad, M, N
        )
    return in_grad, weight_grad, bias_grad


def native_layer_norm(
    input: torch.Tensor,
    normalized_shape: Sequence[int],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logger.debug("GEMS_HYGON NATIVE_LAYER_NORM")
    output, mean, rstd = layer_norm(input, normalized_shape, weight, bias, eps)
    stats_shape = input.shape[: -len(normalized_shape)] + (1,) * len(normalized_shape)
    return output, mean.view(stats_shape), rstd.view(stats_shape)
