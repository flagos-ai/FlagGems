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
from typing import Optional, Sequence, Union

import torch
import triton
import triton.language as tl

from flag_gems import runtime
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

    mean = tl.load(Mean, mask=row_mask).to(tl.float32)
    rstd = tl.load(Rstd, mask=row_mask).to(tl.float32)

    dx_part2 = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    dx_part3 = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)

    for off in range(0, N, BLOCK_COL_SIZE):
        cols = off + tl.arange(0, BLOCK_COL_SIZE)
        col_mask = cols[None, :] < N
        mask = row_mask and col_mask
        dy = tl.load(dY + cols[None, :], mask).to(tl.float32)
        x = tl.load(X + cols[None, :], mask).to(tl.float32)
        x = tl.where(mask, x - mean, 0.0)
        x_hat = x * rstd
        if W is None:
            w = 1
        else:
            w = tl.load(W + cols, mask=cols < N).to(tl.float32)
        dx_hat = dy * w
        dx_part2 += dx_hat
        dx_part3 += dx_hat * x_hat

    dx_2 = tl.sum(dx_part2, axis=1)[:, None]
    dx_3 = tl.sum(dx_part3, axis=1)[:, None]

    for off in range(0, N, BLOCK_COL_SIZE):
        cols = off + tl.arange(0, BLOCK_COL_SIZE)
        col_mask = cols[None, :] < N
        mask = row_mask and col_mask
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
    dY += pid[:]
    X += pid[:]
    accW = tl.zeros([BLOCK_COL_SIZE], dtype=tl.float32)
    accB = tl.zeros([BLOCK_COL_SIZE], dtype=tl.float32)
    for off in range(0, M, BLOCK_ROW_SIZE):
        rows = off
        mask = col_mask[:]
        dy = tl.load(dY + rows * N, mask).to(tl.float32)
        x = tl.load(X + rows * N, mask).to(tl.float32)
        mean = tl.load(Mean + rows, mask=rows < M).to(tl.float32)
        rstd = tl.load(Rstd + rows, mask=rows < M).to(tl.float32)
        x = tl.where(mask, x - mean, 0.0)
        accW += dy * x * rstd
        accB += dy
    if dW:
        tl.store(dW + pid, accW, mask=col_mask)
    if dB:
        tl.store(dB + pid, accB, mask=col_mask)


def layer_norm(
    input: torch.Tensor,
    normalized_shape: Union[int, Sequence[int]],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logger.debug("GEMS_METAX LAYERNORM_FORWARD")
    normalized_shape = (
        (normalized_shape,)
        if isinstance(normalized_shape, int)
        else tuple(normalized_shape)
    )
    if not normalized_shape or any(size <= 0 for size in normalized_shape):
        raise ValueError("normalized_shape must contain positive dimensions")
    if tuple(input.shape[-len(normalized_shape) :]) != normalized_shape:
        raise ValueError("normalized_shape must match the trailing input dimensions")
    if input.device.type != "cuda":
        raise NotImplementedError("layer_norm requires a MetaX CUDA-compatible device")
    if input.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise NotImplementedError("layer_norm supports FP16, BF16 and FP32")
    for affine in (weight, bias):
        if affine is None:
            continue
        if tuple(affine.shape) != normalized_shape:
            raise ValueError("affine shape must equal normalized_shape")
        if affine.device != input.device:
            raise ValueError("affine device must match input")
        if affine.dtype != input.dtype and not (
            input.dtype in (torch.float16, torch.bfloat16)
            and affine.dtype == torch.float32
        ):
            raise ValueError("affine dtype must match input or be FP32 for half input")
    width = math.prod(normalized_shape)
    rows = input.numel() // width
    output = torch.empty(input.shape, dtype=input.dtype, device=input.device)
    # Preserve the flat, input-dtype statistics expected by the existing backward.
    mean = torch.empty(rows, dtype=input.dtype, device=input.device)
    rstd = torch.empty_like(mean)
    if rows == 0:
        return output, mean, rstd
    if width > 65536:
        # The existing streaming kernel covers normalized widths beyond one tile.
        input = input.contiguous()
        weight = None if weight is None else weight.contiguous()
        bias = None if bias is None else bias.contiguous()
        with torch_device_fn.device(input.device):
            layer_norm_loop_kernel[(rows, 1, 1)](
                input, output, weight, bias, mean, rstd, rows, width, eps
            )
        return output, mean, rstd
    warps = 1 if width <= 4096 else (4 if width <= 8192 else 8)
    is_dense_affine = all(
        affine is None or affine.is_contiguous() for affine in (weight, bias)
    )
    body = 1 << (width.bit_length() - 1)
    tail = width - body
    use_split = (
        input.is_contiguous()
        and is_dense_affine
        and width <= 4096
        and rows >= 128
        and body >= 256
        and tail >= 64
        and tail & (tail - 1) == 0
    )
    with torch_device_fn.device(input.device):
        if use_split:
            layernorm_split_kernel[(rows,)](
                input,
                weight,
                bias,
                output,
                mean,
                rstd,
                width,
                eps,
                body,
                tail,
                weight is not None,
                bias is not None,
                num_warps=1,
                enable_fp_fusion=False,
            )
        else:
            layernorm_kernel[(rows,)](
                input,
                weight,
                bias,
                output,
                mean,
                rstd,
                rows,
                width,
                eps,
                tuple(input.shape),
                tuple(input.stride()),
                normalized_shape,
                tuple(weight.stride()) if weight is not None else (),
                tuple(bias.stride()) if bias is not None else (),
                input.is_contiguous(),
                weight is not None,
                bias is not None,
                1,
                triton.next_power_of_2(width),
                num_warps=warps,
                enable_fp_fusion=False,
            )
    return output, mean, rstd


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
    logger.debug("GEMS_METAX LAYERNORM_BACKWARD")

    grad_out = grad_out.contiguous()
    input = input.contiguous()
    mean = mean.contiguous()
    rstd = rstd.contiguous()
    weight = None if weight is None else weight.contiguous()
    bias = None if bias is None else bias.contiguous()

    N = math.prod(normalized_shape)
    M = input.numel() // N

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


@triton.jit
def layernorm_split_kernel(
    Input,
    Weight,
    Bias,
    Output,
    Mean,
    Rstd,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BODY: tl.constexpr,
    TAIL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    row = tl.program_id(0)
    body_cols = tl.arange(0, BODY)
    tail_cols = BODY + tl.arange(0, TAIL)
    body = tl.load(Input + row * N + body_cols).to(tl.float32)
    tail = tl.load(Input + row * N + tail_cols).to(tl.float32)
    mean = (tl.sum(body, 0) + tl.sum(tail, 0)) / N
    body_delta = body - mean
    tail_delta = tail - mean
    variance = (
        tl.sum(body_delta * body_delta, 0) + tl.sum(tail_delta * tail_delta, 0)
    ) / N
    rstd = tl.rsqrt(variance + EPS)
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    body_out = body_delta * rstd
    tail_out = tail_delta * rstd
    if HAS_WEIGHT:
        body_out = body_out * tl.load(Weight + body_cols).to(tl.float32)
        tail_out = tail_out * tl.load(Weight + tail_cols).to(tl.float32)
    if HAS_BIAS:
        body_out = body_out + tl.load(Bias + body_cols).to(tl.float32)
        tail_out = tail_out + tl.load(Bias + tail_cols).to(tl.float32)
    tl.store(Output + row * N + body_cols, body_out)
    tl.store(Output + row * N + tail_cols, tail_out)


@triton.jit
def layernorm_kernel(
    Input,
    Weight,
    Bias,
    Output,
    Mean,
    Rstd,
    M: tl.constexpr,
    N: tl.constexpr,
    EPS: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    NORMALIZED: tl.constexpr,
    WSTRIDES: tl.constexpr,
    BSTRIDES: tl.constexpr,
    CONTIGUOUS: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, COLS)
    if CONTIGUOUS:
        offsets = rows[:, None] * N + cols[None, :]
    else:
        row_offsets = tl.full((ROWS,), 0, tl.int64)
        col_offsets = tl.full((COLS,), 0, tl.int64)
        row_index = rows
        col_index = cols
        for dim in tl.static_range(len(SHAPE) - len(NORMALIZED) - 1, -1, -1):
            row_offsets += (row_index % SHAPE[dim]) * STRIDES[dim]
            row_index = row_index // SHAPE[dim]
        for dim in tl.static_range(len(NORMALIZED) - 1, -1, -1):
            col_offsets += (col_index % NORMALIZED[dim]) * STRIDES[
                len(SHAPE) - len(NORMALIZED) + dim
            ]
            col_index = col_index // NORMALIZED[dim]
        offsets = row_offsets[:, None] + col_offsets[None, :]
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    values = tl.load(Input + offsets, mask, other=0).to(tl.float32)
    mean = tl.sum(values, 1) / N
    centered = values - mean[:, None]
    centered_masked = tl.where(cols[None, :] < N, centered, 0.0)
    variance = tl.sum(centered_masked * centered_masked, 1) / N
    rstd = tl.rsqrt(variance + EPS)
    tl.store(Mean + rows, mean, rows < M)
    tl.store(Rstd + rows, rstd, rows < M)
    normalized = centered * rstd[:, None]
    if HAS_WEIGHT:
        weight_offsets = tl.full((COLS,), 0, tl.int64)
        index = cols
        for dim in tl.static_range(len(NORMALIZED) - 1, -1, -1):
            weight_offsets += (index % NORMALIZED[dim]) * WSTRIDES[dim]
            index = index // NORMALIZED[dim]
        weight = tl.load(Weight + weight_offsets, cols < N, other=0).to(tl.float32)
        normalized = normalized * weight[None, :]
    if HAS_BIAS:
        bias_offsets = tl.full((COLS,), 0, tl.int64)
        index = cols
        for dim in tl.static_range(len(NORMALIZED) - 1, -1, -1):
            bias_offsets += (index % NORMALIZED[dim]) * BSTRIDES[dim]
            index = index // NORMALIZED[dim]
        bias = tl.load(Bias + bias_offsets, cols < N, other=0).to(tl.float32)
        normalized = normalized + bias[None, :]
    tl.store(Output + rows[:, None] * N + cols[None, :], normalized, mask)
