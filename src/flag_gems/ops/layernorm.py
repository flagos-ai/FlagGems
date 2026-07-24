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

import functools
import logging
import math

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


@functools.lru_cache(maxsize=None)
def _layer_norm_parallel_units(device_index):
    props = torch_device_fn.get_device_properties(device_index)
    # Device runtimes expose the compute-unit count under different names.
    property_names = (
        "multi_processor_count",
        "multiProcessorCount",
        "vector_core_num",
        "num_vectorcore",
        "cube_core_num",
        "num_aicore",
    )
    for name in property_names:
        value = (
            props.get(name) if isinstance(props, dict) else getattr(props, name, None)
        )
        if value:
            return int(value)
    return None


def _fused_layer_norm_backward_config(input, output_mask, M, N, weight, bias):
    if output_mask is None or M <= 0 or N <= 0:
        return None

    compute_dx, compute_dw, compute_db = map(bool, output_mask)
    if (
        input.dtype not in (torch.float32, torch.float16, torch.bfloat16)
        or not compute_dx
        or not (compute_dw or compute_db)
        or (compute_dw and weight is None)
        or (compute_db and bias is None)
    ):
        return None

    tile_n = triton.next_power_of_2(N)
    args = {
        "M": M,
        "N": N,
        "TILE_N": tile_n,
        "IS_LOW_PRECISION": input.dtype != torch.float32,
    }
    # Backends opt in to the resident path by providing this heuristic group.
    fused_heuristics = runtime.get_heuristic_config("layer_norm_backward_fused")
    if fused_heuristics is None:
        return None

    max_resident_n = fused_heuristics["MAX_RESIDENT_N"](args)
    enough_work = M * N >= fused_heuristics["MIN_ELEMENTS"](args)
    if tile_n > max_resident_n or not enough_work:
        return None

    parallel_units = _layer_norm_parallel_units(torch_device_fn.current_device())
    if parallel_units is None:
        return None

    direct_lowp_atomic_fn = fused_heuristics.get("DIRECT_LOWP_ATOMIC")
    direct_lowp_atomic = (
        direct_lowp_atomic_fn(args) if direct_lowp_atomic_fn is not None else False
    )
    # Accumulate through FP32 scratch when low-precision atomics are unavailable.
    use_fp32_scratch = input.dtype != torch.float32 and not direct_lowp_atomic

    # Large M is handled by the kernel's grid-stride row loop.
    tile_elements = fused_heuristics["TILE_ELEMENTS"](args)
    program_waves = fused_heuristics["PROGRAM_WAVES"](args)
    block_m = max(1, tile_elements // tile_n)
    max_block_m_fn = fused_heuristics.get("MAX_BLOCK_M")
    if max_block_m_fn is not None:
        block_m = min(block_m, max_block_m_fn(args))
    row_tiles = triton.cdiv(M, block_m)
    program_count = min(row_tiles, parallel_units * program_waves)
    return (
        block_m,
        tile_n,
        program_count,
        compute_dw,
        compute_db,
        use_fp32_scratch,
    )


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
@triton.jit
def layer_norm_backward_zero_affine_kernel(
    dW,
    dB,
    N,
    BLOCK_SIZE: tl.constexpr,
    COMPUTE_DW: tl.constexpr,
    COMPUTE_DB: tl.constexpr,
):
    cols = ext.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    if COMPUTE_DW:
        tl.store(dW + cols, 0.0, mask=mask)
    if COMPUTE_DB:
        tl.store(dB + cols, 0.0, mask=mask)


@libentry()
@triton.jit
def layer_norm_backward_cast_affine_kernel(
    dWAcc,
    dBAcc,
    dW,
    dB,
    N,
    BLOCK_SIZE: tl.constexpr,
    COMPUTE_DW: tl.constexpr,
    COMPUTE_DB: tl.constexpr,
):
    cols = ext.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    if COMPUTE_DW:
        tl.store(dW + cols, tl.load(dWAcc + cols, mask=mask), mask=mask)
    if COMPUTE_DB:
        tl.store(dB + cols, tl.load(dBAcc + cols, mask=mask), mask=mask)


@libentry()
@triton.jit
def layer_norm_backward_resident_kernel(
    dY,
    X,
    W,
    Mean,
    Rstd,
    dX,
    dW,
    dB,
    M,
    N: tl.constexpr,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    COMPUTE_DW: tl.constexpr,
    COMPUTE_DB: tl.constexpr,
):
    pid = ext.program_id(0)
    program_count = ext.num_programs(0)
    cols = tl.arange(0, BLOCK_COL_SIZE)
    col_mask = cols < N
    if HAS_WEIGHT:
        weight = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
    else:
        weight = 1.0
    if COMPUTE_DW:
        partial_dw = tl.zeros([BLOCK_COL_SIZE], dtype=tl.float32)
    if COMPUTE_DB:
        partial_db = tl.zeros([BLOCK_COL_SIZE], dtype=tl.float32)

    # Reuse the dX input scan and reduce one affine partial per program.
    for row_start in range(pid * BLOCK_ROW_SIZE, M, program_count * BLOCK_ROW_SIZE):
        rows = row_start + tl.arange(0, BLOCK_ROW_SIZE)
        row_mask = rows < M
        mask = row_mask[:, None] & col_mask[None, :]
        offsets = rows[:, None] * N + cols[None, :]

        dy = tl.load(dY + offsets, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(Mean + rows, mask=row_mask, other=0.0)[:, None].to(tl.float32)
        rstd = tl.load(Rstd + rows, mask=row_mask, other=0.0)[:, None].to(tl.float32)

        x_hat = (x - mean) * rstd
        dx_hat = dy * weight
        sum_dx_hat = tl.sum(tl.where(mask, dx_hat, 0.0), axis=1)[:, None]
        sum_dx_hat_x = tl.sum(tl.where(mask, dx_hat * x_hat, 0.0), axis=1)[:, None]
        dx = rstd * (dx_hat - (sum_dx_hat + x_hat * sum_dx_hat_x) / N)
        tl.store(dX + offsets, dx, mask=mask)

        if COMPUTE_DW:
            partial_dw += tl.sum(tl.where(mask, dy * x_hat, 0.0), axis=0)
        if COMPUTE_DB:
            partial_db += tl.sum(tl.where(mask, dy, 0.0), axis=0)

    if COMPUTE_DW:
        tl.atomic_add(dW + cols, partial_dw, mask=col_mask)
    if COMPUTE_DB:
        tl.atomic_add(dB + cols, partial_db, mask=col_mask)


def _launch_layer_norm_backward_resident(
    grad_out,
    input,
    mean,
    rstd,
    weight,
    bias,
    M,
    N,
    block_row_size,
    block_col_size,
    program_count,
    compute_dw,
    compute_db,
    use_fp32_scratch,
):
    in_grad = torch.empty_like(input)
    weight_grad = torch.empty_like(weight) if compute_dw else None
    bias_grad = torch.empty_like(bias) if compute_db else None

    # Atomic affine updates require zero-initialized accumulation buffers.
    affine_scratch = (
        torch.empty((2, N), dtype=torch.float32, device=input.device)
        if use_fp32_scratch
        else None
    )
    weight_acc = affine_scratch[0] if use_fp32_scratch and compute_dw else weight_grad
    bias_acc = affine_scratch[1] if use_fp32_scratch and compute_db else bias_grad

    with torch_device_fn.device(input.device):
        affine_block_size = 256
        if compute_dw or compute_db:
            layer_norm_backward_zero_affine_kernel[
                (triton.cdiv(N, affine_block_size), 1, 1)
            ](
                weight_acc,
                bias_acc,
                N,
                BLOCK_SIZE=affine_block_size,
                COMPUTE_DW=compute_dw,
                COMPUTE_DB=compute_db,
            )
        layer_norm_backward_resident_kernel[(program_count, 1, 1)](
            grad_out,
            input,
            weight,
            mean,
            rstd,
            in_grad,
            weight_acc,
            bias_acc,
            M,
            N=N,
            BLOCK_ROW_SIZE=block_row_size,
            BLOCK_COL_SIZE=block_col_size,
            HAS_WEIGHT=weight is not None,
            COMPUTE_DW=compute_dw,
            COMPUTE_DB=compute_db,
        )
        if use_fp32_scratch:
            layer_norm_backward_cast_affine_kernel[
                (triton.cdiv(N, affine_block_size), 1, 1)
            ](
                weight_acc,
                bias_acc,
                weight_grad,
                bias_grad,
                N,
                BLOCK_SIZE=affine_block_size,
                COMPUTE_DW=compute_dw,
                COMPUTE_DB=compute_db,
            )
    return in_grad, weight_grad, bias_grad


def _launch_fused_layer_norm_backward(
    grad_out,
    input,
    mean,
    rstd,
    weight,
    bias,
    output_mask,
    M,
    N,
):
    config = _fused_layer_norm_backward_config(input, output_mask, M, N, weight, bias)
    if config is None:
        return None

    return _launch_layer_norm_backward_resident(
        grad_out,
        input,
        mean,
        rstd,
        weight,
        bias,
        M,
        N,
        *config,
    )


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
        mask = row_mask and col_mask[None, :]
        dy = tl.load(dY + rows * N, mask).to(tl.float32)
        x = tl.load(X + rows * N, mask).to(tl.float32)
        mean = tl.load(Mean + rows, mask=rows < M).to(tl.float32)
        rstd = tl.load(Rstd + rows, mask=rows < M).to(tl.float32)
        x = tl.where(mask, x - mean, 0.0)
        accW += dy * x * rstd
        accB += dy
    if dW:
        dw = tl.sum(accW, axis=0)
        tl.store(dW + pid, dw, mask=col_mask)
    if dB:
        db = tl.sum(accB, axis=0)
        tl.store(dB + pid, db, mask=col_mask)


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    logger.debug("GEMS LAYERNORM FORWARD")

    N = math.prod(normalized_shape)
    M = input.numel() // N

    input = input.contiguous()
    weight = None if weight is None else weight.contiguous()
    bias = None if bias is None else bias.contiguous()
    y = torch.empty_like(input)

    # NOTE: when the input is half-precision(either float16 or bfloat16)
    # these statistical data saved for backward is in single precision
    mean = torch.empty(M, dtype=input.dtype, device=input.device)
    rstd = torch.empty(M, dtype=input.dtype, device=input.device)

    with torch_device_fn.device(input.device):
        if N <= 128:
            TILE_N = triton.next_power_of_2(N)
            TILE_M = triton.cdiv(1024, TILE_N)
            grid = (triton.cdiv(M, TILE_M), 1, 1)
            layer_norm_persistent_kernel_multiline[grid](
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
            )
        elif N <= 4096:
            TILE_N = triton.next_power_of_2(N)
            grid = (M, 1, 1)
            layer_norm_persistent_kernel[grid](
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
            )
        else:
            grid = (M, 1, 1)
            layer_norm_loop_kernel[grid](
                input,
                y,
                weight,
                bias,
                mean,
                rstd,
                M,
                N,
                eps,
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
    logger.debug("GEMS LAYERNORM BACKWARD")

    grad_out = grad_out if grad_out.is_contiguous() else grad_out.contiguous()
    input = input if input.is_contiguous() else input.contiguous()
    mean = mean if mean.is_contiguous() else mean.contiguous()
    rstd = rstd if rstd.is_contiguous() else rstd.contiguous()
    if weight is not None and not weight.is_contiguous():
        weight = weight.contiguous()
    if bias is not None and not bias.is_contiguous():
        bias = bias.contiguous()

    # LayerNorm flattens all leading dimensions into M and normalizes over N.
    N = math.prod(normalized_shape)
    M = input.numel() // N

    # Unsupported shapes or backends without heuristics keep the upstream path.
    fused_grads = _launch_fused_layer_norm_backward(
        grad_out,
        input,
        mean,
        rstd,
        weight,
        bias,
        output_mask,
        M,
        N,
    )
    if fused_grads is not None:
        return fused_grads

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

    weight_grad = torch.empty_like(weight) if output_mask[1] else None
    bias_grad = torch.empty_like(bias) if output_mask[2] else None
    with torch_device_fn.device(input.device):
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_COL_SIZE"]), 1, 1)
        weight_bias_backward_kernel[grid](
            grad_out, input, mean, rstd, weight_grad, bias_grad, M, N
        )
    return in_grad, weight_grad, bias_grad
