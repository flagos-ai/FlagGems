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
from triton.runtime import driver

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

_HYGON = runtime.device.vendor_name == "hygon"
_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_LAUNCH_CACHE = {}
try:
    from c_operators import allocate_layer_norm_outputs_1d as ALLOCATE_1D
except ImportError:
    # Pure-Python installations remain usable without the optional C++ wheel.
    def ALLOCATE_1D(input):
        new_empty = input.new_empty
        shape = input.shape[:-1] + (1,)
        return (
            new_empty(input.shape),
            new_empty(shape, dtype=torch.float32),
            new_empty(shape, dtype=torch.float32),
        )


@libentry()
@triton.jit(do_not_specialize=["X", "Y", "W", "B", "Mean", "Rstd", "EPS"])
def _strided_row_layer_norm(
    X,
    Y,
    W,
    B,
    Mean,
    Rstd,
    N: tl.constexpr,
    EPS,
    SHAPE: tl.constexpr,
    STRIDE: tl.constexpr,
    WS: tl.constexpr,
    BS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    remaining = row
    offset = tl.full((), 0, tl.int64)
    for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
        offset += (remaining % SHAPE[axis]).to(tl.int64) * STRIDE[axis]
        remaining = remaining // SHAPE[axis]
    col = tl.arange(0, BLOCK)
    valid = col < N
    x = tl.load(X + offset + col, valid, other=0).to(tl.float32)
    mean = tl.sum(x, 0) / N
    delta = x - mean
    var = tl.sum(tl.where(valid, delta * delta, 0), 0) / N
    rstd = tl.rsqrt(var + EPS)
    out = delta * rstd
    if W is not None:
        out = out * tl.load(W + col * WS, valid, other=1).to(tl.float32)
    if B is not None:
        out = out + tl.load(B + col * BS, valid, other=0).to(tl.float32)
    tl.store(Y + row * N + col, out, valid)
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)


def _can_use_strided(input, normalized_shape, weight, bias):
    if not _HYGON or input.device.type not in ("cuda", "flagos", "privateuseone"):
        return False
    if input.dtype not in _DTYPES or len(normalized_shape) != 1 or input.ndim < 1:
        return False
    n = normalized_shape[0]
    if not (0 < n <= 4096) or input.shape[-1] != n or input.numel() == 0:
        return False
    if input.stride(-1) != 1 or input.ndim > 8:
        return False
    for affine in (weight, bias):
        if affine is not None and (
            affine.shape != (n,)
            or affine.dtype != input.dtype
            or affine.device != input.device
            or affine.stride(0) < 0
        ):
            return False
    return True


def native_layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    logger.debug("GEMS_HYGON NATIVE_LAYER_NORM")
    # Leave compiler/subclass dispatch on the original implementation. The
    # prepared eager launch assumes concrete metadata and ordinary Tensors.
    if (
        type(input) is not torch.Tensor
        or torch.compiler.is_compiling()
        or (weight is not None and type(weight) is not torch.Tensor)
        or (bias is not None and type(bias) is not torch.Tensor)
    ):
        return _generic_native(input, normalized_shape, weight, bias, eps)
    # Materialize logical negative values before passing storage to Triton.
    if input.is_neg():
        input = input.resolve_neg()
    if weight is not None and weight.is_neg():
        weight = weight.resolve_neg()
    if bias is not None and bias.is_neg():
        bias = bias.resolve_neg()
    device, dtype = input.device, input.dtype
    shape, stride = input.shape, input.stride()
    wmeta = (
        None
        if weight is None
        else (weight.shape, weight.stride(), weight.dtype, weight.device)
    )
    bmeta = (
        None if bias is None else (bias.shape, bias.stride(), bias.dtype, bias.device)
    )
    # All metadata consulted by eligibility and compilation participates in the
    # key. A new Tensor with the same metadata is safe; a resized or re-strided
    # Tensor cannot silently reuse an incompatible validated plan.
    key = (
        device,
        dtype,
        shape,
        stride,
        tuple(normalized_shape),
        wmeta,
        bmeta,
        type(eps),
    )
    plan = _LAUNCH_CACHE.get(key)
    if plan is None:
        if not _can_use_strided(input, normalized_shape, weight, bias):
            return _generic_native(input, normalized_shape, weight, bias, eps)
        n = normalized_shape[0]
        stats_shape = shape[:-1] + (1,)
        block = 1 << (n - 1).bit_length()
        grid = (input.numel() // n, 1, 1)
        ws = 0 if weight is None else wmeta[1][0]
        bs = 0 if bias is None else bmeta[1][0]
        constants = (n, shape[:-1], stride[:-1], ws, bs, block)
        # Only immutable metadata and compiled objects are saved. Outputs,
        # current streams, Tensor objects and their addresses are never saved.
        plan = [stats_shape, grid, constants, None]
        if len(_LAUNCH_CACHE) >= 128:
            _LAUNCH_CACHE.clear()
        _LAUNCH_CACHE[key] = plan
    stats_shape, grid, constants, executable = plan
    out, mean, rstd = ALLOCATE_1D(input)
    n, leading_shape, leading_stride, ws, bs, block = constants
    args = (
        input,
        out,
        weight,
        bias,
        mean,
        rstd,
        n,
        eps,
        leading_shape,
        leading_stride,
        ws,
        bs,
        block,
    )
    # Only avoid an exchange when the current thread is already on the input
    # device. A real device change retains the restoring standard guard.
    if torch_device_fn.current_device() == device.index:
        _submit(plan, args, device.index)
    else:
        with torch_device_fn.device(device.index):
            _submit(plan, args, device.index)
    return out, mean, rstd


def _submit(plan, args, index):
    _, grid, _, executable = plan
    if executable is None:
        kernel, _ = _strided_row_layer_norm[grid](
            *args,
            num_warps=4,
            enable_fp_fusion=False,
        )
        # Cache only the compiled kernel; construct its standard runner at
        # each warm invocation rather than retaining a separate callable.
        plan[3] = kernel
        return
    # Standard runner owns stream lookup, metadata, hooks and launch.
    kernel = executable
    runner = kernel[grid]
    runner(*args, stream=driver.active.get_current_stream(index))


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


def _layer_norm_fallback(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    logger.debug("GEMS_HYGON LAYERNORM_FORWARD")

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


def _generic_native(
    input: torch.Tensor,
    normalized_shape: Sequence[int],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    output, mean, rstd = _layer_norm_fallback(
        input, normalized_shape, weight, bias, eps
    )
    stats_shape = input.shape[: -len(normalized_shape)] + (1,) * len(normalized_shape)
    return output, mean.view(stats_shape), rstd.view(stats_shape)


layer_norm = native_layer_norm
