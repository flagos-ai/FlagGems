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
from functools import lru_cache

import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl

from flag_gems import runtime
from flag_gems.ops.layernorm import (
    _launch_layer_norm_backward_resident,
    _layer_norm_parallel_units,
)
from flag_gems.ops.layernorm import layer_norm as _generic_layer_norm
from flag_gems.ops.layernorm import (
    layer_norm_backward_kernel,
    weight_bias_backward_kernel,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

_ROW_LAYOUT = tle.gpu.BlockEncoding(
    size_per_thread=[1, 4],
    threads_per_warp=[1, 32],
    warps_per_cta=[4, 1],
    order=[1, 0],
)
_WEIGHT_LAYOUT = tle.gpu.SlicedEncoding(0, _ROW_LAYOUT)


@libentry()
@triton.jit(do_not_specialize=["M", "eps"])
def _layer_norm_kernel(
    X,
    W,
    B,
    Y,
    Mean,
    Rstd,
    M,
    N: tl.constexpr,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_TLE: tl.constexpr,
    ROW_LAYOUT: tl.constexpr,
    WEIGHT_LAYOUT: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    # S5000 profiling: keep affine values out of the wide reductions' live
    # register set. Retain the measured scheduling for other row widths.
    LATE_AFFINE: tl.constexpr = N == 65536
    if not LATE_AFFINE:
        w = tl.full((BLOCK_N,), 1, tl.float32)
        b = tl.full((BLOCK_N,), 0, tl.float32)
        if W is not None:
            w = tl.load(W + cols, cols < N, other=0).to(tl.float32)
        if B is not None:
            b = tl.load(B + cols, cols < N, other=0).to(tl.float32)
        if USE_TLE:
            w = tle.gpu.set_layout(w, WEIGHT_LAYOUT)
            b = tle.gpu.set_layout(b, WEIGHT_LAYOUT)
    offsets = rows[:, None] * N + cols[None, :]
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    x = tl.load(X + offsets, mask, other=0).to(tl.float32)
    anchor = tl.load(X + rows * N, rows < M, other=0).to(tl.float32)
    # Translation invariance improves constant / large-offset rows. Restore
    # the original mean when exporting the statistics used by backward.
    z = tl.where(cols[None, :] < N, x - anchor[:, None], 0.0)
    if USE_TLE:
        z = tle.gpu.set_layout(z, ROW_LAYOUT)
    mean = tl.sum(z, 1) / N
    centered = tl.where(cols[None, :] < N, z - mean[:, None], 0.0)
    variance = tl.sum(centered * centered, 1) / N
    rstd = tl.rsqrt(variance + eps)
    if LATE_AFFINE:
        y = centered * rstd[:, None]
        if W is not None:
            w = tl.load(W + cols, cols < N, other=0).to(tl.float32)
            if USE_TLE:
                w = tle.gpu.set_layout(w, WEIGHT_LAYOUT)
            y *= w[None, :]
        if B is not None:
            b = tl.load(B + cols, cols < N, other=0).to(tl.float32)
            if USE_TLE:
                b = tle.gpu.set_layout(b, WEIGHT_LAYOUT)
            y += b[None, :]
    else:
        y = centered * rstd[:, None] * w[None, :] + b[None, :]
    tl.store(Y + offsets, y, mask)
    tl.store(Mean + rows, mean + anchor, rows < M)
    tl.store(Rstd + rows, rstd, rows < M)


_TLE_SHAPES = {
    torch.float16: {
        (256, 128),
        (256, 1024),
        (4096, 128),
        (4096, 768),
        (4096, 1024),
        (32768, 256),
    },
    torch.bfloat16: {
        (65536, 1152),
        (256, 128),
        (4096, 128),
        (4096, 768),
        (4096, 1024),
        (32768, 256),
    },
}


@lru_cache(maxsize=None)
def _device_name(index):
    return torch_device_fn.get_device_name(index)


def _select_config(m, n, dtype):
    # S5000 three-output sweep: retain >=5% wins over the stable single-row
    # kernel. Unknown shapes use the baseline; no Y-only tuning table is used.
    if (m, n) in _TLE_SHAPES.get(dtype, ()):
        return 4, 4, True
    return 1, 4 if n <= 8192 else 16, False


@libentry()
@triton.jit(do_not_specialize=["eps"])
def _layer_norm_loop(X, W, B, Y, Mean, Rstd, N: tl.constexpr, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    anchor = tl.load(X + row * N).to(tl.float32)
    sums = tl.zeros((BLOCK,), tl.float32)
    for start in range(tl.cdiv(N, BLOCK)):
        idx = start * BLOCK + cols
        x = tl.load(X + row * N + idx, idx < N, other=0).to(tl.float32)
        sums += tl.where(idx < N, x - anchor, 0.0)
    mean = tl.sum(sums) / N
    squares = tl.zeros((BLOCK,), tl.float32)
    for start in range(tl.cdiv(N, BLOCK)):
        idx = start * BLOCK + cols
        x = tl.load(X + row * N + idx, idx < N, other=0).to(tl.float32)
        centered = tl.where(idx < N, (x - anchor) - mean, 0.0)
        squares += centered * centered
    rstd = tl.rsqrt(tl.sum(squares) / N + eps)
    tl.store(Mean + row, mean + anchor)
    tl.store(Rstd + row, rstd)
    for start in range(tl.cdiv(N, BLOCK)):
        idx = start * BLOCK + cols
        x = tl.load(X + row * N + idx, idx < N, other=0).to(tl.float32)
        y = ((x - anchor) - mean) * rstd
        if W is not None:
            y *= tl.load(W + idx, idx < N, other=0).to(tl.float32)
        if B is not None:
            y += tl.load(B + idx, idx < N, other=0).to(tl.float32)
        tl.store(Y + row * N + idx, y, idx < N)


@libentry()
@triton.jit
def _layer_norm_partial_stats(
    X, PartialMean, PartialM2, N: tl.constexpr, PARTS: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    part = tl.program_id(1)
    cols = part * BLOCK + tl.arange(0, BLOCK)
    count = tl.minimum(BLOCK, N - part * BLOCK)
    anchor = tl.load(X + row * N).to(tl.float32)
    values = tl.load(X + row * N + cols, cols < N, other=0).to(tl.float32)
    shifted = tl.where(cols < N, values - anchor, 0.0)
    mean = tl.sum(shifted, 0) / count
    centered = tl.where(cols < N, shifted - mean, 0.0)
    m2 = tl.sum(centered * centered, 0)
    tl.store(PartialMean + row * PARTS + part, mean)
    tl.store(PartialM2 + row * PARTS + part, m2)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def _layer_norm_partition_output(
    X,
    W,
    B,
    Y,
    Mean,
    Rstd,
    PartialMean,
    PartialM2,
    N: tl.constexpr,
    eps,
    PARTS: tl.constexpr,
    REDUCE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    part = tl.program_id(1)
    parts = tl.arange(0, REDUCE)
    counts = tl.where(parts < PARTS, tl.minimum(BLOCK, N - parts * BLOCK), 0)
    means = tl.load(PartialMean + row * PARTS + parts, parts < PARTS, other=0)
    m2 = tl.load(PartialM2 + row * PARTS + parts, parts < PARTS, other=0)
    mean = tl.sum(means * counts, 0) / N
    # Merge centered moments to avoid subtracting large squared means.
    deltas = means - mean
    variance = tl.sum(m2 + deltas * deltas * counts, 0) / N
    rstd = tl.rsqrt(variance + eps)
    anchor = tl.load(X + row * N).to(tl.float32)
    cols = part * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(X + row * N + cols, cols < N, other=0).to(tl.float32)
    output = ((values - anchor) - mean) * rstd
    if W is not None:
        output *= tl.load(W + cols, cols < N, other=0).to(tl.float32)
    if B is not None:
        output += tl.load(B + cols, cols < N, other=0).to(tl.float32)
    tl.store(Y + row * N + cols, output, cols < N)
    if part == 0:
        tl.store(Mean + row, mean + anchor)
        tl.store(Rstd + row, rstd)


def _layer_norm_impl(input, normalized_shape, weight, bias, eps, keepdim):
    logger.debug("GEMS_MTHREADS LAYERNORM_FORWARD")
    normalized_shape = tuple(normalized_shape)
    input_shape = input.shape
    if (
        not normalized_shape
        or len(normalized_shape) > input.ndim
        or tuple(input_shape[-len(normalized_shape) :]) != normalized_shape
    ):
        raise RuntimeError("normalized_shape must match the trailing input dimensions")
    n = math.prod(normalized_shape)
    if n <= 0:
        raise RuntimeError("normalized dimensions must be positive")
    input_device = input.device
    input_dtype = input.dtype
    for name, value in (("weight", weight), ("bias", bias)):
        if value is not None:
            if tuple(value.shape) != normalized_shape:
                raise RuntimeError(f"{name} shape must equal normalized_shape")
            if value.device != input_device:
                raise RuntimeError(f"{name} must be on the input device")
            if value.dtype != input_dtype and not (
                input_dtype in (torch.float16, torch.bfloat16)
                and value.dtype == torch.float32
            ):
                raise RuntimeError(f"{name} must match input dtype or be float32")
    if weight is not None and bias is not None and weight.dtype != bias.dtype:
        raise RuntimeError("weight and bias must have the same dtype")
    if input_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        result = _generic_layer_norm(input, normalized_shape, weight, bias, eps)
        if keepdim:
            stats_shape = input_shape[: -len(normalized_shape)] + (1,) * len(
                normalized_shape
            )
            return result[0], result[1].view(stats_shape), result[2].view(stats_shape)
        return result
    x = input.contiguous()
    w = None if weight is None else weight.contiguous()
    b = None if bias is None else bias.contiguous()
    m = input.numel() // n
    y = torch.empty_like(x)
    # ATen MUSA exports FP32 statistics for half / BF16 as well as FP32 input.
    # Native LayerNorm exposes keepdim statistics. Allocate their final shape
    # directly; the kernels still address the same contiguous M elements.
    stats_shape = (
        input_shape[: -len(normalized_shape)] + (1,) * len(normalized_shape)
        if keepdim
        else (m,)
    )
    mean = torch.empty(stats_shape, dtype=torch.float32, device=input_device)
    rstd = torch.empty_like(mean)
    if m:
        bm, nw, use_tle = _select_config(m, n, input_dtype)
        if use_tle and _device_name(input_device.index) != "MTT S5000":
            bm, nw, use_tle = 1, 4 if n <= 8192 else 16, False
        with torch_device_fn.device(input_device):
            # Bound each output tile to merging at most 128 partial moments.
            if 65536 < n <= 524288:
                parts = triton.cdiv(n, 4096)
                partial_mean = torch.empty(
                    (m, parts), dtype=torch.float32, device=x.device
                )
                partial_m2 = torch.empty_like(partial_mean)
                _layer_norm_partial_stats[(m, parts)](
                    x, partial_mean, partial_m2, n, parts, 4096, num_warps=4
                )
                _layer_norm_partition_output[(m, parts)](
                    x,
                    w,
                    b,
                    y,
                    mean,
                    rstd,
                    partial_mean,
                    partial_m2,
                    n,
                    eps,
                    parts,
                    triton.next_power_of_2(parts),
                    4096,
                    num_warps=4,
                )
            elif n > 65536:
                _layer_norm_loop[(m,)](
                    x, w, b, y, mean, rstd, n, eps, 4096, num_warps=4
                )
            else:
                _layer_norm_kernel[((m + bm - 1) // bm,)](
                    x,
                    w,
                    b,
                    y,
                    mean,
                    rstd,
                    m,
                    n,
                    eps,
                    bm,
                    triton.next_power_of_2(n),
                    use_tle,
                    _ROW_LAYOUT if use_tle else None,
                    _WEIGHT_LAYOUT if use_tle else None,
                    num_warps=nw,
                )
    return y, mean, rstd


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    return _layer_norm_impl(input, normalized_shape, weight, bias, eps, False)


def native_layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    logger.debug("GEMS_MTHREADS NATIVE_LAYER_NORM")
    return _layer_norm_impl(input, normalized_shape, weight, bias, eps, True)


def _layer_norm_backward_resident_dx_config(input, output_mask, M, N):
    if output_mask is None or M <= 0 or N <= 0:
        return None

    compute_dx, compute_dw, compute_db = map(bool, output_mask)
    # This dX launch is paired with the vendor affine reduction below.
    if not compute_dx or not (compute_dw or compute_db):
        return None
    if input.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return None

    heuristics = runtime.get_heuristic_config("layer_norm_backward_resident_dx")
    if heuristics is None:
        return None

    tile_n = triton.next_power_of_2(N)
    args = {
        "M": M,
        "N": N,
        "TILE_N": tile_n,
        "IS_LOW_PRECISION": input.dtype != torch.float32,
    }
    if M < heuristics["MIN_ROWS"](args):
        return None
    if tile_n > heuristics["MAX_RESIDENT_N"](args):
        return None

    parallel_units = _layer_norm_parallel_units(torch_device_fn.current_device())
    if parallel_units is None:
        return None

    if heuristics["FULL_ROW_GRID"](args):
        return 1, tile_n, M

    block_m = max(1, heuristics["TILE_ELEMENTS"](args) // tile_n)
    row_tiles = triton.cdiv(M, block_m)
    program_count = min(
        row_tiles,
        parallel_units * heuristics["PROGRAM_WAVES"](args),
    )
    return block_m, tile_n, program_count


# Stage 1: reduce one row group and one column tile into FP32 partials.
@libentry()
@triton.jit
def weight_bias_backward_partial_kernel(
    dY,
    X,
    Mean,
    Rstd,
    PartialW,
    PartialB,
    M,
    N,
    ROWS_PER_PROGRAM: tl.constexpr,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    row_group = ext.program_id(0)
    col_group = ext.program_id(1)
    row_start = row_group * ROWS_PER_PROGRAM
    cols = col_group * BLOCK_COL_SIZE + tl.arange(0, BLOCK_COL_SIZE)
    col_mask = cols < N
    acc_w = tl.zeros([BLOCK_COL_SIZE], dtype=tl.float32)
    acc_b = tl.zeros([BLOCK_COL_SIZE], dtype=tl.float32)

    for row_offset in range(0, ROWS_PER_PROGRAM, BLOCK_ROW_SIZE):
        rows = row_start + row_offset + tl.arange(0, BLOCK_ROW_SIZE)[:, None]
        mask = (rows < M) & col_mask[None, :]
        offsets = rows * N + cols[None, :]
        dy = tl.load(dY + offsets, mask=mask, other=0.0).to(tl.float32)
        if PartialW is not None:
            x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
            mean = tl.load(Mean + rows, mask=rows < M, other=0.0).to(tl.float32)
            rstd = tl.load(Rstd + rows, mask=rows < M, other=0.0).to(tl.float32)
            acc_w += tl.sum(dy * (x - mean) * rstd, axis=0)
        if PartialB is not None:
            acc_b += tl.sum(dy, axis=0)

    partial_offsets = row_group * N + cols
    if PartialW is not None:
        tl.store(PartialW + partial_offsets, acc_w, mask=col_mask)
    if PartialB is not None:
        tl.store(PartialB + partial_offsets, acc_b, mask=col_mask)


# Stage 2: reduce row-group partials into the final affine gradients.
@libentry()
@triton.jit
def weight_bias_backward_reduce_kernel(
    PartialW,
    PartialB,
    dW,
    dB,
    row_group_count,
    N,
    BLOCK_GROUP_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    cols = ext.program_id(0) * BLOCK_COL_SIZE + tl.arange(0, BLOCK_COL_SIZE)
    col_mask = cols < N
    acc_w = tl.zeros([BLOCK_COL_SIZE], dtype=tl.float32)
    acc_b = tl.zeros([BLOCK_COL_SIZE], dtype=tl.float32)

    for group_offset in range(0, row_group_count, BLOCK_GROUP_SIZE):
        groups = group_offset + tl.arange(0, BLOCK_GROUP_SIZE)[:, None]
        mask = (groups < row_group_count) & col_mask[None, :]
        offsets = groups * N + cols[None, :]
        if PartialW is not None:
            acc_w += tl.sum(tl.load(PartialW + offsets, mask=mask, other=0.0), axis=0)
        if PartialB is not None:
            acc_b += tl.sum(tl.load(PartialB + offsets, mask=mask, other=0.0), axis=0)

    if dW is not None:
        tl.store(dW + cols, acc_w, mask=col_mask)
    if dB is not None:
        tl.store(dB + cols, acc_b, mask=col_mask)


def _weight_bias_backward_reduction_config(input, M, N):
    heuristics = runtime.get_heuristic_config("layer_norm_weight_bias_backward")
    if heuristics is None:
        return None

    args = {
        "M": M,
        "N": N,
        "IS_LOW_PRECISION": input.dtype != torch.float32,
    }
    rows_per_program = heuristics["ROWS_PER_PROGRAM"](args)
    row_group_count = triton.cdiv(M, rows_per_program)
    # The temporary reduction buffer pays off only with enough row parallelism.
    if row_group_count < heuristics["MIN_ROW_GROUPS"](args):
        return None

    block_row_size_fn = heuristics.get("BLOCK_ROW_SIZE")
    block_group_size_fn = heuristics.get("BLOCK_GROUP_SIZE")
    block_row_size = block_row_size_fn(args) if block_row_size_fn is not None else 32
    block_group_size = (
        block_group_size_fn(args) if block_group_size_fn is not None else 32
    )
    block_col_size = heuristics["BLOCK_COL_SIZE"](args)
    return (
        row_group_count,
        rows_per_program,
        block_row_size,
        block_col_size,
        block_group_size,
    )


def _launch_weight_bias_backward_reduction(
    grad_out,
    input,
    mean,
    rstd,
    weight_grad,
    bias_grad,
    M,
    N,
):
    config = _weight_bias_backward_reduction_config(input, M, N)
    if config is None:
        return False

    (
        row_group_count,
        rows_per_program,
        block_row_size,
        block_col_size,
        block_group_size,
    ) = config
    # Store inter-stage partials in FP32 so all dtypes use accumulation precision.
    partial_weight = (
        torch.empty((row_group_count, N), dtype=torch.float32, device=input.device)
        if weight_grad is not None
        else None
    )
    partial_bias = (
        torch.empty((row_group_count, N), dtype=torch.float32, device=input.device)
        if bias_grad is not None
        else None
    )

    with torch_device_fn.device(input.device):
        partial_grid = (row_group_count, triton.cdiv(N, block_col_size), 1)
        weight_bias_backward_partial_kernel[partial_grid](
            grad_out,
            input,
            mean,
            rstd,
            partial_weight,
            partial_bias,
            M,
            N,
            ROWS_PER_PROGRAM=rows_per_program,
            BLOCK_ROW_SIZE=block_row_size,
            BLOCK_COL_SIZE=block_col_size,
        )
        final_grid = (triton.cdiv(N, block_col_size), 1, 1)
        weight_bias_backward_reduce_kernel[final_grid](
            partial_weight,
            partial_bias,
            weight_grad,
            bias_grad,
            row_group_count,
            N,
            BLOCK_GROUP_SIZE=block_group_size,
            BLOCK_COL_SIZE=block_col_size,
        )
    return True


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

    grad_out = grad_out.contiguous()
    input = input.contiguous()
    mean = mean.contiguous()
    rstd = rstd.contiguous()
    weight = None if weight is None else weight.contiguous()
    bias = None if bias is None else bias.contiguous()

    # Match LayerNorm semantics when normalized_shape covers only an input suffix.
    N = math.prod(normalized_shape)
    M = input.numel() // N

    # MThreads launches resident dX separately from the grouped affine reduction.
    resident_config = _layer_norm_backward_resident_dx_config(input, output_mask, M, N)
    if resident_config is None:
        in_grad = None
    else:
        in_grad, _, _ = _launch_layer_norm_backward_resident(
            grad_out,
            input,
            mean,
            rstd,
            weight,
            bias,
            M,
            N,
            *resident_config,
            compute_dw=False,
            compute_db=False,
            use_fp32_scratch=False,
        )

    if output_mask[0] and in_grad is None:
        in_grad = torch.empty_like(input)
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_ROW_SIZE"]), 1, 1)
        with torch_device_fn.device(input.device):
            layer_norm_backward_kernel[grid](
                grad_out, input, weight, mean, rstd, in_grad, M, N
            )

    if output_mask[1] is False and output_mask[2] is False:
        return in_grad, None, None

    weight_grad = torch.empty_like(weight) if output_mask[1] else None
    bias_grad = torch.empty_like(bias) if output_mask[2] else None
    used_grouped_reduction = _launch_weight_bias_backward_reduction(
        grad_out,
        input,
        mean,
        rstd,
        weight_grad,
        bias_grad,
        M,
        N,
    )
    if not used_grouped_reduction:
        # Small reductions keep the common one-stage affine kernel.
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_COL_SIZE"]), 1, 1)
        with torch_device_fn.device(input.device):
            weight_bias_backward_kernel[grid](
                grad_out, input, mean, rstd, weight_grad, bias_grad, M, N
            )
    return in_grad, weight_grad, bias_grad
