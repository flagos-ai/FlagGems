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

from flag_gems.ops.layernorm import layer_norm as _generic_layer_norm
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

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
                    _ROW_LAYOUT,
                    _WEIGHT_LAYOUT,
                    num_warps=nw,
                )
    return y, mean, rstd


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    return _layer_norm_impl(input, normalized_shape, weight, bias, eps, False)


def native_layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    logger.debug("GEMS_MTHREADS NATIVE_LAYER_NORM")
    return _layer_norm_impl(input, normalized_shape, weight, bias, eps, True)
