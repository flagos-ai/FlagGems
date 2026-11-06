# Copyright 2026, The FlagOS Contributors.
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
#
import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.cudnn_batch_norm import (
    _cudnn_batch_norm_impl,
    _cudnn_batch_norm_out_impl,
)
from flag_gems.ops.cudnn_batch_norm import _forward as _generic_forward
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from .cudnn_batch_norm_backward import cudnn_batch_norm_backward

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _train_fused(
    X,
    W,
    Bias,
    RM,
    RV,
    Y,
    Mean,
    Inv,
    M: tl.constexpr,
    C: tl.constexpr,
    S: tl.constexpr,
    UPDATE: tl.constexpr,
    FACTOR: tl.constexpr,
    EPS: tl.constexpr,
    B: tl.constexpr,
):
    # One CTA per channel computes stable statistics and writes the output.
    c = tl.program_id(0)
    lane = tl.arange(0, B)
    anchor = tl.load(X + c * S).to(tl.float32)
    anchor = tl.where(tl.abs(anchor) < float("inf"), anchor, 0)
    sums = tl.full((B,), 0, tl.float32)
    for block in range(tl.cdiv(M, B)):
        r = block * B + lane
        x = tl.load(X + r // S * C * S + c * S + r % S, r < M, 0).to(tl.float32)
        sums += tl.where(r < M, x - anchor, 0)
    shift = tl.sum(sums, 0) / M
    squares = tl.full((B,), 0, tl.float32)
    for block in range(tl.cdiv(M, B)):
        r = block * B + lane
        x = tl.load(X + r // S * C * S + c * S + r % S, r < M, 0).to(tl.float32)
        delta = tl.where(r < M, (x - anchor) - shift, 0)
        squares += delta * delta
    var = tl.sum(squares, 0) / M
    inv = tl.rsqrt(var + EPS)
    mean = anchor + shift
    tl.store(Mean + c, mean)
    tl.store(Inv + c, inv)
    if UPDATE:
        old_mean = tl.load(RM + c)
        old_var = tl.load(RV + c)
        tl.store(RM + c, (1 - FACTOR) * old_mean + FACTOR * mean)
        tl.store(RV + c, (1 - FACTOR) * old_var + FACTOR * var * M / (M - 1))
    weight = tl.load(W + c)
    bias = tl.load(Bias + c)
    for block in range(tl.cdiv(M, B)):
        r = block * B + lane
        offsets = r // S * C * S + c * S + r % S
        x = tl.load(X + offsets, r < M, 0).to(tl.float32)
        tl.store(Y + offsets, ((x - anchor) - shift) * inv * weight + bias, r < M)


# Direct JIT dispatch avoids host gaps between these two short NVIDIA launches.
@triton.jit
def _train_partial(
    X,
    Partial,
    M: tl.constexpr,
    C: tl.constexpr,
    S: tl.constexpr,
    P: tl.constexpr,
    B: tl.constexpr,
):
    # Center each tile before accumulating its second moment.
    c = tl.program_id(0)
    p = tl.program_id(1)
    r = p * B + tl.arange(0, B)
    anchor = tl.load(X + c * S).to(tl.float32)
    anchor = tl.where(tl.abs(anchor) < float("inf"), anchor, 0)
    x = tl.load(X + r // S * C * S + c * S + r % S, r < M, 0).to(tl.float32)
    x = tl.where(r < M, x - anchor, 0)
    count = tl.minimum(B, M - p * B)
    mean = tl.sum(x, 0) / count
    delta = tl.where(r < M, x - mean, 0)
    tl.store(Partial + c * P * 2 + p, mean)
    tl.store(Partial + c * P * 2 + P + p, tl.sum(delta * delta, 0))


@triton.jit
def _train_finish(
    X,
    W,
    Bias,
    RM,
    RV,
    Y,
    Mean,
    Inv,
    Partial,
    M: tl.constexpr,
    C: tl.constexpr,
    S: tl.constexpr,
    UPDATE: tl.constexpr,
    FACTOR: tl.constexpr,
    EPS: tl.constexpr,
    P: tl.constexpr,
    PB: tl.constexpr,
    STAT_B: tl.constexpr,
    B: tl.constexpr,
):
    c = tl.program_id(0)
    block = tl.program_id(1)
    p = tl.arange(0, PB)
    means = tl.load(Partial + c * P * 2 + p, p < P, 0)
    m2 = tl.load(Partial + c * P * 2 + P + p, p < P, 0)
    # The final tile may be partial, so merge with its actual sample count.
    counts = tl.where(p < P, tl.minimum(STAT_B, M - p * STAT_B), 0)
    shift = tl.sum(means * counts, 0) / M
    diff = means - shift
    var = tl.sum(tl.where(p < P, m2 + counts * diff * diff, 0), 0) / M
    inv = tl.rsqrt(var + EPS)
    anchor = tl.load(X + c * S).to(tl.float32)
    anchor = tl.where(tl.abs(anchor) < float("inf"), anchor, 0)
    if block == 0:
        mean = anchor + shift
        tl.store(Mean + c, mean)
        tl.store(Inv + c, inv)
        if UPDATE:
            tl.store(RM + c, (1 - FACTOR) * tl.load(RM + c) + FACTOR * mean)
            tl.store(
                RV + c, (1 - FACTOR) * tl.load(RV + c) + FACTOR * var * M / (M - 1)
            )
    r = block * B + tl.arange(0, B)
    offsets = r // S * C * S + c * S + r % S
    x = tl.load(X + offsets, r < M, 0).to(tl.float32)
    y = ((x - anchor) - shift) * inv * tl.load(W + c) + tl.load(Bias + c)
    tl.store(Y + offsets, y, r < M)


def _forward(
    input,
    weight,
    bias,
    running_mean,
    running_var,
    training,
    exponential_average_factor,
    epsilon,
    output,
    mean,
    inv,
):
    n, c = input.shape[:2]
    s = math.prod(input.shape[2:])
    m = n * s
    double = input.dtype == torch.float64
    if training and 2048 < m <= 131072 and not double and input.is_contiguous():
        with torch_device_fn.device(input.device):
            args = (
                input,
                weight,
                bias,
                running_mean,
                running_var,
                output,
                mean,
                inv,
            )
            if m <= 32768:
                # A single launch avoids intermediate allocations for short reductions.
                _train_fused[(c,)](
                    *args,
                    m,
                    c,
                    s,
                    running_mean is not None,
                    exponential_average_factor,
                    epsilon,
                    min(triton.next_power_of_2(m), 8192),
                    num_warps=8,
                )
            else:
                # Merge partial moments inside each output block to save a launch.
                block = 2048
                parts = triton.cdiv(m, block)
                partial = torch.empty(
                    (c, 2, parts), dtype=torch.float32, device=input.device
                )
                _train_partial[(c, parts)](input, partial, m, c, s, parts, block)
                _train_finish[(c, parts)](
                    *args,
                    partial,
                    m,
                    c,
                    s,
                    running_mean is not None,
                    exponential_average_factor,
                    epsilon,
                    parts,
                    triton.next_power_of_2(parts),
                    block,
                    block,
                )
            return
    _generic_forward(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        exponential_average_factor,
        epsilon,
        output,
        mean,
        inv,
    )


def cudnn_batch_norm(
    input,
    weight,
    bias,
    running_mean,
    running_var,
    training,
    exponential_average_factor,
    epsilon,
):
    logger.debug("GEMS_NVIDIA CUDNN_BATCH_NORM")
    return _cudnn_batch_norm_impl(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        exponential_average_factor,
        epsilon,
        _forward,
        cudnn_batch_norm_backward,
    )


def cudnn_batch_norm_out(
    input,
    weight,
    bias,
    running_mean,
    running_var,
    training,
    exponential_average_factor,
    epsilon,
    *,
    out0,
    out1,
    out2,
    out3,
):
    logger.debug("GEMS_NVIDIA CUDNN_BATCH_NORM_OUT")
    return _cudnn_batch_norm_out_impl(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        exponential_average_factor,
        epsilon,
        out0=out0,
        out1=out1,
        out2=out2,
        out3=out3,
        forward_fn=_forward,
    )
