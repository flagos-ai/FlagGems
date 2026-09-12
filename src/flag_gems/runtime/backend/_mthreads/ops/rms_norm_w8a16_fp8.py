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

logger = logging.getLogger(__name__)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_w8a16_fp8_kernel(
    X,
    W,
    S,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    W_STRIDE: tl.constexpr,
    S_STRIDE: tl.constexpr,
    eps,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    NUM_WARPS: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    offsets = rows[:, None] * N + cols[None, :]
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(x * x, axis=1) / N + eps)
    w = tl.load(W + cols * W_STRIDE, mask=cols < N, other=0.0).to(tl.float32)
    if GROUP_SIZE <= BLOCK_N and GROUP_SIZE & (GROUP_SIZE - 1) == 0:
        groups = tl.arange(0, BLOCK_N // GROUP_SIZE)
        scale = tl.load(
            S + groups * S_STRIDE, mask=groups * GROUP_SIZE < N, other=0.0
        ).to(tl.float32)
        w = tl.reshape(
            tl.reshape(w, (BLOCK_N // GROUP_SIZE, GROUP_SIZE)) * scale[:, None],
            (BLOCK_N,),
        )
    else:
        scale = tl.load(
            S + (cols // GROUP_SIZE) * S_STRIDE, mask=cols < N, other=0.0
        ).to(tl.float32)
        w *= scale
    # Share decoded weights across rows without materializing a device buffer.
    y = (x * inv_rms[:, None]) * w[None, :]
    tl.store(Y + offsets, y, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_w8a16_fp8_tiled_kernel(
    X,
    W,
    S,
    Y,
    N: tl.constexpr,
    W_STRIDE: tl.constexpr,
    S_STRIDE: tl.constexpr,
    eps,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TILE_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + row * N + cols, cols < N, 0.0).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(x * x, 0) / N + eps)
    # Few rows cannot occupy the device. Recompute their small reduction in
    # each output tile, spreading weight decoding and stores across blocks.
    out_cols = tl.program_id(1) * TILE_N + tl.arange(0, TILE_N)
    mask = out_cols < N
    x_out = tl.load(X + row * N + out_cols, mask, 0.0).to(tl.float32)
    w = tl.load(W + out_cols * W_STRIDE, mask, 0.0).to(tl.float32)
    scale = tl.load(S + (out_cols // GROUP_SIZE) * S_STRIDE, mask, 0.0).to(tl.float32)
    tl.store(Y + row * N + out_cols, (x_out * inv_rms) * (w * scale), mask)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_w8a16_fp8_loop_kernel(
    X,
    W,
    S,
    Y,
    N: tl.constexpr,
    W_STRIDE: tl.constexpr,
    S_STRIDE: tl.constexpr,
    eps,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), tl.float32)
    for start in range(tl.cdiv(N, BLOCK_N)):
        offsets = start * BLOCK_N + cols
        x = tl.load(X + row * N + offsets, offsets < N, 0.0).to(tl.float32)
        acc += x * x
    inv_rms = tl.rsqrt(tl.sum(acc, 0) / N + eps)
    for step in range(tl.cdiv(N, BLOCK_N)):
        offsets = (tl.cdiv(N, BLOCK_N) - 1 - step) * BLOCK_N + cols
        mask = offsets < N
        x = tl.load(X + row * N + offsets, mask, 0.0).to(tl.float32)
        w = tl.load(W + offsets * W_STRIDE, mask, 0.0).to(tl.float32)
        scale = tl.load(S + (offsets // GROUP_SIZE) * S_STRIDE, mask, 0.0).to(
            tl.float32
        )
        y = (x * inv_rms) * (w * scale)
        tl.store(Y + row * N + offsets, y, mask)


def rms_norm_w8a16_fp8(
    x, normalized_shape, weight_fp8, weight_scale, eps=1e-5, group_size=128
):
    logger.debug("GEMS_MTHREADS RMS_NORM W8A16 FP8 FORWARD")
    normalized_shape = tuple(normalized_shape)
    if (
        not normalized_shape
        or tuple(x.shape[-len(normalized_shape) :]) != normalized_shape
    ):
        raise ValueError("normalized_shape must match the trailing input dimensions")
    n = math.prod(normalized_shape)
    if n <= 0 or group_size <= 0:
        raise ValueError("normalized dimensions and group_size must be positive")
    if weight_fp8.dtype != torch.float8_e4m3fn:
        raise TypeError("weight_fp8 must have dtype torch.float8_e4m3fn")
    if weight_fp8.numel() != n or weight_scale.numel() != triton.cdiv(n, group_size):
        raise ValueError(
            "weight and scale sizes must match normalized_shape and group_size"
        )
    x = x.contiguous()
    weight_fp8 = weight_fp8.reshape(-1)
    weight_scale = weight_scale.reshape(-1)
    m = x.numel() // n
    y = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    if m == 0:
        return y
    with torch_device_fn.device(x.device):
        if m <= 4 and 16384 <= n <= 65536:
            rms_norm_w8a16_fp8_tiled_kernel[(m, triton.cdiv(n, 1024))](
                x,
                weight_fp8,
                weight_scale,
                y,
                n,
                weight_fp8.stride(0),
                weight_scale.stride(0),
                eps,
                group_size,
                triton.next_power_of_2(n),
                1024,
                num_warps=16,
            )
        elif n <= 65536:
            block_m = 1
            if n <= 1024:
                num_warps = 4
            elif n <= 4096:
                if m <= 16:
                    num_warps = 32
                elif m <= 64:
                    num_warps = 16
                elif m < 512:
                    num_warps = 8
                else:
                    block_m, num_warps = 2, 16
            else:
                num_warps = 32
            rms_norm_w8a16_fp8_kernel[(triton.cdiv(m, block_m),)](
                x,
                weight_fp8,
                weight_scale,
                y,
                m,
                n,
                weight_fp8.stride(0),
                weight_scale.stride(0),
                eps,
                group_size,
                triton.next_power_of_2(n),
                block_m,
                num_warps,
                num_warps=num_warps,
            )
        else:
            rms_norm_w8a16_fp8_loop_kernel[(m,)](
                x,
                weight_fp8,
                weight_scale,
                y,
                n,
                weight_fp8.stride(0),
                weight_scale.stride(0),
                eps,
                group_size,
                4096,
                num_warps=4,
            )
    return y
