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
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

MAX_BLOCK_SIZE = 4096
MAX_GRID_SIZE = 65535
# gcu400 has 24 SIPs; 48 programs (2 waves) measured best for the row-wise
# kernel on L600.  The old grid=min(M, 65535) launched up to 65535 programs
# for M=1M rows and paid scheduling overhead without extra parallelism.
OPTIMAL_GRID_SIZE = 48


@libentry()
@triton.jit(do_not_specialize=["eps"])
def skip_layer_norm_kernel_generic(
    Y,  # pointer to the output
    X,  # pointer to the input
    R,  # pointer to the residual
    W,  # pointer to the weights
    B,  # pointer to the biases
    y_stride_r,
    y_stride_c,
    x_stride_r,  # how much to increase the pointer when moving by 1 row
    x_stride_c,  # how much to increase the pointer when moving by 1 col
    r_stride_r,  # how much to increase the pointer when moving by 1 row
    r_stride_c,  # how much to increase the pointer when moving by 1 col
    M,  # total number of rows
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    num_programs = ext.num_programs(0)

    for row_idx in range(pid, M, num_programs):
        row_Y = Y + row_idx * y_stride_r
        row_X = X + row_idx * x_stride_r
        row_R = R + row_idx * r_stride_r

        # Pass 1: accumulate sum for mean
        sum_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(row_X + cols * x_stride_c, mask, other=0.0).to(tl.float32)
            r = tl.load(row_R + cols * r_stride_c, mask, other=0.0).to(tl.float32)
            sum_acc += tl.where(mask, x + r, 0.0)

        mean = tl.sum(sum_acc, axis=0) / N

        # Pass 2: compute variance
        var_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(row_X + cols * x_stride_c, mask, other=0.0).to(tl.float32)
            r = tl.load(row_R + cols * r_stride_c, mask, other=0.0).to(tl.float32)
            val = tl.where(mask, x + r - mean, 0.0)
            var_acc += val * val

        var = tl.sum(var_acc, axis=0) / N
        rstd = 1 / tl.sqrt(var + eps)

        # Pass 3: normalize and apply weight/bias, write output
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(row_X + cols * x_stride_c, mask, other=0.0).to(tl.float32)
            r = tl.load(row_R + cols * r_stride_c, mask, other=0.0).to(tl.float32)
            w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
            b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
            x_hat = (x + r - mean) * rstd
            y = w * x_hat + b
            y = y.to(row_Y.dtype.element_ty)
            tl.store(row_Y + cols * y_stride_c, y, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def skip_layer_norm_kernel_fused(
    Y,
    X,
    R,
    W,
    B,
    y_stride_r,
    y_stride_c,
    x_stride_r,
    x_stride_c,
    r_stride_r,
    r_stride_c,
    M,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Single-pass row LayerNorm for N <= BLOCK_SIZE.

    The generic kernel walks each row three times and reloads X / R on every
    pass (9 memory ops per row).  Keeping x+residual in registers lets one
    load of each serve mean, variance and normalization (5 ops per row).
    Measured on gcu400 L600: 1024x1024 108us -> 44us, 4096x4096 493us ->
    193us.  Relative error vs torch.layer_norm is <= 4e-4 for fp16.
    """
    pid = ext.program_id(0)
    num_programs = ext.num_programs(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    for row_idx in range(pid, M, num_programs):
        row_X = X + row_idx * x_stride_r
        row_R = R + row_idx * r_stride_r
        row_Y = Y + row_idx * y_stride_r
        x = tl.load(row_X + cols * x_stride_c, mask, other=0.0).to(tl.float32)
        r = tl.load(row_R + cols * r_stride_c, mask, other=0.0).to(tl.float32)
        val = tl.where(mask, x + r, 0.0)
        mean = tl.sum(val, axis=0) / N
        diff = tl.where(mask, val - mean, 0.0)
        var = tl.sum(diff * diff, axis=0) / N
        rstd = tl.rsqrt(var + eps)
        w = tl.load(W + cols, mask, other=0.0).to(tl.float32)
        b = tl.load(B + cols, mask, other=0.0).to(tl.float32)
        y = w * (diff * rstd) + b
        tl.store(row_Y + cols * y_stride_c, y, mask)


class SkipLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual, normalized_shape, weight, bias, eps=1e-5):
        logger.debug("GEMS_ENFLAME SKIP_LAYERNORM_FORWARD")
        dim = x.ndim - len(normalized_shape)
        M = math.prod(x.shape[:dim])
        N = math.prod(normalized_shape)

        BLOCK_SIZE = min(triton.next_power_of_2(N), MAX_BLOCK_SIZE)
        grid_size = min(M, OPTIMAL_GRID_SIZE)
        x = x.contiguous()
        residual = residual.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()
        y = torch.empty_like(x)

        args = (y, x, residual, weight, bias, N, 1, N, 1, N, 1, M, N, eps, BLOCK_SIZE)
        # A single warp is fastest for narrow rows (measured 1024 columns
        # 44.2us -> 34.9us, 512 columns with M=32768 1480us -> 1015us);
        # wide rows need 4 warps to cover the 4096-element reduction.
        num_warps = 1 if N <= 1024 else 4
        with torch_device_fn.device(x.device):
            if N <= BLOCK_SIZE:
                skip_layer_norm_kernel_fused[grid_size,](*args, num_warps=num_warps)
            else:
                skip_layer_norm_kernel_generic[grid_size,](*args, num_warps=4)
        return y


def skip_layer_norm(x, residual, normalized_shape, weight, bias, eps=1e-5):
    return SkipLayerNorm.apply(x, residual, normalized_shape, weight, bias, eps)
