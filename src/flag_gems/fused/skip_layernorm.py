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


@libentry()
@triton.jit(do_not_specialize=["eps"])
def skip_layer_norm_kernel(
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
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    Y += pid * y_stride_r
    X += pid * x_stride_r
    R += pid * r_stride_r

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + cols * x_stride_c, mask, other=0.0).to(tl.float32)
    r = tl.load(R + cols * r_stride_c, mask, other=0.0).to(tl.float32)

    x += r

    mean = tl.sum(x, axis=0) / N

    # Compute variance
    _var = tl.where(mask, x - mean, 0.0)
    _var = _var * _var
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    w = tl.load(W + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0).to(tl.float32)

    x_hat = (x - mean) * rstd
    y = w * x_hat + b
    y = y.to(Y.dtype.element_ty)
    tl.store(Y + cols * y_stride_c, y, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def skip_layer_norm_loop_kernel(
    Y,  # pointer to the output
    X,  # pointer to the input
    R,  # pointer to the residual
    W,  # pointer to the weights
    B,  # pointer to the biases
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    row_start = pid * N

    # Pass 1: add residual and compute mean (read from X and R directly)
    mean_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    num_steps = tl.cdiv(N, BLOCK_SIZE)

    for step in range(0, num_steps):
        start_n = step * BLOCK_SIZE
        cols = start_n + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + row_start + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(R + row_start + cols, mask=mask, other=0.0).to(tl.float32)
        x = x + r
        mean_acc += x

    mean = tl.sum(mean_acc, axis=0) / N

    # Pass 2: compute variance (re-read from X and R to avoid fp16 truncation)
    var_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for step in range(0, num_steps):
        start_n = step * BLOCK_SIZE
        cols = start_n + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + row_start + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(R + row_start + cols, mask=mask, other=0.0).to(tl.float32)
        x = x + r
        diff = tl.where(mask, x - mean, 0.0)
        var_acc += diff * diff

    var = tl.sum(var_acc, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    # Pass 3: normalize (re-read from X and R to avoid fp16 truncation)
    for step in range(0, num_steps):
        start_n = step * BLOCK_SIZE
        cols = start_n + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + row_start + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(R + row_start + cols, mask=mask, other=0.0).to(tl.float32)
        x = x + r
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = w * x_hat + b
        y = y.to(Y.dtype.element_ty)
        tl.store(Y + row_start + cols, y, mask=mask)


class SkipLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual, normalized_shape, weight, bias, eps=1e-5):
        logger.debug("GEMS SKIP LAYERNORM FORWARD")
        dim = x.ndim - len(normalized_shape)
        M = math.prod(x.shape[:dim])
        N = math.prod(normalized_shape)

        x = x.contiguous()
        residual = residual.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()
        y = torch.empty_like(x)

        with torch_device_fn.device(x.device):
            if N <= 4096:
                BLOCK_SIZE = triton.next_power_of_2(N)
                skip_layer_norm_kernel[M,](
                    y, x, residual, weight, bias, N, 1, N, 1, N, 1, N, eps, BLOCK_SIZE
                )
            else:
                BLOCK_SIZE = 1024
                skip_layer_norm_loop_kernel[M,](
                    y, x, residual, weight, bias, N, eps, BLOCK_SIZE
                )
        return y


def skip_layer_norm(x, residual, normalized_shape, weight, bias, eps=1e-5):
    return SkipLayerNorm.apply(x, residual, normalized_shape, weight, bias, eps)
