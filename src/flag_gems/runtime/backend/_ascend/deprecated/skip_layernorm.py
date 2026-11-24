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
from flag_gems.runtime.backend._ascend.tensorview import (
    HAS_TENSORVIEW,
    MIN_TILE_ROWS,
    tensorview_call,
    tile_rows_for_ub,
)
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

if HAS_TENSORVIEW:
    import triton.experimental.tle as tle_exp
    from triton.experimental.tle import tile_load, tile_store


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
    x_stride_r,  # pointer increment for moving 1 row
    x_stride_c,  # pointer increment for moving 1 column
    r_stride_r,  # pointer increment for moving 1 row in residual
    r_stride_c,  # pointer increment for moving 1 column in residual
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    loops = tl.cdiv(N, BLOCK_SIZE)

    # Initialize accumulators for sum of x and sum of squared x (for mean/variance)
    sum_x = tl.zeros((), dtype=tl.float32)  # Explicitly specify as float32
    sum_sq = tl.zeros((), dtype=tl.float32)  # For variance calculation

    # Pointer offsets for current row (based on program ID)
    X += pid * x_stride_r
    R += pid * r_stride_r
    Y += pid * y_stride_r

    # This partitioning is special: need to load entire row data for computation, so N is required
    # First pass: compute sum(x) and sum(x²) in one traversal (optimized from 2 passes)
    for process in range(loops):
        cols = process * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        # Load input and residual, compute x = X + residual
        x = tl.load(X + cols * x_stride_c, mask, other=0.0).to(tl.float32)
        r = tl.load(R + cols * r_stride_c, mask, other=0.0).to(tl.float32)
        x += r
        # Accumulate sum of x and sum of x squared
        sum_x += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    # Compute mean and variance from accumulated sums
    mean = sum_x / N
    var = (sum_sq / N) - (mean * mean)  # Equivalent to E[x²] - (E[x])²
    rstd = 1 / tl.sqrt(var + eps)  # Reciprocal of standard deviation

    # Second pass: compute final output (y = w * (x-mean)/std + b)
    for process in range(loops):
        cols = process * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        # Load weights and biases
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
        # Re-load x (X + residual) for normalization
        x = tl.load(X + cols * x_stride_c, mask, other=0.0).to(tl.float32)
        r = tl.load(R + cols * r_stride_c, mask, other=0.0).to(tl.float32)
        x += r
        # Apply layer norm and linear transformation
        x_hat = (x - mean) * rstd
        y = w * x_hat + b
        # Cast to output dtype and store
        y = y.to(Y.dtype.element_ty)
        tl.store(Y + cols * y_stride_c, y, mask=mask)


class SkipLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual, normalized_shape, weight, bias, eps=1e-5):
        logger.debug("GEMS_ASCEND SKIP_LAYERNORM_FORWARD")
        dim = x.ndim - len(normalized_shape)
        M = min(math.prod(x.shape[:dim]), 65535)
        N = math.prod(normalized_shape)
        BLOCK_SIZE = min(triton.next_power_of_2(N), 4096)
        x = x.contiguous()
        residual = residual.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()
        y = torch.empty_like(x)
        with torch_device_fn.device(x.device):
            skip_layer_norm_kernel[M,](
                y, x, residual, weight, bias, N, 1, N, 1, N, 1, N, eps, BLOCK_SIZE
            )
        return y


def _skip_layer_norm_generic(x, residual, normalized_shape, weight, bias, eps=1e-5):
    return SkipLayerNorm.apply(x, residual, normalized_shape, weight, bias, eps)


# ---------------------------------------------------------------------------
# TensorView fast path
#
# The generic kernel runs one program per row and walks that row in BLOCK_SIZE
# chunks, re-reading x and the residual for the second pass. Here one tile
# covers BLOCK_M whole rows, so the sum and the sum of squares are plain
# reductions along the tile width and the data is read once. The weight and
# bias rows come in through their own 1-D views, fetched once per program.
# ---------------------------------------------------------------------------
TV_BLOCK_M = 32

if HAS_TENSORVIEW:

    @triton.jit(do_not_specialize=["eps"])
    def skip_layer_norm_tv_kernel(
        y,
        x,
        residual,
        weight,
        bias,
        num_rows,
        eps,
        WIDTH: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_WIDTH: tl.constexpr,
    ):
        row_block = tl.program_id(0)

        x_view = tle_exp.make_partition_view(
            x,
            shape=[num_rows, WIDTH],
            strides=[WIDTH, 1],
            tile=[BLOCK_M, BLOCK_WIDTH],
        )
        residual_view = tle_exp.make_partition_view(
            residual,
            shape=[num_rows, WIDTH],
            strides=[WIDTH, 1],
            tile=[BLOCK_M, BLOCK_WIDTH],
        )
        weight_view = tle_exp.make_partition_view(
            weight, shape=[WIDTH], strides=[1], tile=[BLOCK_WIDTH]
        )
        bias_view = tle_exp.make_partition_view(
            bias, shape=[WIDTH], strides=[1], tile=[BLOCK_WIDTH]
        )
        summed = tile_load(
            x_view, index=(row_block, 0), dst_space=tle_exp.language.dsa.ascend.UB
        ).to(tl.float32)
        summed += tile_load(
            residual_view,
            index=(row_block, 0),
            dst_space=tle_exp.language.dsa.ascend.UB,
        ).to(tl.float32)
        scale = tile_load(
            weight_view, index=(0,), dst_space=tle_exp.language.dsa.ascend.UB
        ).to(tl.float32)
        shift = tile_load(
            bias_view, index=(0,), dst_space=tle_exp.language.dsa.ascend.UB
        ).to(tl.float32)

        mean = tl.sum(summed, axis=1) / WIDTH
        var = tl.sum(summed * summed, axis=1) / WIDTH - mean * mean
        rstd = 1.0 / tl.sqrt(var + eps)
        normalized = (summed - mean[:, None]) * rstd[:, None] * scale[None, :] + shift[
            None, :
        ]

        # Rows past the end of the tensor fall outside the view and are dropped
        # on store, which covers the tail block without a mask.
        y_view = tle_exp.make_partition_view(
            y,
            shape=[num_rows, WIDTH],
            strides=[WIDTH, 1],
            tile=[BLOCK_M, BLOCK_WIDTH],
        )
        tile_store(
            y_view,
            value=normalized.to(y.dtype.element_ty),
            index=(row_block, 0),
            src_space=tle_exp.language.dsa.ascend.UB,
        )

    def _tv_rows_that_fit(width):
        """Rows per tile UB can actually hold: x, the residual and their sum
        are all live as fp32. A width whose tile cannot reach MIN_TILE_ROWS
        reports 1, which the eligibility check rejects."""
        return tile_rows_for_ub(triton.next_power_of_2(width) * 4, 3, TV_BLOCK_M)

    def _tv_block_m(width):
        """Tile height to launch with, never below the liftable minimum."""
        return max(_tv_rows_that_fit(width), MIN_TILE_ROWS)

    def _skip_layer_norm_tensorview(
        x, residual, normalized_shape, weight, bias, eps=1e-5, block_m=None
    ):
        logger.debug("GEMS_ASCEND SKIP_LAYERNORM_FORWARD(TENSORVIEW)")
        width = math.prod(normalized_shape)
        block_m = _tv_block_m(width) if block_m is None else block_m
        x = x.contiguous()
        residual = residual.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()
        rows = x.numel() // width
        y = torch.empty_like(x)
        grid = (triton.cdiv(rows, block_m),)
        with torch_device_fn.device(x.device):
            skip_layer_norm_tv_kernel[grid](
                y,
                x,
                residual,
                weight,
                bias,
                rows,
                eps,
                WIDTH=width,
                BLOCK_M=block_m,
                BLOCK_WIDTH=triton.next_power_of_2(width),
            )
        return y

    def _tv_eligible(x, residual, normalized_shape, weight, bias):
        # The view is 2-D [rows, width], so only the last axis may be
        # normalized and the weight and bias must cover exactly that width.
        if len(normalized_shape) != 1:
            return False
        width = normalized_shape[0]
        if x.numel() == 0 or x.shape != residual.shape or x.shape[-1] != width:
            return False
        if weight is None or bias is None:
            return False
        if weight.numel() != width or bias.numel() != width:
            return False
        if x.dtype != residual.dtype:
            return False
        # A width that is not a power of two makes the dense tile wider than
        # the view. The padding path handles it but costs more than it saves.
        if triton.next_power_of_2(width) != width:
            return False
        # The win comes from batching rows into one tile, and the tile has to
        # be at least MIN_TILE_ROWS tall to lower at all. A row so wide that
        # UB cannot hold that many leaves nothing to gain over the per-row
        # generic kernel, and launching the clamped height would overflow UB.
        return _tv_rows_that_fit(width) >= MIN_TILE_ROWS

    def skip_layer_norm(x, residual, normalized_shape, weight, bias, eps=1e-5):
        if not _tv_eligible(x, residual, normalized_shape, weight, bias):
            return _skip_layer_norm_generic(
                x, residual, normalized_shape, weight, bias, eps
            )
        return tensorview_call(
            "SKIP_LAYERNORM",
            _skip_layer_norm_tensorview,
            _skip_layer_norm_generic,
            x,
            residual,
            normalized_shape,
            weight,
            bias,
            eps,
        )

else:
    skip_layer_norm = _skip_layer_norm_generic
