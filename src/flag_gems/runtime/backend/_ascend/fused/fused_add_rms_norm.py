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

import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._ascend.tensorview import (
    HAS_TENSORVIEW,
    tensorview_call,
    tile_rows_for_ub,
)
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

if HAS_TENSORVIEW:
    import triton.experimental.tle as tle_exp
    from triton.experimental.tle.language.dsa import tile_load, tile_store


@libentry()
@triton.jit(do_not_specialize=["eps"])
def fused_add_rms_norm_kernel(
    X,  # pointer to the input
    R,  # pointer to the residual
    W,  # pointer to the weight
    x_stride_r,  # how much to increase the pointer when moving by 1 row
    x_stride_c,  # how much to increase the pointer when moving by 1 col
    r_stride_r,  # how much to increase the pointer when moving by 1 row
    r_stride_c,  # how much to increase the pointer when moving by 1 col
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    X += pid * x_stride_r
    R += pid * r_stride_r

    _var_base = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        r = tl.load(R + cols, mask, other=0.0).to(tl.float32)
        x += r
        _var_base += x * x / N
    var = tl.sum(_var_base)

    rrms = 1 / tl.sqrt(var + eps)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        r = tl.load(R + cols, mask, other=0.0).to(tl.float32)
        x += r
        w = tl.load(W + cols, mask, other=0.0)
        y = (x * rrms).to(X.dtype.element_ty) * w
        # write back to residual and input
        tl.store(R + cols * r_stride_c, x, mask=mask)
        tl.store(X + cols * x_stride_c, y, mask=mask)


def _fused_add_rms_norm_generic(x, residual, normalized_shape, weight, eps=1e-5):
    """
    This function performs fused residual addition and RMS normalization **in-place**.
    Both `x` and `residual` tensors will be modified. Use with caution if these tensors
    are reused elsewhere or require gradients.
    """
    logger.debug("GEMS_ASCEND FUSED_ADD_RMS_NORM")
    dim = x.ndim - len(normalized_shape)
    M = min(math.prod(x.shape[:dim]), 65535)
    N = math.prod(normalized_shape)

    BLOCK_SIZE = min(triton.next_power_of_2(N), 8192)
    x = x.contiguous()
    residual = residual.contiguous()
    weight = weight.contiguous()

    with torch_device_fn.device(x.device):
        fused_add_rms_norm_kernel[M,](
            x, residual, weight, N, 1, N, 1, N, eps, BLOCK_SIZE
        )
    return x, residual


# ---------------------------------------------------------------------------
# TensorView fast path
#
# The generic kernel runs one program per row and loops over the width twice
# (once to accumulate the variance, once to write back), re-reading x and the
# residual on the second pass. Here a PartitionView brings a whole
# [BLOCK_M, WIDTH] tile in once as a block DMA, so the sum is computed a single
# time and reused, and the two in-place results go back out through
# GatherScatterViews.
#
# Note the generic path clamps its grid to 65535 programs, which silently
# leaves rows beyond that untouched. The fast path covers BLOCK_M rows per
# program, so the same row count needs far fewer programs.
# ---------------------------------------------------------------------------
TV_BLOCK_M = 32

if HAS_TENSORVIEW:

    @triton.jit(do_not_specialize=["eps"])
    def fused_add_rms_norm_tv_kernel(
        x,
        residual,
        weight,
        num_rows,
        eps,
        WIDTH: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_WIDTH: tl.constexpr,
    ):
        row_block = tl.program_id(0)
        rows = (row_block * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int32)

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
        tile = tile_load(
            x_view, index=(row_block, 0), dst_space=tle_exp.language.dsa.ascend.UB
        ).to(tl.float32)
        res = tile_load(
            residual_view,
            index=(row_block, 0),
            dst_space=tle_exp.language.dsa.ascend.UB,
        ).to(tl.float32)
        scale = tile_load(
            weight_view, index=(0,), dst_space=tle_exp.language.dsa.ascend.UB
        ).to(tl.float32)

        summed = tile + res
        rrms = 1.0 / tl.sqrt(tl.sum(summed * summed, axis=1) / WIDTH + eps)
        normalized = summed * rrms[:, None] * scale[None, :]

        x_out = tle_exp.make_gather_scatter_view(
            x,
            shape=[num_rows, WIDTH],
            strides=[WIDTH, 1],
            tile=[BLOCK_M, BLOCK_WIDTH],
            sparse_dim=[0],
        )
        residual_out = tle_exp.make_gather_scatter_view(
            residual,
            shape=[num_rows, WIDTH],
            strides=[WIDTH, 1],
            tile=[BLOCK_M, BLOCK_WIDTH],
            sparse_dim=[0],
        )
        tile_store(
            residual_out,
            value=summed.to(residual.dtype.element_ty),
            index=(rows, 0),
            src_space=tle_exp.language.dsa.ascend.UB,
        )
        tile_store(
            x_out,
            value=normalized.to(x.dtype.element_ty),
            index=(rows, 0),
            src_space=tle_exp.language.dsa.ascend.UB,
        )

    def _tv_block_m(width):
        """Rows per tile: x, residual and their sum are all live as fp32."""
        return tile_rows_for_ub(triton.next_power_of_2(width) * 4, 3, TV_BLOCK_M)

    def _fused_add_rms_norm_tensorview(
        x, residual, normalized_shape, weight, eps=1e-5, block_m=None
    ):
        logger.debug("GEMS_ASCEND FUSED_ADD_RMS_NORM(TENSORVIEW)")
        width = math.prod(normalized_shape)
        block_m = _tv_block_m(width) if block_m is None else block_m
        rows = x.numel() // width
        grid = (triton.cdiv(rows, block_m),)
        with torch_device_fn.device(x.device):
            fused_add_rms_norm_tv_kernel[grid](
                x,
                residual,
                weight,
                rows,
                eps,
                WIDTH=width,
                BLOCK_M=block_m,
                BLOCK_WIDTH=triton.next_power_of_2(width),
            )
        return x, residual

    def _tv_eligible(x, residual, normalized_shape, weight):
        width = math.prod(normalized_shape)
        return (
            len(normalized_shape) == 1
            and x.numel() > 0
            and x.shape[-1] == width
            and weight.numel() == width
            and residual.shape == x.shape
            and residual.dtype == x.dtype
            # A width that is not a power of two makes the dense tile wider
            # than the view; leave those to the looping generic kernel.
            and triton.next_power_of_2(width) == width
            # Only one row per tile means no batching advantage left, see the
            # same note in ops/rms_norm.py.
            and _tv_block_m(width) > 1
        )

    def fused_add_rms_norm(x, residual, normalized_shape, weight, eps=1e-5):
        """Fused residual addition and RMS normalization, in place on x and residual."""
        if not _tv_eligible(x, residual, normalized_shape, weight):
            return _fused_add_rms_norm_generic(
                x, residual, normalized_shape, weight, eps
            )
        # Both tensors are written in place, so they must be contiguous for the
        # view strides to describe them.
        if not (x.is_contiguous() and residual.is_contiguous()):
            return _fused_add_rms_norm_generic(
                x, residual, normalized_shape, weight, eps
            )
        return tensorview_call(
            "FUSED_ADD_RMS_NORM",
            _fused_add_rms_norm_tensorview,
            _fused_add_rms_norm_generic,
            x,
            residual,
            normalized_shape,
            weight.contiguous(),
            eps,
        )

else:
    fused_add_rms_norm = _fused_add_rms_norm_generic
