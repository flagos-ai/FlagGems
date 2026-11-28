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

"""gcu400 fake-quantize per channel, one flat masked gather.

The generated ``pointwise_dynamic`` kernel used to own this op.  On gcu400 it
issues its loads/stores either as DTE DMA through block pointers (rounded up to
the tile width) or as plain masked loads whose masked lanes still touch their
own address.  Both overrun the last tile: at ``(16,128,64,60)`` the allocation
ends on a page boundary, the overrun leaves the mapping and the MMU faults --
the process aborts at the next sync and the card goes to ``HEALTH FAIL``
(the same failure that took the card down through ``copy_`` on 2026-11-10).

This kernel keeps the exact same per-element arithmetic (fp32 reciprocal,
``nearbyint``, int zero point, clamp) but flattens to 1-D and clamps every
address: masked lanes address element 0, so no lane can leave the allocation.
The channel index is ``(flat // inner) % channels`` on the (contiguous) input.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import tl_extra_shim

logger = logging.getLogger(__name__)

_RUN_MIN_INNER = 1024
_RUN_MAX_BLOCK = 16384
_FLAT_BLOCK = 4096
_SMALL_N = 4096
_INNER1_BLOCK = 8192
_MAX_GRID = 65535


def _is_pow2(value):
    return value > 0 and (value & (value - 1)) == 0


def _pow2_at_most(value, cap):
    block = 1
    while block * 2 <= value and block * 2 <= cap:
        block *= 2
    return max(block, 1)


def _next_pow2(value):
    block = 1
    while block < value:
        block *= 2
    return block


@triton.jit
def _learnable_run_kernel(
    x_ptr,
    scale_ptr,
    zero_point_ptr,
    out_ptr,
    inner,
    n_channels,
    quant_min,
    quant_max,
    BLOCK: tl.constexpr,
):
    run_id = tl.program_id(0)
    channel = run_id % n_channels
    chunk_start = tl.program_id(1) * BLOCK
    lane = tl.arange(0, BLOCK)
    mask = lane < inner - chunk_start
    offsets = run_id * inner + chunk_start + lane
    # Masked-out lanes all point to element 0, avoiding reads past the end of the allocation
    safe = tl.where(mask, offsets, 0)

    scale = tl.load(scale_ptr + channel).to(tl.float32)
    zero_point = tl.load(zero_point_ptr + channel).to(tl.float32)
    zero_point = tl.minimum(
        tl.maximum(tl_extra_shim.rint(zero_point), quant_min), quant_max
    )
    lower = quant_min - zero_point
    upper = quant_max - zero_point

    x = tl.load(x_ptr + safe, mask=mask, other=0.0).to(tl.float32)
    quantized = tl_extra_shim.nearbyint(x * tl_extra_shim.rcp_rn(scale))
    out = tl.minimum(tl.maximum(quantized, lower), upper) * scale
    tl.store(out_ptr + safe, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _learnable_inner1_kernel(
    x_ptr,
    scale_ptr,
    zero_point_ptr,
    out_ptr,
    numel,
    n_channels,
    quant_min,
    quant_max,
    COL: tl.constexpr,
    TILE_ROWS: tl.constexpr,
):
    tile_start = tl.program_id(0) * (TILE_ROWS * n_channels)
    columns = tl.arange(0, COL)
    offsets = (
        tile_start + tl.arange(0, TILE_ROWS)[:, None] * n_channels + columns[None, :]
    )
    mask = (columns[None, :] < n_channels) & (offsets < numel)
    safe = tl.where(mask, offsets, 0)

    col_mask = columns < n_channels
    safe_col = tl.where(col_mask, columns, 0)
    scale = tl.load(scale_ptr + safe_col, mask=col_mask, other=1.0).to(tl.float32)[
        None, :
    ]
    zero_point = tl.load(zero_point_ptr + safe_col, mask=col_mask, other=0.0).to(
        tl.float32
    )
    zero_point = tl.minimum(
        tl.maximum(tl_extra_shim.rint(zero_point), quant_min), quant_max
    )
    lower = quant_min - zero_point[None, :]
    upper = quant_max - zero_point[None, :]

    x = tl.load(x_ptr + safe, mask=mask, other=0.0).to(tl.float32)
    quantized = tl_extra_shim.nearbyint(x * tl_extra_shim.rcp_rn(scale))
    out = tl.minimum(tl.maximum(quantized, lower), upper) * scale
    tl.store(out_ptr + safe, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _learnable_flat_kernel(
    x_ptr,
    scale_ptr,
    zero_point_ptr,
    out_ptr,
    numel,
    quant_min,
    quant_max,
    INNER: tl.constexpr,
    N_CHANNELS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    safe = tl.where(mask, offsets, 0)
    channel = (offsets // INNER) % N_CHANNELS

    scale = tl.load(scale_ptr + channel, mask=mask, other=1.0).to(tl.float32)
    zero_point = tl.load(zero_point_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    zero_point = tl.minimum(
        tl.maximum(tl_extra_shim.rint(zero_point), quant_min), quant_max
    )
    lower = quant_min - zero_point
    upper = quant_max - zero_point

    x = tl.load(x_ptr + safe, mask=mask, other=0.0).to(tl.float32)
    quantized = tl_extra_shim.nearbyint(x * tl_extra_shim.rcp_rn(scale))
    out = tl.minimum(tl.maximum(quantized, lower), upper) * scale
    tl.store(out_ptr + safe, out.to(out_ptr.dtype.element_ty), mask=mask)


def _launch(self, scale, zero_point, axis, quant_min, quant_max, out):
    if axis < 0:
        axis += self.dim()
    x = self if self.is_contiguous() else self.contiguous()
    scale = scale if scale.is_contiguous() else scale.contiguous()
    zero_point = zero_point if zero_point.is_contiguous() else zero_point.contiguous()
    n_channels = int(x.shape[axis])
    inner = 1
    for d in x.shape[axis + 1 :]:
        inner *= int(d)
    numel = x.numel()

    target = out
    if target is None:
        target = torch.empty_like(x)
    elif target.dtype != x.dtype or tuple(target.shape) != tuple(x.shape):
        raise RuntimeError(
            "_fake_quantize_learnable_per_channel_affine: out must match the "
            f"input shape {tuple(x.shape)} and dtype {x.dtype}"
        )
    contiguous_out = target.is_contiguous()
    store = target if contiguous_out else torch.empty_like(x)

    if numel:
        quant_min = float(quant_min)
        quant_max = float(quant_max)
        if inner == 1 and n_channels <= 4096 and numel // n_channels <= _MAX_GRID:
            col = _next_pow2(n_channels)
            rows_total = numel // n_channels
            tile_rows = max(
                min(
                    _INNER1_BLOCK // col,
                    _pow2_at_most(rows_total, _INNER1_BLOCK // col),
                ),
                1,
            )
            grid = (triton.cdiv(rows_total, tile_rows),)
            _learnable_inner1_kernel[grid](
                x,
                scale,
                zero_point,
                store,
                numel,
                n_channels,
                quant_min,
                quant_max,
                COL=col,
                TILE_ROWS=tile_rows,
                num_warps=4,
            )
        elif inner >= _RUN_MIN_INNER and (numel // inner) <= _MAX_GRID:
            block = min(max(_pow2_at_most(inner, _RUN_MAX_BLOCK), 256), _RUN_MAX_BLOCK)
            num_warps = 4 if block >= 8192 else 2 if block >= 2048 else 1
            grid = (numel // inner, triton.cdiv(inner, block))
            _learnable_run_kernel[grid](
                x,
                scale,
                zero_point,
                store,
                inner,
                n_channels,
                quant_min,
                quant_max,
                BLOCK=block,
                num_warps=num_warps,
            )
        else:
            block = _SMALL_N if numel <= _SMALL_N else _FLAT_BLOCK
            grid = (triton.cdiv(numel, block),)
            _learnable_flat_kernel[grid](
                x,
                scale,
                zero_point,
                store,
                numel,
                quant_min,
                quant_max,
                INNER=inner,
                N_CHANNELS=n_channels,
                BLOCK=block,
                num_warps=4,
            )
    if not contiguous_out:
        target.copy_(store)
    return target


def _fake_quantize_learnable_per_channel_affine_inner(
    self, scale, zero_point, axis, quant_min, quant_max, out=None
):
    return _launch(self, scale, zero_point, axis, quant_min, quant_max, out)


def _fake_quantize_learnable_per_channel_affine(
    self, scale, zero_point, axis, quant_min, quant_max, grad_factor=1.0
):
    logger.debug("GEMS_ENFLAME FAKE_QUANTIZE_LEARNABLE_PER_CHANNEL_AFFINE")
    if self.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError(
            f"unsupported dtype {self.dtype}; only floating-point dtypes are supported"
        )
    return _fake_quantize_learnable_per_channel_affine_inner(
        self, scale, zero_point, axis, quant_min, quant_max
    )


def _fake_quantize_learnable_per_channel_affine_out_impl(
    self,
    scale,
    zero_point,
    axis,
    quant_min,
    quant_max,
    grad_factor=1.0,
    *,
    out=None,
):
    logger.debug("GEMS_ENFLAME FAKE_QUANTIZE_LEARNABLE_PER_CHANNEL_AFFINE_OUT")
    return _fake_quantize_learnable_per_channel_affine_inner(
        self, scale, zero_point, axis, quant_min, quant_max, out=out
    )
