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

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import tl_extra_shim

logger = logging.getLogger(__name__)

# When inner reaches this width, one contiguous segment is enough to fill the vector unit, so take the scalar-scale run
# path
_RUN_MIN_INNER = 2048
# Upper limit on elements per contiguous segment / tile; beyond this the benefit per program drops
_RUN_MAX_BLOCK = 16384
# Block for the flat path: small tensors are dominated by launch overhead, and larger blocks are slower (measured
# (64,64) B4096 19.5us
# vs B256 5.8us), so bucket by numel instead of fixing one large block
_TINY_N = 8192
_MID_N = 65536
_TINY_BLOCK = 256
_MID_BLOCK = 1024
_FLAT_BLOCK = 4096
_INNER1_BLOCK = 8192
# When inner == 1, the 2D (TILE_ROWS, N_CHANNELS) tile only helps on large tensors that do not write cachemask
# (1M elements: no mask 67.8us vs flat 71.4us; with mask, flat 72.9us vs tile 148.3us)
_INNER1_MIN_N = 1 << 19
_MAX_GRID = 65535


def _is_pow2(value):
    return value > 0 and (value & (value - 1)) == 0


def _pow2_at_most(value, cap):
    block = 1
    while block * 2 <= value and block * 2 <= cap:
        block *= 2
    return max(block, 1)


@triton.jit
def _fake_quant_channel_run_kernel(
    input_ptr,
    scale_ptr,
    zero_point_ptr,
    output_ptr,
    mask_ptr,
    inner,
    n_channels,
    quant_min,
    quant_max,
    STORE_MASK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # A program covers BLOCK elements of the run_id-th (outer, channel) contiguous segment
    run_id = tl.program_id(0)
    channel = run_id % n_channels
    chunk_start = tl.program_id(1) * BLOCK
    inner_offsets = tl.arange(0, BLOCK)
    offsets = run_id * inner + chunk_start + inner_offsets
    mask = inner_offsets < inner - chunk_start

    scale = tl.load(scale_ptr + channel)
    zero_point = tl.load(zero_point_ptr + channel).to(tl.float32)
    lower = quant_min - zero_point
    upper = quant_max - zero_point

    x_fp32 = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    quantized = tl_extra_shim.nearbyint(x_fp32 / scale)
    if STORE_MASK:
        tl.store(
            mask_ptr + offsets, (quantized >= lower) & (quantized <= upper), mask=mask
        )
    clamped = tl.minimum(tl.maximum(quantized, lower), upper) * scale
    tl.store(output_ptr + offsets, clamped.to(output_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _fake_quant_channel_inner1_kernel(
    input_ptr,
    scale_ptr,
    zero_point_ptr,
    output_ptr,
    mask_ptr,
    n_elements,
    quant_min,
    quant_max,
    N_CHANNELS: tl.constexpr,
    TILE_ROWS: tl.constexpr,
    STORE_MASK: tl.constexpr,
):
    # inner == 1: the tensor is a (rows, n_channels) row-major matrix, and the channel index is the column index
    tile_start = tl.program_id(0) * (TILE_ROWS * N_CHANNELS)
    offsets = (
        tile_start
        + tl.arange(0, TILE_ROWS)[:, None] * N_CHANNELS
        + tl.arange(0, N_CHANNELS)[None, :]
    )
    mask = offsets < n_elements

    channels = tl.arange(0, N_CHANNELS)
    scale = tl.load(scale_ptr + channels)[None, :].to(tl.float32)
    zero_point = tl.load(zero_point_ptr + channels)[None, :].to(tl.float32)
    lower = quant_min - zero_point
    upper = quant_max - zero_point

    x_fp32 = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    quantized = tl_extra_shim.nearbyint(x_fp32 / scale)
    if STORE_MASK:
        tl.store(
            mask_ptr + offsets, (quantized >= lower) & (quantized <= upper), mask=mask
        )
    clamped = tl.minimum(tl.maximum(quantized, lower), upper) * scale
    tl.store(output_ptr + offsets, clamped.to(output_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _fake_quant_channel_flat_kernel(
    input_ptr,
    scale_ptr,
    zero_point_ptr,
    output_ptr,
    mask_ptr,
    n_elements,
    quant_min,
    quant_max,
    INNER: tl.constexpr,
    N_CHANNELS: tl.constexpr,
    STORE_MASK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    channel = (offsets // INNER) % N_CHANNELS

    scale = tl.load(scale_ptr + channel, mask=mask, other=1.0).to(tl.float32)
    zero_point = tl.load(zero_point_ptr + channel, mask=mask, other=0).to(tl.float32)
    lower = quant_min - zero_point
    upper = quant_max - zero_point

    x_fp32 = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    quantized = tl_extra_shim.nearbyint(x_fp32 / scale)
    if STORE_MASK:
        tl.store(
            mask_ptr + offsets, (quantized >= lower) & (quantized <= upper), mask=mask
        )
    clamped = tl.minimum(tl.maximum(quantized, lower), upper) * scale
    tl.store(output_ptr + offsets, clamped.to(output_ptr.dtype.element_ty), mask=mask)


def _channel_geometry(shape, axis):
    """Returns (n_channels, inner), where inner is the product of the dims after axis."""
    n_channels = shape[axis]
    inner = 1
    for dim in range(axis + 1, len(shape)):
        inner *= shape[dim]
    return n_channels, inner


def _pick_path(n_elements, inner, n_channels, store_mask):
    if (
        inner == 1
        and not store_mask
        and n_elements >= _INNER1_MIN_N
        and _is_pow2(n_channels)
        and n_channels <= 4096
        and triton.cdiv(n_elements, n_channels) <= _MAX_GRID
    ):
        return "inner1"
    if inner >= _RUN_MIN_INNER and (n_elements // inner) <= _MAX_GRID:
        return "run"
    return "flat"


def _flat_block(n_elements):
    """The flat path tiers blocks by numel; small tensors use small blocks to avoid idle lanes."""
    if n_elements <= _TINY_N:
        return _TINY_BLOCK
    if n_elements <= _MID_N:
        return _MID_BLOCK
    return _FLAT_BLOCK


def launch_fake_quantize_per_channel_affine(
    input,
    scale,
    zero_point,
    output,
    mask_output,
    axis,
    quant_min,
    quant_max,
):
    """Dispatches on the inner shape to one of three kernels; store_mask also writes cachemask."""
    store_mask = mask_output is not None
    n_elements = input.numel()
    if n_elements == 0:
        return
    n_channels, inner = _channel_geometry(input.shape, axis)
    path = _pick_path(n_elements, inner, n_channels, store_mask)
    quant_min = float(quant_min)
    quant_max = float(quant_max)

    with torch_device_fn.device(input.device):
        if path == "run":
            block = min(max(_pow2_at_most(inner, _RUN_MAX_BLOCK), 256), _RUN_MAX_BLOCK)
            num_warps = 4 if block >= 8192 else 2 if block >= 2048 else 1
            grid = (n_elements // inner, triton.cdiv(inner, block))
            _fake_quant_channel_run_kernel[grid](
                input,
                scale,
                zero_point,
                output,
                mask_output,
                inner,
                n_channels,
                quant_min,
                quant_max,
                STORE_MASK=store_mask,
                BLOCK=block,
                num_warps=num_warps,
            )
        elif path == "inner1":
            rows_total = n_elements // n_channels
            tile_rows = max(
                min(
                    _INNER1_BLOCK // n_channels,
                    _pow2_at_most(rows_total, _INNER1_BLOCK // n_channels),
                ),
                1,
            )
            grid = (triton.cdiv(rows_total, tile_rows),)
            _fake_quant_channel_inner1_kernel[grid](
                input,
                scale,
                zero_point,
                output,
                mask_output,
                n_elements,
                quant_min,
                quant_max,
                N_CHANNELS=n_channels,
                TILE_ROWS=tile_rows,
                STORE_MASK=store_mask,
                num_warps=4,
            )
        else:
            block = _flat_block(n_elements)
            grid = (triton.cdiv(n_elements, block),)
            _fake_quant_channel_flat_kernel[grid](
                input,
                scale,
                zero_point,
                output,
                mask_output,
                n_elements,
                quant_min,
                quant_max,
                INNER=inner,
                N_CHANNELS=n_channels,
                STORE_MASK=store_mask,
                BLOCK=block,
                num_warps=4,
            )


def fake_quantize_per_channel_affine(
    input, scale, zero_point, axis, quant_min, quant_max
):
    logger.debug("GEMS_ENFLAME FAKE_QUANTIZE_PER_CHANNEL_AFFINE")

    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")

    input = input.contiguous()
    scale = scale.contiguous()
    zero_point = zero_point.contiguous()

    output = torch.empty_like(input)
    launch_fake_quantize_per_channel_affine(
        input, scale, zero_point, output, None, axis, quant_min, quant_max
    )
    return output
