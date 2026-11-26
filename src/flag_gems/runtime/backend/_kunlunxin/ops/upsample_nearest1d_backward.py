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
"""Kunlunxin copy of the generic module.

The element loops are unrolled at trace time; a runtime ``range`` does not compile
for these kernels on this backend.  Dense (N, C, W) float operands take a flat
kernel whose runtime arguments are fixed in number; everything else takes the
strided per-lane kernel.
"""

import logging
import math
import struct
from typing import Optional, Sequence

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn
from flag_gems.runtime.backend._kunlunxin.ops.upsample_nearest3d_backward import (
    _reciprocal_scale,
)

logger = logging.getLogger(__name__)

# Lanes per program: the dominant cost on this backend is bytes per program,
# not the arithmetic, so one wide tile per launch is what carries throughput.
_BLOCK_SIZE = 4096

_DENSE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
# These sizes may legitimately be 1, and a size of 1 must stay a runtime
# argument so that the argument list keeps its shape.
_DENSE_INT_ARGS = ("numel", "input_w", "output_w")


@triton.jit(do_not_specialize=list(_DENSE_INT_ARGS))
def upsample_nearest1d_backward_dense_kernel(
    grad_output,
    grad_input,
    numel,
    input_w,
    output_w,
    backward_scale,
    # ATen's forward scale (in / out).  Unused by this body; it completes the
    # fixed runtime argument list.
    reciprocal_scale,
    MAX_CONTRIBUTORS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    # Tail lanes recompute the last row; the store mask drops them.
    plane = tl.minimum(offsets // input_w, numel // input_w - 1)
    input_x = offsets % input_w

    input_x_f32 = input_x.to(tl.float32)
    start = tl.ceil(input_x_f32 * backward_scale).to(tl.int32)
    end = tl.ceil((input_x_f32 + 1.0) * backward_scale).to(tl.int32)
    start = tl.minimum(tl.maximum(start, 0), output_w)
    end = tl.minimum(tl.maximum(end, 0), output_w)

    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for contributor in tl.static_range(MAX_CONTRIBUTORS):
        output_x = start + contributor
        contributor_mask = mask & (output_x < end)
        # Clamped address: a masked-off lane still performs its access.
        output_x_c = tl.minimum(output_x, output_w - 1)
        value = tl.load(grad_output + plane * output_w + output_x_c)
        accumulator += value.to(tl.float32) * contributor_mask.to(tl.float32)
    tl.store(
        grad_input + offsets,
        accumulator.to(grad_input.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _upsample_nearest1d_backward_kernel(
    grad_output,
    grad_input,
    numel,
    channels,
    output_w,
    input_w,
    grad_output_stride_n,
    grad_output_stride_c,
    grad_output_stride_w,
    grad_input_stride_n,
    grad_input_stride_c,
    grad_input_stride_w,
    backward_scale,
    BATCH: tl.constexpr,
    BATCH_LOOP: tl.constexpr,
    IS_FP64: tl.constexpr,
    IS_UINT8: tl.constexpr,
    MAX_CONTRIBUTORS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    input_x = offsets % input_w
    acc_dtype: tl.constexpr = (
        tl.int32 if IS_UINT8 else (tl.float64 if IS_FP64 else tl.float32)
    )
    if BATCH_LOOP:
        channel = tl.minimum(offsets // input_w, channels - 1)
    else:
        nc = tl.minimum(offsets // input_w, BATCH * channels - 1)
        channel = nc % channels
        batch = nc // channels

    # Match nearest_neighbor_bw_compute_source_index from PyTorch:
    # ceil(input_index * scale), clamped to output_w.
    input_x_f32 = input_x.to(tl.float32)
    start = tl.ceil(input_x_f32 * backward_scale).to(tl.int32)
    end = tl.ceil((input_x_f32 + 1.0) * backward_scale).to(tl.int32)
    start = tl.minimum(tl.maximum(start, 0), output_w)
    end = tl.minimum(tl.maximum(end, 0), output_w)

    if BATCH_LOOP:
        for batch_index in tl.static_range(BATCH):
            accumulator = tl.zeros((BLOCK_SIZE,), dtype=acc_dtype)
            output_base = (
                batch_index * grad_output_stride_n + channel * grad_output_stride_c
            )
            for contributor in tl.static_range(MAX_CONTRIBUTORS):
                output_x = start + contributor
                contributor_mask = mask & (output_x < end)
                # Clamped address: a masked-off lane still performs its access.
                output_x_c = tl.minimum(output_x, output_w - 1)
                value = tl.load(
                    grad_output + output_base + output_x_c * grad_output_stride_w,
                )
                accumulator += value.to(acc_dtype) * contributor_mask.to(acc_dtype)
            input_offset = (
                batch_index * grad_input_stride_n
                + channel * grad_input_stride_c
                + input_x * grad_input_stride_w
            )
            tl.store(grad_input + input_offset, accumulator, mask=mask)
    else:
        accumulator = tl.zeros((BLOCK_SIZE,), dtype=acc_dtype)
        output_base = batch * grad_output_stride_n + channel * grad_output_stride_c
        for contributor in tl.static_range(MAX_CONTRIBUTORS):
            output_x = start + contributor
            contributor_mask = mask & (output_x < end)
            output_x_c = tl.minimum(output_x, output_w - 1)
            value = tl.load(
                grad_output + output_base + output_x_c * grad_output_stride_w,
            )
            accumulator += value.to(acc_dtype) * contributor_mask.to(acc_dtype)
        input_offset = (
            batch * grad_input_stride_n
            + channel * grad_input_stride_c
            + input_x * grad_input_stride_w
        )
        tl.store(grad_input + input_offset, accumulator, mask=mask)


def _validate_args(
    grad_output: torch.Tensor,
    output_size: Sequence[int],
    input_size: Sequence[int],
) -> tuple[int, int, int, int]:
    if grad_output.device.type != device.name:
        raise RuntimeError(
            f"Expected grad_output on {device.name}, but got {grad_output.device.type}"
        )
    if grad_output.ndim != 3:
        raise RuntimeError("Expected grad_output to be a 3D tensor")
    if len(output_size) != 1:
        raise RuntimeError("Expected output_size to contain one element")
    if len(input_size) != 3:
        raise RuntimeError("Expected input_size to contain three elements")
    if not grad_output.is_floating_point() and grad_output.dtype != torch.uint8:
        raise RuntimeError(
            f'"upsample_nearest1d_backward_out_frame" not implemented for '
            f"'{grad_output.dtype}'"
        )

    output_w = int(output_size[0])
    batch, channels, input_w = (int(value) for value in input_size)
    if input_w <= 0 or output_w <= 0:
        raise RuntimeError("Input and output sizes should be greater than 0")
    if tuple(grad_output.shape) != (batch, channels, output_w):
        raise RuntimeError(
            f"Expected grad_output shape {(batch, channels, output_w)}, "
            f"but got {tuple(grad_output.shape)}"
        )
    return batch, channels, input_w, output_w


def _as_float32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def _upsample_nearest1d_backward_impl(
    grad_output: torch.Tensor,
    output_size: Sequence[int],
    input_size: Sequence[int],
    scales: Optional[float],
    grad_input: Optional[torch.Tensor],
) -> torch.Tensor:
    batch, channels, input_w, output_w = _validate_args(
        grad_output, output_size, input_size
    )
    if grad_input is None:
        grad_input = torch.empty(
            (batch, channels, input_w),
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
    else:
        if grad_input.device != grad_output.device:
            raise RuntimeError(
                f"Expected grad_input on {grad_output.device}, "
                f"but got {grad_input.device}"
            )
        if grad_input.dtype != grad_output.dtype:
            raise RuntimeError(
                f"Expected grad_input dtype {grad_output.dtype}, "
                f"but got {grad_input.dtype}"
            )
        grad_input.resize_((batch, channels, input_w))

    if grad_input.numel() == 0:
        return grad_input

    backward_scale = _as_float32(
        float(scales) if scales is not None and scales > 0.0 else output_w / input_w
    )
    # ceil(i*s) and ceil((i+1)*s) differ by at most ceil(s) contributors, and the
    # [0, output_w] clamp can only shrink that difference.
    max_contributors = min(output_w, max(1, math.ceil(backward_scale)))
    numel = grad_input.numel()
    block_size = min(_BLOCK_SIZE, triton.next_power_of_2(numel))
    with torch_device_fn.device(grad_output.device):
        if (
            grad_output.is_contiguous()
            and grad_input.is_contiguous()
            and grad_output.dtype in _DENSE_DTYPES
        ):
            upsample_nearest1d_backward_dense_kernel[(triton.cdiv(numel, block_size),)](
                grad_output,
                grad_input,
                numel,
                input_w,
                output_w,
                backward_scale,
                _reciprocal_scale(input_w, output_w, scales),
                MAX_CONTRIBUTORS=max_contributors,
                BLOCK_SIZE=block_size,
            )
            return grad_input
        # One lane per (batch, channel, input) element.  The per-batch device
        # loop is only used when the batch count divides the lane tile exactly,
        # which keeps every launch aligned with a whole (batch, channel) plane.
        batch_loop = batch > 1 and channels * input_w >= block_size
        _upsample_nearest1d_backward_kernel[(triton.cdiv(numel, block_size),)](
            grad_output,
            grad_input,
            numel,
            channels,
            output_w,
            input_w,
            *grad_output.stride(),
            *grad_input.stride(),
            backward_scale,
            BATCH=batch,
            BATCH_LOOP=batch_loop,
            IS_FP64=grad_output.dtype == torch.float64,
            IS_UINT8=grad_output.dtype == torch.uint8,
            MAX_CONTRIBUTORS=max_contributors,
            BLOCK_SIZE=block_size,
        )
    return grad_input


def upsample_nearest1d_backward(
    grad_output: torch.Tensor,
    output_size: Sequence[int],
    input_size: Sequence[int],
    scales: Optional[float] = None,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN UPSAMPLE_NEAREST1D_BACKWARD")
    return _upsample_nearest1d_backward_impl(
        grad_output, output_size, input_size, scales, None
    )


def upsample_nearest1d_backward_grad_input(
    grad_output: torch.Tensor,
    output_size: Sequence[int],
    input_size: Sequence[int],
    scales: Optional[float] = None,
    *,
    grad_input: torch.Tensor,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN UPSAMPLE_NEAREST1D_BACKWARD_GRAD_INPUT")
    return _upsample_nearest1d_backward_impl(
        grad_output, output_size, input_size, scales, grad_input
    )
