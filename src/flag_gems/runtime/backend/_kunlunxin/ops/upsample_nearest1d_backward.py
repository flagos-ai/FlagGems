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

logger = logging.getLogger(__name__)

# Lanes per program: the dominant cost on this backend is bytes per program,
# not the arithmetic, so one wide tile per launch is what carries throughput.
_BLOCK_SIZE = 4096
_MIN_BLOCK = 64

_DENSE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
# Triton would specialize a runtime argument that equals 1 into a constant, so
# these sizes stay ordinary runtime arguments.
_DENSE_INT_ARGS = ("numel", "input_w", "output_w")

# An integer backward scale s maps every grad_input[i] onto the fixed window
# grad_output[i*s .. i*s + s), an affine lane index, so the loads stay contiguous.
# Any other scale needs the per-output window search below.
_INT_SCALE_DEFAULT_MAX = 16
_RECIP_SCALE_MAX = 4
_HALF_SCALE_MIN = 1 << 14


@triton.jit(do_not_specialize=["numel"])
def upsample_nearest1d_backward_int_scale_kernel(
    grad_output,
    grad_input,
    numel,
    WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    FULL: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if FULL:
        # Every lane is in range; keeping the mask and the index clamp out of
        # the address is what keeps these loads contiguous.
        base = offsets * WIDTH
        accumulator = tl.load(grad_output + base).to(tl.float32)
        for contributor in tl.static_range(1, WIDTH):
            accumulator += tl.load(grad_output + base + contributor).to(tl.float32)
        tl.store(grad_input + offsets, accumulator.to(grad_input.dtype.element_ty))
    else:
        mask = offsets < numel
        base = tl.minimum(offsets, numel - 1) * WIDTH
        accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for contributor in tl.static_range(WIDTH):
            accumulator += tl.load(
                grad_output + base + contributor, mask=mask, other=0.0
            ).to(tl.float32)
        tl.store(
            grad_input + offsets,
            accumulator.to(grad_input.dtype.element_ty),
            mask=mask,
        )


# A scale of 1/r (integer r) is the downsampling mirror image: grad_input[f] is
# grad_output[f//r] when f is a multiple of r and 0 otherwise.  Indexing by the
# output keeps the store contiguous and the load a fixed-shift broadcast, which
# is what keeps both accesses vectorizable.
@triton.jit(do_not_specialize=["numel"])
def upsample_nearest1d_backward_recip_scale_kernel(
    grad_output,
    grad_input,
    numel,
    RATIO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    FULL: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if FULL:
        value = tl.load(grad_output + offsets // RATIO).to(tl.float32)
        value = tl.where((offsets % RATIO) == 0, value, 0.0)
        tl.store(grad_input + offsets, value.to(grad_input.dtype.element_ty))
    else:
        mask = offsets < numel
        lane = tl.minimum(offsets, numel - 1)
        value = tl.load(grad_output + lane // RATIO).to(tl.float32)
        value = tl.where((lane % RATIO) == 0, value, 0.0)
        tl.store(grad_input + lane, value.to(grad_input.dtype.element_ty), mask=mask)


@triton.jit(do_not_specialize=["numel"])
def upsample_nearest1d_backward_half_scale_kernel(
    grad_output,
    grad_input,
    numel,
    BLOCK_SIZE: tl.constexpr,
    FULL: tl.constexpr,
):
    # scale 1.5: grad_input[2j] = grad_output[3j] + grad_output[3j+1] and
    # grad_input[2j+1] = grad_output[3j+2], so every output lane reads at most
    # two contiguous grad_output elements at a fixed shift.
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if FULL:
        lane = offsets
        mask = None
    else:
        lane = tl.minimum(offsets, numel - 1)
        mask = offsets < numel
    base = (lane // 2) * 3
    odd = lane % 2
    first = tl.load(grad_output + base + 2 * odd, mask=mask).to(tl.float32)
    second = tl.load(grad_output + base + 1, mask=mask).to(tl.float32)
    value = tl.where(odd == 0, first + second, first)
    tl.store(grad_input + lane, value.to(grad_input.dtype.element_ty), mask=mask)


@triton.jit(do_not_specialize=list(_DENSE_INT_ARGS))
def upsample_nearest1d_backward_dense_kernel(
    grad_output,
    grad_input,
    numel,
    input_w,
    output_w,
    backward_scale,
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

    accumulator = tl.zeros((BLOCK_SIZE,), dtype=acc_dtype)
    output_base = batch * grad_output_stride_n + channel * grad_output_stride_c
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


def _block_for(numel: int, dtype: torch.dtype) -> int:
    """Lanes per program.

    A tile of exactly 1024 lanes returns unrelated values for bf16, while 512 and
    2048 are exact, so bf16 steps around that size.  Nothing narrower than
    _MIN_BLOCK is used: short tails run masked instead.
    """
    block = min(_BLOCK_SIZE, max(_MIN_BLOCK, triton.next_power_of_2(numel)))
    if dtype == torch.bfloat16 and block == 1024:
        block = 512
    return block


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
    block_size = _block_for(numel, grad_output.dtype)
    with torch_device_fn.device(grad_output.device):
        if (
            grad_output.is_contiguous()
            and grad_input.is_contiguous()
            and grad_output.dtype in _DENSE_DTYPES
        ):
            # Three scale families keep every window affine in the lane index:
            # an integer scale s (window [i*s, i*s+s)), its mirror 1/r (only one
            # of every r outputs is non-zero), and 1.5 (one or two neighbours).
            int_width = (
                int(backward_scale)
                if backward_scale == int(backward_scale)
                and 1 <= backward_scale <= _INT_SCALE_DEFAULT_MAX
                and (input_w - 1) * int(backward_scale) + int(backward_scale)
                <= output_w
                else 0
            )
            recip_ratio = (
                int(round(1.0 / backward_scale))
                if int(round(1.0 / backward_scale)) * backward_scale == 1
                and 2 <= round(1.0 / backward_scale) <= _RECIP_SCALE_MAX
                and output_w * int(round(1.0 / backward_scale)) == input_w
                else 0
            )
            # The 1.5 path trades two loads per lane for a contiguous store, which
            # only pays once the tensor is big enough to be bandwidth bound.
            half_scale = (
                backward_scale == 1.5
                and input_w % 2 == 0
                and 2 * output_w == 3 * input_w
                and numel >= _HALF_SCALE_MIN
            )

            def _run(kernel, lanes, source, target, source_step, consts, group=1):
                # Full-width programs run unmasked; anything left over runs as a
                # single masked program, since a mask or an index clamp in the
                # address is what turns these loads into a gather.  `group`
                # keeps the split aligned to both the block and the scale.
                step_num, step_den = source_step
                unit = block_size * group
                fast = (lanes // unit) * unit
                if fast:
                    kernel[(fast // block_size,)](
                        source,
                        target,
                        lanes,
                        BLOCK_SIZE=block_size,
                        FULL=fast == lanes,
                        **consts,
                    )
                if fast < lanes:
                    tail = lanes - fast
                    tail_block = _block_for(tail, grad_output.dtype)
                    kernel[(triton.cdiv(tail, tail_block),)](
                        source.view(-1)[(fast * step_num) // step_den :],
                        target.view(-1)[fast:],
                        tail,
                        BLOCK_SIZE=tail_block,
                        FULL=False,
                        **consts,
                    )

            if recip_ratio:
                _run(
                    upsample_nearest1d_backward_recip_scale_kernel,
                    numel,
                    grad_output,
                    grad_input,
                    (1, recip_ratio),
                    {"RATIO": recip_ratio},
                    group=recip_ratio,
                )
                return grad_input
            if int_width:
                _run(
                    upsample_nearest1d_backward_int_scale_kernel,
                    numel,
                    grad_output,
                    grad_input,
                    (int_width, 1),
                    {"WIDTH": int_width},
                )
                return grad_input
            if half_scale:
                _run(
                    upsample_nearest1d_backward_half_scale_kernel,
                    numel,
                    grad_output,
                    grad_input,
                    (3, 2),
                    {},
                )
                return grad_input
            upsample_nearest1d_backward_dense_kernel[(triton.cdiv(numel, block_size),)](
                grad_output,
                grad_input,
                numel,
                input_w,
                output_w,
                backward_scale,
                MAX_CONTRIBUTORS=max_contributors,
                BLOCK_SIZE=block_size,
            )
            return grad_input
        # One lane per (batch, channel, input) element, addressed through the
        # strides so any layout works.
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
