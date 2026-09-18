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
import struct
from typing import Optional, Sequence

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _upsample_nearest_exact1d_backward_kernel(
    grad_output,
    grad_input,
    numel,
    channels,
    output_w,
    input_w,
    go_stride_n,
    go_stride_c,
    go_stride_w,
    gi_stride_n,
    gi_stride_c,
    gi_stride_w,
    backward_scale,
    GI_CONTIGUOUS: tl.constexpr,
    MAX_CONTRIBUTORS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # One lane per input element over the whole [N, C, W_in] grad_input. The
    # generic kernel took `BATCH` as a constexpr and unrolled
    # `for batch_index in range(BATCH)` whenever batch <= 32, emitting up to
    # 32 * MAX_CONTRIBUTORS masked loads and stores per program. The backend
    # compiler gives up on that with `uni_sram out of resource: Required: 0,
    # Hardware limit: 0` followed by `PassManager::run failed` - 44 of 50 cases
    # here plus all 8 grad_input cases.
    #
    # The window walk uses `tl.static_range` and does every clamp in fp32
    # before narrowing to int32. A plain `range` loop, or clamping an int32
    # tensor against a runtime scalar, makes the backend emit
    # `arith.addi`/`arith.addf`/`arith.cmpi` with mismatched operand types and
    # the compiler aborts. This is the shape that upsample_linear1d_backward
    # already compiles with.
    #
    # Reads are clamped and zeroed with `tl.where` rather than masked: a data
    # dependent load predicate leaks the real in-range value on this backend.
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offsets_c = tl.minimum(offsets, numel - 1)

    input_x = offsets_c % input_w
    nc = offsets_c // input_w
    channel = nc % channels
    batch = nc // channels

    # nearest_neighbor_exact_bw_compute_source_index: ceil(idx * scale - 0.5).
    input_x_f = input_x.to(tl.float32)
    out_w_f = tl.cast(output_w, tl.float32)
    start_f = tl.ceil(input_x_f * backward_scale - 0.5)
    end_f = tl.ceil((input_x_f + 1.0) * backward_scale - 0.5)
    start_f = tl.minimum(tl.maximum(start_f, 0.0), out_w_f)
    end_f = tl.minimum(tl.maximum(end_f, 0.0), out_w_f)

    go_base = grad_output + batch * go_stride_n + channel * go_stride_c
    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for contributor in tl.static_range(MAX_CONTRIBUTORS):
        output_x_f = start_f + float(contributor)
        contributes = output_x_f < end_f
        output_x = tl.minimum(tl.maximum(output_x_f, 0.0), out_w_f - 1.0).to(tl.int32)
        value = tl.load(go_base + output_x * go_stride_w).to(tl.float32)
        accumulator += tl.where(contributes, value, 0.0)

    # Stores are clamped rather than masked. A tail lane recomputes the last
    # element from the clamped index, so writing it again is a redundant write
    # of the correct value; a masked store past the end of the buffer segfaults
    # here when numel is small relative to BLOCK_SIZE (numel == 1 with
    # BLOCK_SIZE 256 walks 255 elements off the end).
    result = accumulator.to(grad_input.dtype.element_ty)
    if GI_CONTIGUOUS:
        tl.store(grad_input + offsets_c, result)
    else:
        # The out= overload accepts a strided grad_input, so write through its
        # strides instead of materialising a contiguous buffer and copying it
        # back (a strided copy_ fails on this backend).
        gi_offset = batch * gi_stride_n + channel * gi_stride_c + input_x * gi_stride_w
        tl.store(grad_input + gi_offset, result)


def _validate_args(
    grad_output: torch.Tensor,
    output_size: Sequence[int],
    input_size: Sequence[int],
):
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


def _impl(
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
    max_contributors = min(output_w, max(1, math.ceil(backward_scale) + 1))
    numel = grad_input.numel()
    block_size = 256

    with torch_device_fn.device(grad_output.device):
        _upsample_nearest_exact1d_backward_kernel[(triton.cdiv(numel, block_size),)](
            grad_output,
            grad_input,
            numel,
            channels,
            output_w,
            input_w,
            *grad_output.stride(),
            *grad_input.stride(),
            backward_scale,
            GI_CONTIGUOUS=grad_input.is_contiguous(),
            MAX_CONTRIBUTORS=max_contributors,
            BLOCK_SIZE=block_size,
        )
    return grad_input


def _upsample_nearest_exact1d_backward(
    grad_output: torch.Tensor,
    output_size: Sequence[int],
    input_size: Sequence[int],
    scales: Optional[float] = None,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN _UPSAMPLE_NEAREST_EXACT1D_BACKWARD")
    return _impl(grad_output, output_size, input_size, scales, None)


def _upsample_nearest_exact1d_backward_grad_input(
    grad_output: torch.Tensor,
    output_size: Sequence[int],
    input_size: Sequence[int],
    scales: Optional[float] = None,
    *,
    grad_input: torch.Tensor,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN _UPSAMPLE_NEAREST_EXACT1D_BACKWARD.GRAD_INPUT")
    return _impl(grad_output, output_size, input_size, scales, grad_input)
