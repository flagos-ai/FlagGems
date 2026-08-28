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
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _upsample_nearest_exact2d_backward_kernel(
    grad_output,
    grad_input,
    numel,
    channels,
    input_h,
    input_w,
    output_h,
    output_w,
    go_stride_n,
    go_stride_c,
    go_stride_h,
    go_stride_w,
    bw_scale_h,
    bw_scale_w,
    MAX_CONTRIB_H: tl.constexpr,
    MAX_CONTRIB_W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Gather form: one lane per grad_input element sums its own contributor
    # window. The generic kernel scatters with `tl.atomic_add`, and atomics on
    # this backend drop concurrent updates - a handful of elements per run end
    # up short by exactly one or two whole contributions (abs diff 1.0/2.0 on
    # a ones gradient, tens of elements out of millions, 15 of 30 cases).
    #
    # The window walk uses `tl.static_range` and does every clamp in fp32
    # before narrowing to int32: a plain `range` loop, or clamping an int32
    # tensor against a runtime scalar, makes the backend emit
    # `arith.addi`/`arith.addf`/`arith.cmpi` with mismatched operand types
    # and compilation aborts with `PassManager::run failed`.
    #
    # Reads are clamped and zeroed with `tl.where` rather than masked: a data
    # dependent load predicate leaks the real in-range value on this backend.
    # Stores are clamped rather than masked: a masked store past the end of a
    # small buffer segfaults here.
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offsets_c = tl.minimum(offsets, numel - 1)

    iw = offsets_c % input_w
    tmp = offsets_c // input_w
    ih = tmp % input_h
    nc = tmp // input_h
    channel = nc % channels
    batch = nc // channels

    # nearest_neighbor_exact_bw_compute_source_index: the output indices that
    # scatter into input index i span [ceil(i * scale - 0.5),
    # ceil((i + 1) * scale - 0.5)).
    ih_f = ih.to(tl.float32)
    iw_f = iw.to(tl.float32)
    out_h_f = tl.cast(output_h, tl.float32)
    out_w_f = tl.cast(output_w, tl.float32)
    start_h_f = tl.ceil(ih_f * bw_scale_h - 0.5)
    end_h_f = tl.ceil((ih_f + 1.0) * bw_scale_h - 0.5)
    start_h_f = tl.minimum(tl.maximum(start_h_f, 0.0), out_h_f)
    end_h_f = tl.minimum(tl.maximum(end_h_f, 0.0), out_h_f)
    start_w_f = tl.ceil(iw_f * bw_scale_w - 0.5)
    end_w_f = tl.ceil((iw_f + 1.0) * bw_scale_w - 0.5)
    start_w_f = tl.minimum(tl.maximum(start_w_f, 0.0), out_w_f)
    end_w_f = tl.minimum(tl.maximum(end_w_f, 0.0), out_w_f)

    go_base = grad_output + batch * go_stride_n + channel * go_stride_c
    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for i in tl.static_range(MAX_CONTRIB_H):
        oh_f = start_h_f + float(i)
        h_contributes = oh_f < end_h_f
        oh = tl.minimum(tl.maximum(oh_f, 0.0), out_h_f - 1.0).to(tl.int32)
        row_base = go_base + oh * go_stride_h
        for j in tl.static_range(MAX_CONTRIB_W):
            ow_f = start_w_f + float(j)
            contributes = h_contributes & (ow_f < end_w_f)
            ow = tl.minimum(tl.maximum(ow_f, 0.0), out_w_f - 1.0).to(tl.int32)
            value = tl.load(row_base + ow * go_stride_w).to(tl.float32)
            accumulator += tl.where(contributes, value, 0.0)

    result = accumulator.to(grad_input.dtype.element_ty)
    tl.store(grad_input + offsets_c, result)


def _as_float32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def _upsample_nearest_exact2d_backward(
    grad_output: torch.Tensor,
    output_size: Tuple[int, int],
    input_size: Tuple[int, int, int, int],
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN _UPSAMPLE_NEAREST_EXACT2D_BACKWARD")

    assert grad_output.device.type == device.name
    assert grad_output.ndim == 4, "The ndim of grad_output must be 4"
    assert len(output_size) == 2, "The len of output_size must be 2"
    assert len(input_size) == 4, "The len of input_size must be 4"

    output_h, output_w = (int(value) for value in output_size)
    batch, channels, input_h, input_w = (int(value) for value in input_size)
    assert tuple(grad_output.shape) == (batch, channels, output_h, output_w), (
        f"grad_output shape {tuple(grad_output.shape)} does not match "
        f"expected shape (N={batch}, C={channels}, OH={output_h}, OW={output_w})"
    )

    grad_input = torch.empty(
        (batch, channels, input_h, input_w),
        dtype=grad_output.dtype,
        device=grad_output.device,
    )
    numel = grad_input.numel()
    if numel == 0:
        return grad_input

    bw_scale_h = _as_float32(
        float(scales_h)
        if scales_h is not None and scales_h > 0.0
        else output_h / input_h
    )
    bw_scale_w = _as_float32(
        float(scales_w)
        if scales_w is not None and scales_w > 0.0
        else output_w / input_w
    )
    max_contrib_h = min(output_h, max(1, math.ceil(bw_scale_h) + 1))
    max_contrib_w = min(output_w, max(1, math.ceil(bw_scale_w) + 1))
    block_size = 1024

    with torch_device_fn.device(grad_output.device):
        _upsample_nearest_exact2d_backward_kernel[(triton.cdiv(numel, block_size),)](
            grad_output,
            grad_input,
            numel,
            channels,
            input_h,
            input_w,
            output_h,
            output_w,
            *grad_output.stride(),
            bw_scale_h,
            bw_scale_w,
            MAX_CONTRIB_H=max_contrib_h,
            MAX_CONTRIB_W=max_contrib_w,
            BLOCK_SIZE=block_size,
        )
    return grad_input
