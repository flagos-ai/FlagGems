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

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

logger = logging.getLogger(__name__)


@libentry()
@triton.jit(
    do_not_specialize=[
        "scale",
        "inv_scale",
        "zero_point",
        "quant_min",
        "quant_max",
    ]
)
def fake_quantize_per_tensor_affine_cachemask_kernel(
    input_ptr,
    output_ptr,
    mask_ptr,
    n_elements,
    scale,
    inv_scale,
    zero_point,
    quant_min,
    quant_max,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    active = offsets < n_elements

    value = tl.load(input_ptr + offsets, mask=active).to(tl.float32)
    scale = scale.to(tl.float32)
    inv_scale = inv_scale.to(tl.float32)
    quantized = tl_extra_shim.nearbyint(value * inv_scale) + zero_point
    cachemask = (quantized >= quant_min) & (quantized <= quant_max)
    quantized = tl.minimum(tl.maximum(quantized, quant_min), quant_max)
    output = (quantized - zero_point) * scale

    tl.store(output_ptr + offsets, output, mask=active)
    tl.store(mask_ptr + offsets, cachemask, mask=active)


def _validate_and_convert_qparams(scale, zero_point, quant_min, quant_max):
    if quant_min > quant_max:
        raise RuntimeError("quant_min must be less than or equal to quant_max")
    if zero_point < quant_min or zero_point > quant_max:
        raise RuntimeError("zero_point must be between quant_min and quant_max")

    scale = struct.unpack("f", struct.pack("f", float(scale)))[0]
    if scale == 0.0:
        inv_scale = math.copysign(math.inf, scale)
    else:
        inv_scale = struct.unpack("f", struct.pack("f", 1.0 / scale))[0]
    return scale, inv_scale


def _launch(
    input,
    output,
    cachemask,
    scale,
    zero_point,
    quant_min,
    quant_max,
):
    scale, inv_scale = _validate_and_convert_qparams(
        scale, zero_point, quant_min, quant_max
    )
    contiguous_input = input.contiguous()
    n_elements = contiguous_input.numel()
    if n_elements == 0:
        return

    contiguous_output = output
    if not output.is_contiguous():
        contiguous_output = torch.empty_like(contiguous_input)
    contiguous_mask = cachemask
    if not cachemask.is_contiguous():
        contiguous_mask = torch.empty_like(contiguous_input, dtype=torch.bool)

    block_size = 1024
    grid = (triton.cdiv(n_elements, block_size),)
    with torch_device_fn.device(input.device):
        fake_quantize_per_tensor_affine_cachemask_kernel[grid](
            contiguous_input,
            contiguous_output,
            contiguous_mask,
            n_elements,
            scale,
            inv_scale,
            int(zero_point),
            int(quant_min),
            int(quant_max),
            BLOCK_SIZE=block_size,
        )

    if contiguous_output is not output:
        output.copy_(contiguous_output)
    if contiguous_mask is not cachemask:
        cachemask.copy_(contiguous_mask)


def fake_quantize_per_tensor_affine_cachemask(
    input, scale, zero_point, quant_min, quant_max
):
    logger.debug("GEMS FAKE_QUANTIZE_PER_TENSOR_AFFINE_CACHEMASK")
    contiguous_input = input.contiguous()
    output = torch.empty_like(contiguous_input)
    cachemask = torch.empty_like(contiguous_input, dtype=torch.bool)
    _launch(
        contiguous_input,
        output,
        cachemask,
        scale,
        zero_point,
        quant_min,
        quant_max,
    )
    return output, cachemask


def fake_quantize_per_tensor_affine_cachemask_out(
    input,
    scale,
    zero_point,
    quant_min,
    quant_max,
    *,
    out0,
    out1,
):
    logger.debug("GEMS FAKE_QUANTIZE_PER_TENSOR_AFFINE_CACHEMASK_OUT")
    if out0.dtype != input.dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {input.dtype}, but got {out0.dtype} instead"
        )
    if out1.dtype != torch.bool:
        raise RuntimeError(
            f"Expected out tensor to have dtype bool, but got {out1.dtype} instead"
        )
    if out0.device != input.device or out1.device != input.device:
        raise RuntimeError("input and out tensors must be on the same device")

    out0.resize_(input.shape)
    out1.resize_(input.shape)
    _launch(input, out0, out1, scale, zero_point, quant_min, quant_max)
    return out0, out1
