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
@triton.jit(do_not_specialize=["scale", "zero_point", "quant_min", "quant_max"])
def fake_quantize_per_tensor_affine_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    scale,
    inv_scale,
    zero_point,
    quant_min,
    quant_max,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    value = tl.load(input_ptr + offsets, mask=mask).to(tl.float32)
    scale = scale.to(tl.float32)
    inv_scale = inv_scale.to(tl.float32)
    quantized = tl_extra_shim.nearbyint(value * inv_scale) + zero_point
    quantized = tl.minimum(tl.maximum(quantized, quant_min), quant_max)
    output = (quantized - zero_point) * scale
    tl.store(output_ptr + offsets, output, mask=mask)


def fake_quantize_per_tensor_affine(input, scale, zero_point, quant_min, quant_max):
    logger.debug("GEMS FAKE_QUANTIZE_PER_TENSOR_AFFINE")

    input = input.contiguous()
    if isinstance(scale, torch.Tensor):
        scale = scale.item()
    if isinstance(zero_point, torch.Tensor):
        zero_point = zero_point.item()
    if quant_min > quant_max:
        raise RuntimeError("quant_min must be less than or equal to quant_max")
    if zero_point < quant_min or zero_point > quant_max:
        raise RuntimeError("zero_point must be between quant_min and quant_max")

    output = torch.empty_like(input)
    n_elements = input.numel()
    if n_elements == 0:
        return output

    scale = struct.unpack("f", struct.pack("f", float(scale)))[0]
    if scale == 0.0:
        inv_scale = math.copysign(math.inf, scale)
    else:
        inv_scale = struct.unpack("f", struct.pack("f", 1.0 / scale))[0]
    block_size = 1024
    grid = (triton.cdiv(n_elements, block_size),)
    with torch_device_fn.device(input.device):
        fake_quantize_per_tensor_affine_kernel[grid](
            input,
            output,
            n_elements,
            scale,
            inv_scale,
            int(zero_point),
            int(quant_min),
            int(quant_max),
            BLOCK_SIZE=block_size,
        )
    return output
