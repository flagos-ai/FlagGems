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
from flag_gems.utils import libentry, tl_extra_shim
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)
rsqrt = tl_extra_shim.rsqrt


@libentry()
@triton.jit(do_not_specialize=["n_elements", "eps"])
def _batch_norm_no_update_kernel(
    input,
    weight,
    bias,
    running_mean,
    running_var,
    output,
    n_elements,
    eps,
    C: tl.constexpr,
    INNER: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = ext.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    channels = (offsets // INNER) % C

    x = tl.load(input + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(running_mean + channels, mask=mask, other=0.0).to(tl.float32)
    variance = tl.load(running_var + channels, mask=mask, other=0.0).to(tl.float32)
    result = (x - mean) * rsqrt(variance + eps)

    if HAS_WEIGHT:
        scale = tl.load(weight + channels, mask=mask, other=0.0).to(tl.float32)
        result *= scale
    if HAS_BIAS:
        shift = tl.load(bias + channels, mask=mask, other=0.0).to(tl.float32)
        result += shift

    tl.store(output + offsets, result, mask=mask)


def _batch_norm_no_update(
    input,
    weight=None,
    bias=None,
    running_mean=None,
    running_var=None,
    momentum=0.1,
    eps=1e-5,
):
    logger.debug("GEMS_KUNLUNXIN _BATCH_NORM_NO_UPDATE")
    if input.ndim < 2:
        raise RuntimeError("batch_norm expects input with at least 2 dimensions")
    if running_mean is None or running_var is None:
        raise RuntimeError("running_mean and running_var are required for no-update batch_norm")

    channels = input.shape[1]
    if running_mean.numel() != channels or running_var.numel() != channels:
        raise RuntimeError("running statistics must contain one value per channel")

    input_contiguous = input.contiguous()
    output = torch.empty_like(input_contiguous)
    n_elements = input_contiguous.numel()
    inner = n_elements // input.shape[0] // channels

    if n_elements > 0:
        block_size = 256
        grid = (triton.cdiv(n_elements, block_size),)
        weight_pointer = input_contiguous if weight is None else weight
        bias_pointer = input_contiguous if bias is None else bias
        with torch_device_fn.device(input.device):
            _batch_norm_no_update_kernel[grid](
                input_contiguous,
                weight_pointer,
                bias_pointer,
                running_mean,
                running_var,
                output,
                n_elements,
                eps,
                C=channels,
                INNER=inner,
                HAS_WEIGHT=weight is not None,
                HAS_BIAS=bias is not None,
                BLOCK_SIZE=block_size,
                isCloseVectorization=True,
                buffer_size_limit=2048,
            )

    save_mean = torch.empty((0,), dtype=input.dtype, device=input.device)
    save_var = torch.empty((0,), dtype=input.dtype, device=input.device)
    reserved = torch.empty((0,), dtype=torch.uint8, device=input.device)
    return output.view_as(input), save_mean, save_var, reserved
