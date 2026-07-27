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
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _max_unpool3d_kernel(
    input_ptr,
    indices_ptr,
    output_ptr,
    in_plane_size,
    out_plane_size,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    indices = tl.load(indices_ptr + offsets, mask=mask, other=-1)
    plane = offsets // in_plane_size
    valid = mask & (indices >= 0) & (indices < out_plane_size)
    tl.store(output_ptr + plane * out_plane_size + indices, values, mask=valid)


def max_unpool3d(
    input: torch.Tensor,
    indices: torch.Tensor,
    kernel_size,
    stride=None,
    padding=0,
    output_size=None,
):
    logger.debug("GEMS_KUNLUNXIN MAX_UNPOOL3D")
    input = input.contiguous()
    indices = indices.contiguous()

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * 3
    if stride is None:
        stride = kernel_size
    elif isinstance(stride, int):
        stride = (stride,) * 3
    if isinstance(padding, int):
        padding = (padding,) * 3

    in_n, in_c, in_d, in_h, in_w = input.shape
    if output_size is None:
        out_d = (in_d - 1) * stride[0] - 2 * padding[0] + kernel_size[0]
        out_h = (in_h - 1) * stride[1] - 2 * padding[1] + kernel_size[1]
        out_w = (in_w - 1) * stride[2] - 2 * padding[2] + kernel_size[2]
    else:
        out_d, out_h, out_w = output_size

    output = torch.zeros(
        (in_n, in_c, out_d, out_h, out_w), device=input.device, dtype=input.dtype
    )
    numel = input.numel()
    if numel == 0:
        return output

    in_plane_size = in_d * in_h * in_w
    out_plane_size = out_d * out_h * out_w
    with torch_device_fn.device(input.device):
        _max_unpool3d_kernel[(triton.cdiv(numel, 256),)](
            input,
            indices,
            output,
            in_plane_size,
            out_plane_size,
            numel,
            BLOCK_SIZE=256,
        )
    return output
