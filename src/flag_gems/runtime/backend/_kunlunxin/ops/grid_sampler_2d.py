# Copyright 2026, The FlagOS Contributors.
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

from flag_gems.runtime import torch_device_fn

# Reuse the Kunlunxin XPU `grid_sample` kernel:
#   * constant-folded output/input dimensions + strides (`tl.constexpr`),
#   * `NEED_MASK` so an exactly divisible grid avoids the slow masked path,
#   * clamped-address + explicit `tl.where` zeroing for `padding_mode=zeros`
#     (the XPU masked-load `other=0.0` value is ignored by the backend).
# The ATen `grid_sampler_2d` mode codes differ from the shared kernel's codes:
#   ATen:  0 = bilinear, 1 = nearest, 2 = bicubic
#   shared: 0 = nearest,  1 = bilinear, 2 = bicubic
from .grid_sample import _grid_sample_2d_kunlunxin_kernel

logger = logging.getLogger(__name__)

_ATEN_TO_KERNEL_MODE = {
    0: 1,  # bilinear
    1: 0,  # nearest
    2: 2,  # bicubic
}

_BLOCK = 512


def grid_sampler_2d(
    input: torch.Tensor,
    grid: torch.Tensor,
    interpolation_mode: int = 0,
    padding_mode: int = 0,
    align_corners: bool = False,
):
    """Performs grid sampling of the input tensor using the given grid.

    Uses bilinear, nearest, or bicubic interpolation with zeros, border, or
    reflection padding.

    Args:
        input: 4D tensor of shape (N, C, IH, IW).
        grid: 4D tensor of shape (N, OH, OW, 2) with normalized coords.
        interpolation_mode: 0=Bilinear, 1=Nearest, 2=Bicubic.
        padding_mode: 0=Zeros, 1=Border, 2=Reflection.
        align_corners: If True, corner pixels are aligned.
    """
    logger.debug("GEMS_KUNLUNXIN GRID_SAMPLER_2D")

    assert input.ndim == 4, "Input must be 4D"
    assert grid.ndim == 4, "Grid must be 4D"
    assert grid.shape[-1] == 2, "Grid must have last dimension of size 2"

    N, C, IH, IW = input.shape
    OH, OW = grid.shape[1:3]

    assert grid.shape[0] == N, "Batch size mismatch"

    output = torch.empty((N, C, OH, OW), dtype=input.dtype, device=input.device)
    total = output.numel()
    if total == 0:
        return output

    # The kernel walks every dimension with explicit strides, so input/grid do
    # not need to be materialized as contiguous copies.
    with torch_device_fn.device(input.device):
        _grid_sample_2d_kunlunxin_kernel[(triton.cdiv(total, _BLOCK),)](
            output,
            input,
            grid,
            total,
            C=C,
            H_IN=IH,
            W_IN=IW,
            H_OUT=OH,
            W_OUT=OW,
            IN_STRIDE_N=input.stride(0),
            IN_STRIDE_C=input.stride(1),
            IN_STRIDE_H=input.stride(2),
            IN_STRIDE_W=input.stride(3),
            GRID_STRIDE_N=grid.stride(0),
            GRID_STRIDE_H=grid.stride(1),
            GRID_STRIDE_W=grid.stride(2),
            GRID_STRIDE_C=grid.stride(3),
            MODE=_ATEN_TO_KERNEL_MODE[interpolation_mode],
            PADDING=padding_mode,
            ALIGN_CORNERS=align_corners,
            BLOCK=_BLOCK,
            NEED_MASK=(total % _BLOCK != 0),
            num_warps=4,
            num_stages=1,
            isCloseVectorization=True,
            buffer_size_limit=2048,
            unroll_num=8,
        )

    return output
