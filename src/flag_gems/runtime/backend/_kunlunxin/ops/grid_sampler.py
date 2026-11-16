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

# Reuse the Kunlunxin XPU `grid_sample` kernels:
#   * constant-folded output/input dimensions + strides (`tl.constexpr`),
#   * `NEED_MASK` so an exactly divisible grid avoids the slow masked path,
#   * clamped-address + explicit `tl.where` zeroing for `padding_mode=zeros`
#     (the XPU masked-load `other=0.0` value is ignored by the backend).
# The kernels walk every dimension with explicit strides, so input/grid do not
# need to be materialized as contiguous copies.
#
# ATen `grid_sampler` mode codes differ from the shared kernels' codes:
#   ATen 4D:  0 = bilinear, 1 = nearest, 2 = bicubic
#   ATen 5D:  0 = bilinear, 1 = nearest
#   shared 2D kernel: 0 = nearest, 1 = bilinear, 2 = bicubic
#   shared 3D kernel: 0 = nearest, 1 = bilinear
from .grid_sample import (
    _grid_sample_2d_kunlunxin_kernel,
    _grid_sample_3d_kunlunxin_kernel,
)

logger = logging.getLogger(__name__)

_ATEN_TO_KERNEL_MODE_2D = {
    0: 1,  # bilinear
    1: 0,  # nearest
    2: 2,  # bicubic
}
_ATEN_TO_KERNEL_MODE_3D = {
    0: 1,  # bilinear (trilinear)
    1: 0,  # nearest
}

_BLOCK = 512


def grid_sampler(
    input: torch.Tensor,
    grid: torch.Tensor,
    interpolation_mode: int = 0,
    padding_mode: int = 0,
    align_corners: bool = False,
):
    """Samples the input tensor at normalized grid coordinates.

    Uses bilinear, nearest, or bicubic interpolation with zeros, border, or
    reflection padding. 4D (N, C, IH, IW) inputs use bilinear/nearest/bicubic
    interpolation; 5D (N, C, ID, IH, IW) inputs use trilinear/nearest.

    Args:
        input: 4D (N, C, IH, IW) or 5D (N, C, ID, IH, IW) tensor.
        grid: (N, OH, OW, 2) or (N, OD, OH, OW, 3) flow field in [-1, 1].
        interpolation_mode: 0=Bilinear, 1=Nearest, 2=Bicubic (4D only).
        padding_mode: 0=Zeros, 1=Border, 2=Reflection.
        align_corners: If True, extrema (-1, 1) refer to the corner pixel centers.
    """
    logger.debug("GEMS_KUNLUNXIN GRID_SAMPLER")

    if input.ndim not in (4, 5):
        raise ValueError(
            f"grid_sampler(): expected 4D or 5D input, but got {input.ndim}D input"
        )
    if grid.shape[-1] != input.ndim - 2:
        raise ValueError(
            f"grid_sampler(): expected grid to have {input.ndim - 2} in its last "
            f"dimension, but got {grid.shape[-1]}"
        )
    if grid.shape[0] != input.shape[0]:
        raise ValueError(
            f"grid_sampler(): expected grid and input to have the same size of "
            f"the first dimension, but got grid.size(0) = {grid.shape[0]} and "
            f"input.size(0) = {input.shape[0]}"
        )
    if interpolation_mode not in (0, 1, 2):
        raise ValueError(
            f"grid_sampler(): expected interpolation_mode to be 0, 1 or 2, "
            f"but got {interpolation_mode}"
        )
    if padding_mode not in (0, 1, 2):
        raise ValueError(
            f"grid_sampler(): expected padding_mode to be 0, 1 or 2, "
            f"but got {padding_mode}"
        )
    if interpolation_mode == 2 and input.ndim == 5:
        raise ValueError("grid_sampler(): bicubic interpolation only supports 4D input")

    with torch_device_fn.device(input.device):
        if input.ndim == 5:
            N, C, ID, IH, IW = input.shape
            OD, OH, OW = grid.shape[1], grid.shape[2], grid.shape[3]
            output = torch.empty(
                (N, C, OD, OH, OW), dtype=input.dtype, device=input.device
            )
            total = output.numel()
            if total == 0:
                return output
            _grid_sample_3d_kunlunxin_kernel[(triton.cdiv(total, _BLOCK),)](
                output,
                input,
                grid,
                total,
                C=C,
                D_IN=ID,
                H_IN=IH,
                W_IN=IW,
                D_OUT=OD,
                H_OUT=OH,
                W_OUT=OW,
                IN_SN=input.stride(0),
                IN_SC=input.stride(1),
                IN_SD=input.stride(2),
                IN_SH=input.stride(3),
                IN_SW=input.stride(4),
                GR_SN=grid.stride(0),
                GR_SD=grid.stride(1),
                GR_SH=grid.stride(2),
                GR_SW=grid.stride(3),
                GR_SC=grid.stride(4),
                MODE=_ATEN_TO_KERNEL_MODE_3D[interpolation_mode],
                PADDING=padding_mode,
                ALIGN=align_corners,
                BLOCK=_BLOCK,
                NEED_MASK=(total % _BLOCK != 0),
                num_warps=4,
                num_stages=1,
                isCloseVectorization=True,
                buffer_size_limit=2048,
                unroll_num=8,
            )
            return output

        N, C, IH, IW = input.shape
        OH, OW = grid.shape[1], grid.shape[2]
        output = torch.empty((N, C, OH, OW), dtype=input.dtype, device=input.device)
        total = output.numel()
        if total == 0:
            return output
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
            MODE=_ATEN_TO_KERNEL_MODE_2D[interpolation_mode],
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
