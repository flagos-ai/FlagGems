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

"""Kunlunxin XPU override for ``aten::grid_sampler_3d``.

Why this override exists
------------------------
The generic implementation (``flag_gems/ops/grid_sampler_3d.py``) launches one
*scalar* program per output point and selects the input values with
``tl.load(..., mask=<in-bounds>, other=0.0)`` for ``padding_mode=zeros``.  On
this backend the masked scalar-load path ignores ``other`` and reads the
(theoretically masked-off) address anyway: mask-false lanes return whatever
lives at the out-of-bounds address (huge values / NaN), and when that address
is unmapped the launch dies with ``KL_XID_KERNEL_EXCEPTION`` (status 700).
Consequently every ``padding_mode=zeros`` sample whose 8/1-tap footprint leaves
the input volume was wrong (83/252 functional cases), and the perf benchmark
crashed on its first case.

``grid_sampler_3d`` on 5-D input is exactly ``aten::grid_sampler`` for 5-D
input, whose Kunlunxin kernel already exists and is validated by
``tests/test_grid_sampler.py::test_grid_sampler_5d``.  This override reuses that
kernel:

  * BLOCK-tiled flat offsets (output points x channels) instead of one scalar
    program per spatial point, dims/strides folded as ``tl.constexpr``;
  * every tap index is clamped before the load, so no load ever leaves the
    input volume, and invalid taps are zeroed with an explicit ``tl.where`` --
    no reliance on ``other=0.0`` and no masked load for the gather itself;
  * ``NEED_MASK`` drops the mask when ``total % BLOCK == 0``.

ATen trilinear/nearest codes map to the shared kernel's codes as
``0 -> 1`` (bilinear) and ``1 -> 0`` (nearest).
"""

import logging

import torch
import triton

from flag_gems.runtime import torch_device_fn

# Reuse the committed Kunlunxin XPU `grid_sample` 3-D kernel (same kernel the
# `grid_sampler` 5-D path uses). It walks input/grid with explicit strides, so
# non-contiguous input/grid tensors are supported without materializing copies.
from .grid_sample import _grid_sample_3d_kunlunxin_kernel

logger = logging.getLogger(__name__)

_BLOCK = 512

# ATen interpolation codes -> shared kernel MODE (0 = nearest, 1 = trilinear)
_ATEN_TO_KERNEL_MODE = {
    0: 1,
    1: 0,
}


def grid_sampler_3d(
    input: torch.Tensor,
    grid: torch.Tensor,
    interpolation_mode: int = 0,
    padding_mode: int = 0,
    align_corners: bool = False,
):
    """Samples a 5-D input at normalized 3-D grid coordinates.

    Args:
        input: 5-D (N, C, ID, IH, IW) tensor.
        grid: (N, OD, OH, OW, 3) flow field in [-1, 1].
        interpolation_mode: 0=trilinear (bilinear), 1=nearest.
        padding_mode: 0=zeros, 1=border, 2=reflection.
        align_corners: If True, extrema (-1, 1) refer to corner pixel centers.
    """
    logger.debug("GEMS_KUNLUNXIN GRID_SAMPLER_3D")

    if input.ndim != 5:
        raise ValueError(
            f"grid_sampler_3d(): expected 5D input, but got {input.ndim}D input"
        )
    if grid.shape[-1] != 3:
        raise ValueError(
            f"grid_sampler_3d(): expected grid to have 3 in its last dimension, "
            f"but got {grid.shape[-1]}"
        )
    if grid.shape[0] != input.shape[0]:
        raise ValueError(
            f"grid_sampler_3d(): expected grid and input to have the same size of "
            f"the first dimension, but got grid.size(0) = {grid.shape[0]} and "
            f"input.size(0) = {input.shape[0]}"
        )
    if interpolation_mode not in (0, 1):
        raise ValueError(
            f"grid_sampler_3d(): expected interpolation_mode to be 0 or 1, "
            f"but got {interpolation_mode}"
        )
    if padding_mode not in (0, 1, 2):
        raise ValueError(
            f"grid_sampler_3d(): expected padding_mode to be 0, 1 or 2, "
            f"but got {padding_mode}"
        )

    N, C, ID, IH, IW = input.shape
    OD, OH, OW = grid.shape[1], grid.shape[2], grid.shape[3]
    output = torch.empty((N, C, OD, OH, OW), dtype=input.dtype, device=input.device)
    total = output.numel()
    if total == 0:
        # Host-side short-circuit: nothing to launch, never touch the device.
        return output

    with torch_device_fn.device(input.device):
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
            MODE=_ATEN_TO_KERNEL_MODE[interpolation_mode],
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
