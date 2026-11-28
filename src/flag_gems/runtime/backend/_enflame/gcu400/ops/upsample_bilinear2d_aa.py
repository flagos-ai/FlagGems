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

"""GCU400-specific upsample_bilinear2d_aa implementation.

The common implementation uses a 2D (cdiv(OW, BLOCK_X), cdiv(OH, BLOCK_Y)) grid; with large OH
grid.y exceeds GCU400's 255 hardware limit and raises OutOfResources. Here the 2D tiles are
flattened into a 1D task queue, with a fixed program count plus an in-kernel stride loop over all output tiles,
removing the grid dimension limit and decoupling program count from output size.
"""

import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn

logger = logging.getLogger(__name__)

GCU_NUM_PROGRAMS = 48
BLOCK_X = 256
BLOCK_Y = 8


@triton.jit(
    do_not_specialize=[
        "N",
        "C",
        "OH",
        "OW",
        "IH",
        "IW",
        "reciprocal_scale_h",
        "reciprocal_scale_w",
        "num_tiles_x",
        "num_tiles",
    ]
)
def _upsample_bilinear2d_aa_kernel(
    ptr_o,
    ptr_i,
    N,
    C,
    OH,
    OW,
    IH,
    IW,
    reciprocal_scale_h,
    reciprocal_scale_w,
    num_tiles_x,
    num_tiles,
    NUM_PROGRAMS: tl.constexpr,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    pid = tl.program_id(0)
    range_x = tl.arange(0, BLOCK_SIZE_X)
    range_y = tl.arange(0, BLOCK_SIZE_Y)

    # Each program strides over several output tiles; tile count does not grow with output size.
    for tile_id in tl.range(pid, num_tiles, NUM_PROGRAMS):
        tile_y = tile_id // num_tiles_x
        tile_x = tile_id % num_tiles_x
        oh = tile_y * BLOCK_SIZE_Y + range_y
        ow = tile_x * BLOCK_SIZE_X + range_x
        out_mask = (oh[:, None] < OH) & (ow[None, :] < OW)

        # Antialiased bilinear: support window 2x2, linear interpolation along x first then y.
        center_w = (ow + 0.5) * reciprocal_scale_w
        center_h = (oh + 0.5) * reciprocal_scale_h
        span_start_w = tl.maximum(center_w - 0.5, 0).to(tl.int32)
        span_start_h = tl.maximum(center_h - 0.5, 0).to(tl.int32)
        span_size_w = (tl.minimum(center_w + 1.5, IW) - span_start_w).to(tl.int32)
        span_size_h = (tl.minimum(center_h + 1.5, IH) - span_start_h).to(tl.int32)
        start_minus_center_w = span_start_w - center_w
        start_minus_center_h = span_start_h - center_h

        wy0 = 1.0 - tl.abs(start_minus_center_h + 0.5)
        wy1 = 1.0 - tl.abs(start_minus_center_h + 1.5)
        weight_y0 = tl.where(0 < span_size_h, tl.maximum(wy0, 0), 0)
        weight_y1 = tl.where(1 < span_size_h, tl.maximum(wy1, 0), 0)
        weight_y_total = weight_y0 + weight_y1
        weight_y_total = tl.where(weight_y_total != 0, weight_y_total, 1)
        weight_y0 = weight_y0 / weight_y_total
        weight_y1 = weight_y1 / weight_y_total

        wx0 = 1.0 - tl.abs(start_minus_center_w + 0.5)
        wx1 = 1.0 - tl.abs(start_minus_center_w + 1.5)
        weight_x0 = tl.where(0 < span_size_w, tl.maximum(wx0, 0), 0)
        weight_x1 = tl.where(1 < span_size_w, tl.maximum(wx1, 0), 0)
        weight_x_total = weight_x0 + weight_x1
        weight_x_total = tl.where(weight_x_total != 0, weight_x_total, 1)
        weight_x0 = weight_x0 / weight_x_total
        weight_x1 = weight_x1 / weight_x_total

        mask_y0 = span_start_h[:, None] + 0 < IH
        mask_y1 = span_start_h[:, None] + 1 < IH
        mask_x0 = span_start_w[None, :] + 0 < IW
        mask_x1 = span_start_w[None, :] + 1 < IW

        # 2D traversal of (n, c) and output tiles equals 1D plane traversal; merging loops reduces nesting and
        # multiplication.
        for plane in range(0, N * C, 1):
            offset_base = (plane * IH + span_start_h[:, None]) * IW + span_start_w[
                None, :
            ]

            data00 = tl.load(
                ptr_i + (offset_base + 0 * IW + 0),
                mask=(mask_y0 & mask_x0),
                other=0,
            )
            data01 = tl.load(
                ptr_i + (offset_base + 0 * IW + 1),
                mask=(mask_y0 & mask_x1),
                other=0,
            )
            data10 = tl.load(
                ptr_i + (offset_base + 1 * IW + 0),
                mask=(mask_y1 & mask_x0),
                other=0,
            )
            data11 = tl.load(
                ptr_i + (offset_base + 1 * IW + 1),
                mask=(mask_y1 & mask_x1),
                other=0,
            )

            row0 = data00 * weight_x0[None, :] + data01 * weight_x1[None, :]
            row1 = data10 * weight_x0[None, :] + data11 * weight_x1[None, :]
            result = row0 * weight_y0[:, None] + row1 * weight_y1[:, None]

            offset_o = (plane * OH + oh[:, None]) * OW + ow[None, :]
            tl.store(ptr_o + offset_o, result, mask=out_mask)


def bilinear_reciprocal_scale(src_size, dst_size, align_corners, scale):
    if align_corners:
        if dst_size > 1:
            return (src_size - 1) / (dst_size - 1)
        else:
            return 0
    else:
        if scale is not None and scale > 0:
            return 1.0 / scale
        else:
            return src_size / dst_size


def _upsample_bilinear2d_aa(
    input: torch.Tensor,
    output_size: Tuple[int],
    align_corners: bool = False,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
):
    logger.debug("GEMS_ENFLAME UPSAMPLE_BILINEAR2D_AA")
    assert input.device.type == device.name
    assert input.ndim == 4, "The ndim of input must be 4"
    assert len(output_size) == 2, "The len of output_size must be 2"

    OH, OW = output_size
    N, C, IH, IW = input.shape

    reciprocal_scale_h = bilinear_reciprocal_scale(IH, OH, align_corners, scales_h)
    reciprocal_scale_w = bilinear_reciprocal_scale(IW, OW, align_corners, scales_w)

    output = torch.empty((N, C, OH, OW), device=input.device, dtype=input.dtype)
    num_tiles_x = triton.cdiv(OW, BLOCK_X)
    num_tiles_y = triton.cdiv(OH, BLOCK_Y)
    num_tiles = num_tiles_x * num_tiles_y
    num_programs = min(GCU_NUM_PROGRAMS, num_tiles)

    with torch_device_fn.device(input.device):
        _upsample_bilinear2d_aa_kernel[(num_programs,)](
            output,
            input,
            N,
            C,
            OH,
            OW,
            IH,
            IW,
            reciprocal_scale_h,
            reciprocal_scale_w,
            num_tiles_x,
            num_tiles,
            NUM_PROGRAMS=num_programs,
            BLOCK_SIZE_X=BLOCK_X,
            BLOCK_SIZE_Y=BLOCK_Y,
            num_warps=4,
        )
    return output
