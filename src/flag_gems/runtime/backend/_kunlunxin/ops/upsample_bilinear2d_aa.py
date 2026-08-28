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
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn

logger = logging.getLogger(__name__)
device = device.name


@triton.jit
def upsample_bilinear2d_aa_kernel(
    ptr_o,
    ptr_i,
    total,
    OH,
    OW,
    IH,
    IW,
    reciprocal_scale_h,
    reciprocal_scale_w,
    BLOCK_SIZE: tl.constexpr,
):
    # Flat 1D grid over the whole [N*C, OH, OW] output, so the store index is the
    # raw stride-1 `o` (contiguous block DMA) and every output pixel is an
    # independent lane. The generic kernel used a 2D (OW, OH) grid and walked
    # N*C with `for n: for c:` inside, issuing 4 * N * C gathers per program;
    # that serialisation wedges the NoC watchdog on the large test shapes.
    #
    # Antialiasing support stays 1.0 (2x2 window): aten only widens the window
    # when downsampling (aten scale = IH / OH >= 1), and this kernel is only
    # dispatched on the upsampling path.
    pid = tl.program_id(0)
    o = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = o < total

    nc_oh = o // OW
    ow = o - nc_oh * OW
    nc = nc_oh // OH
    oh = nc_oh - nc * OH

    support_h = 1.0
    support_w = 1.0

    center_h = (oh.to(tl.float32) + 0.5) * reciprocal_scale_h
    center_w = (ow.to(tl.float32) + 0.5) * reciprocal_scale_w

    span_start_h = tl.maximum(center_h - support_h + 0.5, 0.0).to(tl.int32)
    span_start_w = tl.maximum(center_w - support_w + 0.5, 0.0).to(tl.int32)
    span_size_h = (tl.minimum(center_h + support_h + 0.5, IH * 1.0) - span_start_h).to(
        tl.int32
    )
    span_size_w = (tl.minimum(center_w + support_w + 0.5, IW * 1.0) - span_start_w).to(
        tl.int32
    )

    start_minus_center_h = span_start_h.to(tl.float32) - center_h
    start_minus_center_w = span_start_w.to(tl.float32) - center_w

    wy0 = tl.where(
        0 < span_size_h, tl.maximum(1.0 - tl.abs(start_minus_center_h + 0.5), 0.0), 0.0
    )
    wy1 = tl.where(
        1 < span_size_h, tl.maximum(1.0 - tl.abs(start_minus_center_h + 1.5), 0.0), 0.0
    )
    wy_total = wy0 + wy1
    wy_total = tl.where(wy_total != 0.0, wy_total, 1.0)
    wy0 = wy0 / wy_total
    wy1 = wy1 / wy_total

    wx0 = tl.where(
        0 < span_size_w, tl.maximum(1.0 - tl.abs(start_minus_center_w + 0.5), 0.0), 0.0
    )
    wx1 = tl.where(
        1 < span_size_w, tl.maximum(1.0 - tl.abs(start_minus_center_w + 1.5), 0.0), 0.0
    )
    wx_total = wx0 + wx1
    wx_total = tl.where(wx_total != 0.0, wx_total, 1.0)
    wx0 = wx0 / wx_total
    wx1 = wx1 / wx_total

    # Out-of-range neighbours carry weight 0 already (span_size guards), so the
    # index is clamped instead of masked: a clamped read is always in bounds and
    # its contribution is multiplied by zero.
    h0 = tl.minimum(span_start_h, IH - 1)
    h1 = tl.minimum(span_start_h + 1, IH - 1)
    w0 = tl.minimum(span_start_w, IW - 1)
    w1 = tl.minimum(span_start_w + 1, IW - 1)

    row0 = (nc * IH + h0) * IW
    row1 = (nc * IH + h1) * IW
    x00 = tl.load(ptr_i + row0 + w0, mask=mask, other=0.0).to(tl.float32)
    x01 = tl.load(ptr_i + row0 + w1, mask=mask, other=0.0).to(tl.float32)
    x10 = tl.load(ptr_i + row1 + w0, mask=mask, other=0.0).to(tl.float32)
    x11 = tl.load(ptr_i + row1 + w1, mask=mask, other=0.0).to(tl.float32)

    top = x00 * wx0 + x01 * wx1
    bot = x10 * wx0 + x11 * wx1
    result = top * wy0 + bot * wy1

    tl.store(ptr_o + o, result.to(ptr_o.dtype.element_ty), mask=mask)


def bilinear_reciprocal_scale(src_size, dst_size, align_corners, scale):
    if align_corners:
        if dst_size > 1:
            return (src_size - 1) / (dst_size - 1)
        return 0.0
    if scale is not None and scale > 0:
        return 1.0 / scale
    return src_size / dst_size


def _upsample_bilinear2d_aa(
    input: torch.Tensor,
    output_size: Tuple[int],
    align_corners: bool = False,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN UPSAMPLE_BILINEAR2D_AA")
    assert input.device.type == device
    assert input.ndim == 4, "The ndim of input must be 4"
    assert len(output_size) == 2, "The len of output_size must be 2"

    OH, OW = output_size
    N, C, IH, IW = input.shape

    reciprocal_scale_h = bilinear_reciprocal_scale(IH, OH, align_corners, scales_h)
    reciprocal_scale_w = bilinear_reciprocal_scale(IW, OW, align_corners, scales_w)

    inp = input.contiguous()
    output = torch.empty((N, C, OH, OW), device=input.device, dtype=input.dtype)

    total = N * C * OH * OW
    BLOCK_SIZE = 4096
    grid = (triton.cdiv(total, BLOCK_SIZE),)

    with torch_device_fn.device(input.device):
        upsample_bilinear2d_aa_kernel[grid](
            output,
            inp,
            total,
            OH,
            OW,
            IH,
            IW,
            reciprocal_scale_h,
            reciprocal_scale_w,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    return output
