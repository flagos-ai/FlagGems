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
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)
device = device.name


@triton.jit
def upsample_bilinear2d_kernel(
    ptr_o,
    ptr_i,
    OH,
    OW,
    IH,
    IW,
    reciprocal_scale_h,
    reciprocal_scale_w,
    N: tl.constexpr,
    C: tl.constexpr,
    ALIGN_CORNERS: tl.constexpr,
    BX: tl.constexpr,
):
    # One program per output row: pid = n*C*OH + oh. All per-lane index math is
    # confined to the column axis (ow = arange(BX)), so N/C/OH/OW only ever
    # appear in SCALAR (uniform) div/mod, hoisted out of the lane loop. This
    # matters on KunlunXin: per-lane integer division/modulo is emulated and
    # costs ~2.3ms per 3.1M lanes (measured), i.e. 50x the flat copy kernel;
    # the flat 1D-grid variant that derives (n, c, oh, ow) via per-lane div/mod
    # was ~2x slower than this row-grid form on the benchmark shapes.
    #
    # The four input reads are data-dependent gathers (w0 = f(ow) with
    # min/max/floor clamps). The XPU Triton backend lowers any non-affine load
    # to a discrete/scalarized access (no warp coalescing observed: even a
    # perfectly coalescable `(o>>1)<<1` index pattern costs ~10x an affine
    # stride-1 load). Measured ceiling: ~10G 4B-transactions/s, which caps any
    # bilinear triton kernel at ~1.5ms for (1,3,512,512)->(1024,1024) versus
    # 0.037ms for the native C++ kernel (see harness README for the full
    # evidence chain). This kernel is retained because it is 20-90x faster than
    # the generic 2D-tile implementation (serial N*C loop) and is numerically
    # exact; it does not meet the 0.8x dtype-equal Gems Speedup gate.
    row = ext.program_id(axis=0)
    oh = row % OH
    nc = row // OH
    c = nc % C
    n = nc // C

    ow = tl.arange(0, BX)
    mask = ow < OW

    # Compute the source coordinates (ATen area_pixel_compute_source_index).
    if ALIGN_CORNERS:
        # When align_corners is True, map corners to corners:
        # real = oh * (IH - 1) / (OH - 1), 0 when OH == 1.
        real_h = tl.where(
            OH > 1,
            oh.to(tl.float32) * (IH - 1) / (OH - 1),
            0.0,
        )
        real_w = tl.where(
            OW > 1,
            ow.to(tl.float32) * (IW - 1) / (OW - 1),
            0.0,
        )
    else:
        # When align_corners is False: real = (oh + 0.5) * scale - 0.5.
        real_h = (oh.to(tl.float32) + 0.5) * reciprocal_scale_h - 0.5
        real_w = (ow.to(tl.float32) + 0.5) * reciprocal_scale_w - 0.5

    # Clamp to valid range.
    real_h = tl.maximum(real_h, 0.0)
    real_w = tl.maximum(real_w, 0.0)

    # Top-left corner of the 2x2 region.
    h0 = tl.minimum(real_h.to(tl.int32), IH - 1)
    w0 = tl.minimum(real_w.to(tl.int32), IW - 1)
    h1 = tl.minimum(h0 + 1, IH - 1)
    w1 = tl.minimum(w0 + 1, IW - 1)

    # Interpolation weights, clamped to [0, 1].
    h_weight = real_h - h0.to(tl.float32)
    w_weight = real_w - w0.to(tl.float32)
    h_weight = tl.maximum(tl.minimum(h_weight, 1.0), 0.0)
    w_weight = tl.maximum(tl.minimum(w_weight, 1.0), 0.0)

    base = (n * C + c) * (IH * IW)
    off_00 = base + h0 * IW + w0
    off_01 = base + h0 * IW + w1
    off_10 = base + h1 * IW + w0
    off_11 = base + h1 * IW + w1

    data_00 = tl.load(ptr_i + off_00, mask=mask, other=0.0)
    data_01 = tl.load(ptr_i + off_01, mask=mask, other=0.0)
    data_10 = tl.load(ptr_i + off_10, mask=mask, other=0.0)
    data_11 = tl.load(ptr_i + off_11, mask=mask, other=0.0)

    w00 = (1.0 - h_weight) * (1.0 - w_weight)
    w01 = (1.0 - h_weight) * w_weight
    w10 = h_weight * (1.0 - w_weight)
    w11 = h_weight * w_weight

    result = (
        data_00.to(tl.float32) * w00
        + data_01.to(tl.float32) * w01
        + data_10.to(tl.float32) * w10
        + data_11.to(tl.float32) * w11
    )
    result = result.to(data_00.dtype)

    # Store index row*OW + ow is affine/stride-1 in the lanes -> block DMA.
    tl.store(ptr_o + row * OW + ow, result, mask=mask)


def _bilinear_reciprocal_scale(src_size, dst_size, align_corners, scale):
    if align_corners:
        if dst_size > 1:
            return (src_size - 1) / (dst_size - 1)
        else:
            return 0.0
    else:
        if scale is not None and scale > 0:
            return 1.0 / scale
        else:
            return src_size / dst_size


def upsample_bilinear2d(
    input: torch.Tensor,
    output_size: Tuple[int],
    align_corners: bool = False,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN UPSAMPLE_BILINEAR2D")
    assert input.device.type == device
    assert input.ndim == 4, "The ndim of input must be 4"
    assert len(output_size) == 2, "The len of output_size must be 2"

    OH, OW = output_size
    N, C, IH, IW = input.shape

    reciprocal_scale_h = _bilinear_reciprocal_scale(IH, OH, align_corners, scales_h)
    reciprocal_scale_w = _bilinear_reciprocal_scale(IW, OW, align_corners, scales_w)

    output = torch.empty((N, C, OH, OW), device=input.device, dtype=input.dtype)
    # Row-grid: one program per output row; BX covers the full row (padded to a
    # power of two, masked tail).
    block_size = triton.next_power_of_2(OW) if OW > 0 else 1
    grid = (N * C * OH,)

    with torch_device_fn.device(input.device):
        upsample_bilinear2d_kernel[grid](
            output,
            input,
            OH,
            OW,
            IH,
            IW,
            reciprocal_scale_h,
            reciprocal_scale_w,
            N,
            C,
            align_corners,
            BX=block_size,
        )
    return output