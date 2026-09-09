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
def upsample_bilinear2d_kernel(
    ptr_o,
    ptr_i,
    total,
    OH,
    OW,
    IH,
    IW,
    scale_h,
    bias_h,
    scale_w,
    bias_w,
    BLOCK_SIZE: tl.constexpr,
):
    # Flat 1D grid over the whole [N*C, OH, OW] output. The store index is the
    # raw `o = pid * BLOCK + arange`, provably stride-1, so the store lowers to a
    # contiguous block DMA (same reasoning as upsample_linear1d on this backend).
    #
    # The generic kernel instead used a 2D (OW, OH) grid and walked N*C inside
    # the kernel with `for n: for c:`. Each program then issued 4 * N * C gather
    # loads over the whole input - up to 24576 planes for the (128, 192, 42, 51)
    # test shape - which drives the NoC into `wait for noc idle timeout` and
    # wedges the card (the whole 120-case marker died on the 1220s watchdog).
    pid = tl.program_id(0)
    o = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = o < total

    nc_oh = o // OW
    ow = o - nc_oh * OW
    nc = nc_oh // OH
    oh = nc_oh - nc * OH

    # aten maps the destination pixel centre back onto the source grid. Clamping
    # the source position to [0, I-1] is equivalent to aten's
    # `offset = (idx < I - 1) ? 1 : 0`: once the position saturates both
    # neighbours coincide and the two weights sum to 1.
    src_h = tl.maximum(0.0, tl.minimum(oh.to(tl.float32) * scale_h + bias_h, IH - 1.0))
    src_w = tl.maximum(0.0, tl.minimum(ow.to(tl.float32) * scale_w + bias_w, IW - 1.0))

    # src is non-negative here, so int truncation equals floor.
    h0 = src_h.to(tl.int32)
    w0 = src_w.to(tl.int32)
    h1 = tl.minimum(h0 + 1, IH - 1)
    w1 = tl.minimum(w0 + 1, IW - 1)

    th = src_h - h0.to(tl.float32)
    tw = src_w - w0.to(tl.float32)

    row0 = (nc * IH + h0) * IW
    row1 = (nc * IH + h1) * IW
    x00 = tl.load(ptr_i + row0 + w0, mask=mask, other=0.0).to(tl.float32)
    x01 = tl.load(ptr_i + row0 + w1, mask=mask, other=0.0).to(tl.float32)
    x10 = tl.load(ptr_i + row1 + w0, mask=mask, other=0.0).to(tl.float32)
    x11 = tl.load(ptr_i + row1 + w1, mask=mask, other=0.0).to(tl.float32)

    top = x00 * (1.0 - tw) + x01 * tw
    bot = x10 * (1.0 - tw) + x11 * tw
    result = top * (1.0 - th) + bot * th

    tl.store(ptr_o + o, result.to(ptr_o.dtype.element_ty), mask=mask)


def bilinear_scale_bias(src_size, dst_size, align_corners, scale):
    """Return (scale, bias) so that src_pos = dst_index * scale + bias."""
    if align_corners:
        if dst_size > 1:
            return (src_size - 1) / (dst_size - 1), 0.0
        return 0.0, 0.0
    if scale is not None and scale > 0:
        reciprocal_scale = 1.0 / scale
    else:
        reciprocal_scale = src_size / dst_size
    return reciprocal_scale, 0.5 * reciprocal_scale - 0.5


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

    scale_h, bias_h = bilinear_scale_bias(IH, OH, align_corners, scales_h)
    scale_w, bias_w = bilinear_scale_bias(IW, OW, align_corners, scales_w)

    inp = input.contiguous()
    output = torch.empty((N, C, OH, OW), device=input.device, dtype=input.dtype)

    total = N * C * OH * OW
    BLOCK_SIZE = 4096
    grid = (triton.cdiv(total, BLOCK_SIZE),)

    with torch_device_fn.device(input.device):
        upsample_bilinear2d_kernel[grid](
            output,
            inp,
            total,
            OH,
            OW,
            IH,
            IW,
            scale_h,
            bias_h,
            scale_w,
            bias_w,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    return output
