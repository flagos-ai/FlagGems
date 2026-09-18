import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn

logger = logging.getLogger(__name__)
device = device.name


@triton.jit
def upsample_trilinear3d_kernel(
    ptr_o,
    ptr_i,
    scale_d,
    scale_h,
    scale_w,
    bias_d,
    bias_h,
    bias_w,
    OD: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    ID: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    SAME_D: tl.constexpr,
    SAME_H: tl.constexpr,
    SAME_W: tl.constexpr,
    BX: tl.constexpr,
    USE_INT32_IDX: tl.constexpr,
):
    row = tl.program_id(axis=0)
    if not USE_INT32_IDX:
        row = row.to(tl.int64)
    oh = row % OH
    od = (row // OH) % OD
    nc = row // (OH * OD)

    ow = tl.arange(0, BX)
    mask = ow < OW

    if SAME_D:
        src_d = od.to(tl.float32)
    else:
        src_d = od.to(tl.float32) * scale_d + bias_d
    if SAME_H:
        src_h = oh.to(tl.float32)
    else:
        src_h = oh.to(tl.float32) * scale_h + bias_h
    if SAME_W:
        src_w = ow.to(tl.float32)
    else:
        src_w = ow.to(tl.float32) * scale_w + bias_w

    src_d = tl.maximum(0.0, tl.minimum(src_d, ID - 1.0))
    src_h = tl.maximum(0.0, tl.minimum(src_h, IH - 1.0))
    src_w = tl.maximum(0.0, tl.minimum(src_w, IW - 1.0))

    id0 = tl.floor(src_d).to(tl.int32)
    ih0 = tl.floor(src_h).to(tl.int32)
    iw0 = tl.floor(src_w).to(tl.int32)

    id1 = tl.minimum(id0 + 1, ID - 1)
    ih1 = tl.minimum(ih0 + 1, IH - 1)
    iw1 = tl.minimum(iw0 + 1, IW - 1)

    td = src_d - id0.to(tl.float32)
    th = src_h - ih0.to(tl.float32)
    tw = src_w - iw0.to(tl.float32)

    wd0 = 1.0 - td
    wh0 = 1.0 - th
    ww0 = 1.0 - tw

    d_stride_in = IH * IW
    h_stride_in = IW
    spatial_in_stride = ID * IH * IW
    base = nc * spatial_in_stride

    b00 = base + id0 * d_stride_in + ih0 * h_stride_in
    b01 = base + id0 * d_stride_in + ih1 * h_stride_in
    b10 = base + id1 * d_stride_in + ih0 * h_stride_in
    b11 = base + id1 * d_stride_in + ih1 * h_stride_in

    x000 = tl.load(ptr_i + b00 + iw0).to(tl.float32)
    x001 = tl.load(ptr_i + b00 + iw1).to(tl.float32)
    x010 = tl.load(ptr_i + b01 + iw0).to(tl.float32)
    x011 = tl.load(ptr_i + b01 + iw1).to(tl.float32)
    x100 = tl.load(ptr_i + b10 + iw0).to(tl.float32)
    x101 = tl.load(ptr_i + b10 + iw1).to(tl.float32)
    x110 = tl.load(ptr_i + b11 + iw0).to(tl.float32)
    x111 = tl.load(ptr_i + b11 + iw1).to(tl.float32)

    c000 = x000 * ww0 + x001 * tw
    c001 = x010 * ww0 + x011 * tw
    front = c000 * wh0 + c001 * th

    c100 = x100 * ww0 + x101 * tw
    c101 = x110 * ww0 + x111 * tw
    back = c100 * wh0 + c101 * th

    out = front * wd0 + back * td

    tl.store(ptr_o + row * OW + ow, out, mask=mask)


def upsample_trilinear3d(
    self: torch.Tensor,
    output_size: Tuple[int, int, int],
    align_corners: bool,
    scales_d: Optional[float] = None,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN UPSAMPLE_TRILINEAR3D")
    assert self.device.type == device
    assert self.ndim == 5, f"Input must be 5D (NCDHW), got {self.ndim}D"

    N, C, ID, IH, IW = self.shape
    OD, OH, OW = output_size
    NC = N * C

    def calculate_scale_and_bias(in_sz, out_sz, scale):
        if align_corners:
            if out_sz > 1:
                scale_val = (in_sz - 1.0) / (out_sz - 1.0)
            else:
                scale_val = 0.0
            bias_val = 0.0
        else:
            if scale is not None:
                real_scale = 1.0 / scale
            else:
                real_scale = in_sz / out_sz
            scale_val = real_scale
            bias_val = 0.5 * real_scale - 0.5
        return scale_val, bias_val

    scale_d, bias_d = calculate_scale_and_bias(ID, OD, scales_d)
    scale_h, bias_h = calculate_scale_and_bias(IH, OH, scales_h)
    scale_w, bias_w = calculate_scale_and_bias(IW, OW, scales_w)

    inp = self.reshape(NC, ID, IH, IW).contiguous()
    out = torch.empty((N, C, OD, OH, OW), device=self.device, dtype=self.dtype)

    if out.numel() == 0:
        return out

    total_out = NC * OD * OH * OW
    BX = 1 << max(0, (OW - 1).bit_length())
    grid = (NC * OD * OH,)
    num_warps = min(8, max(1, BX // 64))

    with torch_device_fn.device(self.device):
        upsample_trilinear3d_kernel[grid](
            out,
            inp,
            scale_d,
            scale_h,
            scale_w,
            bias_d,
            bias_h,
            bias_w,
            OD=OD,
            OH=OH,
            OW=OW,
            ID=ID,
            IH=IH,
            IW=IW,
            SAME_D=(OD == ID),
            SAME_H=(OH == IH),
            SAME_W=(OW == IW),
            BX=BX,
            USE_INT32_IDX=(total_out + BX <= (2**31 - 1)),
            num_warps=num_warps,
        )

    return out
