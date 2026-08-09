import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn

logger = logging.getLogger(__name__)
device = device.name


# NOTE (kunlunxin/XPU): flat 1D grid over ALL output elements (decode nc from
# the flat index, no inner loop) exposes full program-level parallelism.
# XPU perf pass 2026-08-17 (XPU 7): geometry (OD/OH/OW/ID/IH/IW) is
# tl.constexpr so the per-lane div/mod chain (ow=%OW, oh=(//OW)%OH, od=//(OW*OH),
# nc=//(OD*OH*OW)) strength-reduces to constant arithmetic instead of generic
# vector integer divides; the 8 corner indices are clamped before use, which
# makes every interpolation load in-bounds for ANY decoded lane, so the loads
# drop the mask (masked-memory path is penalized on XPU, see upsample family
# rpm: nearest2d/bilinear2d_aa use the same clamp + unmasked pattern); the
# tail store is guarded by a NEED_MASK constexpr only when total_out does not
# divide BLOCK_SIZE. The residual gap to torch is the XPU discrete-gather wall
# (8 data-dependent neighbour loads), same structural ceiling as
# grid_sample / reflection_pad2d; torch runs a fused vendor kernel.
@triton.jit
def upsample_trilinear3d_kernel(
    ptr_o,
    ptr_i,
    NC,
    total_out,
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
    BLOCK_SIZE: tl.constexpr,
    USE_INT32_IDX: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    if USE_INT32_IDX:
        pid = tl.program_id(axis=0)
    else:
        pid = tl.program_id(axis=0).to(tl.int64)

    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    total_spatial = OD * OH * OW
    sp = idx % total_spatial
    ow = sp % OW
    oh = (sp // OW) % OH
    od = sp // (OW * OH)
    # Tail lanes beyond total_out may decode nc >= NC; clamping keeps the
    # (unmasked) corner loads in-bounds -- the store is masked instead.
    nc = tl.minimum(idx // total_spatial, NC - 1)

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

    o000 = base + id0 * d_stride_in + ih0 * h_stride_in + iw0
    o001 = base + id0 * d_stride_in + ih0 * h_stride_in + iw1
    o010 = base + id0 * d_stride_in + ih1 * h_stride_in + iw0
    o011 = base + id0 * d_stride_in + ih1 * h_stride_in + iw1
    o100 = base + id1 * d_stride_in + ih0 * h_stride_in + iw0
    o101 = base + id1 * d_stride_in + ih0 * h_stride_in + iw1
    o110 = base + id1 * d_stride_in + ih1 * h_stride_in + iw0
    o111 = base + id1 * d_stride_in + ih1 * h_stride_in + iw1

    x000 = tl.load(ptr_i + o000).to(tl.float32)
    x001 = tl.load(ptr_i + o001).to(tl.float32)
    x010 = tl.load(ptr_i + o010).to(tl.float32)
    x011 = tl.load(ptr_i + o011).to(tl.float32)
    x100 = tl.load(ptr_i + o100).to(tl.float32)
    x101 = tl.load(ptr_i + o101).to(tl.float32)
    x110 = tl.load(ptr_i + o110).to(tl.float32)
    x111 = tl.load(ptr_i + o111).to(tl.float32)

    c000 = x000 * ww0 + x001 * tw
    c001 = x010 * ww0 + x011 * tw
    front = c000 * wh0 + c001 * th

    c100 = x100 * ww0 + x101 * tw
    c101 = x110 * ww0 + x111 * tw
    back = c100 * wh0 + c101 * th

    out = front * wd0 + back * td

    if NEED_MASK:
        tl.store(ptr_o + idx, out, mask=idx < total_out)
    else:
        tl.store(ptr_o + idx, out)


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
    BLOCK_SIZE = 4096
    need_mask = total_out % BLOCK_SIZE != 0
    grid = (triton.cdiv(total_out, BLOCK_SIZE),)

    with torch_device_fn.device(self.device):
        upsample_trilinear3d_kernel[grid](
            out,
            inp,
            NC,
            total_out,
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
            BLOCK_SIZE=BLOCK_SIZE,
            USE_INT32_IDX=(total_out + BLOCK_SIZE <= (2**31 - 1)),
            NEED_MASK=need_mask,
            num_warps=8,
        )

    return out