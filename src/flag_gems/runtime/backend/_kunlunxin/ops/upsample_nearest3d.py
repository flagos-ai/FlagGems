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


# NOTE (kunlunxin/XPU): flat 1D grid over ALL output elements (decode nc/od/oh/ow
# from the flat index, no per-plane loop) exposes full program-level parallelism.
# XPU perf pass 2026-08-21 (XPU 4, following trilinear3d XPU7 / nearest2d XPU7
# precedent): geometry (OD/OH/OW/ID/IH/IW) is tl.constexpr so the per-lane
# div/mod chain (ow=%OW, oh=(//OW)%OH, od=//(OW*OH), nc=//(OD*OH*OW)) is
# strength-reduced to constant arithmetic instead of generic vector integer
# divides; nc is clamped to NC-1 and the nearest indices are clamped to the
# source extent before use so every load is in-bounds for ANY decoded lane and
# the loads drop the mask (masked-memory path is penalized on XPU, see upsample
# family records); the tail store is guarded by a NEED_MASK constexpr only when
# total_out does not divide BLOCK_SIZE. The residual gap to torch is the XPU
# discrete-gather wall (1 data-dependent gather per output lane).
@triton.jit
def upsample_nearest3d_kernel(
    ptr_o,
    ptr_i,
    NC,
    total_out,
    reciprocal_scale_d,
    reciprocal_scale_h,
    reciprocal_scale_w,
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
    # (unmasked) nearest load in-bounds -- the store is masked instead.
    nc = tl.minimum(idx // total_spatial, NC - 1)

    if SAME_D:
        id = od
    else:
        id = tl.minimum(
            tl.math.floor(od.to(tl.float32) * reciprocal_scale_d).to(tl.int32),
            ID - 1,
        )
    if SAME_H:
        ih = oh
    else:
        ih = tl.minimum(
            tl.math.floor(oh.to(tl.float32) * reciprocal_scale_h).to(tl.int32),
            IH - 1,
        )
    if SAME_W:
        iw = ow
    else:
        iw = tl.minimum(
            tl.math.floor(ow.to(tl.float32) * reciprocal_scale_w).to(tl.int32),
            IW - 1,
        )

    d_stride_in = IH * IW
    h_stride_in = IW
    spatial_in_stride = ID * IH * IW
    base = nc * spatial_in_stride
    input_offset = base + id * d_stride_in + ih * h_stride_in + iw

    data = tl.load(ptr_i + input_offset)
    if NEED_MASK:
        tl.store(ptr_o + idx, data, mask=idx < total_out)
    else:
        tl.store(ptr_o + idx, data)


def upsample_nearest3d(
    input: torch.Tensor,
    output_size: Tuple[int, int, int],
    scales_d: Optional[float] = None,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
):
    logger.debug("GEMS_KUNLUNXIN UPSAMPLE_NEAREST3D")
    assert input.device.type == device
    assert input.ndim == 5, "The ndim of input must be 5"

    OD, OH, OW = output_size
    N, C, ID, IH, IW = input.shape
    NC = N * C

    def calculate_scale(in_sz, out_sz, s):
        if s is not None:
            return float(torch.tensor(1.0 / s, dtype=torch.float32).item())
        return float(
            (
                torch.tensor(in_sz, dtype=torch.float32)
                / torch.tensor(out_sz, dtype=torch.float32)
            ).item()
        )

    reciprocal_scale_d = calculate_scale(ID, OD, scales_d)
    reciprocal_scale_h = calculate_scale(IH, OH, scales_h)
    reciprocal_scale_w = calculate_scale(IW, OW, scales_w)

    output = torch.empty((N, C, OD, OH, OW), device=input.device, dtype=input.dtype)
    if output.numel() == 0:
        return output

    # Exact 2x fast path (XPU 2026-08-21): when every output dimension is
    # exactly twice the input dimension, out[2*d+od, 2*h+oh, 2*w+ow] == in[d,h,w]
    # for the 8 parity classes (od, oh, ow) in {0,1}^3, identical to the
    # floor((o * in/out)) nearest-neighbour mapping.  The 8 copies are plain
    # strided views, so torch.ops.aten._copy_from (never overridden by gems)
    # dispatches straight to the native strided-copy engine -- far cheaper than
    # one discrete-gather Triton lane per output voxel (see upsample_nearest1d
    # vendor fast path; 3D keeps the whole ~50MB in the copy engine).
    if OD == 2 * ID and OH == 2 * IH and OW == 2 * IW:
        for od in (0, 1):
            for oh in (0, 1):
                for ow in (0, 1):
                    dst = output[:, :, od::2, oh::2, ow::2]
                    torch.ops.aten._copy_from(input, dst, False)
        return output

    total_out = NC * OD * OH * OW
    BLOCK_SIZE = 1024
    need_mask = total_out % BLOCK_SIZE != 0
    grid = (triton.cdiv(total_out, BLOCK_SIZE),)

    with torch_device_fn.device(input.device):
        upsample_nearest3d_kernel[grid](
            output,
            input,
            NC,
            total_out,
            reciprocal_scale_d,
            reciprocal_scale_h,
            reciprocal_scale_w,
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
            num_warps=4,
        )
    return output