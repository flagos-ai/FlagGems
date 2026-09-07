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
import math
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn

logger = logging.getLogger(__name__)
device = device.name


# NOTE (kunlunxin/XPU): flat 1D grid over ALL output elements (decode nc/od/oh/ow
# from the flat index, no per-plane loop) exposes full program-level parallelism.
# Follows the vendor upsample_nearest3d precedent: geometry (OD/OH/OW/ID/IH/IW)
# is tl.constexpr so the per-lane div/mod chain is strength-reduced to constant
# arithmetic; nc is clamped to NC-1 and the nearest-exact source indices are
# clamped to the source extent before use so every load is in-bounds for ANY
# decoded lane and the loads drop the mask (masked-memory path is penalized on
# XPU); the tail store is guarded by a NEED_MASK constexpr only when total_out
# does not divide BLOCK_SIZE.  The nearest-exact source mapping is
#   src = min(floor((dst + 0.5) / scale), in - 1)
# where scale = out / in (reciprocal_scale = in / out, or 1 / scale when the
# scale factor was given directly).
@triton.jit
def _upsample_nearest_exact3d_kernel(
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
            tl.math.floor((od.to(tl.float32) + 0.5) * reciprocal_scale_d).to(
                tl.int32
            ),
            ID - 1,
        )
    if SAME_H:
        ih = oh
    else:
        ih = tl.minimum(
            tl.math.floor((oh.to(tl.float32) + 0.5) * reciprocal_scale_h).to(
                tl.int32
            ),
            IH - 1,
        )
    if SAME_W:
        iw = ow
    else:
        iw = tl.minimum(
            tl.math.floor((ow.to(tl.float32) + 0.5) * reciprocal_scale_w).to(
                tl.int32
            ),
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


def _upsample_nearest_exact3d(
    input: torch.Tensor,
    output_size: Optional[Tuple[int, int, int]] = None,
    scales_d: Optional[float] = None,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
):
    logger.debug("GEMS_KUNLUNXIN _UPSAMPLE_NEAREST_EXACT3D")
    assert input.device.type == device
    assert input.ndim == 5, "The ndim of input must be 5"

    input = input.contiguous()
    N, C, ID, IH, IW = input.shape

    if output_size is not None:
        OD, OH, OW = output_size
        OD, OH, OW = int(OD), int(OH), int(OW)
    else:
        # output_size omitted: derive from the scale factors, matching the
        # aten convention OD = floor(ID * scale_d).
        OD = int(math.floor(ID * (scales_d if scales_d is not None else 1.0)))
        OH = int(math.floor(IH * (scales_h if scales_h is not None else 1.0)))
        OW = int(math.floor(IW * (scales_w if scales_w is not None else 1.0)))

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

    output = torch.empty(
        (N, C, OD, OH, OW), device=input.device, dtype=input.dtype
    )
    if output.numel() == 0:
        return output

    # NOTE (kunlunxin/XPU, 2026-09-05): the exact-2x "8 strided _copy_from"
    # fast path used by vendor upsample_nearest3d was measured on this op and
    # is *slower* than the flat gather kernel for these shapes (8 launches,
    # ~0.29-0.39ms total on card 6, vs 0.14-0.05ms kernel); dropped in favor
    # of the single-kernel path below.  The W-pair / W-quad 2x variants (2 or
    # 4 stores per lane) were also measured and are slower (0.5-0.9ms) -- the
    # 2nd/8th discrete stores are penalized on XPU, see upsample family notes.

    total_out = N * C * OD * OH * OW
    BLOCK_SIZE = 1024
    need_mask = total_out % BLOCK_SIZE != 0
    grid = (triton.cdiv(total_out, BLOCK_SIZE),)

    with torch_device_fn.device(input.device):
        _upsample_nearest_exact3d_kernel[grid](
            output,
            input,
            N * C,
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