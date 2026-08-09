# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn


def _calculate_scale(in_size: int, out_size: int, scale: Optional[float]) -> float:
    if scale is not None:
        return float(torch.tensor(1.0 / scale, dtype=torch.float32).item())
    return float(
        (
            torch.tensor(in_size, dtype=torch.float32)
            / torch.tensor(out_size, dtype=torch.float32)
        ).item()
    )


@triton.heuristics({"BLOCK_SIZE": lambda args: 2048})
@triton.jit
def upsample_nearest3d_kernel(
    ptr_o,
    ptr_i,
    NC,
    OD,
    OH,
    OW,
    ID,
    IH,
    IW,
    reciprocal_scale_d,
    reciprocal_scale_h,
    reciprocal_scale_w,
    total_out,
    BLOCK_SIZE: tl.constexpr,
    SAME_D: tl.constexpr,
    SAME_H: tl.constexpr,
    SAME_W: tl.constexpr,
    USE_INT32_IDX: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if not USE_INT32_IDX:
        pid = pid.to(tl.int64)

    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < OD * OH * OW
    nc = tl.program_id(axis=1)

    ow = idx % OW
    oh = (idx // OW) % OH
    od = idx // (OH * OW)

    if SAME_D:
        id = od
    else:
        id = tl.minimum(
            tl.math.floor(od.to(tl.float32) * reciprocal_scale_d).to(tl.int32), ID - 1
        )
    if SAME_H:
        ih = oh
    else:
        ih = tl.minimum(
            tl.math.floor(oh.to(tl.float32) * reciprocal_scale_h).to(tl.int32), IH - 1
        )
    if SAME_W:
        iw = ow
    else:
        iw = tl.minimum(
            tl.math.floor(ow.to(tl.float32) * reciprocal_scale_w).to(tl.int32), IW - 1
        )

    input_spatial = ID * IH * IW
    output_spatial = OD * OH * OW
    input_offset = nc * input_spatial + id * IH * IW + ih * IW + iw
    output_offset = nc * output_spatial + idx
    value = tl.load(ptr_i + input_offset, mask=mask)
    tl.store(ptr_o + output_offset, value, mask=mask)


def upsample_nearest3d(
    input: torch.Tensor,
    output_size: Tuple[int, int, int],
    scales_d: Optional[float] = None,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
):
    assert input.device.type == device.name
    assert input.ndim == 5, "The ndim of input must be 5"

    OD, OH, OW = output_size
    N, C, ID, IH, IW = input.shape
    reciprocal_scale_d = _calculate_scale(ID, OD, scales_d)
    reciprocal_scale_h = _calculate_scale(IH, OH, scales_h)
    reciprocal_scale_w = _calculate_scale(IW, OW, scales_w)
    output = torch.empty((N, C, OD, OH, OW), device=input.device, dtype=input.dtype)
    total_out = N * C * OD * OH * OW
    spatial = OD * OH * OW
    grid = lambda meta: (
        triton.cdiv(spatial, meta["BLOCK_SIZE"]),
        N * C,
    )
    with torch_device_fn.device(input.device):
        upsample_nearest3d_kernel[grid](
            output,
            input,
            N * C,
            OD,
            OH,
            OW,
            ID,
            IH,
            IW,
            reciprocal_scale_d,
            reciprocal_scale_h,
            reciprocal_scale_w,
            total_out,
            SAME_D=(OD == ID),
            SAME_H=(OH == IH),
            SAME_W=(OW == IW),
            USE_INT32_IDX=(total_out <= (2**31 - 1)),
            num_warps=8,
        )
    return output
