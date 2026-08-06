# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from typing import Optional, Tuple

import torch
import triton

from flag_gems.runtime import torch_device_fn
from flag_gems.ops.upsample_nearest3d import upsample_nearest3d_kernel


def _calculate_scale(in_size: int, out_size: int, scale: Optional[float]) -> float:
    if scale is not None:
        return float(torch.tensor(1.0 / scale, dtype=torch.float32).item())
    return float(
        (torch.tensor(in_size, dtype=torch.float32) / torch.tensor(out_size, dtype=torch.float32)).item()
    )


def upsample_nearest3d(
    input: torch.Tensor,
    output_size: Tuple[int, int, int],
    scales_d: Optional[float] = None,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
):
    OD, OH, OW = output_size
    N, C, ID, IH, IW = input.shape
    reciprocal_scale_d = _calculate_scale(ID, OD, scales_d)
    reciprocal_scale_h = _calculate_scale(IH, OH, scales_h)
    reciprocal_scale_w = _calculate_scale(IW, OW, scales_w)
    output = torch.empty((N, C, OD, OH, OW), device=input.device, dtype=input.dtype)
    total_threads = OD * OH * OW
    nc_per_program = 1 if N * C <= 4 else 4
    grid = lambda meta: (
        triton.cdiv(total_threads, meta["BLOCK_SIZE"]),
        triton.cdiv(N * C, nc_per_program),
    )
    with torch_device_fn.device(input.device):
        upsample_nearest3d_kernel[grid](
            output,
            input,
            N,
            C,
            OD,
            OH,
            OW,
            ID,
            IH,
            IW,
            reciprocal_scale_d,
            reciprocal_scale_h,
            reciprocal_scale_w,
        )
    return output
