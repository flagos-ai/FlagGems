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

import numpy as np
import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn

logger = logging.getLogger(__name__)
device = device.name

PROGRAMS = 12
MAX_BLOCK = 16384
MIN_BLOCK = 512
NUM_WARPS = 4

# Staging copy (only reached for non-contiguous inputs): aim for ~12 programs
# like the main kernel, but keep the tile at or below 4096 -- 32768 is a
# registered hard-fail on this backend.
STAGE_PROGRAMS = 12
STAGE_MAX_BLOCK = 4096
STAGE_MIN_BLOCK = 512
STAGE_NUM_WARPS = 4


@triton.jit
def _upsample_nearest_exact2d_kernel(
    ptr_o,
    ptr_i,
    total,
    NC: tl.constexpr,
    C: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    rheight,
    rwidth,
    soN,
    soC,
    soH,
    soW,
    OUT_STRIDED: tl.constexpr,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)

    sp = idx % (OH * OW)
    nc = idx // (OH * OW)
    ow = sp % OW
    oh = sp // OW

    ih = tl.minimum(((oh + 0.5) * rheight).to(tl.int32), IH - 1)
    iw = tl.minimum(((ow + 0.5) * rwidth).to(tl.int32), IW - 1)
    nc = tl.minimum(nc, NC - 1)
    data = tl.load(ptr_i + (nc * IH + ih) * IW + iw)
    if OUT_STRIDED:
        n = nc // C
        c = nc - n * C
        o_off = n * soN + c * soC + oh * soH + ow * soW
    else:
        o_off = idx
    if NEED_MASK:
        tl.store(ptr_o + o_off, data, mask=idx < total)
    else:
        tl.store(ptr_o + o_off, data)


def _f32(value):
    return float(np.float32(value))


def _reciprocal_scale(in_size, out_size, scale):
    if scale is not None and scale > 0:
        return _f32(1.0 / scale)
    return float(np.float32(in_size) / np.float32(out_size))


def _pick_block(total):
    return min(
        MAX_BLOCK, max(MIN_BLOCK, triton.next_power_of_2(triton.cdiv(total, PROGRAMS)))
    )


def _shape_args(input, output_size, scales_h, scales_w):
    N, C, IH, IW = input.shape
    if output_size is not None:
        OH, OW = int(output_size[-2]), int(output_size[-1])
    else:
        scale_h = scales_h if scales_h is not None else 1.0
        scale_w = scales_w if scales_w is not None else 1.0
        OH, OW = int(math.floor(IH * scale_h)), int(math.floor(IW * scale_w))
    if OH < 0 or OW < 0:
        raise ValueError("Output size must be non-negative.")
    if OH * OW > 0 and (IH == 0 or IW == 0):
        # Same rejection as ATen; the generic implementation silently loads out
        # of bounds here.
        raise RuntimeError(
            "Input and output sizes should be greater than 0, but got input "
            f"(H: {IH}, W: {IW}) output (H: {OH}, W: {OW})"
        )
    return N, C, IH, IW, OH, OW


@triton.jit
def _stage_contiguous_kernel(
    ptr_dst,
    ptr_src,
    N: tl.constexpr,
    C: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    sN,
    sC,
    sH,
    sW,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    flat = pid * BLOCK + tl.arange(0, BLOCK)
    iw = flat % IW
    ih = (flat // IW) % IH
    c = (flat // (IH * IW)) % C
    n = tl.minimum(flat // (C * IH * IW), N - 1)
    off = n * sN + c * sC + ih * sH + iw * sW
    tl.store(ptr_dst + flat, tl.load(ptr_src + off))


def _stage_contiguous(input):
    N, C, IH, IW = input.shape
    if input.stride() == (C * IH * IW, IH * IW, IW, 1):
        return input
    total = N * C * IH * IW
    sN, sC, sH, sW = input.stride()
    block = min(
        STAGE_MAX_BLOCK,
        max(
            STAGE_MIN_BLOCK,
            triton.next_power_of_2(triton.cdiv(total, STAGE_PROGRAMS)),
        ),
    )
    grid = (triton.cdiv(total, block),)
    buf = torch.empty(grid[0] * block, device=input.device, dtype=input.dtype)
    staged = buf[:total].view(N, C, IH, IW)
    with torch_device_fn.device(input.device):
        _stage_contiguous_kernel[grid](
            staged,
            input,
            N=N,
            C=C,
            IH=IH,
            IW=IW,
            sN=sN,
            sC=sC,
            sH=sH,
            sW=sW,
            BLOCK=block,
            num_warps=STAGE_NUM_WARPS,
        )
    return staged


def _launch(input, out, total, rheight, rwidth, out_strided, need_mask):
    N, C, IH, IW = input.shape
    OH, OW = out.shape[-2:]
    block = _pick_block(total)
    grid = (triton.cdiv(total, block),)
    soN, soC, soH, soW = out.stride()
    with torch_device_fn.device(input.device):
        _upsample_nearest_exact2d_kernel[grid](
            out,
            input,
            total,
            NC=N * C,
            C=C,
            IH=IH,
            IW=IW,
            OH=OH,
            OW=OW,
            rheight=rheight,
            rwidth=rwidth,
            soN=soN,
            soC=soC,
            soH=soH,
            soW=soW,
            OUT_STRIDED=out_strided,
            BLOCK=block,
            NEED_MASK=need_mask,
            num_warps=NUM_WARPS,
        )
    return out


def _upsample_nearest_exact2d(
    input: torch.Tensor,
    output_size: Optional[Tuple[int, int]] = None,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN _UPSAMPLE_NEAREST_EXACT2D")
    assert input.device.type == device
    if input.ndim != 4:
        raise ValueError(
            "_upsample_nearest_exact2d expects a 4D tensor (N, C, H, W); "
            f"got shape {tuple(input.shape)}"
        )
    N, C, IH, IW, OH, OW = _shape_args(input, output_size, scales_h, scales_w)
    total = N * C * OH * OW
    if total == 0:
        return torch.empty((N, C, OH, OW), device=input.device, dtype=input.dtype)
    rheight = _reciprocal_scale(IH, OH, scales_h)
    rwidth = _reciprocal_scale(IW, OW, scales_w)
    input = _stage_contiguous(input)

    block = _pick_block(total)
    buf = torch.empty(
        triton.cdiv(total, block) * block, device=input.device, dtype=input.dtype
    )
    output = buf[:total].view(N, C, OH, OW)
    return _launch(input, output, total, rheight, rwidth, False, False)


def _upsample_nearest_exact2d_out(
    input: torch.Tensor,
    output_size: Optional[Tuple[int, int]] = None,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN _UPSAMPLE_NEAREST_EXACT2D_OUT")
    assert input.device.type == device
    if input.ndim != 4:
        raise ValueError(
            "_upsample_nearest_exact2d expects a 4D tensor (N, C, H, W); "
            f"got shape {tuple(input.shape)}"
        )
    N, C, IH, IW, OH, OW = _shape_args(input, output_size, scales_h, scales_w)
    if tuple(out.shape) != (N, C, OH, OW):
        raise ValueError(
            "Provided out tensor has shape "
            f"{tuple(out.shape)} but expected {(N, C, OH, OW)}."
        )
    if out.dtype != input.dtype:
        raise ValueError(
            f"Provided out tensor has dtype {out.dtype} but expected {input.dtype}."
        )
    total = out.numel()
    if total == 0:
        return out
    rheight = _reciprocal_scale(IH, OH, scales_h)
    rwidth = _reciprocal_scale(IW, OW, scales_w)
    input = _stage_contiguous(input)
    out_strided = not out.is_contiguous()
    # The caller's buffer cannot be over-allocated, so a partial tail tile has
    # to fall back to a guarded store.
    need_mask = out_strided or total % _pick_block(total) != 0
    return _launch(input, out, total, rheight, rwidth, out_strided, need_mask)


__all__ = ["_upsample_nearest_exact2d", "_upsample_nearest_exact2d_out"]
