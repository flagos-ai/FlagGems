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

# Per-program lane caps: 1-D blocks below 256 lanes are expanded with OOB
# writes on this backend, and large single tiles are slow, so both caps keep
# the compiled tile bounded.
_MAX_ROW_BLOCK = 1024
_MAX_FLAT_BLOCK = 2048
# Row-wise kernels launch one program per output row; for small row widths
# that becomes launch-bound (tens of thousands of tiny programs), so the flat
# variant (many rows per program) is used below this threshold.
_MIN_ROW_IW = 512
_NUM_WARPS = 4

# Staging copy (only reached for non-contiguous inputs): aim for ~12 programs
# like the main kernel, but keep the tile at or below 4096 -- 32768 is a
# registered hard-fail on this backend.
STAGE_PROGRAMS = 12
STAGE_MAX_BLOCK = 4096
STAGE_MIN_BLOCK = 512
STAGE_NUM_WARPS = 4


@triton.jit
def _upsample_nearest_exact2d_2x16_kernel(
    out32_ptr,  # output viewed as uint32 (two f16/bf16 per element)
    in16_ptr,  # input viewed as uint16
    OH,
    OW,
    IH,
    IW,
    rheight,
    NC,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    # Exact 2x fast path (OH == 2*IH, OW == 2*IW, rheight == rwidth == 0.5):
    # output[oh, 2j] == output[oh, 2j+1] == input[ih, j], where
    # ih = min(fptosi((oh+0.5)*rheight), IH-1) == oh // 2 (the f32 product is
    # exact, so fptosi equals the integer floor). One program per output row
    # block: each lane loads one input element (u16 view, contiguous 1-D) and
    # stores one packed uint32 (two output elements, contiguous 1-D), so both
    # sides stay block-DMA and no per-element div/rem or float-to-int chain
    # is generated. The input row is re-read for the paired output row, which
    # keeps the store fully contiguous at the cost of 2x loads (both cached).
    j = tl.program_id(axis=0) * BLOCK + tl.arange(0, BLOCK)
    oh = tl.program_id(axis=1)
    nc = tl.program_id(axis=2)
    ih = tl.minimum(((oh + 0.5) * rheight).to(tl.int32), IH - 1)
    if NEED_MASK:
        v = tl.load(
            in16_ptr + (nc * IH + ih).to(tl.int64) * IW + j, mask=j < IW, other=0
        )
    else:
        v = tl.load(in16_ptr + (nc * IH + ih).to(tl.int64) * IW + j)
    u = v.to(tl.uint32)
    packed = u | (u << 16)
    o_off = (nc * OH + oh).to(tl.int64) * IW + j
    if NEED_MASK:
        tl.store(out32_ptr + o_off, packed, mask=j < IW)
    else:
        tl.store(out32_ptr + o_off, packed)


@triton.jit
def _upsample_nearest_exact2d_2x32_kernel(
    out64_ptr,  # output viewed as uint64 (two f32 per element)
    in32_ptr,  # input viewed as uint32
    OH,
    OW,
    IH,
    IW,
    rheight,
    NC,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    # Same as _2x16_kernel but for f32: one uint64 stores the two identical
    # output elements (bit pattern duplicated, little-endian order).
    j = tl.program_id(axis=0) * BLOCK + tl.arange(0, BLOCK)
    oh = tl.program_id(axis=1)
    nc = tl.program_id(axis=2)
    ih = tl.minimum(((oh + 0.5) * rheight).to(tl.int32), IH - 1)
    if NEED_MASK:
        v = tl.load(
            in32_ptr + (nc * IH + ih).to(tl.int64) * IW + j, mask=j < IW, other=0
        )
    else:
        v = tl.load(in32_ptr + (nc * IH + ih).to(tl.int64) * IW + j)
    w = v.to(tl.uint64)
    packed = w | (w << 32)
    o_off = (nc * OH + oh).to(tl.int64) * IW + j
    if NEED_MASK:
        tl.store(out64_ptr + o_off, packed, mask=j < IW)
    else:
        tl.store(out64_ptr + o_off, packed)


@triton.jit
def _upsample_nearest_exact2d_2x16_flat_kernel(
    out32_ptr,
    in16_ptr,
    OH,
    OW,
    IH,
    IW,
    rheight,
    NC,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    # Flat variant of the 2x16 kernel: one program covers a chunk of the whole
    # (OH x IW) uint32 plane (many rows), recovering the row with a single
    # idx // IW pair. Used for small IW where one-row-per-program would launch
    # too many tiny programs.
    idx = tl.program_id(axis=0) * BLOCK + tl.arange(0, BLOCK)
    nc = tl.program_id(axis=1)
    oh = idx // IW
    ow = idx % IW
    ih = tl.minimum(((oh + 0.5) * rheight).to(tl.int32), IH - 1)
    if NEED_MASK:
        v = tl.load(
            in16_ptr + (nc * IH + ih).to(tl.int64) * IW + ow,
            mask=idx < OH * IW,
            other=0,
        )
    else:
        v = tl.load(in16_ptr + (nc * IH + ih).to(tl.int64) * IW + ow)
    u = v.to(tl.uint32)
    packed = u | (u << 16)
    o_off = (nc * OH * IW).to(tl.int64) + idx
    if NEED_MASK:
        tl.store(out32_ptr + o_off, packed, mask=idx < OH * IW)
    else:
        tl.store(out32_ptr + o_off, packed)


@triton.jit
def _upsample_nearest_exact2d_2x32_flat_kernel(
    out64_ptr,
    in32_ptr,
    OH,
    OW,
    IH,
    IW,
    rheight,
    NC,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    idx = tl.program_id(axis=0) * BLOCK + tl.arange(0, BLOCK)
    nc = tl.program_id(axis=1)
    oh = idx // IW
    ow = idx % IW
    ih = tl.minimum(((oh + 0.5) * rheight).to(tl.int32), IH - 1)
    if NEED_MASK:
        v = tl.load(
            in32_ptr + (nc * IH + ih).to(tl.int64) * IW + ow,
            mask=idx < OH * IW,
            other=0,
        )
    else:
        v = tl.load(in32_ptr + (nc * IH + ih).to(tl.int64) * IW + ow)
    w = v.to(tl.uint64)
    packed = w | (w << 32)
    o_off = (nc * OH * IW).to(tl.int64) + idx
    if NEED_MASK:
        tl.store(out64_ptr + o_off, packed, mask=idx < OH * IW)
    else:
        tl.store(out64_ptr + o_off, packed)


@triton.jit
def _upsample_nearest_exact2d_row_kernel(
    out_ptr,
    in_ptr,
    OH,
    OW,
    IH,
    IW,
    rheight,
    rwidth,
    NC,
    C,
    soN,
    soC,
    soH,
    soW,
    OUT_STRIDED: tl.constexpr,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    # General (any scale) kernel: one program per output row block, so the row
    # index is a scalar program id and no per-element div/rem chain is
    # generated. The output store is an affine 1-D block (contiguous when
    # OUT_STRIDED is false); the input load is a per-row gather with the
    # ATen-compatible exact mapping (clamped, always in-bounds, so only the
    # store needs a tail mask).
    ow = tl.program_id(axis=0) * BLOCK + tl.arange(0, BLOCK)
    oh = tl.program_id(axis=1)
    nc = tl.program_id(axis=2)
    ih = tl.minimum(((oh + 0.5) * rheight).to(tl.int32), IH - 1)
    iw = tl.minimum(((ow + 0.5) * rwidth).to(tl.int32), IW - 1)
    data = tl.load(in_ptr + (nc * IH + ih).to(tl.int64) * IW + iw)
    if OUT_STRIDED:
        o_off = (nc // C) * soN + (nc % C) * soC + oh * soH + ow * soW
    else:
        o_off = (nc * OH + oh).to(tl.int64) * OW + ow
    if NEED_MASK:
        tl.store(out_ptr + o_off, data, mask=ow < OW)
    else:
        tl.store(out_ptr + o_off, data)


def _f32(value):
    return float(np.float32(value))


def _reciprocal_scale(in_size, out_size, scale):
    if scale is not None and scale > 0:
        return _f32(1.0 / scale)
    return float(np.float32(in_size) / np.float32(out_size))


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


def _launch_2x(input, output, rheight, NC, OH, OW, IH, IW):
    # Exact-2x fast path: both load and store stay contiguous 1-D blocks
    # (packed u32/u64 stores, one packed element per lane).
    if input.dtype in (torch.float16, torch.bfloat16):
        ptr_i = input.view(torch.uint16)
        ptr_o = output.view(torch.uint32)
        if IW >= _MIN_ROW_IW:
            block = min(_MAX_ROW_BLOCK, triton.next_power_of_2(IW))
            grid = (triton.cdiv(IW, block), OH, NC)
            with torch_device_fn.device(input.device):
                _upsample_nearest_exact2d_2x16_kernel[grid](
                    ptr_o,
                    ptr_i,
                    OH,
                    OW,
                    IH,
                    IW,
                    rheight,
                    NC,
                    BLOCK=block,
                    NEED_MASK=(IW % block != 0),
                    num_warps=_NUM_WARPS,
                )
        else:
            block = min(_MAX_FLAT_BLOCK, triton.next_power_of_2(OH * IW))
            grid = (triton.cdiv(OH * IW, block), NC)
            with torch_device_fn.device(input.device):
                _upsample_nearest_exact2d_2x16_flat_kernel[grid](
                    ptr_o,
                    ptr_i,
                    OH,
                    OW,
                    IH,
                    IW,
                    rheight,
                    NC,
                    BLOCK=block,
                    NEED_MASK=((OH * IW) % block != 0),
                    num_warps=_NUM_WARPS,
                )
    else:
        ptr_i = input.view(torch.uint32)
        ptr_o = output.view(torch.uint64)
        if IW >= _MIN_ROW_IW:
            block = min(_MAX_ROW_BLOCK, triton.next_power_of_2(IW))
            grid = (triton.cdiv(IW, block), OH, NC)
            with torch_device_fn.device(input.device):
                _upsample_nearest_exact2d_2x32_kernel[grid](
                    ptr_o,
                    ptr_i,
                    OH,
                    OW,
                    IH,
                    IW,
                    rheight,
                    NC,
                    BLOCK=block,
                    NEED_MASK=(IW % block != 0),
                    num_warps=_NUM_WARPS,
                )
        else:
            block = min(_MAX_FLAT_BLOCK, triton.next_power_of_2(OH * IW))
            grid = (triton.cdiv(OH * IW, block), NC)
            with torch_device_fn.device(input.device):
                _upsample_nearest_exact2d_2x32_flat_kernel[grid](
                    ptr_o,
                    ptr_i,
                    OH,
                    OW,
                    IH,
                    IW,
                    rheight,
                    NC,
                    BLOCK=block,
                    NEED_MASK=((OH * IW) % block != 0),
                    num_warps=_NUM_WARPS,
                )
    return output


def _launch_row(input, output, rheight, rwidth, NC, C, OH, OW, IH, IW, need_mask):
    block = min(_MAX_ROW_BLOCK, triton.next_power_of_2(OW))
    grid = (triton.cdiv(OW, block), OH, NC)
    out_strided = not output.is_contiguous()
    soN, soC, soH, soW = output.stride()
    with torch_device_fn.device(input.device):
        _upsample_nearest_exact2d_row_kernel[grid](
            output,
            input,
            OH,
            OW,
            IH,
            IW,
            rheight,
            rwidth,
            NC,
            C,
            soN,
            soC,
            soH,
            soW,
            OUT_STRIDED=out_strided,
            BLOCK=block,
            NEED_MASK=need_mask,
            num_warps=_NUM_WARPS,
        )
    return output


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
    output = torch.empty((N, C, OH, OW), device=input.device, dtype=input.dtype)
    NC = N * C
    is_2x = (
        OH == 2 * IH
        and OW == 2 * IW
        and (scales_h is None or float(scales_h) == 2.0)
        and (scales_w is None or float(scales_w) == 2.0)
    )
    if is_2x:
        return _launch_2x(input, output, rheight, NC, OH, OW, IH, IW)
    block = min(_MAX_ROW_BLOCK, triton.next_power_of_2(OW))
    return _launch_row(
        input, output, rheight, rwidth, NC, C, OH, OW, IH, IW, OW % block != 0
    )


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
    NC = N * C
    is_2x = (
        OH == 2 * IH
        and OW == 2 * IW
        and (scales_h is None or float(scales_h) == 2.0)
        and (scales_w is None or float(scales_w) == 2.0)
    )
    if is_2x and out.is_contiguous():
        # The fast path views the buffer as u32/u64; a user buffer is not
        # over-allocated, so guarded stores are needed on partial tiles.
        return _launch_2x(input, out, rheight, NC, OH, OW, IH, IW)
    # General path. The row kernel reads the caller's buffer with its own
    # strides; a partial tail tile needs a guarded store (the buffer cannot be
    # over-allocated). The tail condition is per output row (ow >= OW would
    # leave the row and must not be written); a strided out also needs guards.
    block = min(_MAX_ROW_BLOCK, triton.next_power_of_2(OW))
    need_mask = OW % block != 0 or not out.is_contiguous()
    return _launch_row(input, out, rheight, rwidth, NC, C, OH, OW, IH, IW, need_mask)


__all__ = ["_upsample_nearest_exact2d", "_upsample_nearest_exact2d_out"]
