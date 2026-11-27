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

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.autotune(
    configs=runtime.get_tuned_config("upsample_nearest2d"), key=["N", "C", "OH", "OW"]
)
@triton.heuristics(runtime.get_heuristic_config("upsample_nearest2d"))
@triton.jit
def upsample_nearest2d_kernel(
    ptr_o,
    ptr_i,
    N,
    C,
    OH,
    OW,
    IH,
    IW,
    reciprocal_scale_h,
    reciprocal_scale_w,
    BLOCK_SIZE: tl.constexpr,
    NUM_TILE: tl.constexpr,
    TOTAL_TILE: tl.constexpr,
    SAME_H: tl.constexpr,
    SAME_W: tl.constexpr,
    USE_INT32_IDX: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    NC = N * C
    nc_iter = tl.program_id(axis=1)
    pid = tl.program_id(axis=0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    ow = idx % OW
    oh = idx // OW % OH
    if SAME_H:
        ih = oh
    else:
        # tl.floor() cannot be found in 2.3.1, using int trunc
        ih = tl.minimum((oh * reciprocal_scale_h).to(tl.int32), IH - 1)
    if SAME_W:
        iw = ow
    else:
        iw = tl.minimum((ow * reciprocal_scale_w).to(tl.int32), IW - 1)

    for i in range(NUM_TILE):
        nc_iter_start = nc_iter * NUM_TILE + i
        ptr_i_start = ptr_i
        ptr_o_start = ptr_o
        offset_o = (nc_iter_start * OH + oh) * OW + ow
        offset_i = (nc_iter_start * IH + ih) * IW + iw
        src_index_stride = TOTAL_TILE * IH * IW
        dst_index_stride = TOTAL_TILE * OH * OW
        nc_iter_tmp = nc_iter_start
        while nc_iter_tmp < NC:
            data = tl.load(ptr_i_start + offset_i)
            tl.store(ptr_o_start + offset_o, data)
            ptr_i_start += src_index_stride
            ptr_o_start += dst_index_stride
            nc_iter_tmp += TOTAL_TILE


# Fast path for power-of-two integer scales (OH == IH * 2**log2_sh, OW == IW * 2**log2_sw).
# Loads each unique input row block once (W axis gathered with run-length duplication),
# then replicates it over SH output rows. Stores are fully contiguous row segments.
@triton.jit
def upsample_nearest2d_pow2_element_kernel(
    ptr_o,
    ptr_i,
    IH,
    IW,
    OH,
    OW,
    NC,
    GW,
    LOG2_SH: tl.constexpr,
    LOG2_SW: tl.constexpr,
    BH: tl.constexpr,
    TOW: tl.constexpr,
    NTP: tl.constexpr,
    NCZ: tl.constexpr,
    SINGLE_NC: tl.constexpr,
    EVEN: tl.constexpr,
):
    SH: tl.constexpr = 1 << LOG2_SH
    pid_g = tl.program_id(0)
    pid_h = tl.program_id(1)
    nc0 = tl.program_id(2)
    r = tl.arange(0, BH)
    c = tl.arange(0, TOW)
    cw = c >> LOG2_SW
    ih = pid_h * BH + r
    if SINGLE_NC:
        base_i = nc0.to(tl.int64) * (IH * IW)
        base_o = nc0.to(tl.int64) * (OH * OW)
        for t in tl.static_range(NTP):
            tw = pid_g * NTP + t
            if tw < GW:
                ow = tw * TOW + c
                iw = (tw * TOW >> LOG2_SW) + cw
                if EVEN:
                    src = tl.load(ptr_i + base_i + ih[:, None] * IW + iw[None, :])
                    for sh in tl.static_range(SH):
                        ohr = ih * SH + sh
                        tl.store(ptr_o + base_o + ohr[:, None] * OW + ow[None, :], src)
                else:
                    mh = ih < IH
                    mw = ow < OW
                    src = tl.load(
                        ptr_i + base_i + ih[:, None] * IW + iw[None, :],
                        mask=mh[:, None] & mw[None, :],
                        other=0.0,
                    )
                    for sh in tl.static_range(SH):
                        ohr = ih * SH + sh
                        tl.store(
                            ptr_o + base_o + ohr[:, None] * OW + ow[None, :],
                            src,
                            mask=(ohr < OH)[:, None] & mw[None, :],
                        )
    else:
        for nc in range(nc0, NC, NCZ):
            base_i = nc.to(tl.int64) * (IH * IW)
            base_o = nc.to(tl.int64) * (OH * OW)
            for t in tl.static_range(NTP):
                tw = pid_g * NTP + t
                if tw < GW:
                    ow = tw * TOW + c
                    iw = (tw * TOW >> LOG2_SW) + cw
                    if EVEN:
                        src = tl.load(ptr_i + base_i + ih[:, None] * IW + iw[None, :])
                        for sh in tl.static_range(SH):
                            ohr = ih * SH + sh
                            tl.store(
                                ptr_o + base_o + ohr[:, None] * OW + ow[None, :], src
                            )
                    else:
                        mh = ih < IH
                        mw = ow < OW
                        src = tl.load(
                            ptr_i + base_i + ih[:, None] * IW + iw[None, :],
                            mask=mh[:, None] & mw[None, :],
                            other=0.0,
                        )
                        for sh in tl.static_range(SH):
                            ohr = ih * SH + sh
                            tl.store(
                                ptr_o + base_o + ohr[:, None] * OW + ow[None, :],
                                src,
                                mask=(ohr < OH)[:, None] & mw[None, :],
                            )


# Packed fast path for 2-byte dtypes with SW == 2: each output int32 word is the
# input element duplicated into both halves ((x << 16) | x), so stores run at
# 4-byte granularity instead of the 2-byte element-rate limit.
@triton.jit
def upsample_nearest2d_pow2_pack16_kernel(
    ptr_o32,
    ptr_i,
    IH,
    IW,
    OH,
    OW32,
    NC,
    GW,
    LOG2_SH: tl.constexpr,
    BH: tl.constexpr,
    BW: tl.constexpr,
    NTP: tl.constexpr,
    NCZ: tl.constexpr,
    SINGLE_NC: tl.constexpr,
    EVEN: tl.constexpr,
):
    SH: tl.constexpr = 1 << LOG2_SH
    pid_g = tl.program_id(0)
    pid_h = tl.program_id(1)
    nc0 = tl.program_id(2)
    r = tl.arange(0, BH)
    c = tl.arange(0, BW)
    ih = pid_h * BH + r
    if SINGLE_NC:
        base_i = nc0.to(tl.int64) * (IH * IW)
        base_o = nc0.to(tl.int64) * (OH * OW32)
        for t in tl.static_range(NTP):
            tw = pid_g * NTP + t
            if tw < GW:
                iw = tw * BW + c
                if EVEN:
                    x = tl.load(ptr_i + base_i + ih[:, None] * IW + iw[None, :])
                    xi = x.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
                    w = xi | (xi << 16)
                    for sh in tl.static_range(SH):
                        ohr = ih * SH + sh
                        tl.store(
                            ptr_o32 + base_o + ohr[:, None] * OW32 + iw[None, :], w
                        )
                else:
                    mh = ih < IH
                    mw = iw < IW
                    x = tl.load(
                        ptr_i + base_i + ih[:, None] * IW + iw[None, :],
                        mask=mh[:, None] & mw[None, :],
                        other=0.0,
                    )
                    xi = x.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
                    w = xi | (xi << 16)
                    for sh in tl.static_range(SH):
                        ohr = ih * SH + sh
                        tl.store(
                            ptr_o32 + base_o + ohr[:, None] * OW32 + iw[None, :],
                            w,
                            mask=(ohr < OH)[:, None] & mw[None, :],
                        )
    else:
        for nc in range(nc0, NC, NCZ):
            base_i = nc.to(tl.int64) * (IH * IW)
            base_o = nc.to(tl.int64) * (OH * OW32)
            for t in tl.static_range(NTP):
                tw = pid_g * NTP + t
                if tw < GW:
                    iw = tw * BW + c
                    if EVEN:
                        x = tl.load(ptr_i + base_i + ih[:, None] * IW + iw[None, :])
                        xi = x.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
                        w = xi | (xi << 16)
                        for sh in tl.static_range(SH):
                            ohr = ih * SH + sh
                            tl.store(
                                ptr_o32 + base_o + ohr[:, None] * OW32 + iw[None, :], w
                            )
                    else:
                        mh = ih < IH
                        mw = iw < IW
                        x = tl.load(
                            ptr_i + base_i + ih[:, None] * IW + iw[None, :],
                            mask=mh[:, None] & mw[None, :],
                            other=0.0,
                        )
                        xi = x.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
                        w = xi | (xi << 16)
                        for sh in tl.static_range(SH):
                            ohr = ih * SH + sh
                            tl.store(
                                ptr_o32 + base_o + ohr[:, None] * OW32 + iw[None, :],
                                w,
                                mask=(ohr < OH)[:, None] & mw[None, :],
                            )


_POW2_SCALE_MAX = 16
_GRID_Y_MAX = 255
_NTP = 4
_NUM_WARPS = 4


def _pow2_floor(value):
    result = 1
    while result * 2 <= value:
        result *= 2
    return result


def _pow2_log_scale(out_size, in_size, reciprocal):
    """Return log2(scale) if out_size == in_size * 2**k (k <= 16) and the
    effective reciprocal scale is exactly 1 / 2**k; otherwise None."""
    if in_size <= 0 or out_size % in_size != 0:
        return None
    scale = out_size // in_size
    if scale < 1 or scale > _POW2_SCALE_MAX or (scale & (scale - 1)) != 0:
        return None
    if reciprocal != 1.0 / scale:
        return None
    return scale.bit_length() - 1


def _launch_pow2_fastpath(
    output, input, OH, OW, reciprocal_scale_h, reciprocal_scale_w
):
    N, C, IH, IW = input.shape
    NC = N * C
    if NC <= 0 or IH <= 0 or IW <= 0 or OH <= 0 or OW <= 0:
        return True
    if IH * IW >= 2**31 or OH * OW >= 2**31:
        return False
    log2_sh = _pow2_log_scale(OH, IH, reciprocal_scale_h)
    log2_sw = _pow2_log_scale(OW, IW, reciprocal_scale_w)
    if log2_sh is None or log2_sw is None:
        return False
    dtype = input.dtype
    esize = input.element_size()
    with torch_device_fn.device(input.device):
        if esize == 2 and log2_sw == 1 and dtype in (torch.float16, torch.bfloat16):
            BH = 64
            BW = min(256, _pow2_floor(IW))
            while triton.cdiv(IH, BH) > _GRID_Y_MAX and BH < 256:
                BH *= 2
            if triton.cdiv(IH, BH) > _GRID_Y_MAX:
                return False
            OW32 = OW // 2
            gw = triton.cdiv(IW, BW)
            gh = triton.cdiv(IH, BH)
            ncz = min(NC, 128)
            even = (IH % BH == 0) and (IW % BW == 0)
            grid = (triton.cdiv(gw, _NTP), gh, ncz)
            upsample_nearest2d_pow2_pack16_kernel[grid](
                output.view(torch.int32),
                input,
                IH,
                IW,
                OH,
                OW32,
                NC,
                gw,
                log2_sh,
                BH,
                BW,
                _NTP,
                ncz,
                ncz >= NC,
                even,
                num_warps=_NUM_WARPS,
            )
            return True
        if esize in (2, 4) and dtype in (torch.float16, torch.bfloat16, torch.float32):
            if esize == 2 and OW >= 512:
                BH, TOW = 16, 512
            else:
                BH, TOW = 64, min(256, _pow2_floor(OW))
            while triton.cdiv(IH, BH) > _GRID_Y_MAX and BH < 256:
                BH *= 2
            if triton.cdiv(IH, BH) > _GRID_Y_MAX:
                return False
            gw = triton.cdiv(OW, TOW)
            gh = triton.cdiv(IH, BH)
            ncz = min(NC, 128)
            even = (IH % BH == 0) and (OW % TOW == 0)
            grid = (triton.cdiv(gw, _NTP), gh, ncz)
            upsample_nearest2d_pow2_element_kernel[grid](
                output,
                input,
                IH,
                IW,
                OH,
                OW,
                NC,
                gw,
                log2_sh,
                log2_sw,
                BH,
                TOW,
                _NTP,
                ncz,
                ncz >= NC,
                even,
                num_warps=_NUM_WARPS,
            )
            return True
    return False


def upsample_nearest2d(
    input: torch.Tensor,
    output_size: Tuple[int],
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME UPSAMPLE_NEAREST2D")
    assert input.ndim == 4, "The ndim of input must be 4"
    assert len(output_size) == 2, "The len of output_size must be 2"
    OH, OW = output_size
    N, C, IH, IW = input.shape
    if scales_h is not None:
        reciprocal_scale_h = 1 / scales_h
    else:
        reciprocal_scale_h = IH / OH
    if scales_w is not None:
        reciprocal_scale_w = 1 / scales_w
    else:
        reciprocal_scale_w = IW / OW
    # allocate output
    output = torch.empty((N, C, OH, OW), device=input.device, dtype=input.dtype)
    if not _launch_pow2_fastpath(
        output, input, OH, OW, reciprocal_scale_h, reciprocal_scale_w
    ):
        total_threads = OH * OW
        grid = lambda META: (
            triton.cdiv(total_threads, META["BLOCK_SIZE"]),
            min(128, triton.cdiv(N * C, 4)),
        )

        with torch_device_fn.device(input.device):
            upsample_nearest2d_kernel[grid](
                output,
                input,
                N,
                C,
                OH,
                OW,
                IH,
                IW,
                reciprocal_scale_h,
                reciprocal_scale_w,
            )
    return output
