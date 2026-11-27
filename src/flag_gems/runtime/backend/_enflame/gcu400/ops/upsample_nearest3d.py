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
from flag_gems.runtime import device, torch_device_fn

device = device.name
logger = logging.getLogger(__name__)


@triton.autotune(
    configs=runtime.get_tuned_config("upsample_nearest3d"),
    key=["N", "C", "OD", "OH", "OW"],
)
@triton.heuristics(runtime.get_heuristic_config("upsample_nearest3d"))
@triton.jit
def upsample_nearest3d_kernel(
    ptr_o,
    ptr_i,
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
    BLOCK_SIZE: tl.constexpr,
    SAME_D: tl.constexpr,
    SAME_H: tl.constexpr,
    SAME_W: tl.constexpr,
    USE_INT32_IDX: tl.constexpr,
):
    if USE_INT32_IDX:
        pid0 = tl.program_id(axis=0)
    else:
        pid0 = tl.program_id(axis=0).to(tl.int64)

    nc_stride = tl.num_programs(axis=1)
    NC = N * C
    nc_iter = tl.program_id(axis=1)

    idx = pid0 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total_spatial_size = OD * OH * OW

    mask = idx < total_spatial_size

    ow = idx % OW
    oh = (idx // OW) % OH
    od = idx // (OW * OH)

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

    offset_o = nc_iter * (OD * OH * OW) + idx
    offset_i = nc_iter * (ID * IH * IW) + (id * IH * IW + ih * IW + iw)

    src_nc_stride = nc_stride * (ID * IH * IW)
    dst_nc_stride = nc_stride * (OD * OH * OW)

    while nc_iter < NC:
        data = tl.load(ptr_i + offset_i, mask=mask)
        tl.store(ptr_o + offset_o, data, mask=mask)

        offset_i += src_nc_stride
        offset_o += dst_nc_stride
        nc_iter += nc_stride


# Fast path for power-of-two integer scales (OD == ID * 2**log2_sd etc.).
# Input-driven affine tiles: each (id, ih) input row block is loaded once
# (purely affine addresses), then replicated into SD * SH output rows; the W
# axis is handled by lane-level run-length gathers. No per-element div/mod or
# floor survives. Row-shift dup-gathers on the load side are a measured 4x
# regression (non-affine row vector triggers the per-lane gather path) -- keep
# the H-axis duplication on the store side.
@triton.jit
def upsample_nearest3d_pow2_element_kernel(
    ptr_o,
    ptr_i,
    ID,
    IH,
    IW,
    OD,
    OH,
    OW,
    NC,
    GHH,
    GW,
    LOG2_SD: tl.constexpr,
    LOG2_SH: tl.constexpr,
    LOG2_SW: tl.constexpr,
    BH: tl.constexpr,
    TOW: tl.constexpr,
    NTP: tl.constexpr,
    NCZ: tl.constexpr,
    SINGLE_NC: tl.constexpr,
    EVEN: tl.constexpr,
):
    SD: tl.constexpr = 1 << LOG2_SD
    SH: tl.constexpr = 1 << LOG2_SH
    pid_g = tl.program_id(0)
    pid_r = tl.program_id(1)
    nc0 = tl.program_id(2)
    id_ = pid_r // GHH
    ih = (pid_r % GHH) * BH + tl.arange(0, BH)
    c = tl.arange(0, TOW)
    ri = (id_ * IH + ih) * IW
    if SINGLE_NC:
        base_i = nc0.to(tl.int64) * (ID * IH * IW)
        base_o = nc0.to(tl.int64) * (OD * OH * OW)
        for t in tl.static_range(NTP):
            tw = pid_g * NTP + t
            if tw < GW:
                ow = tw * TOW + c
                iw = (tw * TOW >> LOG2_SW) + (c >> LOG2_SW)
                if EVEN:
                    src = tl.load(ptr_i + base_i + ri[:, None] + iw[None, :])
                    for sd in tl.static_range(SD):
                        row_o = ((id_ * SD + sd) * OH + ih * SH) * OW
                        for sh in tl.static_range(SH):
                            tl.store(
                                ptr_o + base_o + row_o[:, None] + sh * OW + ow[None, :],
                                src,
                            )
                else:
                    mh = ih < IH
                    mw = ow < OW
                    src = tl.load(
                        ptr_i + base_i + ri[:, None] + iw[None, :],
                        mask=mh[:, None] & mw[None, :],
                        other=0.0,
                    )
                    for sd in tl.static_range(SD):
                        row_o = ((id_ * SD + sd) * OH + ih * SH) * OW
                        for sh in tl.static_range(SH):
                            tl.store(
                                ptr_o + base_o + row_o[:, None] + sh * OW + ow[None, :],
                                src,
                                mask=mh[:, None] & mw[None, :],
                            )
    else:
        for nc in range(nc0, NC, NCZ):
            base_i = nc.to(tl.int64) * (ID * IH * IW)
            base_o = nc.to(tl.int64) * (OD * OH * OW)
            for t in tl.static_range(NTP):
                tw = pid_g * NTP + t
                if tw < GW:
                    ow = tw * TOW + c
                    iw = (tw * TOW >> LOG2_SW) + (c >> LOG2_SW)
                    if EVEN:
                        src = tl.load(ptr_i + base_i + ri[:, None] + iw[None, :])
                        for sd in tl.static_range(SD):
                            row_o = ((id_ * SD + sd) * OH + ih * SH) * OW
                            for sh in tl.static_range(SH):
                                tl.store(
                                    ptr_o
                                    + base_o
                                    + row_o[:, None]
                                    + sh * OW
                                    + ow[None, :],
                                    src,
                                )
                    else:
                        mh = ih < IH
                        mw = ow < OW
                        src = tl.load(
                            ptr_i + base_i + ri[:, None] + iw[None, :],
                            mask=mh[:, None] & mw[None, :],
                            other=0.0,
                        )
                        for sd in tl.static_range(SD):
                            row_o = ((id_ * SD + sd) * OH + ih * SH) * OW
                            for sh in tl.static_range(SH):
                                tl.store(
                                    ptr_o
                                    + base_o
                                    + row_o[:, None]
                                    + sh * OW
                                    + ow[None, :],
                                    src,
                                    mask=mh[:, None] & mw[None, :],
                                )


# Packed fast path for 2-byte dtypes with SW == 2: each output int32 word is the
# input element duplicated into both halves ((x << 16) | x), so stores run at
# 4-byte granularity instead of the 2-byte element-rate limit.
@triton.jit
def upsample_nearest3d_pow2_pack16_kernel(
    ptr_o32,
    ptr_i,
    ID,
    IH,
    IW,
    OD,
    OH,
    OW32,
    NC,
    GHH,
    GW,
    LOG2_SD: tl.constexpr,
    LOG2_SH: tl.constexpr,
    BH: tl.constexpr,
    BW: tl.constexpr,
    NTP: tl.constexpr,
    NCZ: tl.constexpr,
    SINGLE_NC: tl.constexpr,
    EVEN: tl.constexpr,
):
    SD: tl.constexpr = 1 << LOG2_SD
    SH: tl.constexpr = 1 << LOG2_SH
    pid_g = tl.program_id(0)
    pid_r = tl.program_id(1)
    nc0 = tl.program_id(2)
    id_ = pid_r // GHH
    ih = (pid_r % GHH) * BH + tl.arange(0, BH)
    c = tl.arange(0, BW)
    ri = (id_ * IH + ih) * IW
    if SINGLE_NC:
        base_i = nc0.to(tl.int64) * (ID * IH * IW)
        base_o = nc0.to(tl.int64) * (OD * OH * OW32)
        for t in tl.static_range(NTP):
            tw = pid_g * NTP + t
            if tw < GW:
                iw = tw * BW + c
                if EVEN:
                    x = tl.load(ptr_i + base_i + ri[:, None] + iw[None, :])
                    xi = x.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
                    w = xi | (xi << 16)
                    for sd in tl.static_range(SD):
                        row_o = ((id_ * SD + sd) * OH + ih * SH) * OW32
                        for sh in tl.static_range(SH):
                            tl.store(
                                ptr_o32
                                + base_o
                                + row_o[:, None]
                                + sh * OW32
                                + iw[None, :],
                                w,
                            )
                else:
                    mh = ih < IH
                    mw = iw < IW
                    x = tl.load(
                        ptr_i + base_i + ri[:, None] + iw[None, :],
                        mask=mh[:, None] & mw[None, :],
                        other=0.0,
                    )
                    xi = x.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
                    w = xi | (xi << 16)
                    for sd in tl.static_range(SD):
                        row_o = ((id_ * SD + sd) * OH + ih * SH) * OW32
                        for sh in tl.static_range(SH):
                            tl.store(
                                ptr_o32
                                + base_o
                                + row_o[:, None]
                                + sh * OW32
                                + iw[None, :],
                                w,
                                mask=mh[:, None] & mw[None, :],
                            )
    else:
        for nc in range(nc0, NC, NCZ):
            base_i = nc.to(tl.int64) * (ID * IH * IW)
            base_o = nc.to(tl.int64) * (OD * OH * OW32)
            for t in tl.static_range(NTP):
                tw = pid_g * NTP + t
                if tw < GW:
                    iw = tw * BW + c
                    if EVEN:
                        x = tl.load(ptr_i + base_i + ri[:, None] + iw[None, :])
                        xi = x.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
                        w = xi | (xi << 16)
                        for sd in tl.static_range(SD):
                            row_o = ((id_ * SD + sd) * OH + ih * SH) * OW32
                            for sh in tl.static_range(SH):
                                tl.store(
                                    ptr_o32
                                    + base_o
                                    + row_o[:, None]
                                    + sh * OW32
                                    + iw[None, :],
                                    w,
                                )
                    else:
                        mh = ih < IH
                        mw = iw < IW
                        x = tl.load(
                            ptr_i + base_i + ri[:, None] + iw[None, :],
                            mask=mh[:, None] & mw[None, :],
                            other=0.0,
                        )
                        xi = x.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
                        w = xi | (xi << 16)
                        for sd in tl.static_range(SD):
                            row_o = ((id_ * SD + sd) * OH + ih * SH) * OW32
                            for sh in tl.static_range(SH):
                                tl.store(
                                    ptr_o32
                                    + base_o
                                    + row_o[:, None]
                                    + sh * OW32
                                    + iw[None, :],
                                    w,
                                    mask=mh[:, None] & mw[None, :],
                                )


_POW2_SCALE_MAX = 16
_SD_SH_STORE_MAX = 16
_GRID_Y_MAX = 255
_TILE_LANE_CAP = 32768
_ELEM_TILE_LANES = 16384
_NW1_TILE_LANES = 4096
_NTP = 4
_NCS_PER_PROG = 32
_MIN_PROGRAMS = 48


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


def _fit_tile(ID, IH, BH, WLANS):
    while BH > 1 and (
        ID * triton.cdiv(IH, BH) > _GRID_Y_MAX or BH * WLANS > _TILE_LANE_CAP
    ):
        BH //= 2
    if ID * triton.cdiv(IH, BH) > _GRID_Y_MAX or BH * WLANS > _TILE_LANE_CAP:
        return None
    return BH


def _launch_pow2_fastpath(
    output,
    input,
    OD,
    OH,
    OW,
    reciprocal_scale_d,
    reciprocal_scale_h,
    reciprocal_scale_w,
):
    N, C, ID, IH, IW = input.shape
    NC = N * C
    if NC <= 0 or ID <= 0 or IH <= 0 or IW <= 0 or OD <= 0 or OH <= 0 or OW <= 0:
        return True
    if not input.is_contiguous():
        return False
    if ID * IH * IW >= 2**31 or OD * OH * OW >= 2**31:
        return False
    log2_sd = _pow2_log_scale(OD, ID, reciprocal_scale_d)
    log2_sh = _pow2_log_scale(OH, IH, reciprocal_scale_h)
    log2_sw = _pow2_log_scale(OW, IW, reciprocal_scale_w)
    if log2_sd is None or log2_sh is None or log2_sw is None:
        return False
    if (1 << log2_sd) * (1 << log2_sh) > _SD_SH_STORE_MAX:
        return False
    dtype = input.dtype
    esize = input.element_size()
    with torch_device_fn.device(input.device):
        if esize == 2 and log2_sw == 1 and dtype in (torch.float16, torch.bfloat16):
            BW = min(256, _pow2_floor(IW))
            BH = min(64, _pow2_floor(IH))
            BH = _fit_tile(ID, IH, BH, BW)
            if BH is None:
                return False
            num_warps = 1 if BH * BW <= _NW1_TILE_LANES else 2
            OW32 = OW // 2
            ghh = triton.cdiv(IH, BH)
            gw = triton.cdiv(IW, BW)
            ncz = min(NC, _NCS_PER_PROG)
            even = (IH % BH == 0) and (IW % BW == 0)
            grid = (triton.cdiv(gw, _NTP), ID * ghh, ncz)
            upsample_nearest3d_pow2_pack16_kernel[grid](
                output.view(torch.int32),
                input,
                ID,
                IH,
                IW,
                OD,
                OH,
                OW32,
                NC,
                ghh,
                gw,
                log2_sd,
                log2_sh,
                BH,
                BW,
                _NTP,
                ncz,
                ncz >= NC,
                even,
                num_warps=num_warps,
            )
            return True
        if esize in (2, 4) and dtype in (torch.float16, torch.bfloat16, torch.float32):
            TOW = min(512, _pow2_floor(OW))
            BH = min(64, _pow2_floor(IH), _ELEM_TILE_LANES // TOW)
            while (
                BH > 1
                and NC * ID * triton.cdiv(IH, BH) * triton.cdiv(OW, TOW) < _MIN_PROGRAMS
            ):
                BH //= 2
            BH = _fit_tile(ID, IH, BH, TOW)
            if BH is None:
                return False
            num_warps = 1 if BH * TOW <= _NW1_TILE_LANES else 2
            ghh = triton.cdiv(IH, BH)
            gw = triton.cdiv(OW, TOW)
            ncz = min(NC, _NCS_PER_PROG)
            even = (IH % BH == 0) and (OW % TOW == 0)
            grid = (triton.cdiv(gw, _NTP), ID * ghh, ncz)
            upsample_nearest3d_pow2_element_kernel[grid](
                output,
                input,
                ID,
                IH,
                IW,
                OD,
                OH,
                OW,
                NC,
                ghh,
                gw,
                log2_sd,
                log2_sh,
                log2_sw,
                BH,
                TOW,
                _NTP,
                ncz,
                ncz >= NC,
                even,
                num_warps=num_warps,
            )
            return True
    return False


def upsample_nearest3d(
    input: torch.Tensor,
    output_size: Tuple[int, int, int],
    scales_d: Optional[float] = None,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME UPSAMPLE_NEAREST3D")
    assert input.device.type == device
    assert input.ndim == 5, "The ndim of input must be 5"

    OD, OH, OW = output_size
    N, C, ID, IH, IW = input.shape

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

    if not _launch_pow2_fastpath(
        output,
        input,
        OD,
        OH,
        OW,
        reciprocal_scale_d,
        reciprocal_scale_h,
        reciprocal_scale_w,
    ):
        total_threads = OD * OH * OW
        grid = lambda meta: (
            triton.cdiv(total_threads, meta["BLOCK_SIZE"]),
            triton.cdiv(N * C, 4),
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
