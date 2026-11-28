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

GRID_Y_LIMIT = 128


@triton.autotune(
    configs=runtime.get_tuned_config("upsample_nearest1d"), key=["N", "C", "OL"]
)
@triton.heuristics(runtime.get_heuristic_config("upsample_nearest1d"))
@triton.jit
def upsample_nearest1d_kernel(
    ptr_o,
    ptr_i,
    N,
    C,
    OL,
    IL,
    reciprocal_scale_l,
    BLOCK_SIZE: tl.constexpr,
    SAME_L: tl.constexpr,
    USE_INT32_IDX: tl.constexpr,
):
    if USE_INT32_IDX:
        pid = tl.program_id(axis=0)
    else:
        pid = tl.program_id(axis=0).to(tl.int64)
    nc_stride = tl.num_programs(axis=1)
    NC = N * C
    nc_iter = tl.program_id(axis=1)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    ol = idx % OL
    if SAME_L:
        il = ol
    else:
        il = tl.minimum(
            tl.math.floor(ol.to(tl.float32) * reciprocal_scale_l).to(tl.int32), IL - 1
        )

    offset_o = nc_iter * OL + ol
    offset_i = nc_iter * IL + il
    if USE_INT32_IDX:
        src_index_stride = nc_stride * IL
        dst_index_stride = nc_stride * OL
    else:
        src_index_stride = nc_stride.to(tl.int64) * IL
        dst_index_stride = nc_stride.to(tl.int64) * OL

    while nc_iter < NC:
        data = tl.load(ptr_i + offset_i)
        tl.store(ptr_o + offset_o, data)
        ptr_i += src_index_stride
        ptr_o += dst_index_stride
        nc_iter += nc_stride


# Fast path for power-of-two integer scales, 4-byte dtypes. Each program
# loads one contiguous input tile once and stores it SCALE times with
# consecutive phase offsets (out[i * SCALE + j] = x[i]). Strided stores beat
# both register-lane expansion (interleave/broadcast+reshape/2D-tile store
# are pathologically slow on this backend, measured 8-25x) and dup-gather
# loads (measured ~10% slower on large shapes).
@triton.jit
def upsample_nearest1d_pow2_dupstore_kernel(
    ptr_o,
    ptr_i,
    IL,
    OL,
    NC,
    GW,
    LOG2_SL: tl.constexpr,
    BW: tl.constexpr,
    NTP: tl.constexpr,
    NCZ: tl.constexpr,
    SINGLE_NC: tl.constexpr,
    EVEN: tl.constexpr,
):
    SCALE: tl.constexpr = 1 << LOG2_SL
    pid_g = tl.program_id(0)
    nc0 = tl.program_id(1)
    c = tl.arange(0, BW)
    if SINGLE_NC:
        base_i = nc0.to(tl.int64) * IL
        base_o = nc0.to(tl.int64) * OL
        for t in tl.static_range(NTP):
            tw = pid_g * NTP + t
            if tw < GW:
                iw = tw * BW + c
                if EVEN:
                    x = tl.load(ptr_i + base_i + iw)
                    for j in tl.static_range(SCALE):
                        tl.store(ptr_o + base_o + iw * SCALE + j, x)
                else:
                    m = iw < IL
                    x = tl.load(ptr_i + base_i + iw, mask=m, other=0)
                    for j in tl.static_range(SCALE):
                        tl.store(
                            ptr_o + base_o + iw * SCALE + j,
                            x,
                            mask=(iw * SCALE + j) < OL,
                        )
    else:
        for nc in range(nc0, NC, NCZ):
            base_i = nc.to(tl.int64) * IL
            base_o = nc.to(tl.int64) * OL
            for t in tl.static_range(NTP):
                tw = pid_g * NTP + t
                if tw < GW:
                    iw = tw * BW + c
                    if EVEN:
                        x = tl.load(ptr_i + base_i + iw)
                        for j in tl.static_range(SCALE):
                            tl.store(ptr_o + base_o + iw * SCALE + j, x)
                    else:
                        m = iw < IL
                        x = tl.load(ptr_i + base_i + iw, mask=m, other=0)
                        for j in tl.static_range(SCALE):
                            tl.store(
                                ptr_o + base_o + iw * SCALE + j,
                                x,
                                mask=(iw * SCALE + j) < OL,
                            )


# Packed fast path for 2-byte dtypes: each output int32 word duplicates one
# input element into both halves ((x << 16) | x), so stores run at 4-byte
# granularity instead of the 2-byte element-rate limit. Word j covers output
# elements (2j, 2j+1), which both map to in[j >> (log2_sl - 1)].
@triton.jit
def upsample_nearest1d_pow2_pack16_kernel(
    ptr_o32,
    ptr_i,
    IL,
    OW32,
    NC,
    GW,
    LOG2_SL: tl.constexpr,
    TW: tl.constexpr,
    NTP: tl.constexpr,
    NCZ: tl.constexpr,
    SINGLE_NC: tl.constexpr,
    EVEN: tl.constexpr,
):
    SHIFTP: tl.constexpr = LOG2_SL - 1
    pid_g = tl.program_id(0)
    nc0 = tl.program_id(1)
    c = tl.arange(0, TW)
    if SINGLE_NC:
        base_i = nc0.to(tl.int64) * IL
        base_o = nc0.to(tl.int64) * OW32
        for t in tl.static_range(NTP):
            tw = pid_g * NTP + t
            if tw < GW:
                w = tw * TW + c
                iw = w >> SHIFTP
                if EVEN:
                    x = tl.load(ptr_i + base_i + iw)
                    xi = x.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
                    tl.store(ptr_o32 + base_o + w, xi | (xi << 16))
                else:
                    m = w < OW32
                    x = tl.load(ptr_i + base_i + iw, mask=m, other=0)
                    xi = x.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
                    tl.store(ptr_o32 + base_o + w, xi | (xi << 16), mask=m)
    else:
        for nc in range(nc0, NC, NCZ):
            base_i = nc.to(tl.int64) * IL
            base_o = nc.to(tl.int64) * OW32
            for t in tl.static_range(NTP):
                tw = pid_g * NTP + t
                if tw < GW:
                    w = tw * TW + c
                    iw = w >> SHIFTP
                    if EVEN:
                        x = tl.load(ptr_i + base_i + iw)
                        xi = x.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
                        tl.store(ptr_o32 + base_o + w, xi | (xi << 16))
                    else:
                        m = w < OW32
                        x = tl.load(ptr_i + base_i + iw, mask=m, other=0)
                        xi = x.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
                        tl.store(ptr_o32 + base_o + w, xi | (xi << 16), mask=m)


_POW2_SCALE_MAX = 16
_NUM_WARPS = 4


def _pow2_floor(value):
    result = 1
    while result * 2 <= value:
        result *= 2
    return result


def _pow2_log_scale(out_size, in_size, reciprocal):
    """Return log2(scale) if out_size == in_size * 2**k (k <= 4) and the
    effective reciprocal scale is exactly 1 / 2**k; otherwise None."""
    if in_size <= 0 or out_size % in_size != 0:
        return None
    scale = out_size // in_size
    if scale < 2 or scale > _POW2_SCALE_MAX or (scale & (scale - 1)) != 0:
        return None
    if reciprocal != 1.0 / scale:
        return None
    return scale.bit_length() - 1


def _launch_pow2_fastpath(output, input, IL, OL, reciprocal_scale_l):
    N, C = input.shape[0], input.shape[1]
    NC = N * C
    if NC <= 0 or OL <= 0:
        return True
    if IL >= 2**31 or OL >= 2**31 or OL > 2**24:
        return False
    log2_sl = _pow2_log_scale(OL, IL, reciprocal_scale_l)
    if log2_sl is None:
        return False
    dtype = input.dtype
    esize = input.element_size()
    if not input.is_contiguous():
        return False
    with torch_device_fn.device(input.device):
        if esize == 2 and dtype in (torch.float16, torch.bfloat16):
            ow32 = OL // 2
            if NC * ow32 <= 4 * 1024 * 1024:
                tw = min(16384, _pow2_floor(ow32)) if ow32 <= 32768 else 8192
                ncz = min(NC, 48)
            else:
                tw = min(32768, _pow2_floor(ow32))
                ncz = min(NC, 128)
            gw = triton.cdiv(ow32, tw)
            grid = (triton.cdiv(gw, 4), ncz)
            upsample_nearest1d_pow2_pack16_kernel[grid](
                output.view(torch.int32),
                input,
                IL,
                ow32,
                NC,
                gw,
                log2_sl,
                tw,
                4,
                ncz,
                NC <= ncz,
                ow32 % tw == 0,
                num_warps=_NUM_WARPS,
            )
            return True
        if esize == 4 and dtype == torch.float32:
            in_total = NC * IL
            if 1024 * 1024 < in_total <= 8 * 1024 * 1024:
                bw = min(16384, _pow2_floor(IL))
            else:
                bw = 8192
            ncz = min(NC, 48)
            bw = min(bw, _pow2_floor(IL))
            gw = triton.cdiv(IL, bw)
            grid = (triton.cdiv(gw, 4), ncz)
            upsample_nearest1d_pow2_dupstore_kernel[grid](
                output,
                input,
                IL,
                OL,
                NC,
                gw,
                log2_sl,
                bw,
                4,
                ncz,
                NC <= ncz,
                IL % bw == 0,
                num_warps=_NUM_WARPS,
            )
            return True
    return False


def upsample_nearest1d(
    input: torch.Tensor,
    output_size: Optional[Tuple[int]] = None,
    scales: Optional[float] = None,
) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME UPSAMPLE_NEAREST1D")
    assert input.device.type == device
    assert input.ndim == 3, "The ndim of input must be 3"
    assert (
        output_size is not None or scales is not None
    ), "Either output_size or scales should be defined."

    OL = output_size[0] if output_size is not None else int(input.shape[2] * scales)
    N, C, IL = input.shape

    if scales is not None:
        reciprocal_scale_l = float(
            torch.tensor(1.0 / scales, dtype=torch.float32).item()
        )
    else:
        # Use float32 division to match PyTorch's behavior
        reciprocal_scale_l = float(
            (
                torch.tensor(IL, dtype=torch.float32)
                / torch.tensor(OL, dtype=torch.float32)
            ).item()
        )

    # allocate output
    output = torch.empty((N, C, OL), device=input.device, dtype=input.dtype)
    if not _launch_pow2_fastpath(output, input, IL, OL, reciprocal_scale_l):
        total_threads = OL
        grid = lambda meta: (
            triton.cdiv(total_threads, meta["BLOCK_SIZE"]),
            min(triton.cdiv(N * C, 4), GRID_Y_LIMIT),
        )

        with torch_device_fn.device(input.device):
            upsample_nearest1d_kernel[grid](
                output,
                input,
                N,
                C,
                OL,
                IL,
                reciprocal_scale_l,
            )
    return output
