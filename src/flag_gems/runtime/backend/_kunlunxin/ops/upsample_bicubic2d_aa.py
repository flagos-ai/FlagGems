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

# from flag_gems import runtime
from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)
device = device.name


def configs():
    block = [(bx, by) for bx in (512, 256, 128, 64) for by in (2, 1)]
    warps = [4, 8]
    return [
        triton.Config(
            {
                "BLOCK_X": bs[0],
                "BLOCK_Y": bs[1],
            },
            num_warps=wp,
        )
        for bs in block
        for wp in warps
    ]


def heur_m_block_size(args):
    return triton.next_power_of_2(triton.cdiv(args["OW"], 12))  # cluster_num


def heur_n_block_size(args):
    return 1
    import builtins

    return builtins.min(triton.next_power_of_2(args["OH"]), 8192)


# XPU perf repair 2026-08-17: the upsample fast path (reciprocal scale < 1) was a
# single 25-tap kernel issuing 25 masked gathers per output pixel (~400-510ms per
# official-benchmark case). It is replaced by two separable passes: a horizontal
# bicubic pass over every input row into an (N*C, IH, OW) intermediate, then a
# vertical pass. Per-pixel gathers drop from 25 to 10 and all load masks are
# removed via clamped indices (weights are already zero wherever the old mask
# could clear a load). Fixed bounded dispatch, no @triton.autotune: measured on
# the official 6-case matrix at ~6-44ms/case (see
# harness/solution/performance/upsample_bicubic2d_aa_xpu5_20260817.md).


@triton.jit
def bicubic2d_aa_horz_kernel(
    ptr_t,
    ptr_i,
    OH,
    OW,
    IH,
    IW,
    reciprocal_scale_w,
    BLOCK_W: tl.constexpr,
):
    pid_w = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_nc = tl.program_id(axis=2)
    ow = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    center = (ow + 0.5) * reciprocal_scale_w
    span_start = tl.maximum(center - 2.0 + 0.5, 0).to(tl.int32)
    span_size = (tl.minimum(center + 2.0 + 0.5, IW) - span_start).to(tl.int32)
    start_minus_center = span_start - center
    a = -0.5
    ix0 = tl.minimum(span_start + 0, IW - 1)
    ix1 = tl.minimum(span_start + 1, IW - 1)
    ix2 = tl.minimum(span_start + 2, IW - 1)
    ix3 = tl.minimum(span_start + 3, IW - 1)
    ix4 = tl.minimum(span_start + 4, IW - 1)
    base = (pid_nc * IH + pid_h) * IW
    d0 = tl.load(ptr_i + base + ix0)
    d1 = tl.load(ptr_i + base + ix1)
    d2 = tl.load(ptr_i + base + ix2)
    d3 = tl.load(ptr_i + base + ix3)
    d4 = tl.load(ptr_i + base + ix4)
    y0 = tl.abs((0 + start_minus_center + 0.5) * 1.0)
    w0 = tl.where(0 < span_size, tl.where(y0 < 1.0, ((a + 2) * y0 - (a + 3)) * y0 * y0 + 1, tl.where(y0 < 2.0, (((y0 - 5) * y0 + 8) * y0 - 4) * a, 0)), 0)
    y1 = tl.abs((1 + start_minus_center + 0.5) * 1.0)
    w1 = tl.where(1 < span_size, tl.where(y1 < 1.0, ((a + 2) * y1 - (a + 3)) * y1 * y1 + 1, tl.where(y1 < 2.0, (((y1 - 5) * y1 + 8) * y1 - 4) * a, 0)), 0)
    y2 = tl.abs((2 + start_minus_center + 0.5) * 1.0)
    w2 = tl.where(2 < span_size, tl.where(y2 < 1.0, ((a + 2) * y2 - (a + 3)) * y2 * y2 + 1, tl.where(y2 < 2.0, (((y2 - 5) * y2 + 8) * y2 - 4) * a, 0)), 0)
    y3 = tl.abs((3 + start_minus_center + 0.5) * 1.0)
    w3 = tl.where(3 < span_size, tl.where(y3 < 1.0, ((a + 2) * y3 - (a + 3)) * y3 * y3 + 1, tl.where(y3 < 2.0, (((y3 - 5) * y3 + 8) * y3 - 4) * a, 0)), 0)
    y4 = tl.abs((4 + start_minus_center + 0.5) * 1.0)
    w4 = tl.where(4 < span_size, tl.where(y4 < 1.0, ((a + 2) * y4 - (a + 3)) * y4 * y4 + 1, tl.where(y4 < 2.0, (((y4 - 5) * y4 + 8) * y4 - 4) * a, 0)), 0)
    wt = w0 + w1 + w2 + w3 + w4
    wt = tl.where(wt != 0, wt, 1)
    res = (d0 * w0 + d1 * w1 + d2 * w2 + d3 * w3 + d4 * w4) / wt
    tl.store(ptr_t + (pid_nc * IH + pid_h) * OW + ow, res, mask=ow < OW)


@triton.jit
def bicubic2d_aa_vert_kernel(
    ptr_o,
    ptr_t,
    OH,
    OW,
    IH,
    IW,
    reciprocal_scale_h,
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr,
):
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)
    pid_nc = tl.program_id(axis=2)
    ow = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)
    oh = pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)
    mask = (ow[None, :] < OW) & (oh[:, None] < OH)
    center_h = (oh + 0.5) * reciprocal_scale_h
    span_start = tl.maximum(center_h - 2.0 + 0.5, 0).to(tl.int32)
    span_size = (tl.minimum(center_h + 2.0 + 0.5, IH) - span_start).to(tl.int32)
    start_minus_center = span_start - center_h
    a = -0.5
    iy0 = tl.minimum(span_start + 0, IH - 1)
    iy1 = tl.minimum(span_start + 1, IH - 1)
    iy2 = tl.minimum(span_start + 2, IH - 1)
    iy3 = tl.minimum(span_start + 3, IH - 1)
    iy4 = tl.minimum(span_start + 4, IH - 1)
    r0 = tl.load(ptr_t + (pid_nc * IH + iy0[:, None]) * OW + ow[None, :])
    r1 = tl.load(ptr_t + (pid_nc * IH + iy1[:, None]) * OW + ow[None, :])
    r2 = tl.load(ptr_t + (pid_nc * IH + iy2[:, None]) * OW + ow[None, :])
    r3 = tl.load(ptr_t + (pid_nc * IH + iy3[:, None]) * OW + ow[None, :])
    r4 = tl.load(ptr_t + (pid_nc * IH + iy4[:, None]) * OW + ow[None, :])
    y0v = tl.abs((0 + start_minus_center[:, None] + 0.5) * 1.0)
    w0 = tl.where(0 < span_size[:, None], tl.where(y0v < 1.0, ((a + 2) * y0v - (a + 3)) * y0v * y0v + 1, tl.where(y0v < 2.0, (((y0v - 5) * y0v + 8) * y0v - 4) * a, 0)), 0)
    y1v = tl.abs((1 + start_minus_center[:, None] + 0.5) * 1.0)
    w1 = tl.where(1 < span_size[:, None], tl.where(y1v < 1.0, ((a + 2) * y1v - (a + 3)) * y1v * y1v + 1, tl.where(y1v < 2.0, (((y1v - 5) * y1v + 8) * y1v - 4) * a, 0)), 0)
    y2v = tl.abs((2 + start_minus_center[:, None] + 0.5) * 1.0)
    w2 = tl.where(2 < span_size[:, None], tl.where(y2v < 1.0, ((a + 2) * y2v - (a + 3)) * y2v * y2v + 1, tl.where(y2v < 2.0, (((y2v - 5) * y2v + 8) * y2v - 4) * a, 0)), 0)
    y3v = tl.abs((3 + start_minus_center[:, None] + 0.5) * 1.0)
    w3 = tl.where(3 < span_size[:, None], tl.where(y3v < 1.0, ((a + 2) * y3v - (a + 3)) * y3v * y3v + 1, tl.where(y3v < 2.0, (((y3v - 5) * y3v + 8) * y3v - 4) * a, 0)), 0)
    y4v = tl.abs((4 + start_minus_center[:, None] + 0.5) * 1.0)
    w4 = tl.where(4 < span_size[:, None], tl.where(y4v < 1.0, ((a + 2) * y4v - (a + 3)) * y4v * y4v + 1, tl.where(y4v < 2.0, (((y4v - 5) * y4v + 8) * y4v - 4) * a, 0)), 0)
    wt = w0 + w1 + w2 + w3 + w4
    wt = tl.where(wt != 0, wt, 1)
    res = (r0 * w0 + r1 * w1 + r2 * w2 + r3 * w3 + r4 * w4) / wt
    tl.store(ptr_o + (pid_nc * OH + oh[:, None]) * OW + ow[None, :], res, mask=mask)
# upsample and downsample
# @triton.autotune(
#     configs=runtime.get_tuned_config("upsample_bicubic2d_aa"),
#     key=["N", "C", "OH", "OW"],
# )
@triton.heuristics(
    values={
        "BLOCK_X": heur_m_block_size,
        "BLOCK_Y": heur_n_block_size,
    },
)
@triton.jit
def general_interpolate_bicubic2d_aa_kernel(
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
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr,
):
    pid_x = ext.program_id(axis=0)
    pid_y = ext.program_id(axis=1)
    ow = (pid_x * BLOCK_X + tl.arange(0, BLOCK_X)) % OW
    oh = (pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)) % OH

    if reciprocal_scale_w >= 1.0:
        support_w = 2 * reciprocal_scale_w
    else:
        support_w = 2.0
    if reciprocal_scale_h >= 1.0:
        support_h = 2 * reciprocal_scale_h
    else:
        support_h = 2.0

    interpolate_w = (support_w + 0.5).to(tl.int32) * 2 + 1
    interpolate_h = (support_h + 0.5).to(tl.int32) * 2 + 1

    # _compute_weights_span
    center_w = (ow + 0.5) * reciprocal_scale_w
    center_h = (oh + 0.5) * reciprocal_scale_h
    span_start_w = tl.maximum(center_w - support_w + 0.5, 0).to(tl.int32)
    span_start_h = tl.maximum(center_h - support_h + 0.5, 0).to(tl.int32)
    span_size_w = (tl.minimum(center_w + support_w + 0.5, IW) - span_start_w).to(
        tl.int32
    )
    span_size_h = (tl.minimum(center_h + support_h + 0.5, IH) - span_start_h).to(
        tl.int32
    )

    if reciprocal_scale_w >= 1.0:
        invscale_w = 1.0 / reciprocal_scale_w
    else:
        invscale_w = 1.0
    if reciprocal_scale_h >= 1.0:
        invscale_h = 1.0 / reciprocal_scale_h
    else:
        invscale_h = 1.0
    start_minus_center_w = span_start_w - center_w
    start_minus_center_h = span_start_h - center_h

    a = -0.5
    for n in range(0, N, 1):
        for c in range(0, C, 1):
            offset_base = ((n * C + c) * IH + span_start_h[:, None]) * IW + span_start_w
            weight_y_total = tl.zeros((BLOCK_Y,), dtype=tl.float32)
            result = tl.zeros((BLOCK_Y, BLOCK_X), dtype=tl.float32)
            for y in range(0, interpolate_h, 1):
                wy = tl.abs((y + start_minus_center_h + 0.5) * invscale_h)
                weight_y = tl.where(
                    y < span_size_h,
                    tl.where(
                        wy < 1.0,
                        ((a + 2) * wy - (a + 3)) * wy * wy + 1,
                        tl.where(wy < 2.0, (((wy - 5) * wy + 8) * wy - 4) * a, 0),
                    ),
                    0,
                )
                weight_y_total += weight_y
                weight_x_total = tl.zeros((BLOCK_X,), dtype=tl.float32)
                buffer = tl.zeros((BLOCK_Y, BLOCK_X), dtype=tl.float32)
                for x in range(0, interpolate_w, 1):
                    wx = tl.abs((x + start_minus_center_w + 0.5) * invscale_w)
                    weight_x = tl.where(
                        x < span_size_w,
                        tl.where(
                            wx < 1.0,
                            ((a + 2) * wx - (a + 3)) * wx * wx + 1,
                            tl.where(wx < 2.0, (((wx - 5) * wx + 8) * wx - 4) * a, 0),
                        ),
                        0,
                    )
                    weight_x_total += weight_x
                    data = tl.load(
                        ptr_i + (offset_base + y * IW + x),
                        mask=(span_start_h[:, None] + y < IH)
                        & (span_start_w[None, :] + x < IW),
                        other=0,
                    )
                    buffer += data * weight_x[None, :]
                weight_x_total = tl.where(weight_x_total != 0, weight_x_total, 1)
                result += buffer / weight_x_total[None, :] * weight_y[:, None]
            weight_y_total = tl.where(weight_y_total != 0, weight_y_total, 1)
            result /= weight_y_total[:, None]
            offset_o = ((n * C + c) * OH + oh[:, None]) * OW + ow[None, :]
            tl.store(ptr_o + offset_o, result)


def bicubic_reciprocal_scale(src_size, dst_size, align_corners, scale):
    if align_corners:
        if dst_size > 1:
            return (src_size - 1) / (dst_size - 1)
        else:
            return 0
    else:
        if scale is not None and scale > 0:
            return 1.0 / scale
        else:
            return src_size / dst_size


# https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml#L12547
def _upsample_bicubic2d_aa(
    input: torch.Tensor,
    output_size: Tuple[int],
    align_corners: bool = False,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
):
    logger.debug("GEMS_KUNLUNXIN UPSAMPLE_BICUBIC2D_AA")
    assert input.device.type == device
    assert input.ndim == 4, "The ndim of input must be 4"
    assert len(output_size) == 2, "The len of output_size must be 2"

    OH, OW = output_size
    N, C, IH, IW = input.shape

    reciprocal_scale_h = bicubic_reciprocal_scale(IH, OH, align_corners, scales_h)
    reciprocal_scale_w = bicubic_reciprocal_scale(IW, OW, align_corners, scales_w)

    # allocate output
    output = torch.empty((N, C, OH, OW), device=input.device, dtype=input.dtype)
    if (reciprocal_scale_w >= 1.0) or (reciprocal_scale_h >= 1.0):
        # Downsample / general path: unchanged 2D spatial grid (kernel still
        # loops N*C internally). The masked loads with other=0 require the
        # TRITONXPU_*_SIM env helpers on the XPU backend.
        kernel = general_interpolate_bicubic2d_aa_kernel
        grid = lambda META: (
            triton.cdiv(OW, META["BLOCK_X"]),
            triton.cdiv(OH, META["BLOCK_Y"]),
        )
        import os

        os.environ["TRITONXPU_OTHER_SIM"] = "1"
        os.environ["TRITONXPU_STORE_MASK_SIM"] = "1"
        with torch_device_fn.device(input.device):
            kernel[grid](
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
        if "TRITONXPU_OTHER_SIM" in os.environ:
            del os.environ["TRITONXPU_OTHER_SIM"]
        if "TRITONXPU_STORE_MASK_SIM" in os.environ:
            del os.environ["TRITONXPU_STORE_MASK_SIM"]
    else:
        # Upsample path: two-pass separable bicubic. BLOCK_W covers the whole
        # (pow2-padded) output row so one program handles one input row
        # (horizontal pass) and the vertical pass reuses the intermediate.
        # Fixed bounded dispatch, no @triton.autotune.
        tmp = torch.empty((N, C, IH, OW), device=input.device, dtype=input.dtype)
        block_w = min(triton.next_power_of_2(OW) if OW <= 2048 else 2048, 2048)
        with torch_device_fn.device(input.device):
            bicubic2d_aa_horz_kernel[(triton.cdiv(OW, block_w), IH, N * C)](
                tmp,
                input,
                OH,
                OW,
                IH,
                IW,
                reciprocal_scale_w,
                BLOCK_W=block_w,
                num_warps=8,
            )
            bicubic2d_aa_vert_kernel[
                (triton.cdiv(OW, block_w), triton.cdiv(OH, 2), N * C)
            ](
                output,
                tmp,
                OH,
                OW,
                IH,
                IW,
                reciprocal_scale_h,
                BLOCK_X=block_w,
                BLOCK_Y=2,
                num_warps=8,
            )

    return output
