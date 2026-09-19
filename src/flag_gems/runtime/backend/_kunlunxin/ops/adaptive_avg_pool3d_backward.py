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

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _adaptive_avg_pool3d_backward_exact_kernel(
    grad_output_ptr,
    grad_input_ptr,
    in_d,
    in_h,
    in_w,
    out_d,
    out_h,
    out_w,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    AREA: tl.constexpr,
    n_elems: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Fast path: exact integer ratio (in % out == 0 on every dim).
    # Each input element belongs to exactly one output's pooling region, so
    # grad_input[i] = grad_output[i // K] / (KD*KH*KW).  One load, one store.
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elems

    d = (offsets // (in_h * in_w)) % in_d
    h = (offsets // in_w) % in_h
    w = offsets % in_w
    nc = offsets // (in_d * in_h * in_w)

    # All loads below are provably in-bounds (indices are exact), so only the
    # flat tail mask is needed.
    o_off = (
        nc * (out_d * out_h * out_w)
        + (d // KD) * (out_h * out_w)
        + (h // KH) * out_w
        + (w // KW)
    )
    val = tl.load(grad_output_ptr + o_off, mask=mask)
    tl.store(grad_input_ptr + offsets, val / AREA, mask=mask)


@libentry()
@triton.jit
def _adaptive_avg_pool3d_backward_general_kernel(
    grad_output_ptr,
    grad_input_ptr,
    in_d,
    in_h,
    in_w,
    out_d,
    out_h,
    out_w,
    BLOCK: tl.constexpr,
    MAX_D,
    MAX_H,
    MAX_W: tl.constexpr,
):
    # General path (non-integer ratios, handles upsampling).
    # One program per (n*c, d, h) input row; lanes cover w.  For each input
    # element we enumerate the (few) output positions whose pooling region may
    # contain it: o in [o_min, o_max) per dim, o_max - o_min <= ceil(out/in)+1.
    # Loads are made safe by clamping the candidate to [0, out-1] (never OOB
    # even if the hardware drops the predicate) and the contribution is zeroed
    # with tl.where.  Region bounds are recomputed host-side as constexpr, so
    # the per-dim candidate counts stay small: the runtime od/oh loops keep the
    # static unroll of the innermost (vector) ow loop at MAX_W iterations.
    pid = tl.program_id(0)
    h = pid % in_h
    t = pid // in_h
    d = t % in_d
    nc = t // in_d

    w = tl.arange(0, BLOCK)
    valid = w < in_w

    d_min = (d * out_d) // in_d
    d_max = tl.minimum(((d + 1) * out_d + in_d - 1) // in_d, out_d)
    h_min = (h * out_h) // in_h
    h_max = tl.minimum(((h + 1) * out_h + in_h - 1) // in_h, out_h)
    w_min = (w * out_w) // in_w
    w_max = tl.minimum(((w + 1) * out_w + in_w - 1) // in_w, out_w)

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    gop = grad_output_ptr + nc * (out_d * out_h * out_w)
    for od in range(0, MAX_D):
        o_d = d_min + od
        d_ok = o_d < d_max
        c_d = tl.minimum(o_d, out_d - 1)
        for oh in range(0, MAX_H):
            o_h = h_min + oh
            h_ok = o_h < h_max
            c_h = tl.minimum(o_h, out_h - 1)
            for ow in tl.static_range(0, MAX_W):
                o_w = w_min + ow
                w_ok = o_w < w_max
                active = valid & d_ok & h_ok & w_ok
                c_w = tl.minimum(o_w, out_w - 1)
                # Pooling region of output (c_d, c_h, c_w); for active lanes
                # this equals the region of (o_d, o_h, o_w).
                ds = (c_d * in_d) // out_d
                de = tl.minimum(((c_d + 1) * in_d + out_d - 1) // out_d, in_d)
                hs = (c_h * in_h) // out_h
                he = tl.minimum(((c_h + 1) * in_h + out_h - 1) // out_h, in_h)
                ws = (c_w * in_w) // out_w
                we = tl.minimum(((c_w + 1) * in_w + out_w - 1) // out_w, in_w)
                in_region = (
                    (d >= ds) & (d < de) & (h >= hs) & (h < he) & (w >= ws) & (w < we)
                )
                area = (de - ds) * (he - hs) * (we - ws)
                val = tl.load(gop + c_d * (out_h * out_w) + c_h * out_w + c_w)
                contrib = tl.where(in_region, val, 0.0) / tl.cast(area, tl.float32)
                acc += tl.where(active, contrib, 0.0)

    gip = grad_input_ptr + ((nc * in_d + d) * in_h + h) * in_w
    tl.store(gip + w, acc, mask=valid)


def _adaptive_avg_pool3d_backward(grad_output, input):
    """Gradient of adaptive_avg_pool3d (Kunlunxin/XPU implementation)."""
    logger.debug("GEMS_KUNLUNXIN _ADAPTIVE_AVG_POOL3D_BACKWARD")

    grad_output = grad_output.contiguous()
    input = input.contiguous()
    in_n, in_c, in_d, in_h, in_w = input.shape
    out_d, out_h, out_w = grad_output.shape[-3:]

    grad_input = torch.empty_like(input)
    if grad_output.numel() == 0 or input.numel() == 0:
        return grad_input

    with torch_device_fn.device(input.device):
        if in_d % out_d == 0 and in_h % out_h == 0 and in_w % out_w == 0:
            kd, kh, kw = in_d // out_d, in_h // out_h, in_w // out_w
            n_elems = in_n * in_c * in_d * in_h * in_w
            grid = (triton.cdiv(n_elems, 1024),)
            _adaptive_avg_pool3d_backward_exact_kernel[grid](
                grad_output,
                grad_input,
                in_d,
                in_h,
                in_w,
                out_d,
                out_h,
                out_w,
                KD=kd,
                KH=kh,
                KW=kw,
                AREA=kd * kh * kw,
                n_elems=n_elems,
                BLOCK=1024,
                num_warps=4,
            )
        else:
            block = triton.next_power_of_2(in_w)
            grid = (in_n * in_c * in_d * in_h,)
            _adaptive_avg_pool3d_backward_general_kernel[grid](
                grad_output,
                grad_input,
                in_d,
                in_h,
                in_w,
                out_d,
                out_h,
                out_w,
                BLOCK=block,
                MAX_D=(out_d + in_d - 1) // in_d + 1,
                MAX_H=(out_h + in_h - 1) // in_h + 1,
                MAX_W=(out_w + in_w - 1) // in_w + 1,
                num_warps=4,
            )
    return grad_input
