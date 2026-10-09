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

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


def _upsample_nearest_exact2d_backward_kernel(
    grad_out_ptr,
    out_ptr,
    H_in,
    W_in,
    H_out,
    W_out,
    rheight,
    rwidth,
    total,
    MAX_H: tl.constexpr,
    MAX_W: tl.constexpr,
    BLOCK: tl.constexpr,
    USE_F64: tl.constexpr,
    USE_I32: tl.constexpr,
):
    pid = tl.program_id(0)
    if USE_I32:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < total
        w = offs % W_in
        t = offs // W_in
        h = t % H_in
        nc = t // H_in
    else:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        mask = offs < total
        w = offs % W_in
        t = offs // W_in
        h = t % H_in
        nc = t // H_in

    if USE_F64:
        rh = rheight
        rw = rwidth
        hf = h.to(tl.float64)
        wf = w.to(tl.float64)
        lo_h = tl.ceil(hf * rh - 0.5).to(tl.int64)
        hi_h = tl.ceil((hf + 1.0) * rh - 0.5).to(tl.int64)
        lo_w = tl.ceil(wf * rw - 0.5).to(tl.int64)
        hi_w = tl.ceil((wf + 1.0) * rw - 0.5).to(tl.int64)
        acc = tl.zeros([BLOCK], dtype=tl.float64)
    else:
        rh = rheight.to(tl.float32)
        rw = rwidth.to(tl.float32)
        hf = h.to(tl.float32)
        wf = w.to(tl.float32)
        lo_h = tl.ceil(hf * rh - 0.5).to(tl.int64)
        hi_h = tl.ceil((hf + 1.0) * rh - 0.5).to(tl.int64)
        lo_w = tl.ceil(wf * rw - 0.5).to(tl.int64)
        hi_w = tl.ceil((wf + 1.0) * rw - 0.5).to(tl.int64)
        acc = tl.zeros([BLOCK], dtype=tl.float32)

    base = grad_out_ptr + nc * (H_out * W_out) + lo_h * W_out + lo_w
    wlen = hi_w - lo_w
    hlen = hi_h - lo_h
    for dh in range(0, MAX_H):
        dh_ok = (dh < hlen) & mask
        for dw in range(0, MAX_W):
            m = (dw < wlen) & dh_ok
            val = tl.load(base + dh * W_out + dw, mask=m, other=0.0)
            acc += val.to(acc.dtype)

    out_off = nc * (H_in * W_in) + h * W_in + w
    tl.store(out_ptr + out_off, acc.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _upsample_nearest_exact2d_backward_int_kernel(
    grad_out_ptr,
    out_ptr,
    H_in,
    W_in,
    H_out,
    W_out,
    total,
    RH: tl.constexpr,
    RW: tl.constexpr,
    BLOCK: tl.constexpr,
    USE_F64: tl.constexpr,
    USE_I32: tl.constexpr,
):
    # Integer-ratio specialization: ranges are exactly RH x RW wide, so the
    # per-pixel output patch is loaded as contiguous (BLOCK, RW) 2D blocks
    # (fully coalesced) and reduced along the RW axis.
    pid = tl.program_id(0)
    if USE_I32:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < total
        w = offs % W_in
        t = offs // W_in
        h = t % H_in
        nc = t // H_in
    else:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        mask = offs < total
        w = offs % W_in
        t = offs // W_in
        h = t % H_in
        nc = t // H_in

    lo_h = h * RH
    lo_w = w * RW
    if USE_F64:
        acc = tl.zeros([BLOCK], dtype=tl.float64)
    else:
        acc = tl.zeros([BLOCK], dtype=tl.float32)

    dw = tl.arange(0, RW)
    for dh in range(0, RH):
        row_off = nc * (H_out * W_out) + (lo_h + dh) * W_out + lo_w
        vals = tl.load(
            grad_out_ptr + row_off[:, None] + dw[None, :],
            mask=mask[:, None],
            other=0.0,
        )
        acc += tl.sum(vals.to(acc.dtype), axis=1)

    out_off = nc * (H_in * W_in) + h * W_in + w
    tl.store(out_ptr + out_off, acc.to(out_ptr.dtype.element_ty), mask=mask)


def _upsample_nearest_exact2d_backward(
    grad_output, output_size, input_size, scales_h=None, scales_w=None
):
    H_out, W_out = int(output_size[0]), int(output_size[1])
    N, C, H_in, W_in = (int(x) for x in input_size)

    out = torch.empty(
        (N, C, H_in, W_in), device=grad_output.device, dtype=grad_output.dtype
    )

    rh = float(scales_h) if (scales_h is not None and scales_h > 0) else (H_out / H_in)
    rw = float(scales_w) if (scales_w is not None and scales_w > 0) else (W_out / W_in)

    total = N * C * H_in * W_in
    if total == 0:
        return out

    use_f64 = grad_output.dtype == torch.float64
    use_i32 = max(total, N * C * H_out * W_out) < (1 << 31)
    BLOCK = 256
    grid = (triton.cdiv(total, BLOCK),)
    rh_int = float(rh).is_integer() and rh >= 1.0
    rw_int = float(rw).is_integer() and rw >= 1.0
    if rh_int and rw_int:
        _upsample_nearest_exact2d_backward_int_kernel[grid](
            grad_output,
            out,
            H_in,
            W_in,
            H_out,
            W_out,
            total,
            RH=int(rh),
            RW=int(rw),
            BLOCK=BLOCK,
            USE_F64=use_f64,
            USE_I32=use_i32,
        )
    else:
        # Tight static loop bounds: the per-pixel output range is at most
        # ceil(r) for representable r; reserve one extra iteration only for
        # non-integer ratios where f32 rounding can nudge a boundary.
        if float(rh).is_integer():
            max_h = max(1, int(rh))
        else:
            max_h = max(1, int(math.ceil(rh)) + 1)
        if float(rw).is_integer():
            max_w = max(1, int(rw))
        else:
            max_w = max(1, int(math.ceil(rw)) + 1)
        _upsample_nearest_exact2d_backward_kernel[grid](
            grad_output,
            out,
            H_in,
            W_in,
            H_out,
            W_out,
            rh,
            rw,
            total,
            MAX_H=max_h,
            MAX_W=max_w,
            BLOCK=BLOCK,
            USE_F64=use_f64,
            USE_I32=use_i32,
        )
    return out
