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
"""Kunlunxin XPU3 override for ``aten::_adaptive_avg_pool3d`` (forward).

The generic implementation launches **one program per output element** and runs
three nested scalar ``while`` loops (D/H/W) with a scalar accumulator: tens of
thousands of one-lane programs each doing serial gathers.  Measured on XPU3 it
runs at ~0.001-0.03x of the native torch baseline (pre-profiling: the single
``adaptive_avg_pool3d_kernel`` = 10311 ms / 99.98% of the run).

Adaptive average pooling is **separable**:

    out[o] = (1/(dd*hh*ww)) * sum_{d,h,w in window} x
           = (1/dd) sum_d (1/hh) sum_h (1/ww) sum_w x

so it equals three independent 1-D windowed sums along W, then H, then D.  This
module runs them as three flat 1-D-tile kernels (one lane per partial-sum
element, a bounded ``static_range`` window loop accumulates in fp32), plus a
single-pass contiguous plane mean for ``output_size == (1, 1, 1)``.  Compared to
the generic kernel:

  * the launch grid collapses to ``cdiv(numel, BLOCK)`` fat programs instead of
    one program per output;
  * each pass reduces a **single** axis, so the per-lane window is
    ``cdiv(in, out) + 1`` taps (<= ~11 for every benchmarked shape), never the
    3-D window product (up to 1000) -- this is what keeps the static unroll both
    cheap to compile and fast;
  * memory traffic stays ~input-size per pass instead of re-gathering the whole
    3-D box per output.

bf16 detail (same XPU3 trap as the 2-D override): strided / per-lane gather
bf16 loads mis-map lanes, so the W pass loads the raw int16 bits and
reconstructs fp32 via ``(bits & 0xFFFF) << 16``.  Contiguous loads (plane mean)
use the native bf16 pointer.  Intermediates are fp32 and the window area is
divided exactly once at the end, matching ATen's fp32 accumulation.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

BLOCK = 1024
# Plane-mean vector width; bounded like the 2-D override (the tile is reduced
# with tl.sum, whose safe tile width is capped by the XPU buffer tuner).
PLANE_VEC_MAX = 2048
# Target elements per plane-partial program: small enough that even tiny plane
# counts (e.g. 3) launch ~hw/8192 CTAs, large enough to amortise launch.
PLANE_TARGET_CHUNK = 8192
# A runtime `range` window loop carrying a 1-D vector accumulator trips
# `TritonXPUUnrollControl OutOfResources: uni_sram` on XPU3.  A fully unrolled
# `tl.static_range` over the (single-axis) max window is the proven-safe form
# (cf. the 2-D flat kernels, capped at 64 taps).  Above this cap we fall back to
# a scalar per-output kernel (correct, not fast) so arbitrary shapes still work.
STATIC_CAP = 64


@triton.jit
def _load_f32(ptr, mask, BF16: tl.constexpr):
    # bf16 strided/per-lane gather loads on XPU3 run through a vmerge float
    # de-interleave that mis-maps lanes; loading the raw int16 bits and
    # reconstructing fp32 via (bits << 16) is bit-exact and avoids that path.
    if BF16:
        raw = tl.load(ptr, mask=mask, other=0)
        return ((raw.to(tl.int32) & 0xFFFF) << 16).to(tl.float32, bitcast=True)
    return tl.load(ptr, mask=mask, other=0.0).to(tl.float32)


@libentry()
@triton.jit
def _aap3d_plane_fused_kernel(
    input,
    output,
    HW,
    CHUNK: tl.constexpr,
    VEC: tl.constexpr,
    ITERS: tl.constexpr,
    EVEN: tl.constexpr,
):
    # output_size == (1, 1, 1) with one program per plane (splits == 1).
    pid = tl.program_id(0)
    base = input + pid * HW
    acc = tl.zeros((VEC,), dtype=tl.float32)
    if EVEN:
        for it in tl.static_range(ITERS):
            idx = it * VEC + tl.arange(0, VEC)
            acc += tl.load(base + idx).to(tl.float32)
    else:
        for it in tl.static_range(ITERS):
            idx = it * VEC + tl.arange(0, VEC)
            acc += tl.load(base + idx, mask=idx < CHUNK, other=0.0).to(tl.float32)
    tl.store(output + pid, tl.sum(acc) / HW)


@libentry()
@triton.jit
def _aap3d_plane_partial_kernel(
    input,
    partial,
    HW,
    CHUNK: tl.constexpr,
    SPLITS: tl.constexpr,
    VEC: tl.constexpr,
    ITERS: tl.constexpr,
    EVEN: tl.constexpr,
):
    # output_size == (1, 1, 1): average each (N, C) plane over its D*H*W extent.
    # Stage 1: every program sums one contiguous CHUNK of one plane, so the grid
    # is planes*SPLITS (enough CTAs to fill the machine even when planes is tiny)
    # and each load is a coalesced VEC-wide stream.  EVEN (chunk divides the
    # plane and VEC divides chunk) drops the per-lane mask, which lets the XPU
    # vectorizer emit full-width block loads.
    pid = tl.program_id(0)
    plane = pid // SPLITS
    sp = pid % SPLITS
    start = sp * CHUNK
    base = input + plane * HW + start
    acc = tl.zeros((VEC,), dtype=tl.float32)
    if EVEN:
        for it in tl.static_range(ITERS):
            idx = it * VEC + tl.arange(0, VEC)
            acc += tl.load(base + idx).to(tl.float32)
    else:
        n = tl.minimum(start + CHUNK, HW) - start
        for it in tl.static_range(ITERS):
            idx = it * VEC + tl.arange(0, VEC)
            acc += tl.load(base + idx, mask=idx < n, other=0.0).to(tl.float32)
    tl.store(partial + pid, tl.sum(acc))


@libentry()
@triton.jit
def _aap3d_plane_reduce_kernel(
    partial,
    output,
    HW,
    SPLITS: tl.constexpr,
    VEC: tl.constexpr,
    ITERS: tl.constexpr,
):
    plane = tl.program_id(0)
    base = partial + plane * SPLITS
    acc = tl.zeros((VEC,), dtype=tl.float32)
    for it in tl.static_range(ITERS):
        idx = it * VEC + tl.arange(0, VEC)
        acc += tl.load(base + idx, mask=idx < SPLITS, other=0.0)
    tl.store(output + plane, tl.sum(acc) / HW)


@libentry()
@triton.jit
def _aap3d_pass_w_kernel(
    input,
    output,
    total,
    IW: tl.constexpr,
    OW: tl.constexpr,
    MAXW: tl.constexpr,
    BLOCK: tl.constexpr,
    BF16: tl.constexpr,
):
    # Pass 1: sum each row (fixed n,c,d,h) over its W window -> t1[N*C*D*H, OW].
    pid = tl.program_id(0)
    q = pid * BLOCK + tl.arange(0, BLOCK)
    mask = q < total
    ow = q % OW
    row = q // OW
    w0 = (ow * IW) // OW
    w1 = ((ow + 1) * IW + OW - 1) // OW
    base = row * IW
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in tl.static_range(MAXW):
        wi = w0 + k
        active = mask & (wi < w1)
        safe = tl.minimum(wi, w1 - 1)
        v = _load_f32(input + base + safe, mask, BF16)
        acc += tl.where(active, v, 0.0)
    tl.store(output + q, acc, mask=mask)


@libentry()
@triton.jit
def _aap3d_pass_h_kernel(
    input,
    output,
    total,
    IH: tl.constexpr,
    OW: tl.constexpr,
    OH: tl.constexpr,
    MAXH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Pass 2: sum t1[N*C*D, IH, OW] over its H window -> t2[N*C*D, OH, OW].
    pid = tl.program_id(0)
    q = pid * BLOCK + tl.arange(0, BLOCK)
    mask = q < total
    wo = q % OW
    t = q // OW
    ho = t % OH
    ncd = t // OH
    h0 = (ho * IH) // OH
    h1 = ((ho + 1) * IH + OH - 1) // OH
    base = ncd * (IH * OW) + wo
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in tl.static_range(MAXH):
        hi = h0 + k
        active = mask & (hi < h1)
        safe = tl.minimum(hi, h1 - 1)
        acc += tl.where(
            active, tl.load(input + base + safe * OW, mask=mask, other=0.0), 0.0
        )
    tl.store(output + q, acc, mask=mask)


@libentry()
@triton.jit
def _aap3d_pass_d_kernel(
    input,
    output,
    total,
    ID: tl.constexpr,
    OD: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    MAXD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Pass 3: sum t2[N*C, ID, OH, OW] over its D window, then divide by the full
    # 3-D window volume and store into ``output`` (already the final dtype).
    pid = tl.program_id(0)
    q = pid * BLOCK + tl.arange(0, BLOCK)
    mask = q < total
    wo = q % OW
    t = q // OW
    ho = t % OH
    t2 = t // OH
    od = t2 % OD
    nc = t2 // OD
    d0 = (od * ID) // OD
    d1 = ((od + 1) * ID + OD - 1) // OD
    # t2 plane stride is OH*OW; the per-nc base is nc * ID * OH * OW.
    base = nc * (ID * OH * OW) + ho * OW + wo
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in tl.static_range(MAXD):
        di = d0 + k
        active = mask & (di < d1)
        safe = tl.minimum(di, d1 - 1)
        acc += tl.where(
            active,
            tl.load(input + base + safe * (OH * OW), mask=mask, other=0.0),
            0.0,
        )

    # Control-flow-independent per-lane true window volume (sum/area matches ATen).
    w0 = (wo * IW) // OW
    w1 = ((wo + 1) * IW + OW - 1) // OW
    h0 = (ho * IH) // OH
    h1 = ((ho + 1) * IH + OH - 1) // OH
    area = (d1 - d0) * (h1 - h0) * (w1 - w0)
    tl.store(
        output + q,
        acc / tl.cast(area, tl.float32),
        mask=mask,
    )


@libentry()
@triton.jit
def _aap3d_scalar_kernel(
    input,
    output,
    ID: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    OD: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    BF16: tl.constexpr,
):
    # Correctness fallback for windows larger than STATIC_CAP.  One program per
    # output element, scalar triple loop, contiguous 5-D layout.  Not on any
    # benchmarked path.
    pid = tl.program_id(0)
    wo = pid % OW
    t = pid // OW
    ho = t % OH
    t2 = t // OH
    od = t2 % OD
    nc = t2 // OD
    d0 = (od * ID) // OD
    d1 = ((od + 1) * ID + OD - 1) // OD
    h0 = (ho * IH) // OH
    h1 = ((ho + 1) * IH + OH - 1) // OH
    w0 = (wo * IW) // OW
    w1 = ((wo + 1) * IW + OW - 1) // OW
    plane = nc * (ID * IH * IW)
    acc = tl.zeros((), dtype=tl.float32)
    di = d0
    while di < d1:
        hi = h0
        while hi < h1:
            wi = w0
            while wi < w1:
                off = plane + di * (IH * IW) + hi * IW + wi
                if BF16:
                    raw = tl.load(input + off)
                    v = ((raw.to(tl.int32) & 0xFFFF) << 16).to(
                        tl.float32, bitcast=True
                    )
                else:
                    v = tl.load(input + off).to(tl.float32)
                acc += v
                wi += 1
            hi += 1
        di += 1
    area = (d1 - d0) * (h1 - h0) * (w1 - w0)
    tl.store(output + pid, acc / tl.cast(area, tl.float32))


def _normalize_output_size_3d(output_size):
    if isinstance(output_size, torch.Size):
        output_size = tuple(output_size)
    if isinstance(output_size, (list, tuple)):
        if len(output_size) != 3:
            raise ValueError(
                "output_size for _adaptive_avg_pool3d must have 3 elements (D_out, H_out, W_out)"
            )
        return tuple(int(x) for x in output_size)
    raise TypeError("output_size must be a sequence of three integers")


def _prepare_5d_input(t):
    if t.dim() == 5:
        return t, False
    if t.dim() == 4:
        return t.unsqueeze(0), True  # add N=1
    raise ValueError(
        "input for _adaptive_avg_pool3d must be 4D (C,D,H,W) or 5D (N,C,D,H,W)"
    )


def _cdiv(a, b):
    return (a + b - 1) // b


def adaptive_avg_pool3d(input: torch.Tensor, output_size):
    logger.debug("GEMS_KUNLUNXIN ADAPTIVE_AVG_POOL3D")
    x5d, squeezed = _prepare_5d_input(input)
    D_out, H_out, W_out = _normalize_output_size_3d(output_size)

    N, C, D_in, H_in, W_in = x5d.shape
    out5d = torch.empty(
        (N, C, D_out, H_out, W_out),
        dtype=input.dtype,
        device=input.device,
        layout=input.layout,
    )
    if out5d.numel() == 0:
        return out5d.squeeze(0) if squeezed else out5d

    # Layout normalisation only: no-op for the contiguous test/benchmark inputs.
    # Under use_gems() the (rare) copy dispatches to the gems copy kernel.
    x5d = x5d.contiguous()
    bf16 = out5d.dtype == torch.bfloat16

    with torch_device_fn.device(x5d.device):
        # Adaptive windows overlap: safe upper bound on the true max window.
        maxd = _cdiv(D_in, D_out) + 1
        maxh = _cdiv(H_in, H_out) + 1
        maxw = _cdiv(W_in, W_out) + 1
        if D_out == 1 and H_out == 1 and W_out == 1:
            planes = N * C
            hw = D_in * H_in * W_in
            # Split each plane into ~PLANE_TARGET_CHUNK pieces so the grid fills
            # the machine even for tiny plane counts (e.g. 3).
            splits = max(1, _cdiv(hw, PLANE_TARGET_CHUNK))
            chunk = _cdiv(hw, splits)
            pvec = max(64, min(PLANE_VEC_MAX, triton.next_power_of_2(chunk)))
            piters = _cdiv(chunk, pvec)
            even = (hw % chunk == 0) and (chunk % pvec == 0)
            if splits == 1:
                _aap3d_plane_fused_kernel[(planes,)](
                    x5d.view(-1),
                    out5d.view(-1),
                    hw,
                    CHUNK=chunk,
                    VEC=pvec,
                    ITERS=piters,
                    EVEN=even,
                )
            else:
                partial = torch.empty(
                    (planes * splits,), dtype=torch.float32, device=x5d.device
                )
                _aap3d_plane_partial_kernel[(planes * splits,)](
                    x5d.view(-1),
                    partial,
                    hw,
                    CHUNK=chunk,
                    SPLITS=splits,
                    VEC=pvec,
                    ITERS=piters,
                    EVEN=even,
                )
                rvec = max(64, min(PLANE_VEC_MAX, triton.next_power_of_2(splits)))
                _aap3d_plane_reduce_kernel[(planes,)](
                    partial,
                    out5d.view(-1),
                    hw,
                    SPLITS=splits,
                    VEC=rvec,
                    ITERS=_cdiv(splits, rvec),
                )
        elif maxd > STATIC_CAP or maxh > STATIC_CAP or maxw > STATIC_CAP:
            # Extreme windows: keep the fully-unrolled passes off the table.
            w_input = x5d.view(torch.int16) if bf16 else x5d
            _aap3d_scalar_kernel[(N * C * D_out * H_out * W_out,)](
                w_input.view(-1),
                out5d.view(-1),
                ID=D_in,
                IH=H_in,
                IW=W_in,
                OD=D_out,
                OH=H_out,
                OW=W_out,
                BF16=bf16,
            )
        else:
            # bf16 gather loads need the int16-bit reconstruction path.
            w_input = x5d.view(torch.int16) if bf16 else x5d
            rows_w = N * C * D_in * H_in
            total_w = rows_w * W_out
            t1 = torch.empty(
                (rows_w, W_out), dtype=torch.float32, device=x5d.device
            )
            _aap3d_pass_w_kernel[(_cdiv(total_w, BLOCK),)](
                w_input.view(-1),
                t1.view(-1),
                total_w,
                IW=W_in,
                OW=W_out,
                MAXW=maxw,
                BLOCK=BLOCK,
                BF16=bf16,
            )

            total_h = N * C * D_in * H_out * W_out
            t2 = torch.empty(
                (N * C * D_in, H_out, W_out), dtype=torch.float32, device=x5d.device
            )
            _aap3d_pass_h_kernel[(_cdiv(total_h, BLOCK),)](
                t1.view(-1),
                t2.view(-1),
                total_h,
                IH=H_in,
                OW=W_out,
                OH=H_out,
                MAXH=maxh,
                BLOCK=BLOCK,
            )

            total_d = N * C * D_out * H_out * W_out
            _aap3d_pass_d_kernel[(_cdiv(total_d, BLOCK),)](
                t2.view(-1),
                out5d.view(-1),
                total_d,
                ID=D_in,
                OD=D_out,
                IH=H_in,
                IW=W_in,
                OH=H_out,
                OW=W_out,
                MAXD=maxd,
                BLOCK=BLOCK,
            )

    return out5d.squeeze(0) if squeezed else out5d
