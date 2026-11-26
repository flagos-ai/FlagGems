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

"""Triton implementation of aten::upsample_trilinear3d_backward for this target.

The authoritative Gems pytest compares against ``aten`` on this device, whose
upsample_trilinear3d_backward distributes each output element to exactly ONE
input element with weight 1.0, i.e. a block (nearest-style) resize backward:

    grad_input[i] = sum of grad_output[o] for o in [s(i), s(i+1))

with the half-open block boundaries

    s(i) = ceil_fp32(i * fp32(out/in))

(align_corners does not influence the mapping; the fp32 rounding of the
multiply matters at exact-integer coordinates and is reproduced bit-for-bit).

Because the blocks partition the output range, the whole operator is a
separable block-sum gather: every grad_output element is read exactly once
and every grad_input element is written exactly once, with no atomics.

Two kernels:
  * `_ut3d_bwd_gather_int`: integral scale factors (out/in == 1, 2, 3, ...),
    so block i is exactly [i*c, (i+1)*c) and needs no per-element bounds
    masks.  Each (td, th) tap loads a contiguous (BLOCK, CI_W) tile that is
    reduced in registers; IW is a compile-time constant so the lane decode
    uses no runtime integer division.
  * `_ut3d_bwd_gather`: the general path (non-integral scales) with explicit
    ceil-derived block bounds and masked scalar taps.
"""

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.upsample_trilinear3d_backward import (
    upsample_trilinear3d_backward as default_upsample_trilinear3d_backward,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Moore Threads hardware does not support fp64.
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@triton.jit
def _ut3d_bwd_gather_int(
    go_ptr,
    gi_ptr,
    ID,
    NB,
    OD,
    OH,
    OW,
    IH: tl.constexpr,
    IW: tl.constexpr,
    CI_D: tl.constexpr,
    CI_H: tl.constexpr,
    CI_W: tl.constexpr,
    MASKED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    wb = pid % NB
    t = pid // NB
    i = t % ID
    nc = t // ID

    plane = IH * IW
    offs = wb * BLOCK + tl.arange(0, BLOCK)
    if MASKED:
        m = offs < plane
    j = offs // IW
    k = offs % IW

    base = nc * (OD * OH * OW)
    p0 = i * CI_D
    q = j * CI_H
    w0 = k * CI_W
    cw = tl.arange(0, CI_W)

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for td in tl.static_range(0, CI_D):
        rb = base + (p0 + td) * (OH * OW) + q * OW
        for th in tl.static_range(0, CI_H):
            ptr = go_ptr + (rb + th * OW)[:, None] + (w0[:, None] + cw[None, :])
            if MASKED:
                v = tl.load(ptr, mask=m[:, None], other=0.0)
            else:
                v = tl.load(ptr)
            acc += tl.sum(v.to(tl.float32), axis=1)

    out_ptr = gi_ptr + (nc * (ID * IH * IW) + i * plane) + offs
    if MASKED:
        tl.store(out_ptr, acc.to(gi_ptr.dtype.element_ty), mask=m)
    else:
        tl.store(out_ptr, acc.to(gi_ptr.dtype.element_ty))


@triton.jit
def _ut3d_bwd_gather(
    go_ptr,
    gi_ptr,
    ID,
    IH,
    IW,
    OD,
    OH,
    OW,
    c_d,
    c_h,
    c_w,
    NB,
    BLOCK: tl.constexpr,
    TD: tl.constexpr,
    TH: tl.constexpr,
    TW: tl.constexpr,
):
    pid = tl.program_id(0)
    wb = pid % NB
    t = pid // NB
    i = t % ID
    nc = t // ID

    cd = c_d.to(tl.float32)
    ch = c_h.to(tl.float32)
    cw = c_w.to(tl.float32)

    plane = IH * IW
    offs = wb * BLOCK + tl.arange(0, BLOCK)
    m = offs < plane
    j = offs // IW
    k = offs % IW

    fi = i.to(tl.float32)
    fj = j.to(tl.float32)
    fk = k.to(tl.float32)

    d_lo = tl.math.ceil(fi * cd).to(tl.int32)
    d_hi = tl.math.ceil((fi + 1.0) * cd).to(tl.int32)
    h_lo = tl.math.ceil(fj * ch).to(tl.int32)
    h_hi = tl.math.ceil((fj + 1.0) * ch).to(tl.int32)
    w_lo = tl.math.ceil(fk * cw).to(tl.int32)
    w_hi = tl.math.ceil((fk + 1.0) * cw).to(tl.int32)

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    base_nc = nc * (OD * OH * OW)

    for td in tl.static_range(0, TD):
        p = d_lo + td
        pm = (p < d_hi) & (p < OD)
        for th in tl.static_range(0, TH):
            q = h_lo + th
            qm = pm & (q < h_hi) & (q < OH)
            rb = base_nc + p * (OH * OW) + q * OW
            for tw in tl.static_range(0, TW):
                r = w_lo + tw
                rm = qm & m & (r < w_hi) & (r < OW)
                acc += tl.load(go_ptr + rb + r, mask=rm, other=0.0).to(tl.float32)

    tl.store(
        gi_ptr + (nc * (ID * IH * IW) + i * plane) + offs,
        acc.to(gi_ptr.dtype.element_ty),
        mask=m,
    )


def _taps(ratio):
    # max block width = ceil(c) with c = fp32(out/in); at least 1.
    return max(1, int(math.ceil(ratio)))


def _use_triton_kernel(x):
    """Whether ``x`` can be handled by the MUSA kernels."""
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if x.is_complex():
        return False
    return True


def upsample_trilinear3d_backward(
    grad_output,
    output_size,
    input_size,
    align_corners,
    scales_d=None,
    scales_h=None,
    scales_w=None,
):
    logger.debug("GEMS_MTHREADS UPSAMPLE_TRILINEAR3D_BACKWARD")

    if not _use_triton_kernel(grad_output):
        return default_upsample_trilinear3d_backward(
            grad_output,
            output_size,
            input_size,
            align_corners,
            scales_d,
            scales_h,
            scales_w,
        )

    with torch_device_fn.device(grad_output.device):
        return _launch_trilinear_bwd(
            grad_output,
            output_size,
            input_size,
            align_corners,
            scales_d,
            scales_h,
            scales_w,
        )


def _launch_trilinear_bwd(
    grad_output,
    output_size,
    input_size,
    align_corners,
    scales_d=None,
    scales_h=None,
    scales_w=None,
):
    N, C, ID, IH, IW = (int(v) for v in input_size)
    OD, OH, OW = (int(v) for v in output_size)
    device = grad_output.device
    out_dtype = grad_output.dtype
    out_shape = (N, C, ID, IH, IW)

    if N * C * ID * IH * IW == 0 or N * C * OD * OH * OW == 0:
        return torch.zeros(out_shape, dtype=out_dtype, device=device)

    # fp32 block scales, matching the reference arithmetic exactly.
    c_d = float(OD) / float(ID)
    c_h = float(OH) / float(IH)
    c_w = float(OW) / float(IW)

    out = torch.empty(out_shape, dtype=out_dtype, device=device)

    BLOCK = 256
    plane = IH * IW
    NB = (plane + BLOCK - 1) // BLOCK
    grid = (NB * N * C * ID,)

    integral = (
        c_d >= 1.0
        and c_h >= 1.0
        and c_w >= 1.0
        and float(c_d).is_integer()
        and float(c_h).is_integer()
        and float(c_w).is_integer()
    )
    if integral:
        _ut3d_bwd_gather_int[grid](
            grad_output,
            out,
            ID,
            NB,
            OD,
            OH,
            OW,
            IH=IH,
            IW=IW,
            CI_D=int(c_d),
            CI_H=int(c_h),
            CI_W=int(c_w),
            MASKED=(plane % BLOCK) != 0,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out

    _ut3d_bwd_gather[grid](
        grad_output,
        out,
        ID,
        IH,
        IW,
        OD,
        OH,
        OW,
        c_d,
        c_h,
        c_w,
        NB,
        BLOCK=BLOCK,
        TD=_taps(c_d),
        TH=_taps(c_h),
        TW=_taps(c_w),
        num_warps=4,
    )
    return out
