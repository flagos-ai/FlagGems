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

"""Triton upsample_nearest3d_backward (exact gather formulation).

grad_input[n, c, d1, h1, w1] is the sum of grad_output over the preimage box
    [S_D(d1), S_D(d1+1)) x [S_H(h1), S_H(h1+1)) x [S_W(w1), S_W(w1+1))
with S_X(s) = min{ i in [0, out_X] : floor(i * r_X) >= s } and
r_X = fp32(1/scale_X) when scale_X > 0, else fp32(in_X) / fp32(out_X).

Two kernels share that contract:

* ``_uniform_kernel`` is used when out_X == in_X * k_X with k_X a power of two
  (and the scale, if given, equals k_X).  Then r_X = 1/k_X is exact in fp32,
  i * r_X is exact, so floor(i/k_X) == i // k_X and the preimage of s is exactly
  [k_X*s, k_X*(s+1)).  The gradient is a k_D x k_H x k_W box sum of contiguous
  grad_output slices: no index search, no masks beyond the lane tail.  Spatial
  dims are compile-time constants, so the per-lane index decomposition lowers
  to multiply-shift instead of runtime integer division.

* ``_upsample_nearest3d_backward_kernel`` is the general fallback.  It inverts
  the ATen map per lane with a branchless integer binary search on the very same
  fp32 multiply that ATen uses, so knife-edge ratios (e.g. 3/147) cannot drift,
  and it folds ATen's min(..., in-1) clamp into the last input index owning the
  whole tail.

Both write every input element exactly once, accumulate in fp32 registers and
need no atomics and no zero-fill.
"""

import logging

import numpy as np
import torch
import triton
import triton.language as tl

from flag_gems.ops.upsample_nearest3d_backward import (
    upsample_nearest3d_backward as default_upsample_nearest3d_backward,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Moore Threads hardware does not support fp64.
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@triton.jit
def _bound(s, r, out_size, iters):
    """Smallest i in [0, out_size] with floor(i*r) >= s (fp32 arithmetic)."""
    lo = s * 0
    hi = s * 0 + out_size
    for _ in range(iters):
        mid = (lo + hi) // 2
        cond = mid.to(tl.float32) * r >= s.to(tl.float32)
        lo = tl.where(cond, lo, mid + 1)
        hi = tl.where(cond, mid, hi)
    return hi


@triton.jit
def _upsample_nearest3d_backward_kernel(
    gout_ptr,
    gin_ptr,
    OD,
    OH,
    OW,
    ID,
    IH,
    IW,
    sn,
    sc,
    C,
    r_d,
    r_h,
    r_w,
    d_iters,
    h_iters,
    w_iters,
    s2,
    s3,
    s4,
    BLOCK: tl.constexpr,
):
    nc = tl.program_id(1)
    pid = tl.program_id(0)

    n = nc // C
    c = nc - n * C
    base = n.to(tl.int64) * sn + c.to(tl.int64) * sc

    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < ID * IH * IW
    iw = idx % IW
    t = idx // IW
    ih = t % IH
    id_ = t // IH

    dlo = _bound(id_, r_d, OD, d_iters)
    hlo = _bound(ih, r_h, OH, h_iters)
    wlo = _bound(iw, r_w, OW, w_iters)
    dhi = tl.where(id_ == ID - 1, OD, _bound(id_ + 1, r_d, OD, d_iters))
    hhi = tl.where(ih == IH - 1, OH, _bound(ih + 1, r_h, OH, h_iters))
    whi = tl.where(iw == IW - 1, OW, _bound(iw + 1, r_w, OW, w_iters))
    dlen = tl.maximum(dhi - dlo, 0)
    hlen = tl.maximum(hhi - hlo, 0)
    wlen = tl.maximum(whi - wlo, 0)
    md = tl.max(dlen)
    mh = tl.max(hlen)
    mw = tl.max(wlen)

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for dd in range(0, md):
        for hh in range(0, mh):
            run = (dlo + dd) * s2 + (hlo + hh) * s3
            for ww in range(0, mw):
                m = mask & (dd < dlen) & (hh < hlen) & (ww < wlen)
                v = tl.load(gout_ptr + base + run + (wlo + ww) * s4, mask=m, other=0.0)
                acc += v.to(tl.float32)

    out_off = nc.to(tl.int64) * (ID * IH * IW) + idx
    tl.store(gin_ptr + out_off, acc, mask=mask)


@triton.jit
def _uniform_kernel(
    gout_ptr,
    gin_ptr,
    sn,
    sc,
    C,
    s2,
    s3,
    s4,
    BLOCK: tl.constexpr,
    K_D: tl.constexpr,
    K_H: tl.constexpr,
    K_W: tl.constexpr,
    ID: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    INVOL: tl.constexpr,
):
    nc = tl.program_id(1)
    pid = tl.program_id(0)

    n = nc // C
    c = nc - n * C
    base = n.to(tl.int64) * sn + c.to(tl.int64) * sc

    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < INVOL
    iw = idx % IW
    t = idx // IW
    ih = t % IH
    id_ = t // IH

    d0 = id_ * K_D
    h0 = ih * K_H
    w0 = iw * K_W

    woffs = tl.arange(0, K_W)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for dd in tl.static_range(K_D):
        for hh in tl.static_range(K_H):
            row = base + (d0 + dd) * s2 + (h0 + hh) * s3 + w0 * s4
            tile = tl.load(
                gout_ptr + row[:, None] + woffs[None, :] * s4,
                mask=mask[:, None],
                other=0.0,
            )
            acc += tl.sum(tile.to(tl.float32), axis=1)

    out_off = nc.to(tl.int64) * INVOL + idx
    tl.store(gin_ptr + out_off, acc, mask=mask)


def _recip(in_size, out_size, scale):
    if scale is not None:
        s = float(scale)
        if s > 0.0:
            return float(np.float32(1.0 / s))
    return float(np.float32(in_size) / np.float32(out_size))


def _nbits(v):
    return max(1, int(v).bit_length())


def _pow2_ratio(in_size, out_size, scale):
    """k when out == in*k with k a power of two and the scale (if any) is k."""
    if in_size <= 0 or out_size <= 0 or out_size % in_size:
        return 0
    k = out_size // in_size
    if k & (k - 1):
        return 0
    if scale is not None and float(scale) != k:
        return 0
    return k


_BLOCK = 128
_NUM_WARPS = 4
_SMALL_BLOCK = 64
_SMALL_WARPS = 2


def _use_triton_kernel(x):
    """Whether ``x`` can be handled by the MUSA kernels."""
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if x.is_complex():
        return False
    return True


def upsample_nearest3d_backward(
    grad_output,
    output_size,
    input_size,
    scales_d=None,
    scales_h=None,
    scales_w=None,
    grad_input=None,
    **kwargs,
):
    logger.debug("GEMS_MTHREADS UPSAMPLE_NEAREST3D_BACKWARD")

    if not _use_triton_kernel(grad_output):
        return default_upsample_nearest3d_backward(
            grad_output,
            output_size,
            input_size,
            scales_d,
            scales_h,
            scales_w,
            grad_input=grad_input,
        )

    with torch_device_fn.device(grad_output.device):
        return _launch_upsample_nearest3d_backward(
            grad_output,
            output_size,
            input_size,
            scales_d,
            scales_h,
            scales_w,
            grad_input,
        )


def _launch_upsample_nearest3d_backward(
    grad_output, output_size, input_size, scales_d, scales_h, scales_w, grad_input
):
    if (
        scales_h is None
        and scales_w is None
        and scales_d is not None
        and not isinstance(scales_d, (int, float))
    ):
        seq = tuple(scales_d)
        if len(seq) == 3:
            scales_d, scales_h, scales_w = seq

    outs = tuple(int(v) for v in output_size)
    ins = tuple(int(v) for v in input_size)
    if len(ins) == 3:
        ins = (int(grad_output.shape[0]), int(grad_output.shape[1])) + ins
    if len(outs) == 5:
        outs = outs[2:]

    N, C, ID, IH, IW = ins[0], ins[1], ins[2], ins[3], ins[4]
    OD, OH, OW = outs[0], outs[1], outs[2]

    dev = grad_output.device
    dt = grad_output.dtype

    if grad_input is None:
        if OD <= 0 or OH <= 0 or OW <= 0:
            return torch.zeros((N, C, ID, IH, IW), device=dev, dtype=dt)
        out = torch.empty((N, C, ID, IH, IW), device=dev, dtype=dt)
    else:
        out = grad_input
        if out.numel() == 0:
            return out
        if OD <= 0 or OH <= 0 or OW <= 0:
            out.zero_()
            return out

    in_vol = ID * IH * IW
    if out.numel() == 0:
        return out

    s0, s1, s2, s3, s4 = (int(v) for v in grad_output.stride())

    kd = _pow2_ratio(ID, OD, scales_d)
    kh = _pow2_ratio(IH, OH, scales_h)
    kw = _pow2_ratio(IW, OW, scales_w)
    if kd and kh and kw:
        # Tiny inputs activate only a handful of CTAs, so shrink the block to
        # spread the same work over more SMs; larger inputs are already
        # byte-proportional at BLOCK=128.
        blk, nw = (
            (_SMALL_BLOCK, _SMALL_WARPS) if in_vol <= 1024 else (_BLOCK, _NUM_WARPS)
        )
        grid = (triton.cdiv(in_vol, blk), N * C)
        _uniform_kernel[grid](
            grad_output,
            out,
            s0,
            s1,
            C,
            s2,
            s3,
            s4,
            BLOCK=blk,
            K_D=kd,
            K_H=kh,
            K_W=kw,
            ID=ID,
            IH=IH,
            IW=IW,
            INVOL=in_vol,
            num_warps=nw,
        )
        return out

    r_d = _recip(ID, OD, scales_d)
    r_h = _recip(IH, OH, scales_h)
    r_w = _recip(IW, OW, scales_w)

    grid = (triton.cdiv(in_vol, _BLOCK), N * C)
    _upsample_nearest3d_backward_kernel[grid](
        grad_output,
        out,
        OD,
        OH,
        OW,
        ID,
        IH,
        IW,
        s0,
        s1,
        C,
        r_d,
        r_h,
        r_w,
        _nbits(OD),
        _nbits(OH),
        _nbits(OW),
        s2,
        s3,
        s4,
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
    )
    return out
