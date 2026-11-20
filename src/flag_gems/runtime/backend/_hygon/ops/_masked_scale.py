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

logger = logging.getLogger(__name__)

# Launch geometry: 4 contiguous elements per thread (one 32-bit mask load, one
# 128-bit self load, one 128-bit store) is the measured optimum for the
# bandwidth-bound shapes. For inputs that fit in a single CTA we widen the tile
# so the whole tensor is handled by one program.
_BLOCK = 2048
_SHIFT = 11
_MASK = _BLOCK - 1
_SMALL_BLOCK = 4096
_SMALL_SHIFT = 12
_SMALL_MASK = _SMALL_BLOCK - 1
_WARPS = 8


@triton.jit
def _ms_bulk_kernel(x_ptr, m_ptr, o_ptr, scale, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = tl.load(m_ptr + offs)
    x = tl.load(x_ptr + offs)
    y = tl.where(m != 0, x * scale, 0.0)
    tl.store(o_ptr + offs, y.to(o_ptr.dtype.element_ty))


@triton.jit(do_not_specialize=["n"])
def _ms_flat_kernel(x_ptr, m_ptr, o_ptr, n, scale, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if (pid + 1) * BLOCK <= n:
        m = tl.load(m_ptr + offs)
        x = tl.load(x_ptr + offs)
        y = tl.where(m != 0, x * scale, 0.0)
        tl.store(o_ptr + offs, y.to(o_ptr.dtype.element_ty))
    else:
        inb = offs < n
        m = tl.load(m_ptr + offs, mask=inb, other=0)
        x = tl.load(x_ptr + offs, mask=inb, other=0.0)
        y = tl.where(m != 0, x * scale, 0.0)
        tl.store(o_ptr + offs, y.to(o_ptr.dtype.element_ty), mask=inb)


@triton.jit(do_not_specialize=["n"])
def _ms_strided_kernel(
    x_ptr,
    m_ptr,
    o_ptr,
    n,
    scale,
    xs0,
    xs1,
    xs2,
    xs3,
    ms0,
    ms1,
    ms2,
    ms3,
    d0,
    d1,
    d2,
    d3,
    R: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    inb = offs < n
    rem = offs
    xoff = offs * 0
    moff = offs * 0
    if R >= 4:
        i3 = rem % d3
        rem = rem // d3
        xoff += i3 * xs3
        moff += i3 * ms3
    if R >= 3:
        i2 = rem % d2
        rem = rem // d2
        xoff += i2 * xs2
        moff += i2 * ms2
    if R >= 2:
        i1 = rem % d1
        rem = rem // d1
        xoff += i1 * xs1
        moff += i1 * ms1
    if R >= 1:
        xoff += rem * xs0
        moff += rem * ms0
    m = tl.load(m_ptr + moff, mask=inb, other=0)
    x = tl.load(x_ptr + xoff, mask=inb, other=0.0)
    y = tl.where(m != 0, x * scale, 0.0)
    tl.store(o_ptr + offs, y.to(o_ptr.dtype.element_ty), mask=inb)


_BULK_RUN = _ms_bulk_kernel.run
_FLAT_RUN = _ms_flat_kernel.run


def _aligned_strides(t, bshape):
    ts = t.shape
    st = t.stride()
    nd = len(bshape)
    off = nd - len(ts)
    sts = [0] * nd
    for j in range(len(ts)):
        sts[off + j] = st[j] if ts[j] != 1 else 0
    for j in range(nd):
        if bshape[j] == 1:
            sts[j] = 0
    return sts


def _launch_strided(x, mask, out, scale, bshape):
    n = out.numel()
    if n == 0:
        return
    if len(bshape) > 4:
        for i in range(bshape[0]):
            _launch_strided(x[i], mask[i], out[i], scale, bshape[1:])
        return
    xs = _aligned_strides(x, bshape)
    ms = _aligned_strides(mask, bshape)
    R = len(bshape)
    pad = 4 - R
    xs = xs + [0] * pad
    ms = ms + [0] * pad
    dims = list(bshape) + [1] * pad
    if R == 0:
        _ms_strided_kernel[(1,)](
            x,
            mask,
            out,
            1,
            scale,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            R=0,
            BLOCK=64,
            num_warps=1,
        )
        return
    block = 1024
    grid = ((n + block - 1) // block,)
    _ms_strided_kernel[grid](
        x,
        mask,
        out,
        n,
        scale,
        xs[0],
        xs[1],
        xs[2],
        xs[3],
        ms[0],
        ms[1],
        ms[2],
        ms[3],
        dims[0],
        dims[1],
        dims[2],
        dims[3],
        R=R,
        BLOCK=block,
        num_warps=4,
    )


def _masked_scale(self, mask, scale, *args, **kwargs):
    logger.debug("GEMS_HYGON _MASKED_SCALE")
    x = self
    if kwargs:
        if "input" in kwargs:
            x = kwargs.pop("input")
        if "scale" in kwargs:
            scale = kwargs.pop("scale")
        if "mask" in kwargs:
            mask = kwargs.pop("mask")
    shape = x.shape
    out = x.new_empty(shape)
    n = out.numel()
    if n == 0:
        return out
    if shape == mask.shape and x.is_contiguous() and mask.is_contiguous():
        if n <= _SMALL_BLOCK:
            if n & _SMALL_MASK:
                _FLAT_RUN(
                    x,
                    mask,
                    out,
                    n,
                    float(scale),
                    BLOCK=_SMALL_BLOCK,
                    num_warps=_WARPS,
                    grid=((n + _SMALL_MASK) >> _SMALL_SHIFT,),
                    warmup=False,
                )
            else:
                _BULK_RUN(
                    x,
                    mask,
                    out,
                    float(scale),
                    BLOCK=_SMALL_BLOCK,
                    num_warps=_WARPS,
                    grid=(n >> _SMALL_SHIFT,),
                    warmup=False,
                )
        elif n & _MASK:
            _FLAT_RUN(
                x,
                mask,
                out,
                n,
                float(scale),
                BLOCK=_BLOCK,
                num_warps=_WARPS,
                grid=((n + _MASK) >> _SHIFT,),
                warmup=False,
            )
        else:
            _BULK_RUN(
                x,
                mask,
                out,
                float(scale),
                BLOCK=_BLOCK,
                num_warps=_WARPS,
                grid=(n >> _SHIFT,),
                warmup=False,
            )
        return out
    bshape = torch.broadcast_shapes(shape, mask.shape)
    if bshape != shape:
        out = torch.empty(bshape, dtype=x.dtype, device=x.device)
        if out.numel() == 0:
            return out
    _launch_strided(x, mask, out, float(scale), list(bshape))
    return out
