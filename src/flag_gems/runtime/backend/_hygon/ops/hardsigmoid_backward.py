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

"""Triton implementation of hardsigmoid_backward.

out = (self > -3 && self < 3) ? grad_output * (1/6) : 0
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_I32_MAX = 0x7FFFFFFF


@triton.jit
def _hsb_apply(g, s, F64: tl.constexpr):
    # ATen computes in opmath: float32 for fp16/bf16/fp32 inputs, float64 for fp64.
    if F64:
        gf = g.to(tl.float64)
        sf = s.to(tl.float64)
        # 6.0 is exactly representable, so dividing yields the correctly rounded 1/6.
        return tl.where((sf > -3.0) & (sf < 3.0), gf / 6.0, 0.0)
    else:
        gf = g.to(tl.float32)
        sf = s.to(tl.float32)
        return tl.where((sf > -3.0) & (sf < 3.0), gf * 0.16666666666666666, 0.0)


@triton.jit
def _hsb_flat(
    g_ptr,
    s_ptr,
    o_ptr,
    n_elem,
    BLOCK: tl.constexpr,
    F64: tl.constexpr,
    EVEN: tl.constexpr,
    I64: tl.constexpr,
):
    pid = tl.program_id(0)
    if I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        n = n_elem.to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        n = n_elem
    if EVEN:
        g = tl.load(g_ptr + offs)
        s = tl.load(s_ptr + offs)
        tl.store(o_ptr + offs, _hsb_apply(g, s, F64).to(o_ptr.dtype.element_ty))
    else:
        mask = offs < n
        g = tl.load(g_ptr + offs, mask=mask, other=0.0)
        s = tl.load(s_ptr + offs, mask=mask, other=0.0)
        tl.store(
            o_ptr + offs, _hsb_apply(g, s, F64).to(o_ptr.dtype.element_ty), mask=mask
        )


@triton.jit
def _hsb_strided(
    g_ptr, s_ptr, o_ptr, meta_ptr, rank, n_elem, BLOCK: tl.constexpr, F64: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n_elem.to(tl.int64)
    idx = offs
    goff = tl.zeros([BLOCK], dtype=tl.int64)
    soff = tl.zeros([BLOCK], dtype=tl.int64)
    r = rank
    for i in tl.range(0, r):
        j = r - 1 - i
        sz = tl.load(meta_ptr + j)
        sg = tl.load(meta_ptr + r + j)
        ss = tl.load(meta_ptr + 2 * r + j)
        rem = idx % sz
        idx = idx // sz
        goff += rem * sg
        soff += rem * ss
    g = tl.load(g_ptr + goff, mask=mask, other=0.0)
    s = tl.load(s_ptr + soff, mask=mask, other=0.0)
    tl.store(o_ptr + offs, _hsb_apply(g, s, F64).to(o_ptr.dtype.element_ty), mask=mask)


def _pick(n):
    """Launch configuration for the flat streaming kernel."""
    if n >= 1024:
        return 1024, 4
    block = 64
    while block < n:
        block <<= 1
    return block, max(1, min(4, block // 128))


def _launch_flat(g, s, out, n):
    block, warps = _pick(n)
    grid = ((n + block - 1) // block,)
    _hsb_flat[grid](
        g,
        s,
        out,
        n,
        BLOCK=block,
        F64=(out.dtype == torch.float64),
        EVEN=(n % block == 0),
        I64=(n > _I32_MAX),
        num_warps=warps,
    )


def _compact(bs, gstride, sstride):
    sizes = []
    gst = []
    sst = []
    for d in range(len(bs)):
        sz = bs[d]
        if sz == 1:
            continue
        sizes.append(sz)
        gst.append(gstride[d])
        sst.append(sstride[d])
    i = len(sizes) - 2
    while i >= 0:
        if sizes[i + 1] * gst[i + 1] == gst[i] and sizes[i + 1] * sst[i + 1] == sst[i]:
            sizes[i] *= sizes[i + 1]
            del sizes[i + 1]
            del gst[i + 1]
            del sst[i + 1]
        i -= 1
    return sizes, gst, sst


def _launch_strided(ge, se, out, sizes, gst, sst, n):
    rank = len(sizes)
    meta = torch.tensor(sizes + gst + sst, dtype=torch.int64, device=out.device)
    block = 1024
    warps = 4
    if n < block:
        block = 64
        while block < n:
            block <<= 1
        warps = max(1, min(4, block // 128))
    grid = ((n + block - 1) // block,)
    _hsb_strided[grid](
        ge,
        se,
        out,
        meta,
        rank,
        n,
        BLOCK=block,
        F64=(out.dtype == torch.float64),
        num_warps=warps,
    )


def hardsigmoid_backward(grad_output, self):
    logger.debug("GEMS_HYGON HARDSIGMOID_BACKWARD")
    g = grad_output
    s = self
    gs = g.shape
    ss = s.shape
    out_dtype = g.dtype if g.dtype is s.dtype else torch.result_type(g, s)

    # Fast path: identical shapes, both densely packed in linear order.
    if gs == ss and g.is_contiguous() and s.is_contiguous():
        n = g.numel()
        if out_dtype is g.dtype:
            out = torch.empty_like(g)
        else:
            out = torch.empty(gs, dtype=out_dtype, device=g.device)
        if n == 0:
            return out
        _launch_flat(g, s, out, n)
        return out

    bs = torch.broadcast_shapes(gs, ss)
    out = torch.empty(bs, dtype=out_dtype, device=g.device)
    n = out.numel()
    if n == 0:
        return out

    ge = g.expand(bs)
    se = s.expand(bs)
    sizes, gst, sst = _compact(bs, ge.stride(), se.stride())
    rank = len(sizes)
    if rank == 0 or (rank == 1 and gst[0] == 1 and sst[0] == 1):
        _launch_flat(ge, se, out, n)
        return out
    _launch_strided(ge, se, out, sizes, gst, sst, n)
    return out
