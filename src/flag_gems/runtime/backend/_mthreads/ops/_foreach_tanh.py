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
from triton.language.extra import libdevice

from flag_gems.ops._foreach_unary import _foreach_tanh as default__foreach_tanh

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_FUSE_MAX = 16
_MAX_RANK = 8
_SHARE_MAX_BYTES = 1 << 28
_FUSE_MAX_TENSOR_BYTES = 1 << 30


@triton.jit
def _tanh_body(x_ptr, o_ptr, off, mask):
    z = tl.load(x_ptr + off, mask=mask, other=0.0)
    zf = z.to(tl.float32)
    # tanh(z) = 1 - 2/(exp2(2*log2(e)*z) + 1); saturates correctly to +-1
    # (exp2 overflows to +inf for z > ~44, giving 2/inf = 0), so no clamping.
    y = 1.0 - 2.0 / (tl.exp2(2.8853900817779268 * zf) + 1.0)
    tl.store(o_ptr + off, y.to(o_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _tanh_flat_kernel(x_ptr, o_ptr, n, I64: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    if I64:
        off = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    else:
        off = pid * BLOCK + tl.arange(0, BLOCK)
    _tanh_body(x_ptr, o_ptr, off, off < n)


@triton.jit
def _tanh_fused_shared_kernel(
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    x8,
    x9,
    x10,
    x11,
    x12,
    x13,
    x14,
    x15,
    obase,
    n,
    BLOCK: tl.constexpr,
):
    tile = tl.program_id(0)
    t = tl.program_id(1)
    off = tile * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    o = obase + t.to(tl.int64) * n
    if t == 0:
        _tanh_body(x0, o, off, mask)
    elif t == 1:
        _tanh_body(x1, o, off, mask)
    elif t == 2:
        _tanh_body(x2, o, off, mask)
    elif t == 3:
        _tanh_body(x3, o, off, mask)
    elif t == 4:
        _tanh_body(x4, o, off, mask)
    elif t == 5:
        _tanh_body(x5, o, off, mask)
    elif t == 6:
        _tanh_body(x6, o, off, mask)
    elif t == 7:
        _tanh_body(x7, o, off, mask)
    elif t == 8:
        _tanh_body(x8, o, off, mask)
    elif t == 9:
        _tanh_body(x9, o, off, mask)
    elif t == 10:
        _tanh_body(x10, o, off, mask)
    elif t == 11:
        _tanh_body(x11, o, off, mask)
    elif t == 12:
        _tanh_body(x12, o, off, mask)
    elif t == 13:
        _tanh_body(x13, o, off, mask)
    elif t == 14:
        _tanh_body(x14, o, off, mask)
    else:
        _tanh_body(x15, o, off, mask)


@triton.jit
def _tanh_fused_kernel(
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    x8,
    x9,
    x10,
    x11,
    x12,
    x13,
    x14,
    x15,
    o0,
    o1,
    o2,
    o3,
    o4,
    o5,
    o6,
    o7,
    o8,
    o9,
    o10,
    o11,
    o12,
    o13,
    o14,
    o15,
    n,
    BLOCK: tl.constexpr,
):
    tile = tl.program_id(0)
    t = tl.program_id(1)
    off = tile * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    if t == 0:
        _tanh_body(x0, o0, off, mask)
    elif t == 1:
        _tanh_body(x1, o1, off, mask)
    elif t == 2:
        _tanh_body(x2, o2, off, mask)
    elif t == 3:
        _tanh_body(x3, o3, off, mask)
    elif t == 4:
        _tanh_body(x4, o4, off, mask)
    elif t == 5:
        _tanh_body(x5, o5, off, mask)
    elif t == 6:
        _tanh_body(x6, o6, off, mask)
    elif t == 7:
        _tanh_body(x7, o7, off, mask)
    elif t == 8:
        _tanh_body(x8, o8, off, mask)
    elif t == 9:
        _tanh_body(x9, o9, off, mask)
    elif t == 10:
        _tanh_body(x10, o10, off, mask)
    elif t == 11:
        _tanh_body(x11, o11, off, mask)
    elif t == 12:
        _tanh_body(x12, o12, off, mask)
    elif t == 13:
        _tanh_body(x13, o13, off, mask)
    elif t == 14:
        _tanh_body(x14, o14, off, mask)
    else:
        _tanh_body(x15, o15, off, mask)


@triton.jit
def _tanh_complex_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = idx < n
    re = tl.load(x_ptr + 2 * idx, mask=mask, other=0.0)
    im = tl.load(x_ptr + 2 * idx + 1, mask=mask, other=0.0)
    th = libdevice.tanh(re)
    tn = libdevice.tan(im)
    t2 = tn * tn
    den = 1.0 + th * th * t2
    ore = th * (1.0 + t2) / den
    oim = tn * (1.0 - th * th) / den
    tl.store(o_ptr + 2 * idx, ore, mask=mask)
    tl.store(o_ptr + 2 * idx + 1, oim, mask=mask)


@triton.jit
def _tanh_strided_kernel(
    x_ptr,
    o_ptr,
    n,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    xs0,
    xs1,
    xs2,
    xs3,
    xs4,
    xs5,
    xs6,
    xs7,
    os0,
    os1,
    os2,
    os3,
    os4,
    os5,
    os6,
    os7,
    R: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n
    rem = offs
    xoff = tl.zeros([BLOCK], tl.int64)
    ooff = tl.zeros([BLOCK], tl.int64)
    for d in tl.static_range(R - 1, -1, -1):
        if d == 0:
            sz = s0
            xsd = xs0
            osd = os0
        elif d == 1:
            sz = s1
            xsd = xs1
            osd = os1
        elif d == 2:
            sz = s2
            xsd = xs2
            osd = os2
        elif d == 3:
            sz = s3
            xsd = xs3
            osd = os3
        elif d == 4:
            sz = s4
            xsd = xs4
            osd = os4
        elif d == 5:
            sz = s5
            xsd = xs5
            osd = os5
        elif d == 6:
            sz = s6
            xsd = xs6
            osd = os6
        else:
            sz = s7
            xsd = xs7
            osd = os7
        idx = rem % sz
        rem = rem // sz
        xoff += idx * xsd
        ooff += idx * osd
    z = tl.load(x_ptr + xoff, mask=mask, other=0.0)
    zf = z.to(tl.float32)
    y = 1.0 - 2.0 / (tl.exp2(2.8853900817779268 * zf) + 1.0)
    tl.store(o_ptr + ooff, y.to(o_ptr.dtype.element_ty), mask=mask)


def _block_for(itemsize):
    if itemsize <= 2:
        return 2048
    if itemsize == 4:
        return 1024
    return 512


def _collapse(shape, xstrides, ostrides):
    """Metadata-only rank reduction: drop size-1 dims and merge adjacent dims
    that are dense in both the input and the output."""
    dims = [
        (int(s), int(xs), int(os_))
        for s, xs, os_ in zip(shape, xstrides, ostrides)
        if int(s) != 1
    ]
    merged = []
    for s, xs, os_ in dims:
        if merged:
            ps, pxs, pos = merged[-1]
            if pxs == xs * s and pos == os_ * s:
                merged[-1] = (ps * s, pxs, pos)
                continue
        merged.append((s, xs, os_))
    return merged


def _launch_one(t, o):
    n = t.numel()
    if n == 0:
        return
    block = _block_for(o.element_size())
    grid = ((n + block - 1) // block,)
    if t.is_contiguous() and o.is_contiguous():
        _tanh_flat_kernel[grid](t, o, n, n >= 2**31, BLOCK=block, num_warps=4)
        return
    dims = _collapse(t.shape, t.stride(), o.stride())
    rank = len(dims)
    if rank > _MAX_RANK:
        raise RuntimeError("_foreach_tanh: unsupported tensor rank")
    sizes = [d[0] for d in dims] + [1] * (_MAX_RANK - rank)
    xs = [d[1] for d in dims] + [0] * (_MAX_RANK - rank)
    os_ = [d[2] for d in dims] + [0] * (_MAX_RANK - rank)
    _tanh_strided_kernel[grid](
        t, o, n, *sizes, *xs, *os_, R=rank, BLOCK=block, num_warps=4
    )


def _launch_complex(t, o):
    n = t.numel()
    if n == 0:
        return
    fd = torch.float32 if t.dtype is torch.complex64 else torch.float64
    xt = t.unsqueeze(0) if t.dim() == 0 else t
    ot = o.unsqueeze(0) if o.dim() == 0 else o
    xf = xt.view(fd)
    of = ot.view(fd)
    if not (xf.is_contiguous() and of.is_contiguous()):
        raise RuntimeError("_foreach_tanh: non-contiguous complex tensors unsupported")
    block = 256
    grid = ((n + block - 1) // block,)
    _tanh_complex_kernel[grid](xf, of, n, BLOCK=block, num_warps=4)


def _fused_uniform(ts, dt, n, L):
    t0 = ts[0]
    esize = t0.element_size()
    block = _block_for(esize)
    tpb = (n + block - 1) // block
    if L * n * esize <= _SHARE_MAX_BYTES:
        buf = torch.empty((L,) + t0.shape, dtype=dt, device=t0.device)
        args = list(ts)
        if L < _FUSE_MAX:
            args.extend([t0] * (_FUSE_MAX - L))
        _tanh_fused_shared_kernel[(tpb, L)](*args, buf, n, BLOCK=block, num_warps=4)
        return list(buf.unbind(0))
    outs = [torch.empty_like(t) for t in ts]
    args = list(ts) + outs
    if L < _FUSE_MAX:
        pad = _FUSE_MAX - L
        args.extend([t0] * pad)
        args.extend([outs[0]] * pad)
    _tanh_fused_kernel[(tpb, L)](*args, n, BLOCK=block, num_warps=4)
    return outs


def _general(ts):
    pairs = []
    for t in ts:
        if not isinstance(t, torch.Tensor):
            raise TypeError("_foreach_tanh: expected a list of Tensors")
        dt = t.dtype
        if dt.is_complex:
            pairs.append((t, torch.empty_like(t), True))
        else:
            odt = dt if dt.is_floating_point else torch.get_default_dtype()
            pairs.append((t, torch.empty_like(t, dtype=odt), False))
    for t, o, cplx in pairs:
        if cplx:
            _launch_complex(t, o)
        else:
            _launch_one(t, o)
    return [p[1] for p in pairs]


def _foreach_tanh(self):
    logger.debug("GEMS_MTHREADS _FOREACH_TANH")
    if not isinstance(self, (list, tuple)) or any(
        not isinstance(t, torch.Tensor) or t.device.type != "musa" for t in self
    ):
        return default__foreach_tanh(self)
    ts = list(self)
    L = len(ts)
    if L == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")
    t0 = ts[0]
    if not isinstance(t0, torch.Tensor):
        raise TypeError("_foreach_tanh: expected a list of Tensors")
    dt = t0.dtype
    n = t0.numel()
    fast = (
        L > 1
        and L <= _FUSE_MAX
        and n > 0
        and n * t0.element_size() < _FUSE_MAX_TENSOR_BYTES
        and dt.is_floating_point
        and t0.is_contiguous()
    )
    if fast:
        contig = torch.Tensor.is_contiguous
        for t in ts[1:]:
            if t.dtype is not dt or t.numel() != n or not contig(t):
                fast = False
                break
    if fast:
        return _fused_uniform(ts, dt, n, L)
    return _general(ts)
