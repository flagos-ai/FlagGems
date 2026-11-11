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

"""Triton implementation of aten::_foreach_tan (elementwise tan over a tensor list)."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

_BLOCK = 1024
_NUM_WARPS = 4
_FUSE_BLOCK = 512
_FUSE_WARPS = 4
_FUSE_MAX_TENSORS = 16
_FUSE_MAX_BYTES = 8 << 20
_MAX_DIMS = 8

_FLAT_CONFIGS = [
    triton.Config({"BLOCK": 1024}, num_warps=4),
    triton.Config({"BLOCK": 2048}, num_warps=4),
    triton.Config({"BLOCK": 4096}, num_warps=4),
    triton.Config({"BLOCK": 8192}, num_warps=4),
    triton.Config({"BLOCK": 2048}, num_warps=8),
    triton.Config({"BLOCK": 4096}, num_warps=8),
    triton.Config({"BLOCK": 8192}, num_warps=8),
    triton.Config({"BLOCK": 16384}, num_warps=8),
]

_DT_CODES = {
    torch.bool: 0,
    torch.uint8: 1,
    torch.int8: 2,
    torch.int16: 3,
    torch.int32: 4,
    torch.int64: 5,
    torch.float16: 6,
    torch.bfloat16: 7,
    torch.float32: 8,
    torch.float64: 9,
}


@triton.jit
def _offset_of(
    idx,
    n0,
    n1,
    n2,
    n3,
    n4,
    n5,
    n6,
    n7,
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
    t6,
    t7,
    BLOCK: tl.constexpr,
    NDIM: tl.constexpr,
):
    """Map a linear index over a permuted (memory-order) shape to element offsets.

    The plan lists the fastest-varying (innermost) dimension first, so the
    linear index is decomposed least-significant-first.
    """
    off = tl.zeros([BLOCK], dtype=tl.int64)
    rem = idx
    if NDIM > 0:
        q = rem % n0
        rem = rem // n0
        off += q.to(tl.int64) * t0
    if NDIM > 1:
        q = rem % n1
        rem = rem // n1
        off += q.to(tl.int64) * t1
    if NDIM > 2:
        q = rem % n2
        rem = rem // n2
        off += q.to(tl.int64) * t2
    if NDIM > 3:
        q = rem % n3
        rem = rem // n3
        off += q.to(tl.int64) * t3
    if NDIM > 4:
        q = rem % n4
        rem = rem // n4
        off += q.to(tl.int64) * t4
    if NDIM > 5:
        q = rem % n5
        rem = rem // n5
        off += q.to(tl.int64) * t5
    if NDIM > 6:
        q = rem % n6
        rem = rem // n6
        off += q.to(tl.int64) * t6
    if NDIM > 7:
        q = rem % n7
        rem = rem // n7
        off += q.to(tl.int64) * t7
    return off


@triton.autotune(configs=_FLAT_CONFIGS, key=["n", "dt"])
@triton.jit
def _tan_flat_kernel(x_ptr, y_ptr, n, dt, BLOCK: tl.constexpr, FP64: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(x_ptr + offs, mask=mask, other=0)
    if FP64:
        r = libdevice.tan(v.to(tl.float64))
    else:
        r = libdevice.tan(v.to(tl.float32))
    tl.store(y_ptr + offs, r.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _tan_flat_multi_kernel(
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6,
    a7,
    a8,
    a9,
    a10,
    a11,
    a12,
    a13,
    a14,
    a15,
    b0,
    b1,
    b2,
    b3,
    b4,
    b5,
    b6,
    b7,
    b8,
    b9,
    b10,
    b11,
    b12,
    b13,
    b14,
    b15,
    n,
    nblk,
    BLOCK: tl.constexpr,
    NT: tl.constexpr,
    FP64: tl.constexpr,
):
    """One launch for a whole list of equally sized contiguous tensors."""
    pid = tl.program_id(0)
    t = pid // nblk
    offs = (pid - t * nblk) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    xp = a0
    yp = b0
    if t == 1:
        xp = a1
        yp = b1
    elif t == 2:
        xp = a2
        yp = b2
    elif t == 3:
        xp = a3
        yp = b3
    elif t == 4:
        xp = a4
        yp = b4
    elif t == 5:
        xp = a5
        yp = b5
    elif t == 6:
        xp = a6
        yp = b6
    elif t == 7:
        xp = a7
        yp = b7
    elif t == 8:
        xp = a8
        yp = b8
    elif t == 9:
        xp = a9
        yp = b9
    elif t == 10:
        xp = a10
        yp = b10
    elif t == 11:
        xp = a11
        yp = b11
    elif t == 12:
        xp = a12
        yp = b12
    elif t == 13:
        xp = a13
        yp = b13
    elif t == 14:
        xp = a14
        yp = b14
    elif t == 15:
        xp = a15
        yp = b15
    v = tl.load(xp + offs, mask=mask, other=0)
    if FP64:
        r = libdevice.tan(v.to(tl.float64))
    else:
        r = libdevice.tan(v.to(tl.float32))
    tl.store(yp + offs, r.to(yp.dtype.element_ty), mask=mask)


@triton.jit
def _tan_strided_kernel(
    x_ptr,
    y_ptr,
    n,
    n0,
    n1,
    n2,
    n3,
    n4,
    n5,
    n6,
    n7,
    i0,
    i1,
    i2,
    i3,
    i4,
    i5,
    i6,
    i7,
    o0,
    o1,
    o2,
    o3,
    o4,
    o5,
    o6,
    o7,
    BLOCK: tl.constexpr,
    NDIM: tl.constexpr,
    FP64: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n
    xo = _offset_of(
        idx, n0, n1, n2, n3, n4, n5, n6, n7, i0, i1, i2, i3, i4, i5, i6, i7, BLOCK, NDIM
    )
    yo = _offset_of(
        idx, n0, n1, n2, n3, n4, n5, n6, n7, o0, o1, o2, o3, o4, o5, o6, o7, BLOCK, NDIM
    )
    v = tl.load(x_ptr + xo, mask=mask, other=0)
    if FP64:
        r = libdevice.tan(v.to(tl.float64))
    else:
        r = libdevice.tan(v.to(tl.float32))
    tl.store(y_ptr + yo, r.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _tan_complex_kernel(
    x_ptr,
    y_ptr,
    n,
    n0,
    n1,
    n2,
    n3,
    n4,
    n5,
    n6,
    n7,
    i0,
    i1,
    i2,
    i3,
    i4,
    i5,
    i6,
    i7,
    o0,
    o1,
    o2,
    o3,
    o4,
    o5,
    o6,
    o7,
    BLOCK: tl.constexpr,
    NDIM: tl.constexpr,
    FP64: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n
    xo = _offset_of(
        idx, n0, n1, n2, n3, n4, n5, n6, n7, i0, i1, i2, i3, i4, i5, i6, i7, BLOCK, NDIM
    )
    yo = _offset_of(
        idx, n0, n1, n2, n3, n4, n5, n6, n7, o0, o1, o2, o3, o4, o5, o6, o7, BLOCK, NDIM
    )
    xo = 2 * xo
    yo = 2 * yo
    a = tl.load(x_ptr + xo, mask=mask, other=0)
    b = tl.load(x_ptr + xo + 1, mask=mask, other=0)
    if FP64:
        a = a.to(tl.float64)
        b = b.to(tl.float64)
    else:
        a = a.to(tl.float32)
        b = b.to(tl.float32)
    den = tl.cos(2.0 * a) + libdevice.cosh(2.0 * b)
    re = tl.sin(2.0 * a) / den
    im = libdevice.sinh(2.0 * b) / den
    tl.store(y_ptr + yo, re.to(y_ptr.dtype.element_ty), mask=mask)
    tl.store(y_ptr + yo + 1, im.to(y_ptr.dtype.element_ty), mask=mask)


def _plan(shape, in_stride, out_stride):
    """Memory-order dim descriptor; keeps the iteration a coalesced bijection."""
    dims = [
        (int(s), int(a), int(b))
        for s, a, b in zip(shape, in_stride, out_stride)
        if int(s) != 1
    ]
    if not dims:
        return [(1, 1, 1)]
    dims = sorted(dims, key=lambda d: (d[1], d[2]))
    merged = []
    cs, ci, co = dims[0]
    for s, i, o in dims[1:]:
        if ci != 0 and co != 0 and i == ci * cs and o == co * cs:
            cs *= s
        else:
            merged.append((cs, ci, co))
            cs, ci, co = s, i, o
    merged.append((cs, ci, co))
    return merged


def _pad(plan):
    sizes = [1] * _MAX_DIMS
    istr = [0] * _MAX_DIMS
    ostr = [0] * _MAX_DIMS
    for k, (s, i, o) in enumerate(plan):
        sizes[k] = s
        istr[k] = i
        ostr[k] = o
    return sizes, istr, ostr


def _launch_real(x, y):
    n = x.numel()
    if n == 0:
        return
    if x.is_contiguous() and y.is_contiguous():
        dt = _DT_CODES.get(x.dtype, 15) * 16 + _DT_CODES.get(y.dtype, 15)
        fp64 = y.dtype == torch.float64
        grid = lambda meta: ((n + meta["BLOCK"] - 1) // meta["BLOCK"],)
        _tan_flat_kernel[grid](x, y, n, dt, FP64=fp64)
        return
    plan = _plan(x.shape, x.stride(), y.stride())
    if len(plan) > _MAX_DIMS:
        for i in range(x.shape[0]):
            _launch_real(x.select(0, i), y.select(0, i))
        return
    sizes, istr, ostr = _pad(plan)
    grid = ((n + _BLOCK - 1) // _BLOCK,)
    _tan_strided_kernel[grid](
        x,
        y,
        n,
        *sizes,
        *istr,
        *ostr,
        _BLOCK,
        len(plan),
        y.dtype == torch.float64,
        num_warps=_NUM_WARPS,
    )


def _launch_complex(x, y):
    n = x.numel()
    if n == 0:
        return
    # Strides stay in complex-element units: the kernel doubles the offsets.
    plan = _plan(x.shape, x.stride(), y.stride())
    if len(plan) > _MAX_DIMS:
        for i in range(x.shape[0]):
            _launch_complex(x.select(0, i), y.select(0, i))
        return
    sizes, istr, ostr = _pad(plan)
    grid = ((n + _BLOCK - 1) // _BLOCK,)
    _tan_complex_kernel[grid](
        torch.view_as_real(x),
        torch.view_as_real(y),
        n,
        *sizes,
        *istr,
        *ostr,
        _BLOCK,
        len(plan),
        y.dtype == torch.complex128,
        num_warps=_NUM_WARPS,
    )


def _try_fuse(xs, ys):
    """One launch for a list of equally sized contiguous tensors (launch-bound case)."""
    nt = len(xs)
    n = xs[0].numel()
    if n == 0 or n * xs[0].element_size() * nt > _FUSE_MAX_BYTES:
        return False
    dtype = xs[0].dtype
    y_dtype = ys[0].dtype
    for x, y in zip(xs, ys):
        if x.dtype is not dtype or y.dtype is not y_dtype:
            return False
        if not x.is_contiguous() or not y.is_contiguous():
            return False
        if x.numel() != n:
            return False
    pad_x = [xs[0]] * (_FUSE_MAX_TENSORS - nt)
    pad_y = [ys[0]] * (_FUSE_MAX_TENSORS - nt)
    nblk = (n + _FUSE_BLOCK - 1) // _FUSE_BLOCK
    _tan_flat_multi_kernel[(nt * nblk,)](
        *xs,
        *pad_x,
        *ys,
        *pad_y,
        n,
        nblk,
        _FUSE_BLOCK,
        nt,
        ys[0].dtype == torch.float64,
        num_warps=_FUSE_WARPS,
    )
    return True


def foreach_tan(self):
    tensors = self if isinstance(self, (list, tuple)) else [self]
    n = len(tensors)
    if n == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")
    if n > 1 and n <= _FUSE_MAX_TENSORS:
        simple = True
        for t in tensors:
            if t.dtype.is_complex:
                simple = False
                break
        if simple:
            default_dtype = torch.get_default_dtype()
            outs = []
            live = []
            for t in tensors:
                if t.dtype.is_floating_point:
                    out = torch.empty_like(t)
                else:
                    out = torch.empty_like(t, dtype=default_dtype)
                outs.append(out)
                if t.numel():
                    live.append((t, out))
            if len(live) > 1:
                xs = [p[0] for p in live]
                ys = [p[1] for p in live]
                if _try_fuse(xs, ys):
                    return outs
            for x, y in live:
                _launch_real(x, y)
            return outs

    default_dtype = torch.get_default_dtype()
    outs = []
    for t in tensors:
        if t.dtype.is_complex:
            out = torch.empty_like(t)
            _launch_complex(t, out)
        elif t.dtype.is_floating_point:
            out = torch.empty_like(t)
            _launch_real(t, out)
        else:
            out = torch.empty_like(t, dtype=default_dtype)
            _launch_real(t, out)
        outs.append(out)
    return outs
