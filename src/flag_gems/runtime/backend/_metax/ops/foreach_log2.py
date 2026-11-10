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

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

_BLOCK = 1024
_WIDE_BLOCK = 4096
_WIDE_WARPS = 8
_WIDE_MAX_ELEMS = 1 << 27
_MAX_DIMS = 8
_MAXK = 4
_MAXK_SMALL = 16
_SMALL_ELEMS = 1 << 20


@triton.jit
def _log2_flat_kernel(x_ptr, y_ptr, n, STREAM: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    if STREAM:
        x = tl.load(x_ptr + offs, mask=m, other=1, eviction_policy="evict_first")
    else:
        x = tl.load(x_ptr + offs, mask=m, other=1)
    if x_ptr.dtype.element_ty == tl.float64:
        y = tl.log2(x.to(tl.float64))
    else:
        y = tl.log2(x.to(tl.float32))
    yv = y.to(y_ptr.dtype.element_ty)
    if STREAM:
        tl.store(y_ptr + offs, yv, mask=m, eviction_policy="evict_first")
    else:
        tl.store(y_ptr + offs, yv, mask=m)


@triton.jit
def _log2_body(x_ptr, y_ptr, n, pid, STREAM: tl.constexpr, BLOCK: tl.constexpr):
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    if STREAM:
        x = tl.load(x_ptr + offs, mask=m, other=1, eviction_policy="evict_first")
    else:
        x = tl.load(x_ptr + offs, mask=m, other=1)
    if x_ptr.dtype.element_ty == tl.float64:
        y = tl.log2(x.to(tl.float64))
    else:
        y = tl.log2(x.to(tl.float32))
    yv = y.to(y_ptr.dtype.element_ty)
    if STREAM:
        tl.store(y_ptr + offs, yv, mask=m, eviction_policy="evict_first")
    else:
        tl.store(y_ptr + offs, yv, mask=m)


@triton.jit
def _log2_multi_kernel(
    x0,
    y0,
    n0,
    x1,
    y1,
    n1,
    x2,
    y2,
    n2,
    x3,
    y3,
    n3,
    K: tl.constexpr,
    STREAM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    b1 = tl.cdiv(n0, BLOCK)
    if pid < b1:
        _log2_body(x0, y0, n0, pid, STREAM, BLOCK)
    elif K > 1:
        b2 = b1 + tl.cdiv(n1, BLOCK)
        if pid < b2:
            _log2_body(x1, y1, n1, pid - b1, STREAM, BLOCK)
        elif K > 2:
            b3 = b2 + tl.cdiv(n2, BLOCK)
            if pid < b3:
                _log2_body(x2, y2, n2, pid - b2, STREAM, BLOCK)
            elif K > 3:
                _log2_body(x3, y3, n3, pid - b3, STREAM, BLOCK)


@triton.jit
def _log2_multi16_kernel(
    x0,
    y0,
    n0,
    x1,
    y1,
    n1,
    x2,
    y2,
    n2,
    x3,
    y3,
    n3,
    x4,
    y4,
    n4,
    x5,
    y5,
    n5,
    x6,
    y6,
    n6,
    x7,
    y7,
    n7,
    x8,
    y8,
    n8,
    x9,
    y9,
    n9,
    x10,
    y10,
    n10,
    x11,
    y11,
    n11,
    x12,
    y12,
    n12,
    x13,
    y13,
    n13,
    x14,
    y14,
    n14,
    x15,
    y15,
    n15,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    b1 = tl.cdiv(n0, BLOCK)
    b2 = b1 + tl.cdiv(n1, BLOCK)
    b3 = b2 + tl.cdiv(n2, BLOCK)
    b4 = b3 + tl.cdiv(n3, BLOCK)
    b5 = b4 + tl.cdiv(n4, BLOCK)
    b6 = b5 + tl.cdiv(n5, BLOCK)
    b7 = b6 + tl.cdiv(n6, BLOCK)
    b8 = b7 + tl.cdiv(n7, BLOCK)
    b9 = b8 + tl.cdiv(n8, BLOCK)
    b10 = b9 + tl.cdiv(n9, BLOCK)
    b11 = b10 + tl.cdiv(n10, BLOCK)
    b12 = b11 + tl.cdiv(n11, BLOCK)
    b13 = b12 + tl.cdiv(n12, BLOCK)
    b14 = b13 + tl.cdiv(n13, BLOCK)
    b15 = b14 + tl.cdiv(n14, BLOCK)
    if pid < b1:
        _log2_body(x0, y0, n0, pid, False, BLOCK)
    if (pid >= b1) & (pid < b2):
        _log2_body(x1, y1, n1, pid - b1, False, BLOCK)
    if (pid >= b2) & (pid < b3):
        _log2_body(x2, y2, n2, pid - b2, False, BLOCK)
    if (pid >= b3) & (pid < b4):
        _log2_body(x3, y3, n3, pid - b3, False, BLOCK)
    if (pid >= b4) & (pid < b5):
        _log2_body(x4, y4, n4, pid - b4, False, BLOCK)
    if (pid >= b5) & (pid < b6):
        _log2_body(x5, y5, n5, pid - b5, False, BLOCK)
    if (pid >= b6) & (pid < b7):
        _log2_body(x6, y6, n6, pid - b6, False, BLOCK)
    if (pid >= b7) & (pid < b8):
        _log2_body(x7, y7, n7, pid - b7, False, BLOCK)
    if (pid >= b8) & (pid < b9):
        _log2_body(x8, y8, n8, pid - b8, False, BLOCK)
    if (pid >= b9) & (pid < b10):
        _log2_body(x9, y9, n9, pid - b9, False, BLOCK)
    if (pid >= b10) & (pid < b11):
        _log2_body(x10, y10, n10, pid - b10, False, BLOCK)
    if (pid >= b11) & (pid < b12):
        _log2_body(x11, y11, n11, pid - b11, False, BLOCK)
    if (pid >= b12) & (pid < b13):
        _log2_body(x12, y12, n12, pid - b12, False, BLOCK)
    if (pid >= b13) & (pid < b14):
        _log2_body(x13, y13, n13, pid - b13, False, BLOCK)
    if (pid >= b14) & (pid < b15):
        _log2_body(x14, y14, n14, pid - b14, False, BLOCK)
    if pid >= b15:
        _log2_body(x15, y15, n15, pid - b15, False, BLOCK)


@triton.jit
def _log2_gen_kernel(
    xp,
    xq,
    yp,
    yq,
    n,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
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
    NDIM: tl.constexpr,
    SAME: tl.constexpr,
    CPLX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    sizes = [s0, s1, s2, s3, s4, s5, s6, s7]
    ist = [i0, i1, i2, i3, i4, i5, i6, i7]
    rem = offs
    ia = offs * 0
    if SAME:
        for d in tl.static_range(NDIM):
            ia = ia + (rem % sizes[d]) * ist[d]
            rem = rem // sizes[d]
        oa = ia
    else:
        ost = [o0, o1, o2, o3, o4, o5, o6, o7]
        oa = offs * 0
        for d in tl.static_range(NDIM):
            idx = rem % sizes[d]
            rem = rem // sizes[d]
            ia = ia + idx * ist[d]
            oa = oa + idx * ost[d]
    if CPLX:
        re = tl.load(xp + 2 * ia, mask=m, other=1.0)
        im = tl.load(xq + 2 * ia, mask=m, other=0.0)
        r = tl.sqrt(re * re + im * im)
        yre = tl.log2(r)
        yim = libdevice.atan2(im, re) * 1.4426950408889634
        yim = tl.where(r == 0, float("nan"), yim)
        tl.store(yp + 2 * oa, yre, mask=m)
        tl.store(yq + 2 * oa, yim, mask=m)
    else:
        x = tl.load(xp + ia, mask=m, other=1)
        if xp.dtype.element_ty == tl.float64:
            y = tl.log2(x.to(tl.float64))
        else:
            y = tl.log2(x.to(tl.float32))
        tl.store(yp + oa, y.to(yp.dtype.element_ty), mask=m)


def _cdiv(n, b):
    return (n + b - 1) // b


def _tile(dt, n):
    """Tile matched to a 16-byte per-thread access, capped for pure streams.

    16-bit types need twice the elements per thread to reach the same access
    width as fp32 (measured 3-4% faster at 16.78M); once a tensor is a pure
    DRAM stream (>= 2^27 elements) the narrower tile measured better
    (1.531 vs 1.507 TB/s), so the wide tile is capped there.
    """
    if dt.itemsize <= 2 and _SMALL_ELEMS <= n <= _WIDE_MAX_ELEMS:
        return _WIDE_BLOCK, _WIDE_WARPS
    return _BLOCK, None


def _flat_launch(x, y, n, dt):
    block, warps = _tile(dt, n)
    stream = n >= _SMALL_ELEMS
    if warps is None:
        _log2_flat_kernel[(_cdiv(n, block),)](x, y, n, STREAM=stream, BLOCK=block)
    else:
        _log2_flat_kernel[(_cdiv(n, block),)](
            x, y, n, STREAM=stream, BLOCK=block, num_warps=warps
        )


def _one(x):
    dt = x.dtype
    n = x.numel()
    if x.is_contiguous():
        if dt.is_floating_point:
            y = torch.empty_like(x)
            if n:
                _flat_launch(x, y, n, dt)
            return y
        cplx = dt.is_complex
        y = (
            torch.empty_like(x)
            if cplx
            else torch.empty_like(x, dtype=torch.get_default_dtype())
        )
        if n == 0:
            return y
        if cplx:
            _log2_gen_kernel[(_cdiv(n, _BLOCK),)](
                x.real,
                x.imag,
                y.real,
                y.imag,
                n,
                n,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                NDIM=1,
                SAME=True,
                CPLX=True,
                BLOCK=_BLOCK,
            )
        else:
            _flat_launch(x, y, n, dt)
        return y
    # non-contiguous input: torch.empty_like reproduces ATen's output strides
    # (identical strides for non-overlapping dense tensors, contiguous otherwise)
    cplx = dt.is_complex
    y = (
        torch.empty_like(x)
        if (dt.is_floating_point or cplx)
        else torch.empty_like(x, dtype=torch.get_default_dtype())
    )
    if n == 0:
        return y
    shape = x.shape
    xs = x.stride()
    ys = y.stride()
    dims = [i for i in range(len(shape)) if shape[i] != 1]
    if dims:
        dims = sorted(dims, key=xs.__getitem__)
    nd = len(dims)
    sizes = [shape[i] for i in dims]
    ist = [xs[i] for i in dims]
    ost = [ys[i] for i in dims]
    same = ist == ost
    grid = (_cdiv(n, _BLOCK),)
    if cplx:
        _log2_gen_kernel[grid](
            x.real,
            x.imag,
            y.real,
            y.imag,
            n,
            *(sizes + [1] * (_MAX_DIMS - nd)),
            *(ist + [0] * (_MAX_DIMS - nd)),
            *(ost + [0] * (_MAX_DIMS - nd)),
            NDIM=nd,
            SAME=same,
            CPLX=True,
            BLOCK=_BLOCK,
        )
    else:
        _log2_gen_kernel[grid](
            x,
            x,
            y,
            y,
            n,
            *(sizes + [1] * (_MAX_DIMS - nd)),
            *(ist + [0] * (_MAX_DIMS - nd)),
            *(ost + [0] * (_MAX_DIMS - nd)),
            NDIM=nd,
            SAME=same,
            CPLX=False,
            BLOCK=_BLOCK,
        )
    return y


def foreach_log2(self):
    if not len(self):
        raise RuntimeError("Tensor list must have at least one tensor.")
    if len(self) == 1:
        return [_one(self[0])]
    outs = [None] * len(self)
    groups = {}
    for i in range(len(self)):
        x = self[i]
        n = x.numel()
        if n and x.dtype.is_floating_point and x.is_contiguous():
            g = groups.get(x.dtype)
            if g is None:
                groups[x.dtype] = [[i], n, n]
            else:
                g[0].append(i)
                g[1] += n
                if n > g[2]:
                    g[2] = n
        else:
            outs[i] = _one(x)
    for dt, (idxs, total, maxn) in groups.items():
        small = total <= _SMALL_ELEMS
        if small:
            kmax = _MAXK_SMALL
            block, warps, stream = _BLOCK, None, False
        else:
            kmax = _MAXK
            block, warps = _tile(dt, maxn)
            stream = True
        for c in range(0, len(idxs), kmax):
            chunk = idxs[c : c + kmax]
            args = []
            blocks = 0
            for i in chunk:
                x = self[i]
                y = torch.empty_like(x)
                outs[i] = y
                n = x.numel()
                blocks += (n + block - 1) // block
                args.append(x)
                args.append(y)
                args.append(n)
            while len(args) < 3 * kmax:
                args.append(args[0])
                args.append(args[1])
                args.append(0)
            if small:
                _log2_multi16_kernel[(blocks,)](*args, BLOCK=block)
            elif warps is None:
                _log2_multi_kernel[(blocks,)](
                    *args, K=len(chunk), STREAM=stream, BLOCK=block
                )
            else:
                _log2_multi_kernel[(blocks,)](
                    *args, K=len(chunk), STREAM=stream, BLOCK=block, num_warps=warps
                )
    return outs
