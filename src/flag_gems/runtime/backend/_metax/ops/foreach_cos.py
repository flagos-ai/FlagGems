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

_MAX_DIMS = 8
_MAX_BATCH = 16
_FUSE_MAX_ELEMS = 1 << 20
_LARGE_THRESHOLD = 1 << 18
_FLAT_BLOCK = 1024
_FLAT_BLOCK_LARGE = 2048
_FLAT_WARPS = 4
_STRIDED_BLOCK = 1024
_STRIDED_WARPS = 4
_I32_LIMIT = 2**31 - 1


@triton.jit
def _cos_flat_kernel(
    x_ptr, y_ptr, n_elem, FP64: tl.constexpr, I64: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    if I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    x = tl.load(x_ptr + offs, mask=mask)
    if FP64:
        y = libdevice.cos(x)
    else:
        y = libdevice.cos(x.to(tl.float32))
    tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def _cos_tile(x_ptr, y_ptr, offs, mask, FP64: tl.constexpr):
    x = tl.load(x_ptr + offs, mask=mask)
    if FP64:
        y = libdevice.cos(x)
    else:
        y = libdevice.cos(x.to(tl.float32))
    tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def _cos_multi_kernel(
    x0,
    y0,
    x1,
    y1,
    x2,
    y2,
    x3,
    y3,
    x4,
    y4,
    x5,
    y5,
    x6,
    y6,
    x7,
    y7,
    x8,
    y8,
    x9,
    y9,
    x10,
    y10,
    x11,
    y11,
    x12,
    y12,
    x13,
    y13,
    x14,
    y14,
    x15,
    y15,
    n_elem,
    blocks_per_tensor,
    FP64: tl.constexpr,
    NT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    t = pid // blocks_per_tensor
    local = pid - t * blocks_per_tensor
    offs = local * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    if t == 0:
        _cos_tile(x0, y0, offs, mask, FP64)
    if NT > 1:
        if t == 1:
            _cos_tile(x1, y1, offs, mask, FP64)
    if NT > 2:
        if t == 2:
            _cos_tile(x2, y2, offs, mask, FP64)
    if NT > 3:
        if t == 3:
            _cos_tile(x3, y3, offs, mask, FP64)
    if NT > 4:
        if t == 4:
            _cos_tile(x4, y4, offs, mask, FP64)
    if NT > 5:
        if t == 5:
            _cos_tile(x5, y5, offs, mask, FP64)
    if NT > 6:
        if t == 6:
            _cos_tile(x6, y6, offs, mask, FP64)
    if NT > 7:
        if t == 7:
            _cos_tile(x7, y7, offs, mask, FP64)
    if NT > 8:
        if t == 8:
            _cos_tile(x8, y8, offs, mask, FP64)
    if NT > 9:
        if t == 9:
            _cos_tile(x9, y9, offs, mask, FP64)
    if NT > 10:
        if t == 10:
            _cos_tile(x10, y10, offs, mask, FP64)
    if NT > 11:
        if t == 11:
            _cos_tile(x11, y11, offs, mask, FP64)
    if NT > 12:
        if t == 12:
            _cos_tile(x12, y12, offs, mask, FP64)
    if NT > 13:
        if t == 13:
            _cos_tile(x13, y13, offs, mask, FP64)
    if NT > 14:
        if t == 14:
            _cos_tile(x14, y14, offs, mask, FP64)
    if NT > 15:
        if t == 15:
            _cos_tile(x15, y15, offs, mask, FP64)


@triton.jit
def _cos_strided_kernel(
    x_ptr,
    y_ptr,
    n_elem,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    FP64: tl.constexpr,
    NDIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    rem = offs.to(tl.int64)
    off = tl.zeros([BLOCK], dtype=tl.int64)
    if NDIM >= 1:
        i0 = rem % d0
        rem = rem // d0
        off += i0 * s0
    if NDIM >= 2:
        i1 = rem % d1
        rem = rem // d1
        off += i1 * s1
    if NDIM >= 3:
        i2 = rem % d2
        rem = rem // d2
        off += i2 * s2
    if NDIM >= 4:
        i3 = rem % d3
        rem = rem // d3
        off += i3 * s3
    if NDIM >= 5:
        i4 = rem % d4
        rem = rem // d4
        off += i4 * s4
    if NDIM >= 6:
        i5 = rem % d5
        rem = rem // d5
        off += i5 * s5
    if NDIM >= 7:
        i6 = rem % d6
        rem = rem // d6
        off += i6 * s6
    if NDIM >= 8:
        i7 = rem % d7
        rem = rem // d7
        off += i7 * s7
    x = tl.load(x_ptr + off, mask=mask)
    if FP64:
        y = libdevice.cos(x)
    else:
        y = libdevice.cos(x.to(tl.float32))
    tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def _cos_complex_kernel(
    x_ptr,
    y_ptr,
    n_elem,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    F64: tl.constexpr,
    NDIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    rem = offs.to(tl.int64)
    off = tl.zeros([BLOCK], dtype=tl.int64)
    if NDIM >= 1:
        i0 = rem % d0
        rem = rem // d0
        off += i0 * s0
    if NDIM >= 2:
        i1 = rem % d1
        rem = rem // d1
        off += i1 * s1
    if NDIM >= 3:
        i2 = rem % d2
        rem = rem // d2
        off += i2 * s2
    if NDIM >= 4:
        i3 = rem % d3
        rem = rem // d3
        off += i3 * s3
    if NDIM >= 5:
        i4 = rem % d4
        rem = rem // d4
        off += i4 * s4
    if NDIM >= 6:
        i5 = rem % d5
        rem = rem // d5
        off += i5 * s5
    if NDIM >= 7:
        i6 = rem % d6
        rem = rem // d6
        off += i6 * s6
    if NDIM >= 8:
        i7 = rem % d7
        rem = rem // d7
        off += i7 * s7
    voff = off * 2
    doff = offs * 2
    xr = tl.load(x_ptr + voff, mask=mask)
    xi = tl.load(x_ptr + voff + 1, mask=mask)
    if F64:
        cr = libdevice.cos(xr) * libdevice.cosh(xi)
        ci = -libdevice.sin(xr) * libdevice.sinh(xi)
    else:
        ar = xr.to(tl.float32)
        ai = xi.to(tl.float32)
        cr = libdevice.cos(ar) * libdevice.cosh(ai)
        ci = -libdevice.sin(ar) * libdevice.sinh(ai)
    tl.store(y_ptr + doff, cr, mask=mask)
    tl.store(y_ptr + doff + 1, ci, mask=mask)


def _coalesce(shape, stride):
    """Merge adjacent dimensions that form a contiguous run (row-major order)."""
    dims = [(int(s), int(st)) for s, st in zip(shape, stride) if int(s) != 1]
    if not dims:
        return [], []
    merged = [[dims[0][0], dims[0][1]]]
    for s, st in dims[1:]:
        ps, pst = merged[-1]
        if pst == st * s:
            merged[-1][0] = ps * s
            merged[-1][1] = st
        else:
            merged.append([s, st])
    return [m[0] for m in merged], [m[1] for m in merged]


def _launch_cos(inp, out, fp64):
    n = inp.numel()
    if n == 0:
        return out
    if inp.is_contiguous():
        block = _FLAT_BLOCK_LARGE if n >= _LARGE_THRESHOLD else _FLAT_BLOCK
        grid = (triton.cdiv(n, block),)
        _cos_flat_kernel[grid](
            inp,
            out,
            n,
            FP64=fp64,
            I64=n > _I32_LIMIT,
            BLOCK=block,
            num_warps=_FLAT_WARPS,
            num_stages=1,
        )
        return out
    shapes, strides = _coalesce(inp.shape, inp.stride())
    nd = len(shapes)
    if nd == 0:
        # All surviving dims were size 1: single element, contiguous-addressable.
        block = _FLAT_BLOCK
        _cos_flat_kernel[(1,)](
            inp,
            out,
            n,
            FP64=fp64,
            I64=False,
            BLOCK=block,
            num_warps=_FLAT_WARPS,
            num_stages=1,
        )
        return out
    if nd > _MAX_DIMS:
        raise NotImplementedError(
            "foreach_cos: unsupported tensor rank after coalescing"
        )
    pad = _MAX_DIMS - nd
    rs = shapes[::-1] + [1] * pad
    rt = strides[::-1] + [0] * pad
    block = _STRIDED_BLOCK
    grid = (triton.cdiv(n, block),)
    _cos_strided_kernel[grid](
        inp,
        out,
        n,
        *rs,
        *rt,
        FP64=fp64,
        NDIM=nd,
        BLOCK=block,
        num_warps=_STRIDED_WARPS,
        num_stages=1,
    )
    return out


def _launch_cos_multi(group, n, fp64):
    nt = len(group)
    x0 = group[0][0]
    y0 = group[0][1]
    args = []
    for i in range(_MAX_BATCH):
        if i < nt:
            args.append(group[i][0])
            args.append(group[i][1])
        else:
            args.append(x0)
            args.append(y0)
    block = _FLAT_BLOCK
    bpt = triton.cdiv(n, block)
    grid = (nt * bpt,)
    _cos_multi_kernel[grid](
        *args,
        n,
        bpt,
        FP64=fp64,
        NT=nt,
        BLOCK=block,
        num_warps=_FLAT_WARPS,
        num_stages=1,
    )


def _launch_cos_complex(t, out, fp64):
    n = t.numel()
    if n == 0:
        return out
    xr = torch.view_as_real(t)
    yr = torch.view_as_real(out)
    shapes, strides = _coalesce(t.shape, t.stride())
    nd = len(shapes)
    if nd == 0:
        shapes, strides, nd = [1], [1], 1
    if nd > _MAX_DIMS:
        raise NotImplementedError(
            "foreach_cos: unsupported tensor rank after coalescing"
        )
    pad = _MAX_DIMS - nd
    rs = shapes[::-1] + [1] * pad
    rt = strides[::-1] + [0] * pad
    block = _STRIDED_BLOCK
    grid = (triton.cdiv(n, block),)
    _cos_complex_kernel[grid](
        xr,
        yr,
        n,
        *rs,
        *rt,
        F64=fp64,
        NDIM=nd,
        BLOCK=block,
        num_warps=_STRIDED_WARPS,
        num_stages=1,
    )
    return out


def _describe(t):
    """Map an input tensor to (kernel operand, output dtype, fp64 flag)."""
    dt = t.dtype
    if dt is torch.bool:
        return t.view(torch.uint8), torch.float32, False
    if dt is torch.float64:
        return t, dt, True
    if dt.is_floating_point:
        return t, dt, False
    # Integer inputs promote to the default floating dtype (as ATen does).
    return t, torch.float32, False


def _cos_single(t):
    dt = t.dtype
    if dt.is_complex:
        out = torch.empty(t.shape, dtype=dt, device=t.device)
        return _launch_cos_complex(t, out, dt is torch.complex128)
    inp, out_dtype, fp64 = _describe(t)
    out = torch.empty(t.shape, dtype=out_dtype, device=t.device)
    _launch_cos(inp, out, fp64)
    return out


def foreach_cos(self=None, *args, **kwargs):
    tensors = self
    if tensors is None:
        tensors = args[0] if args else None
    if tensors is None:
        for key in ("self", "tensors", "input", "inputs"):
            if key in kwargs:
                tensors = kwargs[key]
                break
    if tensors is None:
        raise TypeError("foreach_cos expects a sequence of tensors")
    if torch.is_tensor(tensors):
        return [_cos_single(tensors)]
    if not isinstance(tensors, (list, tuple)):
        tensors = list(tensors)
    if len(tensors) == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")
    outs = [None] * len(tensors)
    buckets = {}
    order = []
    for i, t in enumerate(tensors):
        if not torch.is_tensor(t):
            raise TypeError("foreach_cos expects a sequence of tensors")
        dt = t.dtype
        if dt.is_complex:
            out = torch.empty(t.shape, dtype=dt, device=t.device)
            outs[i] = _launch_cos_complex(t, out, dt is torch.complex128)
            continue
        inp, out_dtype, fp64 = _describe(t)
        out = torch.empty(t.shape, dtype=out_dtype, device=t.device)
        outs[i] = out
        n = t.numel()
        if n == 0:
            continue
        if inp.is_contiguous() and n <= _FUSE_MAX_ELEMS:
            key = (inp.dtype, out_dtype, n, fp64)
            bucket = buckets.get(key)
            if bucket is None:
                bucket = buckets[key] = []
                order.append(key)
            bucket.append((inp, out))
        else:
            _launch_cos(inp, out, fp64)
    for key in order:
        group = buckets[key]
        n = key[2]
        fp64 = key[3]
        if len(group) == 1:
            _launch_cos(group[0][0], group[0][1], fp64)
            continue
        for s in range(0, len(group), _MAX_BATCH):
            chunk = group[s : s + _MAX_BATCH]
            if len(chunk) == 1:
                _launch_cos(chunk[0][0], chunk[0][1], fp64)
            else:
                _launch_cos_multi(chunk, n, fp64)
    return outs
