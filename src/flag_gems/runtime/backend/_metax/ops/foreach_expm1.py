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

"""Triton implementation of aten::_foreach_expm1 (elementwise expm1 over a tensor list)."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice as _libdevice

_MAX_RANK = 8
_FLAT_BLOCK = 1024
_FLAT_WARPS = 4
_STRIDED_BLOCK = 512
_STRIDED_WARPS = 4
_GROUP_MAX = 16
# Grouping trades per-tensor host dispatch for a per-program tensor-selection
# chain, so it only pays off while each tensor is small enough that one launch
# (~2.3 us) is comparable to the kernel time.  Measured crossover is far below
# the multi-million-element regime; keep it as a named, tunable constant.
_GROUP_MAX_ELEMS = 65536

_FLOAT8_DTYPES = tuple(
    getattr(torch, name)
    for name in (
        "float8_e4m3fn",
        "float8_e5m2",
        "float8_e4m3fnuz",
        "float8_e5m2fnuz",
        "float8_e8m0fnu",
    )
    if hasattr(torch, name)
)


@triton.jit
def _strided_offset(
    offs,
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
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Signed int64 flat memory offset for each logical flat index in ``offs``."""
    rem = offs
    off = tl.zeros([BLOCK], dtype=tl.int64)
    if RANK > 0:
        c = rem % d0
        rem = rem // d0
        off = off + c * s0
    if RANK > 1:
        c = rem % d1
        rem = rem // d1
        off = off + c * s1
    if RANK > 2:
        c = rem % d2
        rem = rem // d2
        off = off + c * s2
    if RANK > 3:
        c = rem % d3
        rem = rem // d3
        off = off + c * s3
    if RANK > 4:
        c = rem % d4
        rem = rem // d4
        off = off + c * s4
    if RANK > 5:
        c = rem % d5
        rem = rem // d5
        off = off + c * s5
    if RANK > 6:
        c = rem % d6
        rem = rem // d6
        off = off + c * s6
    if RANK > 7:
        c = rem % d7
        rem = rem // d7
        off = off + c * s7
    return off


@triton.jit
def _expm1_flat_kernel(
    x_ptr,
    y_ptr,
    numel,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    WIDE: tl.constexpr,
    FP32: tl.constexpr,
    FAST: tl.constexpr,
):
    # ATen uses fp32 opmath for half/bfloat16.  For 16-bit outputs an exp2-based
    # form is within the representation step and avoids the longer libdevice
    # expm1 expansion that profiling showed on the critical path; fp32/fp64 keep
    # the relative-accuracy-preserving libdevice expm1.
    pid = tl.program_id(0)
    if WIDE:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)
    if EVEN:
        x = tl.load(x_ptr + offs)
        if FP32:
            xf = x.to(tl.float32)
            if FAST:
                y = _libdevice.exp2(xf * 1.4426950408889634) - 1.0
            else:
                y = _libdevice.expm1(xf)
        else:
            y = _libdevice.expm1(x)
        tl.store(y_ptr + offs, y)
    else:
        mask = offs < numel
        x = tl.load(x_ptr + offs, mask=mask)
        if FP32:
            xf = x.to(tl.float32)
            if FAST:
                y = _libdevice.exp2(xf * 1.4426950408889634) - 1.0
            else:
                y = _libdevice.expm1(xf)
        else:
            y = _libdevice.expm1(x)
        tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def _expm1_group_kernel(
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
    FP32: tl.constexpr,
    FAST: tl.constexpr,
):
    """One launch for up to 16 contiguous same-dtype tensors.

    Unused slots are passed with numel 0, which contributes no blocks, so the
    cumulative-block comparison never selects them.  The selection chain costs a
    fixed number of scalar ops per program, which is why callers only group
    tensors whose per-tensor size is launch-overhead dominated.
    """
    pid = tl.program_id(0)
    xp = x0
    yp = y0
    nn = n0
    start = 0
    cum = tl.cdiv(n0, BLOCK)
    if pid >= cum:
        xp = x1
        yp = y1
        nn = n1
        start = cum
    cum += tl.cdiv(n1, BLOCK)
    if pid >= cum:
        xp = x2
        yp = y2
        nn = n2
        start = cum
    cum += tl.cdiv(n2, BLOCK)
    if pid >= cum:
        xp = x3
        yp = y3
        nn = n3
        start = cum
    cum += tl.cdiv(n3, BLOCK)
    if pid >= cum:
        xp = x4
        yp = y4
        nn = n4
        start = cum
    cum += tl.cdiv(n4, BLOCK)
    if pid >= cum:
        xp = x5
        yp = y5
        nn = n5
        start = cum
    cum += tl.cdiv(n5, BLOCK)
    if pid >= cum:
        xp = x6
        yp = y6
        nn = n6
        start = cum
    cum += tl.cdiv(n6, BLOCK)
    if pid >= cum:
        xp = x7
        yp = y7
        nn = n7
        start = cum
    cum += tl.cdiv(n7, BLOCK)
    if pid >= cum:
        xp = x8
        yp = y8
        nn = n8
        start = cum
    cum += tl.cdiv(n8, BLOCK)
    if pid >= cum:
        xp = x9
        yp = y9
        nn = n9
        start = cum
    cum += tl.cdiv(n9, BLOCK)
    if pid >= cum:
        xp = x10
        yp = y10
        nn = n10
        start = cum
    cum += tl.cdiv(n10, BLOCK)
    if pid >= cum:
        xp = x11
        yp = y11
        nn = n11
        start = cum
    cum += tl.cdiv(n11, BLOCK)
    if pid >= cum:
        xp = x12
        yp = y12
        nn = n12
        start = cum
    cum += tl.cdiv(n12, BLOCK)
    if pid >= cum:
        xp = x13
        yp = y13
        nn = n13
        start = cum
    cum += tl.cdiv(n13, BLOCK)
    if pid >= cum:
        xp = x14
        yp = y14
        nn = n14
        start = cum
    cum += tl.cdiv(n14, BLOCK)
    if pid >= cum:
        xp = x15
        yp = y15
        nn = n15
        start = cum
    offs = (pid - start) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < nn
    x = tl.load(xp + offs, mask=mask)
    if FP32:
        xf = x.to(tl.float32)
        if FAST:
            y = _libdevice.exp2(xf * 1.4426950408889634) - 1.0
        else:
            y = _libdevice.expm1(xf)
    else:
        y = _libdevice.expm1(x)
    tl.store(yp + offs, y, mask=mask)


@triton.jit
def _expm1_strided_kernel(
    x_ptr,
    y_ptr,
    numel,
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
    BLOCK: tl.constexpr,
    RANK: tl.constexpr,
    SAME_LAYOUT: tl.constexpr,
    FP32: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < numel
    off = _strided_offset(
        offs,
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
        RANK,
        BLOCK,
    )
    x = tl.load(x_ptr + off, mask=mask)
    if FP32:
        y = _libdevice.expm1(x.to(tl.float32))
    else:
        y = _libdevice.expm1(x)
    if SAME_LAYOUT:
        tl.store(y_ptr + off, y, mask=mask)
    else:
        tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def _expm1_complex_kernel(
    x_ptr,
    y_ptr,
    numel,
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
    BLOCK: tl.constexpr,
    RANK: tl.constexpr,
    SAME_LAYOUT: tl.constexpr,
    FP64: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < numel
    off = _strided_offset(
        offs,
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
        RANK,
        BLOCK,
    )
    load_off = off * 2
    if SAME_LAYOUT:
        store_off = off * 2
    else:
        store_off = offs * 2
    xr = tl.load(x_ptr + load_off, mask=mask)
    xi = tl.load(x_ptr + load_off + 1, mask=mask)
    if FP64:
        e = _libdevice.exp(xr)
        yr = e * _libdevice.cos(xi) - 1.0
        yi = e * _libdevice.sin(xi)
    else:
        re = xr.to(tl.float32)
        im = xi.to(tl.float32)
        e = _libdevice.exp(re)
        yr = e * _libdevice.cos(im) - 1.0
        yi = e * _libdevice.sin(im)
    tl.store(y_ptr + store_off, yr, mask=mask)
    tl.store(y_ptr + store_off + 1, yi, mask=mask)


def _collapse_dims(shape, strides):
    """Merge adjacent dims that are contiguous in memory; drop size-1 dims."""
    dims = [[int(s), int(st)] for s, st in zip(shape, strides) if int(s) != 1]
    if not dims:
        return [1], [0]
    merged = [dims[-1]]
    for s, st in reversed(dims[:-1]):
        cs, cst = merged[-1]
        if st == cst * cs:
            merged[-1] = [s * cs, cst]
        else:
            merged.append([s, st])
    merged.reverse()
    return [m[0] for m in merged], [m[1] for m in merged]


def _is_contiguous(shape, stride):
    expected = 1
    for s, st in zip(reversed(shape), reversed(stride)):
        if int(s) == 1:
            continue
        if int(st) != expected:
            return False
        expected *= int(s)
    return True


def _is_dense(shape, stride):
    dims = [(int(st), int(s)) for s, st in zip(shape, stride) if int(s) > 1]
    if not dims:
        return True
    running = 1
    for st, s in sorted(dims):
        if st != running:
            return False
        running *= s
    return True


def _alloc_out(t, dtype):
    shape = t.shape
    stride = t.stride()
    if _is_contiguous(shape, stride):
        return torch.empty(shape, dtype=dtype, device=t.device)
    if all(st >= 0 for st in stride) and _is_dense(shape, stride):
        return torch.empty_strided(shape, stride, dtype=dtype, device=t.device)
    return torch.empty(shape, dtype=dtype, device=t.device)


def _pad(seq, fill):
    vals = list(seq) + [fill] * (_MAX_RANK - len(seq))
    return vals


def _launch_group(items):
    """One launch for a small chunk (<= _GROUP_MAX) of contiguous, same-dtype tensors."""
    n = len(items)
    t0, y0 = items[0]
    total = 0
    args = []
    for i in range(_GROUP_MAX):
        if i < n:
            t, out = items[i]
            ni = t.numel()
            total += (ni + _FLAT_BLOCK - 1) // _FLAT_BLOCK
            args.extend((t, out, ni))
        else:
            args.extend((t0, y0, 0))
    _expm1_group_kernel[(total,)](
        *args,
        BLOCK=_FLAT_BLOCK,
        FP32=(y0.dtype is not torch.float64),
        FAST=_is_fast_dtype(y0.dtype),
        num_warps=_FLAT_WARPS,
    )


def _launch_strided(
    x_ptr, y_ptr, numel, shape, strides, same_layout, fp32, block, warps
):
    rank = len(shape)
    dims = _pad(shape, 1)
    strs = _pad(strides, 0)
    grid = (triton.cdiv(numel, block),)
    _expm1_strided_kernel[grid](
        x_ptr,
        y_ptr,
        numel,
        *dims,
        *strs,
        BLOCK=block,
        RANK=rank,
        SAME_LAYOUT=same_layout,
        FP32=fp32,
        num_warps=warps,
    )


def _is_fast_dtype(dtype):
    return dtype is torch.float16 or dtype is torch.bfloat16


def _launch_flat(t, out):
    numel = t.numel()
    block = _FLAT_BLOCK
    grid = (triton.cdiv(numel, block),)
    _expm1_flat_kernel[grid](
        t,
        out,
        numel,
        BLOCK=block,
        EVEN=(numel % block == 0),
        WIDE=(numel >= 2**31),
        FP32=(out.dtype is not torch.float64),
        FAST=_is_fast_dtype(out.dtype),
        num_warps=_FLAT_WARPS,
    )


def _launch_float(t, out):
    numel = t.numel()
    if numel == 0:
        return
    fp32 = out.dtype is not torch.float64
    if _is_contiguous(t.shape, t.stride()):
        _launch_flat(t, out)
        return
    shape, strides = _collapse_dims(t.shape, t.stride())
    if len(shape) > _MAX_RANK:
        for i in range(t.shape[0]):
            _launch_float(t.narrow(0, i, 1).squeeze(0), out.narrow(0, i, 1).squeeze(0))
        return
    same = tuple(out.stride()) == tuple(t.stride())
    _launch_strided(
        t, out, numel, shape, strides, same, fp32, _STRIDED_BLOCK, _STRIDED_WARPS
    )


def _launch_complex(t, out):
    numel = t.numel()
    if numel == 0:
        return
    fp64 = out.dtype is torch.complex128
    xr = torch.view_as_real(t)
    yr = torch.view_as_real(out)
    if _is_contiguous(t.shape, t.stride()):
        shape, strides = [numel], [1]
    else:
        shape, strides = _collapse_dims(t.shape, t.stride())
    while len(shape) > _MAX_RANK:
        for i in range(shape[0]):
            _launch_complex(
                t.narrow(0, i, 1).squeeze(0), out.narrow(0, i, 1).squeeze(0)
            )
        return
    same = tuple(out.stride()) == tuple(t.stride())
    block = _STRIDED_BLOCK
    dims = _pad(shape, 1)
    strs = _pad(strides, 0)
    grid = (triton.cdiv(numel, block),)
    _expm1_complex_kernel[grid](
        xr,
        yr,
        numel,
        *dims,
        *strs,
        BLOCK=block,
        RANK=len(shape),
        SAME_LAYOUT=same,
        FP64=fp64,
        num_warps=_STRIDED_WARPS,
    )


def foreach_expm1(self):
    if not isinstance(self, (list, tuple)):
        raise TypeError("run() expects a sequence of tensors")
    if len(self) == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")

    outs = []
    groups = {}
    strided = []
    complex_items = []
    for t in self:
        dt = t.dtype
        if dt.is_complex:
            out = _alloc_out(t, dt)
            outs.append(out)
            if t.numel():
                complex_items.append((t, out))
            continue
        if dt in _FLOAT8_DTYPES:
            raise NotImplementedError(
                "\"foreach_unary_op_cuda\" not implemented for '%s'"
                % str(dt).replace("torch.", "")
            )
        odtype = dt if dt.is_floating_point else torch.get_default_dtype()
        out = _alloc_out(t, odtype)
        outs.append(out)
        if t.numel() == 0:
            continue
        if _is_contiguous(t.shape, t.stride()):
            groups.setdefault(dt, []).append((t, out))
        else:
            strided.append((t, out))

    for items in groups.values():
        small = []
        for t, out in items:
            if t.numel() <= _GROUP_MAX_ELEMS:
                small.append((t, out))
            else:
                _launch_flat(t, out)
        for k in range(0, len(small), _GROUP_MAX):
            chunk = small[k : k + _GROUP_MAX]
            if len(chunk) == 1:
                _launch_flat(*chunk[0])
            else:
                _launch_group(chunk)
    for t, out in strided:
        _launch_float(t, out)
    for t, out in complex_items:
        _launch_complex(t, out)
    return outs
