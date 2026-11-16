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

"""Triton implementation of aten::_foreach_cosh (elementwise cosh over a tensor list)."""

import torch
import triton
import triton.language as tl

try:  # Triton >= 3.0 layout
    from triton.language.extra import libdevice as _libdevice

    _USE_LIBDEVICE = True
except Exception:  # pragma: no cover
    try:
        from triton.language.extra.cuda import libdevice as _libdevice

        _USE_LIBDEVICE = True
    except Exception:
        _libdevice = None
        _USE_LIBDEVICE = False

_USE_LIBDEVICE_CONST = tl.constexpr(bool(_USE_LIBDEVICE))

_BLOCK = 1024
_MAX_RANK = 8
_OP_COSH = 0
_OP_RE = 1
_OP_IM = 2
_FUSE_MAX = 16
_FUSE_BLOCK = 1024
_FUSE_TOTAL_LIMIT = 1 << 20

# Bounded configuration search for the contiguous flat kernel.  Every entry only
# changes elements-per-thread / warps-per-CTA for identical source math; the
# round-2 instruction profile showed one 16 B load in flight per thread and a
# 4 KB per-program footprint with full vec4 access width.
_FLAT_CONFIGS = [
    triton.Config({"BLOCK": 1024, "COSHMODE": 1}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK": 2048, "COSHMODE": 1}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK": 4096, "COSHMODE": 1}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK": 8192, "COSHMODE": 1}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK": 16384, "COSHMODE": 1}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK": 16384, "COSHMODE": 1}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK": 4096, "COSHMODE": 0}, num_warps=4, num_stages=1),
]


@triton.jit
def _cosh_vals(x, F64: tl.constexpr):
    if _USE_LIBDEVICE_CONST:
        return _libdevice.cosh(x)
    else:
        ax = tl.abs(x)
        if F64:
            thr = 709.0
        else:
            thr = 88.0
        s = tl.where(ax > thr, 0.6931471805599453, 0.0)
        e = tl.exp(ax - s) + tl.exp(-ax - s)
        return e * tl.where(s > 0.0, 1.0, 0.5)


@triton.jit
def _cosh_fast(x, F64: tl.constexpr):
    """cosh via the exp pair, with a single ln2 shift in the overflow band."""
    ax = tl.abs(x)
    if F64:
        thr = 709.0
    else:
        thr = 88.0
    big = ax > thr
    t = tl.exp(ax - tl.where(big, 0.6931471805599453, 0.0))
    c = 0.5 * (t + 1.0 / t)
    return tl.where(big, 2.0 * c, c)


@triton.jit
def _cosh_sel(x, F64: tl.constexpr, MODE: tl.constexpr):
    if MODE == 0:
        return _cosh_vals(x, F64)
    else:
        return _cosh_fast(x, F64)


@triton.jit
def _elem(
    x_ptr,
    b_ptr,
    o_ptr,
    xoff,
    mask,
    OP: tl.constexpr,
    F64: tl.constexpr,
    MODE: tl.constexpr,
):
    x = tl.load(x_ptr + xoff, mask=mask, other=0)
    if F64:
        xf = x.to(tl.float64)
    else:
        xf = x.to(tl.float32)
    if OP == 0:
        y = _cosh_sel(xf, F64, MODE)
    else:
        b = tl.load(b_ptr + xoff, mask=mask, other=0)
        if F64:
            bf = b.to(tl.float64)
        else:
            bf = b.to(tl.float32)
        if OP == 1:
            y = _cosh_sel(xf, F64, MODE) * _libdevice.cos(bf)
        else:
            y = _libdevice.sinh(xf) * _libdevice.sin(bf)
    tl.store(o_ptr + xoff, y.to(o_ptr.dtype.element_ty), mask=mask)


@triton.autotune(configs=_FLAT_CONFIGS, key=["numel", "tag"], warmup=10, rep=40)
@triton.jit
def _cosh_flat_kernel(
    x_ptr,
    b_ptr,
    o_ptr,
    numel,
    tag,
    BLOCK: tl.constexpr,
    F64: tl.constexpr,
    IDX64: tl.constexpr,
    OP: tl.constexpr,
    COSHMODE: tl.constexpr,
):
    pid = tl.program_id(0)
    if IDX64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    _elem(x_ptr, b_ptr, o_ptr, offs, mask, OP, F64, COSHMODE)


@triton.jit
def _cosh_strided_kernel(
    x_ptr,
    b_ptr,
    o_ptr,
    numel,
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
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    BLOCK: tl.constexpr,
    F64: tl.constexpr,
    OP: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < numel
    rem = offs
    xoff = tl.zeros([BLOCK], dtype=tl.int64)

    c = rem % d0
    rem = rem // d0
    xoff += c * xs0
    c = rem % d1
    rem = rem // d1
    xoff += c * xs1
    c = rem % d2
    rem = rem // d2
    xoff += c * xs2
    c = rem % d3
    rem = rem // d3
    xoff += c * xs3
    c = rem % d4
    rem = rem // d4
    xoff += c * xs4
    c = rem % d5
    rem = rem // d5
    xoff += c * xs5
    c = rem % d6
    rem = rem // d6
    xoff += c * xs6
    c = rem % d7
    xoff += c * xs7

    rem = offs
    ooff = tl.zeros([BLOCK], dtype=tl.int64)
    c = rem % d0
    rem = rem // d0
    ooff += c * os0
    c = rem % d1
    rem = rem // d1
    ooff += c * os1
    c = rem % d2
    rem = rem // d2
    ooff += c * os2
    c = rem % d3
    rem = rem // d3
    ooff += c * os3
    c = rem % d4
    rem = rem // d4
    ooff += c * os4
    c = rem % d5
    rem = rem // d5
    ooff += c * os5
    c = rem % d6
    rem = rem // d6
    ooff += c * os6
    c = rem % d7
    ooff += c * os7

    x = tl.load(x_ptr + xoff, mask=mask, other=0)
    if F64:
        xf = x.to(tl.float64)
    else:
        xf = x.to(tl.float32)
    if OP == 0:
        y = _cosh_vals(xf, F64)
    else:
        b = tl.load(b_ptr + xoff, mask=mask, other=0)
        if F64:
            bf = b.to(tl.float64)
        else:
            bf = b.to(tl.float32)
        if OP == 1:
            y = _cosh_vals(xf, F64) * _libdevice.cos(bf)
        else:
            y = _libdevice.sinh(xf) * _libdevice.sin(bf)
    tl.store(o_ptr + ooff, y.to(o_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _cosh_fused_kernel(
    x0,
    o0,
    x1,
    o1,
    x2,
    o2,
    x3,
    o3,
    x4,
    o4,
    x5,
    o5,
    x6,
    o6,
    x7,
    o7,
    x8,
    o8,
    x9,
    o9,
    x10,
    o10,
    x11,
    o11,
    x12,
    o12,
    x13,
    o13,
    x14,
    o14,
    x15,
    o15,
    numel,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    F64: tl.constexpr,
):
    pid = tl.program_id(0)
    tid = tl.program_id(1)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel

    # Exactly one of the N mutually exclusive predicates is true, so the
    # predicated loads contribute no traffic and the sum selects one tensor.
    x = tl.load(x0 + offs, mask=mask & (tid == 0), other=0)
    x += tl.load(x1 + offs, mask=mask & (tid == 1), other=0)
    x += tl.load(x2 + offs, mask=mask & (tid == 2), other=0)
    x += tl.load(x3 + offs, mask=mask & (tid == 3), other=0)
    x += tl.load(x4 + offs, mask=mask & (tid == 4), other=0)
    x += tl.load(x5 + offs, mask=mask & (tid == 5), other=0)
    x += tl.load(x6 + offs, mask=mask & (tid == 6), other=0)
    x += tl.load(x7 + offs, mask=mask & (tid == 7), other=0)
    x += tl.load(x8 + offs, mask=mask & (tid == 8), other=0)
    x += tl.load(x9 + offs, mask=mask & (tid == 9), other=0)
    x += tl.load(x10 + offs, mask=mask & (tid == 10), other=0)
    x += tl.load(x11 + offs, mask=mask & (tid == 11), other=0)
    x += tl.load(x12 + offs, mask=mask & (tid == 12), other=0)
    x += tl.load(x13 + offs, mask=mask & (tid == 13), other=0)
    x += tl.load(x14 + offs, mask=mask & (tid == 14), other=0)
    x += tl.load(x15 + offs, mask=mask & (tid == 15), other=0)

    if F64:
        xf = x.to(tl.float64)
    else:
        xf = x.to(tl.float32)
    y = _cosh_vals(xf, F64)

    tl.store(o0 + offs, y.to(o0.dtype.element_ty), mask=mask & (tid == 0))
    tl.store(o1 + offs, y.to(o1.dtype.element_ty), mask=mask & (tid == 1))
    tl.store(o2 + offs, y.to(o2.dtype.element_ty), mask=mask & (tid == 2))
    tl.store(o3 + offs, y.to(o3.dtype.element_ty), mask=mask & (tid == 3))
    tl.store(o4 + offs, y.to(o4.dtype.element_ty), mask=mask & (tid == 4))
    tl.store(o5 + offs, y.to(o5.dtype.element_ty), mask=mask & (tid == 5))
    tl.store(o6 + offs, y.to(o6.dtype.element_ty), mask=mask & (tid == 6))
    tl.store(o7 + offs, y.to(o7.dtype.element_ty), mask=mask & (tid == 7))
    tl.store(o8 + offs, y.to(o8.dtype.element_ty), mask=mask & (tid == 8))
    tl.store(o9 + offs, y.to(o9.dtype.element_ty), mask=mask & (tid == 9))
    tl.store(o10 + offs, y.to(o10.dtype.element_ty), mask=mask & (tid == 10))
    tl.store(o11 + offs, y.to(o11.dtype.element_ty), mask=mask & (tid == 11))
    tl.store(o12 + offs, y.to(o12.dtype.element_ty), mask=mask & (tid == 12))
    tl.store(o13 + offs, y.to(o13.dtype.element_ty), mask=mask & (tid == 13))
    tl.store(o14 + offs, y.to(o14.dtype.element_ty), mask=mask & (tid == 14))
    tl.store(o15 + offs, y.to(o15.dtype.element_ty), mask=mask & (tid == 15))


def _fused(tensors):
    """One launch for a short list of uniform, contiguous, small tensors."""
    n = len(tensors)
    if n < 2 or n > _FUSE_MAX:
        return None
    t0 = tensors[0]
    if not t0.is_floating_point() or not t0.is_contiguous():
        return None
    dtype = t0.dtype
    numel = t0.numel()
    if numel == 0 or numel * n > _FUSE_TOTAL_LIMIT:
        return None
    for t in tensors:
        if t.dtype != dtype or t.numel() != numel or not t.is_contiguous():
            return None
    outs = [torch.empty_like(t) for t in tensors]
    pad_x = [t0] * (_FUSE_MAX - n)
    pad_o = [outs[0]] * (_FUSE_MAX - n)
    xs = list(tensors) + pad_x
    os_ = outs + pad_o
    args = []
    for i in range(_FUSE_MAX):
        args.append(xs[i])
        args.append(os_[i])
    grid = (triton.cdiv(numel, _FUSE_BLOCK), n)
    _cosh_fused_kernel[grid](
        *args,
        numel,
        N=n,
        BLOCK=_FUSE_BLOCK,
        F64=(dtype == torch.float64),
        num_warps=4,
        num_stages=1,
    )
    return outs


def _tag(dtype, op):
    if dtype == torch.float16:
        k = 0
    elif dtype == torch.bfloat16:
        k = 1
    elif dtype == torch.float32:
        k = 2
    elif dtype == torch.float64:
        k = 3
    else:
        k = 4
    return k * 4 + op


def _collapse(dims, xs, os_):
    """Drop size-1 dims and merge adjacent dims contiguous in both layouts."""
    dims2, xs2, os2 = [], [], []
    for d, sx, so in zip(dims, xs, os_):
        if d == 1:
            continue
        dims2.append(d)
        xs2.append(sx)
        os2.append(so)
    changed = True
    while changed:
        changed = False
        i = len(dims2) - 2
        while i >= 0:
            nxt = dims2[i + 1]
            if xs2[i] == xs2[i + 1] * nxt and os2[i] == os2[i + 1] * nxt:
                dims2[i] = dims2[i] * dims2[i + 1]
                del dims2[i + 1]
                del xs2[i + 1]
                del os2[i + 1]
                changed = True
            i -= 1
    return dims2, xs2, os2


def _launch(t, out, op=0, b=None):
    numel = t.numel()
    if numel == 0:
        return
    if b is None:
        b = t
    f64 = out.dtype == torch.float64

    if t.is_contiguous() and out.is_contiguous():
        idx64 = numel > 0x7FFFFFFF

        def _grid(meta):
            return (triton.cdiv(numel, meta["BLOCK"]),)

        _cosh_flat_kernel[_grid](
            t,
            b,
            out,
            numel,
            _tag(t.dtype, op),
            F64=f64,
            IDX64=idx64,
            OP=op,
        )
        return

    dims = list(t.shape)
    xs = list(t.stride())
    os_ = list(out.stride())
    dims, xs, os_ = _collapse(dims, xs, os_)

    if len(dims) > _MAX_RANK:
        n0 = dims[0]
        for i in range(n0):
            _launch(
                t.narrow(0, i, 1).squeeze(0),
                out.narrow(0, i, 1).squeeze(0),
                op,
                b.narrow(0, i, 1).squeeze(0),
            )
        return

    # Reverse so that d0 is the innermost (fastest varying) logical dim.
    dims = dims[::-1]
    xs = xs[::-1]
    os_ = os_[::-1]
    pad = _MAX_RANK - len(dims)
    dims = dims + [1] * pad
    xs = xs + [0] * pad
    os_ = os_ + [0] * pad

    grid = (triton.cdiv(numel, _BLOCK),)
    _cosh_strided_kernel[grid](
        t,
        b,
        out,
        numel,
        xs[0],
        xs[1],
        xs[2],
        xs[3],
        xs[4],
        xs[5],
        xs[6],
        xs[7],
        os_[0],
        os_[1],
        os_[2],
        os_[3],
        os_[4],
        os_[5],
        os_[6],
        os_[7],
        dims[0],
        dims[1],
        dims[2],
        dims[3],
        dims[4],
        dims[5],
        dims[6],
        dims[7],
        BLOCK=_BLOCK,
        F64=f64,
        OP=op,
        num_warps=4,
        num_stages=1,
    )


def _cosh_one(t):
    if t.is_complex():
        out = torch.empty_like(t)
        if t.numel() == 0:
            return out
        rt = torch.view_as_real(t)
        ro = torch.view_as_real(out)
        a = rt[..., 0]
        b = rt[..., 1]
        _launch(a, ro[..., 0], op=_OP_RE, b=b)
        _launch(a, ro[..., 1], op=_OP_IM, b=b)
        return out

    if t.is_floating_point():
        out_dtype = t.dtype
    else:
        out_dtype = torch.get_default_dtype()
    out = torch.empty_like(t, dtype=out_dtype)
    _launch(t, out, op=_OP_COSH)
    return out


def foreach_cosh(self):
    if isinstance(self, torch.Tensor):
        tensors = [self]
    else:
        tensors = list(self)
    if len(tensors) == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")
    fused = _fused(tensors)
    if fused is not None:
        return fused
    return [_cosh_one(t) for t in tensors]
