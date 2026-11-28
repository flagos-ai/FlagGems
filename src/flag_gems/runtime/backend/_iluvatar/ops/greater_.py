"""In-place elementwise greater-than (aten::greater_) implemented with Triton.

`run(self, other)` compares elementwise and writes 0/1 into `self` using `self`'s
dtype, returning the same tensor object/storage.  Nothing is allocated and the
out-of-place bool temporary of the eager reference is never materialised.

float64 note: this device cannot read float64 inside a compute kernel (an fp64
tl.load returns zero bits), so the float64 path loads the operands as raw int64
bit patterns and compares them through the IEEE-754 total-order key (sign-flip
transform), storing the bit pattern of 0.0/1.0.  That reproduces exact double
semantics, including NaN (unordered) behaviour.
"""

import logging
import struct

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_BLOCK = 1024
_WARPS = 4
_I32MAX = 2147483647
_I64MIN = -9223372036854775807 - 1


# --------------------------------------------------------------------------- #
# shared compare/store body
# --------------------------------------------------------------------------- #
@triton.jit
def _body(x_ptr, y_ptr, xo, yo, m, F64: tl.constexpr):
    """Load x[xo], y[yo], compare, store 0/1 back into x[xo]."""
    if F64:
        xp = x_ptr.to(tl.pointer_type(tl.int64))
        yp = y_ptr.to(tl.pointer_type(tl.int64))
        ab = tl.load(xp + xo, mask=m)
        bb = tl.load(yp + yo, mask=m)
        imin = tl.full((), -9223372036854775807 - 1, tl.int64)
        ka = tl.where(ab < 0, imin - ab, ab)
        kb = tl.where(bb < 0, imin - bb, bb)
        ok = ((ab & 9223372036854775807) <= 9218868437227405312) & (
            (bb & 9223372036854775807) <= 9218868437227405312
        )
        r = (ka > kb) & ok
        tl.store(xp + xo, r.to(tl.int64) * 4607182418800017408, mask=m)
    else:
        a = tl.load(x_ptr + xo, mask=m, eviction_policy="evict_first")
        b = tl.load(y_ptr + yo, mask=m, eviction_policy="evict_first")
        tl.store(x_ptr + xo, (a > b).to(a.dtype), mask=m, eviction_policy="evict_first")


@triton.jit
def _body_s(x_ptr, xo, m, sv, F64: tl.constexpr):
    """Scalar operand: compare x[xo] against the scalar value `sv`."""
    if F64:
        xp = x_ptr.to(tl.pointer_type(tl.int64))
        ab = tl.load(xp + xo, mask=m)
        imin = tl.full((), -9223372036854775807 - 1, tl.int64)
        ka = tl.where(ab < 0, imin - ab, ab)
        ok = (ab & 9223372036854775807) <= 9218868437227405312
        r = (ka > sv) & ok
        tl.store(xp + xo, r.to(tl.int64) * 4607182418800017408, mask=m)
    else:
        a = tl.load(x_ptr + xo, mask=m)
        tl.store(x_ptr + xo, (a > tl.cast(sv, a.dtype)).to(a.dtype), mask=m)


# --------------------------------------------------------------------------- #
# kernels: tensor operand
# --------------------------------------------------------------------------- #
@triton.jit
def _gt_lin(
    x_ptr, y_ptr, n, s, F64: tl.constexpr, I64: tl.constexpr, BLOCK: tl.constexpr
):
    """Both operands share one linear stride `s`."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    if I64:
        xo = offs.to(tl.int64) * s
    else:
        xo = offs * s
    _body(x_ptr, y_ptr, xo, xo, m, F64)


@triton.jit
def _gt_2(
    x_ptr, y_ptr, n, xs, ys, F64: tl.constexpr, I64: tl.constexpr, BLOCK: tl.constexpr
):
    """Rank-1 with distinct linear strides."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    if I64:
        offs = offs.to(tl.int64)
    _body(x_ptr, y_ptr, offs * xs, offs * ys, m, F64)


@triton.jit
def _gt_r2sl(
    x_ptr,
    y_ptr,
    n,
    d1,
    xs,
    y0,
    y1,
    F64: tl.constexpr,
    I64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Rank-2, self linear (stride `xs`), other with strides (y0, y1)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    if I64:
        rem = offs.to(tl.int64)
    else:
        rem = offs
    i0 = rem // d1
    i1 = rem - i0 * d1
    _body(x_ptr, y_ptr, rem * xs, i0 * y0 + i1 * y1, m, F64)


@triton.jit
def _gt_r2(
    x_ptr,
    y_ptr,
    n,
    d1,
    x0,
    x1,
    y0,
    y1,
    F64: tl.constexpr,
    I64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Rank-2 general."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    if I64:
        rem = offs.to(tl.int64)
    else:
        rem = offs
    i0 = rem // d1
    i1 = rem - i0 * d1
    _body(x_ptr, y_ptr, i0 * x0 + i1 * x1, i0 * y0 + i1 * y1, m, F64)


@triton.jit
def _gt_gen(
    x_ptr,
    y_ptr,
    n,
    xs,
    yl,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    y0,
    y1,
    y2,
    y3,
    y4,
    y5,
    y6,
    NDIM: tl.constexpr,
    XSLIN: tl.constexpr,
    F64: tl.constexpr,
    I64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Rank 3..8.  XSLIN: self is one linear run of stride `xs`."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    if I64:
        base = offs.to(tl.int64)
    else:
        base = offs
    rem = base
    i = rem // d1
    rem = rem - i * d1
    yo = i * y0
    if XSLIN:
        xo = base * xs
    else:
        xo = i * x0
    if NDIM > 2:
        i = rem // d2
        rem = rem - i * d2
        yo += i * y1
        if not XSLIN:
            xo += i * x1
    if NDIM > 3:
        i = rem // d3
        rem = rem - i * d3
        yo += i * y2
        if not XSLIN:
            xo += i * x2
    if NDIM > 4:
        i = rem // d4
        rem = rem - i * d4
        yo += i * y3
        if not XSLIN:
            xo += i * x3
    if NDIM > 5:
        i = rem // d5
        rem = rem - i * d5
        yo += i * y4
        if not XSLIN:
            xo += i * x4
    if NDIM > 6:
        i = rem // d6
        rem = rem - i * d6
        yo += i * y5
        if not XSLIN:
            xo += i * x5
    if NDIM > 7:
        i = rem // d7
        rem = rem - i * d7
        yo += i * y6
        if not XSLIN:
            xo += i * x6
    yo += rem * yl
    if not XSLIN:
        xo += rem * xs
    _body(x_ptr, y_ptr, xo, yo, m, F64)


# --------------------------------------------------------------------------- #
# kernels: scalar operand
# --------------------------------------------------------------------------- #
@triton.jit
def _gts_lin(
    x_ptr, n, s, val, F64: tl.constexpr, I64: tl.constexpr, BLOCK: tl.constexpr
):
    """Self is one linear run of stride `s`; scalar comparison value."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    if I64:
        xo = offs.to(tl.int64) * s
    else:
        xo = offs * s
    _body_s(x_ptr, xo, m, val, F64)


@triton.jit
def _gts_gen(
    x_ptr,
    n,
    xl,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    val,
    NDIM: tl.constexpr,
    F64: tl.constexpr,
    I64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Rank 2..8 general strided self; scalar comparison value."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    if I64:
        rem = offs.to(tl.int64)
    else:
        rem = offs
    i = rem // d1
    rem = rem - i * d1
    xo = i * x0
    if NDIM > 2:
        i = rem // d2
        rem = rem - i * d2
        xo += i * x1
    if NDIM > 3:
        i = rem // d3
        rem = rem - i * d3
        xo += i * x2
    if NDIM > 4:
        i = rem // d4
        rem = rem - i * d4
        xo += i * x3
    if NDIM > 5:
        i = rem // d5
        rem = rem - i * d5
        xo += i * x4
    if NDIM > 6:
        i = rem // d6
        rem = rem - i * d6
        xo += i * x5
    if NDIM > 7:
        i = rem // d7
        rem = rem - i * d7
        xo += i * x6
    xo += rem * xl
    _body_s(x_ptr, xo, m, val, F64)


# --------------------------------------------------------------------------- #
# host helpers
# --------------------------------------------------------------------------- #
def _collapse(shape, xs, ys):
    """Merge affine-consistent adjacent dims (result ordered outermost first)."""
    cs = []
    cxs = []
    cys = []
    for d in range(len(shape) - 1, -1, -1):
        sd = shape[d]
        if sd == 1:
            continue
        sx = xs[d]
        sy = ys[d]
        if cs:
            size = cs[-1]
            if sx == cxs[-1] * size and sy == cys[-1] * size:
                cs[-1] = size * sd
                continue
        cs.append(sd)
        cxs.append(sx)
        cys.append(sy)
    cs.reverse()
    cxs.reverse()
    cys.reverse()
    return cs, cxs, cys


def _collapse1(shape, xs):
    """Merge affine-consistent adjacent dims of a single layout."""
    cs = []
    cxs = []
    for d in range(len(shape) - 1, -1, -1):
        sd = shape[d]
        if sd == 1:
            continue
        sx = xs[d]
        if cs:
            size = cs[-1]
            if sx == cxs[-1] * size:
                cs[-1] = size * sd
                continue
        cs.append(sd)
        cxs.append(sx)
    cs.reverse()
    cxs.reverse()
    return cs, cxs


def _is_lin(cs, cxs, r):
    for k in range(r - 1):
        if cxs[k] != cxs[k + 1] * cs[k + 1]:
            return False
    return True


def _divs(cs, r):
    """divs[k] = prod(cs[k+1:]) for k in 0..r-2 (padded to 7 entries)."""
    out = [1] * 7
    acc = 1
    for k in range(r - 1, 0, -1):
        acc *= cs[k]
        out[k - 1] = acc
    return out


def _pad7(vals):
    out = list(vals[:7])
    out.extend([0] * (7 - len(out)))
    return out


def _f64key(v):
    """IEEE-754 total-order key of a python float, as a signed int64."""
    bits = struct.unpack("<q", struct.pack("<d", float(v)))[0]
    if bits < 0:
        return _I64MIN - bits
    return bits


def _run_tensor(x, y):
    shape = x.shape
    if y.shape != shape:
        y = y.expand(shape)  # raises for non-broadcastable operands
    n = shape.numel()
    if n == 0:
        return x
    f64 = x.dtype is torch.float64 and y.dtype is torch.float64
    if x.is_contiguous() and y.is_contiguous():
        _gt_lin[((n + _BLOCK - 1) // _BLOCK,)](
            x, y, n, 1, f64, n > _I32MAX, BLOCK=_BLOCK, num_warps=_WARPS
        )
        return x

    cs, cxs, cys = _collapse(shape, x.stride(), y.stride())
    r = len(cs)
    if r == 0:  # single element
        cs = [1]
        cxs = [0]
        cys = [0]
        r = 1
    grid = ((n + _BLOCK - 1) // _BLOCK,)
    mx = max(max(cxs), max(cys))
    i64 = (n - 1) * mx > _I32MAX
    kw = {"BLOCK": _BLOCK, "num_warps": _WARPS}

    if r == 1:
        if cxs[0] == cys[0]:
            _gt_lin[grid](x, y, n, cxs[0], f64, i64, **kw)
        else:
            _gt_2[grid](x, y, n, cxs[0], cys[0], f64, i64, **kw)
        return x

    xsl = cxs[-1]
    if _is_lin(cs, cxs, r):
        if r == 2:
            _gt_r2sl[grid](x, y, n, cs[1], xsl, cys[0], cys[1], f64, i64, **kw)
            return x
        dv = _divs(cs, r)
        yp = _pad7(cys)
        _gt_gen[grid](
            x,
            y,
            n,
            xsl,
            cys[-1],
            dv[0],
            dv[1],
            dv[2],
            dv[3],
            dv[4],
            dv[5],
            dv[6],
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            yp[0],
            yp[1],
            yp[2],
            yp[3],
            yp[4],
            yp[5],
            yp[6],
            r,
            True,
            f64,
            i64,
            **kw,
        )
        return x

    if r == 2:
        _gt_r2[grid](x, y, n, cs[1], cxs[0], cxs[1], cys[0], cys[1], f64, i64, **kw)
        return x
    dv = _divs(cs, r)
    xp = _pad7(cxs)
    yp = _pad7(cys)
    _gt_gen[grid](
        x,
        y,
        n,
        xsl,
        cys[-1],
        dv[0],
        dv[1],
        dv[2],
        dv[3],
        dv[4],
        dv[5],
        dv[6],
        xp[0],
        xp[1],
        xp[2],
        xp[3],
        xp[4],
        xp[5],
        xp[6],
        yp[0],
        yp[1],
        yp[2],
        yp[3],
        yp[4],
        yp[5],
        yp[6],
        r,
        False,
        f64,
        i64,
        **kw,
    )
    return x


def _run_scalar(x, s):
    n = x.numel()
    if n == 0:
        return x
    f64 = x.dtype is torch.float64
    if s is True or s is False:
        val = int(s)
        if f64:
            val = _f64key(float(val))
    elif isinstance(s, float):
        val = _f64key(s) if f64 else s
    elif isinstance(s, int):
        val = _f64key(float(s)) if f64 else s
    else:
        z = complex(s)
        v = z.real if z.imag == 0 else float("nan")
        val = _f64key(v) if f64 else v
    kw = {"BLOCK": _BLOCK, "num_warps": _WARPS}

    if x.is_contiguous():
        _gts_lin[((n + _BLOCK - 1) // _BLOCK,)](x, n, 1, val, f64, n > _I32MAX, **kw)
        return x

    cs, cxs = _collapse1(x.shape, x.stride())
    r = len(cs)
    if r == 0:
        _gts_lin[(1,)](x, n, 0, val, f64, False, **kw)
        return x
    mx = max(cxs)
    i64 = (n - 1) * mx > _I32MAX
    grid = ((n + _BLOCK - 1) // _BLOCK,)
    if _is_lin(cs, cxs, r):
        _gts_lin[grid](x, n, cxs[-1], val, f64, i64, **kw)
        return x
    dv = _divs(cs, r)
    xp = _pad7(cxs)
    _gts_gen[grid](
        x,
        n,
        cxs[-1],
        dv[0],
        dv[1],
        dv[2],
        dv[3],
        dv[4],
        dv[5],
        dv[6],
        xp[0],
        xp[1],
        xp[2],
        xp[3],
        xp[4],
        xp[5],
        xp[6],
        val,
        r,
        f64,
        i64,
        **kw,
    )
    return x


def greater_(self, other):
    logger.debug("GEMS_ILUVATAR GREATER_INPLACE")
    if isinstance(other, torch.Tensor):
        return _run_tensor(self, other)
    return _run_scalar(self, other)
