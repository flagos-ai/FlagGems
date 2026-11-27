import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# quantized dtype -> declared output code dtype
_QCODE = {
    torch.quint8: torch.uint8,
    torch.qint8: torch.int8,
    torch.qint32: torch.int32,
}
# quantized dtype -> dtype the copy kernel is allowed to touch.
# NOTE (measured on this target): passing a torch.int8 tensor that aliases quantized
# storage to a Triton kernel SIGSEGVs the process, while uint8 aliases of the same
# storage are fine.  1-byte codes are therefore always moved as raw uint8 bytes.
_KDT = {
    torch.quint8: torch.uint8,
    torch.qint8: torch.uint8,
    torch.qint32: torch.int32,
}

_BLOCK = 1024
_NWARPS = 4
_MAXD = 8
_I32MAX = 2147483647
# The row schedule needs a long inner sweep to pay off; below this the flat
# schedule (few fat blocks) is used instead of thousands of one-warp rows.
_MIN_INNER = 1024
# Square tile used when the source and the destination are dense along different
# axes (transpose / channels_last), so both sides move in tile-sized runs
# instead of one scattered element per lane.
_TILE = 64


@triton.jit
def _flat_copy(src, dst, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(dst + offs, tl.load(src + offs, mask=m), mask=m)


@triton.jit
def _tile_copy(
    src,
    dst,
    s_b,
    d_a,
    a_ext,
    b_ext,
    o_sh0,
    o_sh1,
    o_sh2,
    o_sh3,
    o_sh4,
    o_sh5,
    o_sh6,
    o_sh7,
    o_st0,
    o_st1,
    o_st2,
    o_st3,
    o_st4,
    o_st5,
    o_st6,
    o_st7,
    o_dt0,
    o_dt1,
    o_dt2,
    o_dt3,
    o_dt4,
    o_dt5,
    o_dt6,
    o_dt7,
    NDIM: tl.constexpr,
    T: tl.constexpr,
):
    """Tiled copy for views whose source is dense along axis a and whose
    destination is dense along axis b (a != b).

    A T x T block is loaded with runs along the source's fast axis and stored with
    runs along the destination's fast axis; the block is transposed in between.
    """
    cb = tl.program_id(0)
    ca = tl.program_id(1)
    row = tl.program_id(2)
    rem = row
    so = row - row
    do = row - row
    if NDIM > 7:
        q = rem % o_sh7
        rem = rem // o_sh7
        so += q * o_st7
        do += q * o_dt7
    if NDIM > 6:
        q = rem % o_sh6
        rem = rem // o_sh6
        so += q * o_st6
        do += q * o_dt6
    if NDIM > 5:
        q = rem % o_sh5
        rem = rem // o_sh5
        so += q * o_st5
        do += q * o_dt5
    if NDIM > 4:
        q = rem % o_sh4
        rem = rem // o_sh4
        so += q * o_st4
        do += q * o_dt4
    if NDIM > 3:
        q = rem % o_sh3
        rem = rem // o_sh3
        so += q * o_st3
        do += q * o_dt3
    if NDIM > 2:
        q = rem % o_sh2
        rem = rem // o_sh2
        so += q * o_st2
        do += q * o_dt2
    if NDIM > 1:
        q = rem % o_sh1
        rem = rem // o_sh1
        so += q * o_st1
        do += q * o_dt1
    if NDIM > 0:
        q = rem % o_sh0
        so += q * o_st0
        do += q * o_dt0
    ai = ca * T + tl.arange(0, T)
    bi = cb * T + tl.arange(0, T)
    ma = ai < a_ext
    mb = bi < b_ext
    x = tl.load(
        src + (so + bi[:, None] * s_b + ai[None, :]),
        mask=mb[:, None] & ma[None, :],
    )
    tl.store(
        dst + (do + bi[None, :] + ai[:, None] * d_a),
        tl.trans(x),
        mask=ma[:, None] & mb[None, :],
    )


@triton.jit
def _row_copy(
    src,
    dst,
    chunks,
    inner,
    s_in,
    d_in,
    o_sh0,
    o_sh1,
    o_sh2,
    o_sh3,
    o_sh4,
    o_sh5,
    o_sh6,
    o_sh7,
    o_st0,
    o_st1,
    o_st2,
    o_st3,
    o_st4,
    o_st5,
    o_st6,
    o_st7,
    o_dt0,
    o_dt1,
    o_dt2,
    o_dt3,
    o_dt4,
    o_dt5,
    o_dt6,
    o_dt7,
    NDIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program copies BLOCK elements of one outer row.

    The outer row index is decomposed once per program (not per element), and the
    inner axis is swept with a single multiply per element.
    """
    pid = tl.program_id(0)
    row = pid // chunks
    chunk = pid - row * chunks
    rem = row
    so = row - row
    do = row - row
    if NDIM > 7:
        q = rem % o_sh7
        rem = rem // o_sh7
        so += q * o_st7
        do += q * o_dt7
    if NDIM > 6:
        q = rem % o_sh6
        rem = rem // o_sh6
        so += q * o_st6
        do += q * o_dt6
    if NDIM > 5:
        q = rem % o_sh5
        rem = rem // o_sh5
        so += q * o_st5
        do += q * o_dt5
    if NDIM > 4:
        q = rem % o_sh4
        rem = rem // o_sh4
        so += q * o_st4
        do += q * o_dt4
    if NDIM > 3:
        q = rem % o_sh3
        rem = rem // o_sh3
        so += q * o_st3
        do += q * o_dt3
    if NDIM > 2:
        q = rem % o_sh2
        rem = rem // o_sh2
        so += q * o_st2
        do += q * o_dt2
    if NDIM > 1:
        q = rem % o_sh1
        rem = rem // o_sh1
        so += q * o_st1
        do += q * o_dt1
    if NDIM > 0:
        q = rem % o_sh0
        so += q * o_st0
        do += q * o_dt0
    j = chunk * BLOCK + tl.arange(0, BLOCK)
    m = j < inner
    tl.store(dst + (do + j * d_in), tl.load(src + (so + j * s_in), mask=m), mask=m)


@triton.jit
def _flat_viewed(
    src,
    dst,
    n,
    s_st0,
    s_st1,
    s_st2,
    s_st3,
    s_st4,
    s_st5,
    s_st6,
    s_st7,
    s_sh0,
    s_sh1,
    s_sh2,
    s_sh3,
    s_sh4,
    s_sh5,
    s_sh6,
    s_sh7,
    d_st0,
    d_st1,
    d_st2,
    d_st3,
    d_st4,
    d_st5,
    d_st6,
    d_st7,
    NDIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Flat schedule: one lane per element, int32 per-element decomposition.

    Used when the view's dense inner run is shorter than one block, where a
    row-per-program grid would launch thousands of one-warp blocks.
    """
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    rem = i
    so = i - i
    do = i - i
    if NDIM > 7:
        q = rem % s_sh7
        rem = rem // s_sh7
        so += q * s_st7
        do += q * d_st7
    if NDIM > 6:
        q = rem % s_sh6
        rem = rem // s_sh6
        so += q * s_st6
        do += q * d_st6
    if NDIM > 5:
        q = rem % s_sh5
        rem = rem // s_sh5
        so += q * s_st5
        do += q * d_st5
    if NDIM > 4:
        q = rem % s_sh4
        rem = rem // s_sh4
        so += q * s_st4
        do += q * d_st4
    if NDIM > 3:
        q = rem % s_sh3
        rem = rem // s_sh3
        so += q * s_st3
        do += q * d_st3
    if NDIM > 2:
        q = rem % s_sh2
        rem = rem // s_sh2
        so += q * s_st2
        do += q * d_st2
    if NDIM > 1:
        q = rem % s_sh1
        rem = rem // s_sh1
        so += q * s_st1
        do += q * d_st1
    if NDIM > 0:
        q = rem % s_sh0
        so += q * s_st0
        do += q * d_st0
    tl.store(dst + do, tl.load(src + so, mask=m), mask=m)


@triton.jit
def _strided_copy(
    src,
    dst,
    n,
    s_st0,
    s_st1,
    s_st2,
    s_st3,
    s_st4,
    s_st5,
    s_st6,
    s_st7,
    s_sh0,
    s_sh1,
    s_sh2,
    s_sh3,
    s_sh4,
    s_sh5,
    s_sh6,
    s_sh7,
    d_st0,
    d_st1,
    d_st2,
    d_st3,
    d_st4,
    d_st5,
    d_st6,
    d_st7,
    NDIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """int64 fallback: per-element decomposition (kept for huge/odd strides)."""
    pid = tl.program_id(0)
    i = (pid * BLOCK).to(tl.int64) + tl.arange(0, BLOCK).to(tl.int64)
    m = i < n
    rem = i
    so = i - i
    do = i - i
    if NDIM > 7:
        q = rem % s_sh7
        rem = rem // s_sh7
        so += q * s_st7
        do += q * d_st7
    if NDIM > 6:
        q = rem % s_sh6
        rem = rem // s_sh6
        so += q * s_st6
        do += q * d_st6
    if NDIM > 5:
        q = rem % s_sh5
        rem = rem // s_sh5
        so += q * s_st5
        do += q * d_st5
    if NDIM > 4:
        q = rem % s_sh4
        rem = rem // s_sh4
        so += q * s_st4
        do += q * d_st4
    if NDIM > 3:
        q = rem % s_sh3
        rem = rem // s_sh3
        so += q * s_st3
        do += q * d_st3
    if NDIM > 2:
        q = rem % s_sh2
        rem = rem // s_sh2
        so += q * s_st2
        do += q * d_st2
    if NDIM > 1:
        q = rem % s_sh1
        rem = rem // s_sh1
        so += q * s_st1
        do += q * d_st1
    if NDIM > 0:
        q = rem % s_sh0
        so += q * s_st0
        do += q * d_st0
    tl.store(dst + do, tl.load(src + so, mask=m), mask=m)


def _next_pow2(v):
    p = 1
    while p < v:
        p <<= 1
    return p


def _row_plan(dims):
    """Split the view into (outer dims, inner run) collapsing the trailing dims that
    are dense in *both* tensors, so the inner axis has one stride per side.

    Returns (outer, run, s_inner_stride, d_inner_stride).
    """
    s, s_in, d_in = dims[-1]
    run = s
    k = len(dims) - 2
    while k >= 0:
        s, ss, ds = dims[k]
        if ss == s_in * run and ds == d_in * run:
            run *= s
            k -= 1
        else:
            break
    return dims[: k + 1], run, s_in, d_in


def _span(dims, idx):
    tot = 0
    for d in dims:
        tot += (d[0] - 1) * abs(d[idx])
    return tot + 1


def int_repr(self, *args, out=None, **kwargs):
    logger.debug("GEMS_ILUVATAR INT_REPR")
    if not isinstance(self, torch.Tensor):
        raise TypeError(
            "int_repr(): argument 'self' must be Tensor, not " + type(self).__name__
        )
    qdtype = self.dtype
    kdt = _KDT.get(qdtype)
    if kdt is None:
        raise RuntimeError(
            "int_repr: only quantized tensors are supported, got " + str(qdtype)
        )
    if out is None and args:
        out = args[0]

    if out is None:
        ret = torch.empty(self.shape, dtype=_QCODE[qdtype], device=self.device)
        dst = ret
    else:
        if not isinstance(out, torch.Tensor):
            raise TypeError(
                "int_repr(): argument 'out' must be Tensor, not " + type(out).__name__
            )
        code = _QCODE[qdtype]
        if out.dtype != code:
            raise RuntimeError(
                "Expected out tensor to have dtype "
                + str(code)
                + ", but got "
                + str(out.dtype)
                + " instead"
            )
        if out.device != self.device:
            raise RuntimeError(
                "Expected out tensor to be on device " + str(self.device)
            )
        if out.shape != self.shape:
            out.resize_(self.shape)
        dst = out
        ret = out

    n = self.numel()
    if n == 0:
        # No elements to move: aten also does no work here, so do not launch.
        return ret
    src = self.view(kdt)
    if kdt is not torch.uint8:
        ksrc = src
        kdst = dst
    else:
        ksrc = src if src.dtype is torch.uint8 else src.view(torch.uint8)
        kdst = dst if dst.dtype is torch.uint8 else dst.view(torch.uint8)

    if ksrc.is_contiguous() and kdst.is_contiguous():
        grid = ((n + _BLOCK - 1) // _BLOCK,)
        _flat_copy[grid](ksrc, kdst, n, BLOCK=_BLOCK, num_warps=_NWARPS)
        return ret

    _copy_viewed(ksrc, kdst, n)
    return ret


def _outer_count(outer):
    p = 1
    for d in outer:
        p *= d[0]
    return p


def _tile_plan(dims):
    """Return (outer, a_ext, b_ext, s_b, d_a) when the source is dense along one
    axis and the destination is dense along a different axis, else None."""
    a = -1
    b = -1
    for idx, d in enumerate(dims):
        if d[1] == 1:
            a = idx
        if d[2] == 1:
            b = idx
    if a < 0 or b < 0 or a == b:
        return None
    outer = [d for i, d in enumerate(dims) if i != a and i != b]
    return outer, dims[a][0], dims[b][0], dims[b][1], dims[a][2]


def _launch_tile(src, dst, outer, a_ext, b_ext, s_b, d_a):
    rows = 1
    for d in outer:
        rows *= d[0]
    cb = (b_ext + _TILE - 1) // _TILE
    ca = (a_ext + _TILE - 1) // _TILE
    if rows > 65535 or cb > 65535 or ca > 65535:
        return False
    pod = len(outer)
    pad_h = [d[0] for d in outer] + [1] * (_MAXD - pod)
    pad_s = [d[1] for d in outer] + [0] * (_MAXD - pod)
    pad_d = [d[2] for d in outer] + [0] * (_MAXD - pod)
    _tile_copy[(cb, ca, rows)](
        src,
        dst,
        s_b,
        d_a,
        a_ext,
        b_ext,
        pad_h[0],
        pad_h[1],
        pad_h[2],
        pad_h[3],
        pad_h[4],
        pad_h[5],
        pad_h[6],
        pad_h[7],
        pad_s[0],
        pad_s[1],
        pad_s[2],
        pad_s[3],
        pad_s[4],
        pad_s[5],
        pad_s[6],
        pad_s[7],
        pad_d[0],
        pad_d[1],
        pad_d[2],
        pad_d[3],
        pad_d[4],
        pad_d[5],
        pad_d[6],
        pad_d[7],
        NDIM=pod,
        T=_TILE,
        num_warps=_NWARPS,
    )
    return True


def _copy_viewed(src, dst, n):
    dims = [
        (s, ss, ds)
        for s, ss, ds in zip(src.shape, src.stride(), dst.stride())
        if s != 1
    ]
    nd = len(dims)
    if (
        nd > _MAXD
        or n > _I32MAX - _BLOCK
        or max(_span(dims, 1), _span(dims, 2)) > _I32MAX - _BLOCK
    ):
        _strided(src, dst, n)
        return

    outer, inner, s_in, d_in = _row_plan(dims)
    pod = len(outer)
    if max(s_in, d_in) > 2:
        tile = _tile_plan(dims)
        if tile is not None:
            t_outer, a_ext, b_ext, s_b, d_a = tile
            if len(t_outer) <= _MAXD and _launch_tile(
                src, dst, t_outer, a_ext, b_ext, s_b, d_a
            ):
                return
    if inner >= _MIN_INNER and pod <= _MAXD:
        chunks = (inner + _BLOCK - 1) // _BLOCK
        grid = (_outer_count(outer) * chunks,)
        pad_h = [d[0] for d in outer] + [1] * (_MAXD - pod)
        pad_s = [d[1] for d in outer] + [0] * (_MAXD - pod)
        pad_d = [d[2] for d in outer] + [0] * (_MAXD - pod)
        _row_copy[grid](
            src,
            dst,
            chunks,
            inner,
            s_in,
            d_in,
            pad_h[0],
            pad_h[1],
            pad_h[2],
            pad_h[3],
            pad_h[4],
            pad_h[5],
            pad_h[6],
            pad_h[7],
            pad_s[0],
            pad_s[1],
            pad_s[2],
            pad_s[3],
            pad_s[4],
            pad_s[5],
            pad_s[6],
            pad_s[7],
            pad_d[0],
            pad_d[1],
            pad_d[2],
            pad_d[3],
            pad_d[4],
            pad_d[5],
            pad_d[6],
            pad_d[7],
            NDIM=pod,
            BLOCK=_BLOCK,
            num_warps=_NWARPS,
        )
        return

    pad_s = [d[1] for d in dims] + [0] * (_MAXD - nd)
    pad_h = [d[0] for d in dims] + [1] * (_MAXD - nd)
    pad_d = [d[2] for d in dims] + [0] * (_MAXD - nd)
    grid = ((n + _BLOCK - 1) // _BLOCK,)
    _flat_viewed[grid](
        src,
        dst,
        n,
        pad_s[0],
        pad_s[1],
        pad_s[2],
        pad_s[3],
        pad_s[4],
        pad_s[5],
        pad_s[6],
        pad_s[7],
        pad_h[0],
        pad_h[1],
        pad_h[2],
        pad_h[3],
        pad_h[4],
        pad_h[5],
        pad_h[6],
        pad_h[7],
        pad_d[0],
        pad_d[1],
        pad_d[2],
        pad_d[3],
        pad_d[4],
        pad_d[5],
        pad_d[6],
        pad_d[7],
        NDIM=nd,
        BLOCK=_BLOCK,
        num_warps=_NWARPS,
    )


def _strided(src, dst, n):
    dims = [
        (s, ss, ds)
        for s, ss, ds in zip(src.shape, src.stride(), dst.stride())
        if s != 1
    ]
    nd = len(dims)
    if nd > _MAXD:
        raise RuntimeError("int_repr: unsupported non-contiguous rank")
    pad_s = [d[1] for d in dims] + [0] * (_MAXD - nd)
    pad_h = [d[0] for d in dims] + [1] * (_MAXD - nd)
    pad_d = [d[2] for d in dims] + [0] * (_MAXD - nd)
    grid = ((n + _BLOCK - 1) // _BLOCK,)
    _strided_copy[grid](
        src,
        dst,
        n,
        pad_s[0],
        pad_s[1],
        pad_s[2],
        pad_s[3],
        pad_s[4],
        pad_s[5],
        pad_s[6],
        pad_s[7],
        pad_h[0],
        pad_h[1],
        pad_h[2],
        pad_h[3],
        pad_h[4],
        pad_h[5],
        pad_h[6],
        pad_h[7],
        pad_d[0],
        pad_d[1],
        pad_d[2],
        pad_d[3],
        pad_d[4],
        pad_d[5],
        pad_d[6],
        pad_d[7],
        NDIM=nd,
        BLOCK=_BLOCK,
        num_warps=_NWARPS,
    )
