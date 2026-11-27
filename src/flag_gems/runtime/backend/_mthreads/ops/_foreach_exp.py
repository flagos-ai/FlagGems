"""Triton implementation of aten::_foreach_exp (elementwise exp over a tensor list)."""

import torch
import triton
import triton.language as tl

_BLOCK = 1024
_MAXD = 6
_FUSED_MAX = 16
_BIG = 1 << 21

_CFG = [
    triton.Config({"BLOCK": 1024}, num_warps=4),
    triton.Config({"BLOCK": 2048}, num_warps=4),
    triton.Config({"BLOCK": 4096}, num_warps=4),
]


@triton.jit
def _exp_flat_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(x_ptr + offs, mask=mask, other=0.0)
    r = tl.exp(v.to(tl.float32))
    tl.store(o_ptr + offs, r.to(o_ptr.dtype.element_ty), mask=mask)


@triton.autotune(configs=_CFG, key=["n"], warmup=10, rep=20)
@triton.jit
def _exp_flat_big(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(x_ptr + offs, mask=mask, other=0.0)
    r = tl.exp(v.to(tl.float32))
    tl.store(o_ptr + offs, r.to(o_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _exp_fused16(
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
    n,
    bpt,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    t = pid // bpt
    lb = pid - t * bpt
    offs = lb * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    xp = x0
    op = o0
    if t >= 1:
        xp = x1
        op = o1
    if t >= 2:
        xp = x2
        op = o2
    if t >= 3:
        xp = x3
        op = o3
    if t >= 4:
        xp = x4
        op = o4
    if t >= 5:
        xp = x5
        op = o5
    if t >= 6:
        xp = x6
        op = o6
    if t >= 7:
        xp = x7
        op = o7
    if t >= 8:
        xp = x8
        op = o8
    if t >= 9:
        xp = x9
        op = o9
    if t >= 10:
        xp = x10
        op = o10
    if t >= 11:
        xp = x11
        op = o11
    if t >= 12:
        xp = x12
        op = o12
    if t >= 13:
        xp = x13
        op = o13
    if t >= 14:
        xp = x14
        op = o14
    if t >= 15:
        xp = x15
        op = o15
    v = tl.load(xp + offs, mask=mask, other=0.0)
    r = tl.exp(v.to(tl.float32))
    tl.store(op + offs, r.to(op.dtype.element_ty), mask=mask)


@triton.jit
def _exp_strided_kernel(
    x_ptr,
    o_ptr,
    n,
    sh0,
    sh1,
    sh2,
    sh3,
    sh4,
    sh5,
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
    out_contig,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    c4 = sh5
    c3 = sh4 * c4
    c2 = sh3 * c3
    c1 = sh2 * c2
    c0 = sh1 * c1
    i0 = (offs // c0) % sh0
    i1 = (offs // c1) % sh1
    i2 = (offs // c2) % sh2
    i3 = (offs // c3) % sh3
    i4 = (offs // c4) % sh4
    i5 = offs % sh5
    in_off = i0 * t0 + i1 * t1 + i2 * t2 + i3 * t3 + i4 * t4 + i5 * t5
    v = tl.load(x_ptr + in_off, mask=mask, other=0.0)
    r = tl.exp(v.to(tl.float32))
    out_off = tl.where(out_contig != 0, offs, in_off)
    tl.store(o_ptr + out_off, r.to(o_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _exp_complex_kernel(
    x_ptr,
    o_ptr,
    n,
    sh0,
    sh1,
    sh2,
    sh3,
    sh4,
    sh5,
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
    out_contig,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    c4 = sh5
    c3 = sh4 * c4
    c2 = sh3 * c3
    c1 = sh2 * c2
    c0 = sh1 * c1
    i0 = (offs // c0) % sh0
    i1 = (offs // c1) % sh1
    i2 = (offs // c2) % sh2
    i3 = (offs // c3) % sh3
    i4 = (offs // c4) % sh4
    i5 = offs % sh5
    in_off = (i0 * t0 + i1 * t1 + i2 * t2 + i3 * t3 + i4 * t4 + i5 * t5) * 2
    a = tl.load(x_ptr + in_off, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(x_ptr + in_off + 1, mask=mask, other=0.0).to(tl.float32)
    ea = tl.exp(a)
    re = ea * tl.cos(b)
    im = ea * tl.sin(b)
    out_off = tl.where(out_contig != 0, offs, in_off // 2) * 2
    tl.store(o_ptr + out_off, re.to(o_ptr.dtype.element_ty), mask=mask)
    tl.store(o_ptr + out_off + 1, im.to(o_ptr.dtype.element_ty), mask=mask)


def _out_dtype(dt):
    if dt.is_floating_point or dt.is_complex:
        return dt
    return torch.get_default_dtype()


def _collapse(shape, stride):
    dims = [(s, t) for s, t in zip(shape, stride) if s != 1]
    merged = []
    for s, t in dims:
        if merged:
            ps, pt = merged[-1]
            if pt == t * s:
                merged[-1] = (ps * s, pt)
                continue
        merged.append((s, t))
    return merged


def _pad6(dims):
    sh = [1] * _MAXD
    st = [0] * _MAXD
    for i, (s, t) in enumerate(dims):
        sh[i] = s
        st[i] = t
    return sh, st


def _launch_strided(kernel, x, out, n):
    dims = _collapse(x.shape, x.stride())
    if len(dims) > _MAXD:
        raise RuntimeError("unsupported tensor rank for strided exp")
    sh, st = _pad6(dims)
    out_contig = 1 if out.is_contiguous() else 0
    grid = (triton.cdiv(n, _BLOCK),)
    kernel[grid](
        x,
        out,
        n,
        sh[0],
        sh[1],
        sh[2],
        sh[3],
        sh[4],
        sh[5],
        st[0],
        st[1],
        st[2],
        st[3],
        st[4],
        st[5],
        out_contig,
        BLOCK=_BLOCK,
    )


def _launch_complex(x, out, n):
    xr = torch.view_as_real(x)
    orr = torch.view_as_real(out)
    dims = _collapse(x.shape, x.stride())
    if len(dims) > _MAXD:
        raise RuntimeError("unsupported tensor rank for strided exp")
    sh, st = _pad6(dims)
    out_contig = 1 if out.is_contiguous() else 0
    grid = (triton.cdiv(n, _BLOCK),)
    _exp_complex_kernel[grid](
        xr,
        orr,
        n,
        sh[0],
        sh[1],
        sh[2],
        sh[3],
        sh[4],
        sh[5],
        st[0],
        st[1],
        st[2],
        st[3],
        st[4],
        st[5],
        out_contig,
        BLOCK=_BLOCK,
    )


def _launch_one(x, out):
    n = x.numel()
    if n == 0:
        return
    if x.is_complex():
        _launch_complex(x, out, n)
    elif x.is_contiguous():
        if n >= _BIG:
            _exp_flat_big[lambda meta: (triton.cdiv(n, meta["BLOCK"]),)](x, out, n)
        else:
            grid = (triton.cdiv(n, _BLOCK),)
            _exp_flat_kernel[grid](x, out, n, BLOCK=_BLOCK)
    else:
        _launch_strided(_exp_strided_kernel, x, out, n)


def _fusable(xs, outs, idx):
    x0 = xs[idx[0]]
    dt = x0.dtype
    if dt.is_complex:
        return False
    odt = outs[idx[0]].dtype
    n = x0.numel()
    for i in idx:
        x = xs[i]
        if x.dtype != dt or x.numel() != n or not x.is_contiguous():
            return False
        o = outs[i]
        if o.dtype != odt or not o.is_contiguous():
            return False
    return True


def _launch_fused(xs, outs, chunk):
    n = xs[chunk[0]].numel()
    bpt = triton.cdiv(n, _BLOCK)
    k = len(chunk)
    args = []
    for j in range(_FUSED_MAX):
        i = chunk[j] if j < k else chunk[0]
        args.append(xs[i])
        args.append(outs[i])
    grid = (bpt * k,)
    _exp_fused16[grid](*args, n, bpt, BLOCK=_BLOCK)


def _dispatch(xs, outs):
    idx = [i for i, x in enumerate(xs) if x.numel() > 0]
    if not idx:
        return
    if len(idx) >= 2 and _fusable(xs, outs, idx):
        for start in range(0, len(idx), _FUSED_MAX):
            _launch_fused(xs, outs, idx[start : start + _FUSED_MAX])
        return
    for i in idx:
        _launch_one(xs[i], outs[i])


def _foreach_exp(self):
    tensors = list(self)
    if len(tensors) == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")
    outs = []
    for x in tensors:
        odt = _out_dtype(x.dtype)
        if odt == x.dtype:
            outs.append(torch.empty_like(x))
        else:
            outs.append(torch.empty_like(x, dtype=odt))
    _dispatch(tensors, outs)
    return outs
