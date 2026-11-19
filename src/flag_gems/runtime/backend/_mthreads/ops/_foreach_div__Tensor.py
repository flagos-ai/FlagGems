"""Triton implementation of aten::_foreach_div_.Tensor (and .Scalar).

In-place elementwise division of every tensor in ``self`` by one 0-dim divisor
tensor.  Storage, strides and shapes of every mutated tensor are preserved and
the call returns ``None`` like the in-place ATen op.

Two launch shapes:
  * ``_div_fused``  - one launch for a group of up to 8 contiguous tensors that
    share a dtype.  The grid is the exact number of blocks over the group, so a
    program never straddles two tensors and no element is visited twice.
  * ``_div_strided`` - one launch per non-contiguous tensor, using the tensor's
    unit-stride axis as the coalesced inner axis.
"""

import torch
import triton
import triton.language as tl

_MAX_GROUP = 8
_MAX_OUTER = 8


@triton.jit
def _div_fused(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    n0,
    n1,
    n2,
    n3,
    n4,
    n5,
    n6,
    n7,
    d_ptr,
    d_scalar,
    BLOCK: tl.constexpr,
    IS_TENSOR: tl.constexpr,
    IS_F64: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    if IS_TENSOR:
        dvv = tl.load(d_ptr)
    else:
        dvv = d_scalar
    if IS_F64:
        dv = dvv.to(tl.float64)
    else:
        dv = dvv.to(tl.float32)
    lane = tl.arange(0, BLOCK)
    b = BLOCK - 1
    c0 = (n0 + b) // BLOCK
    c1 = c0 + (n1 + b) // BLOCK
    c2 = c1 + (n2 + b) // BLOCK
    c3 = c2 + (n3 + b) // BLOCK
    c4 = c3 + (n4 + b) // BLOCK
    c5 = c4 + (n5 + b) // BLOCK
    c6 = c5 + (n6 + b) // BLOCK
    c7 = c6 + (n7 + b) // BLOCK
    if pid < c0:
        off = pid * BLOCK + lane
        mk = off < n0
        v = tl.load(p0 + off, mask=mk, other=0.0)
        if IS_F64:
            r = v / dv
        else:
            r = v.to(tl.float32) / dv
        tl.store(p0 + off, r.to(p0.dtype.element_ty), mask=mk)
    elif pid < c1:
        off = (pid - c0) * BLOCK + lane
        mk = off < n1
        v = tl.load(p1 + off, mask=mk, other=0.0)
        if IS_F64:
            r = v / dv
        else:
            r = v.to(tl.float32) / dv
        tl.store(p1 + off, r.to(p1.dtype.element_ty), mask=mk)
    elif pid < c2:
        off = (pid - c1) * BLOCK + lane
        mk = off < n2
        v = tl.load(p2 + off, mask=mk, other=0.0)
        if IS_F64:
            r = v / dv
        else:
            r = v.to(tl.float32) / dv
        tl.store(p2 + off, r.to(p2.dtype.element_ty), mask=mk)
    elif pid < c3:
        off = (pid - c2) * BLOCK + lane
        mk = off < n3
        v = tl.load(p3 + off, mask=mk, other=0.0)
        if IS_F64:
            r = v / dv
        else:
            r = v.to(tl.float32) / dv
        tl.store(p3 + off, r.to(p3.dtype.element_ty), mask=mk)
    elif pid < c4:
        off = (pid - c3) * BLOCK + lane
        mk = off < n4
        v = tl.load(p4 + off, mask=mk, other=0.0)
        if IS_F64:
            r = v / dv
        else:
            r = v.to(tl.float32) / dv
        tl.store(p4 + off, r.to(p4.dtype.element_ty), mask=mk)
    elif pid < c5:
        off = (pid - c4) * BLOCK + lane
        mk = off < n5
        v = tl.load(p5 + off, mask=mk, other=0.0)
        if IS_F64:
            r = v / dv
        else:
            r = v.to(tl.float32) / dv
        tl.store(p5 + off, r.to(p5.dtype.element_ty), mask=mk)
    elif pid < c6:
        off = (pid - c5) * BLOCK + lane
        mk = off < n6
        v = tl.load(p6 + off, mask=mk, other=0.0)
        if IS_F64:
            r = v / dv
        else:
            r = v.to(tl.float32) / dv
        tl.store(p6 + off, r.to(p6.dtype.element_ty), mask=mk)
    elif pid < c7:
        off = (pid - c6) * BLOCK + lane
        mk = off < n7
        v = tl.load(p7 + off, mask=mk, other=0.0)
        if IS_F64:
            r = v / dv
        else:
            r = v.to(tl.float32) / dv
        tl.store(p7 + off, r.to(p7.dtype.element_ty), mask=mk)


@triton.jit
def _div_strided(
    x_ptr,
    d_ptr,
    d_scalar,
    inner,
    si,
    nb_inner,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    s8,
    t1,
    t2,
    t3,
    t4,
    t5,
    t6,
    t7,
    t8,
    BLOCK: tl.constexpr,
    RANK: tl.constexpr,
    IS_TENSOR: tl.constexpr,
    IS_F64: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    row = pid // nb_inner
    blk = pid - row * nb_inner
    base = row * 0
    r = row
    if RANK >= 1:
        base = base + (r % s1) * t1
        r = r // s1
    if RANK >= 2:
        base = base + (r % s2) * t2
        r = r // s2
    if RANK >= 3:
        base = base + (r % s3) * t3
        r = r // s3
    if RANK >= 4:
        base = base + (r % s4) * t4
        r = r // s4
    if RANK >= 5:
        base = base + (r % s5) * t5
        r = r // s5
    if RANK >= 6:
        base = base + (r % s6) * t6
        r = r // s6
    if RANK >= 7:
        base = base + (r % s7) * t7
        r = r // s7
    if RANK >= 8:
        base = base + (r % s8) * t8
        r = r // s8
    offs = blk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < inner
    idx = base + offs * si
    v = tl.load(x_ptr + idx, mask=mask, other=0.0)
    if IS_TENSOR:
        dvv = tl.load(d_ptr)
    else:
        dvv = d_scalar
    if IS_F64:
        res = v / dvv.to(tl.float64)
    else:
        res = v.to(tl.float32) / dvv.to(tl.float32)
    tl.store(x_ptr + idx, res.to(x_ptr.dtype.element_ty), mask=mask)


def _strided_plan(t):
    """Return (inner, inner_stride, rows, outer_sizes, outer_strides)."""
    shape = t.shape
    strides = t.stride()
    nd = len(shape)
    dims = [d for d in range(nd) if shape[d] != 1]
    if not dims:
        return None
    unit = None
    for d in dims:
        if strides[d] == 1:
            unit = d
            break
    if unit is None:
        unit = dims[0]
        for d in dims:
            if abs(strides[d]) < abs(strides[unit]):
                unit = d
    inner = shape[unit]
    si = strides[unit]
    outer = [d for d in dims if d != unit]
    ss = [shape[d] for d in outer]
    tt = [strides[d] for d in outer]
    rows = 1
    for s in ss:
        rows *= s
    return inner, si, rows, ss, tt


def _launch_strided(t, dptr, dval, is_tensor):
    plan = _strided_plan(t)
    if plan is None:
        return
    inner, si, rows, ss, tt = plan
    if len(ss) > _MAX_OUTER:
        # Extremely rare: fold the leading dimension away with zero-copy views.
        for i in range(ss[0]):
            _launch_strided(t.select(0, i), dptr, dval, is_tensor)
        return
    rank = len(ss)
    ss = ss + [1] * (_MAX_OUTER - rank)
    tt = tt + [0] * (_MAX_OUTER - rank)
    block = (
        1024 if inner > 1024 else (triton.next_power_of_2(inner) if inner > 16 else 16)
    )
    nb = (inner + block - 1) // block
    _div_strided[(rows * nb,)](
        t,
        dptr,
        dval,
        inner,
        si,
        nb,
        ss[0],
        ss[1],
        ss[2],
        ss[3],
        ss[4],
        ss[5],
        ss[6],
        ss[7],
        tt[0],
        tt[1],
        tt[2],
        tt[3],
        tt[4],
        tt[5],
        tt[6],
        tt[7],
        BLOCK=block,
        RANK=rank,
        IS_TENSOR=is_tensor,
        IS_F64=t.dtype == torch.float64,
    )


def _launch_group(group, dptr, dval, is_tensor):
    """One fused launch for up to _MAX_GROUP contiguous, same-dtype tensors."""
    k = len(group)
    sizes = [t.numel() for t in group]
    nmax = max(sizes)
    if nmax == 0:
        return
    block = (
        1024 if nmax >= 1024 else (triton.next_power_of_2(nmax) if nmax > 16 else 16)
    )
    total = 0
    for s in sizes:
        total += (s + block - 1) // block
    if k < _MAX_GROUP:
        sizes = sizes + [0] * (_MAX_GROUP - k)
        group = group + [group[0]] * (_MAX_GROUP - k)
    _div_fused[(total,)](
        group[0],
        group[1],
        group[2],
        group[3],
        group[4],
        group[5],
        group[6],
        group[7],
        sizes[0],
        sizes[1],
        sizes[2],
        sizes[3],
        sizes[4],
        sizes[5],
        sizes[6],
        sizes[7],
        dptr,
        dval,
        BLOCK=block,
        IS_TENSOR=is_tensor,
        IS_F64=group[0].dtype == torch.float64,
    )


def _foreach_div__Tensor(self, other):
    tensors = self if isinstance(self, list) else list(self)
    if not tensors:
        raise RuntimeError("Tensor list must have at least one tensor.")
    if isinstance(other, torch.Tensor):
        if other.numel() != 1:
            raise RuntimeError(
                "scalar tensor expected to be 0 dim but has %d dimensions and %d elements."
                % (other.dim(), other.numel())
            )
        dptr = other
        dval = 0.0
        is_tensor = True
    else:
        dptr = tensors[0]
        dval = float(other)
        is_tensor = False

    n = len(tensors)
    i = 0
    while i < n:
        t = tensors[i]
        if not t.is_contiguous():
            _launch_strided(t, dptr, dval, is_tensor)
            i += 1
            continue
        dt = t.dtype
        j = i + 1
        while (
            j < n
            and j - i < _MAX_GROUP
            and tensors[j].dtype is dt
            and tensors[j].is_contiguous()
        ):
            j += 1
        _launch_group(tensors[i:j], dptr, dval, is_tensor)
        i = j
    return None
