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

"""Triton implementation of ``aten::_foreach_mul.Scalar``.

``run(self, scalar)`` takes a sequence of tensors and one scalar (or a sequence
of scalars) and returns a freshly allocated list of tensors with
``out_i[p] = self_i[p] * s``.

Fast path: inputs whose storage mapping is identical to the freshly allocated
output (i.e. dense tensors -- contiguous *or* transposed/permuted views) are
processed as a flat elementwise map over the storage, which keeps both the load
and the store fully coalesced.  Up to four same-dtype dense tensors share one
launch through a slot-dispatched kernel (cumulative block-start mapping) so a
list of small tensors costs a single launch instead of one per tensor.  Other
layouts use a rank-collapsed strided kernel.
"""

import struct

import torch
import triton
import triton.language as tl

_BLOCK = 1024
_NUM_WARPS = 4
_MAX_RANK = 8
_MAX_SLOTS = 4
_INT32_MAX = 2**31 - 1

_MODE_F32 = 0  # fp16 / bf16 / fp32 -> same dtype, fp32 opmath
_MODE_F64 = 1  # fp64
_MODE_INT = 2  # integral tensor & integral scalar -> same dtype
_MODE_INT2F32 = 3  # integral tensor & float scalar -> default float dtype


@triton.jit
def _mul_body(
    _x_ptr,
    _out_ptr,
    _xoff,
    _ooff,
    _mask,
    _s_hi,
    _s_lo,
    _s_int,
    _MODE: tl.constexpr,
    _SCALAR_PTR: tl.constexpr,
):
    x = tl.load(_x_ptr + _xoff, mask=_mask, other=0)
    if _MODE == 1:
        if _SCALAR_PTR:
            s = tl.load(_s_hi).to(tl.float64)
        else:
            s = _s_hi.to(tl.float64) + _s_lo.to(tl.float64)
        y = x.to(tl.float64) * s
        tl.store(_out_ptr + _ooff, y, mask=_mask)
    elif _MODE == 2:
        if _SCALAR_PTR:
            si = tl.load(_s_hi).to(tl.int64)
        else:
            si = _s_int
        y = x.to(tl.int64) * si
        tl.store(_out_ptr + _ooff, y.to(x.dtype), mask=_mask)
    elif _MODE == 3:
        if _SCALAR_PTR:
            s = tl.load(_s_hi).to(tl.float32)
        else:
            s = _s_hi
        y = x.to(tl.float32) * s
        tl.store(_out_ptr + _ooff, y, mask=_mask)
    else:
        if _SCALAR_PTR:
            s = tl.load(_s_hi).to(tl.float32)
        else:
            s = _s_hi
        y = x.to(tl.float32) * s
        tl.store(_out_ptr + _ooff, y.to(x.dtype), mask=_mask)


@triton.jit
def _mul_flat_kernel(
    x_ptr,
    out_ptr,
    n,
    s_hi,
    s_lo,
    s_int,
    BLOCK: tl.constexpr,
    MODE: tl.constexpr,
    I64OFS: tl.constexpr,
    SCALAR_PTR: tl.constexpr,
):
    pid = tl.program_id(0)
    if I64OFS:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    _mul_body(x_ptr, out_ptr, offs, offs, offs < n, s_hi, s_lo, s_int, MODE, SCALAR_PTR)


@triton.jit
def _mul_slots_kernel(
    x0,
    x1,
    x2,
    x3,
    o0,
    o1,
    o2,
    o3,
    n0,
    n1,
    n2,
    n3,
    b1,
    b2,
    b3,
    s_hi,
    s_lo,
    s_int,
    BLOCK: tl.constexpr,
    MODE: tl.constexpr,
    I64OFS: tl.constexpr,
    SCALAR_PTR: tl.constexpr,
):
    pid = tl.program_id(0)
    slot = (
        (pid >= b1).to(tl.int32) + (pid >= b2).to(tl.int32) + (pid >= b3).to(tl.int32)
    )
    if I64OFS:
        lan = tl.arange(0, BLOCK).to(tl.int64)
    else:
        lan = tl.arange(0, BLOCK)
    if slot < 1:
        if I64OFS:
            offs = pid.to(tl.int64) * BLOCK + lan
        else:
            offs = pid * BLOCK + lan
        _mul_body(x0, o0, offs, offs, offs < n0, s_hi, s_lo, s_int, MODE, SCALAR_PTR)
    elif slot < 2:
        if I64OFS:
            offs = (pid - b1).to(tl.int64) * BLOCK + lan
        else:
            offs = (pid - b1) * BLOCK + lan
        _mul_body(x1, o1, offs, offs, offs < n1, s_hi, s_lo, s_int, MODE, SCALAR_PTR)
    elif slot < 3:
        if I64OFS:
            offs = (pid - b2).to(tl.int64) * BLOCK + lan
        else:
            offs = (pid - b2) * BLOCK + lan
        _mul_body(x2, o2, offs, offs, offs < n2, s_hi, s_lo, s_int, MODE, SCALAR_PTR)
    else:
        if I64OFS:
            offs = (pid - b3).to(tl.int64) * BLOCK + lan
        else:
            offs = (pid - b3) * BLOCK + lan
        _mul_body(x3, o3, offs, offs, offs < n3, s_hi, s_lo, s_int, MODE, SCALAR_PTR)


@triton.jit
def _flat_pair(xp, op, pid, n, s, BLOCK: tl.constexpr, EXACT: tl.constexpr):
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if EXACT:
        x = tl.load(xp + offs)
        tl.store(op + offs, (x.to(tl.float32) * s).to(x.dtype))
    else:
        m = offs < n
        x = tl.load(xp + offs, mask=m, other=0)
        tl.store(op + offs, (x.to(tl.float32) * s).to(x.dtype), mask=m)


@triton.jit
def _mul4_kernel(
    x0, x1, x2, x3, o0, o1, o2, o3, n0, n1, n2, n3, s, EXACT: tl.constexpr
):
    """Four homogeneous contiguous tensors in one launch (fp16/bf16/fp32).

    BLOCK is the literal 1024 and the cumulative block starts are derived from
    the element counts inside the kernel, so the launch carries no extra
    scalar arguments.  EXACT drops the bounds mask for exact multiples.
    """
    pid = tl.program_id(0)
    c1 = (n0 + 1023) >> 10
    c2 = c1 + ((n1 + 1023) >> 10)
    c3 = c2 + ((n2 + 1023) >> 10)
    slot = (
        (pid >= c1).to(tl.int32) + (pid >= c2).to(tl.int32) + (pid >= c3).to(tl.int32)
    )
    if slot < 1:
        _flat_pair(x0, o0, pid, n0, s, 1024, EXACT)
    elif slot < 2:
        _flat_pair(x1, o1, pid - c1, n1, s, 1024, EXACT)
    elif slot < 3:
        _flat_pair(x2, o2, pid - c2, n2, s, 1024, EXACT)
    else:
        _flat_pair(x3, o3, pid - c3, n3, s, 1024, EXACT)


@triton.jit
def _mul_strided_kernel(
    x_ptr,
    out_ptr,
    n,
    s_hi,
    s_lo,
    s_int,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
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
    BLOCK: tl.constexpr,
    MODE: tl.constexpr,
    RANK: tl.constexpr,
    I64OFS: tl.constexpr,
    SCALAR_PTR: tl.constexpr,
):
    pid = tl.program_id(0)
    if I64OFS:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        xoff = tl.zeros([BLOCK], dtype=tl.int64)
        ooff = tl.zeros([BLOCK], dtype=tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        xoff = tl.zeros([BLOCK], dtype=tl.int32)
        ooff = tl.zeros([BLOCK], dtype=tl.int32)
    rem = offs
    if RANK > 7:
        i = rem % d7
        rem = rem // d7
        xoff += i * xs7
        ooff += i * os7
    if RANK > 6:
        i = rem % d6
        rem = rem // d6
        xoff += i * xs6
        ooff += i * os6
    if RANK > 5:
        i = rem % d5
        rem = rem // d5
        xoff += i * xs5
        ooff += i * os5
    if RANK > 4:
        i = rem % d4
        rem = rem // d4
        xoff += i * xs4
        ooff += i * os4
    if RANK > 3:
        i = rem % d3
        rem = rem // d3
        xoff += i * xs3
        ooff += i * os3
    if RANK > 2:
        i = rem % d2
        rem = rem // d2
        xoff += i * xs2
        ooff += i * os2
    if RANK > 1:
        i = rem % d1
        rem = rem // d1
        xoff += i * xs1
        ooff += i * os1
    if RANK > 0:
        i = rem % d0
        xoff += i * xs0
        ooff += i * os0

    _mul_body(x_ptr, out_ptr, xoff, ooff, offs < n, s_hi, s_lo, s_int, MODE, SCALAR_PTR)


def _split_scalar(s):
    """Split a Python float into an exactly summable fp32 hi/lo pair."""
    fs = float(s)
    hi = struct.unpack("f", struct.pack("f", fs))[0]
    return hi, fs - hi


def _mode_of(dtype, scalar):
    if dtype == torch.float64:
        return _MODE_F64
    if dtype.is_floating_point:
        return _MODE_F32
    if isinstance(scalar, float):
        return _MODE_INT2F32
    return _MODE_INT


def _is_dense_layout(t):
    """Metadata-only equivalent of ``is_non_overlapping_and_dense``."""
    sizes = t.shape
    strides = t.stride()
    order = sorted(range(len(sizes)), key=lambda d: strides[d])
    dense_stride = 1
    for d in order:
        sz = sizes[d]
        if sz == 0:
            return True
        if sz == 1:
            continue
        if strides[d] != dense_stride:
            return False
        dense_stride *= sz
    return True


def _flat_ok(x, out):
    if x.is_contiguous():
        return out.is_contiguous()
    return x.stride() == out.stride() and _is_dense_layout(x)


def _strided_meta(t, out):
    """Collapse dims of a non-dense tensor into at most _MAX_RANK entries."""
    dims = []
    for d in range(t.dim() - 1, -1, -1):
        sz = t.size(d)
        if sz == 1:
            continue
        dims.append((sz, t.stride(d), out.stride(d)))
    merged = []
    for sz, xs, os_ in dims:
        if merged:
            psz, pxs, pos = merged[-1]
            if xs == pxs * psz and os_ == pos * psz:
                merged[-1] = (psz * sz, pxs, pos)
                continue
        merged.append((sz, xs, os_))
    merged.reverse()
    return merged


def _launch_flat(x, out, n, mode, s_hi, s_lo, s_int, scalar_ptr):
    grid = ((n + _BLOCK - 1) // _BLOCK,)
    _mul_flat_kernel[grid](
        x,
        out,
        n,
        s_hi,
        s_lo,
        s_int,
        BLOCK=_BLOCK,
        MODE=mode,
        I64OFS=n > _INT32_MAX,
        SCALAR_PTR=scalar_ptr,
        num_warps=_NUM_WARPS,
    )


def _launch_slots(xs, outs, ns, mode, s_hi, s_lo, s_int, scalar_ptr):
    k = len(xs)
    total = 0
    bounds = []
    for n in ns:
        total += (n + _BLOCK - 1) // _BLOCK
        bounds.append(total)
    b1 = bounds[0] if k > 1 else total
    b2 = bounds[1] if k > 2 else total
    b3 = bounds[2] if k > 3 else total
    px = list(xs)
    po = list(outs)
    pn = list(ns)
    while len(px) < _MAX_SLOTS:
        px.append(px[0])
        po.append(po[0])
        pn.append(0)
    _mul_slots_kernel[(total,)](
        px[0],
        px[1],
        px[2],
        px[3],
        po[0],
        po[1],
        po[2],
        po[3],
        pn[0],
        pn[1],
        pn[2],
        pn[3],
        b1,
        b2,
        b3,
        s_hi,
        s_lo,
        s_int,
        BLOCK=_BLOCK,
        MODE=mode,
        I64OFS=max(pn) > _INT32_MAX,
        SCALAR_PTR=scalar_ptr,
        num_warps=_NUM_WARPS,
    )


def _launch_strided(x, out, n, mode, s_hi, s_lo, s_int, scalar_ptr, meta):
    rank = len(meta)
    if rank > _MAX_RANK:
        # Extremely rare: peel the outermost dim and recurse (views only).
        for i in range(x.size(0)):
            _dispatch_one(x[i], out[i], mode, s_hi, s_lo, s_int, scalar_ptr)
        return
    sizes = [0] * _MAX_RANK
    xstr = [0] * _MAX_RANK
    ostr = [0] * _MAX_RANK
    for i in range(rank):
        sizes[i], xstr[i], ostr[i] = meta[i]
    max_off = 0
    for sz, xs in zip(sizes[:rank], xstr[:rank]):
        max_off += (sz - 1) * xs
    grid = ((n + _BLOCK - 1) // _BLOCK,)
    _mul_strided_kernel[grid](
        x,
        out,
        n,
        s_hi,
        s_lo,
        s_int,
        sizes[0],
        sizes[1],
        sizes[2],
        sizes[3],
        sizes[4],
        sizes[5],
        sizes[6],
        sizes[7],
        xstr[0],
        xstr[1],
        xstr[2],
        xstr[3],
        xstr[4],
        xstr[5],
        xstr[6],
        xstr[7],
        ostr[0],
        ostr[1],
        ostr[2],
        ostr[3],
        ostr[4],
        ostr[5],
        ostr[6],
        ostr[7],
        BLOCK=_BLOCK,
        MODE=mode,
        RANK=rank,
        I64OFS=max_off > _INT32_MAX,
        SCALAR_PTR=scalar_ptr,
        num_warps=_NUM_WARPS,
    )


def _dispatch_one(x, out, mode, s_hi, s_lo, s_int, scalar_ptr):
    n = out.numel()
    if n == 0:
        return
    if _flat_ok(x, out):
        _launch_flat(x, out, n, mode, s_hi, s_lo, s_int, scalar_ptr)
        return
    _launch_strided(
        x, out, n, mode, s_hi, s_lo, s_int, scalar_ptr, _strided_meta(x, out)
    )


def _alloc_out(x, mode):
    if mode == _MODE_INT2F32:
        return torch.empty(x.shape, dtype=torch.get_default_dtype(), device=x.device)
    return torch.empty_like(x)


def foreach_mul_scalar(self, scalar):
    if isinstance(self, torch.Tensor):
        tensors = [self]
    elif type(self) is list:
        tensors = self
    else:
        tensors = list(self)
    n_t = len(tensors)

    # --- device-resident 0-dim tensor scalar -------------------------------
    if isinstance(scalar, torch.Tensor):
        outs = []
        for x in tensors:
            mode = _mode_of(x.dtype, 0.0)
            out = _alloc_out(x, mode)
            outs.append(out)
            if out.numel() != 0:
                _dispatch_one(x, out, mode, scalar, 0.0, 0, True)
        return outs

    # --- per-tensor scalar list -------------------------------------------
    if not isinstance(scalar, (int, float)):
        sl = list(scalar)
        if len(sl) != n_t:
            raise RuntimeError(
                "foreach_mul: the length of the scalar list (%d) must match "
                "the length of the tensor list (%d)" % (len(sl), n_t)
            )
        outs = []
        for i in range(n_t):
            x = tensors[i]
            s = sl[i]
            mode = _mode_of(x.dtype, s)
            out = _alloc_out(x, mode)
            outs.append(out)
            if out.numel() == 0:
                continue
            if mode == _MODE_INT:
                _dispatch_one(x, out, mode, 0.0, 0.0, int(s), False)
            else:
                hi, lo = _split_scalar(s)
                _dispatch_one(x, out, mode, hi, lo, 0, False)
        return outs

    # --- one uniform scalar ------------------------------------------------
    s_single = scalar
    if isinstance(s_single, float):
        s_hi, s_lo = _split_scalar(s_single)
        s_int = 0
    else:
        s_hi, s_lo = float(s_single), 0.0
        s_int = int(s_single)

    # Homogeneous four-tensor call: one fully specialised launch.
    if n_t == 4 and isinstance(s_single, float):
        t0, t1, t2, t3 = tensors
        dt = t0.dtype
        if (
            (dt == torch.float16 or dt == torch.float32 or dt == torch.bfloat16)
            and t1.dtype == dt
            and t2.dtype == dt
            and t3.dtype == dt
            and t0.is_contiguous()
            and t1.is_contiguous()
            and t2.is_contiguous()
            and t3.is_contiguous()
        ):
            n0 = t0.numel()
            n1 = t1.numel()
            n2 = t2.numel()
            n3 = t3.numel()
            if n0 and n1 and n2 and n3:
                o0 = torch.empty_like(t0)
                o1 = torch.empty_like(t1)
                o2 = torch.empty_like(t2)
                o3 = torch.empty_like(t3)
                _mul4_kernel[
                    (
                        ((n0 + 1023) >> 10)
                        + ((n1 + 1023) >> 10)
                        + ((n2 + 1023) >> 10)
                        + ((n3 + 1023) >> 10),
                    )
                ](
                    t0,
                    t1,
                    t2,
                    t3,
                    o0,
                    o1,
                    o2,
                    o3,
                    n0,
                    n1,
                    n2,
                    n3,
                    s_single,
                    EXACT=not ((n0 | n1 | n2 | n3) & 1023),
                )
                return [o0, o1, o2, o3]

    outs = []
    append = outs.append
    flat_groups = []
    group = None
    others = []
    for i in range(n_t):
        x = tensors[i]
        mode = _mode_of(x.dtype, s_single)
        out = _alloc_out(x, mode)
        append(out)
        if out.numel() == 0:
            continue
        if _flat_ok(x, out):
            if (
                group is not None
                and group[0] == mode
                and group[1] == x.dtype
                and len(group[2]) < _MAX_SLOTS
            ):
                group[2].append(i)
            else:
                group = (mode, x.dtype, [i])
                flat_groups.append(group)
        else:
            others.append(i)

    for mode, _dtype, idxs in flat_groups:
        if len(idxs) == 1:
            i = idxs[0]
            out = outs[i]
            _launch_flat(tensors[i], out, out.numel(), mode, s_hi, s_lo, s_int, False)
        else:
            xs = [tensors[i] for i in idxs]
            os_ = [outs[i] for i in idxs]
            _launch_slots(
                xs, os_, [o.numel() for o in os_], mode, s_hi, s_lo, s_int, False
            )

    for i in others:
        x = tensors[i]
        _dispatch_one(x, outs[i], _mode_of(x.dtype, s_single), s_hi, s_lo, s_int, False)

    return outs
