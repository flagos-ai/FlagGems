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

"""Triton implementation of aten::_foreach_mul.List.

Out-of-place elementwise multiply for paired tensor lists:
    out_i = self_i * other_i   (torch broadcasting, torch type promotion)
A fresh dense output tensor is allocated for every pair; inputs are never
written to and never aliased.

Contiguous equal-numel pairs of one dtype are fused into a single multi-pair
launch (grid = (blocks_per_pair, num_pairs)); the paired pointers travel as
ordinary kernel arguments so no host-to-device pointer table is needed.
"""

import numbers

import torch
import triton
import triton.language as tl

_MAX_RANK = 8
_MAX_FUSE = 4
# Fuse outputs into one allocation (one torch.empty + one unbind) only while the
# combined buffer stays small; large groups keep per-pair allocations so a
# multi-GiB contiguous block is never requested.
_SHARED_BUF_BYTES = 512 << 20


@triton.jit(do_not_specialize=["n"])
def _mul_flat_kernel(
    a_ptr,
    b_ptr,
    o_ptr,
    n,
    b_scalar,
    BLOCK: tl.constexpr,
    I64: tl.constexpr,
    SCALAR_B: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    if I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)
    dt = o_ptr.dtype.element_ty
    if EVEN:
        a = tl.load(a_ptr + offs).to(dt)
        if SCALAR_B:
            b = (tl.zeros([BLOCK], dtype=tl.float32) + b_scalar).to(dt)
        else:
            b = tl.load(b_ptr + offs).to(dt)
        tl.store(o_ptr + offs, a * b)
    else:
        mask = offs < n
        a = tl.load(a_ptr + offs, mask=mask, other=0).to(dt)
        if SCALAR_B:
            b = (tl.zeros([BLOCK], dtype=tl.float32) + b_scalar).to(dt)
        else:
            b = tl.load(b_ptr + offs, mask=mask, other=0).to(dt)
        tl.store(o_ptr + offs, a * b, mask=mask)


@triton.jit(do_not_specialize=["n"])
def _mul_flat_multi_kernel(
    a0,
    a1,
    a2,
    a3,
    b0,
    b1,
    b2,
    b3,
    o0,
    o1,
    o2,
    o3,
    n,
    BLOCK: tl.constexpr,
    I64: tl.constexpr,
    P: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    pair = tl.program_id(1)
    if I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)

    ap = a0
    bp = b0
    op = o0
    if pair == 1:
        ap = a1
        bp = b1
        op = o1
    elif pair == 2:
        ap = a2
        bp = b2
        op = o2
    elif pair == 3:
        ap = a3
        bp = b3
        op = o3

    dt = op.dtype.element_ty
    if EVEN:
        a = tl.load(ap + offs).to(dt)
        b = tl.load(bp + offs).to(dt)
        tl.store(op + offs, a * b)
    else:
        mask = offs < n
        a = tl.load(ap + offs, mask=mask, other=0).to(dt)
        b = tl.load(bp + offs, mask=mask, other=0).to(dt)
        tl.store(op + offs, a * b, mask=mask)


@triton.jit(do_not_specialize=["n", "d0", "d1", "as0", "as1", "bs0", "bs1"])
def _mul_2d_kernel(
    a_ptr,
    b_ptr,
    o_ptr,
    n,
    b_scalar,
    d0,
    d1,
    as0,
    as1,
    bs0,
    bs1,
    BLOCK: tl.constexpr,
    I64: tl.constexpr,
    SCALAR_B: tl.constexpr,
):
    pid = tl.program_id(0)
    if I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    i0 = offs // d1
    i1 = offs % d1
    ao = i0 * as0 + i1 * as1
    dt = o_ptr.dtype.element_ty
    a = tl.load(a_ptr + ao, mask=mask, other=0).to(dt)
    if SCALAR_B:
        b = (tl.zeros([BLOCK], dtype=tl.float32) + b_scalar).to(dt)
    else:
        bo = i0 * bs0 + i1 * bs1
        b = tl.load(b_ptr + bo, mask=mask, other=0).to(dt)
    tl.store(o_ptr + offs, a * b, mask=mask)


@triton.jit(
    do_not_specialize=[
        "n",
        "d0",
        "d1",
        "d2",
        "d3",
        "d4",
        "d5",
        "d6",
        "d7",
        "as0",
        "as1",
        "as2",
        "as3",
        "as4",
        "as5",
        "as6",
        "as7",
        "bs0",
        "bs1",
        "bs2",
        "bs3",
        "bs4",
        "bs5",
        "bs6",
        "bs7",
    ]
)
def _mul_nd_kernel(
    a_ptr,
    b_ptr,
    o_ptr,
    n,
    b_scalar,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    as0,
    as1,
    as2,
    as3,
    as4,
    as5,
    as6,
    as7,
    bs0,
    bs1,
    bs2,
    bs3,
    bs4,
    bs5,
    bs6,
    bs7,
    BLOCK: tl.constexpr,
    I64: tl.constexpr,
    SCALAR_B: tl.constexpr,
):
    pid = tl.program_id(0)
    if I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n

    i7 = offs % d7
    rem = offs // d7
    i6 = rem % d6
    rem = rem // d6
    i5 = rem % d5
    rem = rem // d5
    i4 = rem % d4
    rem = rem // d4
    i3 = rem % d3
    rem = rem // d3
    i2 = rem % d2
    rem = rem // d2
    i1 = rem % d1
    i0 = rem // d1

    ao = (
        i0 * as0
        + i1 * as1
        + i2 * as2
        + i3 * as3
        + i4 * as4
        + i5 * as5
        + i6 * as6
        + i7 * as7
    )
    dt = o_ptr.dtype.element_ty
    a = tl.load(a_ptr + ao, mask=mask, other=0).to(dt)
    if SCALAR_B:
        b = (tl.zeros([BLOCK], dtype=tl.float32) + b_scalar).to(dt)
    else:
        bo = (
            i0 * bs0
            + i1 * bs1
            + i2 * bs2
            + i3 * bs3
            + i4 * bs4
            + i5 * bs5
            + i6 * bs6
            + i7 * bs7
        )
        b = tl.load(b_ptr + bo, mask=mask, other=0).to(dt)
    tl.store(o_ptr + offs, a * b, mask=mask)


def _pick_block(n):
    if n >= (1 << 20):
        return 8192, 8
    return 1024, 4


def _as_strides(t, shape):
    """Right-aligned strides of ``t`` broadcast to ``shape`` (0 for size-1 dims)."""
    rank = len(shape)
    strides = [0] * rank
    base = rank - t.dim()
    for i in range(t.dim()):
        if shape[base + i] != 1:
            strides[base + i] = t.stride(i)
    return strides


def _needs_i64(n, shape, sa, sb):
    if n >= (1 << 31):
        return True
    off = 0
    for d, s in zip(shape, sa):
        off += (d - 1) * abs(s)
    for d, s in zip(shape, sb):
        off += (d - 1) * abs(s)
    return off >= (1 << 31)


def _pad(seq, length):
    pad = [1] * (length - len(seq))
    return pad + list(seq)


def _launch_one(a, b, out, scalar_b=None, shape=None):
    """One launch for one pair (flat or strided)."""
    n = out.numel()
    if n == 0:
        return out
    blk, warps = _pick_block(n)
    scalar = scalar_b is not None
    b_arg = a if scalar else b
    sval = 0.0 if scalar_b is None else float(scalar_b)

    if shape is None:
        i64 = n >= (1 << 31)
        grid = (triton.cdiv(n, blk),)
        _mul_flat_kernel[grid](
            a,
            b_arg,
            out,
            n,
            sval,
            BLOCK=blk,
            I64=i64,
            SCALAR_B=scalar,
            EVEN=(n % blk == 0),
            num_warps=warps,
        )
        return out

    sa = _as_strides(a, shape)
    sb = _as_strides(b, shape)
    i64 = _needs_i64(n, shape, sa, sb)
    rank = len(shape)
    grid = (triton.cdiv(n, blk),)

    if rank <= 2:
        d = _pad(shape, 2)
        pa = _pad(sa, 2)
        pb = _pad(sb, 2)
        _mul_2d_kernel[grid](
            a,
            b_arg,
            out,
            n,
            sval,
            d[0],
            d[1],
            pa[0],
            pa[1],
            pb[0],
            pb[1],
            BLOCK=blk,
            I64=i64,
            SCALAR_B=scalar,
            num_warps=warps,
        )
        return out

    d = _pad(shape, _MAX_RANK)
    pa = _pad(sa, _MAX_RANK)
    pb = _pad(sb, _MAX_RANK)
    _mul_nd_kernel[grid](
        a,
        b_arg,
        out,
        n,
        sval,
        d[0],
        d[1],
        d[2],
        d[3],
        d[4],
        d[5],
        d[6],
        d[7],
        pa[0],
        pa[1],
        pa[2],
        pa[3],
        pa[4],
        pa[5],
        pa[6],
        pa[7],
        pb[0],
        pb[1],
        pb[2],
        pb[3],
        pb[4],
        pb[5],
        pb[6],
        pb[7],
        BLOCK=blk,
        I64=i64,
        SCALAR_B=scalar,
        num_warps=warps,
    )
    return out


def _launch_fused(triples, n):
    """Single launch for up to _MAX_FUSE flat pairs of equal numel/dtype."""
    blk, warps = _pick_block(n)
    pairs = len(triples)
    i64 = n >= (1 << 31)
    grid = (triton.cdiv(n, blk), pairs)
    _mul_flat_multi_kernel[grid](
        triples[0][0],
        triples[1][0],
        triples[2][0],
        triples[3][0],
        triples[0][1],
        triples[1][1],
        triples[2][1],
        triples[3][1],
        triples[0][2],
        triples[1][2],
        triples[2][2],
        triples[3][2],
        n,
        BLOCK=blk,
        I64=i64,
        P=pairs,
        EVEN=(n % blk == 0),
        num_warps=warps,
    )


def _pair(a, b):
    """Single-pair entry (Tensor op Tensor or Tensor op scalar)."""
    if not isinstance(a, torch.Tensor):
        raise TypeError("_foreach_mul expects a Tensor list as the first argument")
    if isinstance(b, numbers.Number) and not isinstance(b, bool):
        out = torch.empty(tuple(a.shape), dtype=a.dtype, device=a.device)
        if out.numel() == 0:
            return out
        if a.is_contiguous():
            return _launch_one(a, a, out, scalar_b=b)
        return _launch_one(a, a, out, scalar_b=b, shape=tuple(a.shape))

    if not isinstance(b, torch.Tensor):
        raise TypeError("_foreach_mul expects a Tensor list as the second argument")

    odt = a.dtype if a.dtype == b.dtype else torch.promote_types(a.dtype, b.dtype)
    if a.shape == b.shape:
        shape = tuple(a.shape)
    else:
        shape = tuple(torch.broadcast_shapes(tuple(a.shape), tuple(b.shape)))
    out = torch.empty(shape, dtype=odt, device=a.device)
    if out.numel() == 0:
        return out
    if a.shape == b.shape and a.is_contiguous() and b.is_contiguous():
        return _launch_one(a, b, out)
    return _launch_one(a, b, out, shape=shape)


def _is_seq(obj):
    return isinstance(obj, (list, tuple))


def foreach_mul_list(self, other):
    if isinstance(self, torch.Tensor):
        if isinstance(other, torch.Tensor) or isinstance(other, numbers.Number):
            return _pair(self, other)
        self_list = [self]
    else:
        self_list = list(self)

    if _is_seq(other):
        other_list = list(other)
        if len(other_list) != len(self_list):
            raise RuntimeError(
                "_foreach_mul(): Tensor list length mismatch: "
                "expected {} elements, got {}".format(len(self_list), len(other_list))
            )
    else:
        other_list = None

    count = len(self_list)
    outs = [None] * count
    groups = {}
    for i in range(count):
        a = self_list[i]
        b = other_list[i] if other_list is not None else other
        if not isinstance(a, torch.Tensor):
            raise TypeError("_foreach_mul expects a Tensor list as the first argument")

        if not isinstance(b, torch.Tensor):
            if isinstance(b, numbers.Number) and not isinstance(b, bool):
                out = torch.empty(tuple(a.shape), dtype=a.dtype, device=a.device)
                outs[i] = out
                if out.numel() == 0:
                    continue
                if a.is_contiguous():
                    _launch_one(a, a, out, scalar_b=b)
                else:
                    _launch_one(a, a, out, scalar_b=b, shape=tuple(a.shape))
                continue
            raise TypeError("_foreach_mul expects a Tensor list as the second argument")

        odt = a.dtype if a.dtype == b.dtype else torch.promote_types(a.dtype, b.dtype)
        same = a.shape == b.shape
        if same:
            shape = tuple(a.shape)
        else:
            shape = tuple(torch.broadcast_shapes(tuple(a.shape), tuple(b.shape)))

        if same and a.is_contiguous() and b.is_contiguous():
            numel = a.numel()
            if numel == 0:
                outs[i] = torch.empty(shape, dtype=odt, device=a.device)
                continue
            groups.setdefault((odt, a.device, numel), []).append((i, a, b, shape))
            continue

        out = torch.empty(shape, dtype=odt, device=a.device)
        outs[i] = out
        if out.numel() != 0:
            _launch_one(a, b, out, shape=shape)

    for (odt, dev, numel), plans in groups.items():
        chunk_bytes = numel * odt.itemsize * min(len(plans), _MAX_FUSE)
        for start in range(0, len(plans), _MAX_FUSE):
            chunk = plans[start : start + _MAX_FUSE]
            if len(chunk) == 1:
                i, a, b, shape = chunk[0]
                out = torch.empty(shape, dtype=odt, device=dev)
                outs[i] = out
                _launch_one(a, b, out)
                continue
            if chunk_bytes <= _SHARED_BUF_BYTES:
                shapes = [p[3] for p in chunk]
                if all(s == shapes[0] for s in shapes):
                    buf = torch.empty((len(chunk),) + shapes[0], dtype=odt, device=dev)
                    parts = buf.unbind(0)
                else:
                    buf = torch.empty(len(chunk) * numel, dtype=odt, device=dev)
                    parts = tuple(
                        buf[k * numel : (k + 1) * numel].view(shapes[k])
                        for k in range(len(chunk))
                    )
            else:
                parts = tuple(torch.empty(p[3], dtype=odt, device=dev) for p in chunk)
            for k, plan in enumerate(chunk):
                outs[plan[0]] = parts[k]
            triples = [(chunk[k][1], chunk[k][2], parts[k]) for k in range(len(chunk))]
            while len(triples) < _MAX_FUSE:
                triples.append(triples[0])
            _launch_fused(triples, numel)

    return outs
