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

_BLOCK = 1024
_NUM_WARPS = 4
_MAX_RANK = 8
_SLOTS = 4


@triton.jit
def _cm_tile(ap, bp, op, n, pid, BLOCK: tl.constexpr):
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(ap + offs, mask=mask, other=0)
    b = tl.load(bp + offs, mask=mask, other=0)
    out = tl.where(a != a, a, tl.minimum(a, b))
    tl.store(op + offs, out, mask=mask)


@triton.jit
def _cm_flat(a_ptr, b_ptr, out_ptr, n, B_SCALAR: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    if B_SCALAR:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        a = tl.load(a_ptr + offs, mask=mask, other=0)
        b = tl.load(b_ptr)
        out = tl.where(a != a, a, tl.minimum(a, b))
        tl.store(out_ptr + offs, out, mask=mask)
    else:
        _cm_tile(a_ptr, b_ptr, out_ptr, n, pid, BLOCK)


@triton.jit
def _cm_flat_slots(
    a0,
    b0,
    o0,
    n0,
    a1,
    b1,
    o1,
    n1,
    a2,
    b2,
    o2,
    n2,
    a3,
    b3,
    o3,
    n3,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    t = tl.program_id(1)
    if t == 0:
        _cm_tile(a0, b0, o0, n0, pid, BLOCK)
    if K > 1:
        if t == 1:
            _cm_tile(a1, b1, o1, n1, pid, BLOCK)
    if K > 2:
        if t == 2:
            _cm_tile(a2, b2, o2, n2, pid, BLOCK)
    if K > 3:
        if t == 3:
            _cm_tile(a3, b3, o3, n3, pid, BLOCK)


@triton.jit
def _cm_strided(
    a_ptr,
    b_ptr,
    out_ptr,
    n,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6,
    a7,
    b0,
    b1,
    b2,
    b3,
    b4,
    b5,
    b6,
    b7,
    A_CONTIG: tl.constexpr,
    B_SCALAR: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    flat = pid * BLOCK + tl.arange(0, BLOCK)
    mask = flat < n

    if A_CONTIG:
        offs_a = flat
    else:
        rem = flat
        offs_a = tl.zeros_like(flat)
        if RANK > 0:
            i0 = rem % s0
            rem = rem // s0
            offs_a += i0 * a0
        if RANK > 1:
            i1 = rem % s1
            rem = rem // s1
            offs_a += i1 * a1
        if RANK > 2:
            i2 = rem % s2
            rem = rem // s2
            offs_a += i2 * a2
        if RANK > 3:
            i3 = rem % s3
            rem = rem // s3
            offs_a += i3 * a3
        if RANK > 4:
            i4 = rem % s4
            rem = rem // s4
            offs_a += i4 * a4
        if RANK > 5:
            i5 = rem % s5
            rem = rem // s5
            offs_a += i5 * a5
        if RANK > 6:
            i6 = rem % s6
            rem = rem // s6
            offs_a += i6 * a6
        if RANK > 7:
            i7 = rem % s7
            rem = rem // s7
            offs_a += i7 * a7

    a = tl.load(a_ptr + offs_a, mask=mask, other=0)

    if B_SCALAR:
        b = tl.load(b_ptr)
    else:
        rem2 = flat
        offs_b = tl.zeros_like(flat)
        if RANK > 0:
            j0 = rem2 % s0
            rem2 = rem2 // s0
            offs_b += j0 * b0
        if RANK > 1:
            j1 = rem2 % s1
            rem2 = rem2 // s1
            offs_b += j1 * b1
        if RANK > 2:
            j2 = rem2 % s2
            rem2 = rem2 // s2
            offs_b += j2 * b2
        if RANK > 3:
            j3 = rem2 % s3
            rem2 = rem2 // s3
            offs_b += j3 * b3
        if RANK > 4:
            j4 = rem2 % s4
            rem2 = rem2 // s4
            offs_b += j4 * b4
        if RANK > 5:
            j5 = rem2 % s5
            rem2 = rem2 // s5
            offs_b += j5 * b5
        if RANK > 6:
            j6 = rem2 % s6
            rem2 = rem2 // s6
            offs_b += j6 * b6
        if RANK > 7:
            j7 = rem2 % s7
            rem2 = rem2 // s7
            offs_b += j7 * b7
        b = tl.load(b_ptr + offs_b, mask=mask, other=0)

    out = tl.where(a != a, a, tl.minimum(a, b))
    tl.store(out_ptr + flat, out, mask=mask)


def _collapse_dims(shape, sa, sb):
    # Build a minimal set of dims (innermost-first) such that flat row-major
    # enumeration of the collapsed shape reproduces each operand's offsets.
    # The kernel consumes sizes/strides innermost-first (flat % size).
    dims = list(zip(shape, sa, sb))[::-1]
    res = []
    for s, x, y in dims:
        if s == 1:
            continue
        if res:
            ps, px, py = res[-1]
            if x == px * ps and y == py * ps:
                res[-1] = (ps * s, px, py)
                continue
        res.append((s, x, y))
    return res


def _pad(values, fill):
    out = list(values[:_MAX_RANK])
    while len(out) < _MAX_RANK:
        out.append(fill)
    return out


def _launch_flat(a, b, out, n, b_scalar):
    grid = (triton.cdiv(n, _BLOCK),)
    _cm_flat[grid](a, b, out, n, B_SCALAR=b_scalar, BLOCK=_BLOCK, num_warps=_NUM_WARPS)


def _launch_slots(a_list, b_list, outs, idxs):
    ap = []
    bp = []
    op = []
    ns = []
    for t in idxs:
        a = a_list[t]
        out = torch.empty_like(a)
        outs[t] = out
        ap.append(a)
        bp.append(b_list[t])
        op.append(out)
        ns.append(a.numel())
    k = len(ap)
    nmax = max(ns)
    while len(ap) < _SLOTS:
        ap.append(ap[0])
        bp.append(bp[0])
        op.append(op[0])
        ns.append(0)
    grid = (triton.cdiv(nmax, _BLOCK), k)
    _cm_flat_slots[grid](
        ap[0],
        bp[0],
        op[0],
        ns[0],
        ap[1],
        bp[1],
        op[1],
        ns[1],
        ap[2],
        bp[2],
        op[2],
        ns[2],
        ap[3],
        bp[3],
        op[3],
        ns[3],
        K=k,
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
    )


def _launch_strided(a, b, out, n, out_shape, a_strides, b_strides, a_contig, b_scalar):
    dims = _collapse_dims(out_shape, a_strides, b_strides)
    sizes = _pad([d[0] for d in dims], 1)
    ast = _pad([d[1] for d in dims], 0)
    bst = _pad([d[2] for d in dims], 0)
    grid = (triton.cdiv(n, _BLOCK),)
    _cm_strided[grid](
        a,
        b,
        out,
        n,
        sizes[0],
        sizes[1],
        sizes[2],
        sizes[3],
        sizes[4],
        sizes[5],
        sizes[6],
        sizes[7],
        ast[0],
        ast[1],
        ast[2],
        ast[3],
        ast[4],
        ast[5],
        ast[6],
        ast[7],
        bst[0],
        bst[1],
        bst[2],
        bst[3],
        bst[4],
        bst[5],
        bst[6],
        bst[7],
        A_CONTIG=a_contig,
        B_SCALAR=b_scalar,
        RANK=len(dims),
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
    )


def _tensor_pair(a, b):
    n = a.numel()
    if (
        a.shape == b.shape
        and a.dtype == b.dtype
        and a.is_contiguous()
        and b.is_contiguous()
    ):
        out = torch.empty_like(a)
        if n:
            _launch_flat(a, b, out, n, False)
        return out

    out_shape = torch.broadcast_shapes(a.shape, b.shape)
    av = a.expand(out_shape)
    bv = b.expand(out_shape)
    out_dtype = torch.promote_types(a.dtype, b.dtype)
    out = torch.empty(out_shape, dtype=out_dtype, device=a.device)
    n = out.numel()
    if n:
        a_contig = av.is_contiguous()
        if a_contig and bv.is_contiguous():
            _launch_flat(av, bv, out, n, False)
        else:
            _launch_strided(
                av, bv, out, n, out_shape, av.stride(), bv.stride(), a_contig, False
            )
    return out


def _scalar_pair(a, scalar):
    out = torch.empty(a.shape, dtype=a.dtype, device=a.device)
    n = a.numel()
    if n:
        bt = torch.tensor(scalar, dtype=a.dtype, device=a.device)
        if a.is_contiguous():
            _launch_flat(a, bt, out, n, True)
        else:
            _launch_strided(
                a, bt, out, n, tuple(a.shape), a.stride(), a.stride(), False, True
            )
    return out


def _fusable(a, b):
    return (
        a.shape == b.shape
        and a.dtype == b.dtype
        and a.numel() > 0
        and a.is_contiguous()
        and b.is_contiguous()
    )


def _tensor_list(a_list, b_list):
    k = len(a_list)
    outs = [None] * k
    i = 0
    while i < k:
        a = a_list[i]
        if not _fusable(a, b_list[i]):
            outs[i] = _tensor_pair(a, b_list[i])
            i += 1
            continue
        dt = a.dtype
        dev = a.device
        j = i + 1
        while j < k:
            aj = a_list[j]
            if aj.dtype != dt or aj.device != dev or not _fusable(aj, b_list[j]):
                break
            j += 1
        p = i
        while p < j:
            q = p + _SLOTS
            if q > j:
                q = j
            if q - p == 1:
                outs[p] = _tensor_pair(a_list[p], b_list[p])
            else:
                _launch_slots(a_list, b_list, outs, range(p, q))
            p = q
        i = j
    return outs


def foreach_clamp_max_list(self, other):
    if isinstance(self, torch.Tensor):
        self = [self]
    if isinstance(other, torch.Tensor):
        other = [other]
    if isinstance(other, (list, tuple)):
        if len(other) != len(self):
            raise RuntimeError(
                "Tensor list lengths must match (self: {}, other: {})".format(
                    len(self), len(other)
                )
            )
        if len(other) == 0:
            return []
        if isinstance(other[0], torch.Tensor):
            return _tensor_list(self, other)
        return [_scalar_pair(a, s) for a, s in zip(self, other)]
    return [_scalar_pair(a, other) for a in self]
