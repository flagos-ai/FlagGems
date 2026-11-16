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

"""Fused foreach minimum (out-of-place) for the Gems `run(self, other)` contract.

Architecture
------------
* One 1-D launch services the whole pair list, so a list of N tensors costs one
  kernel launch instead of N.
* Two launch shapes:
  - uniform dense list (all pairs the same dtype, shape, contiguous layout and
    numel divisible by BLOCK): the smallest possible signature -- one pointer
    per operand plus a single blocks-per-tensor count, slot decoded branch-free
    with pid // bpt -- and no tail mask at all.  For the measured 64x64x4
    workloads ~83% of the wall time is host dispatch, so argument count is the
    dominant cost there.
  - general path: per-slot (ptr, ptr, ptr, numel, scalar) arguments with
    cumulative block-offset decoding, bucketed to 1/2/4/8 slots.
* Layout handling keeps every access stride-1 in storage order:
  - tensors that are non-overlapping-and-dense (contiguous or permuted-dense)
    with a matching partner stride pattern are streamed linearly; the output is
    `empty_like(self)` so its strides equal the reference's.
  - everything else (sliced / mismatched layouts / broadcasting) falls back to a
    rank<=6 strided kernel with a contiguous logical-order output, which is what
    the reference produces for non-dense inputs.
* NaN semantics follow ATen minimum: isnan(a) -> a, isnan(b) -> b, else
  (a < b) ? a : b (also gives ATen's signed-zero behaviour).
* Overloads: tensor list vs tensor list, tensor list vs scalar, tensor list vs
  scalar list. Mismatched lengths and empty lists raise like the reference.
"""

import torch
import triton
import triton.language as tl

_BLOCK = 1024
_NUM_WARPS = 4


def _geom(numel):
    """Launch geometry by element count.

    Measured on the target twice (rounds 4 and 5): the 1G-element regime gains
    ~1.5% from a wider block with more warps (fewer CTAs, better DRAM row
    locality), while the 16.7M- and 4K-element regimes prefer the narrow block.
    The choice depends only on the amount of work, so it transfers across device
    families.
    """
    if numel >= (1 << 28) and numel % 4096 == 0:
        return 4096, 4
    return _BLOCK, _NUM_WARPS


@triton.jit
def _min2(a, bl, sf, si, MODE: tl.constexpr, SCALAR: tl.constexpr):
    """NaN-propagating minimum of two operands in the promoted compute dtype."""
    if MODE == 0:  # float32 compute
        ca = a.to(tl.float32)
        if SCALAR:
            cb = (tl.zeros_like(ca) + sf).to(tl.float32)
        else:
            cb = bl.to(tl.float32)
        r = tl.where(ca != ca, ca, tl.where(cb != cb, cb, tl.where(ca < cb, ca, cb)))
    elif MODE == 1:  # float64 compute
        ca = a.to(tl.float64)
        if SCALAR:
            cb = (tl.zeros_like(ca) + sf).to(tl.float64)
        else:
            cb = bl.to(tl.float64)
        r = tl.where(ca != ca, ca, tl.where(cb != cb, cb, tl.where(ca < cb, ca, cb)))
    elif MODE == 2:  # int32 compute
        ca = a.to(tl.int32)
        if SCALAR:
            cb = (tl.zeros_like(ca) + si).to(tl.int32)
        else:
            cb = bl.to(tl.int32)
        r = tl.where(ca < cb, ca, cb)
    else:  # int64 compute
        ca = a.to(tl.int64)
        if SCALAR:
            cb = (tl.zeros_like(ca) + si).to(tl.int64)
        else:
            cb = bl.to(tl.int64)
        r = tl.where(ca < cb, ca, cb)
    return r


@triton.jit
def _slot(
    a_ptr,
    b_ptr,
    o_ptr,
    n,
    lb,
    sf,
    si,
    BLOCK: tl.constexpr,
    MODE: tl.constexpr,
    SCALAR: tl.constexpr,
):
    offs = lb * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    a = tl.load(a_ptr + offs, mask=m, other=0, eviction_policy="evict_first")
    if SCALAR:
        bl = a
    else:
        bl = tl.load(b_ptr + offs, mask=m, other=0, eviction_policy="evict_first")
    r = _min2(a, bl, sf, si, MODE, SCALAR)
    tl.store(o_ptr + offs, r.to(o_ptr.dtype.element_ty), mask=m, cache_modifier=".cs")


@triton.jit
def _slot_nb(a_ptr, b_ptr, o_ptr, lb, BLOCK: tl.constexpr, MODE: tl.constexpr):
    """Tail-mask-free slot: the caller guarantees numel is a multiple of BLOCK."""
    offs = lb * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(a_ptr + offs, eviction_policy="evict_first")
    bl = tl.load(b_ptr + offs, eviction_policy="evict_first")
    r = _min2(a, bl, 0.0, 0, MODE, False)
    tl.store(o_ptr + offs, r.to(o_ptr.dtype.element_ty), cache_modifier=".cs")


@triton.jit
def _fusedeq2(a0, a1, b0, b1, o0, o1, bpt, BLOCK: tl.constexpr, MODE: tl.constexpr):
    pid = tl.program_id(0)
    t = pid // bpt
    lb = pid - t * bpt
    if t == 0:
        _slot_nb(a0, b0, o0, lb, BLOCK, MODE)
    else:
        _slot_nb(a1, b1, o1, lb, BLOCK, MODE)


@triton.jit
def _fusedeq4(
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
    bpt,
    BLOCK: tl.constexpr,
    MODE: tl.constexpr,
):
    pid = tl.program_id(0)
    t = pid // bpt
    lb = pid - t * bpt
    if t == 0:
        _slot_nb(a0, b0, o0, lb, BLOCK, MODE)
    elif t == 1:
        _slot_nb(a1, b1, o1, lb, BLOCK, MODE)
    elif t == 2:
        _slot_nb(a2, b2, o2, lb, BLOCK, MODE)
    else:
        _slot_nb(a3, b3, o3, lb, BLOCK, MODE)


@triton.jit
def _fused1(
    a0,
    b0,
    o0,
    n0,
    s0f,
    s0i,
    BLOCK: tl.constexpr,
    MODE: tl.constexpr,
    SCALAR: tl.constexpr,
):
    _slot(a0, b0, o0, n0, tl.program_id(0), s0f, s0i, BLOCK, MODE, SCALAR)


@triton.jit
def _fused2(
    a0,
    b0,
    o0,
    n0,
    s0f,
    s0i,
    a1,
    b1,
    o1,
    n1,
    s1f,
    s1i,
    k1,
    BLOCK: tl.constexpr,
    MODE: tl.constexpr,
    SCALAR: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < k1:
        _slot(a0, b0, o0, n0, pid, s0f, s0i, BLOCK, MODE, SCALAR)
    else:
        _slot(a1, b1, o1, n1, pid - k1, s1f, s1i, BLOCK, MODE, SCALAR)


@triton.jit
def _fused4(
    a0,
    b0,
    o0,
    n0,
    s0f,
    s0i,
    a1,
    b1,
    o1,
    n1,
    s1f,
    s1i,
    a2,
    b2,
    o2,
    n2,
    s2f,
    s2i,
    a3,
    b3,
    o3,
    n3,
    s3f,
    s3i,
    k1,
    k2,
    k3,
    BLOCK: tl.constexpr,
    MODE: tl.constexpr,
    SCALAR: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < k1:
        _slot(a0, b0, o0, n0, pid, s0f, s0i, BLOCK, MODE, SCALAR)
    elif pid < k2:
        _slot(a1, b1, o1, n1, pid - k1, s1f, s1i, BLOCK, MODE, SCALAR)
    elif pid < k3:
        _slot(a2, b2, o2, n2, pid - k2, s2f, s2i, BLOCK, MODE, SCALAR)
    else:
        _slot(a3, b3, o3, n3, pid - k3, s3f, s3i, BLOCK, MODE, SCALAR)


@triton.jit
def _fused8(
    a0,
    b0,
    o0,
    n0,
    s0f,
    s0i,
    a1,
    b1,
    o1,
    n1,
    s1f,
    s1i,
    a2,
    b2,
    o2,
    n2,
    s2f,
    s2i,
    a3,
    b3,
    o3,
    n3,
    s3f,
    s3i,
    a4,
    b4,
    o4,
    n4,
    s4f,
    s4i,
    a5,
    b5,
    o5,
    n5,
    s5f,
    s5i,
    a6,
    b6,
    o6,
    n6,
    s6f,
    s6i,
    a7,
    b7,
    o7,
    n7,
    s7f,
    s7i,
    k1,
    k2,
    k3,
    k4,
    k5,
    k6,
    k7,
    BLOCK: tl.constexpr,
    MODE: tl.constexpr,
    SCALAR: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < k1:
        _slot(a0, b0, o0, n0, pid, s0f, s0i, BLOCK, MODE, SCALAR)
    elif pid < k2:
        _slot(a1, b1, o1, n1, pid - k1, s1f, s1i, BLOCK, MODE, SCALAR)
    elif pid < k3:
        _slot(a2, b2, o2, n2, pid - k2, s2f, s2i, BLOCK, MODE, SCALAR)
    elif pid < k4:
        _slot(a3, b3, o3, n3, pid - k3, s3f, s3i, BLOCK, MODE, SCALAR)
    elif pid < k5:
        _slot(a4, b4, o4, n4, pid - k4, s4f, s4i, BLOCK, MODE, SCALAR)
    elif pid < k6:
        _slot(a5, b5, o5, n5, pid - k5, s5f, s5i, BLOCK, MODE, SCALAR)
    elif pid < k7:
        _slot(a6, b6, o6, n6, pid - k6, s6f, s6i, BLOCK, MODE, SCALAR)
    else:
        _slot(a7, b7, o7, n7, pid - k7, s7f, s7i, BLOCK, MODE, SCALAR)


_FUSED = {1: _fused1, 2: _fused2, 4: _fused4, 8: _fused8}
_FUSED_EQ = {2: _fusedeq2, 4: _fusedeq4}


@triton.jit
def _strided_min(
    a_ptr,
    b_ptr,
    o_ptr,
    n,
    sa0,
    sa1,
    sa2,
    sa3,
    sa4,
    sa5,
    sb0,
    sb1,
    sb2,
    sb3,
    sb4,
    sb5,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    sf,
    si,
    BLOCK: tl.constexpr,
    RANK: tl.constexpr,
    MODE: tl.constexpr,
    SCALAR: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    rem = offs
    ia = tl.zeros_like(offs)
    ib = tl.zeros_like(offs)
    if RANK > 0:
        i0 = rem % d0
        rem = rem // d0
        ia += i0 * sa0
        ib += i0 * sb0
    if RANK > 1:
        i1 = rem % d1
        rem = rem // d1
        ia += i1 * sa1
        ib += i1 * sb1
    if RANK > 2:
        i2 = rem % d2
        rem = rem // d2
        ia += i2 * sa2
        ib += i2 * sb2
    if RANK > 3:
        i3 = rem % d3
        rem = rem // d3
        ia += i3 * sa3
        ib += i3 * sb3
    if RANK > 4:
        i4 = rem % d4
        rem = rem // d4
        ia += i4 * sa4
        ib += i4 * sb4
    if RANK > 5:
        i5 = rem % d5
        ia += i5 * sa5
        ib += i5 * sb5
    a = tl.load(a_ptr + ia, mask=m, other=0)
    if SCALAR:
        bl = a
    else:
        bl = tl.load(b_ptr + ib, mask=m, other=0)
    r = _min2(a, bl, sf, si, MODE, SCALAR)
    tl.store(o_ptr + offs, r.to(o_ptr.dtype.element_ty), mask=m)


def _mode_of(dtype):
    if dtype is torch.float32 or dtype is torch.float16 or dtype is torch.bfloat16:
        return 0
    if dtype is torch.float64:
        return 1
    if dtype is torch.int64:
        return 3
    return 2


def _is_dense(t):
    """True when the tensor's storage is exactly its elements in a contiguous order."""
    d = t.dim()
    if d <= 1:
        return False
    perm = sorted(
        range(d), key=lambda k: (-t.stride(k)) if t.size(k) > 1 else (1 << 62)
    )
    return t.permute(perm).is_contiguous()


def _collapse(shape, sa, sb):
    sh = list(shape)
    A = list(sa)
    B = list(sb)
    i = 0
    while i < len(sh):
        if sh[i] == 1:
            del sh[i], A[i], B[i]
        else:
            i += 1
    i = len(sh) - 1
    while i > 0:
        if sh[i] > 1 and A[i - 1] == A[i] * sh[i] and B[i - 1] == B[i] * sh[i]:
            sh[i - 1] *= sh[i]
            A[i - 1] = A[i]
            B[i - 1] = B[i]
            del sh[i], A[i], B[i]
        i -= 1
    if not sh:
        sh, A, B = [1], [0], [0]
    return sh, A, B


def _launch_fused(chunk, mode, scalar_mode):
    n = len(chunk)
    if n <= 2:
        cap = n
    elif n <= 4:
        cap = 4
    else:
        cap = 8
    mx = 0
    for it in chunk:
        if it[3] > mx:
            mx = it[3]
    block, warps = _geom(mx)
    tot = 0
    cums = []
    for it in chunk:
        tot += -(-it[3] // block)
        cums.append(tot)
    src = chunk
    if n < cap:
        p = chunk[0]
        src = chunk + [(p[0], p[0], p[2], 0, 0.0, 0)] * (cap - n)
    args = []
    ap = args.append
    for it in src:
        a = it[0]
        b = it[1]
        o = it[2]
        ap(a.view(torch.uint8) if a.dtype is torch.bool else a)
        ap(b.view(torch.uint8) if b.dtype is torch.bool else b)
        ap(o.view(torch.uint8) if o.dtype is torch.bool else o)
        ap(it[3])
        ap(it[4])
        ap(it[5])
    for i in range(cap - 1):
        ap(cums[i] if i < n else tot)
    _FUSED[cap][(tot,)](
        *args, BLOCK=block, MODE=mode, SCALAR=scalar_mode, num_warps=warps
    )


def _launch_strided(a, b, out, mode, sf, si, scalar_mode):
    if b is None:
        sh, sa, sb = _collapse(a.shape, a.stride(), [0] * a.dim())
    else:
        sh, sa, sb = _collapse(a.shape, a.stride(), b.stride())
    rank = len(sh)
    if rank > 6:
        raise RuntimeError("foreach_minimum fallback supports rank <= 6 only")
    sh = sh + [1] * (6 - rank)
    sa = sa + [0] * (6 - rank)
    sb = sb + [0] * (6 - rank)
    sh.reverse()
    sa.reverse()
    sb.reverse()
    numel = a.numel()
    av = a.view(torch.uint8) if a.dtype is torch.bool else a
    bv = av if b is None else (b.view(torch.uint8) if b.dtype is torch.bool else b)
    ov = out.view(torch.uint8) if out.dtype is torch.bool else out
    _strided_min[(-(-numel // _BLOCK),)](
        av,
        bv,
        ov,
        numel,
        sa[0],
        sa[1],
        sa[2],
        sa[3],
        sa[4],
        sa[5],
        sb[0],
        sb[1],
        sb[2],
        sb[3],
        sb[4],
        sb[5],
        sh[0],
        sh[1],
        sh[2],
        sh[3],
        sh[4],
        sh[5],
        sf,
        si,
        BLOCK=_BLOCK,
        RANK=rank,
        MODE=mode,
        SCALAR=scalar_mode,
        num_warps=_NUM_WARPS,
    )


def _uniform(self, other, n):
    """Smallest-signature launch for an equal-shaped contiguous tensor-vs-tensor list."""
    a0 = self[0]
    b0 = other[0]
    dt = a0.dtype
    if b0.dtype is not dt or b0.shape != a0.shape:
        return None
    numel = a0.numel()
    block, warps = _geom(numel)
    if numel == 0 or numel % block or not a0.is_contiguous() or not b0.is_contiguous():
        return None
    if dt is torch.bool:
        return None
    mode = _mode_of(dt)
    outs = [torch.empty_like(a0)]
    as_ = [a0]
    bs = [b0]
    for i in range(1, n):
        a = self[i]
        b = other[i]
        if (
            a.dtype is not dt
            or a.shape != a0.shape
            or b.dtype is not dt
            or b.shape != a0.shape
            or not a.is_contiguous()
            or not b.is_contiguous()
        ):
            return None
        as_.append(a)
        bs.append(b)
        outs.append(torch.empty_like(a))
    bpt = numel // block
    _FUSED_EQ[n][(n * bpt,)](
        *as_, *bs, *outs, bpt, BLOCK=block, MODE=mode, num_warps=warps
    )
    return tuple(outs)


def foreach_minimum_list(self, other, *args, **kwargs):
    if not isinstance(self, (list, tuple)):
        raise TypeError(
            "_foreach_minimum() received an invalid combination of arguments - "
            "expected a tuple of Tensors for self"
        )
    n = len(self)
    if n == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")
    if torch.is_tensor(other):
        raise TypeError(
            "_foreach_minimum() received an invalid combination of arguments - "
            "got (list, Tensor)"
        )
    list_other = isinstance(other, (list, tuple))
    if list_other:
        m = len(other)
        if m != n:
            raise RuntimeError(
                "Tensor lists must have the same number of tensors, got %d and %d"
                % (n, m)
            )
        scalar_mode = not torch.is_tensor(other[0])
    else:
        scalar_mode = True

    if not scalar_mode and n in (2, 4):
        res = _uniform(self, other, n)
        if res is not None:
            return res

    outs = [None] * n
    plan = []  # [(mode, [slot, ...]), ...] flat-path launches
    cur = None
    cur_mode = -1
    strided = None

    for i in range(n):
        a = self[i]
        if scalar_mode:
            s = other[i] if list_other else other
            b = None
            if isinstance(s, float):
                sf, si = float(s), 0
                odtype = (
                    a.dtype if a.dtype.is_floating_point else torch.get_default_dtype()
                )
            else:
                sf, si = 0.0, int(s)
                odtype = a.dtype
        else:
            b = other[i]
            if b.shape != a.shape:
                b = b.expand(a.shape)
            sf, si = 0.0, 0
            bdtype = b.dtype
            if bdtype is a.dtype:
                odtype = bdtype
            else:
                odtype = torch.promote_types(a.dtype, bdtype)

        numel = a.numel()
        if numel == 0:
            outs[i] = torch.empty_like(
                a, dtype=odtype, memory_format=torch.contiguous_format
            )
            continue

        mode = _mode_of(odtype)

        if a.is_contiguous():
            dense = True
        elif a.dim() > 1:
            dense = _is_dense(a)
        else:
            dense = False

        if dense and (
            b is None
            or (
                b.shape == a.shape
                and (
                    (a.is_contiguous() and b.is_contiguous())
                    or b.stride() == a.stride()
                )
            )
        ):
            out = (
                torch.empty_like(a)
                if odtype is a.dtype
                else torch.empty_like(a, dtype=odtype)
            )
            outs[i] = out
            if mode != cur_mode:
                cur = []
                plan.append((mode, cur))
                cur_mode = mode
            cur.append((a, a if b is None else b, out, numel, sf, si))
        else:
            out = torch.empty_like(
                a, dtype=odtype, memory_format=torch.contiguous_format
            )
            outs[i] = out
            if strided is None:
                strided = []
            strided.append((a, b, out, mode, sf, si))
            cur_mode = -1
            cur = None

    for mode, slots in plan:
        for start in range(0, len(slots), 8):
            _launch_fused(slots[start : start + 8], mode, scalar_mode)

    if strided is not None:
        for a, b, out, mode, sf, si in strided:
            _launch_strided(a, b, out, mode, sf, si, scalar_mode)

    return tuple(outs)
