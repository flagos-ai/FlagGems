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

"""Fused Triton implementation of ``aten::_foreach_add_.ScalarList`` (in-place).

Probed target semantics (torch 2.10 / MetaX C550):
  * fp16/bf16: the scalar is kept at fp32 precision, the add is evaluated in
    fp32 and rounded once to the element dtype (verified by probe: torch gives
    0.300048828125 for 1000.0 + (-999.7) in fp16, i.e. the fp32-scalar variant).
  * fp32: fp32 add with the fp32-rounded scalar.
  * fp64: full double add.
  * integral element dtypes raise for a floating scalar.
  * empty tensor list raises "Tensor list must have at least one tensor."
  * mismatched list lengths raise
    "Tensor list must have same number of elements as scalar list."
  * a bare number is accepted for a single-tensor list.

The mutation is fully in place: no tensor is allocated and every element is
read and written exactly once.

Dispatch, cheapest first:
  1. UNIFORM list (one dtype/device, every tensor a dense-range with the same
     numel): one launch of a 2-D grid (blocks, slots) kernel binding one
     pointer and one scalar per tensor and nothing else -- 9 arguments for the
     4-tensor case, which keeps the host launch cost minimal for tiny lists.
  2. Mixed dense-range list: one fused launch per (dtype, device) group of up
     to 8 tensors, addressed straight from ``data_ptr`` with runtime block
     prefix dispatch.
  3. Anything else (non-dense strides): a generic strided kernel per tensor.

"dense-range" means the tensor's storage offsets are exactly [0, numel): it
covers contiguous tensors, transposed / permuted / unit-inner-stride views and
views carrying a storage offset, all of which stay fully coalesced.
"""

import struct

import torch
import triton
import triton.language as tl

_MAX_SLOTS = 8
_MAX_DIMS = 8
_LEN_ERR = "Tensor list must have same number of elements as scalar list."
_EMPTY_ERR = "Tensor list must have at least one tensor."
_INT_NAMES = {
    torch.int8: "Char",
    torch.uint8: "Byte",
    torch.int16: "Short",
    torch.int32: "Int",
    torch.int64: "Long",
    torch.bool: "Bool",
}


@triton.jit
def _add_store(ptr, off, mask, s, sb):
    """In-place ``x <- x + s`` with ATen's rounding for each dtype."""
    x = tl.load(ptr + off, mask=mask)
    if x.dtype == tl.float32:
        y = x + s
    elif x.dtype == tl.float64:
        y = x + sb.to(tl.int64).to(tl.float64, bitcast=True)
    elif x.dtype == tl.float16 or x.dtype == tl.bfloat16:
        y = x.to(tl.float32) + s
    else:
        y = x + s.to(x.dtype)
    tl.store(ptr + off, y.to(x.dtype), mask=mask)


@triton.jit
def _uniform4_kernel(p0, p1, p2, p3, s0, s1, s2, s3, n, BLOCK: tl.constexpr):
    """Four dense-range tensors of equal numel: grid (blocks, 4)."""
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    slot = tl.program_id(1)
    if slot == 0:
        _add_store(p0, off, mask, s0, 0)
    elif slot == 1:
        _add_store(p1, off, mask, s1, 0)
    elif slot == 2:
        _add_store(p2, off, mask, s2, 0)
    else:
        _add_store(p3, off, mask, s3, 0)


@triton.jit
def _uniform8_kernel(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    n,
    BLOCK: tl.constexpr,
):
    """Up to eight dense-range tensors of equal numel: grid (blocks, slots)."""
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    slot = tl.program_id(1)
    if slot == 0:
        _add_store(p0, off, mask, s0, 0)
    elif slot == 1:
        _add_store(p1, off, mask, s1, 0)
    elif slot == 2:
        _add_store(p2, off, mask, s2, 0)
    elif slot == 3:
        _add_store(p3, off, mask, s3, 0)
    elif slot == 4:
        _add_store(p4, off, mask, s4, 0)
    elif slot == 5:
        _add_store(p5, off, mask, s5, 0)
    elif slot == 6:
        _add_store(p6, off, mask, s6, 0)
    else:
        _add_store(p7, off, mask, s7, 0)


@triton.jit
def _dense_kernel(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    q0,
    q1,
    q2,
    q3,
    q4,
    q5,
    q6,
    q7,
    n0,
    n1,
    n2,
    n3,
    n4,
    n5,
    n6,
    n7,
    c0,
    c1,
    c2,
    c3,
    c4,
    c5,
    c6,
    c7,
    BLOCK: tl.constexpr,
    I64: tl.constexpr,
):
    """Mixed dense-range slots: one program per BLOCK of one slot."""
    pid = tl.program_id(0)
    if I64:
        base = pid.to(tl.int64) * BLOCK
    else:
        base = pid * BLOCK
    r = tl.arange(0, BLOCK)
    if pid < c0:
        off = base + r
        _add_store(p0, off, off < n0, s0, q0)
    elif pid < c1:
        off = base - c0 * BLOCK + r
        _add_store(p1, off, off < n1, s1, q1)
    elif pid < c2:
        off = base - c1 * BLOCK + r
        _add_store(p2, off, off < n2, s2, q2)
    elif pid < c3:
        off = base - c2 * BLOCK + r
        _add_store(p3, off, off < n3, s3, q3)
    elif pid < c4:
        off = base - c3 * BLOCK + r
        _add_store(p4, off, off < n4, s4, q4)
    elif pid < c5:
        off = base - c4 * BLOCK + r
        _add_store(p5, off, off < n5, s5, q5)
    elif pid < c6:
        off = base - c5 * BLOCK + r
        _add_store(p6, off, off < n6, s6, q6)
    else:
        off = base - c6 * BLOCK + r
        _add_store(p7, off, off < n7, s7, q7)


@triton.jit
def _strided_kernel(
    p,
    s,
    q,
    n,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
    t6,
    t7,
    NDIM: tl.constexpr,
    BLOCK: tl.constexpr,
    I64: tl.constexpr,
):
    """Generic strided add; dims are supplied innermost logical dim first."""
    pid = tl.program_id(0)
    if I64:
        idx = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n
    rem = idx
    if NDIM == 1:
        off = rem * t0
    else:
        if I64:
            off = tl.zeros([BLOCK], dtype=tl.int64)
        else:
            off = tl.zeros([BLOCK], dtype=tl.int32)
        u = rem % d0
        rem = rem // d0
        off = off + u * t0
        if NDIM > 1:
            u = rem % d1
            rem = rem // d1
            off = off + u * t1
        if NDIM > 2:
            u = rem % d2
            rem = rem // d2
            off = off + u * t2
        if NDIM > 3:
            u = rem % d3
            rem = rem // d3
            off = off + u * t3
        if NDIM > 4:
            u = rem % d4
            rem = rem // d4
            off = off + u * t4
        if NDIM > 5:
            u = rem % d5
            rem = rem // d5
            off = off + u * t5
        if NDIM > 6:
            u = rem % d6
            rem = rem // d6
            off = off + u * t6
        if NDIM > 7:
            u = rem % d7
            rem = rem // d7
            off = off + u * t7
    off = tl.where(mask, off, 0)
    _add_store(p, off, mask, s, q)


def _f64_bits(v):
    return struct.unpack("<q", struct.pack("<d", v))[0]


def _is_dense_perm(t):
    """True when the storage offsets of ``t`` are exactly [0, numel)."""
    shape = t.shape
    strides = t.stride()
    dims = []
    for i in range(len(shape)):
        sz = shape[i]
        if sz != 1:
            dims.append((strides[i], sz))
    if not dims:
        return True
    dims.sort()
    exp = 1
    for st, sz in dims:
        if st != exp:
            return False
        exp *= sz
    return True


def _collapse(t):
    """Collapse mergeable dims; returns (sizes, strides) inner-to-outer."""
    shape = t.shape
    strides = t.stride()
    groups = []
    n = len(shape)
    for i in range(n - 1, -1, -1):
        sz = shape[i]
        if sz == 1:
            continue
        st = strides[i]
        if groups and groups[-1][1] * groups[-1][0] == st:
            groups[-1][0] *= sz
        else:
            groups.append([sz, st])
    return [g[0] for g in groups], [g[1] for g in groups]


def _has_dup(items):
    if len(items) < 2:
        return False
    seen = set()
    for e in items:
        p = e[0].data_ptr()
        if p in seen:
            return True
        seen.add(p)
    return False


def _pick_block(total):
    if total <= 8192:
        return 256
    if total <= 262144:
        return 1024
    if total <= 8388608:
        return 2048
    return 4096


def _uniform_launch(tlist, slist, m, n, block):
    grid = ((n + block - 1) // block, m)
    nw = 4 if block <= 1024 else 8
    if m == 4:
        _uniform4_kernel[grid](
            tlist[0],
            tlist[1],
            tlist[2],
            tlist[3],
            slist[0],
            slist[1],
            slist[2],
            slist[3],
            n,
            BLOCK=block,
            num_warps=nw,
        )
    else:
        ptrs = list(tlist)
        scs = list(slist)
        pad = ptrs[0]
        while len(ptrs) < _MAX_SLOTS:
            ptrs.append(pad)
            scs.append(0.0)
        _uniform8_kernel[grid](
            *ptrs,
            *scs,
            n,
            BLOCK=block,
            num_warps=nw,
        )


def _launch_dense(items, is64):
    m = len(items)
    tot = 0
    for e in items:
        tot += e[1]
    block = _pick_block(tot)
    nw = 4 if block <= 1024 else 8
    ptrs = []
    scs = []
    ns = []
    cs = []
    acc = 0
    for e in items:
        n = e[1]
        ptrs.append(e[0])
        scs.append(e[2])
        ns.append(n)
        acc += (n + block - 1) // block
        cs.append(acc)
    if is64:
        qs = [_f64_bits(v) for v in scs]
    else:
        qs = [0] * m
    i64 = max(ns) > 1073741824
    if m < _MAX_SLOTS:
        ext = _MAX_SLOTS - m
        pad = ptrs[0]
        ptrs = ptrs + [pad] * ext
        scs = scs + [0.0] * ext
        qs = qs + [0] * ext
        ns = ns + [0] * ext
        cs = cs + [acc] * ext
    _dense_kernel[(acc,)](
        *ptrs,
        *scs,
        *qs,
        *ns,
        *cs,
        BLOCK=block,
        I64=i64,
        num_warps=nw,
    )


def _launch_strided(t, n, s, dt):
    sizes, strides = _collapse(t)
    nd = len(sizes)
    if nd > _MAX_DIMS:
        raise RuntimeError("unsupported tensor rank for in-place add")
    span = 1
    for i in range(nd):
        span += (sizes[i] - 1) * strides[i]
    i64 = span > 1073741824 or n > 1073741824
    ds = sizes + [1] * (_MAX_DIMS - nd)
    ts = strides + [0] * (_MAX_DIMS - nd)
    block = _pick_block(n)
    nw = 4 if block <= 1024 else 8
    q = _f64_bits(s) if dt is torch.float64 else 0
    _strided_kernel[((n + block - 1) // block,)](
        t,
        s,
        q,
        n,
        *ds,
        *ts,
        NDIM=nd,
        BLOCK=block,
        I64=i64,
        num_warps=nw,
    )


def foreach_add_scalar_list_(self, scalars):
    ts = self
    if isinstance(ts, torch.Tensor):
        ts = [ts]
    k = len(ts)
    if k == 0:
        raise RuntimeError(_EMPTY_ERR)
    if isinstance(scalars, (list, tuple)):
        if len(scalars) != k:
            raise RuntimeError(_LEN_ERR)
        sc = scalars
    else:
        try:
            ns_sc = len(scalars)
        except TypeError:
            ns_sc = 1
            scalars = (scalars,)
        if ns_sc != k:
            raise RuntimeError(_LEN_ERR)
        sc = scalars

    # ---- lean path: equal numel, one dtype, all contiguous, no alias ----
    t0 = ts[0]
    n = t0.numel()
    if n and k <= _MAX_SLOTS and len({id(u) for u in ts}) == k:
        dt = t0.dtype
        if dt is not torch.float64 and dt.is_floating_point and t0.is_contiguous():
            ok = True
            for i in range(1, k):
                u = ts[i]
                if u.dtype is not dt or u.numel() != n or not u.is_contiguous():
                    ok = False
                    break
            if ok:
                _uniform_launch(ts, [float(s) for s in sc], k, n, _pick_block(n * k))
                return None

    dense = {}
    strided = []
    for i in range(k):
        t = ts[i]
        n = t.numel()
        if n == 0:
            continue
        dt = t.dtype
        s = sc[i]
        if not dt.is_floating_point:
            name = _INT_NAMES.get(dt)
            if name is not None and isinstance(s, float):
                raise RuntimeError(
                    "result type Float can't be cast to the desired output "
                    "type " + name
                )
        sv = float(s)
        if t.is_contiguous() or _is_dense_perm(t):
            key = (dt, t.get_device())
            bucket = dense.get(key)
            if bucket is None:
                dense[key] = [(t, n, sv)]
            else:
                bucket.append((t, n, sv))
        else:
            strided.append((t, n, sv, dt))

    if len(dense) == 1 and not strided:
        key, items = next(iter(dense.items()))
        m = len(items)
        n = items[0][1]
        dup = _has_dup(items)
        is64 = key[0] is torch.float64
        if not dup and m <= _MAX_SLOTS:
            same = not is64
            if same:
                for e in items:
                    if e[1] != n:
                        same = False
                        break
            if same:
                _uniform_launch(
                    [e[0] for e in items],
                    [e[2] for e in items],
                    m,
                    n,
                    _pick_block(n * m),
                )
                return None
            _launch_dense(items, is64)
        else:
            for e in items:
                _launch_dense([e], is64)
    else:
        for key, items in dense.items():
            is64 = key[0] is torch.float64
            if len(items) <= _MAX_SLOTS and not _has_dup(items):
                _launch_dense(items, is64)
            else:
                for e in items:
                    _launch_dense([e], is64)

    for t, n, s, dt in strided:
        _launch_strided(t, n, s, dt)

    return None


_foreach_add_ = foreach_add_scalar_list_
