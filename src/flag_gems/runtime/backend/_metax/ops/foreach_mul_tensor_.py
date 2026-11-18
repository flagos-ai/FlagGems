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

"""In-place foreach multiply (aten::_foreach_mul_.Tensor) with Triton kernels.

The op mutates every tensor of ``self`` in place (``t *= s``) and returns None.
Dense (contiguous or dense-permuted) tensors use a flat kernel; everything else
uses a stride-driven kernel. Same-dtype dense tensors are fused into a single
launch. A dedicated low-argument kernel covers the common case of a short list
of identically shaped contiguous tensors, where the host launch path dominates.
"""

import torch
import triton
import triton.language as tl

_MAX_SLOTS = 8
_SMALL_TOTAL = 262144
_F16_DTYPES = (torch.float16, torch.bfloat16)
_INT_DTYPES = (
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
    torch.bool,
)

_OVERLAP_MSG = (
    "unsupported operation: more than one element of the written-to tensor "
    "refers to a single memory location. Please clone() the tensor before "
    "performing the operation."
)


@triton.jit
def _mul_flat_body(
    x_ptr,
    s_ptr,
    s_val,
    numel,
    pid,
    BLOCK: tl.constexpr,
    SPTR: tl.constexpr,
    WIDE: tl.constexpr,
    INT: tl.constexpr,
    EVEN: tl.constexpr,
):
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        x = tl.load(x_ptr + offs)
    else:
        mask = offs < numel
        x = tl.load(x_ptr + offs, mask=mask, other=0)
    if SPTR:
        s = tl.load(s_ptr)
    else:
        s = s_val
    if INT:
        out = (x.to(tl.int64) * s.to(tl.int64)).to(x.dtype)
    elif WIDE:
        out = (x.to(tl.float32) * s.to(tl.float32)).to(x.dtype)
    else:
        out = x * s.to(x.dtype)
    if EVEN:
        tl.store(x_ptr + offs, out)
    else:
        tl.store(x_ptr + offs, out, mask=mask)


@triton.jit(do_not_specialize=["numel"])
def _mul_u4_kernel(
    x0,
    x1,
    x2,
    x3,
    numel,
    s_ptr,
    s_val,
    BLOCK: tl.constexpr,
    SPTR: tl.constexpr,
    WIDE: tl.constexpr,
    INT: tl.constexpr,
    EVEN: tl.constexpr,
):
    """Uniform shape list of up to 4 tensors: grid is (blocks, len(list))."""
    pid = tl.program_id(0)
    slot = tl.program_id(1)
    if slot == 0:
        _mul_flat_body(x0, s_ptr, s_val, numel, pid, BLOCK, SPTR, WIDE, INT, EVEN)
    elif slot == 1:
        _mul_flat_body(x1, s_ptr, s_val, numel, pid, BLOCK, SPTR, WIDE, INT, EVEN)
    elif slot == 2:
        _mul_flat_body(x2, s_ptr, s_val, numel, pid, BLOCK, SPTR, WIDE, INT, EVEN)
    else:
        _mul_flat_body(x3, s_ptr, s_val, numel, pid, BLOCK, SPTR, WIDE, INT, EVEN)


@triton.jit(
    do_not_specialize=[
        "n0",
        "n1",
        "n2",
        "n3",
        "n4",
        "n5",
        "n6",
        "n7",
        "p0",
        "p1",
        "p2",
        "p3",
        "p4",
        "p5",
        "p6",
        "p7",
    ]
)
def _mul_flat_kernel(
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    n0,
    n1,
    n2,
    n3,
    n4,
    n5,
    n6,
    n7,
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    s_ptr,
    s_val,
    BLOCK: tl.constexpr,
    SPTR: tl.constexpr,
    WIDE: tl.constexpr,
    INT: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < p0:
        _mul_flat_body(x0, s_ptr, s_val, n0, pid, BLOCK, SPTR, WIDE, INT, EVEN)
    elif pid < p1:
        _mul_flat_body(x1, s_ptr, s_val, n1, pid - p0, BLOCK, SPTR, WIDE, INT, EVEN)
    elif pid < p2:
        _mul_flat_body(x2, s_ptr, s_val, n2, pid - p1, BLOCK, SPTR, WIDE, INT, EVEN)
    elif pid < p3:
        _mul_flat_body(x3, s_ptr, s_val, n3, pid - p2, BLOCK, SPTR, WIDE, INT, EVEN)
    elif pid < p4:
        _mul_flat_body(x4, s_ptr, s_val, n4, pid - p3, BLOCK, SPTR, WIDE, INT, EVEN)
    elif pid < p5:
        _mul_flat_body(x5, s_ptr, s_val, n5, pid - p4, BLOCK, SPTR, WIDE, INT, EVEN)
    elif pid < p6:
        _mul_flat_body(x6, s_ptr, s_val, n6, pid - p5, BLOCK, SPTR, WIDE, INT, EVEN)
    else:
        _mul_flat_body(x7, s_ptr, s_val, n7, pid - p6, BLOCK, SPTR, WIDE, INT, EVEN)


@triton.jit(
    do_not_specialize=[
        "numel",
        "d0",
        "d1",
        "d2",
        "d3",
        "d4",
        "d5",
        "d6",
        "d7",
        "t0",
        "t1",
        "t2",
        "t3",
        "t4",
        "t5",
        "t6",
        "t7",
    ]
)
def _mul_strided_kernel(
    x_ptr,
    s_ptr,
    s_val,
    numel,
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
    BLOCK: tl.constexpr,
    SPTR: tl.constexpr,
    WIDE: tl.constexpr,
    INT: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    idx = offs.to(tl.int64)
    off = (idx % d0) * t0
    idx = idx // d0
    off += (idx % d1) * t1
    idx = idx // d1
    off += (idx % d2) * t2
    idx = idx // d2
    off += (idx % d3) * t3
    idx = idx // d3
    off += (idx % d4) * t4
    idx = idx // d4
    off += (idx % d5) * t5
    idx = idx // d5
    off += (idx % d6) * t6
    idx = idx // d6
    off += (idx % d7) * t7
    x = tl.load(x_ptr + off, mask=mask, other=0)
    if SPTR:
        s = tl.load(s_ptr)
    else:
        s = s_val
    if INT:
        out = (x.to(tl.int64) * s.to(tl.int64)).to(x.dtype)
    elif WIDE:
        out = (x.to(tl.float32) * s.to(tl.float32)).to(x.dtype)
    else:
        out = x * s.to(x.dtype)
    tl.store(x_ptr + off, out, mask=mask)


def _classify(t):
    """Return (kind, pairs): kind 0 flat, 1 strided, -1 overlapping memory."""
    shape = t.shape
    strides = t.stride()
    dims = [(sz, st) for sz, st in zip(shape, strides) if sz > 1]
    if not dims:
        return 0, None
    covered = 1
    dense = True
    for sz, st in sorted(dims, key=lambda p: p[1]):
        if st < covered:
            return -1, None
        if st != covered:
            dense = False
        covered = st * sz
    if dense:
        return 0, None
    collapsed = []
    for sz, st in dims:
        if collapsed and collapsed[-1][1] == st * sz:
            psz, _ = collapsed[-1]
            collapsed[-1] = (psz * sz, st)
        else:
            collapsed.append((sz, st))
    collapsed.reverse()
    return 1, collapsed


def _span_bytes(t):
    span = 1
    for sz, st in zip(t.shape, t.stride()):
        if sz > 1:
            span += (sz - 1) * st
    return span * t.element_size()


def _any_overlap(tensors):
    spans = []
    for t in tensors:
        base = t.data_ptr()
        spans.append((base, base + _span_bytes(t)))
    spans.sort()
    for i in range(len(spans) - 1):
        if spans[i][1] > spans[i + 1][0]:
            return True
    return False


def _pick_block(total):
    if total <= _SMALL_TOTAL:
        return 1024, 4
    return 4096, 8


_FAST_LAUNCH = {}


def _wide_flag(dtype, s_ptr, sptr):
    """Skip the fp32 round-trip when torch itself multiplies in 16-bit.

    torch computes a 16-bit multiply in the tensor dtype when the scalar is a
    same-dtype tensor; host scalars and wider scalar tensors go through fp32
    opmath, so they keep the fp32 path.
    """
    if dtype not in _F16_DTYPES:
        return False
    if sptr and s_ptr.dtype == dtype:
        return False
    return True


def _launch_u4(chunk, numel, dtype, s_ptr, s_val, sptr, aligned):
    k = len(chunk)
    if numel * k <= _SMALL_TOTAL:
        block = 1024
        warps = 4
    else:
        block = 4096
        warps = 8
    b = (numel + block - 1) // block
    x0 = chunk[0]
    x1 = chunk[1] if k > 1 else x0
    x2 = chunk[2] if k > 2 else x0
    x3 = chunk[3] if k > 3 else x0
    sp = s_ptr if sptr else x0
    wide = _wide_flag(dtype, s_ptr, sptr)
    is_int = dtype in _INT_DTYPES
    even = (numel % block) == 0
    if aligned:
        key = (
            dtype,
            block,
            warps,
            sptr,
            s_ptr.dtype if sptr else 0,
            type(s_val),
            wide,
            is_int,
            even,
            k,
            b,
        )
        runner = _FAST_LAUNCH.get(key)
        if runner is not None:
            if runner is not True:
                runner(
                    x0, x1, x2, x3, numel, sp, s_val, block, sptr, wide, is_int, even
                )
                return
        else:
            ck = _mul_u4_kernel[(b, k)](
                x0,
                x1,
                x2,
                x3,
                numel,
                sp,
                s_val,
                BLOCK=block,
                SPTR=sptr,
                WIDE=wide,
                INT=is_int,
                EVEN=even,
                num_warps=warps,
            )
            try:
                _FAST_LAUNCH[key] = ck[(b, k, 1)]
            except Exception:
                _FAST_LAUNCH[key] = True
            return
    _mul_u4_kernel[(b, k)](
        x0,
        x1,
        x2,
        x3,
        numel,
        sp,
        s_val,
        BLOCK=block,
        SPTR=sptr,
        WIDE=wide,
        INT=is_int,
        EVEN=even,
        num_warps=warps,
    )


def _launch_flat(chunk, s_ptr, s_val, sptr):
    sizes = [t.numel() for t in chunk]
    total = sum(sizes)
    block, warps = _pick_block(total)
    blocks = []
    acc = 0
    for sz in sizes:
        acc += (sz + block - 1) // block
        blocks.append(acc)
    ns = sizes + [0] * (_MAX_SLOTS - len(sizes))
    ps = blocks + [acc] * (_MAX_SLOTS - len(blocks))
    xs = list(chunk) + [chunk[0]] * (_MAX_SLOTS - len(chunk))
    dtype = chunk[0].dtype
    even = True
    for sz in sizes:
        if sz % block:
            even = False
            break
    _mul_flat_kernel[(acc,)](
        xs[0],
        xs[1],
        xs[2],
        xs[3],
        xs[4],
        xs[5],
        xs[6],
        xs[7],
        ns[0],
        ns[1],
        ns[2],
        ns[3],
        ns[4],
        ns[5],
        ns[6],
        ns[7],
        ps[0],
        ps[1],
        ps[2],
        ps[3],
        ps[4],
        ps[5],
        ps[6],
        ps[7],
        s_ptr if sptr else xs[0],
        s_val,
        BLOCK=block,
        SPTR=sptr,
        WIDE=_wide_flag(dtype, s_ptr, sptr),
        INT=dtype in _INT_DTYPES,
        EVEN=even,
        num_warps=warps,
    )


def _launch_strided(t, pairs, s_ptr, s_val, sptr):
    numel = t.numel()
    block = 1024
    dtype = t.dtype
    ex = [1] * 8
    su = [0] * 8
    for i, (sz, st) in enumerate(pairs[:8]):
        ex[i] = sz
        su[i] = st
    _mul_strided_kernel[((numel + block - 1) // block,)](
        t,
        s_ptr if sptr else t,
        s_val,
        numel,
        ex[0],
        ex[1],
        ex[2],
        ex[3],
        ex[4],
        ex[5],
        ex[6],
        ex[7],
        su[0],
        su[1],
        su[2],
        su[3],
        su[4],
        su[5],
        su[6],
        su[7],
        BLOCK=block,
        SPTR=sptr,
        WIDE=_wide_flag(dtype, s_ptr, sptr),
        INT=dtype in _INT_DTYPES,
        num_warps=4,
    )


def _run_one(t, s_ptr, s_val, sptr):
    """Launch for exactly one tensor."""
    numel = t.numel()
    if numel == 0:
        return
    kind, pairs = _classify(t)
    if kind < 0:
        raise RuntimeError(_OVERLAP_MSG)
    if kind == 0:
        aligned = (t.data_ptr() & 15) == 0
        if aligned and sptr and (s_ptr.data_ptr() & 15):
            aligned = False
        _launch_u4([t], numel, t.dtype, s_ptr, s_val, sptr, aligned)
        return
    if len(pairs) > 8 and t.shape[0] > 1:
        for i in range(t.shape[0]):
            _run_one(t.select(0, i), s_ptr, s_val, sptr)
        return
    _launch_strided(t, pairs, s_ptr, s_val, sptr)


def _resolve_scalar(sc):
    if isinstance(sc, torch.Tensor):
        if sc.dim() != 0:
            raise RuntimeError(
                "scalar tensor expected to be 0 dim but it has %d dimensions "
                "and %d elements." % (sc.dim(), sc.numel())
            )
        if sc.is_cuda:
            return sc, 0.0, True
        return None, sc.item(), False
    if isinstance(sc, bool) or isinstance(sc, int):
        return None, int(sc), False
    return None, float(sc), False


def foreach_mul_tensor_(self, other):
    if isinstance(self, torch.Tensor):
        tensors = (self,)
    elif isinstance(self, (list, tuple)):
        tensors = self
    else:
        tensors = list(self)
    n = len(tensors)
    if n == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")

    if isinstance(other, torch.Tensor):
        if other.dim() != 0:
            raise RuntimeError(
                "scalar tensor expected to be 0 dim but it has %d dimensions "
                "and %d elements." % (other.dim(), other.numel())
            )
        if other.is_cuda:
            s_ptr = other
            s_val = 0.0
            sptr = True
        else:
            s_ptr = None
            s_val = other.item()
            sptr = False
        scalars = None
    elif isinstance(other, (int, float)):
        s_ptr = None
        sptr = False
        s_val = int(other) if isinstance(other, int) else other
        scalars = None
    elif isinstance(other, (list, tuple)):
        if len(other) != n:
            raise RuntimeError(
                "Tensor list must have same number of elements as scalar list."
            )
        scalars = other
        s_ptr = s_val = sptr = None
    else:
        raise TypeError(
            "foreach_mul_ expects a Tensor or a sequence of scalars, got %s"
            % type(other).__name__
        )

    if scalars is not None:
        for t, sc in zip(tensors, scalars):
            p, v, f = _resolve_scalar(sc)
            _run_one(t, p, v, f)
        return None

    first = tensors[0]
    dtype = first.dtype
    numel = first.numel()
    dev = first.get_device()

    if numel > 0 and n <= 4:
        # Low-argument fast path: identically shaped contiguous tensors.
        ptrs = []
        ok = True
        for t in tensors:
            if (
                type(t) is not torch.Tensor
                or t.dtype is not dtype
                or t.numel() != numel
                or t.get_device() != dev
                or not t.is_contiguous()
            ):
                ok = False
                break
            ptrs.append(t.data_ptr())
        if ok:
            span = numel * first.element_size()
            ordered = sorted(ptrs)
            for i in range(n - 1):
                if ordered[i] + span > ordered[i + 1]:
                    ok = False
                    break
        if ok:
            aligned = True
            for p in ptrs:
                if p & 15:
                    aligned = False
                    break
            if aligned and sptr and (s_ptr.data_ptr() & 15):
                aligned = False
            _launch_u4(tensors, numel, dtype, s_ptr, s_val, sptr, aligned)
            return None

    for t in tensors:
        if not isinstance(t, torch.Tensor):
            raise TypeError(
                "foreach_mul_ expects a sequence of Tensors, got %s" % type(t).__name__
            )

    if _any_overlap(tensors):
        # torch multiplies a storage range once even when the list repeats a
        # tensor, so exact duplicates are collapsed; partially overlapping
        # entries keep list order (sequential launches).
        seen = set()
        for t in tensors:
            key = (t.data_ptr(), t.shape, t.stride(), t.dtype)
            if key in seen:
                continue
            seen.add(key)
            _run_one(t, s_ptr, s_val, sptr)
        return None

    groups = []
    index = {}
    strided = []
    for t in tensors:
        if t.numel() == 0:
            continue
        kind, pairs = _classify(t)
        if kind < 0:
            raise RuntimeError(_OVERLAP_MSG)
        if kind == 1:
            strided.append((t, pairs))
        else:
            key = (t.device, t.dtype)
            gi = index.get(key)
            if gi is None:
                index[key] = len(groups)
                groups.append([t])
            else:
                groups[gi].append(t)

    for t, pairs in strided:
        if len(pairs) > 8:
            _run_one(t, s_ptr, s_val, sptr)
        else:
            _launch_strided(t, pairs, s_ptr, s_val, sptr)

    for g in groups:
        if len(g) <= _MAX_SLOTS:
            _launch_flat(g, s_ptr, s_val, sptr)
        else:
            for i in range(0, len(g), _MAX_SLOTS):
                chunk = g[i : i + _MAX_SLOTS]
                _launch_flat(chunk, s_ptr, s_val, sptr)

    return None
