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

"""In-place foreach clamp_max over a list of tensors (Triton implementation).

Semantics (ATen `_foreach_clamp_max_.List`): for every pair i,
``self[i] = where(other[i] < self[i], other[i], self[i])`` written into
``self[i]``'s own storage.  ``other`` may broadcast to ``self``'s shape.
The call returns ``None`` and a length mismatch raises ``RuntimeError``
before anything is written.
"""

import triton
import triton.language as tl

_INT32_MAX = 2**31 - 1
_MAX_SLOTS = 8


# --------------------------------------------------------------------------- #
# Kernels
# --------------------------------------------------------------------------- #
@triton.jit
def _flat_kernel(
    a_ptr, b_ptr, n, BLOCK: tl.constexpr, SCALAR_B: tl.constexpr, LONG: tl.constexpr
):
    """a and b share one dense element order; ``b`` may be a lone value."""
    pid = tl.program_id(0)
    if LONG:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask, other=0)
    if SCALAR_B:
        b = tl.load(b_ptr)
    else:
        b = tl.load(b_ptr + offs, mask=mask, other=0)
    tl.store(a_ptr + offs, tl.where(b < a, b, a), mask=mask)


@triton.jit
def _fused_kernel(
    a0,
    b0,
    n0,
    a1,
    b1,
    n1,
    a2,
    b2,
    n2,
    a3,
    b3,
    n3,
    a4,
    b4,
    n4,
    a5,
    b5,
    n5,
    a6,
    b6,
    n6,
    a7,
    b7,
    n7,
    scalar_mask,
    BLOCK: tl.constexpr,
    LONG: tl.constexpr,
):
    """One launch for up to eight flat (self, other) pairs.

    Program (m, slot) handles element range [m*BLOCK, (m+1)*BLOCK) of pair
    ``slot``; the pair's pointers and length are selected with a static
    branch chain because Triton has no pointer arrays.
    """
    ap = a0
    bp = b0
    n = n0
    slot = tl.program_id(1)
    if slot == 1:
        ap = a1
        bp = b1
        n = n1
    elif slot == 2:
        ap = a2
        bp = b2
        n = n2
    elif slot == 3:
        ap = a3
        bp = b3
        n = n3
    elif slot == 4:
        ap = a4
        bp = b4
        n = n4
    elif slot == 5:
        ap = a5
        bp = b5
        n = n5
    elif slot == 6:
        ap = a6
        bp = b6
        n = n6
    elif slot == 7:
        ap = a7
        bp = b7
        n = n7
    pid = tl.program_id(0)
    if LONG:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(ap + offs, mask=mask, other=0)
    # A lone ``b`` value is fetched by making every lane read index 0, which
    # coalesces into a single broadcast transaction (no divergent branch).
    sc = (scalar_mask >> slot) & 1
    b = tl.load(bp + offs * (1 - sc), mask=mask, other=0)
    tl.store(ap + offs, tl.where(b < a, b, a), mask=mask)


@triton.jit
def _fused4_kernel(
    a0,
    b0,
    n0,
    a1,
    b1,
    n1,
    a2,
    b2,
    n2,
    a3,
    b3,
    n3,
    scalar_mask,
    BLOCK: tl.constexpr,
    LONG: tl.constexpr,
):
    """Four-slot variant of ``_fused_kernel`` for short lists.

    Same device code path with a third fewer launch arguments; slots at or
    beyond the launched grid extent never run.
    """
    ap = a0
    bp = b0
    n = n0
    slot = tl.program_id(1)
    if slot == 1:
        ap = a1
        bp = b1
        n = n1
    elif slot == 2:
        ap = a2
        bp = b2
        n = n2
    elif slot == 3:
        ap = a3
        bp = b3
        n = n3
    pid = tl.program_id(0)
    if LONG:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(ap + offs, mask=mask, other=0)
    sc = (scalar_mask >> slot) & 1
    b = tl.load(bp + offs * (1 - sc), mask=mask, other=0)
    tl.store(ap + offs, tl.where(b < a, b, a), mask=mask)


@triton.jit
def _fused4_same_kernel(
    a0,
    b0,
    a1,
    b1,
    a2,
    b2,
    a3,
    b3,
    n,
    scalar_mask,
    BLOCK: tl.constexpr,
    LONG: tl.constexpr,
):
    """Four pairs that all share one element count ``n``.

    Same device code path with the per-slot lengths collapsed into a single
    scalar; slots at or beyond the launched grid extent never run.
    """
    ap = a0
    bp = b0
    slot = tl.program_id(1)
    if slot == 1:
        ap = a1
        bp = b1
    elif slot == 2:
        ap = a2
        bp = b2
    elif slot == 3:
        ap = a3
        bp = b3
    pid = tl.program_id(0)
    if LONG:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(ap + offs, mask=mask, other=0)
    sc = (scalar_mask >> slot) & 1
    b = tl.load(bp + offs * (1 - sc), mask=mask, other=0)
    tl.store(ap + offs, tl.where(b < a, b, a), mask=mask)


@triton.jit
def _flat_val_kernel(a_ptr, bval, n, BLOCK: tl.constexpr, LONG: tl.constexpr):
    """Flat path with a host-side numeric bound."""
    pid = tl.program_id(0)
    if LONG:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask, other=0)
    b = bval.to(a.dtype)
    tl.store(a_ptr + offs, tl.where(b < a, b, a), mask=mask)


@triton.jit
def _strided_kernel(
    a_ptr,
    b_ptr,
    n,
    abase,
    bbase,
    bval,
    sz0,
    sz1,
    sz2,
    sz3,
    as0,
    as1,
    as2,
    as3,
    bs0,
    bs1,
    bs2,
    bs3,
    NDIM: tl.constexpr,
    BVAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Generic path: ``NDIM`` collapsed axes with explicit strides.

    ``b`` may be broadcast (stride 0) or a host-side numeric value (BVAL).
    """
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n
    rem = offs
    ao = tl.zeros([BLOCK], tl.int64)
    bo = tl.zeros([BLOCK], tl.int64)
    if NDIM > 0:
        i0 = rem % sz0
        rem = rem // sz0
        ao += i0 * as0
        bo += i0 * bs0
    if NDIM > 1:
        i1 = rem % sz1
        rem = rem // sz1
        ao += i1 * as1
        bo += i1 * bs1
    if NDIM > 2:
        i2 = rem % sz2
        rem = rem // sz2
        ao += i2 * as2
        bo += i2 * bs2
    if NDIM > 3:
        i3 = rem % sz3
        ao += i3 * as3
        bo += i3 * bs3
    ao += abase
    bo += bbase
    a = tl.load(a_ptr + ao, mask=mask, other=0)
    if BVAL:
        b = bval.to(a.dtype)
    else:
        b = tl.load(b_ptr + bo, mask=mask, other=0)
    tl.store(a_ptr + ao, tl.where(b < a, b, a), mask=mask)


# --------------------------------------------------------------------------- #
# Layout analysis (metadata only)
# --------------------------------------------------------------------------- #
def _dense_order(t):
    """Axis order in which ``t`` is dense-contiguous, or None."""
    shape = t.shape
    st = t.stride()
    idx = [d for d in range(len(shape)) if shape[d] != 1]
    if len(idx) > 1:
        idx = sorted(idx, key=st.__getitem__)
    exp = 1
    for d in idx:
        if st[d] != exp:
            return None
        exp *= shape[d]
    return idx


def _strided_layout(a, b):
    """Collapsed (size, a_stride, b_stride) axes, fastest-varying axis first."""
    ash = a.shape
    astr = a.stride()
    if b is None:
        bsh = ()
        bstr = ()
    else:
        bsh = b.shape
        bstr = b.stride()
    nd = len(ash) if len(ash) >= len(bsh) else len(bsh)
    dims = []
    for k in range(nd):
        ai = len(ash) - nd + k
        bi = len(bsh) - nd + k
        asz = ash[ai] if ai >= 0 else 1
        bsz = bsh[bi] if bi >= 0 else 1
        as_ = astr[ai] if ai >= 0 else 0
        bs_ = bstr[bi] if bi >= 0 else 0
        if asz != bsz:
            if asz == 1:
                raise RuntimeError(
                    "output with the broadcast shape does not match self"
                )
            if bsz != 1:
                raise RuntimeError("self and other are not broadcastable")
        if bsz == 1:
            bs_ = 0
        if asz == 1:
            as_ = 0
        dims.append((asz, as_, bs_))
    dims = sorted(dims, key=lambda d: d[1])
    out = []
    for sz, sa, sb in dims:
        if sz == 1:
            continue
        if out and sa == out[-1][1] * out[-1][0] and sb == out[-1][2] * out[-1][0]:
            out[-1] = (out[-1][0] * sz, out[-1][1], out[-1][2])
        else:
            out.append((sz, sa, sb))
    if not out:
        out = [(1, 0, 0)]
    return out


# --------------------------------------------------------------------------- #
# Launch helpers
# --------------------------------------------------------------------------- #
_KERNEL_CACHE = {}


def _launch_cached(fn, key, args, grid, warps):
    """Launch ``fn``, reusing an already compiled kernel when ``key`` matches.

    ``args`` lists every kernel parameter in declaration order (constexpr
    values included, because the target launcher takes them positionally).
    The first call goes through Triton's normal JIT launch; the ready-to-call
    runner of the returned CompiledKernel is cached and used afterwards,
    skipping per-argument binding and cache-key bookkeeping.  ``key`` carries
    exactly the inputs Triton's own cache key is derived from - pointer dtypes,
    pointer 16-byte alignment, every runtime scalar value and every constexpr -
    so a reused binary is always the one Triton would have compiled.  All
    indexing and value computation still happens in the kernel on every call.
    """
    entry = _KERNEL_CACHE.get(key)
    if entry is None:
        entry = fn[grid](*args, num_warps=warps)
        _KERNEL_CACHE[key] = entry[grid]
    else:
        entry(*args)


def _align_bits(ptrs):
    bits = 0
    for i, t in enumerate(ptrs):
        if t.data_ptr() & 15:
            bits |= 1 << i
    return bits


def _config(n):
    if n <= 16384:
        return 512, 2
    return 1024, 4


def _launch_flat(a, b, n, scalar_b):
    block, warps = _config(n)
    long_flag = n > _INT32_MAX
    key = (1, a.dtype, b.dtype, _align_bits((a, b)), n, scalar_b, block, long_flag)
    _launch_cached(
        _flat_kernel,
        key,
        (a, b, n, block, scalar_b, long_flag),
        ((n + block - 1) // block, 1, 1),
        warps,
    )


def _launch_flat_val(a, bval, n):
    block, warps = _config(n)
    _flat_val_kernel[((n + block - 1) // block, 1, 1)](
        a, bval, n, block, n > _INT32_MAX, num_warps=warps
    )


def _launch_strided(a, b, n, layout, bval=0.0, bval_mode=False):
    block, warps = _config(n)
    nd = len(layout)
    dims = list(layout) + [(1, 0, 0)] * (4 - nd)
    _strided_kernel[((n + block - 1) // block, 1, 1)](
        a,
        a if bval_mode else b,
        n,
        0,
        0,
        bval,
        dims[0][0],
        dims[1][0],
        dims[2][0],
        dims[3][0],
        dims[0][1],
        dims[1][1],
        dims[2][1],
        dims[3][1],
        dims[0][2],
        dims[1][2],
        dims[2][2],
        dims[3][2],
        nd,
        bval_mode,
        block,
        num_warps=warps,
    )


def _flush_flat(pairs):
    """Run buffered flat pairs with as few launches as possible."""
    total = len(pairs)
    if total == 1:
        a, b, n, sb = pairs[0]
        _launch_flat(a, b, n, sb)
        return
    if total == 4:
        # Common short list: explicit positional launch of the 4-slot kernel
        # avoids building and unpacking an argument list.
        a0, b0, n0, s0 = pairs[0]
        a1, b1, n1, s1 = pairs[1]
        a2, b2, n2, s2 = pairs[2]
        a3, b3, n3, s3 = pairs[3]
        max_n = n0
        if n1 > max_n:
            max_n = n1
        if n2 > max_n:
            max_n = n2
        if n3 > max_n:
            max_n = n3
        scalar_mask = s0 | (s1 << 1) | (s2 << 2) | (s3 << 3)
        block, warps = _config(max_n)
        long_flag = max_n > _INT32_MAX
        align = (
            (a0.data_ptr() & 15 != 0)
            | ((b0.data_ptr() & 15 != 0) << 1)
            | ((a1.data_ptr() & 15 != 0) << 2)
            | ((b1.data_ptr() & 15 != 0) << 3)
            | ((a2.data_ptr() & 15 != 0) << 4)
            | ((b2.data_ptr() & 15 != 0) << 5)
            | ((a3.data_ptr() & 15 != 0) << 6)
            | ((b3.data_ptr() & 15 != 0) << 7)
        )
        key = (
            2,
            a0.dtype,
            b0.dtype,
            align,
            n0,
            n1,
            n2,
            n3,
            scalar_mask,
            block,
            long_flag,
        )
        _launch_cached(
            _fused4_kernel,
            key,
            (
                a0,
                b0,
                n0,
                a1,
                b1,
                n1,
                a2,
                b2,
                n2,
                a3,
                b3,
                n3,
                scalar_mask,
                block,
                long_flag,
            ),
            ((max_n + block - 1) // block, 4, 1),
            warps,
        )
        return
    slots = 4 if total < 4 else _MAX_SLOTS
    fn = _fused4_kernel if slots == 4 else _fused_kernel
    tag = 2 if slots == 4 else 0
    start = 0
    while start < total:
        chunk = pairs[start : start + slots]
        start += len(chunk)
        k = len(chunk)
        if k == 1:
            a, b, n, sb = chunk[0]
            _launch_flat(a, b, n, sb)
            continue
        args = []
        ptrs = []
        lens = []
        max_n = 0
        scalar_mask = 0
        for j, (a, b, n, sb) in enumerate(chunk):
            args.append(a)
            args.append(b)
            args.append(n)
            ptrs.append(a)
            ptrs.append(b)
            lens.append(n)
            if n > max_n:
                max_n = n
            if sb:
                scalar_mask |= 1 << j
        if k < slots:
            a0, b0, _, _ = chunk[0]
            args.extend((a0, b0, 0) * (slots - k))
            lens.extend([0] * (slots - k))
        block, warps = _config(max_n)
        long_flag = max_n > _INT32_MAX
        key = (
            tag,
            ptrs[0].dtype,
            ptrs[1].dtype,
            _align_bits(ptrs),
            tuple(lens),
            scalar_mask,
            block,
            long_flag,
        )
        args.append(scalar_mask)
        args.append(block)
        args.append(long_flag)
        _launch_cached(fn, key, args, ((max_n + block - 1) // block, k, 1), warps)


def _flat_candidate(layout, n):
    """Layout is a single dense axis of length n (b either dense or lone)."""
    return len(layout) == 1 and layout[0][0] == n and layout[0][1] == 1


def _run_four(self, pairs):
    """Straight-line path for four same-dtype, same-length flat pairs.

    Returns True when it launched the fused kernel, False when the pairs need
    the generic path (which then re-derives the same facts).  Every check runs
    before the launch, so a False return never leaves anything half applied.
    """
    a0 = self[0]
    a1 = self[1]
    a2 = self[2]
    a3 = self[3]
    b0 = pairs[0]
    b1 = pairs[1]
    b2 = pairs[2]
    b3 = pairs[3]
    da = a0.dtype
    if not (a1.dtype is da and a2.dtype is da and a3.dtype is da):
        return False
    db = b0.dtype
    if not (b1.dtype is db and b2.dtype is db and b3.dtype is db):
        return False
    n = a0.numel()
    if n == 0:
        return False
    if not (a1.numel() == n and a2.numel() == n and a3.numel() == n):
        return False
    if not (
        a0.is_contiguous()
        and a1.is_contiguous()
        and a2.is_contiguous()
        and a3.is_contiguous()
        and b0.is_contiguous()
        and b1.is_contiguous()
        and b2.is_contiguous()
        and b3.is_contiguous()
    ):
        return False
    scalar_mask = 0
    nb = b0.numel()
    if nb == 1:
        scalar_mask = 1
    elif nb != n or a0.shape != b0.shape:
        return False
    nb = b1.numel()
    if nb == 1:
        scalar_mask |= 2
    elif nb != n or a1.shape != b1.shape:
        return False
    nb = b2.numel()
    if nb == 1:
        scalar_mask |= 4
    elif nb != n or a2.shape != b2.shape:
        return False
    nb = b3.numel()
    if nb == 1:
        scalar_mask |= 8
    elif nb != n or a3.shape != b3.shape:
        return False
    block, warps = _config(n)
    long_flag = n > _INT32_MAX
    align = (
        (a0.data_ptr() & 15 != 0)
        | ((b0.data_ptr() & 15 != 0) << 1)
        | ((a1.data_ptr() & 15 != 0) << 2)
        | ((b1.data_ptr() & 15 != 0) << 3)
        | ((a2.data_ptr() & 15 != 0) << 4)
        | ((b2.data_ptr() & 15 != 0) << 5)
        | ((a3.data_ptr() & 15 != 0) << 6)
        | ((b3.data_ptr() & 15 != 0) << 7)
    )
    key = (3, da, db, align, n, scalar_mask, block, long_flag)
    _launch_cached(
        _fused4_same_kernel,
        key,
        (a0, b0, a1, b1, a2, b2, a3, b3, n, scalar_mask, block, long_flag),
        ((n + block - 1) // block, 4, 1),
        warps,
    )
    return True


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def foreach_clamp_max_list_(self, other):
    count = len(self)
    if isinstance(other, (list, tuple)):
        if len(other) != count:
            raise RuntimeError("Tensor list inputs must have the same length")
        pairs = other
    else:
        if count == 0:
            return None
        pairs = (other,) * count

    if count == 4:
        try:
            if _run_four(self, pairs):
                return None
        except AttributeError:
            pass

    pending = []
    pending_a = None
    pending_b = None
    for a, b in zip(self, pairs):
        try:
            n = a.numel()
        except AttributeError:
            raise TypeError("self must be a sequence of Tensors") from None
        if n == 0:
            continue
        try:
            bn = b.numel()
        except AttributeError:
            bn = None
        if bn is not None:
            if (
                bn == n
                and a.shape == b.shape
                and a.is_contiguous()
                and b.is_contiguous()
            ):
                flat = (a, b, n, False)
            elif bn == 1 and a.is_contiguous():
                flat = (a, b, n, True)
            else:
                layout = _strided_layout(a, b)
                if _flat_candidate(layout, n):
                    flat = (a, b, n, layout[0][2] != 1)
                else:
                    if pending:
                        _flush_flat(pending)
                        pending = []
                        pending_a = None
                        pending_b = None
                    _launch_strided(a, b, n, layout)
                    continue
            if a.dtype is pending_a and b.dtype is pending_b:
                pending.append(flat)
            else:
                if pending:
                    _flush_flat(pending)
                pending = [flat]
                pending_a = a.dtype
                pending_b = b.dtype
        else:
            if not a.is_contiguous():
                layout = _strided_layout(a, None)
                if not _flat_candidate(layout, n):
                    if pending:
                        _flush_flat(pending)
                        pending = []
                        pending_a = None
                        pending_b = None
                    _launch_strided(a, None, n, layout, bval=float(b), bval_mode=True)
                    continue
            if pending:
                _flush_flat(pending)
                pending = []
                pending_a = None
                pending_b = None
            _launch_flat_val(a, float(b), n)
    if pending:
        _flush_flat(pending)
    return None
