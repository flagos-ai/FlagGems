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

"""In-place foreach_abs_ implemented with Triton kernels.

Semantics: for every tensor in the list, T[i] <- |T[i]| for all logical i,
writing back into T's own storage.  Storage pointer, strides and list order are
preserved, and the call returns None.

Addressing modes:

* flat  - the addressed elements form exactly the window [data_ptr, data_ptr+numel).
          This covers contiguous tensors and dense permutations (transpose /
          permute), because an in-place element-wise map is invariant to the
          visiting order of the elements.
* strided - genuinely gappy views (e.g. x[:, ::2]).  Axes are enumerated in
          ascending-stride order so the fastest varying index follows the
          smallest stride.

Tensors of one dtype that are flat addressable *and* share a numel are served by
a single launch: the grid is (nt, chunks) and each program selects its tensor
from program_id(0) through a compile-time-pruned branch chain.  numel/nt are
constexpr, so only pointers are runtime arguments.

Measured on this target, the per-call launch path - not the kernel - dominates
the small and mid-size workloads: a bare launch costs ~29 us while the streamed
kernels already reach the asymptotic ~1.5 TB/s.  After the first call for a
given specialization, the compiled kernel is therefore kept bound and launched
with only its runtime pointer arguments, which skips Triton's per-call key
computation, argument binding and specialization scan.  Pointer alignment is
part of the compile key, so it is re-checked every call; any oddity (unaligned
pointer, unavailable binding, first call) uses the ordinary JIT launch, which
also stays the fallback if a bound launch ever fails.
"""

import torch
import triton
import triton.language as tl

_MAX_DIMS = 8
_BATCH = 16
_BLOCK = 1024

_MISS = object()
# (device, kind, numel, even, nt) -> bound CompiledKernel, or None if unavailable.
_BOUND = {}


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


@triton.jit
def _abs_body(p, idx, N: tl.constexpr, EVEN: tl.constexpr):
    if EVEN:
        tl.store(p + idx, tl.abs(tl.load(p + idx)))
    else:
        m = idx < N
        tl.store(p + idx, tl.abs(tl.load(p + idx, mask=m)), mask=m)


@triton.jit
def _abs_flat_static(p, N: tl.constexpr, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    off = tl.program_id(0) * BLOCK
    _abs_body(p, off + tl.arange(0, BLOCK), N, EVEN)


@triton.jit
def _abs_batch_static(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8,
    p9,
    p10,
    p11,
    p12,
    p13,
    p14,
    p15,
    NT: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    t = tl.program_id(0)
    idx = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    if t == 0:
        _abs_body(p0, idx, N, EVEN)
    if NT > 1:
        if t == 1:
            _abs_body(p1, idx, N, EVEN)
        if t == 2:
            _abs_body(p2, idx, N, EVEN)
        if t == 3:
            _abs_body(p3, idx, N, EVEN)
        if t == 4:
            _abs_body(p4, idx, N, EVEN)
        if t == 5:
            _abs_body(p5, idx, N, EVEN)
        if t == 6:
            _abs_body(p6, idx, N, EVEN)
        if t == 7:
            _abs_body(p7, idx, N, EVEN)
    if NT > 8:
        if t == 8:
            _abs_body(p8, idx, N, EVEN)
        if t == 9:
            _abs_body(p9, idx, N, EVEN)
        if t == 10:
            _abs_body(p10, idx, N, EVEN)
        if t == 11:
            _abs_body(p11, idx, N, EVEN)
        if t == 12:
            _abs_body(p12, idx, N, EVEN)
        if t == 13:
            _abs_body(p13, idx, N, EVEN)
        if t == 14:
            _abs_body(p14, idx, N, EVEN)
        if t == 15:
            _abs_body(p15, idx, N, EVEN)


@triton.jit
def _abs_strided_kernel(
    p,
    n,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
    t6,
    t7,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK
    idx = off + tl.arange(0, BLOCK)
    mask = idx < n
    rem = idx
    ptr_off = tl.zeros([BLOCK], dtype=tl.int64)
    if RANK > 0:
        ptr_off += (rem % s0).to(tl.int64) * t0
        rem = rem // s0
    if RANK > 1:
        ptr_off += (rem % s1).to(tl.int64) * t1
        rem = rem // s1
    if RANK > 2:
        ptr_off += (rem % s2).to(tl.int64) * t2
        rem = rem // s2
    if RANK > 3:
        ptr_off += (rem % s3).to(tl.int64) * t3
        rem = rem // s3
    if RANK > 4:
        ptr_off += (rem % s4).to(tl.int64) * t4
        rem = rem // s4
    if RANK > 5:
        ptr_off += (rem % s5).to(tl.int64) * t5
        rem = rem // s5
    if RANK > 6:
        ptr_off += (rem % s6).to(tl.int64) * t6
        rem = rem // s6
    if RANK > 7:
        ptr_off += (rem % s7).to(tl.int64) * t7
        rem = rem // s7
    x = tl.load(p + ptr_off, mask=mask)
    tl.store(p + ptr_off, tl.abs(x), mask=mask)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _is_flat_addressable(t):
    """True when [data_ptr, data_ptr + numel) is exactly the addressed set."""
    if t.is_contiguous():
        return True
    if t.numel() <= 1:
        return True
    expected = 1
    for stride, size in sorted(zip(t.stride(), t.shape)):
        if size == 1:
            continue
        if stride != expected:
            return False
        expected *= size
    return True


def _has_internal_overlap(t):
    fn = getattr(torch, "_debug_has_internal_overlap", None)
    if fn is None:
        return False
    return fn(t) == 1


def _launch_one(t, num):
    even = num % _BLOCK == 0
    rows = num // _BLOCK if even else triton.cdiv(num, _BLOCK)
    if not t.data_ptr() & 15:
        key = (t.device.index, 0, num, even)
        ck = _BOUND.get(key, _MISS)
        if ck is _MISS:
            # First call for this specialization: launch through the JIT (which
            # also validates the exact kernel) and bind it for the next call.
            _abs_flat_static[(rows,)](t, num, _BLOCK, even, num_warps=4)
            try:
                _BOUND[key] = _abs_flat_static.warmup(
                    t, num, _BLOCK, even, grid=(rows, 1, 1), num_warps=4
                )
            except Exception:
                _BOUND[key] = None
            return
        if ck is not None:
            try:
                ck[(rows, 1, 1)](t)
                return
            except Exception:
                _BOUND[key] = None
    _abs_flat_static[(rows,)](t, num, _BLOCK, even, num_warps=4)


def _launch_group(ts, num):
    nts = len(ts)
    if nts == 1:
        _launch_one(ts[0], num)
        return
    even = num % _BLOCK == 0
    rows = num // _BLOCK if even else triton.cdiv(num, _BLOCK)
    device = ts[0].device.index
    i = 0
    while i < nts:
        chunk = ts[i : i + _BATCH]
        nt = len(chunk)
        first = chunk[0]
        ptrs = chunk + [first] * (_BATCH - nt)
        aligned = True
        for p in ptrs:
            if p.data_ptr() & 15:
                aligned = False
                break
        if aligned:
            key = (device, nt, num, even)
            ck = _BOUND.get(key, _MISS)
            if ck is _MISS:
                _abs_batch_static[(nt, rows)](*ptrs, nt, num, _BLOCK, even, num_warps=4)
                try:
                    _BOUND[key] = _abs_batch_static.warmup(
                        *ptrs,
                        nt,
                        num,
                        _BLOCK,
                        even,
                        grid=(nt, rows, 1),
                        num_warps=4,
                    )
                except Exception:
                    _BOUND[key] = None
                i += _BATCH
                continue
            if ck is not None:
                try:
                    ck[(nt, rows, 1)](*ptrs)
                    i += _BATCH
                    continue
                except Exception:
                    _BOUND[key] = None
        _abs_batch_static[(nt, rows)](*ptrs, nt, num, _BLOCK, even, num_warps=4)
        i += _BATCH


def _launch_strided(t):
    num = t.numel()
    axes = sorted((s, sz) for s, sz in zip(t.stride(), t.shape) if sz != 1)
    rank = len(axes)
    if rank > _MAX_DIMS:
        # Peel the leading axis with zero-copy views until the rank fits.
        for i in range(t.shape[0]):
            _launch_strided(t[i])
        return
    sizes = [1] * _MAX_DIMS
    strides = [0] * _MAX_DIMS
    for i, (s, sz) in enumerate(axes):
        sizes[i] = int(sz)
        strides[i] = int(s)
    _abs_strided_kernel[(triton.cdiv(num, _BLOCK),)](
        t,
        num,
        *sizes,
        *strides,
        rank,
        _BLOCK,
        num_warps=4,
    )


def _launch_gappy(t):
    if _has_internal_overlap(t):
        raise RuntimeError(
            "unsupported operation: more than one element of the written-to "
            "tensor refers to a single memory location. Please clone() the "
            "tensor before performing the operation."
        )
    _launch_strided(t)


def _run_general_one(t):
    if not isinstance(t, torch.Tensor):
        raise TypeError("argument 'self' must be tuple of Tensors")
    dt = t.dtype
    if dt.is_complex:
        raise RuntimeError("In-place abs is not supported for complex tensors.")
    if dt == torch.bool or t.numel() == 0:
        return
    if _is_flat_addressable(t):
        _launch_one(t, t.numel())
    else:
        _launch_gappy(t)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def foreach_abs_(self):
    if not isinstance(self, (list, tuple)):
        raise TypeError("argument 'self' must be tuple of Tensors")
    count = len(self)
    if count == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")

    first = self[0]
    if not isinstance(first, torch.Tensor):
        raise TypeError("argument 'self' must be tuple of Tensors")
    dtype = first.dtype
    if dtype.is_complex:
        raise RuntimeError("In-place abs is not supported for complex tensors.")

    # Fast path: one tensor, or a homogeneous contiguous list (the benchmark
    # shape).  Needs no per-tensor bookkeeping beyond the equality check.
    num = first.numel()
    if num != 0 and dtype != torch.bool and first.is_contiguous():
        homogeneous = True
        for t in self[1:]:
            if (
                not isinstance(t, torch.Tensor)
                or t.dtype != dtype
                or t.numel() != num
                or not t.is_contiguous()
            ):
                homogeneous = False
                break
        if homogeneous:
            _launch_group(list(self), num)
            return None

    # General path: group by dtype, then by numel; gappy views go per tensor.
    groups = {}
    for t in self:
        if not isinstance(t, torch.Tensor):
            raise TypeError("argument 'self' must be tuple of Tensors")
        dt = t.dtype
        if dt.is_complex:
            raise RuntimeError("In-place abs is not supported for complex tensors.")
        if dt == torch.bool or t.numel() == 0:
            continue
        if not _is_flat_addressable(t):
            _launch_gappy(t)
            continue
        num = t.numel()
        bucket = groups.get(dt)
        if bucket is None:
            groups[dt] = bucket = {}
        sub = bucket.get(num)
        if sub is None:
            bucket[num] = [t]
        else:
            sub.append(t)

    for by_num in groups.values():
        for num, ts in by_num.items():
            _launch_group(ts, num)
    return None
