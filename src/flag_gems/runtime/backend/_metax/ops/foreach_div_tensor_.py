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

"""Triton implementation of aten::_foreach_div_.Tensor.

foreach_div_tensor_(self, other): divides every tensor in the sequence `self` in place by the
single-element tensor `other` (PyTorch opmath semantics: float32 for
fp16/bf16/fp32, float64 for fp64) and returns None.

Contiguous tensors of the list are fused into a single multi-pointer launch
(up to _MAXG pointers, per-slot runtime element count), so a short list costs
one launch instead of one per tensor.  Strided tensors fall back to a
per-tensor canonicalised index-decomposition kernel.
"""

import torch
import triton
import triton.language as tl
from triton import knobs as _knobs
from triton.runtime.driver import driver as _driver

_BLOCK = 1024
_WARPS = 4
_MAXG = 8
_I32MAX = 0x7FFFFFFF

# Direct-launch cache: the tiny workloads are launch bound, so the equal-shape
# fast path bypasses JITFunction.run's per-call binder/cache-key/validation
# work and calls the compiled kernel's launcher directly.  Entries are keyed by
# everything that changes the compiled binary (dtypes, pointer alignment,
# integer specialisation) and never contain tensor values.
_DIRECT = {}
_CTX = []


@triton.jit
def _slot(p, offs, n, o, ACC64: tl.constexpr):
    m = offs < n
    v = tl.load(p + offs, mask=m, other=0.0)
    if ACC64:
        r = tl.fdiv(v.to(tl.float64), o, ieee_rounding=True)
    else:
        r = tl.fdiv(v.to(tl.float32), o, ieee_rounding=True)
    tl.store(p + offs, r, mask=m)


@triton.jit
def _fdiv_g4(p0, p1, p2, p3, n, ov, ACC64: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if ACC64:
        o = tl.load(ov).to(tl.float64)
    else:
        o = tl.load(ov).to(tl.float32)
    _slot(p0, offs, n, o, ACC64)
    _slot(p1, offs, n, o, ACC64)
    _slot(p2, offs, n, o, ACC64)
    _slot(p3, offs, n, o, ACC64)


@triton.jit
def _fdiv_group(
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
    ov,
    IDX64: tl.constexpr,
    ACC64: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if IDX64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    if ACC64:
        o = tl.load(ov).to(tl.float64)
    else:
        o = tl.load(ov).to(tl.float32)
    _slot(p0, offs, n0, o, ACC64)
    if GROUP > 1:
        _slot(p1, offs, n1, o, ACC64)
    if GROUP > 2:
        _slot(p2, offs, n2, o, ACC64)
    if GROUP > 3:
        _slot(p3, offs, n3, o, ACC64)
    if GROUP > 4:
        _slot(p4, offs, n4, o, ACC64)
    if GROUP > 5:
        _slot(p5, offs, n5, o, ACC64)
    if GROUP > 6:
        _slot(p6, offs, n6, o, ACC64)
    if GROUP > 7:
        _slot(p7, offs, n7, o, ACC64)


@triton.jit
def _fdiv_scalar_flat(
    p, n, ov, IDX64: tl.constexpr, ACC64: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    if IDX64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    if ACC64:
        o = tl.load(ov).to(tl.float64)
    else:
        o = tl.load(ov).to(tl.float32)
    _slot(p, offs, n, o, ACC64)


@triton.jit
def _fdiv_scalar_strided(
    p,
    n,
    ov,
    d0,
    s0,
    d1,
    s1,
    d2,
    s2,
    d3,
    s3,
    d4,
    s4,
    d5,
    s5,
    d6,
    s6,
    d7,
    s7,
    RANK: tl.constexpr,
    ACC64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    m = idx < n
    rem = idx
    c = rem % d0
    rem = rem // d0
    off = c * s0
    if RANK > 1:
        c = rem % d1
        rem = rem // d1
        off = off + c * s1
    if RANK > 2:
        c = rem % d2
        rem = rem // d2
        off = off + c * s2
    if RANK > 3:
        c = rem % d3
        rem = rem // d3
        off = off + c * s3
    if RANK > 4:
        c = rem % d4
        rem = rem // d4
        off = off + c * s4
    if RANK > 5:
        c = rem % d5
        rem = rem // d5
        off = off + c * s5
    if RANK > 6:
        c = rem % d6
        rem = rem // d6
        off = off + c * s6
    if RANK > 7:
        c = rem % d7
        rem = rem // d7
        off = off + c * s7
    v = tl.load(p + off, mask=m, other=0.0)
    o = tl.load(ov)
    if ACC64:
        r = tl.fdiv(v.to(tl.float64), o.to(tl.float64), ieee_rounding=True)
    else:
        r = tl.fdiv(v.to(tl.float32), o.to(tl.float32), ieee_rounding=True)
    tl.store(p + off, r, mask=m)


def _canonical(t):
    """Return (rank, [d0, s0, ..., d7, s7]) with dims innermost-first."""
    pairs = [(sz, st) for sz, st in zip(t.shape, t.stride()) if sz != 1]
    for sz, st in pairs:
        if st == 0:
            raise RuntimeError(
                "some tensors are expanded (stride 0) and cannot be divided in place"
            )
    pairs.sort(key=lambda x: x[1])
    merged = []
    for sz, st in pairs:
        if merged and st == merged[-1][1] * merged[-1][0]:
            merged[-1] = (merged[-1][0] * sz, merged[-1][1])
        else:
            merged.append((sz, st))
    rank = len(merged)
    if rank == 0:
        merged = [(1, 0)]
        rank = 1
    if rank > 8:
        raise RuntimeError("tensor layout rank exceeds 8 after canonicalisation")
    flat = []
    for sz, st in merged:
        flat.append(sz)
        flat.append(st)
    while len(flat) < 16:
        flat.append(1)
        flat.append(0)
    return rank, flat


def _flush_group(ptrs, ns, ov, acc64):
    g = len(ptrs)
    nmax = max(ns)
    grid = ((nmax + _BLOCK - 1) // _BLOCK,)
    pad = _MAXG - g
    if pad:
        ptrs = ptrs + [ptrs[0]] * pad
        ns = ns + [0] * pad
    _fdiv_group[grid](
        ptrs[0],
        ptrs[1],
        ptrs[2],
        ptrs[3],
        ptrs[4],
        ptrs[5],
        ptrs[6],
        ptrs[7],
        ns[0],
        ns[1],
        ns[2],
        ns[3],
        ns[4],
        ns[5],
        ns[6],
        ns[7],
        ov,
        nmax > _I32MAX,
        acc64,
        g,
        _BLOCK,
        num_warps=_WARPS,
    )


def _launch_strided(t, ov):
    n = t.numel()
    rank, dims = _canonical(t)
    grid = ((n + _BLOCK - 1) // _BLOCK,)
    _fdiv_scalar_strided[grid](
        t,
        n,
        ov,
        *dims,
        RANK=rank,
        ACC64=t.dtype == torch.float64,
        BLOCK=_BLOCK,
        num_warps=_WARPS,
    )


def _run_scalar_operand(tensors, ov):
    ptrs = []
    ns = []
    acc64 = False
    for t in tensors:
        dt = t.dtype
        if not dt.is_floating_point:
            raise RuntimeError(
                "result type Float can't be cast to the desired output type "
                + str(dt).replace("torch.", "")
            )
        n = t.numel()
        if n == 0:
            continue
        if t.is_contiguous():
            is64 = dt == torch.float64
            if ptrs and is64 != acc64:
                _flush_group(ptrs, ns, ov, acc64)
                ptrs = []
                ns = []
            acc64 = is64
            ptrs.append(t)
            ns.append(n)
            if len(ptrs) == _MAXG:
                _flush_group(ptrs, ns, ov, acc64)
                ptrs = []
                ns = []
        else:
            if ptrs:
                _flush_group(ptrs, ns, ov, acc64)
                ptrs = []
                ns = []
            _launch_strided(t, ov)
    if ptrs:
        _flush_group(ptrs, ns, ov, acc64)


def foreach_div_tensor_(self, other):
    # Hot path: the common case is a short list of equal-shaped contiguous
    # tensors of one dtype.  One launch with a single shared element count,
    # issued through the cached compiled kernel when every pointer is 16-byte
    # aligned (the alignment that binary was specialised for); otherwise the
    # ordinary Triton launch is used, which is always correct.
    if type(self) is list and len(self) == 4 and torch.is_tensor(other):
        a, b, c, d = self
        dt = a.dtype
        if (
            b.dtype == dt
            and c.dtype == dt
            and d.dtype == dt
            and a.is_contiguous()
            and b.is_contiguous()
            and c.is_contiguous()
            and d.is_contiguous()
        ):
            n = a.numel()
            if (
                n
                and n <= _I32MAX
                and n == b.numel()
                and n == c.numel()
                and n == d.numel()
                and other.numel() == 1
            ):
                if not dt.is_floating_point:
                    raise RuntimeError(
                        "result type Float can't be cast to the desired output "
                        "type " + str(dt).replace("torch.", "")
                    )
                acc64 = dt == torch.float64
                g = (n + 1023) >> 10
                if not (
                    (
                        a.data_ptr()
                        | b.data_ptr()
                        | c.data_ptr()
                        | d.data_ptr()
                        | other.data_ptr()
                    )
                    & 15
                ):
                    key = (dt, other.dtype, n & 15 == 0, n == 1)
                    entry = _DIRECT.get(key)
                    if entry is None:
                        try:
                            compiled = _fdiv_g4.warmup(
                                a,
                                b,
                                c,
                                d,
                                n,
                                other,
                                ACC64=acc64,
                                BLOCK=_BLOCK,
                                num_warps=_WARPS,
                                grid=(1, 1, 1),
                            )
                            compiled._init_handles()
                            if not _CTX:
                                _CTX.extend(
                                    (
                                        _driver.active.get_current_device(),
                                        _knobs.runtime.launch_enter_hook,
                                        _knobs.runtime.launch_exit_hook,
                                        _driver.active.get_current_stream,
                                    )
                                )
                            entry = (
                                compiled.run,
                                compiled.function,
                                compiled.packed_metadata,
                            )
                        except Exception:
                            entry = False
                        _DIRECT[key] = entry
                    if entry:
                        launcher, fn, meta = entry
                        launcher(
                            g,
                            1,
                            1,
                            _CTX[3](_CTX[0]),
                            fn,
                            meta,
                            None,
                            _CTX[1],
                            _CTX[2],
                            a,
                            b,
                            c,
                            d,
                            n,
                            other,
                            acc64,
                            _BLOCK,
                        )
                        return None
                _fdiv_g4[(g,)](a, b, c, d, n, other, acc64, _BLOCK, num_warps=_WARPS)
                return None

    tensors = self if isinstance(self, (list, tuple)) else (self,)

    if torch.is_tensor(other):
        if other.numel() != 1:
            raise RuntimeError("the tensor operand must have exactly one element")
        ov = other
    elif isinstance(other, (list, tuple)):
        if len(other) != len(tensors):
            raise RuntimeError(
                "Tensors of `self` and `other` must have the same length"
            )
        for t, o in zip(tensors, other):
            if torch.is_tensor(o):
                if o.numel() != 1:
                    raise RuntimeError(
                        "the tensor operand must have exactly one element"
                    )
            else:
                o = torch.tensor(
                    [o],
                    dtype=torch.float64 if t.dtype == torch.float64 else torch.float32,
                    device=t.device,
                )
            if not t.is_contiguous():
                if t.numel():
                    _launch_strided(t, o)
            else:
                _run_scalar_operand((t,), o)
        return None
    else:
        acc64 = any(t.dtype == torch.float64 for t in tensors)
        ov = torch.tensor(
            [other],
            dtype=torch.float64 if acc64 else torch.float32,
            device=tensors[0].device,
        )

    _run_scalar_operand(tensors, ov)
    return None
