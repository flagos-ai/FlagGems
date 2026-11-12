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

"""foreach_minimum_scalar_list_ (in-place foreach minimum with scalar list).

Semantics: for every pair (t, s) in (self, scalars):
    t[i] = minimum(t[i], s)     # NaN propagating, scalar rounded to t.dtype first
The op mutates every tensor in ``self`` and returns ``None``.
"""

import torch
import triton
import triton.language as tl

_MAXDIM = 8
_MAXT = 8
# Above this many elements per tensor, one launch per tensor (single contiguous
# stream per kernel) beats fusing several distant streams into one CTA.
_FUSE_MAX = 1 << 27


@triton.jit
def _min_tile(p, s, offs, IS_FLOAT: tl.constexpr):
    """Unmasked exact tile: minimum of the tile at ``offs`` with scalar ``s``."""
    x = tl.load(p + offs)
    sv = s.to(x.dtype)
    if IS_FLOAT:
        y = tl.minimum(x, sv, propagate_nan=tl.PropagateNan.ALL)
    else:
        y = tl.minimum(x, sv)
    tl.store(p + offs, y)


@triton.jit(do_not_specialize=["s0", "s1", "s2", "s3"])
def _min_multi4_even_kernel(
    x0,
    x1,
    x2,
    x3,
    s0,
    s1,
    s2,
    s3,
    BLOCK: tl.constexpr,
    IS_FLOAT: tl.constexpr,
):
    """Four uniformly shaped contiguous tensors, one unmasked tile each.

    All four tiles are loaded before any store so the warp keeps four
    independent memory streams in flight.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(x0 + offs, eviction_policy="evict_first")
    b = tl.load(x1 + offs, eviction_policy="evict_first")
    c = tl.load(x2 + offs, eviction_policy="evict_first")
    d = tl.load(x3 + offs, eviction_policy="evict_first")
    if IS_FLOAT:
        a = tl.minimum(a, s0.to(a.dtype), propagate_nan=tl.PropagateNan.ALL)
        b = tl.minimum(b, s1.to(b.dtype), propagate_nan=tl.PropagateNan.ALL)
        c = tl.minimum(c, s2.to(c.dtype), propagate_nan=tl.PropagateNan.ALL)
        d = tl.minimum(d, s3.to(d.dtype), propagate_nan=tl.PropagateNan.ALL)
    else:
        a = tl.minimum(a, s0.to(a.dtype))
        b = tl.minimum(b, s1.to(b.dtype))
        c = tl.minimum(c, s2.to(c.dtype))
        d = tl.minimum(d, s3.to(d.dtype))
    tl.store(x0 + offs, a, eviction_policy="evict_first")
    tl.store(x1 + offs, b, eviction_policy="evict_first")
    tl.store(x2 + offs, c, eviction_policy="evict_first")
    tl.store(x3 + offs, d, eviction_policy="evict_first")


@triton.jit(
    do_not_specialize=[
        "s0",
        "s1",
        "s2",
        "s3",
        "s4",
        "s5",
        "s6",
        "s7",
    ]
)
def _min_multi_even_kernel(
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    NT: tl.constexpr,
    BLOCK: tl.constexpr,
    IS_FLOAT: tl.constexpr,
):
    """One launch for up to 8 uniformly shaped contiguous tensors."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    _min_tile(x0, s0, offs, IS_FLOAT)
    if NT > 1:
        _min_tile(x1, s1, offs, IS_FLOAT)
    if NT > 2:
        _min_tile(x2, s2, offs, IS_FLOAT)
    if NT > 3:
        _min_tile(x3, s3, offs, IS_FLOAT)
    if NT > 4:
        _min_tile(x4, s4, offs, IS_FLOAT)
    if NT > 5:
        _min_tile(x5, s5, offs, IS_FLOAT)
    if NT > 6:
        _min_tile(x6, s6, offs, IS_FLOAT)
    if NT > 7:
        _min_tile(x7, s7, offs, IS_FLOAT)


@triton.jit(do_not_specialize=["s"])
def _min_flat_even_kernel(x_ptr, s, BLOCK: tl.constexpr, IS_FLOAT: tl.constexpr):
    """Unmasked exact-tile variant: only launched when n % BLOCK == 0."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, eviction_policy="evict_first")
    sv = s.to(x.dtype)
    if IS_FLOAT:
        y = tl.minimum(x, sv, propagate_nan=tl.PropagateNan.ALL)
    else:
        y = tl.minimum(x, sv)
    tl.store(x_ptr + offs, y, eviction_policy="evict_first")


@triton.jit(do_not_specialize=["s", "n"])
def _min_flat_kernel(x_ptr, s, n, BLOCK: tl.constexpr, IS_FLOAT: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    sv = s.to(x.dtype)
    if IS_FLOAT:
        y = tl.minimum(x, sv, propagate_nan=tl.PropagateNan.ALL)
    else:
        y = tl.minimum(x, sv)
    tl.store(x_ptr + offs, y, mask=mask)


@triton.jit(
    do_not_specialize=[
        "s",
        "n",
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
def _min_strided_kernel(
    x_ptr,
    s,
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
    IS_FLOAT: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    rem = offs.to(tl.int64)
    addr = tl.zeros([BLOCK], dtype=tl.int64)
    # dims are given fastest-varying first; each block is resolved statically
    if NDIM >= 1:
        i = rem % d0
        rem = rem // d0
        addr += i * t0
    if NDIM >= 2:
        i = rem % d1
        rem = rem // d1
        addr += i * t1
    if NDIM >= 3:
        i = rem % d2
        rem = rem // d2
        addr += i * t2
    if NDIM >= 4:
        i = rem % d3
        rem = rem // d3
        addr += i * t3
    if NDIM >= 5:
        i = rem % d4
        rem = rem // d4
        addr += i * t4
    if NDIM >= 6:
        i = rem % d5
        rem = rem // d5
        addr += i * t5
    if NDIM >= 7:
        i = rem % d6
        rem = rem // d6
        addr += i * t6
    if NDIM >= 8:
        i = rem % d7
        rem = rem // d7
        addr += i * t7
    x = tl.load(x_ptr + addr, mask=mask, other=0)
    sv = s.to(x.dtype)
    if IS_FLOAT:
        y = tl.minimum(x, sv, propagate_nan=tl.PropagateNan.ALL)
    else:
        y = tl.minimum(x, sv)
    tl.store(x_ptr + addr, y, mask=mask)


def _tile(n):
    if n <= 128:
        return 128, 1
    if n <= 512:
        return 512, 2
    return 1024, 4


def _collapse(shape, stride):
    """Merge adjacent dims that are contiguous; keep slowest-first order."""
    dims = []
    for d, s in zip(shape, stride):
        if d == 1:
            continue
        if dims and dims[-1][1] == d * s:
            pd, ps = dims[-1]
            dims[-1] = (pd * d, ps)
        else:
            dims.append((d, s))
    return dims


def _min_one(t, s):
    n = t.numel()
    block, warps = _tile(n)
    grid = (triton.cdiv(n, block),)
    isf = bool(t.dtype.is_floating_point)
    if t.is_contiguous():
        if n % block == 0:
            _min_flat_even_kernel[(n // block,)](
                t, s, BLOCK=block, IS_FLOAT=isf, num_warps=warps
            )
        else:
            _min_flat_kernel[grid](t, s, n, BLOCK=block, IS_FLOAT=isf, num_warps=warps)
        return
    dims = _collapse(t.shape, t.stride())
    if not dims:
        _min_flat_kernel[grid](t, s, n, BLOCK=block, IS_FLOAT=isf, num_warps=warps)
        return
    rev = dims[::-1]
    if len(rev) > _MAXDIM:
        raise RuntimeError(
            "foreach_minimum: view with %d effective dims is unsupported" % len(rev)
        )
    args = []
    for k in range(_MAXDIM):
        if k < len(rev):
            args.append(int(rev[k][0]))
        else:
            args.append(1)
    for k in range(_MAXDIM):
        if k < len(rev):
            args.append(int(rev[k][1]))
        else:
            args.append(0)
    _min_strided_kernel[grid](
        t,
        s,
        n,
        *args,
        NDIM=len(rev),
        BLOCK=block,
        IS_FLOAT=isf,
        num_warps=warps,
    )


def _launch_multi(ts, ss, n, block):
    nt = len(ts)
    isf = bool(ts[0].dtype.is_floating_point)
    warps = max(1, block >> 8)
    grid = (n // block,)
    if nt == 4:
        _min_multi4_even_kernel[grid](
            ts[0],
            ts[1],
            ts[2],
            ts[3],
            ss[0],
            ss[1],
            ss[2],
            ss[3],
            BLOCK=block,
            IS_FLOAT=isf,
            num_warps=warps,
        )
        return
    pad = _MAXT - nt
    if pad:
        ts = ts + [ts[0]] * pad
        ss = ss + [ss[0]] * pad
    _min_multi_even_kernel[grid](
        ts[0],
        ts[1],
        ts[2],
        ts[3],
        ts[4],
        ts[5],
        ts[6],
        ts[7],
        ss[0],
        ss[1],
        ss[2],
        ss[3],
        ss[4],
        ss[5],
        ss[6],
        ss[7],
        NT=nt,
        BLOCK=block,
        IS_FLOAT=isf,
        num_warps=warps,
    )


def _try_multi(tensors, scalars):
    """Single launch for uniformly shaped contiguous, non-overlapping tensors."""
    t0 = tensors[0]
    n = t0.numel()
    if n == 0 or n > _FUSE_MAX or not t0.is_contiguous():
        return False
    block = 1024 if n > 512 else (512 if n > 128 else 128)
    if n % block:
        return False
    dt = t0.dtype
    dev = t0.device
    nb = n * t0.element_size()
    p0 = t0.data_ptr()
    lo = p0
    hi = p0 + nb
    for i in range(1, len(tensors)):
        t = tensors[i]
        if t.numel() != n or t.dtype != dt or t.device != dev or not t.is_contiguous():
            return False
        p = t.data_ptr()
        if p < hi and lo < p + nb:
            return False
        if p < lo:
            lo = p
        elif p + nb > hi:
            hi = p + nb
    for i in range(0, len(tensors), _MAXT):
        _launch_multi(tensors[i : i + _MAXT], scalars[i : i + _MAXT], n, block)
    return True


def foreach_minimum_scalar_list_(self, scalars):
    tensors = self
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    if isinstance(scalars, torch.Tensor) and scalars.dim() == 0:
        scalars = scalars.item()
    if isinstance(scalars, (int, float, bool)):
        scalars = [scalars] * len(tensors)
    else:
        scalars = list(scalars)
        if len(scalars) != len(tensors):
            raise RuntimeError("Tensor list and scalar list must have the same length")
    for i, s in enumerate(scalars):
        if isinstance(s, torch.Tensor):
            scalars[i] = s.item()
    nt = len(tensors)
    if nt == 0:
        return None
    if nt == 1:
        if tensors[0].numel():
            _min_one(tensors[0], scalars[0])
        return None
    if _try_multi(tensors, scalars):
        return None
    for t, s in zip(tensors, scalars):
        if t.numel() == 0:
            continue
        _min_one(t, s)
    return None
