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

"""In-place foreach clamp_max between a tensor list and a scalar list.

Mirrors ``aten::_foreach_clamp_max_.ScalarList``: every element of every input
tensor is replaced by ``min(x, bound_i)`` inside the tensor's own storage and
``run`` returns ``None``.  The bound is converted to the tensor dtype before the
comparison (as ATen's ``Scalar::to<scalar_t>()`` does) and NaN propagates.

Layout of the implementation:

* uniform fast path - four contiguous same-dtype same-numel tensors with plain
  number bounds are handled by a single launch of ``_clamp_uni4`` (nine runtime
  arguments, slot = ``pid // blocks_per_tensor``), which keeps the host launch
  path as short as possible because these calls are host-overhead bound;
* general fused path - contiguous same-dtype groups of up to four tensors with
  arbitrary per-tensor numels go through ``_clamp_fused4`` with cumulative
  block counts;
* noncontiguous tensors use a stride-decomposed gather/scatter kernel with the
  smallest-magnitude stride placed innermost (legal because the op is
  elementwise), so transposed inputs still run on unit-stride lanes;
* aliased entries fall back to sequential per-tensor launches so mutation order
  matches the reference, empty tensors are skipped, and a length mismatch
  raises before anything is mutated.
"""

import torch
import triton
import triton.language as tl

_MAX_RANK = 8
_FUSED_SLOTS = 4
_BLOCK = 1024
_NUM_WARPS = 4


@triton.jit
def _clamp_body(
    ptr,
    offs,
    n,
    bound,
    EVEN: tl.constexpr,
    IS_FP: tl.constexpr,
):
    if EVEN:
        v = tl.load(ptr + offs)
        s = bound.to(v.dtype)
        if IS_FP:
            r = tl.minimum(v, s, propagate_nan=tl.PropagateNan.ALL)
        else:
            r = tl.minimum(v, s)
        tl.store(ptr + offs, r)
    else:
        mask = offs < n
        v = tl.load(ptr + offs, mask=mask, other=0)
        s = bound.to(v.dtype)
        if IS_FP:
            r = tl.minimum(v, s, propagate_nan=tl.PropagateNan.ALL)
        else:
            r = tl.minimum(v, s)
        tl.store(ptr + offs, r, mask=mask)


@triton.jit
def _clamp_uni4(
    p0,
    p1,
    p2,
    p3,
    n,
    step,
    b0,
    b1,
    b2,
    b3,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    IS_FP: tl.constexpr,
):
    pid = tl.program_id(0)
    k = pid // step
    offs = (pid - k * step) * BLOCK + tl.arange(0, BLOCK)
    if k == 0:
        _clamp_body(p0, offs, n, b0, EVEN, IS_FP)
    elif k == 1:
        _clamp_body(p1, offs, n, b1, EVEN, IS_FP)
    elif k == 2:
        _clamp_body(p2, offs, n, b2, EVEN, IS_FP)
    else:
        _clamp_body(p3, offs, n, b3, EVEN, IS_FP)


@triton.jit
def _slot_clamp(
    ptr,
    base,
    n,
    bound,
    BLOCK: tl.constexpr,
    LONG: tl.constexpr,
    EVEN: tl.constexpr,
    IS_FP: tl.constexpr,
):
    pid = tl.program_id(0)
    if LONG:
        offs = (pid.to(tl.int64) - base) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = (pid - base) * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        v = tl.load(ptr + offs)
        s = bound.to(v.dtype)
        if IS_FP:
            r = tl.minimum(v, s, propagate_nan=tl.PropagateNan.ALL)
        else:
            r = tl.minimum(v, s)
        tl.store(ptr + offs, r)
    else:
        mask = offs < n
        v = tl.load(ptr + offs, mask=mask, other=0)
        s = bound.to(v.dtype)
        if IS_FP:
            r = tl.minimum(v, s, propagate_nan=tl.PropagateNan.ALL)
        else:
            r = tl.minimum(v, s)
        tl.store(ptr + offs, r, mask=mask)


@triton.jit
def _clamp_fused4(
    p0,
    p1,
    p2,
    p3,
    n0,
    n1,
    n2,
    n3,
    b0,
    b1,
    b2,
    b3,
    c1,
    c2,
    c3,
    c4,
    BLOCK: tl.constexpr,
    LONG: tl.constexpr,
    EVEN: tl.constexpr,
    IS_FP: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < c1:
        _slot_clamp(p0, 0, n0, b0, BLOCK, LONG, EVEN, IS_FP)
    elif pid < c2:
        _slot_clamp(p1, c1, n1, b1, BLOCK, LONG, EVEN, IS_FP)
    elif pid < c3:
        _slot_clamp(p2, c2, n2, b2, BLOCK, LONG, EVEN, IS_FP)
    else:
        _slot_clamp(p3, c3, n3, b3, BLOCK, LONG, EVEN, IS_FP)


@triton.jit
def _clamp_max_contig(
    ptr,
    n_elem,
    bound,
    BLOCK: tl.constexpr,
    LONG: tl.constexpr,
    IS_FP: tl.constexpr,
):
    pid = tl.program_id(0)
    if LONG:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    v = tl.load(ptr + offs, mask=mask, other=0)
    s = bound.to(v.dtype)
    if IS_FP:
        r = tl.minimum(v, s, propagate_nan=tl.PropagateNan.ALL)
    else:
        r = tl.minimum(v, s)
    tl.store(ptr + offs, r, mask=mask)


@triton.jit
def _clamp_max_strided(
    ptr,
    n_elem,
    bound,
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
    BLOCK: tl.constexpr,
    NDIM: tl.constexpr,
    IS_FP: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n_elem
    rem = offs
    off = tl.zeros([BLOCK], tl.int64)
    if NDIM > 0:
        off += (rem % s0) * t0
        rem = rem // s0
    if NDIM > 1:
        off += (rem % s1) * t1
        rem = rem // s1
    if NDIM > 2:
        off += (rem % s2) * t2
        rem = rem // s2
    if NDIM > 3:
        off += (rem % s3) * t3
        rem = rem // s3
    if NDIM > 4:
        off += (rem % s4) * t4
        rem = rem // s4
    if NDIM > 5:
        off += (rem % s5) * t5
        rem = rem // s5
    if NDIM > 6:
        off += (rem % s6) * t6
        rem = rem // s6
    if NDIM > 7:
        off += (rem % s7) * t7
        rem = rem // s7
    v = tl.load(ptr + off, mask=mask, other=0)
    s = bound.to(v.dtype)
    if IS_FP:
        r = tl.minimum(v, s, propagate_nan=tl.PropagateNan.ALL)
    else:
        r = tl.minimum(v, s)
    tl.store(ptr + off, r, mask=mask)


def _as_bound(x):
    if isinstance(x, torch.Tensor):
        return x.item()
    return x


def _normalize_bounds(scalars, n):
    if isinstance(scalars, (list, tuple)):
        if len(scalars) != n:
            raise RuntimeError(
                "foreach_clamp_max_: number of tensors and scalars must match"
            )
        return [_as_bound(s) for s in scalars]
    b = _as_bound(scalars)
    return [b] * n


def _launch_contig(t, bound):
    n = t.numel()
    grid = ((n + _BLOCK - 1) // _BLOCK,)
    _clamp_max_contig[grid](
        t,
        n,
        bound,
        BLOCK=_BLOCK,
        LONG=n > 0x7FFFFFF0,
        IS_FP=t.dtype.is_floating_point,
        num_warps=_NUM_WARPS,
    )


def _launch_strided(t, bound):
    n = t.numel()
    nd = t.dim()
    shape = t.shape
    strides = t.stride()
    dims = [d for d in range(nd) if shape[d] != 1]
    # elementwise op: traverse so the smallest-magnitude stride is innermost
    dims = sorted(dims, key=lambda d: abs(strides[d]))
    sizes = [shape[d] for d in dims]
    steps = [strides[d] for d in dims]
    ndim = len(dims)
    while len(sizes) < _MAX_RANK:
        sizes.append(1)
        steps.append(0)
    grid = ((n + _BLOCK - 1) // _BLOCK,)
    _clamp_max_strided[grid](
        t,
        n,
        bound,
        sizes[0],
        sizes[1],
        sizes[2],
        sizes[3],
        sizes[4],
        sizes[5],
        sizes[6],
        sizes[7],
        steps[0],
        steps[1],
        steps[2],
        steps[3],
        steps[4],
        steps[5],
        steps[6],
        steps[7],
        BLOCK=_BLOCK,
        NDIM=ndim,
        IS_FP=t.dtype.is_floating_point,
        num_warps=_NUM_WARPS,
    )


def _launch_fused4(ts, bounds):
    a = ts[0]
    b = ts[1]
    n0 = a.numel()
    n1 = b.numel()
    c1 = (n0 + _BLOCK - 1) // _BLOCK
    c2 = c1 + (n1 + _BLOCK - 1) // _BLOCK
    even = (n0 % _BLOCK == 0) and (n1 % _BLOCK == 0)
    long_ = n0 > 0x7FFFFFF0 or n1 > 0x7FFFFFF0
    if len(ts) > 2:
        c = ts[2]
        n2 = c.numel()
        c3 = c2 + (n2 + _BLOCK - 1) // _BLOCK
        even = even and (n2 % _BLOCK == 0)
        long_ = long_ or n2 > 0x7FFFFFF0
    else:
        c = a
        n2 = 1
        c3 = c2
    if len(ts) > 3:
        d = ts[3]
        n3 = d.numel()
        c4 = c3 + (n3 + _BLOCK - 1) // _BLOCK
        even = even and (n3 % _BLOCK == 0)
        long_ = long_ or n3 > 0x7FFFFFF0
    else:
        d = a
        n3 = 1
        c4 = c3
    _clamp_fused4[(c4,)](
        a,
        b,
        c,
        d,
        n0,
        n1,
        n2,
        n3,
        bounds[0],
        bounds[1],
        bounds[2] if len(bounds) > 2 else 0.0,
        bounds[3] if len(bounds) > 3 else 0.0,
        c1,
        c2,
        c3,
        c4,
        BLOCK=_BLOCK,
        LONG=long_,
        EVEN=even,
        IS_FP=a.dtype.is_floating_point,
        num_warps=_NUM_WARPS,
    )


def _general_run(self, scalars):
    ts = list(self)
    nt = len(ts)
    bounds = _normalize_bounds(scalars, nt)
    i = 0
    while i < nt:
        t = ts[i]
        if t.numel() == 0:
            i += 1
            continue
        if not t.is_contiguous():
            _launch_strided(t, bounds[i])
            i += 1
            continue
        dt = t.dtype
        group = [t]
        gb = [bounds[i]]
        j = i + 1
        while j < nt and len(group) < _FUSED_SLOTS:
            u = ts[j]
            if u.dtype is not dt or not u.is_contiguous() or u.numel() == 0:
                break
            group.append(u)
            gb.append(bounds[j])
            j += 1
        i = j
        if len(group) == 1:
            _launch_contig(group[0], gb[0])
        elif len({u.data_ptr() for u in group}) != len(group):
            for u, ub in zip(group, gb):
                _launch_contig(u, ub)
        else:
            _launch_fused4(group, gb)


def foreach_clamp_max_scalar_list_(self, scalars):
    # ---- uniform fast path: four contiguous same-dtype same-numel tensors
    if type(self) is list or type(self) is tuple:
        if len(self) == 4:
            t0 = self[0]
            n = t0.numel()
            dt = t0.dtype
            if (
                0 < n <= 0x7FFFFFF0
                and t0.is_contiguous()
                and (type(scalars) is list or type(scalars) is tuple)
                and len(scalars) == 4
            ):
                t1 = self[1]
                t2 = self[2]
                t3 = self[3]
                if (
                    t1.numel() == n
                    and t2.numel() == n
                    and t3.numel() == n
                    and t1.dtype is dt
                    and t2.dtype is dt
                    and t3.dtype is dt
                    and t1.is_contiguous()
                    and t2.is_contiguous()
                    and t3.is_contiguous()
                ):
                    b0 = scalars[0]
                    b1 = scalars[1]
                    b2 = scalars[2]
                    b3 = scalars[3]
                    k0 = type(b0)
                    k1 = type(b1)
                    k2 = type(b2)
                    k3 = type(b3)
                    if (
                        (k0 is float or k0 is int)
                        and (k1 is float or k1 is int)
                        and (k2 is float or k2 is int)
                        and (k3 is float or k3 is int)
                        and len(
                            {t0.data_ptr(), t1.data_ptr(), t2.data_ptr(), t3.data_ptr()}
                        )
                        == 4
                    ):
                        step = (n + _BLOCK - 1) // _BLOCK
                        _clamp_uni4[(step * 4,)](
                            t0,
                            t1,
                            t2,
                            t3,
                            n,
                            step,
                            b0,
                            b1,
                            b2,
                            b3,
                            BLOCK=_BLOCK,
                            EVEN=(n % _BLOCK == 0),
                            IS_FP=dt.is_floating_point,
                            num_warps=_NUM_WARPS,
                        )
                        return None
    _general_run(self, scalars)
    return None
