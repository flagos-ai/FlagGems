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

"""Triton implementation of aten::_foreach_mul.Tensor (tensor list x 0-dim tensor).

out_k = self_k * other  (elementwise, freshly allocated, same shape/dtype as self_k)
"""

import torch
import triton
import triton.language as tl

_MAX_DIMS = 8
_MAX_FUSED = 8


@triton.jit
def _mul_store(in_ptr, out_ptr, offs, mask, s_ptr):
    # zero-reuse streaming rows: keep the lines out of the cache
    x = tl.load(in_ptr + offs, mask=mask, other=0, eviction_policy="evict_first")
    s = tl.load(s_ptr)
    OT = out_ptr.dtype.element_ty
    if OT == tl.float16 or OT == tl.bfloat16:
        y = (x.to(tl.float32) * s.to(tl.float32)).to(OT)
    else:
        y = x * s.to(OT)
    tl.store(out_ptr + offs, y, mask=mask, eviction_policy="evict_first")


@triton.jit
def _mul_flat(in_ptr, out_ptr, n, s_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    _mul_store(in_ptr, out_ptr, offs, offs < n, s_ptr)


@triton.jit
def _mul_fused(
    i0,
    o0,
    n0,
    i1,
    o1,
    n1,
    i2,
    o2,
    n2,
    i3,
    o3,
    n3,
    i4,
    o4,
    n4,
    i5,
    o5,
    n5,
    i6,
    o6,
    n6,
    i7,
    o7,
    n7,
    s_ptr,
    NT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    bid = tl.program_id(0)
    tid = tl.program_id(1)
    offs = bid * BLOCK + tl.arange(0, BLOCK)
    if tid == 0:
        _mul_store(i0, o0, offs, offs < n0, s_ptr)
    if NT > 1:
        if tid == 1:
            _mul_store(i1, o1, offs, offs < n1, s_ptr)
    if NT > 2:
        if tid == 2:
            _mul_store(i2, o2, offs, offs < n2, s_ptr)
    if NT > 3:
        if tid == 3:
            _mul_store(i3, o3, offs, offs < n3, s_ptr)
    if NT > 4:
        if tid == 4:
            _mul_store(i4, o4, offs, offs < n4, s_ptr)
    if NT > 5:
        if tid == 5:
            _mul_store(i5, o5, offs, offs < n5, s_ptr)
    if NT > 6:
        if tid == 6:
            _mul_store(i6, o6, offs, offs < n6, s_ptr)
    if NT > 7:
        if tid == 7:
            _mul_store(i7, o7, offs, offs < n7, s_ptr)


@triton.jit
def _mul_fused4(i0, o0, i1, o1, i2, o2, i3, o3, n, s_ptr, BLOCK: tl.constexpr):
    # Same work as _mul_fused for exactly four equal-size tensors, but with one
    # shared extent and no unused slots, so the host launch binds 11 arguments
    # instead of 27 on the launch-bound shapes.
    bid = tl.program_id(0)
    tid = tl.program_id(1)
    offs = bid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    if tid == 0:
        _mul_store(i0, o0, offs, mask, s_ptr)
    if tid == 1:
        _mul_store(i1, o1, offs, mask, s_ptr)
    if tid == 2:
        _mul_store(i2, o2, offs, mask, s_ptr)
    if tid == 3:
        _mul_store(i3, o3, offs, mask, s_ptr)


@triton.jit
def _mul_strided(
    in_ptr,
    out_ptr,
    n,
    s_ptr,
    z0,
    z1,
    z2,
    z3,
    z4,
    z5,
    z6,
    z7,
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
    t6,
    t7,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n
    rem = idx
    off = tl.zeros([BLOCK], dtype=tl.int64)
    c = rem % z0
    rem = rem // z0
    off += c.to(tl.int64) * t0
    c = rem % z1
    rem = rem // z1
    off += c.to(tl.int64) * t1
    c = rem % z2
    rem = rem // z2
    off += c.to(tl.int64) * t2
    c = rem % z3
    rem = rem // z3
    off += c.to(tl.int64) * t3
    c = rem % z4
    rem = rem // z4
    off += c.to(tl.int64) * t4
    c = rem % z5
    rem = rem // z5
    off += c.to(tl.int64) * t5
    c = rem % z6
    rem = rem // z6
    off += c.to(tl.int64) * t6
    c = rem % z7
    rem = rem // z7
    off += c.to(tl.int64) * t7
    x = tl.load(in_ptr + off, mask=mask, other=0)
    s = tl.load(s_ptr)
    OT = out_ptr.dtype.element_ty
    if OT == tl.float16 or OT == tl.bfloat16:
        y = (x.to(tl.float32) * s.to(tl.float32)).to(OT)
    else:
        y = x * s.to(OT)
    # out is freshly allocated contiguous, so it is indexed by the flat index
    tl.store(out_ptr + idx, y, mask=mask)


def _block_for(n, itemsize=4):
    # CTAs are 4 warps x 64 threads = 256 threads, so elements per thread is
    # BLOCK/256. Reaching a 16-byte per-thread access therefore needs
    # 16/itemsize elements per thread on the large streaming rows.
    if n >= (1 << 20):
        if itemsize == 1:
            return 4096, 4
        if itemsize == 2:
            return 2048, 4
    if n >= 1024:
        return 1024, 4
    b = 32
    while b < n:
        b *= 2
    if b >= 1024:
        return 1024, 4
    if b >= 256:
        return b, 2
    return b, 1


def _collapse(sizes, strides):
    """Merge adjacent dims that are contiguous in memory (natural order)."""
    zs = [sizes[0]]
    ts = [strides[0]]
    for i in range(1, len(sizes)):
        if ts[-1] == strides[i] * sizes[i]:
            zs[-1] *= sizes[i]
        else:
            zs.append(sizes[i])
            ts.append(strides[i])
    return zs, ts


def _scalar_arg(other, device):
    if isinstance(other, torch.Tensor):
        return other
    return torch.tensor(float(other), dtype=torch.float64, device=device)


def foreach_mul_tensor(self, other):
    tensors = self
    if isinstance(other, torch.Tensor) and other.numel() != 1:
        raise RuntimeError(
            "scalar tensor expected to be 0 dim but it has "
            f"{other.dim()} dimensions and {other.numel()} elements."
        )

    outs = []
    groups = {}
    strided = []
    for t in tensors:
        n = t.numel()
        if t.is_contiguous():
            out = torch.empty_like(t)
            outs.append(out)
            if n:
                key = (t.dtype, t.device)
                g = groups.get(key)
                if g is None:
                    groups[key] = [(t, out, n)]
                else:
                    g.append((t, out, n))
        else:
            out = torch.empty(t.shape, dtype=t.dtype, device=t.device)
            outs.append(out)
            if n:
                strided.append((t, out, n))

    for key, g in groups.items():
        sptr = _scalar_arg(other, key[1])
        isz = key[0].itemsize
        ng = len(g)
        if ng == 1 or ng > _MAX_FUSED:
            for t, out, n in g:
                block, warps = _block_for(n, isz)
                _mul_flat[(triton.cdiv(n, block),)](
                    t, out, n, sptr, BLOCK=block, num_warps=warps
                )
        else:
            maxn = max(item[2] for item in g)
            block, warps = _block_for(maxn, isz)
            if (
                ng == 4
                and g[0][2] == maxn
                and g[1][2] == maxn
                and g[2][2] == maxn
                and g[3][2] == maxn
            ):
                (t0, o0, _n), (t1, o1, _n), (t2, o2, _n), (t3, o3, _n) = g
                _mul_fused4[(triton.cdiv(maxn, block), 4)](
                    t0,
                    o0,
                    t1,
                    o1,
                    t2,
                    o2,
                    t3,
                    o3,
                    maxn,
                    sptr,
                    BLOCK=block,
                    num_warps=warps,
                )
                continue
            args = []
            for i in range(_MAX_FUSED):
                if i < ng:
                    t, out, n = g[i]
                else:
                    t, out, _n = g[0]
                    n = 0
                args.append(t)
                args.append(out)
                args.append(n)
            _mul_fused[(triton.cdiv(maxn, block), ng)](
                *args, sptr, NT=ng, BLOCK=block, num_warps=warps
            )

    for t, out, n in strided:
        sizes, strides = _collapse(list(t.shape), list(t.stride()))
        if len(sizes) > _MAX_DIMS:
            raise RuntimeError("foreach_mul: unsupported tensor rank")
        # kernel walks dims innermost-first
        sizes.reverse()
        strides.reverse()
        while len(sizes) < _MAX_DIMS:
            sizes.append(1)
            strides.append(0)
        sptr = _scalar_arg(other, t.device)
        block, warps = _block_for(n, t.dtype.itemsize)
        _mul_strided[(triton.cdiv(n, block),)](
            t,
            out,
            n,
            sptr,
            sizes[0],
            sizes[1],
            sizes[2],
            sizes[3],
            sizes[4],
            sizes[5],
            sizes[6],
            sizes[7],
            strides[0],
            strides[1],
            strides[2],
            strides[3],
            strides[4],
            strides[5],
            strides[6],
            strides[7],
            BLOCK=block,
            num_warps=warps,
        )
    return outs
