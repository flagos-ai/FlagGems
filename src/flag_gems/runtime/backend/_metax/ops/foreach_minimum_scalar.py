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

import struct

import torch
import triton
import triton.language as tl

_BLOCK = 1024
_NUM_WARPS = 4
_FUSE = 4
_MAX_RANK = 8
_I32_LIMIT = 0x7FFFFFFF

_PACK_E = struct.Struct("<e").pack
_UNPACK_E = struct.Struct("<e").unpack
_PACK_F = struct.Struct("<f").pack
_UNPACK_F = struct.Struct("<f").unpack
_PACK_I = struct.Struct("<I").pack
_UNPACK_I = struct.Struct("<I").unpack


def _bf16_value(x):
    """Round a float to bfloat16 the way torch's scalar conversion does."""
    bits = _UNPACK_I(_PACK_F(x))[0]
    sign = bits & 0x80000000
    mag = bits & 0x7FFFFFFF
    low = mag & 0xFFFF
    q = mag >> 16
    if low > 0x8000 or (low == 0x8000 and (q & 1)):
        q += 1
    return _UNPACK_F(_PACK_I(sign | (q << 16)))[0]


def _scalar_for(dtype, scalar):
    if dtype is torch.float32:
        return float(scalar)
    if dtype is torch.float16:
        try:
            return _UNPACK_E(_PACK_E(float(scalar)))[0]
        except OverflowError:
            return float("inf") if scalar > 0 else float("-inf")
    if dtype is torch.bfloat16:
        return _bf16_value(float(scalar))
    if dtype.is_floating_point:
        return float(torch.tensor(scalar, dtype=dtype))
    return int(scalar)


@triton.jit
def _min_apply(in_ptr, out_ptr, chunk, n_elem, s, BLOCK: tl.constexpr):
    offs = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    x = tl.load(in_ptr + offs, mask=mask, other=0.0)
    y = tl.minimum(x, s.to(x.dtype), propagate_nan=tl.PropagateNan.ALL)
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def _min_scalar_flat(in_ptr, out_ptr, n_elem, s, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n_elem
    x = tl.load(in_ptr + offs, mask=mask, other=0.0)
    y = tl.minimum(x, s.to(x.dtype), propagate_nan=tl.PropagateNan.ALL)
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def _min_scalar_fused4(
    i0,
    i1,
    i2,
    i3,
    o0,
    o1,
    o2,
    o3,
    s,
    N0: tl.constexpr,
    N1: tl.constexpr,
    N2: tl.constexpr,
    N3: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off = tl.program_id(0)
    c0 = tl.cdiv(N0, BLOCK)
    if off < c0:
        _min_apply(i0, o0, off, N0, s, BLOCK)
    else:
        off1 = off - c0
        c1 = tl.cdiv(N1, BLOCK)
        if off1 < c1:
            _min_apply(i1, o1, off1, N1, s, BLOCK)
        else:
            off2 = off1 - c1
            c2 = tl.cdiv(N2, BLOCK)
            if off2 < c2:
                _min_apply(i2, o2, off2, N2, s, BLOCK)
            else:
                off3 = off2 - c2
                _min_apply(i3, o3, off3, N3, s, BLOCK)


@triton.jit
def _min_scalar_strided(
    in_ptr,
    out_ptr,
    n_elem,
    s,
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
    u0,
    u1,
    u2,
    u3,
    u4,
    u5,
    u6,
    u7,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n_elem
    i = offs
    off_in = tl.zeros([BLOCK], tl.int64)
    off_out = tl.zeros([BLOCK], tl.int64)
    if RANK > 0:
        c = i % d0
        i = i // d0
        off_in += c * t0
        off_out += c * u0
    if RANK > 1:
        c = i % d1
        i = i // d1
        off_in += c * t1
        off_out += c * u1
    if RANK > 2:
        c = i % d2
        i = i // d2
        off_in += c * t2
        off_out += c * u2
    if RANK > 3:
        c = i % d3
        i = i // d3
        off_in += c * t3
        off_out += c * u3
    if RANK > 4:
        c = i % d4
        i = i // d4
        off_in += c * t4
        off_out += c * u4
    if RANK > 5:
        c = i % d5
        i = i // d5
        off_in += c * t5
        off_out += c * u5
    if RANK > 6:
        c = i % d6
        i = i // d6
        off_in += c * t6
        off_out += c * u6
    if RANK > 7:
        c = i % d7
        i = i // d7
        off_in += c * t7
        off_out += c * u7
    x = tl.load(in_ptr + off_in, mask=mask, other=0.0)
    y = tl.minimum(x, s.to(x.dtype), propagate_nan=tl.PropagateNan.ALL)
    tl.store(out_ptr + off_out, y, mask=mask)


def _strided_triples(x, out):
    """(size, in_stride, out_stride) per logical dim, dropping size-1 dims."""
    return [
        (int(s), int(ti), int(to))
        for s, ti, to in zip(x.shape, x.stride(), out.stride())
        if int(s) != 1
    ]


def _launch_strided(x, out, n, s):
    triples = _strided_triples(x, out)
    if len(triples) > _MAX_RANK:
        # Extremely unusual layout: peel the leading dimension with views.
        for idx in range(x.shape[0]):
            xi = x.select(0, idx)
            oi = out.select(0, idx)
            ni = xi.numel()
            if ni:
                _launch_strided(xi, oi, ni, s)
        return

    rank = max(len(triples), 1)
    shape_p = [1] * _MAX_RANK
    stride_p = [0] * _MAX_RANK
    ostride_p = [0] * _MAX_RANK
    for k, (ds, dt_, du_) in enumerate(reversed(triples)):
        shape_p[k] = ds
        stride_p[k] = dt_
        ostride_p[k] = du_
    grid = ((n + _BLOCK - 1) // _BLOCK,)
    _min_scalar_strided[grid](
        x,
        out,
        n,
        s,
        *shape_p,
        *stride_p,
        *ostride_p,
        RANK=rank,
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
    )


def foreach_minimum_scalar(self, scalar):
    tensors = self
    count = len(tensors)
    if count == 0:
        return []
    if isinstance(scalar, torch.Tensor):
        scalar = scalar.item()

    outs = [torch.empty_like(t) for t in tensors]

    dt0 = tensors[0].dtype
    fast = count <= _FUSE
    if fast:
        for t in tensors:
            if t.dtype is not dt0 or not t.is_contiguous():
                fast = False
                break
    if fast:
        nums = [t.numel() for t in tensors]
        chunks = 0
        big = False
        for n in nums:
            if n > _I32_LIMIT:
                big = True
                break
            chunks += (n + _BLOCK - 1) // _BLOCK
        if not big:
            if chunks == 0:
                return outs
            s = _scalar_for(dt0, scalar)
            if count == _FUSE:
                it = tensors
                ot = outs
            else:
                it = list(tensors) + [tensors[0]] * (_FUSE - count)
                ot = outs + [outs[0]] * (_FUSE - count)
                nums = nums + [0] * (_FUSE - count)
            _min_scalar_fused4[(chunks,)](
                it[0],
                it[1],
                it[2],
                it[3],
                ot[0],
                ot[1],
                ot[2],
                ot[3],
                s,
                N0=nums[0],
                N1=nums[1],
                N2=nums[2],
                N3=nums[3],
                BLOCK=_BLOCK,
                num_warps=_NUM_WARPS,
            )
            return outs

    for idx in range(count):
        x = tensors[idx]
        out = outs[idx]
        n = x.numel()
        if n == 0:
            continue
        s = _scalar_for(x.dtype, scalar)
        if x.is_contiguous():
            grid = ((n + _BLOCK - 1) // _BLOCK,)
            _min_scalar_flat[grid](
                x,
                out,
                n,
                s,
                BLOCK=_BLOCK,
                num_warps=_NUM_WARPS,
            )
        else:
            _launch_strided(x, out, n, s)
    return outs
