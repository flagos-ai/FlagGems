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

"""Triton implementation of ``aten::_foreach_maximum`` (List overload).

Out-of-place elementwise maximum over paired tensor lists.  Every output is a
freshly allocated tensor (never a view of ``self``), NaN propagation follows
``torch.maximum`` semantics, and non-contiguous / broadcast pairs are read
through explicit strides (stride 0 gives free broadcasting, no data copy).

Contiguous pairs that share dtype and shape are fused: up to four pairs go into
a single launch (one program handles one tile of exactly one pair).  For small
groups the outputs are carved out of a single batched allocation, which removes
three of the four per-call allocator round trips that dominate tiny lists.
"""

import torch
import triton
import triton.language as tl

_MAX_DIMS = 8
_GROUP = 4
_BLOCK = 1024
_SMALL_BLOCK = 256
_WARPS = 4
_BATCH_BYTES = 1 << 20


@triton.jit
def _vmax(a, b):
    # torch.maximum NaN propagation: NaN in a wins, then NaN in b, else max.
    return tl.where(a != a, a, tl.where(b != b, b, tl.maximum(a, b)))


@triton.jit
def _fmax_contig(a_ptr, b_ptr, o_ptr, n, BLOCK: tl.constexpr, INT64: tl.constexpr):
    pid = tl.program_id(0)
    if INT64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    tl.store(o_ptr + offs, _vmax(a, b), mask=mask)


@triton.jit
def _fmax_contig4(
    a0,
    b0,
    o0,
    a1,
    b1,
    o1,
    a2,
    b2,
    o2,
    a3,
    b3,
    o3,
    n,
    tiles,
    BLOCK: tl.constexpr,
    INT64: tl.constexpr,
):
    pid = tl.program_id(0)
    pair = pid // tiles
    tile = pid - pair * tiles
    if INT64:
        offs = tile.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = tile * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    if pair == 0:
        a = tl.load(a0 + offs, mask=mask)
        b = tl.load(b0 + offs, mask=mask)
        tl.store(o0 + offs, _vmax(a, b), mask=mask)
    elif pair == 1:
        a = tl.load(a1 + offs, mask=mask)
        b = tl.load(b1 + offs, mask=mask)
        tl.store(o1 + offs, _vmax(a, b), mask=mask)
    elif pair == 2:
        a = tl.load(a2 + offs, mask=mask)
        b = tl.load(b2 + offs, mask=mask)
        tl.store(o2 + offs, _vmax(a, b), mask=mask)
    else:
        a = tl.load(a3 + offs, mask=mask)
        b = tl.load(b3 + offs, mask=mask)
        tl.store(o3 + offs, _vmax(a, b), mask=mask)


@triton.jit
def _fmax_strided(
    a_ptr,
    b_ptr,
    o_ptr,
    n,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    u0,
    u1,
    u2,
    u3,
    u4,
    u5,
    u6,
    u7,
    v0,
    v1,
    v2,
    v3,
    v4,
    v5,
    v6,
    v7,
    w0,
    w1,
    w2,
    w3,
    w4,
    w5,
    w6,
    w7,
    BLOCK: tl.constexpr,
    INT64: tl.constexpr,
):
    pid = tl.program_id(0)
    if INT64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        off_a = tl.zeros([BLOCK], dtype=tl.int64)
        off_b = tl.zeros([BLOCK], dtype=tl.int64)
        off_o = tl.zeros([BLOCK], dtype=tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        off_a = tl.zeros([BLOCK], dtype=tl.int32)
        off_b = tl.zeros([BLOCK], dtype=tl.int32)
        off_o = tl.zeros([BLOCK], dtype=tl.int32)
    mask = offs < n

    # Row-major linear index -> per-tensor element offsets.  Dimensions are
    # unrolled innermost-first; unused dimensions are padded with shape 1 and
    # stride 0, and stride 0 also implements free broadcasting.
    rem = offs

    c = rem % s7
    rem = rem // s7
    off_a += c * u7
    off_b += c * v7
    off_o += c * w7

    c = rem % s6
    rem = rem // s6
    off_a += c * u6
    off_b += c * v6
    off_o += c * w6

    c = rem % s5
    rem = rem // s5
    off_a += c * u5
    off_b += c * v5
    off_o += c * w5

    c = rem % s4
    rem = rem // s4
    off_a += c * u4
    off_b += c * v4
    off_o += c * w4

    c = rem % s3
    rem = rem // s3
    off_a += c * u3
    off_b += c * v3
    off_o += c * w3

    c = rem % s2
    rem = rem // s2
    off_a += c * u2
    off_b += c * v2
    off_o += c * w2

    c = rem % s1
    rem = rem // s1
    off_a += c * u1
    off_b += c * v1
    off_o += c * w1

    c = rem % s0
    off_a += c * u0
    off_b += c * v0
    off_o += c * w0

    a = tl.load(a_ptr + off_a, mask=mask)
    b = tl.load(b_ptr + off_b, mask=mask)
    tl.store(o_ptr + off_o, _vmax(a, b), mask=mask)


def _pad8(vals, fill):
    vals = list(vals)
    return vals + [fill] * (_MAX_DIMS - len(vals))


def _launch_strided(a, b, out, n):
    grid = (triton.cdiv(n, _BLOCK),)
    _fmax_strided[grid](
        a,
        b,
        out,
        n,
        *_pad8(out.shape, 1),
        *_pad8(a.stride(), 0),
        *_pad8(b.stride(), 0),
        *_pad8(out.stride(), 0),
        _BLOCK,
        n > 2147483647,
        num_warps=_WARPS,
    )


def _slow_pair(a, b):
    """Broadcast / mixed-dtype pair: allocate, then launch the strided kernel."""
    dtype = a.dtype if a.dtype is b.dtype else torch.promote_types(a.dtype, b.dtype)
    if a.shape != b.shape:
        a, b = torch.broadcast_tensors(a, b)
        out = torch.empty(a.shape, dtype=dtype, device=a.device)
    elif a.dtype is dtype and b.dtype is dtype:
        out = torch.empty_like(a)
    else:
        out = torch.empty(a.shape, dtype=dtype, device=a.device)
    n = out.numel()
    if n == 0:
        return out
    if out.dim() > _MAX_DIMS:
        raise RuntimeError("_foreach_maximum supports tensors up to 8 dims")
    _launch_strided(a, b, out, n)
    return out


def foreach_maximum_list(self, other):
    lhs = list(self)
    rhs = list(other)
    npairs = len(lhs)
    if npairs != len(rhs):
        raise RuntimeError(
            "Tensor list lengths must be the same, but got %d and %d"
            % (npairs, len(rhs))
        )
    if npairs == 0:
        return []

    outs = [None] * npairs
    groups = None
    for i in range(npairs):
        a = lhs[i]
        b = rhs[i]
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            raise TypeError("_foreach_maximum expects lists of tensors")
        if a.dtype is b.dtype and a.shape == b.shape:
            n = a.numel()
            if n == 0:
                outs[i] = torch.empty_like(a)
            elif a.is_contiguous() and b.is_contiguous():
                if groups is None:
                    groups = {}
                key = (a.dtype, a.shape)
                slot = groups.get(key)
                if slot is None:
                    groups[key] = [i]
                else:
                    slot.append(i)
            else:
                out = torch.empty_like(a)
                outs[i] = out
                _launch_strided(a, b, out, n)
        else:
            outs[i] = _slow_pair(a, b)

    if groups:
        for (dtype, shape), idxs in groups.items():
            n = shape.numel()
            flat = n * lhs[idxs[0]].element_size()
            for base in range(0, len(idxs), _GROUP):
                chunk = idxs[base : base + _GROUP]
                k = len(chunk)
                small = k > 1 and flat * k <= _BATCH_BYTES and flat % 16 == 0
                if small:
                    views = torch.empty(
                        (k,) + tuple(shape), dtype=dtype, device=lhs[chunk[0]].device
                    ).unbind(0)
                else:
                    views = [torch.empty_like(lhs[j]) for j in chunk]
                for pos, j in enumerate(chunk):
                    outs[j] = views[pos]
                block = _SMALL_BLOCK if flat * k <= _BATCH_BYTES else _BLOCK
                tiles = (n + block - 1) // block
                if k == 1:
                    j = chunk[0]
                    _fmax_contig[(tiles,)](
                        lhs[j],
                        rhs[j],
                        outs[j],
                        n,
                        block,
                        n > 2147483647,
                        num_warps=_WARPS,
                    )
                else:
                    args = []
                    for pos in range(_GROUP):
                        j = chunk[pos] if pos < k else chunk[k - 1]
                        args.append(lhs[j])
                        args.append(rhs[j])
                        args.append(outs[j])
                    _fmax_contig4[(k * tiles,)](
                        *args, n, tiles, block, n > 2147483647, num_warps=_WARPS
                    )
    return outs
