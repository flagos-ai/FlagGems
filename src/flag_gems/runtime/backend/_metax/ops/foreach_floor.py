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

"""foreach_floor: elementwise floor over a ragged list of CUDA tensors.

Reference semantics (measured against ATen torch._foreach_floor on the target,
torch 2.10.0+metax):
  * floating dtypes (fp16/bf16/fp32/fp64): IEEE floor, same dtype, new storage
  * integral dtypes (int8/16/32/64, uint8...): identity, same dtype
  * bool: NotImplementedError 'floor_cuda' not implemented for 'Bool'
  * complex: NotImplementedError 'foreach_unary_op_cuda' not implemented
  * empty list: RuntimeError "Tensor list must have at least one tensor."
  * output layout mirrors the input when the input is dense (empty_like),
    and falls back to contiguous for expanding/narrowed inputs.
  * return container is a tuple.

Dispatch:
  * one dense tensor -> single flat kernel (streaming fast path, unchanged)
  * many dense tensors of the same dtype -> one fused launch per group of
    <= 16 tensors: pointers are kernel arguments (like ATen's
    multi_tensor_apply), so no host->device metadata is materialized.
  * non-dense tensors -> rank-specialized strided kernel, one launch each.
"""

import torch
import triton
import triton.language as tl

_MAX_RANK = 10
_SLOTS = 16

# (BLOCK, num_warps) keyed by element size in bytes. Small work keeps the
# narrow CTA (a 64x64 tensor stays at 2 CTAs); once a group carries enough
# elements, doubling BLOCK halves the CTA count, which measured faster on the
# many-large-tensor workloads (round 6) without the single-CTA penalty that
# the same change caused on the latency-floor cases.
_CFG = {1: (4096, 4), 2: (2048, 4), 4: (1024, 4), 8: (1024, 4)}
_CFG_LARGE = {1: (16384, 8), 2: (8192, 8), 4: (4096, 8), 8: (4096, 8)}
_DEFAULT_CFG = (1024, 4)
_LARGE_NUMEL = 1 << 22

_CURRENT_DEVICE = None


@triton.jit
def _apply(x, IS_FLOAT: tl.constexpr, IS_16: tl.constexpr):
    if IS_FLOAT:
        if IS_16:
            y = tl.math.floor(x.to(tl.float32)).to(x.dtype)
        else:
            y = tl.math.floor(x)
    else:
        y = x
    return y


@triton.jit
def _flat_kernel(
    x_ptr,
    o_ptr,
    numel,
    BLOCK: tl.constexpr,
    IS_FLOAT: tl.constexpr,
    IS_16: tl.constexpr,
    IDX64: tl.constexpr,
):
    pid = tl.program_id(0)
    if IDX64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        n = numel.to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        n = numel
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    tl.store(o_ptr + offs, _apply(x, IS_FLOAT, IS_16), mask=mask)


@triton.jit
def _slot(
    p,
    o,
    n,
    base,
    start,
    BLOCK: tl.constexpr,
    IS_FLOAT: tl.constexpr,
    IS_16: tl.constexpr,
):
    offs = (base - start) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(p + offs, mask=m, other=0)
    tl.store(o + offs, _apply(x, IS_FLOAT, IS_16), mask=m)


@triton.jit
def _fused_kernel(
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
    o0,
    o1,
    o2,
    o3,
    o4,
    o5,
    o6,
    o7,
    o8,
    o9,
    o10,
    o11,
    o12,
    o13,
    o14,
    o15,
    n0,
    n1,
    n2,
    n3,
    n4,
    n5,
    n6,
    n7,
    n8,
    n9,
    n10,
    n11,
    n12,
    n13,
    n14,
    n15,
    c0,
    c1,
    c2,
    c3,
    c4,
    c5,
    c6,
    c7,
    c8,
    c9,
    c10,
    c11,
    c12,
    c13,
    c14,
    c15,
    BLOCK: tl.constexpr,
    IS_FLOAT: tl.constexpr,
    IS_16: tl.constexpr,
    IDX64: tl.constexpr,
):
    pid = tl.program_id(0)
    if IDX64:
        base = pid.to(tl.int64)
    else:
        base = pid
    if pid < c0:
        _slot(p0, o0, n0, base, 0, BLOCK, IS_FLOAT, IS_16)
    elif pid < c1:
        _slot(p1, o1, n1, base, c0, BLOCK, IS_FLOAT, IS_16)
    elif pid < c2:
        _slot(p2, o2, n2, base, c1, BLOCK, IS_FLOAT, IS_16)
    elif pid < c3:
        _slot(p3, o3, n3, base, c2, BLOCK, IS_FLOAT, IS_16)
    elif pid < c4:
        _slot(p4, o4, n4, base, c3, BLOCK, IS_FLOAT, IS_16)
    elif pid < c5:
        _slot(p5, o5, n5, base, c4, BLOCK, IS_FLOAT, IS_16)
    elif pid < c6:
        _slot(p6, o6, n6, base, c5, BLOCK, IS_FLOAT, IS_16)
    elif pid < c7:
        _slot(p7, o7, n7, base, c6, BLOCK, IS_FLOAT, IS_16)
    elif pid < c8:
        _slot(p8, o8, n8, base, c7, BLOCK, IS_FLOAT, IS_16)
    elif pid < c9:
        _slot(p9, o9, n9, base, c8, BLOCK, IS_FLOAT, IS_16)
    elif pid < c10:
        _slot(p10, o10, n10, base, c9, BLOCK, IS_FLOAT, IS_16)
    elif pid < c11:
        _slot(p11, o11, n11, base, c10, BLOCK, IS_FLOAT, IS_16)
    elif pid < c12:
        _slot(p12, o12, n12, base, c11, BLOCK, IS_FLOAT, IS_16)
    elif pid < c13:
        _slot(p13, o13, n13, base, c12, BLOCK, IS_FLOAT, IS_16)
    elif pid < c14:
        _slot(p14, o14, n14, base, c13, BLOCK, IS_FLOAT, IS_16)
    else:
        _slot(p15, o15, n15, base, c14, BLOCK, IS_FLOAT, IS_16)


@triton.jit
def _strided_kernel(
    x_ptr,
    o_ptr,
    numel,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    d8,
    d9,
    si0,
    si1,
    si2,
    si3,
    si4,
    si5,
    si6,
    si7,
    si8,
    si9,
    so0,
    so1,
    so2,
    so3,
    so4,
    so5,
    so6,
    so7,
    so8,
    so9,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
    IS_FLOAT: tl.constexpr,
    IS_16: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < numel
    rem = idx
    oi = tl.zeros([BLOCK], dtype=tl.int64)
    oo = tl.zeros([BLOCK], dtype=tl.int64)
    if RANK >= 1:
        c = rem % d0
        rem = rem // d0
        oi += c * si0
        oo += c * so0
    if RANK >= 2:
        c = rem % d1
        rem = rem // d1
        oi += c * si1
        oo += c * so1
    if RANK >= 3:
        c = rem % d2
        rem = rem // d2
        oi += c * si2
        oo += c * so2
    if RANK >= 4:
        c = rem % d3
        rem = rem // d3
        oi += c * si3
        oo += c * so3
    if RANK >= 5:
        c = rem % d4
        rem = rem // d4
        oi += c * si4
        oo += c * so4
    if RANK >= 6:
        c = rem % d5
        rem = rem // d5
        oi += c * si5
        oo += c * so5
    if RANK >= 7:
        c = rem % d6
        rem = rem // d6
        oi += c * si6
        oo += c * so6
    if RANK >= 8:
        c = rem % d7
        rem = rem // d7
        oi += c * si7
        oo += c * so7
    if RANK >= 9:
        c = rem % d8
        rem = rem // d8
        oi += c * si8
        oo += c * so8
    if RANK >= 10:
        c = rem % d9
        rem = rem // d9
        oi += c * si9
        oo += c * so9
    x = tl.load(x_ptr + oi, mask=mask, other=0)
    tl.store(o_ptr + oo, _apply(x, IS_FLOAT, IS_16), mask=mask)


def _coalesce(shape, si, so):
    """Merge adjacent dimensions when both input and output allow it."""
    dims = list(shape)
    sin = list(si)
    sout = list(so)
    i = len(dims) - 1
    while i > 0:
        if dims[i - 1] == 1 or (
            sin[i - 1] == sin[i] * dims[i] and sout[i - 1] == sout[i] * dims[i]
        ):
            dims[i - 1] = dims[i - 1] * dims[i]
            sin[i - 1] = sin[i]
            sout[i - 1] = sout[i]
            del dims[i]
            del sin[i]
            del sout[i]
        i -= 1
    return dims, sin, sout


def _cfg_for(esize, total_numel):
    if total_numel >= _LARGE_NUMEL:
        return _CFG_LARGE.get(esize, _DEFAULT_CFG)
    return _CFG.get(esize, _DEFAULT_CFG)


def _launch_flat(x, o, n, is_float, is_16, cfg):
    blk, warps = cfg
    grid = ((n + blk - 1) // blk,)
    _flat_kernel[grid](x, o, n, blk, is_float, is_16, n >= 2**31, num_warps=warps)


def _launch_fused(chunk, cfg, is_float, is_16):
    """chunk: list of (in_tensor, out_tensor, numel), 1 < len <= _SLOTS."""
    blk, warps = cfg
    k = len(chunk)
    pad = _SLOTS - k
    ps = [c[0] for c in chunk]
    ops = [c[1] for c in chunk]
    ns = [c[2] for c in chunk]
    # Unused slots repeat the first real entry. Their branch condition is
    # `pid < total`, already satisfied by an earlier real branch, so the
    # padded code is unreachable.
    ps.extend([ps[0]] * pad)
    ops.extend([ops[0]] * pad)
    ns.extend([ns[0]] * pad)
    bounds = []
    total = 0
    for i in range(k):
        total += (ns[i] + blk - 1) // blk
        bounds.append(total)
    bounds.extend([total] * pad)
    args = ps + ops + ns + bounds
    args.extend([blk, is_float, is_16, max(ns[:k]) >= 2**31])
    _fused_kernel[(total,)](*args, num_warps=warps)


def _launch_strided(x, o, n, is_float, is_16, cfg):
    blk, warps = cfg
    shape = tuple(x.shape)
    rank = x.dim()
    si = tuple(x.stride())
    so = tuple(o.stride())
    if rank > _MAX_RANK:
        shape, si, so = _coalesce(shape, si, so)
        rank = len(shape)
    if rank > _MAX_RANK:
        raise RuntimeError("foreach_floor: unsupported tensor rank %d" % rank)
    dims = [1] * _MAX_RANK
    sin = [0] * _MAX_RANK
    sout = [0] * _MAX_RANK
    # last dimension first (row-major delinearization order)
    for k in range(rank):
        j = rank - 1 - k
        dims[k] = shape[j]
        sin[k] = si[j]
        sout[k] = so[j]
    grid = ((n + blk - 1) // blk,)
    args = [x, o, n]
    args.extend(dims)
    args.extend(sin)
    args.extend(sout)
    args.extend([rank, blk, is_float, is_16])
    _strided_kernel[grid](*args, num_warps=warps)


def foreach_floor(self):
    global _CURRENT_DEVICE
    if isinstance(self, torch.Tensor):
        tensors = (self,)
    else:
        tensors = self
    if len(tensors) == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")

    first = tensors[0]
    ctx = None
    d = first.device
    if d.index is not None:
        if _CURRENT_DEVICE is None:
            _CURRENT_DEVICE = torch.cuda.current_device()
        if d.index != _CURRENT_DEVICE:
            ctx = torch.cuda.device(d.index)
            ctx.__enter__()

    outs = []
    groups = {}
    strided = []
    try:
        for t in tensors:
            dt = t.dtype
            if dt == torch.bool:
                raise NotImplementedError("\"floor_cuda\" not implemented for 'Bool'")
            if dt.is_complex:
                raise NotImplementedError(
                    "\"foreach_unary_op_cuda\" not implemented for 'ComplexFloat'"
                )
            o = torch.empty_like(t)
            outs.append(o)
            n = t.numel()
            if n == 0:
                continue
            is_float = dt.is_floating_point
            is_16 = dt == torch.float16 or dt == torch.bfloat16
            if n == 1 or t.is_contiguous():
                bucket = groups.get(dt)
                if bucket is None:
                    groups[dt] = bucket = []
                bucket.append((t, o, n))
            else:
                strided.append((t, o, n, is_float, is_16))

        for dt, bucket in groups.items():
            is_float = dt.is_floating_point
            is_16 = dt == torch.float16 or dt == torch.bfloat16
            esize = dt.itemsize
            if len(bucket) == 1:
                cfg = _cfg_for(esize, bucket[0][2])
                _launch_flat(
                    bucket[0][0],
                    bucket[0][1],
                    bucket[0][2],
                    is_float,
                    is_16,
                    cfg,
                )
            else:
                for i in range(0, len(bucket), _SLOTS):
                    chunk = bucket[i : i + _SLOTS]
                    cfg = _cfg_for(esize, sum(c[2] for c in chunk))
                    _launch_fused(chunk, cfg, is_float, is_16)

        for t, o, n, is_float, is_16 in strided:
            _launch_strided(
                t,
                o,
                n,
                is_float,
                is_16,
                _CFG.get(t.element_size(), _DEFAULT_CFG),
            )
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)
    return tuple(outs)
