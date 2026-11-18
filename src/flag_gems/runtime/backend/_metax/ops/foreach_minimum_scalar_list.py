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

"""Triton implementation of ``aten::_foreach_minimum.ScalarList``.

For each list element i:
    out[i] = min_propagate_nan(self[i], scalars[i])

The scalar is converted to ``self[i].dtype`` exactly as ATen's
``Scalar::to<scalar_t>`` does, the output keeps ``self[i].dtype`` and is a
freshly allocated tensor (the reference never aliases its inputs).

Host cost matters for small list elements, so contiguous same-dtype elements
are batched into one 4-slot launch instead of one launch each.
"""

import struct

import torch
import triton
import triton.language as tl

try:  # cheap, exact host-side rounding to reduced precision dtypes
    import numpy as _np
except Exception:  # pragma: no cover - numpy ships with torch in practice
    _np = None

_BLOCK = 1024
_WARPS = 4
_SLOTS = 4
_MAX_RANK = 8
_I32_MAX = 2147483647
_F32_MAX = 3.4028234663852886e38


@triton.jit
def _apply(x_ptr, o_ptr, offs, n, s):
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    sv = s.to(x.dtype)
    # min_propagate_nan: NaN wins over every value.
    res = tl.where(x != x, x, tl.where(sv != sv, sv, tl.minimum(x, sv)))
    tl.store(o_ptr + offs, res, mask=mask)


@triton.jit
def _fmin_flat_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    s_hi,
    s_lo,
    USE_I64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if USE_I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    s = s_hi.to(x.dtype) + s_lo.to(x.dtype)
    res = tl.where(x != x, x, tl.where(s != s, s, tl.minimum(x, s)))
    tl.store(out_ptr + offs, res, mask=mask)


@triton.jit
def _fmin4_kernel(
    x0,
    x1,
    x2,
    x3,
    o0,
    o1,
    o2,
    o3,
    n0,
    n1,
    n2,
    n3,
    h0,
    h1,
    h2,
    h3,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    e0 = (n0 + BLOCK - 1) // BLOCK
    e1 = e0 + (n1 + BLOCK - 1) // BLOCK
    e2 = e1 + (n2 + BLOCK - 1) // BLOCK
    ar = tl.arange(0, BLOCK)
    if pid < e0:
        _apply(x0, o0, pid * BLOCK + ar, n0, h0)
    elif pid < e1:
        _apply(x1, o1, (pid - e0) * BLOCK + ar, n1, h1)
    elif pid < e2:
        _apply(x2, o2, (pid - e1) * BLOCK + ar, n2, h2)
    else:
        _apply(x3, o3, (pid - e2) * BLOCK + ar, n3, h3)


@triton.jit
def _fmin_strided_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    s_hi,
    s_lo,
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
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n_elements
    rem = offs
    off = tl.zeros([BLOCK], dtype=tl.int64)
    if RANK >= 8:
        off += (rem % s7) * t7
        rem = rem // s7
    if RANK >= 7:
        off += (rem % s6) * t6
        rem = rem // s6
    if RANK >= 6:
        off += (rem % s5) * t5
        rem = rem // s5
    if RANK >= 5:
        off += (rem % s4) * t4
        rem = rem // s4
    if RANK >= 4:
        off += (rem % s3) * t3
        rem = rem // s3
    if RANK >= 3:
        off += (rem % s2) * t2
        rem = rem // s2
    if RANK >= 2:
        off += (rem % s1) * t1
        rem = rem // s1
    if RANK >= 1:
        off += (rem % s0) * t0
    x = tl.load(x_ptr + off, mask=mask, other=0)
    s = s_hi.to(x.dtype) + s_lo.to(x.dtype)
    res = tl.where(x != x, x, tl.where(s != s, s, tl.minimum(x, s)))
    tl.store(out_ptr + offs, res, mask=mask)


def _round_f32(v):
    return struct.unpack("<f", struct.pack("<f", v))[0]


def _round_bf16(v):
    bits = struct.unpack("<I", struct.pack("<f", float(v)))[0]
    if (bits & 0x7F800000) != 0x7F800000:  # not inf / nan
        bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    else:
        bits &= 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _scalar_pair(v, dtype):
    """Return (hi, lo) such that hi + lo reconstructs Scalar::to<dtype>(v)."""
    if dtype.is_floating_point:
        if dtype == torch.float32:
            # Triton converts a python float kernel argument to fp32 with RN,
            # identical to ATen's Scalar::to<float>.
            return float(v), 0.0
        if dtype == torch.float64:
            d = float(v)
            if d != d:
                return float("nan"), 0.0
            hi = _round_f32(d)
            if hi == float("inf"):
                hi = _F32_MAX
            elif hi == float("-inf"):
                hi = -_F32_MAX
            return hi, _round_f32(d - hi)
        if dtype == torch.bfloat16:
            return _round_bf16(v), 0.0
        if _np is not None:
            try:
                return float(_np.float16(float(v))), 0.0
            except Exception:
                pass
        return float(torch.tensor(v, dtype=dtype).item()), 0.0
    try:
        return int(torch.tensor(v, dtype=dtype).item()), 0
    except Exception:
        return int(v), 0


def _geometry(t):
    """Collapse a tensor's shape/stride into the minimal strided description."""
    dims = []
    for sz, st in zip(t.shape, t.stride()):
        if sz == 1:
            continue
        if dims and dims[-1][1] == st * sz:
            dims[-1] = (dims[-1][0] * sz, st)
        else:
            dims.append((sz, st))
    return dims


def _min_scalar_one(t, s):
    out = torch.empty(t.shape, dtype=t.dtype, device=t.device)
    n = out.numel()
    if n == 0:
        return out

    hi, lo = _scalar_pair(s, t.dtype)

    if t.is_contiguous():
        grid = ((n + _BLOCK - 1) // _BLOCK,)
        _fmin_flat_kernel[grid](
            t,
            out,
            n,
            hi,
            lo,
            USE_I64=n > _I32_MAX,
            BLOCK=_BLOCK,
            num_warps=_WARPS,
        )
        return out

    dims = _geometry(t)
    if not dims or (len(dims) == 1 and dims[0][0] == n and dims[0][1] == 1):
        grid = ((n + _BLOCK - 1) // _BLOCK,)
        _fmin_flat_kernel[grid](
            t,
            out,
            n,
            hi,
            lo,
            USE_I64=n > _I32_MAX,
            BLOCK=_BLOCK,
            num_warps=_WARPS,
        )
        return out

    rank = len(dims)
    if rank > _MAX_RANK:
        raise RuntimeError("unsupported tensor rank for strided foreach minimum")

    shp = [1] * _MAX_RANK
    std = [0] * _MAX_RANK
    for i, (sz, st) in enumerate(dims):
        shp[i] = sz
        std[i] = st

    grid = ((n + _BLOCK - 1) // _BLOCK,)
    _fmin_strided_kernel[grid](
        t,
        out,
        n,
        hi,
        lo,
        *shp,
        *std,
        RANK=rank,
        BLOCK=_BLOCK,
        num_warps=_WARPS,
    )
    return out


def _fusable(dtype):
    return dtype != torch.float64 and dtype != torch.bool


def _foreach_min(tensors, scalars):
    count = len(tensors)
    if count == 0:
        return []
    dtype0 = tensors[0].dtype
    batch = count > 1 and _fusable(dtype0)
    if batch:
        for t in tensors:
            if t.dtype != dtype0 or not t.is_contiguous():
                batch = False
                break
    if not batch:
        return [_min_scalar_one(t, s) for t, s in zip(tensors, scalars)]

    outs = [torch.empty(t.shape, dtype=t.dtype, device=t.device) for t in tensors]
    ns = [o.numel() for o in outs]
    if max(ns) > _I32_MAX:
        return [_min_scalar_one(t, s) for t, s in zip(tensors, scalars)]
    hs = [_scalar_pair(s, dtype0)[0] for s in scalars]
    pad_h = 0 if not dtype0.is_floating_point else 0.0

    for base in range(0, count, _SLOTS):
        stop = min(base + _SLOTS, count)
        total = 0
        for k in range(base, stop):
            total += (ns[k] + _BLOCK - 1) // _BLOCK
        if total == 0:
            continue
        xt = tensors[base:stop]
        ot = outs[base:stop]
        nt = ns[base:stop]
        ht = hs[base:stop]
        pad = _SLOTS - (stop - base)
        if pad:
            xt = xt + [xt[0]] * pad
            ot = ot + [ot[0]] * pad
            nt = nt + [0] * pad
            ht = ht + [pad_h] * pad
        _fmin4_kernel[(total,)](
            xt[0],
            xt[1],
            xt[2],
            xt[3],
            ot[0],
            ot[1],
            ot[2],
            ot[3],
            nt[0],
            nt[1],
            nt[2],
            nt[3],
            ht[0],
            ht[1],
            ht[2],
            ht[3],
            BLOCK=_BLOCK,
            num_warps=_WARPS,
        )
    return outs


def foreach_minimum_scalar_list(self=None, scalars=None, **kwargs):
    tensors = self
    if tensors is None:
        tensors = kwargs.get("tensors")
    if scalars is None:
        scalars = kwargs.get("scalars")

    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    if isinstance(scalars, (list, tuple)):
        scalar_list = scalars
    elif isinstance(scalars, torch.Tensor):
        scalar_list = [scalars] if scalars.dim() == 0 else list(scalars)
    else:
        try:
            scalar_list = list(scalars)
        except TypeError:
            scalar_list = [scalars]

    if len(tensors) != len(scalar_list):
        raise RuntimeError(
            "Tensor list must have same number of elements as scalar list."
        )

    numbers = []
    for s in scalar_list:
        numbers.append(s.item() if isinstance(s, torch.Tensor) else s)
    return _foreach_min(tensors, numbers)
