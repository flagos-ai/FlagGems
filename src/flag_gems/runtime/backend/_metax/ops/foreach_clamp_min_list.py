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

"""Triton implementation of the foreach clamp_min list operator.

Semantics: per list entry i, out[i] = (self[i] < other[i]) ? other[i] : self[i]
(ATen clamp_min, implemented with an explicit compare/select so that NaN and
signed-zero behaviour matches the reference instead of maxnum semantics).

Layout of the implementation:
  * one fused launch for a homogeneous list (contiguous tensors, same dtype and
    same shape per pair), with the list slot selected from program_id; small
    lists use a fixed launch configuration, bandwidth-bound lists an autotuned
    one,
  * per-entry flat launch for a single contiguous pair,
  * general rank<=8 strided/broadcast kernel for noncontiguous pairs.
"""

import struct

import torch
import triton
import triton.language as tl

_MAX_RANK = 8
_BATCH_SLOTS = 4
# Above this many elements per list entry the fused launch is bandwidth bound and
# its launch configuration is autotuned; below it the call is launch bound and a
# fixed configuration avoids the autotuner dispatch cost.
_SMALL_N = 1 << 20

_MODE_TENSOR = 0  # other is a full tensor, loaded at the element offset
_MODE_SCALAR_TENSOR = 1  # other is a 1-element tensor, loaded once
_MODE_PY_FLOAT = 2  # other is a python float, passed as fp32 scalar
_MODE_PY_F64 = 3  # other is a python float for an fp64 result (raw bits)
_MODE_PY_INT = 4  # other is a python int, passed as int64 scalar

# Element-type tag used as an autotune key so a winner for one dtype (or device
# family) is never reused for another.
_DTYPE_KEY = {
    torch.float16: 0,
    torch.bfloat16: 1,
    torch.float32: 2,
    torch.float64: 3,
    torch.int8: 4,
    torch.int16: 5,
    torch.int32: 6,
    torch.int64: 7,
}


@triton.jit
def _flat_body(
    a_ptr, b_ptr, o_ptr, n, start, pid, BLOCK: tl.constexpr, INT64: tl.constexpr
):
    if INT64:
        offs = (pid - start).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = (pid - start) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    dt = o_ptr.dtype.element_ty
    a = tl.load(a_ptr + offs, mask=mask, other=0).to(dt)
    b = tl.load(b_ptr + offs, mask=mask, other=0).to(dt)
    tl.store(o_ptr + offs, tl.where(a < b, b, a), mask=mask)


@triton.jit
def _flat_kernel(
    a_ptr,
    b_ptr,
    o_ptr,
    n,
    sf,
    si,
    BLOCK: tl.constexpr,
    INT64: tl.constexpr,
    MODE: tl.constexpr,
):
    pid = tl.program_id(0)
    if INT64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    dt = o_ptr.dtype.element_ty
    a = tl.load(a_ptr + offs, mask=mask, other=0).to(dt)
    if MODE == 0:
        b = tl.load(b_ptr + offs, mask=mask, other=0).to(dt)
    elif MODE == 1:
        b = tl.load(b_ptr).to(dt)
    elif MODE == 2:
        b = sf.to(dt)
    elif MODE == 3:
        b = si.to(tl.float64, bitcast=True).to(dt)
    else:
        b = si.to(dt)
    res = tl.where(a < b, b, a)
    tl.store(o_ptr + offs, res, mask=mask)


@triton.jit
def _batch_slots(
    pid,
    a0,
    b0,
    o0,
    n0,
    a1,
    b1,
    o1,
    n1,
    a2,
    b2,
    o2,
    n2,
    a3,
    b3,
    o3,
    n3,
    BLOCK: tl.constexpr,
    INT64: tl.constexpr,
):
    # Slot boundaries are derived from BLOCK inside the kernel so the launch
    # configuration stays tunable without extra host-side metadata.
    c1 = (n0 + BLOCK - 1) // BLOCK
    c2 = c1 + (n1 + BLOCK - 1) // BLOCK
    c3 = c2 + (n2 + BLOCK - 1) // BLOCK
    if pid < c1:
        _flat_body(a0, b0, o0, n0, 0, pid, BLOCK, INT64)
    else:
        if pid < c2:
            _flat_body(a1, b1, o1, n1, c1, pid, BLOCK, INT64)
        else:
            if pid < c3:
                _flat_body(a2, b2, o2, n2, c2, pid, BLOCK, INT64)
            else:
                _flat_body(a3, b3, o3, n3, c3, pid, BLOCK, INT64)


@triton.jit
def _batch4_small(
    a0,
    b0,
    o0,
    n0,
    a1,
    b1,
    o1,
    n1,
    a2,
    b2,
    o2,
    n2,
    a3,
    b3,
    o3,
    n3,
    BLOCK: tl.constexpr,
    INT64: tl.constexpr,
):
    _batch_slots(
        tl.program_id(0),
        a0,
        b0,
        o0,
        n0,
        a1,
        b1,
        o1,
        n1,
        a2,
        b2,
        o2,
        n2,
        a3,
        b3,
        o3,
        n3,
        BLOCK,
        INT64,
    )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 1024}, num_warps=4),
        triton.Config({"BLOCK": 1024}, num_warps=8),
        triton.Config({"BLOCK": 2048}, num_warps=4),
        triton.Config({"BLOCK": 2048}, num_warps=8),
        triton.Config({"BLOCK": 4096}, num_warps=4),
        triton.Config({"BLOCK": 4096}, num_warps=8),
    ],
    key=["KEY_DTYPE", "n0", "n1", "n2", "n3"],
)
@triton.jit
def _batch4_kernel(
    a0,
    b0,
    o0,
    n0,
    a1,
    b1,
    o1,
    n1,
    a2,
    b2,
    o2,
    n2,
    a3,
    b3,
    o3,
    n3,
    KEY_DTYPE: tl.constexpr,
    BLOCK: tl.constexpr,
    INT64: tl.constexpr,
):
    _batch_slots(
        tl.program_id(0),
        a0,
        b0,
        o0,
        n0,
        a1,
        b1,
        o1,
        n1,
        a2,
        b2,
        o2,
        n2,
        a3,
        b3,
        o3,
        n3,
        BLOCK,
        INT64,
    )


@triton.jit
def _strided_kernel(
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
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6,
    a7,
    b0,
    b1,
    b2,
    b3,
    b4,
    b5,
    b6,
    b7,
    sf,
    si,
    BLOCK: tl.constexpr,
    RANK: tl.constexpr,
    MODE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n
    rem = offs
    aoff = tl.zeros((BLOCK,), tl.int64)
    boff = tl.zeros((BLOCK,), tl.int64)
    if RANK >= 8:
        idx = rem % s7
        rem = rem // s7
        aoff += idx * a7
        boff += idx * b7
    if RANK >= 7:
        idx = rem % s6
        rem = rem // s6
        aoff += idx * a6
        boff += idx * b6
    if RANK >= 6:
        idx = rem % s5
        rem = rem // s5
        aoff += idx * a5
        boff += idx * b5
    if RANK >= 5:
        idx = rem % s4
        rem = rem // s4
        aoff += idx * a4
        boff += idx * b4
    if RANK >= 4:
        idx = rem % s3
        rem = rem // s3
        aoff += idx * a3
        boff += idx * b3
    if RANK >= 3:
        idx = rem % s2
        rem = rem // s2
        aoff += idx * a2
        boff += idx * b2
    if RANK >= 2:
        idx = rem % s1
        rem = rem // s1
        aoff += idx * a1
        boff += idx * b1
    if RANK >= 1:
        idx = rem % s0
        aoff += idx * a0
        boff += idx * b0
    dt = o_ptr.dtype.element_ty
    a = tl.load(a_ptr + aoff, mask=mask, other=0).to(dt)
    if MODE == 0:
        b = tl.load(b_ptr + boff, mask=mask, other=0).to(dt)
    elif MODE == 1:
        b = tl.load(b_ptr).to(dt)
    elif MODE == 2:
        b = sf.to(dt)
    elif MODE == 3:
        b = si.to(tl.float64, bitcast=True).to(dt)
    else:
        b = si.to(dt)
    res = tl.where(a < b, b, a)
    tl.store(o_ptr + offs, res, mask=mask)


def _cdiv(x, y):
    return (x + y - 1) // y


def _is_seq(value):
    return isinstance(value, (list, tuple))


def _scalar_mode(value, out_dtype):
    """Return (sf, si, mode) for a python scalar operand."""
    if isinstance(value, bool):
        return 0.0, int(value), _MODE_PY_INT
    if isinstance(value, int):
        return 0.0, int(value), _MODE_PY_INT
    fv = float(value)
    if out_dtype == torch.float64:
        bits = struct.unpack("<q", struct.pack("<d", fv))[0]
        return 0.0, bits, _MODE_PY_F64
    return fv, 0, _MODE_PY_FLOAT


def _bcast_strides(t, oshape):
    """Strides of `t` broadcast to oshape (0 for broadcast / size-1 dims)."""
    rank = len(oshape)
    tshape = t.shape
    tstride = t.stride()
    pad = rank - len(tshape)
    out = []
    for i in range(rank):
        if i < pad:
            out.append(0)
        else:
            j = i - pad
            out.append(0 if tshape[j] == 1 else tstride[j])
    return out


def _block_for(dtype):
    esize = dtype.itemsize
    if esize >= 8:
        return 512
    if esize >= 4:
        return 1024
    return 2048


def _entry(a, b):
    if not isinstance(a, torch.Tensor):
        raise TypeError("foreach_clamp_min expects tensors in the first list")

    if isinstance(b, torch.Tensor):
        out_dtype = torch.result_type(a, b)
        b_is_scalar = b.numel() == 1
        out_shape = torch.broadcast_shapes(a.shape, b.shape) if b_is_scalar else a.shape
    else:
        out_dtype = a.dtype
        b_is_scalar = True
        out_shape = a.shape

    out = torch.empty(out_shape, dtype=out_dtype, device=a.device)
    n = out.numel()
    if n == 0:
        return out

    if b_is_scalar:
        b_ptr = b if isinstance(b, torch.Tensor) else a
        if isinstance(b, torch.Tensor):
            sf, si, mode = 0.0, 0, _MODE_SCALAR_TENSOR
        else:
            sf, si, mode = _scalar_mode(b, out_dtype)
    else:
        b_ptr = b
        sf, si, mode = 0.0, 0, _MODE_TENSOR

    block = _block_for(out_dtype)
    int64 = n >= 2147483648

    a_flat = a.is_contiguous() and a.numel() == n
    if b_is_scalar:
        b_flat = True
    else:
        b_flat = b.is_contiguous() and b.shape == a.shape

    if a_flat and b_flat:
        grid = (_cdiv(n, block),)
        _flat_kernel[grid](
            a, b_ptr, out, n, sf, si, BLOCK=block, INT64=int64, MODE=mode, num_warps=4
        )
        return out

    # General broadcast / strided path (contiguous output, rank <= 8).
    rank = len(out_shape)
    if rank > _MAX_RANK:
        raise RuntimeError(
            "foreach_clamp_min: rank %d exceeds supported maximum" % rank
        )
    sa = _bcast_strides(a, out_shape)
    sb = _bcast_strides(b, out_shape) if not b_is_scalar else [0] * rank
    shapes = list(out_shape) + [1] * (_MAX_RANK - rank)
    sa_p = sa + [0] * (_MAX_RANK - rank)
    sb_p = sb + [0] * (_MAX_RANK - rank)
    grid = (_cdiv(n, block),)
    _strided_kernel[grid](
        a,
        b_ptr,
        out,
        n,
        *shapes,
        *sa_p,
        *sb_p,
        sf,
        si,
        BLOCK=block,
        RANK=rank,
        MODE=mode,
        num_warps=4,
    )
    return out


def _batch_launch(self_list, other_list):
    """Fused launch for a homogeneous list; returns the fresh output list or None."""
    length = len(self_list)
    dtype = None
    device = None
    for a, b in zip(self_list, other_list):
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            return None
        if a.dtype is not b.dtype:
            return None
        ad = a.dtype
        if dtype is None:
            dtype = ad
        elif ad is not dtype:
            return None
        if not a.is_contiguous() or not b.is_contiguous():
            return None
        if a.shape != b.shape:
            return None
        if a.device != b.device:
            return None
        if device is None:
            device = a.device
        elif a.device != device:
            return None
    if dtype == torch.bool:
        return None

    outs = []
    ns = []
    maxn = 0
    for a in self_list:
        n = a.numel()
        ns.append(n)
        outs.append(torch.empty_like(a))
        if n > maxn:
            maxn = n
    if maxn == 0:
        return outs

    if length == _BATCH_SLOTS:
        # Hot path for a full list: no padding copies are needed.
        a0, b0, o0, n0 = self_list[0], other_list[0], outs[0], ns[0]
        a1, b1, o1, n1 = self_list[1], other_list[1], outs[1], ns[1]
        a2, b2, o2, n2 = self_list[2], other_list[2], outs[2], ns[2]
        a3, b3, o3, n3 = self_list[3], other_list[3], outs[3], ns[3]
    else:
        pad = 0
        for i in range(length):
            if ns[i] > 0:
                pad = i
                break
        fill = _BATCH_SLOTS - length
        A = list(self_list) + [self_list[pad]] * fill
        B = list(other_list) + [other_list[pad]] * fill
        OUT_ = list(outs) + [outs[pad]] * fill
        N = list(ns) + [0] * fill
        a0, b0, o0, n0 = A[0], B[0], OUT_[0], N[0]
        a1, b1, o1, n1 = A[1], B[1], OUT_[1], N[1]
        a2, b2, o2, n2 = A[2], B[2], OUT_[2], N[2]
        a3, b3, o3, n3 = A[3], B[3], OUT_[3], N[3]

    int64 = maxn >= 2147483648
    if maxn <= _SMALL_N:
        # Launch-bound regime: fixed configuration, no autotuner dispatch.
        block = _block_for(dtype)
        m = block - 1
        grid = (
            (n0 + m) // block
            + (n1 + m) // block
            + (n2 + m) // block
            + (n3 + m) // block,
        )
        _batch4_small[grid](
            a0,
            b0,
            o0,
            n0,
            a1,
            b1,
            o1,
            n1,
            a2,
            b2,
            o2,
            n2,
            a3,
            b3,
            o3,
            n3,
            BLOCK=block,
            INT64=int64,
            num_warps=4,
        )
        return outs

    key_dtype = _DTYPE_KEY.get(dtype)
    if key_dtype is None:
        return None

    def _grid(meta):
        blk = meta["BLOCK"]
        mm = blk - 1
        return (
            (n0 + mm) // blk + (n1 + mm) // blk + (n2 + mm) // blk + (n3 + mm) // blk,
        )

    _batch4_kernel[_grid](
        a0,
        b0,
        o0,
        n0,
        a1,
        b1,
        o1,
        n1,
        a2,
        b2,
        o2,
        n2,
        a3,
        b3,
        o3,
        n3,
        KEY_DTYPE=key_dtype,
        INT64=int64,
    )
    return outs


def foreach_clamp_min_list(self, other):
    if not _is_seq(self):
        return _entry(self, other)

    self_list = self if isinstance(self, list) else list(self)
    length = len(self_list)
    if _is_seq(other):
        other_list = other if isinstance(other, list) else list(other)
        if len(other_list) != length:
            raise RuntimeError(
                "Expected a list with %d tensors, but got %d instead."
                % (length, len(other_list))
            )
    else:
        other_list = None

    if other_list is not None and 2 <= length <= _BATCH_SLOTS:
        outs = _batch_launch(self_list, other_list)
        if outs is not None:
            return outs

    if other_list is None:
        return [_entry(a, other) for a in self_list]
    return [_entry(a, b) for a, b in zip(self_list, other_list)]
