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

"""In-place foreach clamp_min (scalar) with Triton kernels.

Semantics: for every tensor t in `self`, t[i] = max(t[i], scalar) written back
into t's existing storage, matching ATen `_foreach_clamp_min_.Scalar`:
NaN-propagating max, float32 "opmath" comparison for fp16/bf16, and the same
scalar conversion / rejection rules for integer dtypes.  `run` returns None,
like the reference in-place op.

All flat (contiguous or storage-dense-under-permutation) tensors that share a
dtype and a scalar are fused into a single launch of an 8-slot kernel, so a
4-tensor foreach list costs one Triton launch instead of four.
"""

import struct

import torch
import triton
import triton.language as tl

_MAX_DIMS = 8
_SLOTS = 8
_INT64_LIMIT = 1 << 31

_MODE_BY_DTYPE = {
    torch.float16: 0,
    torch.bfloat16: 0,
    torch.float32: 0,
    torch.float64: 1,
    torch.int8: 2,
    torch.uint8: 2,
    torch.int16: 2,
    torch.int32: 2,
    torch.int64: 2,
}

_ATEN_INT_NAME = {
    torch.uint8: "unsigned char",
    torch.int8: "signed char",
    torch.int16: "short",
    torch.int32: "int",
    torch.int64: "long int",
}


@triton.jit
def _cm_apply(x, s, s_lo, MODE: tl.constexpr, S_NAN: tl.constexpr):
    """clamp_min(x, s) with ATen semantics for the given MODE.

    MODE 0: float compute in fp32 (fp16/bf16/fp32 inputs, matches opmath_t)
    MODE 1: fp64, scalar carried bit-exactly as two int32 halves
    MODE 2: integer (int64 compute, exact for every integer dtype)
    """
    if MODE == 0:
        xf = x.to(tl.float32)
        if S_NAN:
            y = tl.full(xf.shape, float("nan"), tl.float32)
        else:
            y = tl.maximum(xf, s)
            y = tl.where(xf != xf, xf, y)
    elif MODE == 1:
        bits = (s.to(tl.int64) << 32) | (s_lo.to(tl.int64) & 0xFFFFFFFF)
        s64 = bits.to(tl.float64, bitcast=True)
        if S_NAN:
            y = tl.full(x.shape, float("nan"), tl.float64)
        else:
            y = tl.maximum(x, s64)
            y = tl.where(x != x, x, y)
    else:
        y = tl.maximum(x.to(tl.int64), s.to(tl.int64))
    return y


@triton.jit
def _cm_slot(
    ptr, base, n, s, s_lo, MODE: tl.constexpr, S_NAN: tl.constexpr, BLOCK: tl.constexpr
):
    offs = base + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(ptr + offs, mask=mask, other=0)
    y = _cm_apply(x, s, s_lo, MODE, S_NAN)
    tl.store(ptr + offs, y, mask=mask)


@triton.jit
def _cm_slot_f32(ptr, base, n, s):
    """Hot body for the fast path: BLOCK=1024, fp32 opmath, no NaN scalar."""
    offs = base + tl.arange(0, 1024)
    mask = offs < n
    x = tl.load(ptr + offs, mask=mask, other=0)
    xf = x.to(tl.float32)
    y = tl.maximum(xf, s)
    y = tl.where(xf != xf, xf, y)
    tl.store(ptr + offs, y, mask=mask)


@triton.jit
def _cm_fast1(ptr, n, s):
    _cm_slot_f32(ptr, tl.program_id(0) * 1024, n, s)


@triton.jit
def _cm_fast2(p0, p1, n0, n1, s):
    pid = tl.program_id(0)
    b0 = tl.cdiv(n0, 1024)
    if pid < b0:
        _cm_slot_f32(p0, pid * 1024, n0, s)
    else:
        _cm_slot_f32(p1, (pid - b0) * 1024, n1, s)


@triton.jit
def _cm_fast3(p0, p1, p2, n0, n1, n2, s):
    pid = tl.program_id(0)
    b0 = tl.cdiv(n0, 1024)
    if pid < b0:
        _cm_slot_f32(p0, pid * 1024, n0, s)
    else:
        b1 = b0 + tl.cdiv(n1, 1024)
        if pid < b1:
            _cm_slot_f32(p1, (pid - b0) * 1024, n1, s)
        else:
            _cm_slot_f32(p2, (pid - b1) * 1024, n2, s)


@triton.jit
def _cm_fast4(p0, p1, p2, p3, n0, n1, n2, n3, s):
    pid = tl.program_id(0)
    b0 = tl.cdiv(n0, 1024)
    if pid < b0:
        _cm_slot_f32(p0, pid * 1024, n0, s)
    else:
        b1 = b0 + tl.cdiv(n1, 1024)
        if pid < b1:
            _cm_slot_f32(p1, (pid - b0) * 1024, n1, s)
        else:
            b2 = b1 + tl.cdiv(n2, 1024)
            if pid < b2:
                _cm_slot_f32(p2, (pid - b1) * 1024, n2, s)
            else:
                _cm_slot_f32(p3, (pid - b2) * 1024, n3, s)


@triton.jit
def _cm_flat_kernel(
    ptr,
    n,
    s,
    s_lo,
    MODE: tl.constexpr,
    S_NAN: tl.constexpr,
    I64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if I64:
        base = pid.to(tl.int64) * BLOCK
    else:
        base = pid * BLOCK
    _cm_slot(ptr, base, n, s, s_lo, MODE, S_NAN, BLOCK)


@triton.jit
def _cm_multi_kernel(
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
    s,
    s_lo,
    MODE: tl.constexpr,
    S_NAN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One launch for up to 8 flat tensors sharing a dtype and scalar."""
    pid = tl.program_id(0)
    b0 = tl.cdiv(n0, BLOCK)
    if pid < b0:
        _cm_slot(p0, pid * BLOCK, n0, s, s_lo, MODE, S_NAN, BLOCK)
    else:
        b1 = b0 + tl.cdiv(n1, BLOCK)
        if pid < b1:
            _cm_slot(p1, (pid - b0) * BLOCK, n1, s, s_lo, MODE, S_NAN, BLOCK)
        else:
            b2 = b1 + tl.cdiv(n2, BLOCK)
            if pid < b2:
                _cm_slot(p2, (pid - b1) * BLOCK, n2, s, s_lo, MODE, S_NAN, BLOCK)
            else:
                b3 = b2 + tl.cdiv(n3, BLOCK)
                if pid < b3:
                    _cm_slot(p3, (pid - b2) * BLOCK, n3, s, s_lo, MODE, S_NAN, BLOCK)
                else:
                    b4 = b3 + tl.cdiv(n4, BLOCK)
                    if pid < b4:
                        _cm_slot(
                            p4, (pid - b3) * BLOCK, n4, s, s_lo, MODE, S_NAN, BLOCK
                        )
                    else:
                        b5 = b4 + tl.cdiv(n5, BLOCK)
                        if pid < b5:
                            _cm_slot(
                                p5, (pid - b4) * BLOCK, n5, s, s_lo, MODE, S_NAN, BLOCK
                            )
                        else:
                            b6 = b5 + tl.cdiv(n6, BLOCK)
                            if pid < b6:
                                _cm_slot(
                                    p6,
                                    (pid - b5) * BLOCK,
                                    n6,
                                    s,
                                    s_lo,
                                    MODE,
                                    S_NAN,
                                    BLOCK,
                                )
                            else:
                                _cm_slot(
                                    p7,
                                    (pid - b6) * BLOCK,
                                    n7,
                                    s,
                                    s_lo,
                                    MODE,
                                    S_NAN,
                                    BLOCK,
                                )


@triton.jit
def _cm_strided_kernel(
    ptr,
    n,
    s,
    s_lo,
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
    ND: tl.constexpr,
    MODE: tl.constexpr,
    S_NAN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    lin = pid * BLOCK + tl.arange(0, BLOCK)
    mask = lin < n
    rem = lin
    off = tl.zeros([BLOCK], dtype=tl.int64)
    if ND > 0:
        i0 = rem % d0
        rem = rem // d0
        off += i0.to(tl.int64) * t0
    if ND > 1:
        i1 = rem % d1
        rem = rem // d1
        off += i1.to(tl.int64) * t1
    if ND > 2:
        i2 = rem % d2
        rem = rem // d2
        off += i2.to(tl.int64) * t2
    if ND > 3:
        i3 = rem % d3
        rem = rem // d3
        off += i3.to(tl.int64) * t3
    if ND > 4:
        i4 = rem % d4
        rem = rem // d4
        off += i4.to(tl.int64) * t4
    if ND > 5:
        i5 = rem % d5
        rem = rem // d5
        off += i5.to(tl.int64) * t5
    if ND > 6:
        i6 = rem % d6
        rem = rem // d6
        off += i6.to(tl.int64) * t6
    if ND > 7:
        i7 = rem % d7
        off += i7.to(tl.int64) * t7
    x = tl.load(ptr + off, mask=mask, other=0)
    y = _cm_apply(x, s, s_lo, MODE, S_NAN)
    tl.store(ptr + off, y, mask=mask)


def _flat_capable(t):
    """True when the view's memory image is `numel` consecutive elements of t's
    storage starting at t.data_ptr(): contiguous, or dense under a permutation."""
    dims = [(sz, st) for sz, st in zip(t.shape, t.stride()) if sz != 1]
    if not dims:
        return True
    dims = sorted(dims, key=lambda p: p[1])
    expect = 1
    for sz, st in dims:
        if st != expect:
            return False
        expect *= sz
    return True


def _overlaps(t):
    return any(sz > 1 and st == 0 for sz, st in zip(t.shape, t.stride()))


def _dims_for(t):
    """(sizes, strides) in ascending-stride order, size-1 dims dropped."""
    dims = [(sz, st) for sz, st in zip(t.shape, t.stride()) if sz != 1]
    dims = sorted(dims, key=lambda p: p[1])
    return [d[0] for d in dims], [d[1] for d in dims]


def _int_scalar(value, dtype):
    """ATen rejects float scalars for integer tensors and rejects overflow."""
    if isinstance(value, float):
        raise RuntimeError(
            "result type Float can't be cast to the desired output type %s"
            % _ATEN_INT_NAME.get(dtype, "long int")
        )
    info = torch.iinfo(dtype)
    v = int(value)
    if v < info.min or v > info.max:
        raise RuntimeError(
            "value cannot be converted to type %s without overflow"
            % (str(dtype).replace("torch.", ""),)
        )
    return v


def _prep_scalar(mode, dtype, value):
    """(sarg, s_lo, s_nan) for the kernel argument transport."""
    if mode == 0:
        s = float(value)
        return s, 0, s != s
    if mode == 1:
        raw = struct.unpack("<Q", struct.pack("<d", float(value)))[0]
        hi = raw >> 32
        lo = raw & 0xFFFFFFFF
        if hi >= (1 << 31):
            hi -= 1 << 32
        if lo >= (1 << 31):
            lo -= 1 << 32
        sv = float(value)
        return hi, lo, sv != sv
    return _int_scalar(value, dtype), 0, False


def _bucket(n_total):
    if n_total <= (1 << 16):
        return 1024, 4
    if n_total <= (1 << 22):
        return 2048, 4
    return 4096, 8


def _launch_flat(tensors, sizes, mode, sarg, s_lo, s_nan):
    n_total = 0
    for n in sizes:
        n_total += n
    block, warps = _bucket(n_total)
    if len(tensors) == 1:
        n = sizes[0]
        _cm_flat_kernel[((n + block - 1) // block,)](
            tensors[0],
            n,
            sarg,
            s_lo,
            MODE=mode,
            S_NAN=s_nan,
            I64=n >= _INT64_LIMIT,
            BLOCK=block,
            num_warps=warps,
        )
        return
    ptrs = list(tensors)
    ns = list(sizes)
    blocks = 0
    for n in ns:
        blocks += (n + block - 1) // block
    while len(ptrs) < _SLOTS:
        ptrs.append(ptrs[0])
        ns.append(0)
    _cm_multi_kernel[(blocks,)](
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
        sarg,
        s_lo,
        MODE=mode,
        S_NAN=s_nan,
        BLOCK=block,
        num_warps=warps,
    )


def _launch_strided(t, n, mode, sarg, s_lo, s_nan):
    block, warps = _bucket(n)
    sizes, strides = _dims_for(t)
    nd = len(sizes)
    if nd > _MAX_DIMS:
        raise RuntimeError("rank too large for the strided path")
    sizes = sizes + [1] * (_MAX_DIMS - nd)
    strides = strides + [0] * (_MAX_DIMS - nd)
    _cm_strided_kernel[((n + block - 1) // block,)](
        t,
        n,
        sarg,
        s_lo,
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
        ND=nd,
        MODE=mode,
        S_NAN=s_nan,
        BLOCK=block,
        num_warps=warps,
    )


_FLOAT_DTYPES = frozenset((torch.float16, torch.bfloat16, torch.float32))
_FAST_LIMIT = 1 << 16


def _try_fast(tensors, count, scalar):
    """Short list of contiguous, same-dtype float tensors with one scalar.

    Builds no dicts, key tuples or slices: one tight metadata loop, then one
    launch with no constexpr/metadata keyword arguments.
    """
    dt = tensors[0].dtype
    if dt not in _FLOAT_DTYPES:
        return False
    s = float(scalar)
    if s != s:
        return False
    ns = []
    total = 0
    for t in tensors:
        if t.dtype is not dt or not t.is_contiguous():
            return False
        n = t.numel()
        if n == 0:
            return False
        ns.append(n)
        total += n
    if total > _FAST_LIMIT:
        return False
    blocks = (ns[0] + 1023) >> 10
    if count == 1:
        _cm_fast1[(blocks,)](tensors[0], ns[0], s)
        return True
    blocks += (ns[1] + 1023) >> 10
    if count == 2:
        _cm_fast2[(blocks,)](tensors[0], tensors[1], ns[0], ns[1], s)
        return True
    blocks += (ns[2] + 1023) >> 10
    if count == 3:
        _cm_fast3[(blocks,)](tensors[0], tensors[1], tensors[2], ns[0], ns[1], ns[2], s)
        return True
    blocks += (ns[3] + 1023) >> 10
    _cm_fast4[(blocks,)](
        tensors[0], tensors[1], tensors[2], tensors[3], ns[0], ns[1], ns[2], ns[3], s
    )
    return True


def foreach_clamp_min_scalar_(self, scalar):
    if type(self) is list and not isinstance(scalar, (list, tuple)):
        fast_count = len(self)
        if 0 < fast_count <= 4 and _try_fast(self, fast_count, scalar):
            return None
    tensors = self if isinstance(self, (list, tuple)) else (self,)
    count = len(tensors)
    if count == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")
    if isinstance(scalar, (list, tuple)):
        if len(scalar) != count:
            raise RuntimeError(
                "Tensor list must have same number of elements as scalar list."
            )
        scalar_list = scalar
    else:
        scalar_list = None

    groups = {}
    singles = []
    memo = {}
    for i in range(count):
        t = tensors[i]
        dt = t.dtype
        mode = _MODE_BY_DTYPE.get(dt)
        if mode is None:
            if dt == torch.bool:
                raise RuntimeError(
                    "result type Long can't be cast to the desired output type bool"
                )
            raise RuntimeError("unsupported dtype %r" % (dt,))
        n = t.numel()
        if n == 0:
            continue
        if scalar_list is None:
            prep = memo.get(dt)
            if prep is None:
                prep = _prep_scalar(mode, dt, scalar)
                memo[dt] = prep
        else:
            prep = _prep_scalar(mode, dt, scalar_list[i])
        if t.is_contiguous():
            flat = True
        else:
            if _overlaps(t):
                raise RuntimeError(
                    "unsupported operation: more than one element of the "
                    "written-to tensor refers to a single memory location. "
                    "Please clone() the tensor before performing the operation."
                )
            flat = _flat_capable(t)
        if flat and n < _INT64_LIMIT:
            key = (mode, dt, prep[0], prep[1], prep[2])
            group = groups.get(key)
            if group is None:
                groups[key] = ([t], [n])
            else:
                group[0].append(t)
                group[1].append(n)
        else:
            singles.append((t, n, mode, prep[0], prep[1], prep[2], flat))

    for key, (ts, ns) in groups.items():
        mode = key[0]
        sarg, s_lo, s_nan = key[2], key[3], key[4]
        for start in range(0, len(ts), _SLOTS):
            _launch_flat(
                ts[start : start + _SLOTS],
                ns[start : start + _SLOTS],
                mode,
                sarg,
                s_lo,
                s_nan,
            )
    for t, n, mode, sarg, s_lo, s_nan, flat in singles:
        if flat:
            _launch_flat([t], [n], mode, sarg, s_lo, s_nan)
        else:
            _launch_strided(t, n, mode, sarg, s_lo, s_nan)
    return None
