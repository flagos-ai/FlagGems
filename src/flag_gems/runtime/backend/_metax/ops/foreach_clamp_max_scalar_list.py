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

"""Triton implementation of aten::_foreach_clamp_max.ScalarList.

out[i] = isnan(self[i]) ? self[i] : min(self[i], scalars[i])   (elementwise)

Design notes
------------
* The scalar is transmitted as the exact IEEE-754 double bit pattern (two int32
  halves) so fp64 tensors keep full precision and fp16/bf16/fp32 tensors see the
  same operand ATen's float opmath conversion produces.  A direct f64->bf16
  ``arith.truncf`` is not lowerable on this backend, so fp16/bf16/fp32 go
  double -> float -> storage dtype (exactly ATen's opmath path).
* NaN inputs must survive: ``tl.minimum`` is minnum, so NaN is handled with an
  explicit ``tl.where(x != x, x, ...)`` (a NaN *scalar* must still yield x).
* Contiguous lists of one dtype are packed along ``grid.y`` so a single launch
  covers up to four tensors: per-tensor launch gaps cost several microseconds
  and the device body for short lists is only ~2.5 us, so the host path is the
  real budget there.  A tight four-tensor fast path avoids the generic pooling
  machinery and converts all four scalars with one struct call.
"""

import struct

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------- scalar plumbing

_UNPACK_Q = struct.Struct("<Q").unpack
_PACK_D = struct.Struct("<d").pack
_PACK_4D = struct.Struct("<4d").pack
_UNPACK_4Q = struct.Struct("<4Q").unpack


def _hi_lo(bits):
    hi = (bits >> 32) & 0xFFFFFFFF
    lo = bits & 0xFFFFFFFF
    return (
        hi - 0x100000000 if hi >= 0x80000000 else hi,
        lo - 0x100000000 if lo >= 0x80000000 else lo,
    )


def _double_hi_lo(value):
    return _hi_lo(_UNPACK_Q(_PACK_D(float(value)))[0])


# Scalar-conversion modes (constexpr kernel argument).
_MODE_F32 = 0  # fp16 / bf16 / fp32: double -> float -> storage dtype (ATen opmath)
_MODE_F64 = 1  # fp64: keep the double untouched
_MODE_INT = 2  # integers: static_cast semantics from the double

# Above this element count a launch covers one tensor only: the saved launch
# gaps are negligible next to the saturated streaming time.
_FUSE_MAX_N = 1 << 26
_FUSE_SLOTS = 4


# ---------------------------------------------------------------- kernels


@triton.jit
def _clamp_flat_body(
    in_ptr,
    out_ptr,
    n,
    s_hi,
    s_lo,
    BLOCK: tl.constexpr,
    USE_I64: tl.constexpr,
    MODE: tl.constexpr,
    MASKED: tl.constexpr,
):
    pid = tl.program_id(0)
    if USE_I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    if MASKED:
        mask = offs < n
        x = tl.load(in_ptr + offs, mask=mask, other=0)
    else:
        x = tl.load(in_ptr + offs)
    u = ((s_hi.to(tl.int64) & 0xFFFFFFFF) << 32) | (s_lo.to(tl.int64) & 0xFFFFFFFF)
    d = tl.cast(u, tl.float64, bitcast=True)
    if MODE == 0:
        s = d.to(tl.float32).to(x.dtype)
    else:
        s = d.to(x.dtype)
    y = tl.where(x != x, x, tl.minimum(x, s))
    if MASKED:
        tl.store(out_ptr + offs, y, mask=mask)
    else:
        tl.store(out_ptr + offs, y)


@triton.jit
def _clamp_max_flat_kernel(
    in_ptr,
    out_ptr,
    n,
    s_hi,
    s_lo,
    BLOCK: tl.constexpr,
    USE_I64: tl.constexpr,
    MODE: tl.constexpr,
    MASKED: tl.constexpr,
):
    _clamp_flat_body(
        in_ptr,
        out_ptr,
        n,
        s_hi,
        s_lo,
        BLOCK=BLOCK,
        USE_I64=USE_I64,
        MODE=MODE,
        MASKED=MASKED,
    )


@triton.jit
def _clamp_max_multi_kernel(
    in0,
    in1,
    in2,
    in3,
    out0,
    out1,
    out2,
    out3,
    n0,
    n1,
    n2,
    n3,
    h0,
    l0,
    h1,
    l1,
    h2,
    l2,
    h3,
    l3,
    NSLOTS: tl.constexpr,
    BLOCK: tl.constexpr,
    USE_I64: tl.constexpr,
    MODE: tl.constexpr,
    MASKED: tl.constexpr,
):
    """One launch for NSLOTS <= 4 contiguous tensors (slot on axis 1)."""
    slot = tl.program_id(1)
    if slot == 0:
        _clamp_flat_body(
            in0,
            out0,
            n0,
            h0,
            l0,
            BLOCK=BLOCK,
            USE_I64=USE_I64,
            MODE=MODE,
            MASKED=MASKED,
        )
    if NSLOTS > 1:
        if slot == 1:
            _clamp_flat_body(
                in1,
                out1,
                n1,
                h1,
                l1,
                BLOCK=BLOCK,
                USE_I64=USE_I64,
                MODE=MODE,
                MASKED=MASKED,
            )
    if NSLOTS > 2:
        if slot == 2:
            _clamp_flat_body(
                in2,
                out2,
                n2,
                h2,
                l2,
                BLOCK=BLOCK,
                USE_I64=USE_I64,
                MODE=MODE,
                MASKED=MASKED,
            )
    if NSLOTS > 3:
        if slot == 3:
            _clamp_flat_body(
                in3,
                out3,
                n3,
                h3,
                l3,
                BLOCK=BLOCK,
                USE_I64=USE_I64,
                MODE=MODE,
                MASKED=MASKED,
            )


@triton.jit
def _clamp_max_strided_kernel(
    in_ptr,
    out_ptr,
    n,
    s_hi,
    s_lo,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    e0,
    e1,
    e2,
    e3,
    e4,
    e5,
    e6,
    e7,
    ND: tl.constexpr,
    BLOCK: tl.constexpr,
    MODE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n
    rem = offs
    off = tl.zeros((BLOCK,), dtype=tl.int64)
    # d0/e0 is the innermost (fastest varying) dim of the row-major flat index.
    if ND > 0:
        c = rem % d0
        rem = rem // d0
        off += c * e0
    if ND > 1:
        c = rem % d1
        rem = rem // d1
        off += c * e1
    if ND > 2:
        c = rem % d2
        rem = rem // d2
        off += c * e2
    if ND > 3:
        c = rem % d3
        rem = rem // d3
        off += c * e3
    if ND > 4:
        c = rem % d4
        rem = rem // d4
        off += c * e4
    if ND > 5:
        c = rem % d5
        rem = rem // d5
        off += c * e5
    if ND > 6:
        c = rem % d6
        rem = rem // d6
        off += c * e6
    if ND > 7:
        c = rem % d7
        rem = rem // d7
        off += c * e7
    x = tl.load(in_ptr + off, mask=mask, other=0)
    u = ((s_hi.to(tl.int64) & 0xFFFFFFFF) << 32) | (s_lo.to(tl.int64) & 0xFFFFFFFF)
    d = tl.cast(u, tl.float64, bitcast=True)
    if MODE == 0:
        s = d.to(tl.float32).to(x.dtype)
    else:
        s = d.to(x.dtype)
    y = tl.where(x != x, x, tl.minimum(x, s))
    tl.store(out_ptr + offs, y, mask=mask)


# ---------------------------------------------------------------- layout helpers


def _collapse(shape, strides):
    """Drop unit dims and merge memory-adjacent dims.

    Returns dims innermost-first (fastest varying first), the order the kernel's
    div/mod chain consumes: a dim is merged into the group below it when its
    stride equals the span of that already-dense inner group.
    """
    dims = [(size, stride) for size, stride in zip(shape, strides) if size != 1]
    merged = []
    for size, stride in reversed(dims):
        if merged and stride == merged[-1][0] * merged[-1][1]:
            merged[-1][0] *= size
        else:
            merged.append([size, stride])
    return merged


def _block_for(n, itemsize):
    if n <= 1024:
        return 1024, 4
    if n <= 1 << 16:
        return 2048, 4
    if n <= 1 << 22:
        return 4096, 4
    return 8192, 8


def _mode_for(dtype):
    if dtype == torch.float64:
        return _MODE_F64
    if dtype.is_floating_point:
        return _MODE_F32
    return _MODE_INT


# ---------------------------------------------------------------- launch helpers


def _flat_launch(t, out, n, s_hi, s_lo, mode):
    block, warps = _block_for(n, t.element_size())
    _clamp_max_flat_kernel[(triton.cdiv(n, block),)](
        t,
        out,
        n,
        s_hi,
        s_lo,
        block,
        n > 0x7FFFFFFF,
        mode,
        n % block != 0,
        num_warps=warps,
    )


def _strided_launch(t, out, n, s_hi, s_lo, mode):
    dims = _collapse(tuple(t.shape), tuple(t.stride()))
    nd = len(dims)
    if nd == 0:
        dims = [[1, 0]]
        nd = 1
    if nd > 8:
        raise RuntimeError("clamp_max: unsupported tensor rank for strided path")
    ds = [d for d, _ in dims] + [1] * (8 - nd)
    es = [e for _, e in dims] + [0] * (8 - nd)
    block, warps = _block_for(n, t.element_size())
    _clamp_max_strided_kernel[(triton.cdiv(n, block),)](
        t,
        out,
        n,
        s_hi,
        s_lo,
        ds[0],
        ds[1],
        ds[2],
        ds[3],
        ds[4],
        ds[5],
        ds[6],
        ds[7],
        es[0],
        es[1],
        es[2],
        es[3],
        es[4],
        es[5],
        es[6],
        es[7],
        nd,
        block,
        mode,
        num_warps=warps,
    )


def _fused_launch(chunk, block, use_i64, mode, warps):
    """One launch for up to _FUSE_SLOTS same-dtype contiguous tensors."""
    k = len(chunk)
    slots = chunk + [chunk[0]] * (_FUSE_SLOTS - k)
    blocks = 0
    masked = False
    for item in chunk:
        b = -(-item[4] // block)
        if b > blocks:
            blocks = b
        if item[4] % block:
            masked = True
    _clamp_max_multi_kernel[(blocks, k)](
        slots[0][0],
        slots[1][0],
        slots[2][0],
        slots[3][0],
        slots[0][1],
        slots[1][1],
        slots[2][1],
        slots[3][1],
        slots[0][4],
        slots[1][4],
        slots[2][4],
        slots[3][4],
        slots[0][2],
        slots[0][3],
        slots[1][2],
        slots[1][3],
        slots[2][2],
        slots[2][3],
        slots[3][2],
        slots[3][3],
        k,
        block,
        use_i64,
        mode,
        masked,
        num_warps=warps,
    )


def _fuse_four(tensors, scl):
    """Tight host path for exactly four same-dtype contiguous equal-size tensors.

    Returns the four outputs, or None when the list does not qualify.
    """
    t0 = tensors[0]
    t1 = tensors[1]
    t2 = tensors[2]
    t3 = tensors[3]
    dtype = t0.dtype
    if t1.dtype != dtype or t2.dtype != dtype or t3.dtype != dtype:
        return None
    if not (
        t0.is_contiguous()
        and t1.is_contiguous()
        and t2.is_contiguous()
        and t3.is_contiguous()
    ):
        return None
    n = t0.numel()
    if n <= 0 or n > _FUSE_MAX_N:
        return None
    if t1.numel() != n or t2.numel() != n or t3.numel() != n:
        return None
    # One allocator round trip instead of four, then four contiguous views.  The
    # views are distinct tensors with distinct data pointers, correct shape/dtype
    # and no aliasing of the inputs.
    o0, o1, o2, o3 = torch.empty((4,) + t0.shape, dtype=dtype, device=t0.device).unbind(
        0
    )
    q0, q1, q2, q3 = _UNPACK_4Q(
        _PACK_4D(float(scl[0]), float(scl[1]), float(scl[2]), float(scl[3]))
    )
    h0, l0 = _hi_lo(q0)
    h1, l1 = _hi_lo(q1)
    h2, l2 = _hi_lo(q2)
    h3, l3 = _hi_lo(q3)
    block, warps = _block_for(n, t0.element_size())
    _clamp_max_multi_kernel[(-(-n // block), 4)](
        t0,
        t1,
        t2,
        t3,
        o0,
        o1,
        o2,
        o3,
        n,
        n,
        n,
        n,
        h0,
        l0,
        h1,
        l1,
        h2,
        l2,
        h3,
        l3,
        4,
        block,
        False,
        _mode_for(dtype),
        n % block != 0,
        num_warps=warps,
    )
    return [o0, o1, o2, o3]


# ---------------------------------------------------------------- entry point


def foreach_clamp_max_scalar_list(self, scalars):
    tensors = self
    if type(tensors) is not list and type(tensors) is not tuple:
        tensors = [tensors] if torch.is_tensor(tensors) else list(tensors)
    if type(scalars) is list or type(scalars) is tuple:
        scl = scalars
    else:
        try:
            scl = list(scalars)
        except TypeError:
            scl = [scalars] * len(tensors)

    count = len(tensors)
    if len(scl) != count:
        raise RuntimeError(
            "Tensor list and scalar list must have the same length, but got "
            + str(count)
            + " and "
            + str(len(scl))
        )

    if count == 4:
        fast = _fuse_four(tensors, scl)
        if fast is not None:
            return fast

    outs = [None] * count
    pool = {}  # (dtype, block, warps, use_i64) -> [(t, out, hi, lo, n), ...]
    for i in range(count):
        t = tensors[i]
        n = t.numel()
        if n == 0:
            outs[i] = torch.empty_like(t, memory_format=torch.contiguous_format)
            continue
        s_hi, s_lo = _double_hi_lo(scl[i])
        if not t.is_contiguous():
            out = torch.empty_like(t, memory_format=torch.contiguous_format)
            outs[i] = out
            _strided_launch(t, out, n, s_hi, s_lo, _mode_for(t.dtype))
        elif n > _FUSE_MAX_N:
            out = torch.empty_like(t)
            outs[i] = out
            _flat_launch(t, out, n, s_hi, s_lo, _mode_for(t.dtype))
        else:
            out = torch.empty_like(t)
            outs[i] = out
            block, warps = _block_for(n, t.element_size())
            key = (t.dtype, block, warps, n > 0x7FFFFFFF)
            entry = (t, out, s_hi, s_lo, n)
            bucket = pool.get(key)
            if bucket is None:
                pool[key] = [entry]
            else:
                bucket.append(entry)

    for key, items in pool.items():
        mode = _mode_for(key[0])
        block = key[1]
        warps = key[2]
        use_i64 = key[3]
        total = len(items)
        if total == 1:
            t, out, s_hi, s_lo, n = items[0]
            _flat_launch(t, out, n, s_hi, s_lo, mode)
            continue
        start = 0
        while start < total:
            chunk = items[start : start + _FUSE_SLOTS]
            start += _FUSE_SLOTS
            if len(chunk) == 1:
                t, out, s_hi, s_lo, n = chunk[0]
                _flat_launch(t, out, n, s_hi, s_lo, mode)
            else:
                _fused_launch(chunk, block, use_i64, mode, warps)

    return outs
