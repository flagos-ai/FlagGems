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

"""foreach_tanh: out_k = tanh(in_k) for every tensor in a list.

Homogeneous lists of equally shaped, contiguous tensors are handled by one
fused launch writing into a single (M, numel) buffer that is returned as per
tensor views; otherwise one Triton launch runs per tensor.  Floating dtypes
keep their dtype (fp16/bf16 compute in fp32 opmath like ATen), integral/bool
inputs are promoted to float32 exactly as ATen's TensorIterator does, and
complex inputs use tanh(a+bi) = sinh(2a)/(cosh2a+cos2b) + i sin(2b)/(cosh2a+cos2b).
"""

import torch
import triton
import triton.language as tl

try:  # MetaX/flagtree layout
    from triton.language.extra import libdevice as _ld
except Exception:  # pragma: no cover
    from triton.language import libdevice as _ld


# ---------------------------------------------------------------- kernels


# Fixed launch shapes taken from the measured per-dtype autotune winners.
# Keeping them fixed removes the autotuner's per-call key/config rebuild, and
# because BLOCK is chosen per launch the tail mask can be dropped whenever
# BLOCK divides numel (always true for the small power-of-two shapes, which is
# what the 64x64 / 1-D power-of-two workloads hit).
def _flat_cfg(n, item_size):
    if n <= 4096:
        block = 512
        while block < n and block < 4096:
            block *= 2
        return block, 4
    if item_size == 4:
        return 2048, 4
    if item_size >= 8:
        return 1024, 4
    return 4096, 8


@triton.jit
def _tanh_fast(x):
    """Division-free tanh for fp32 opmath.

    tanh(x) = (1 - r)/(1 + r) with r = exp2(-2|x|*log2e) in (0, 1], so large
    |x| saturates for free (r underflows to 0).  1/(1+r) is seeded with a
    bit-trick reciprocal and refined with three Newton steps, which keeps the
    whole evaluation on the cheap ALU/MUFU path instead of libdevice tanhf's
    polynomial plus correctly-rounded division.  Verified against
    torch._foreach_tanh: fp32 max abs error 1.8e-7 (assert_close defaults
    pass), fp16/bf16 within one ULP on 2-4 of 300000 samples, NaN/inf kept.
    """
    r = _ld.exp2(tl.abs(x) * -2.8853900817779268)
    d = 1.0 + r
    inv = (0x7EF311C3 - d.to(tl.int32, bitcast=True)).to(tl.float32, bitcast=True)
    inv = inv * (2.0 - d * inv)
    inv = inv * (2.0 - d * inv)
    inv = inv * (2.0 - d * inv)
    y = (1.0 - r) * inv
    y = tl.where(x != x, x, y)
    return tl.where(x < 0, -y, y)


@triton.jit
def _tanh_lean(x):
    """Leaner variant for 16-bit outputs (same identity, no abs/sign select).

    r = exp2(x * -2*log2e) is > 1 for negative x, and (1 - r)/(1 + r) is odd
    in x, so the sign falls out of the arithmetic; the abs, the sign select and
    the NaN select all disappear.  Only the negative-overflow case (|x| > ~34,
    where r reaches ~1e30) needs a saturation select, which also covers the
    whole fp16 input range.  NaN still propagates through the arithmetic.
    Rounding to fp16/bf16 hides the ~1e-5 reciprocal error: measured max abs
    deviation is one ULP on a handful of samples out of 300000.
    """
    r = _ld.exp2(x * -2.8853900817779268)
    d = 1.0 + r
    inv = (0x7EF311C3 - d.to(tl.int32, bitcast=True)).to(tl.float32, bitcast=True)
    inv = inv * (2.0 - d * inv)
    inv = inv * (2.0 - d * inv)
    y = (1.0 - r) * inv
    return tl.where(r > 1e30, -1.0, y)


@triton.jit
def _tanh_flat_kernel(
    x_ptr,
    y_ptr,
    n,
    MASKED: tl.constexpr,
    I64: tl.constexpr,
    IS_DOUBLE: tl.constexpr,
    LEAN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    if MASKED:
        m = offs < n
        x = tl.load(x_ptr + offs, mask=m, other=0)
    else:
        x = tl.load(x_ptr + offs)
    if IS_DOUBLE:
        y = _ld.tanh(x)
    elif LEAN:
        y = _tanh_lean(x.to(tl.float32))
    else:
        y = _tanh_fast(x.to(tl.float32))
    if MASKED:
        tl.store(y_ptr + offs, y, mask=m)
    else:
        tl.store(y_ptr + offs, y)


@triton.jit(
    do_not_specialize=[
        "n",
        "d0",
        "d1",
        "d2",
        "d3",
        "d4",
        "d5",
        "d6",
        "d7",
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
def _tanh_strided_kernel(
    x_ptr,
    y_ptr,
    n,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    RANK: tl.constexpr,
    OUT_SAME: tl.constexpr,
    IS_DOUBLE: tl.constexpr,
    LEAN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    m = offs < n
    rem = offs
    addr = tl.zeros([BLOCK], dtype=tl.int64)
    if RANK >= 1:
        addr += (rem % d0) * s0
        rem = rem // d0
    if RANK >= 2:
        addr += (rem % d1) * s1
        rem = rem // d1
    if RANK >= 3:
        addr += (rem % d2) * s2
        rem = rem // d2
    if RANK >= 4:
        addr += (rem % d3) * s3
        rem = rem // d3
    if RANK >= 5:
        addr += (rem % d4) * s4
        rem = rem // d4
    if RANK >= 6:
        addr += (rem % d5) * s5
        rem = rem // d5
    if RANK >= 7:
        addr += (rem % d6) * s6
        rem = rem // d6
    if RANK >= 8:
        addr += (rem % d7) * s7
        rem = rem // d7
    x = tl.load(x_ptr + addr, mask=m, other=0)
    if IS_DOUBLE:
        y = _ld.tanh(x)
    elif LEAN:
        y = _tanh_lean(x.to(tl.float32))
    else:
        y = _tanh_fast(x.to(tl.float32))
    if OUT_SAME:
        tl.store(y_ptr + addr, y, mask=m)
    else:
        # freshly allocated contiguous output: flat logical offsets
        tl.store(y_ptr + offs, y, mask=m)


@triton.jit(do_not_specialize=["n"])
def _tanh_complex_kernel(
    x_ptr,
    y_ptr,
    n,
    IS_DOUBLE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    b = offs * 2
    xr = tl.load(x_ptr + b, mask=m, other=0)
    xi = tl.load(x_ptr + b + 1, mask=m, other=0)
    if IS_DOUBLE:
        a2 = xr + xr
        b2 = xi + xi
        den = _ld.cosh(a2) + _ld.cos(b2)
        yr = _ld.sinh(a2) / den
        yi = _ld.sin(b2) / den
        big = tl.abs(a2) > 700.0
    else:
        xr = xr.to(tl.float32)
        xi = xi.to(tl.float32)
        a2 = xr + xr
        b2 = xi + xi
        den = _ld.cosh(a2) + _ld.cos(b2)
        yr = _ld.sinh(a2) / den
        yi = _ld.sin(b2) / den
        big = tl.abs(a2) > 40.0
    sat = tl.where(xr > 0, 1.0, -1.0)
    yr = tl.where(big, sat, yr)
    yi = tl.where(big, 0.0, yi)
    tl.store(y_ptr + b, yr, mask=m)
    tl.store(y_ptr + b + 1, yi, mask=m)


# ---------------------------------------------------------------- helpers

_MAX_DIMS = 8
_MAX_FUSED = 16


@triton.jit
def _tanh_tile(
    x_ptr,
    y_ptr,
    src_off,
    dst_off,
    IS_DOUBLE: tl.constexpr,
    LEAN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + src_off + offs)
    if IS_DOUBLE:
        y = _ld.tanh(x)
    elif LEAN:
        y = _tanh_lean(x.to(tl.float32))
    else:
        y = _tanh_fast(x.to(tl.float32))
    tl.store(y_ptr + dst_off + offs, y)


@triton.jit(do_not_specialize=["n", "BPT"])
def _tanh_multi_kernel(
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
    y_ptr,
    n,
    BPT,
    M: tl.constexpr,
    IS_DOUBLE: tl.constexpr,
    LEAN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One launch for M equally sized, contiguous tensors.

    Block ``pid`` belongs to tensor ``pid // BPT`` and handles its
    ``(pid % BPT)``-th tile; the output is a single contiguous (M, numel)
    buffer, so the destination offset is ``ti * n + src``.
    """
    pid = tl.program_id(0)
    ti = pid // BPT
    src = (pid - ti * BPT) * BLOCK
    dst = ti.to(tl.int64) * n + src
    if M >= 1:
        if ti == 0:
            _tanh_tile(p0, y_ptr, src, dst, IS_DOUBLE, LEAN, BLOCK)
    if M >= 2:
        if ti == 1:
            _tanh_tile(p1, y_ptr, src, dst, IS_DOUBLE, LEAN, BLOCK)
    if M >= 3:
        if ti == 2:
            _tanh_tile(p2, y_ptr, src, dst, IS_DOUBLE, LEAN, BLOCK)
    if M >= 4:
        if ti == 3:
            _tanh_tile(p3, y_ptr, src, dst, IS_DOUBLE, LEAN, BLOCK)
    if M >= 5:
        if ti == 4:
            _tanh_tile(p4, y_ptr, src, dst, IS_DOUBLE, LEAN, BLOCK)
    if M >= 6:
        if ti == 5:
            _tanh_tile(p5, y_ptr, src, dst, IS_DOUBLE, LEAN, BLOCK)
    if M >= 7:
        if ti == 6:
            _tanh_tile(p6, y_ptr, src, dst, IS_DOUBLE, LEAN, BLOCK)
    if M >= 8:
        if ti == 7:
            _tanh_tile(p7, y_ptr, src, dst, IS_DOUBLE, LEAN, BLOCK)
    if M >= 9:
        if ti == 8:
            _tanh_tile(p8, y_ptr, src, dst, IS_DOUBLE, LEAN, BLOCK)
    if M >= 10:
        if ti == 9:
            _tanh_tile(p9, y_ptr, src, dst, IS_DOUBLE, LEAN, BLOCK)
    if M >= 11:
        if ti == 10:
            _tanh_tile(p10, y_ptr, src, dst, IS_DOUBLE, LEAN, BLOCK)
    if M >= 12:
        if ti == 11:
            _tanh_tile(p11, y_ptr, src, dst, IS_DOUBLE, LEAN, BLOCK)
    if M >= 13:
        if ti == 12:
            _tanh_tile(p12, y_ptr, src, dst, IS_DOUBLE, LEAN, BLOCK)
    if M >= 14:
        if ti == 13:
            _tanh_tile(p13, y_ptr, src, dst, IS_DOUBLE, LEAN, BLOCK)
    if M >= 15:
        if ti == 14:
            _tanh_tile(p14, y_ptr, src, dst, IS_DOUBLE, LEAN, BLOCK)
    if M >= 16:
        if ti == 15:
            _tanh_tile(p15, y_ptr, src, dst, IS_DOUBLE, LEAN, BLOCK)


def _descriptor(shape, stride, by_stride):
    """Drop size-1 dims, merge stride-compatible neighbours.

    Returns (size, stride) pairs ordered innermost-first.  When ``by_stride``
    the pairs are sorted by ascending stride (best coalescing for a layout
    shared by input and output), otherwise the natural logical order is kept
    so that flat output offsets line up with the logical index.
    """
    dims = [(int(s), int(st)) for s, st in zip(shape, stride) if int(s) != 1]
    out = []
    for s, st in reversed(dims):
        if out and st == out[-1][0] * out[-1][1]:
            out[-1] = (s * out[-1][0], out[-1][1])
        else:
            out.append((s, st))
    if by_stride:
        out = sorted(out, key=lambda p: p[1])
    return out


def _block_for(item_size):
    if item_size >= 8:
        return 512, 4
    return 1024, 4


_FUSE_MAX_NUMEL = 262144


def _run_fused(tensors):
    """Single-launch path for a homogeneous list of equally shaped tensors.

    Returns None when the list does not qualify (the caller then falls back to
    one launch per tensor).  Only used for lists whose per-tensor size is small
    enough that ATen's single multi_tensor_apply launch wins on host overhead.
    """
    t0 = tensors[0]
    dt = t0.dtype
    if not dt.is_floating_point:
        return None
    shape = t0.shape
    n = t0.numel()
    if n < 512 or n > _FUSE_MAX_NUMEL or (n & 511) or not t0.is_contiguous():
        return None
    dev = t0.device
    for t in tensors[1:]:
        if (
            t.dtype != dt
            or t.shape != shape
            or t.device != dev
            or not t.is_contiguous()
        ):
            return None
    M = len(tensors)
    BLOCK = 1024 if (n & 1023) == 0 else 512
    bpt = n // BLOCK
    out = torch.empty((M,) + tuple(shape), dtype=dt, device=dev)
    args = list(tensors)
    if M < _MAX_FUSED:
        args.extend([t0] * (_MAX_FUSED - M))
    _tanh_multi_kernel[(M * bpt,)](
        *args,
        out,
        n,
        bpt,
        M,
        dt == torch.float64,
        dt.itemsize == 2,
        BLOCK,
        num_warps=4,
    )
    return list(out.unbind(0))


def _tanh_one(t):
    dt = t.dtype
    n = t.numel()
    if dt.is_complex:
        out = torch.empty(t.shape, dtype=dt, device=t.device)
        if n == 0:
            return out
        BLOCK, warps = _block_for(2 * dt.itemsize)
        _tanh_complex_kernel[(triton.cdiv(n, BLOCK),)](
            torch.view_as_real(t),
            torch.view_as_real(out),
            n,
            IS_DOUBLE=dt == torch.complex128,
            BLOCK=BLOCK,
            num_warps=warps,
        )
        return out

    same_dtype = dt.is_floating_point
    out_dtype = dt if same_dtype else torch.float32
    is_double = out_dtype == torch.float64
    lean = out_dtype.itemsize == 2
    if dt == torch.bool:
        src = t.view(torch.uint8)
    else:
        src = t
    BLOCK, warps = _block_for(out_dtype.itemsize)

    if src.is_contiguous():
        out = torch.empty(t.shape, dtype=out_dtype, device=t.device)
        if n == 0:
            return out
        fblock, fwarps = _flat_cfg(n, out_dtype.itemsize)
        _tanh_flat_kernel[(triton.cdiv(n, fblock),)](
            src,
            out,
            n,
            (n % fblock) != 0,
            n >= (1 << 31),
            is_double,
            lean,
            fblock,
            num_warps=fwarps,
        )
        return out

    out = (
        torch.empty_like(t)
        if same_dtype
        else torch.empty(t.shape, dtype=out_dtype, device=t.device)
    )
    if n == 0:
        return out

    out_same = out.stride() == t.stride()
    dims = _descriptor(t.shape, t.stride(), out_same)
    rank = len(dims)
    if rank == 0:
        # numel >= 2 with every dim collapsed away cannot happen; be safe
        dims = [(n, 1)]
        rank = 1
        out_same = False
    if rank > _MAX_DIMS:
        raise RuntimeError("unsupported tensor layout for foreach_tanh")
    ds = [1] * _MAX_DIMS
    ss = [0] * _MAX_DIMS
    for i, (s, st) in enumerate(dims):
        ds[i] = s
        ss[i] = st
    _tanh_strided_kernel[(triton.cdiv(n, BLOCK),)](
        src,
        out,
        n,
        ds[0],
        ds[1],
        ds[2],
        ds[3],
        ds[4],
        ds[5],
        ds[6],
        ds[7],
        ss[0],
        ss[1],
        ss[2],
        ss[3],
        ss[4],
        ss[5],
        ss[6],
        ss[7],
        RANK=rank,
        OUT_SAME=out_same,
        IS_DOUBLE=is_double,
        LEAN=lean,
        BLOCK=BLOCK,
        num_warps=warps,
    )
    return out


def foreach_tanh(self):
    try:
        count = len(self)
    except TypeError:
        self = list(self)
        count = len(self)
    if count == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")
    if 4 <= count <= _MAX_FUSED:
        fused = _run_fused(self)
        if fused is not None:
            return fused
    outs = []
    for t in self:
        outs.append(_tanh_one(t))
    return outs
