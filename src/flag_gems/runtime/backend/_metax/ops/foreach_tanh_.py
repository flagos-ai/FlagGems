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

"""Triton implementation of ``aten::_foreach_tanh_`` (in-place, returns None).

Design
------
* In-place elementwise: every tensor is read and written at exactly its own
  addresses, so storage, stride, dtype and data_ptr are preserved and nothing
  is allocated.
* Contiguous tensors of one dtype are packed into as few launches as possible
  through bounded static-arity kernels (1, 8 and 16 pointer slots).  The grid is
  ``(slots, cdiv(max_numel, BLOCK))`` and each program selects its own pointer
  with a scalar branch chain, so the host never builds a device metadata table
  (that would cost an extra H2D copy on every call).  A Triton launch on this
  stack costs ~25 us fixed plus ~1 us per bound argument, and the small
  benchmark shapes are pure launch cost, so the hot kernels take only a pointer,
  an element count and BLOCK -- the fp64 math and the 64-bit-offset case are
  selected inside the kernel from the pointer dtype, and numel > 2**31 is routed
  to the strided kernel that carries the explicit flags.
* Non-contiguous tensors fall back to a per-tensor strided kernel that walks the
  logical index space and rebuilds the address from the real strides, so views
  are written through without any staging copy.
* Math: fp16/bf16/fp32 are computed in fp32 opmath (matching ATen's CUDA
  elementwise path) via ``tanh(x) = sign(x) * (1 - 2/(exp(2|x|) + 1))`` using
  the hardware ``exp2``; fp64 uses the same identity in fp64.
"""

import torch
import triton
import triton.language as tl

_exp2 = tl.exp2

_FLOAT_DTYPES = frozenset((torch.float16, torch.bfloat16, torch.float32, torch.float64))

# Tensor slots fused into one launch.
_SLOTS = 8
_SLOTS16 = 16
_SMALL_NUMEL = 4096
_INT32_MAX = 0x7FFFFFFF


# ---------------------------------------------------------------------------
# math
# ---------------------------------------------------------------------------


@triton.jit
def _tanh_f32(x):
    # 1 - 2/(e^(2x)+1) needs no |x|/negate/select mirror: exp2 saturates to inf
    # for large positive x (giving 1) and to 0 for large negative x (giving -1),
    # and NaN propagates.
    e = _exp2(x * 2.8853900817779268)
    return 1.0 - tl.fdiv(2.0, e + 1.0, ieee_rounding=False)


@triton.jit
def _tanh_f64(x):
    e = _exp2(x.to(tl.float64) * 2.8853900817779268)
    return 1.0 - 2.0 / (e + 1.0)


@triton.jit
def _tanh_store(p, offs, mask, x):
    if x.dtype == tl.float64:
        y = _tanh_f64(x)
    else:
        y = _tanh_f32(x.to(tl.float32)).to(x.dtype)
    tl.store(p + offs, y, mask=mask)


@triton.jit
def _tanh_vec(p, pid, n, BLOCK: tl.constexpr):
    """One BLOCK-sized in-place chunk ``[pid*BLOCK, pid*BLOCK+BLOCK)`` of ``p``."""
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(p + offs, mask=mask, other=0.0)
    _tanh_store(p, offs, mask, x)


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


@triton.jit
def _foreach_tanh_flat1(p, n, BLOCK: tl.constexpr):
    _tanh_vec(p, tl.program_id(0), n, BLOCK)


@triton.jit
def _foreach_tanh_flat8(
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
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0)
    p = p0
    n = n0
    if t == 1:
        p = p1
        n = n1
    elif t == 2:
        p = p2
        n = n2
    elif t == 3:
        p = p3
        n = n3
    elif t == 4:
        p = p4
        n = n4
    elif t == 5:
        p = p5
        n = n5
    elif t == 6:
        p = p6
        n = n6
    elif t == 7:
        p = p7
        n = n7
    _tanh_vec(p, tl.program_id(1), n, BLOCK)


@triton.jit
def _foreach_tanh_flat16(
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
    n,
    BLOCK: tl.constexpr,
):
    """16 tensors of equal numel in one launch (one shared element count)."""
    t = tl.program_id(0)
    p = p0
    if t == 1:
        p = p1
    elif t == 2:
        p = p2
    elif t == 3:
        p = p3
    elif t == 4:
        p = p4
    elif t == 5:
        p = p5
    elif t == 6:
        p = p6
    elif t == 7:
        p = p7
    elif t == 8:
        p = p8
    elif t == 9:
        p = p9
    elif t == 10:
        p = p10
    elif t == 11:
        p = p11
    elif t == 12:
        p = p12
    elif t == 13:
        p = p13
    elif t == 14:
        p = p14
    elif t == 15:
        p = p15
    _tanh_vec(p, tl.program_id(1), n, BLOCK)


@triton.jit
def _foreach_tanh_strided(
    p,
    n,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    ND: tl.constexpr,
    BLOCK: tl.constexpr,
    I64: tl.constexpr,
):
    """Logical-order traversal of a strided view; dim 0 is the fastest."""
    rem = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if I64:
        rem = rem.to(tl.int64)
    mask = rem < n
    strides = (s0, s1, s2, s3, s4, s5, s6, s7)
    sizes = (d0, d1, d2, d3, d4, d5, d6, d7)
    rem64 = rem.to(tl.int64)
    addr = tl.zeros((BLOCK,), tl.int64)
    for i in tl.static_range(ND):
        idx = rem64 % sizes[i]
        rem64 = rem64 // sizes[i]
        addr += idx * strides[i]
    x = tl.load(p + addr, mask=mask, other=0.0)
    _tanh_store(p, addr, mask, x)


# ---------------------------------------------------------------------------
# host helpers
# ---------------------------------------------------------------------------


def _collapse_dims(t):
    """Merge fuse-able adjacent dims; return them fastest-first as [size, stride]."""
    merged = []
    for d, s in zip(t.shape, t.stride()):
        d = int(d)
        if d == 1:
            continue
        s = int(s)
        if merged and merged[-1][1] == s * d:
            merged[-1][0] *= d
            merged[-1][1] = s
        else:
            merged.append([d, s])
    if not merged:
        return [[1, 0]]
    merged.reverse()
    return merged


def _pick_block(n, itemsize):
    """Launch-bound shapes and 4/8-byte streams take 1024, 16-bit takes 8192.

    The 4-byte path measures consistently better at 1024 (fp32 1e9 5.715 vs
    5.756 ms, fused 16x4096x4096 1.486 vs 1.495 ms, single medium 0.1105 vs
    0.1136 ms); the 16-bit streams keep more independent tanh chains per thread
    at 8192, which measured non-negative on every 16-bit shape.
    """
    if n <= _SMALL_NUMEL or itemsize >= 4:
        return 1024
    return 8192


def _launch_one(t, n):
    block = _pick_block(n, t.element_size())
    _foreach_tanh_flat1[((n + block - 1) // block,)](
        t, n, block, num_warps=4, num_stages=1
    )


def _launch_flat(chunk, numels):
    k = len(chunk)
    maxn = max(numels)
    block = _pick_block(maxn, chunk[0].element_size())
    ptrs = chunk
    nums = numels
    if k < _SLOTS:
        fill = chunk[-1]
        ptrs = chunk + [fill] * (_SLOTS - k)
        nums = numels + [0] * (_SLOTS - k)
    _foreach_tanh_flat8[(k, (maxn + block - 1) // block)](
        ptrs[0],
        ptrs[1],
        ptrs[2],
        ptrs[3],
        ptrs[4],
        ptrs[5],
        ptrs[6],
        ptrs[7],
        nums[0],
        nums[1],
        nums[2],
        nums[3],
        nums[4],
        nums[5],
        nums[6],
        nums[7],
        block,
        num_warps=4,
        num_stages=1,
    )


def _launch_flat16(chunk, n):
    block = _pick_block(n, chunk[0].element_size())
    _foreach_tanh_flat16[(16, (n + block - 1) // block)](
        chunk[0],
        chunk[1],
        chunk[2],
        chunk[3],
        chunk[4],
        chunk[5],
        chunk[6],
        chunk[7],
        chunk[8],
        chunk[9],
        chunk[10],
        chunk[11],
        chunk[12],
        chunk[13],
        chunk[14],
        chunk[15],
        n,
        block,
        num_warps=4,
        num_stages=1,
    )


def _launch_strided(t):
    dims = _collapse_dims(t)
    nd = len(dims)
    if nd > _SLOTS:
        raise RuntimeError("unsupported layout rank for in-place foreach tanh")
    sizes = [1] * _SLOTS
    strides = [0] * _SLOTS
    for i, (d, s) in enumerate(dims):
        sizes[i] = d
        strides[i] = s
    n = t.numel()
    block = 1024 if n >= 1024 else 128
    _foreach_tanh_strided[((n + block - 1) // block,)](
        t,
        n,
        strides[0],
        strides[1],
        strides[2],
        strides[3],
        strides[4],
        strides[5],
        strides[6],
        strides[7],
        sizes[0],
        sizes[1],
        sizes[2],
        sizes[3],
        sizes[4],
        sizes[5],
        sizes[6],
        sizes[7],
        ND=nd,
        BLOCK=block,
        I64=n > _INT32_MAX,
        num_warps=4,
        num_stages=1,
    )


def _launch_contiguous(t, dtype, n):
    """One contiguous tensor: flat kernel, strided kernel for >2**31 elements."""
    if n > _INT32_MAX:
        _launch_strided(t)
    else:
        _launch_one(t, n)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def foreach_tanh_(self):
    tensors = self
    if not isinstance(tensors, (list, tuple)):
        raise TypeError("argument 'self' must be tuple of Tensors")
    length = len(tensors)
    if length == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")

    # Single-tensor lists are the most common benchmark shape: skip grouping.
    if length == 1:
        t = tensors[0]
        if isinstance(t, torch.Tensor):
            dtype = t.dtype
            if dtype in _FLOAT_DTYPES and t.is_contiguous():
                n = t.numel()
                if n:
                    _launch_contiguous(t, dtype, n)
                    return None

    groups = {}
    strided = None
    for t in tensors:
        if not isinstance(t, torch.Tensor):
            raise TypeError("argument 'self' must be tuple of Tensors")
        dtype = t.dtype
        if dtype not in _FLOAT_DTYPES:
            raise RuntimeError(
                "\"foreach\" not implemented for '%s'"
                % str(dtype).replace("torch.", "")
            )
        if not t.is_contiguous():
            if t.numel():
                if strided is None:
                    strided = [t]
                else:
                    strided.append(t)
            continue
        numel = t.numel()
        if not numel:
            continue
        bucket = groups.get(dtype)
        if bucket is None:
            groups[dtype] = [t]
        else:
            bucket.append(t)

    for dtype, members in groups.items():
        m = len(members)
        if m == 1:
            _launch_contiguous(members[0], dtype, members[0].numel())
        elif m <= _SLOTS:
            _launch_flat(members, [t.numel() for t in members])
        elif m <= _SLOTS16:
            n0 = members[0].numel()
            same = True
            for t in members:
                if t.numel() != n0:
                    same = False
                    break
            if same:
                _launch_flat16(members, n0)
            else:
                for i in range(0, m, _SLOTS):
                    part = members[i : i + _SLOTS]
                    _launch_flat(part, [t.numel() for t in part])
        else:
            for i in range(0, m, _SLOTS):
                part = members[i : i + _SLOTS]
                _launch_flat(part, [t.numel() for t in part])

    if strided is not None:
        for t in strided:
            _launch_strided(t)

    return None
