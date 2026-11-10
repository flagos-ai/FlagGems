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

"""In-place foreach arcsine (``aten::_foreach_asin_``) on Triton.

Semantics: for every tensor ``T`` in the input TensorList, overwrite ``T`` in
place with ``asin(T)`` element by element.  Storage, strides, dtype and the
caller's tensor objects are preserved; the call returns ``None``.

Launch strategy.  On this target a Triton launch costs ~17 us of host time while
a 64x64 kernel runs in microseconds, so the short-list regime is decided by
launch count and by how many arguments each launch carries:

  * a single tensor uses a dedicated 2-argument kernel;
  * a same-dtype group of up to 16 flat-addressable tensors is fused into ONE
    launch.  When every tensor has the same element count that count becomes a
    ``constexpr``, so the launch carries pointer arguments only; otherwise
    per-slot runtime counts drive a chunk-range prologue.  No device metadata
    table and no host-to-device pointer copy is involved;
  * above ``_FUSE_MAX_ELEMS`` the fused prologue costs more than the launches it
    saves, so large groups fall back to one lean launch per tensor, where the
    asynchronous launches overlap device execution anyway;
  * gappy views (``x[:, ::2]``) use a stride-decomposed kernel so no element
    outside the view is written.

Traversal inside a tensor is flat memory order, which is valid for contiguous
and for non-overlapping-and-dense tensors (transposed dense views included) and
is fully coalesced.
"""

import torch
import triton
import triton.language as tl

try:  # Triton >= 3 dispatches libdevice per backend
    from triton.language.extra import libdevice
except Exception:  # pragma: no cover - older layout
    from triton.language.extra.cuda import libdevice

_BLOCK = 1024
# Streaming tensors use a wider tile: the compile probe measured 58 registers
# (no spills) at BLOCK 2048 / 4 warps versus 46 at 1024, i.e. 16 elements per
# thread instead of 8 at a similar warp residency, which raises the bytes in
# flight per SM.  Short lists keep the narrow tile so the fused launch stays
# cheap.
_BLOCK_BIG = 2048
_BIG_ELEMS = 1 << 20
_KMAX = 16
_MAX_DIMS = 8
# Measured: 16 x 16.7M fp16 elements take 1.47 ms fused vs 1.06 ms per tensor
# (48 registers, no spills -- the loss is the per-CTA chunk-range prologue), so
# fusion is limited to groups small enough to be launch bound.
_FUSE_MAX_ELEMS = 1 << 24

_FLOAT_DTYPES = frozenset((torch.float16, torch.bfloat16, torch.float32, torch.float64))

_ATEN_TYPE_NAMES = {
    torch.uint8: "Byte",
    torch.int8: "Char",
    torch.int16: "Short",
    torch.int32: "Int",
    torch.int64: "Long",
    torch.bool: "Bool",
    torch.complex64: "ComplexFloat",
    torch.complex128: "ComplexDouble",
}


def _raise_bad_dtype(dtype):
    name = _ATEN_TYPE_NAMES.get(dtype)
    if name is None:
        name = str(dtype).replace("torch.", "").capitalize()
    raise RuntimeError(
        "result type Float can't be cast to the desired output type " + name
    )


@triton.jit
def _asin_body(ptr, base, n, IS64: tl.constexpr, BLOCK: tl.constexpr):
    idx = base + tl.arange(0, BLOCK)
    mask = idx < n
    x = tl.load(ptr + idx, mask=mask, other=0.0)
    if IS64:
        y = libdevice.asin(x)
    else:
        y = libdevice.asin(x.to(tl.float32)).to(x.dtype)
    tl.store(ptr + idx, y, mask=mask)


@triton.jit
def _asin_flat_kernel(x_ptr, n, IS64: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    _asin_body(x_ptr, pid * BLOCK, n, IS64, BLOCK)


@triton.jit
def _asin_multi_kernel(
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
    IS64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One launch for up to 16 same-dtype tensors with individual counts."""
    pid = tl.program_id(0)
    c1 = tl.cdiv(n0, BLOCK)
    c2 = c1 + tl.cdiv(n1, BLOCK)
    c3 = c2 + tl.cdiv(n2, BLOCK)
    c4 = c3 + tl.cdiv(n3, BLOCK)
    c5 = c4 + tl.cdiv(n4, BLOCK)
    c6 = c5 + tl.cdiv(n5, BLOCK)
    c7 = c6 + tl.cdiv(n6, BLOCK)
    c8 = c7 + tl.cdiv(n7, BLOCK)
    c9 = c8 + tl.cdiv(n8, BLOCK)
    c10 = c9 + tl.cdiv(n9, BLOCK)
    c11 = c10 + tl.cdiv(n10, BLOCK)
    c12 = c11 + tl.cdiv(n11, BLOCK)
    c13 = c12 + tl.cdiv(n12, BLOCK)
    c14 = c13 + tl.cdiv(n13, BLOCK)
    c15 = c14 + tl.cdiv(n14, BLOCK)
    if pid < c1:
        _asin_body(p0, pid * BLOCK, n0, IS64, BLOCK)
    if (pid >= c1) & (pid < c2):
        _asin_body(p1, (pid - c1) * BLOCK, n1, IS64, BLOCK)
    if (pid >= c2) & (pid < c3):
        _asin_body(p2, (pid - c2) * BLOCK, n2, IS64, BLOCK)
    if (pid >= c3) & (pid < c4):
        _asin_body(p3, (pid - c3) * BLOCK, n3, IS64, BLOCK)
    if (pid >= c4) & (pid < c5):
        _asin_body(p4, (pid - c4) * BLOCK, n4, IS64, BLOCK)
    if (pid >= c5) & (pid < c6):
        _asin_body(p5, (pid - c5) * BLOCK, n5, IS64, BLOCK)
    if (pid >= c6) & (pid < c7):
        _asin_body(p6, (pid - c6) * BLOCK, n6, IS64, BLOCK)
    if (pid >= c7) & (pid < c8):
        _asin_body(p7, (pid - c7) * BLOCK, n7, IS64, BLOCK)
    if (pid >= c8) & (pid < c9):
        _asin_body(p8, (pid - c8) * BLOCK, n8, IS64, BLOCK)
    if (pid >= c9) & (pid < c10):
        _asin_body(p9, (pid - c9) * BLOCK, n9, IS64, BLOCK)
    if (pid >= c10) & (pid < c11):
        _asin_body(p10, (pid - c10) * BLOCK, n10, IS64, BLOCK)
    if (pid >= c11) & (pid < c12):
        _asin_body(p11, (pid - c11) * BLOCK, n11, IS64, BLOCK)
    if (pid >= c12) & (pid < c13):
        _asin_body(p12, (pid - c12) * BLOCK, n12, IS64, BLOCK)
    if (pid >= c13) & (pid < c14):
        _asin_body(p13, (pid - c13) * BLOCK, n13, IS64, BLOCK)
    if (pid >= c14) & (pid < c15):
        _asin_body(p14, (pid - c14) * BLOCK, n14, IS64, BLOCK)
    if pid >= c15:
        _asin_body(p15, (pid - c15) * BLOCK, n15, IS64, BLOCK)


@triton.jit
def _asin_multi_uniform_kernel(
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
    N: tl.constexpr,
    IS64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One launch for up to 16 same-dtype tensors sharing one element count.

    ``N`` is a ``constexpr``, so the launch carries pointer arguments only (no
    runtime sizes) and every chunk boundary folds away.  Programs beyond
    ``K * cdiv(N, BLOCK)`` never exist, so unused slots stay dead code.
    """
    pid = tl.program_id(0)
    c1 = (N + BLOCK - 1) // BLOCK
    c2 = 2 * c1
    c3 = 3 * c1
    c4 = 4 * c1
    c5 = 5 * c1
    c6 = 6 * c1
    c7 = 7 * c1
    c8 = 8 * c1
    c9 = 9 * c1
    c10 = 10 * c1
    c11 = 11 * c1
    c12 = 12 * c1
    c13 = 13 * c1
    c14 = 14 * c1
    c15 = 15 * c1
    if pid < c1:
        _asin_body(p0, pid * BLOCK, N, IS64, BLOCK)
    if (pid >= c1) & (pid < c2):
        _asin_body(p1, (pid - c1) * BLOCK, N, IS64, BLOCK)
    if (pid >= c2) & (pid < c3):
        _asin_body(p2, (pid - c2) * BLOCK, N, IS64, BLOCK)
    if (pid >= c3) & (pid < c4):
        _asin_body(p3, (pid - c3) * BLOCK, N, IS64, BLOCK)
    if (pid >= c4) & (pid < c5):
        _asin_body(p4, (pid - c4) * BLOCK, N, IS64, BLOCK)
    if (pid >= c5) & (pid < c6):
        _asin_body(p5, (pid - c5) * BLOCK, N, IS64, BLOCK)
    if (pid >= c6) & (pid < c7):
        _asin_body(p6, (pid - c6) * BLOCK, N, IS64, BLOCK)
    if (pid >= c7) & (pid < c8):
        _asin_body(p7, (pid - c7) * BLOCK, N, IS64, BLOCK)
    if (pid >= c8) & (pid < c9):
        _asin_body(p8, (pid - c8) * BLOCK, N, IS64, BLOCK)
    if (pid >= c9) & (pid < c10):
        _asin_body(p9, (pid - c9) * BLOCK, N, IS64, BLOCK)
    if (pid >= c10) & (pid < c11):
        _asin_body(p10, (pid - c10) * BLOCK, N, IS64, BLOCK)
    if (pid >= c11) & (pid < c12):
        _asin_body(p11, (pid - c11) * BLOCK, N, IS64, BLOCK)
    if (pid >= c12) & (pid < c13):
        _asin_body(p12, (pid - c12) * BLOCK, N, IS64, BLOCK)
    if (pid >= c13) & (pid < c14):
        _asin_body(p13, (pid - c13) * BLOCK, N, IS64, BLOCK)
    if (pid >= c14) & (pid < c15):
        _asin_body(p14, (pid - c14) * BLOCK, N, IS64, BLOCK)
    if pid >= c15:
        _asin_body(p15, (pid - c15) * BLOCK, N, IS64, BLOCK)


@triton.jit
def _asin_strided_kernel(
    x_ptr,
    n,
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
    IS64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    r = offs.to(tl.int64)
    i7 = r % s7
    r = r // s7
    i6 = r % s6
    r = r // s6
    i5 = r % s5
    r = r // s5
    i4 = r % s4
    r = r // s4
    i3 = r % s3
    r = r // s3
    i2 = r % s2
    r = r // s2
    i1 = r % s1
    i0 = r // s1
    off = i0 * t0 + i1 * t1 + i2 * t2 + i3 * t3 + i4 * t4 + i5 * t5 + i6 * t6 + i7 * t7
    x = tl.load(x_ptr + off, mask=mask, other=0.0)
    if IS64:
        y = libdevice.asin(x)
    else:
        y = libdevice.asin(x.to(tl.float32)).to(x.dtype)
    tl.store(x_ptr + off, y, mask=mask)


def _is_flat_dense(t):
    """True when ``[data_ptr, data_ptr + numel)`` is exactly this tensor."""
    if t.numel() <= 1:
        return True
    expected = 1
    for stride, size in sorted(zip(t.stride(), t.shape)):
        if size == 1:
            continue
        if stride != expected:
            return False
        expected *= size
    return True


def _pad_dims(t):
    shape = list(t.shape)
    stride = list(t.stride())
    extra = _MAX_DIMS - len(shape)
    if extra > 0:
        shape.extend([1] * extra)
        stride.extend([0] * extra)
    return shape, stride


def _asin_strided(t, is64):
    n = t.numel()
    if torch._debug_has_internal_overlap(t) == 1:
        raise RuntimeError(
            "unsupported operation: more than one element of the written-to "
            "tensor refers to a single memory location. Please clone() the "
            "tensor before performing the operation."
        )
    shape, stride = _pad_dims(t)
    grid = ((n + _BLOCK - 1) // _BLOCK,)
    _asin_strided_kernel[grid](t, n, *shape, *stride, is64, _BLOCK)


def _asin_one(t):
    dt = t.dtype
    if dt not in _FLOAT_DTYPES:
        _raise_bad_dtype(dt)
    n = t.numel()
    if n == 0:
        return
    is64 = dt is torch.float64
    if t.is_contiguous() or _is_flat_dense(t):
        block = _BLOCK_BIG if n >= _BIG_ELEMS else _BLOCK
        _asin_flat_kernel[((n + block - 1) // block,)](t, n, is64, block)
        return
    _asin_strided(t, is64)


def _asin_many(ts, nums, is64):
    """Same-dtype, flat-addressable group: fused when small, else per tensor."""
    count = len(ts)
    if count == 1:
        n = nums[0]
        b = _BLOCK_BIG if n >= _BIG_ELEMS else _BLOCK
        _asin_flat_kernel[((n + b - 1) // b,)](ts[0], n, is64, b)
        return
    block = _BLOCK
    if count > _KMAX:
        for start in range(0, count, _KMAX):
            stop = start + _KMAX
            _asin_many(ts[start:stop], nums[start:stop], is64)
        return
    chunks = 0
    for n in nums:
        chunks += (n + block - 1) // block
    if chunks * block > _FUSE_MAX_ELEMS:
        for t, n in zip(ts, nums):
            b = _BLOCK_BIG if n >= _BIG_ELEMS else _BLOCK
            _asin_flat_kernel[((n + b - 1) // b,)](t, n, is64, b)
        return
    ptrs = list(ts)
    pad = _KMAX - count
    if pad:
        ptrs.extend([ptrs[0]] * pad)
    first = nums[0]
    for n in nums:
        if n != first:
            if pad:
                nums = list(nums)
                nums.extend([0] * pad)
            _asin_multi_kernel[(chunks,)](*ptrs, *nums, is64, block)
            return
    c = (first + block - 1) // block
    _asin_multi_uniform_kernel[(count * c,)](*ptrs, first, is64, block)


def foreach_asin_(self=None, **kwargs):
    tensors = self
    if tensors is None:
        tensors = kwargs.get("tensors", kwargs.get("input"))
    kind = type(tensors)
    if kind is not list and kind is not tuple:
        raise TypeError("argument 'self' must be tuple of Tensors")
    count = len(tensors)
    if count == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")
    if count == 1:
        t = tensors[0]
        if not isinstance(t, torch.Tensor):
            raise TypeError("argument 'self' must be tuple of Tensors")
        _asin_one(t)
        return None

    # Direct fused path: one dtype, one element count, and every tensor flat
    # addressable (contiguous or non-overlapping-and-dense) with the group small
    # enough to be launch bound.  This is the shape of the short-list workloads,
    # so it reaches the single fused launch without the grouping dictionaries.
    # Unqualified cases fall through to the general path below.
    t0 = tensors[0]
    if isinstance(t0, torch.Tensor):
        dt0 = t0.dtype
        if dt0 in _FLOAT_DTYPES:
            n0 = t0.numel()
            if n0 and count <= _KMAX and (t0.is_contiguous() or _is_flat_dense(t0)):
                c = (n0 + _BLOCK - 1) // _BLOCK
                if count * c * _BLOCK <= _FUSE_MAX_ELEMS:
                    uniform = True
                    for t in tensors:
                        if (
                            not isinstance(t, torch.Tensor)
                            or t.dtype is not dt0
                            or t.numel() != n0
                            or not (t.is_contiguous() or _is_flat_dense(t))
                        ):
                            uniform = False
                            break
                    if uniform:
                        is64 = dt0 is torch.float64
                        if count == _KMAX:
                            _asin_multi_uniform_kernel[(count * c,)](
                                *tensors, n0, is64, _BLOCK
                            )
                        else:
                            ptrs = list(tensors)
                            ptrs.extend([t0] * (_KMAX - count))
                            _asin_multi_uniform_kernel[(count * c,)](
                                *ptrs, n0, is64, _BLOCK
                            )
                        return None

    groups = {}
    nums = {}
    for t in tensors:
        if not isinstance(t, torch.Tensor):
            raise TypeError("argument 'self' must be tuple of Tensors")
        dt = t.dtype
        if dt not in _FLOAT_DTYPES:
            _raise_bad_dtype(dt)
        n = t.numel()
        if n == 0:
            continue
        if t.is_contiguous() or _is_flat_dense(t):
            bucket = groups.get(dt)
            if bucket is None:
                groups[dt] = [t]
                nums[dt] = [n]
            else:
                bucket.append(t)
                nums[dt].append(n)
        else:
            _asin_strided(t, dt is torch.float64)

    for dt, ts in groups.items():
        _asin_many(ts, nums[dt], dt is torch.float64)
    return None
