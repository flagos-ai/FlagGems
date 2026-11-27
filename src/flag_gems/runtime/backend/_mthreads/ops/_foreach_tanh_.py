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
"""In-place ``aten::_foreach_tanh_`` implemented with Triton kernels.

Applies ``tanh`` to every element tensor of the input sequence in place,
preserving each element's storage, storage_offset, shape, stride and dtype,
and returning ``None``.

Dispatch (host side, metadata only, no data movement):

* every element tensor that is flat-addressable -- contiguous or dense with
  permuted strides (transpose/permute) -- is visited through a flat unit-stride
  traversal of its own storage span; correct for an elementwise in-place op
  because every element is touched exactly once at its own address;
* gappy views are canonicalised into (size, stride) dims and visited through a
  rank-specialised strided kernel (no ``contiguous()`` copy, so writes land in
  the original storage);
* a list of 2..16 flat-addressable tensors of one dtype is fused into a single
  launch, which keeps launch overhead independent of the list length.

Non-floating dtypes and an empty list are rejected before any launch, matching
ATen's ``RuntimeError``.
"""

import logging

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from flag_gems.ops._foreach_unary import _foreach_tanh_ as default__foreach_tanh_

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_FLOATING = (torch.float16, torch.bfloat16, torch.float32, torch.float64)

_BIG_N = 65536
_BLOCK_BIG = 4096
_BLOCK_MID = 2048
_BLOCK_SMALL = 1024


@triton.jit
def _tanh_of(x, CD: tl.constexpr):
    return libdevice.tanh(x.to(CD)).to(x.dtype)


@triton.jit
def _apply(p, n, offs, EVEN: tl.constexpr, CD: tl.constexpr):
    """In-place tanh of the lanes ``offs`` of the tensor at pointer ``p``."""
    if EVEN:
        x = tl.load(p + offs)
        tl.store(p + offs, _tanh_of(x, CD))
    else:
        mask = offs < n
        x = tl.load(p + offs, mask=mask)
        tl.store(p + offs, _tanh_of(x, CD), mask=mask)


@triton.jit
def _tanh_flat_kernel(p, n, BLOCK: tl.constexpr, EVEN: tl.constexpr, CD: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    _apply(p, n, offs, EVEN, CD)


@triton.jit
def _tanh_fused16_kernel(
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    x8,
    x9,
    x10,
    x11,
    x12,
    x13,
    x14,
    x15,
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
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    CD: tl.constexpr,
):
    """One launch, up to 16 flat-addressable tensors of the same dtype."""
    t = tl.program_id(1)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if t == 0:
        _apply(x0, n0, offs, EVEN, CD)
    elif t == 1:
        _apply(x1, n1, offs, EVEN, CD)
    elif t == 2:
        _apply(x2, n2, offs, EVEN, CD)
    elif t == 3:
        _apply(x3, n3, offs, EVEN, CD)
    elif t == 4:
        _apply(x4, n4, offs, EVEN, CD)
    elif t == 5:
        _apply(x5, n5, offs, EVEN, CD)
    elif t == 6:
        _apply(x6, n6, offs, EVEN, CD)
    elif t == 7:
        _apply(x7, n7, offs, EVEN, CD)
    elif t == 8:
        _apply(x8, n8, offs, EVEN, CD)
    elif t == 9:
        _apply(x9, n9, offs, EVEN, CD)
    elif t == 10:
        _apply(x10, n10, offs, EVEN, CD)
    elif t == 11:
        _apply(x11, n11, offs, EVEN, CD)
    elif t == 12:
        _apply(x12, n12, offs, EVEN, CD)
    elif t == 13:
        _apply(x13, n13, offs, EVEN, CD)
    elif t == 14:
        _apply(x14, n14, offs, EVEN, CD)
    else:
        _apply(x15, n15, offs, EVEN, CD)


@triton.jit
def _tanh_nd_kernel(
    p,
    n,
    s0,
    t0,
    s1,
    t1,
    s2,
    t2,
    s3,
    t3,
    R: tl.constexpr,
    BLOCK: tl.constexpr,
    CD: tl.constexpr,
):
    """Strided traversal of a canonical rank-R layout (R in 1..4, innermost last)."""
    offs = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    mask = offs < n
    rem = offs
    off = tl.zeros((BLOCK,), tl.int64)
    if R >= 4:
        off += (rem % s3) * t3
        rem = rem // s3
    if R >= 3:
        off += (rem % s2) * t2
        rem = rem // s2
    if R >= 2:
        off += (rem % s1) * t1
        rem = rem // s1
    off += rem * t0
    x = tl.load(p + off, mask=mask)
    tl.store(p + off, _tanh_of(x, CD), mask=mask)


def _flat_addressable(t):
    """``[data_ptr, data_ptr + numel)`` covers exactly this tensor's elements.

    Dense-but-permuted views (transpose, permute) qualify: their storage span
    equals ``numel``, so a flat unit-stride traversal visits every element
    exactly once -- which is all an elementwise in-place op needs.

    A C-level ``is_contiguous`` answers the common case; the stride walk only
    runs for genuine views.  This Torch build does not expose
    ``Tensor.is_non_overlapping_and_dense``, so the walk is done in Python.
    """
    if t.is_contiguous():
        return True
    if t.numel() <= 1:
        return True
    expected = 1
    for stride, size in sorted(
        (s, sz) for s, sz in zip(t.stride(), t.shape) if sz != 1
    ):
        if stride != expected:
            return False
        expected *= size
    return True


def _has_zero_stride(t):
    for size, stride in zip(t.shape, t.stride()):
        if size > 1 and stride == 0:
            return True
    return False


def _canonical_dims(t):
    """(size, stride) dims, outermost first, size-1 dims dropped, coalesced."""
    dims = [(sz, st) for sz, st in zip(t.shape, t.stride()) if sz != 1]
    dims.sort(key=lambda d: d[1])
    merged = []
    for sz, st in dims:
        if merged and merged[-1][1] * merged[-1][0] == st:
            merged[-1] = (merged[-1][0] * sz, merged[-1][1])
        else:
            merged.append((sz, st))
    merged.reverse()
    return merged


def _pick_block(n):
    if n >= _BIG_N:
        block = _BLOCK_BIG
    elif n >= 8192:
        block = _BLOCK_MID
    else:
        block = _BLOCK_SMALL
    return block, (n % block == 0)


def _launch_one(t, n):
    if n == 0:
        return
    cd = tl.float64 if t.dtype is torch.float64 else tl.float32
    block, even = _pick_block(n)
    if _flat_addressable(t):
        _tanh_flat_kernel[(triton.cdiv(n, block),)](
            t, n, BLOCK=block, EVEN=even, CD=cd, num_warps=4
        )
        return
    if _has_zero_stride(t):
        raise RuntimeError(
            "unsupported operation: more than one element of the written-to "
            "tensor refers to a single memory location. Please clone() the "
            "tensor before performing the operation."
        )
    dims = _canonical_dims(t)
    if not dims:
        return
    if len(dims) > 4:
        raise RuntimeError("_foreach_tanh_: unsupported layout rank")
    sizes = [1, 1, 1, 1]
    strides = [0, 0, 0, 0]
    for i, (sz, st) in enumerate(dims):
        sizes[i] = sz
        strides[i] = st
    _tanh_nd_kernel[(triton.cdiv(n, block),)](
        t,
        n,
        sizes[0],
        strides[0],
        sizes[1],
        strides[1],
        sizes[2],
        strides[2],
        sizes[3],
        strides[3],
        R=len(dims),
        BLOCK=block,
        CD=cd,
        num_warps=4,
    )


def _foreach_tanh_(self=None, *args, **kwargs):
    logger.debug("GEMS_MTHREADS _FOREACH_TANH_")
    """Apply ``tanh`` in place to every tensor of ``self``; returns ``None``."""
    if self is None:
        self = kwargs.get("self")
    if not isinstance(self, (list, tuple)) or any(
        not isinstance(t, torch.Tensor) or t.device.type != "musa" for t in self
    ):
        return default__foreach_tanh_(self)
    tensors = self
    if not isinstance(tensors, (list, tuple)):
        raise TypeError("argument 'self' must be tuple of Tensors")
    nt = len(tensors)
    if nt == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")

    dtype = None
    same_dtype = True
    flat = True
    ns = []
    for t in tensors:
        if not isinstance(t, torch.Tensor):
            raise TypeError("argument 'self' must be tuple of Tensors")
        d = t.dtype
        if d not in _FLOATING:
            raise RuntimeError(
                "result type Float can't be cast to the desired output type "
                + str(d).replace("torch.", "").capitalize()
            )
        if dtype is None:
            dtype = d
        elif d is not dtype:
            same_dtype = False
        n = t.numel()
        ns.append(n)
        if flat and not _flat_addressable(t):
            flat = False

    if nt == 1:
        _launch_one(tensors[0], ns[0])
        return None

    if flat and same_dtype and nt <= 16:
        max_n = max(ns)
        if max_n == 0:
            return None
        seen = set()
        duplicate = False
        for t in tensors:
            p = t.data_ptr()
            if p in seen:
                duplicate = True
                break
            seen.add(p)
        if not duplicate:
            block, even = _pick_block(max_n)
            if even and min(ns) != max_n:
                even = False
            cd = tl.float64 if dtype is torch.float64 else tl.float32
            args = list(tensors)
            counts = list(ns)
            args.extend([tensors[0]] * (16 - nt))
            counts.extend([0] * (16 - nt))
            grid = (triton.cdiv(max_n, block), nt)
            _tanh_fused16_kernel[grid](
                *args, *counts, BLOCK=block, EVEN=even, CD=cd, num_warps=4
            )
            return None

    for t, n in zip(tensors, ns):
        _launch_one(t, n)
    return None
