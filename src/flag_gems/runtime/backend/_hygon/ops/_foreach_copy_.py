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

"""Triton implementation of aten::_foreach_copy_ (in-place multi-tensor copy).

Semantics: for every pair i, dst[i].copy_(src[i]) with ATen broadcast and
dtype-conversion rules. The destination storages are mutated in place and the
call returns None.

Launch strategy
---------------
* Independent contiguous pairs are fused into a single launch (slot =
  program_id(1)), so a foreach list of N tensors costs one launch instead of N.
* That launch is issued through a cached pre-compiled kernel handle
  (CompiledKernel runner) instead of JITFunction.run, which rebuilds the
  per-argument specialization key on every call.  The runner must receive every
  signature slot positionally, including constexpr slots; the cache key covers
  slot count, grid, block/warps, element types and every length, so a compiled
  binary is never reused for a different signature.
* Pairs that need broadcast or strided addressing use a general index kernel,
  one launch per pair.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

__all__ = ["_foreach_copy_"]

_MAX_SLOTS = 8

_SPECIAL_DTYPES = frozenset(
    d
    for d in (
        torch.bool,
        getattr(torch, "complex32", None),
        torch.complex64,
        torch.complex128,
    )
    if d is not None
)


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@triton.jit
def _copy_chunk(src_ptr, dst_ptr, offs, n):
    mask = offs < n
    val = tl.load(src_ptr + offs, mask=mask, other=0)
    tl.store(dst_ptr + offs, val.to(dst_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _copy_flat_single(src_ptr, dst_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    _copy_chunk(src_ptr, dst_ptr, pid * BLOCK + tl.arange(0, BLOCK), n)


@triton.jit
def _copy_flat_fused(
    s0,
    d0,
    s1,
    d1,
    s2,
    d2,
    s3,
    d3,
    s4,
    d4,
    s5,
    d5,
    s6,
    d6,
    s7,
    d7,
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
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    slot = tl.program_id(1)
    if slot == 0:
        _copy_chunk(s0, d0, offs, n0)
    elif slot == 1:
        _copy_chunk(s1, d1, offs, n1)
    elif slot == 2:
        _copy_chunk(s2, d2, offs, n2)
    elif slot == 3:
        _copy_chunk(s3, d3, offs, n3)
    elif slot == 4:
        _copy_chunk(s4, d4, offs, n4)
    elif slot == 5:
        _copy_chunk(s5, d5, offs, n5)
    elif slot == 6:
        _copy_chunk(s6, d6, offs, n6)
    else:
        _copy_chunk(s7, d7, offs, n7)


@triton.jit
def _copy_strided_kernel(
    src_ptr,
    dst_ptr,
    n,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    y0,
    y1,
    y2,
    y3,
    y4,
    y5,
    y6,
    y7,
    BLOCK: tl.constexpr,
):
    """Copy with arbitrary destination strides and broadcast source strides."""
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)

    i7 = idx % d7
    r = idx // d7
    i6 = r % d6
    r = r // d6
    i5 = r % d5
    r = r // d5
    i4 = r % d4
    r = r // d4
    i3 = r % d3
    r = r // d3
    i2 = r % d2
    r = r // d2
    i1 = r % d1
    i0 = r // d1

    dst_off = (
        i0 * x0 + i1 * x1 + i2 * x2 + i3 * x3 + i4 * x4 + i5 * x5 + i6 * x6 + i7 * x7
    )
    src_off = (
        i0 * y0 + i1 * y1 + i2 * y2 + i3 * y3 + i4 * y4 + i5 * y5 + i6 * y6 + i7 * y7
    )

    mask = idx < n
    val = tl.load(src_ptr + src_off, mask=mask, other=0)
    tl.store(dst_ptr + dst_off, val.to(dst_ptr.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------
# Host helpers
# ---------------------------------------------------------------------------

_MISS = object()
# key -> CompiledKernel runner bound to a grid, or None when unavailable
_RUNNERS = {}


def _block_for(n, esize=4):
    if n <= 256:
        return 256, 2
    if n <= 4096:
        return 1024, 4
    # 16 bytes per thread with 16 warps: BLOCK * esize == 16 KiB per program.
    block = 16384 // esize
    return (block if block > 256 else 256), 16


def _dispatch(kernel, key, args, grid, block, warps):
    """Launch through a cached pre-compiled runner, falling back to the JIT."""
    runner = _RUNNERS.get(key, _MISS)
    if runner is _MISS:
        try:
            compiled = kernel.warmup(*args, grid=grid, num_warps=warps)
            runner = compiled[grid]
        except Exception:
            runner = None
        _RUNNERS[key] = runner
    if runner is not None:
        try:
            runner(*args)
            return
        except Exception:
            _RUNNERS[key] = None
    kernel[grid](*args[:-1], BLOCK=block, num_warps=warps)


def _launch_one(dst, src, n):
    block, warps = _block_for(n, dst.element_size())
    nb = (n + block - 1) // block
    if n <= 1:
        _copy_flat_single[(nb,)](src, dst, n, BLOCK=block, num_warps=warps)
        return
    grid = (nb, 1, 1)
    key = (1, nb, block, warps, n, src.dtype, dst.dtype)
    _dispatch(_copy_flat_single, key, (src, dst, n, block), grid, block, warps)


def _launch_group(dsts, srcs, sizes, k):
    if k == 1:
        _launch_one(dsts[0], srcs[0], sizes[0])
        return
    max_n = max(sizes)
    block, warps = _block_for(max_n, dsts[0].element_size())
    nb = (max_n + block - 1) // block
    grid = (nb, k, 1)

    s0 = srcs[0]
    d0 = dsts[0]
    n0 = sizes[0]
    # Unused slots repeat slot 0's pointers; their length is 0 and their grid
    # dimension is never scheduled.
    if k > 1:
        s1, d1, n1 = srcs[1], dsts[1], sizes[1]
    else:
        s1, d1, n1 = s0, d0, 0
    if k > 2:
        s2, d2, n2 = srcs[2], dsts[2], sizes[2]
    else:
        s2, d2, n2 = s0, d0, 0
    if k > 3:
        s3, d3, n3 = srcs[3], dsts[3], sizes[3]
    else:
        s3, d3, n3 = s0, d0, 0
    if k > 4:
        s4, d4, n4 = srcs[4], dsts[4], sizes[4]
    else:
        s4, d4, n4 = s0, d0, 0
    if k > 5:
        s5, d5, n5 = srcs[5], dsts[5], sizes[5]
    else:
        s5, d5, n5 = s0, d0, 0
    if k > 6:
        s6, d6, n6 = srcs[6], dsts[6], sizes[6]
    else:
        s6, d6, n6 = s0, d0, 0
    if k > 7:
        s7, d7, n7 = srcs[7], dsts[7], sizes[7]
    else:
        s7, d7, n7 = s0, d0, 0

    args = (
        s0,
        d0,
        s1,
        d1,
        s2,
        d2,
        s3,
        d3,
        s4,
        d4,
        s5,
        d5,
        s6,
        d6,
        s7,
        d7,
        n0,
        n1,
        n2,
        n3,
        n4,
        n5,
        n6,
        n7,
        block,
    )
    key = (
        k,
        nb,
        block,
        warps,
        n0,
        n1,
        n2,
        n3,
        n4,
        n5,
        n6,
        n7,
        s0.dtype,
        d0.dtype,
        s1.dtype,
        d1.dtype,
        s2.dtype,
        d2.dtype,
        s3.dtype,
        d3.dtype,
        s4.dtype,
        d4.dtype,
        s5.dtype,
        d5.dtype,
        s6.dtype,
        d6.dtype,
        s7.dtype,
        d7.dtype,
    )
    if (
        n0 <= 1
        or n1 == 1
        or n2 == 1
        or n3 == 1
        or n4 == 1
        or n5 == 1
        or n6 == 1
        or n7 == 1
    ):
        _copy_flat_fused[grid](*args[:-1], BLOCK=block, num_warps=warps)
        return
    _dispatch(_copy_flat_fused, key, args, grid, block, warps)


def _strided_params(d, s):
    """Right-aligned broadcast parameters padded to 8 dims (dims, dst_str, src_str)."""
    dd = d.shape
    ss = s.shape
    nd = len(dd)
    ns = len(ss)
    if nd > 8:
        raise RuntimeError("_foreach_copy_: tensors with more than 8 dims unsupported")
    # Leading unit dims of src are irrelevant (copy_ never expands dst).
    while ns > nd and ss[0] == 1:
        ss = ss[1:]
        ns -= 1
    if ns > nd:
        raise RuntimeError(
            "_foreach_copy_: shape mismatch: src shape cannot broadcast to dst shape"
        )
    src_strides = list(s.stride())[len(s.shape) - ns :] if ns else []
    offset = nd - ns

    dims = [1] * 8
    dstr = [0] * 8
    sstr = [0] * 8
    for k in range(nd):
        i = nd - 1 - k
        pos = 7 - k
        size = int(dd[i])
        dims[pos] = size
        dstr[pos] = int(d.stride(i))
        if i >= offset:
            si = int(ss[i - offset])
            if si == size:
                sstr[pos] = int(src_strides[i - offset])
            elif si == 1:
                sstr[pos] = 0
            else:
                raise RuntimeError(
                    "_foreach_copy_: shape mismatch: src shape cannot broadcast to dst shape"
                )
        else:
            sstr[pos] = 0
    for k in range(8):
        if dims[k] == 1:
            dstr[k] = 0
            sstr[k] = 0
    return dims, dstr, sstr


def _launch_strided(dst, s):
    dims, dstr, sstr = _strided_params(dst, s)
    n = dst.numel()
    block, warps = _block_for(n)
    _copy_strided_kernel[((n + block - 1) // block,)](
        s,
        dst,
        n,
        *dims,
        *dstr,
        *sstr,
        BLOCK=block,
        num_warps=warps,
    )


def _slow_pair(dst, s):
    """Handle dtype reinterpretation, broadcast and non-contiguous pairs."""
    if dst.device != s.device:
        raise RuntimeError(
            "_foreach_copy_: self and src tensors must be on the same device"
        )
    if dst.dtype in _SPECIAL_DTYPES or s.dtype in _SPECIAL_DTYPES:
        if dst.dtype.is_complex or s.dtype.is_complex:
            if (
                dst.dtype is not s.dtype
                or dst.shape != s.shape
                or not dst.is_contiguous()
                or not s.is_contiguous()
            ):
                raise RuntimeError("_foreach_copy_: unsupported complex tensor copy")
            dst = dst.view(torch.float32)
            s = s.view(torch.float32)
        else:
            if dst.dtype is torch.bool:
                dst = dst.view(torch.uint8)
            if s.dtype is torch.bool:
                s = s.view(torch.uint8)

    n = dst.numel()
    if n == 0:
        return
    if dst.shape == s.shape and dst.is_contiguous() and s.is_contiguous():
        _launch_one(dst, s, n)
    else:
        _launch_strided(dst, s)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _foreach_copy_(self, src, non_blocking=False):
    logger.debug("GEMS_HYGON _FOREACH_COPY_")
    if len(self) != len(src):
        raise RuntimeError(
            "_foreach_copy_: expected the same number of tensors in self and src"
        )
    if not len(self):
        raise RuntimeError("_foreach_copy_: tensor list must have at least one tensor")

    dsts = []
    srcs = []
    sizes = []
    slow = []
    for dst, s in zip(self, src):
        if dst.dtype in _SPECIAL_DTYPES or s.dtype in _SPECIAL_DTYPES:
            slow.append((dst, s))
            continue
        if dst.device != s.device:
            raise RuntimeError(
                "_foreach_copy_: self and src tensors must be on the same device"
            )
        n = dst.numel()
        if dst.shape == s.shape and dst.is_contiguous() and s.is_contiguous():
            if n:
                dsts.append(dst)
                srcs.append(s)
                sizes.append(n)
        else:
            slow.append((dst, s))

    count = len(sizes)
    if count == 1:
        _launch_one(dsts[0], srcs[0], sizes[0])
    elif count > 1:
        if count <= _MAX_SLOTS:
            _launch_group(dsts, srcs, sizes, count)
        else:
            for start in range(0, count, _MAX_SLOTS):
                stop = start + _MAX_SLOTS
                if stop > count:
                    stop = count
                _launch_group(
                    dsts[start:stop], srcs[start:stop], sizes[start:stop], stop - start
                )
    for dst, s in slow:
        _slow_pair(dst, s)
    return None


_foreach_copy_ = _foreach_copy_
