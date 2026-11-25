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

"""Triton implementation of ``aten::_foreach_sign_`` (multi-tensor, in place).

``run(self)`` reproduces ATen's elementwise ``(0 < x) - (x < 0)`` with:

* no output allocation and no host<->device metadata: the tensor pointers (and,
  only when the sizes are not uniform, the element counts) travel as kernel
  arguments, so one launch covers up to 16 list elements of the same dtype
  (launch count tracks ``ceil(len(self) / 16)``, not ``len(self)``);
* a lean fast path for the common case -- every element of the group has the
  same ``numel`` and that ``numel`` is a multiple of ``BLOCK`` -- where the
  kernel takes no counts and needs no mask, and each slot branch inlines its
  own body so no pointer ever flows through a phi node;
* the grid ordered chunk-major (``x`` = chunk, ``y`` = slot), the same order
  ATen's ``multi_tensor_apply`` uses, so consecutive blocks stream through one
  buffer instead of interleaving 16 of them;
* a scalar-pointer variant of that launch for groups whose data is smaller than
  the launch itself: the pointers are passed as int64 scalars, which the
  launcher marshals more cheaply than tensor arguments;
* a searched tile/warp choice only for the largest single-tensor launches,
  where the tuner's per-call overhead is negligible against the call itself;
* a stride-aware variant for gappy views (``x[:, ::2]``, transposes), whose
  linear index is decomposed into per-dimension offsets, so the write set is
  exactly the view's own elements;
* bool elements left untouched (ATen's ``sign`` kernel is a no-op for bool)
  and complex rejected like ATen.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_BLOCK = 1024
_NUM_WARPS = 4
_SHIFT = 10  # log2(_BLOCK)
# Below this per-tensor element count a call is launch bound, so the
# cheaper-to-marshal scalar-pointer launch with a fixed tile is used.
_SMALL = 1 << 16
# Above it a single-tensor launch is long enough to hide the tuner's per-call
# cost, and its tile/warp choice is searched. Any size using the tuned launch
# must be a multiple of the largest tile so that no config needs a mask.
_HUGE = 1 << 28
_WIDE = _BLOCK * 4

_TL_DTYPE = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
    torch.float64: tl.float64,
    torch.int8: tl.int8,
    torch.int16: tl.int16,
    torch.int32: tl.int32,
    torch.int64: tl.int64,
    torch.uint8: tl.uint8,
}

_NM1_CONFIGS = [
    triton.Config({"BLOCK": 1024}, num_warps=4),
    triton.Config({"BLOCK": 2048}, num_warps=4),
    triton.Config({"BLOCK": 4096}, num_warps=4),
    triton.Config({"BLOCK": 2048}, num_warps=8),
]


# ---------------------------------------------------------------------------
# element-wise math
# ---------------------------------------------------------------------------


@triton.jit
def _sign_vals(x):
    """``(0 < x) - (x < 0)``: +1 / -1 / 0, and 0 for NaN, exactly like ATen.

    Computed in int32 and narrowed by the store, which is exact for every dtype
    ATen accepts (fp16/bf16/fp32/fp64 and int8..int64/uint8).
    """
    return (x > 0).to(tl.int32) - (x < 0).to(tl.int32)


@triton.jit
def _sign_body(p, offs, mask):
    x = tl.load(p + offs, mask=mask, other=0)
    tl.store(p + offs, _sign_vals(x), mask=mask)


# ---------------------------------------------------------------------------
# uniform-size fast path: no counts, no mask, one body per slot branch
# grid = (chunk, slot)
# ---------------------------------------------------------------------------


@triton.jit
def _sign_nm1(p, BLOCK: tl.constexpr):
    """Single tensor with the fixed launch-bound tile, no mask."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(p + offs)
    tl.store(p + offs, _sign_vals(x))


@triton.autotune(configs=_NM1_CONFIGS, key=["n"])
@triton.jit
def _sign_nm1_t(p, n, BLOCK: tl.constexpr):
    """Largest single-tensor launches: tile and warp count are searched.

    Only chosen when ``n`` is a multiple of every config's ``BLOCK``, so the
    load and store need no mask and stay fully vectorized.
    """
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(p + offs)
    tl.store(p + offs, _sign_vals(x))


@triton.jit
def _sign_nm16(
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
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.program_id(1)
    if slot == 0:
        x = tl.load(p0 + offs)
        tl.store(p0 + offs, _sign_vals(x))
    elif slot == 1:
        x = tl.load(p1 + offs)
        tl.store(p1 + offs, _sign_vals(x))
    elif slot == 2:
        x = tl.load(p2 + offs)
        tl.store(p2 + offs, _sign_vals(x))
    elif slot == 3:
        x = tl.load(p3 + offs)
        tl.store(p3 + offs, _sign_vals(x))
    elif slot == 4:
        x = tl.load(p4 + offs)
        tl.store(p4 + offs, _sign_vals(x))
    elif slot == 5:
        x = tl.load(p5 + offs)
        tl.store(p5 + offs, _sign_vals(x))
    elif slot == 6:
        x = tl.load(p6 + offs)
        tl.store(p6 + offs, _sign_vals(x))
    elif slot == 7:
        x = tl.load(p7 + offs)
        tl.store(p7 + offs, _sign_vals(x))
    elif slot == 8:
        x = tl.load(p8 + offs)
        tl.store(p8 + offs, _sign_vals(x))
    elif slot == 9:
        x = tl.load(p9 + offs)
        tl.store(p9 + offs, _sign_vals(x))
    elif slot == 10:
        x = tl.load(p10 + offs)
        tl.store(p10 + offs, _sign_vals(x))
    elif slot == 11:
        x = tl.load(p11 + offs)
        tl.store(p11 + offs, _sign_vals(x))
    elif slot == 12:
        x = tl.load(p12 + offs)
        tl.store(p12 + offs, _sign_vals(x))
    elif slot == 13:
        x = tl.load(p13 + offs)
        tl.store(p13 + offs, _sign_vals(x))
    elif slot == 14:
        x = tl.load(p14 + offs)
        tl.store(p14 + offs, _sign_vals(x))
    else:
        x = tl.load(p15 + offs)
        tl.store(p15 + offs, _sign_vals(x))


@triton.jit
def _sign_ns16(
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
    DT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Same launch as ``_sign_nm16`` with the pointers as int64 scalars."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.program_id(1)
    if slot == 0:
        q = p0.to(tl.pointer_type(DT))
        x = tl.load(q + offs)
        tl.store(q + offs, _sign_vals(x))
    elif slot == 1:
        q = p1.to(tl.pointer_type(DT))
        x = tl.load(q + offs)
        tl.store(q + offs, _sign_vals(x))
    elif slot == 2:
        q = p2.to(tl.pointer_type(DT))
        x = tl.load(q + offs)
        tl.store(q + offs, _sign_vals(x))
    elif slot == 3:
        q = p3.to(tl.pointer_type(DT))
        x = tl.load(q + offs)
        tl.store(q + offs, _sign_vals(x))
    elif slot == 4:
        q = p4.to(tl.pointer_type(DT))
        x = tl.load(q + offs)
        tl.store(q + offs, _sign_vals(x))
    elif slot == 5:
        q = p5.to(tl.pointer_type(DT))
        x = tl.load(q + offs)
        tl.store(q + offs, _sign_vals(x))
    elif slot == 6:
        q = p6.to(tl.pointer_type(DT))
        x = tl.load(q + offs)
        tl.store(q + offs, _sign_vals(x))
    elif slot == 7:
        q = p7.to(tl.pointer_type(DT))
        x = tl.load(q + offs)
        tl.store(q + offs, _sign_vals(x))
    elif slot == 8:
        q = p8.to(tl.pointer_type(DT))
        x = tl.load(q + offs)
        tl.store(q + offs, _sign_vals(x))
    elif slot == 9:
        q = p9.to(tl.pointer_type(DT))
        x = tl.load(q + offs)
        tl.store(q + offs, _sign_vals(x))
    elif slot == 10:
        q = p10.to(tl.pointer_type(DT))
        x = tl.load(q + offs)
        tl.store(q + offs, _sign_vals(x))
    elif slot == 11:
        q = p11.to(tl.pointer_type(DT))
        x = tl.load(q + offs)
        tl.store(q + offs, _sign_vals(x))
    elif slot == 12:
        q = p12.to(tl.pointer_type(DT))
        x = tl.load(q + offs)
        tl.store(q + offs, _sign_vals(x))
    elif slot == 13:
        q = p13.to(tl.pointer_type(DT))
        x = tl.load(q + offs)
        tl.store(q + offs, _sign_vals(x))
    elif slot == 14:
        q = p14.to(tl.pointer_type(DT))
        x = tl.load(q + offs)
        tl.store(q + offs, _sign_vals(x))
    else:
        q = p15.to(tl.pointer_type(DT))
        x = tl.load(q + offs)
        tl.store(q + offs, _sign_vals(x))


# ---------------------------------------------------------------------------
# general path: arbitrary (possibly heterogeneous) element counts
# ---------------------------------------------------------------------------


@triton.jit
def _sign_m1(p, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    _sign_body(p, offs, offs < n)


@triton.jit
def _sign_m16(
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
    BLOCK: tl.constexpr,
):
    slot = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    if slot == 0:
        _sign_body(p0, offs, offs < n0)
    elif slot == 1:
        _sign_body(p1, offs, offs < n1)
    elif slot == 2:
        _sign_body(p2, offs, offs < n2)
    elif slot == 3:
        _sign_body(p3, offs, offs < n3)
    elif slot == 4:
        _sign_body(p4, offs, offs < n4)
    elif slot == 5:
        _sign_body(p5, offs, offs < n5)
    elif slot == 6:
        _sign_body(p6, offs, offs < n6)
    elif slot == 7:
        _sign_body(p7, offs, offs < n7)
    elif slot == 8:
        _sign_body(p8, offs, offs < n8)
    elif slot == 9:
        _sign_body(p9, offs, offs < n9)
    elif slot == 10:
        _sign_body(p10, offs, offs < n10)
    elif slot == 11:
        _sign_body(p11, offs, offs < n11)
    elif slot == 12:
        _sign_body(p12, offs, offs < n12)
    elif slot == 13:
        _sign_body(p13, offs, offs < n13)
    elif slot == 14:
        _sign_body(p14, offs, offs < n14)
    else:
        _sign_body(p15, offs, offs < n15)


# ---------------------------------------------------------------------------
# stride-aware kernel for non-contiguous views
# ---------------------------------------------------------------------------


@triton.jit
def _sign_strided(
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
    NDIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    lin = offs.to(tl.int64)
    off = tl.zeros([BLOCK], dtype=tl.int64)
    if NDIM > 0:
        off += (lin % d0) * s0
        lin = lin // d0
    if NDIM > 1:
        off += (lin % d1) * s1
        lin = lin // d1
    if NDIM > 2:
        off += (lin % d2) * s2
        lin = lin // d2
    if NDIM > 3:
        off += (lin % d3) * s3
        lin = lin // d3
    if NDIM > 4:
        off += (lin % d4) * s4
        lin = lin // d4
    if NDIM > 5:
        off += (lin % d5) * s5
        lin = lin // d5
    if NDIM > 6:
        off += (lin % d6) * s6
        lin = lin // d6
    if NDIM > 7:
        off += (lin % d7) * s7
        lin = lin // d7

    x = tl.load(p + off, mask=mask, other=0)
    tl.store(p + off, _sign_vals(x), mask=mask)


# ---------------------------------------------------------------------------
# host side
# ---------------------------------------------------------------------------


def _launch_strided(t, n):
    dims = [(sz, st) for sz, st in zip(t.shape, t.stride()) if sz != 1]
    if not dims:
        # rank-0 shaped as all-ones: a single element
        dims = [(1, 0)]
    if len(dims) > 8:
        # Only reachable for tensors of rank > 8; peel the leading dimension so
        # each launch stays within the kernel's unrolled index decomposition.
        for j in range(t.shape[0]):
            sub = t.select(0, j)
            if sub.numel():
                _launch_strided(sub, sub.numel())
        return

    s = [0] * 8
    d = [1] * 8
    for j, (sz, st) in enumerate(dims):
        d[j] = sz
        s[j] = st

    grid = ((n + _BLOCK - 1) // _BLOCK,)
    _sign_strided[grid](
        t,
        n,
        s[0],
        s[1],
        s[2],
        s[3],
        s[4],
        s[5],
        s[6],
        s[7],
        d[0],
        d[1],
        d[2],
        d[3],
        d[4],
        d[5],
        d[6],
        d[7],
        NDIM=len(dims),
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
    )


def _launch_group(ts, ns, dtl):
    total = len(ts)
    maxn = max(ns)
    if maxn == ns[0] and not (maxn & (_BLOCK - 1)):
        # Uniform, BLOCK-aligned sizes: no counts and no mask, so the launch
        # carries the fewest possible arguments.
        if total == 1:
            if maxn >= _HUGE and not (maxn & (_WIDE - 1)):
                _sign_nm1_t[
                    lambda meta: ((maxn + meta["BLOCK"] - 1) // meta["BLOCK"],)
                ](ts[0], maxn)
            else:
                _sign_nm1[(maxn >> _SHIFT,)](ts[0], BLOCK=_BLOCK, num_warps=_NUM_WARPS)
            return
        small = dtl is not None and maxn <= _SMALL
        i = 0
        while i < total:
            k = total - i
            if k > 16:
                k = 16
            chunk = ts[i : i + k]
            if k < 16:
                chunk = chunk + [chunk[0]] * (16 - k)
            if small:
                ptrs = [t.data_ptr() for t in chunk]
                _sign_ns16[(maxn >> _SHIFT, k)](
                    ptrs[0],
                    ptrs[1],
                    ptrs[2],
                    ptrs[3],
                    ptrs[4],
                    ptrs[5],
                    ptrs[6],
                    ptrs[7],
                    ptrs[8],
                    ptrs[9],
                    ptrs[10],
                    ptrs[11],
                    ptrs[12],
                    ptrs[13],
                    ptrs[14],
                    ptrs[15],
                    DT=dtl,
                    BLOCK=_BLOCK,
                    num_warps=_NUM_WARPS,
                )
            else:
                _sign_nm16[(maxn >> _SHIFT, k)](
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
                    BLOCK=_BLOCK,
                    num_warps=_NUM_WARPS,
                )
            i += k
        return

    i = 0
    while i < total:
        k = total - i
        if k > 16:
            k = 16
        chunk = ts[i : i + k]
        nums = ns[i : i + k]
        maxc = max(nums)
        if k == 1:
            grid = ((maxc + _BLOCK - 1) // _BLOCK,)
            _sign_m1[grid](chunk[0], nums[0], BLOCK=_BLOCK, num_warps=_NUM_WARPS)
        else:
            grid = (k, (maxc + _BLOCK - 1) // _BLOCK)
            if k < 16:
                chunk = chunk + [chunk[0]] * (16 - k)
                nums = nums + [0] * (16 - k)
            _sign_m16[grid](
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
                nums[0],
                nums[1],
                nums[2],
                nums[3],
                nums[4],
                nums[5],
                nums[6],
                nums[7],
                nums[8],
                nums[9],
                nums[10],
                nums[11],
                nums[12],
                nums[13],
                nums[14],
                nums[15],
                BLOCK=_BLOCK,
                num_warps=_NUM_WARPS,
            )
        i += k


def _foreach_sign_(self):
    logger.debug("GEMS_HYGON _FOREACH_SIGN_")
    tensors = self
    if isinstance(tensors, torch.Tensor):
        tensors = (tensors,)
    elif not isinstance(tensors, (list, tuple)):
        raise TypeError("argument 'self' must be tuple of Tensors")
    if len(tensors) == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")

    groups = {}
    for t in tensors:
        n = t.numel()
        if n == 0:
            continue
        d = t.dtype
        if not t.is_contiguous():
            if d.is_complex:
                raise RuntimeError(
                    '"foreach" not implemented for \''
                    + str(d).replace("torch.", "")
                    + "'"
                )
            _launch_strided(t, n)
            continue
        if d is torch.bool:
            # ATen's sign kernel returns immediately for bool: nothing to do.
            continue
        g = groups.get(d)
        if g is None:
            if d.is_complex:
                raise RuntimeError(
                    '"foreach" not implemented for \''
                    + str(d).replace("torch.", "")
                    + "'"
                )
            groups[d] = g = ([], [])
        g[0].append(t)
        g[1].append(n)

    for d, (ts, ns) in groups.items():
        _launch_group(ts, ns, _TL_DTYPE.get(d))
    return None
