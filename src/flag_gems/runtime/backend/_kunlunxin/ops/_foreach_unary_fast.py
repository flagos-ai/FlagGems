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



import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_unary import apply_unary
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.foreach import check_tensor_list, is_flat_addressable, tl_dtype

# Vendor overrides log under their own module logger with the vendor prefix;
# tests/test_foreach_unary.py derives both from the resolved function.
logger = logging.getLogger(__name__)

_FAST_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
# Integral/bool inputs promote to float32 for both operators; they are handled
# by an explicit host-side cast (see ``_apply``).
_INT_IN_DTYPES = (
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
    torch.bool,
)
_K16_SLOTS = 16
# A single launch over equal-numel tensors is used only while the list fits the
# threshold below; beyond it the 16-way unrolled chain is replaced by one launch
# per tensor.  GEMS_XPU_FOREACH_K16_MAX overrides the threshold.
_K16_MAX_NUMEL = int(os.environ.get("GEMS_XPU_FOREACH_K16_MAX", "0") or 0) or (
    1 << 19
)


@triton.jit
def _expm1_fn(x):
    # Identical to ops/expm1.py: promote to fp32 first, then subtract one.
    return tl.exp(x.to(tl.float32)) - 1.0


@triton.jit
def _reciprocal_fn(x):
    return 1.0 / x.to(tl.float32)


@triton.jit
def _foreach16_unary_kernel(
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6,
    a7,
    a8,
    a9,
    a10,
    a11,
    a12,
    a13,
    a14,
    a15,
    o0,
    o1,
    o2,
    o3,
    o4,
    o5,
    o6,
    o7,
    o8,
    o9,
    o10,
    o11,
    o12,
    o13,
    o14,
    o15,
    n,
    fn: tl.constexpr,
    NT: tl.constexpr,
    OT: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    """One launch, up to 16 equal-numel tensors; no mask on the ``EVEN`` path."""
    off = tl.program_id(0) * BLOCK
    idx = off + tl.arange(0, BLOCK)
    if EVEN:
        if NT > 0:
            tl.store(o0 + idx, fn(tl.load(a0 + idx).to(OT)).to(OT))
        if NT > 1:
            tl.store(o1 + idx, fn(tl.load(a1 + idx).to(OT)).to(OT))
        if NT > 2:
            tl.store(o2 + idx, fn(tl.load(a2 + idx).to(OT)).to(OT))
        if NT > 3:
            tl.store(o3 + idx, fn(tl.load(a3 + idx).to(OT)).to(OT))
        if NT > 4:
            tl.store(o4 + idx, fn(tl.load(a4 + idx).to(OT)).to(OT))
        if NT > 5:
            tl.store(o5 + idx, fn(tl.load(a5 + idx).to(OT)).to(OT))
        if NT > 6:
            tl.store(o6 + idx, fn(tl.load(a6 + idx).to(OT)).to(OT))
        if NT > 7:
            tl.store(o7 + idx, fn(tl.load(a7 + idx).to(OT)).to(OT))
        if NT > 8:
            tl.store(o8 + idx, fn(tl.load(a8 + idx).to(OT)).to(OT))
        if NT > 9:
            tl.store(o9 + idx, fn(tl.load(a9 + idx).to(OT)).to(OT))
        if NT > 10:
            tl.store(o10 + idx, fn(tl.load(a10 + idx).to(OT)).to(OT))
        if NT > 11:
            tl.store(o11 + idx, fn(tl.load(a11 + idx).to(OT)).to(OT))
        if NT > 12:
            tl.store(o12 + idx, fn(tl.load(a12 + idx).to(OT)).to(OT))
        if NT > 13:
            tl.store(o13 + idx, fn(tl.load(a13 + idx).to(OT)).to(OT))
        if NT > 14:
            tl.store(o14 + idx, fn(tl.load(a14 + idx).to(OT)).to(OT))
        if NT > 15:
            tl.store(o15 + idx, fn(tl.load(a15 + idx).to(OT)).to(OT))
    else:
        m = idx < n
        if NT > 0:
            tl.store(o0 + idx, fn(tl.load(a0 + idx, mask=m).to(OT)).to(OT), mask=m)
        if NT > 1:
            tl.store(o1 + idx, fn(tl.load(a1 + idx, mask=m).to(OT)).to(OT), mask=m)
        if NT > 2:
            tl.store(o2 + idx, fn(tl.load(a2 + idx, mask=m).to(OT)).to(OT), mask=m)
        if NT > 3:
            tl.store(o3 + idx, fn(tl.load(a3 + idx, mask=m).to(OT)).to(OT), mask=m)
        if NT > 4:
            tl.store(o4 + idx, fn(tl.load(a4 + idx, mask=m).to(OT)).to(OT), mask=m)
        if NT > 5:
            tl.store(o5 + idx, fn(tl.load(a5 + idx, mask=m).to(OT)).to(OT), mask=m)
        if NT > 6:
            tl.store(o6 + idx, fn(tl.load(a6 + idx, mask=m).to(OT)).to(OT), mask=m)
        if NT > 7:
            tl.store(o7 + idx, fn(tl.load(a7 + idx, mask=m).to(OT)).to(OT), mask=m)
        if NT > 8:
            tl.store(o8 + idx, fn(tl.load(a8 + idx, mask=m).to(OT)).to(OT), mask=m)
        if NT > 9:
            tl.store(o9 + idx, fn(tl.load(a9 + idx, mask=m).to(OT)).to(OT), mask=m)
        if NT > 10:
            tl.store(o10 + idx, fn(tl.load(a10 + idx, mask=m).to(OT)).to(OT), mask=m)
        if NT > 11:
            tl.store(o11 + idx, fn(tl.load(a11 + idx, mask=m).to(OT)).to(OT), mask=m)
        if NT > 12:
            tl.store(o12 + idx, fn(tl.load(a12 + idx, mask=m).to(OT)).to(OT), mask=m)
        if NT > 13:
            tl.store(o13 + idx, fn(tl.load(a13 + idx, mask=m).to(OT)).to(OT), mask=m)
        if NT > 14:
            tl.store(o14 + idx, fn(tl.load(a14 + idx, mask=m).to(OT)).to(OT), mask=m)
        if NT > 15:
            tl.store(o15 + idx, fn(tl.load(a15 + idx, mask=m).to(OT)).to(OT), mask=m)


@triton.jit
def _foreach1_unary_kernel(
    a_ptr,
    o_ptr,
    num_tasks,
    tiles_per_cta,
    fn: tl.constexpr,
    OT: tl.constexpr,
    tile_size: tl.constexpr,
    EVEN: tl.constexpr,
):
    """One tensor per launch: one tile per CTA, unmasked when possible.

    A masked access is lowered element-wise on this backend (the same conclusion
    as ``_foreach_sigmoid.py``), so the unmasked path is used whenever the sizes
    allow it.  One tile per CTA also lets one CTA move data while another
    computes, so memory and arithmetic overlap instead of being serialised
    inside a single large tile.

    ``EVEN`` is guaranteed by the caller (the tile count is an exact multiple of
    the CTA count and ``n`` is tile-aligned), so no mask is needed here -- and no
    runtime tile-id guard either: a ``if tile_id < num_tiles`` form does not
    lower on this backend.
    """
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    for j in range(0, tiles_per_cta):
        tid = (pid + j * num_ctas) * tile_size + tl.arange(0, tile_size)
        if EVEN:
            tl.store(o_ptr + tid, fn(tl.load(a_ptr + tid).to(OT)).to(OT))
        else:
            m = tid < num_tasks
            tl.store(
                o_ptr + tid,
                fn(tl.load(a_ptr + tid, mask=m).to(OT)).to(OT),
                mask=m,
            )


def _pick_block(n: int):
    for block in (8192, 4096, 2048, 1024):
        if n % block == 0:
            return block, True
    return 1024, False


def _launch16(tensors, dsts, fn, ot):
    n = tensors[0].numel()
    block, even = _pick_block(n)
    nt16 = min(len(tensors), _K16_SLOTS)
    # Dead ``if NT > i`` slots are compiled away; only the argument values need
    # to be valid pointers, so the live slots are reused.
    a = list(tensors[:nt16]) + [tensors[0]] * (_K16_SLOTS - nt16)
    o = list(dsts[:nt16]) + [dsts[0]] * (_K16_SLOTS - nt16)
    _foreach16_unary_kernel[(triton.cdiv(n, block),)](
        *a, *o, n, fn, nt16, ot, block, even, num_warps=8
    )


# Tile-size / CTA knobs (same convention as sort.py's
# GEMS_XPU_RADIX_BLOCK_N): one tile per CTA, so every program owns a single
# contiguous range.
_TILE_DEFAULT = 131072
_CTAS_DEFAULT = 128
_TILE_CAP = int(os.environ.get("GEMS_XPU_FOREACH_TILE", "0") or 0) or _TILE_DEFAULT
_CTAS_CAP = int(os.environ.get("GEMS_XPU_FOREACH_CTAS", "0") or 0) or _CTAS_DEFAULT
# XTriton DMA batching knobs.
_UNROLL = int(os.environ.get("GEMS_XPU_FOREACH_UNROLL", "0") or 0) or 8
_BUF = int(os.environ.get("GEMS_XPU_FOREACH_BUF", "0") or 0) or 4096


def _big_tile_plan(n):
    """``(num_ctas, tiles_per_cta, tile_size, even)`` for the per-tensor path.

    ``even`` requires BOTH that ``n`` is tile-aligned and that the tile count is
    an exact multiple of the CTA count - otherwise a program would address a
    tile past the buffer, and the unmasked body has no guard (a runtime
    ``if tile_id < num_tiles`` does not lower on this backend).
    """
    tile_size = min(_TILE_CAP, triton.next_power_of_2(n))
    # Kept from the merged recipe (clamp_max).
    if tile_size == 32768:
        tile_size = 16384
    num_tiles = max(1, triton.cdiv(n, tile_size))

    want = min(_CTAS_CAP, num_tiles)
    num_ctas = want
    while num_ctas > 1 and num_tiles % num_ctas:
        num_ctas -= 1
    even = n % tile_size == 0
    if num_ctas * 2 < want:
        # The largest divisor is too small (e.g. a prime tile count): prefer
        # concurrency and let the masked body absorb the stray tile ids.
        num_ctas = want
        even = False
    tiles_per_cta = triton.cdiv(num_tiles, num_ctas)
    return num_ctas, tiles_per_cta, tile_size, even


def _launch1(src, dst, fn, ot):
    n = src.numel()
    num_ctas, tiles_per_cta, tile_size, even = _big_tile_plan(n)
    num_warps = 16 if tile_size >= 4096 else (8 if tile_size >= 2048 else 4)
    _foreach1_unary_kernel[(num_ctas,)](
        src,
        dst,
        n,
        tiles_per_cta,
        fn,
        ot,
        tile_size,
        even,
        num_warps=num_warps,
        unroll_num=_UNROLL,
        buffer_size_limit=_BUF,
    )


def _alloc_dsts(tensors, out_dtype):
    """One allocation for the whole list instead of one per tensor.

    ``[torch.empty_like(t) for t in tensors]`` costs one allocator round trip
    per tensor, which dominates the small-shape cost.  A single flat buffer
    sliced into views has the same layout because every tensor here is
    contiguous, and it keeps the fast path allocation-free per tensor.
    """
    if all(t.is_contiguous() for t in tensors):
        n = tensors[0].numel()
        buf = torch.empty(
            len(tensors) * n,
            dtype=out_dtype,
            device=tensors[0].device,
        )
        return [buf[i * n : (i + 1) * n].view(t.shape) for i, t in enumerate(tensors)]
    return [torch.empty_like(t, dtype=out_dtype) for t in tensors]


def _launch_all(srcs, dsts, fn, ot):
    n = srcs[0].numel()
    if len(srcs) <= _K16_SLOTS and n <= _K16_MAX_NUMEL:
        _launch16(srcs, dsts, fn, ot)
    else:
        for a, o in zip(srcs, dsts):
            _launch1(a, o, fn, ot)


def _apply(name, fn, self, inplace):
    tensors = check_tensor_list(self)
    if not tensors:
        return apply_unary(name, self, inplace)
    first = tensors[0]
    n = first.numel()
    if n == 0 or not is_flat_addressable(first):
        return apply_unary(name, self, inplace)
    for t in tensors:
        if (
            t.dtype != first.dtype
            or t.device != first.device
            or t.numel() != n
            or not is_flat_addressable(t)
        ):
            return apply_unary(name, self, inplace)

    if first.dtype in _INT_IN_DTYPES:
        if inplace:
            # ATen refuses to cast a promoted float result into an integral
            # in-place destination; let the generic path raise its message.
            return apply_unary(name, self, inplace)
        # Promote on the host: with an integral pointer the kernel still returns
        # ``2**x - 1`` after ``x.to(tl.float32)`` (the ``exp`` call is lowered as
        # ``exp2``), while the same code is correct for float inputs.  Integral
        # and bool operands never appear in the performance shapes, so one
        # explicit cast is the cheaper trade than handing an integer pointer to
        # the compute kernel.
        srcs = [t.to(torch.float32) for t in tensors]
        dsts = _alloc_dsts(srcs, torch.float32)
        logger.debug(
            "GEMS_KUNLUNXIN _FOREACH_%s%s", name.upper(), "_" if inplace else ""
        )
        with torch_device_fn.device(first.device):
            _launch_all(srcs, dsts, fn, tl.float32)
        return dsts

    if first.dtype not in _FAST_DTYPES:
        return apply_unary(name, self, inplace)

    logger.debug("GEMS_KUNLUNXIN _FOREACH_%s%s", name.upper(), "_" if inplace else "")
    ot = tl_dtype(first.dtype)
    dsts = tensors if inplace else _alloc_dsts(tensors, first.dtype)
    with torch_device_fn.device(first.device):
        _launch_all(tensors, dsts, fn, ot)
    return None if inplace else dsts


def _foreach_expm1(self):
    return _apply("expm1", _expm1_fn, self, False)


def _foreach_expm1_(self):
    return _apply("expm1", _expm1_fn, self, True)


def _foreach_reciprocal(self):
    return _apply("reciprocal", _reciprocal_fn, self, False)


def _foreach_reciprocal_(self):
    return _apply("reciprocal", _reciprocal_fn, self, True)


__all__ = [
    "_foreach_expm1",
    "_foreach_expm1_",
    "_foreach_reciprocal",
    "_foreach_reciprocal_",
]
