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

"""XPU overrides for the ``.Tensor`` overloads of ``_foreach_add`` / ``_foreach_mul``.

Approach
--------
Same recipe as the already-merged ``_foreach_binary_scalar.py`` (PR #7051):

* ``_foreach16_tensor_kernel``: **every pointer stays a kernel argument**, and
  the second operand is passed as **one device pointer** that the kernel reads
  once with ``tl.load`` before broadcasting it.  No ``expand_as``, no
  ``contiguous``, no ``.item()`` (so no synchronisation).  An ``NT``-constexpr
  guarded unrolled chain lets one instruction stream handle up to sixteen
  equal-numel tensors per launch.
* ``_foreach1_tensor_kernel``: large tensors use the big-tile recipe of
  ``clamp_max`` (``unroll_num=8``, ``buffer_size_limit=4096``).
* Split threshold ``_K16_MAX_NUMEL = 1 << 19``: at or below it the whole list
  goes out in one launch, above it each tensor gets its own launch.

Every other case (non-0-dim operand, mixed dtype/device, non-contiguous input,
integral promotion, empty list, rejected dtype) falls back untouched to the
generic ``_apply_tensor``, so semantics and log records are unchanged.
"""

import os

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_binary import BINARY_OPS, _apply_tensor, _check_alpha, _log
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.foreach import check_tensor_list, is_flat_addressable, tl_dtype

# Only these three float dtypes take the fast path; the integral promotion and
# wrapping rules, and the complex kernels, stay with the generic implementation.
_FAST_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

_K16_SLOTS = 16
# A single launch over equal-numel tensors is used only while the list fits the
# threshold below; beyond it the 16-way unrolled chain is replaced by one launch
# per tensor.  GEMS_XPU_FOREACH_K16_MAX overrides the threshold.
_K16_MAX_NUMEL = int(os.environ.get("GEMS_XPU_FOREACH_K16_MAX", "0") or 0) or (
    1 << 19
)


@triton.jit
def _foreach16_tensor_kernel(
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
    b_ptr,
    n,
    fn: tl.constexpr,
    NT: tl.constexpr,
    ALPHA: tl.constexpr,
    OT: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    """One launch, up to 16 equal-numel tensors against a broadcast scalar.

    ``b_ptr`` is a 1-element device tensor (the ATen ``.Tensor`` operand).  It is
    read *once* per program and broadcast from a register, which is what replaces
    the generic path's ``expand_as`` + ``_dense`` materialisation.  Every data
    pointer stays a kernel argument, so the block-DMA analysis can prove
    stride 1 (same recipe as ``_foreach_binary_scalar._foreach16_kernel``).
    """
    off = tl.program_id(0) * BLOCK
    idx = off + tl.arange(0, BLOCK)
    b = tl.load(b_ptr)
    # ALPHA is a constexpr, so this multiply compiles away for the default 1.
    if ALPHA != 1:
        b = b * ALPHA
    b = b.to(OT)
    if EVEN:
        if NT > 0:
            tl.store(o0 + idx, fn(tl.load(a0 + idx).to(OT), b).to(OT))
        if NT > 1:
            tl.store(o1 + idx, fn(tl.load(a1 + idx).to(OT), b).to(OT))
        if NT > 2:
            tl.store(o2 + idx, fn(tl.load(a2 + idx).to(OT), b).to(OT))
        if NT > 3:
            tl.store(o3 + idx, fn(tl.load(a3 + idx).to(OT), b).to(OT))
        if NT > 4:
            tl.store(o4 + idx, fn(tl.load(a4 + idx).to(OT), b).to(OT))
        if NT > 5:
            tl.store(o5 + idx, fn(tl.load(a5 + idx).to(OT), b).to(OT))
        if NT > 6:
            tl.store(o6 + idx, fn(tl.load(a6 + idx).to(OT), b).to(OT))
        if NT > 7:
            tl.store(o7 + idx, fn(tl.load(a7 + idx).to(OT), b).to(OT))
        if NT > 8:
            tl.store(o8 + idx, fn(tl.load(a8 + idx).to(OT), b).to(OT))
        if NT > 9:
            tl.store(o9 + idx, fn(tl.load(a9 + idx).to(OT), b).to(OT))
        if NT > 10:
            tl.store(o10 + idx, fn(tl.load(a10 + idx).to(OT), b).to(OT))
        if NT > 11:
            tl.store(o11 + idx, fn(tl.load(a11 + idx).to(OT), b).to(OT))
        if NT > 12:
            tl.store(o12 + idx, fn(tl.load(a12 + idx).to(OT), b).to(OT))
        if NT > 13:
            tl.store(o13 + idx, fn(tl.load(a13 + idx).to(OT), b).to(OT))
        if NT > 14:
            tl.store(o14 + idx, fn(tl.load(a14 + idx).to(OT), b).to(OT))
        if NT > 15:
            tl.store(o15 + idx, fn(tl.load(a15 + idx).to(OT), b).to(OT))
    else:
        m = idx < n
        if NT > 0:
            tl.store(
                o0 + idx, fn(tl.load(a0 + idx, mask=m).to(OT), b).to(OT), mask=m
            )
        if NT > 1:
            tl.store(
                o1 + idx, fn(tl.load(a1 + idx, mask=m).to(OT), b).to(OT), mask=m
            )
        if NT > 2:
            tl.store(
                o2 + idx, fn(tl.load(a2 + idx, mask=m).to(OT), b).to(OT), mask=m
            )
        if NT > 3:
            tl.store(
                o3 + idx, fn(tl.load(a3 + idx, mask=m).to(OT), b).to(OT), mask=m
            )
        if NT > 4:
            tl.store(
                o4 + idx, fn(tl.load(a4 + idx, mask=m).to(OT), b).to(OT), mask=m
            )
        if NT > 5:
            tl.store(
                o5 + idx, fn(tl.load(a5 + idx, mask=m).to(OT), b).to(OT), mask=m
            )
        if NT > 6:
            tl.store(
                o6 + idx, fn(tl.load(a6 + idx, mask=m).to(OT), b).to(OT), mask=m
            )
        if NT > 7:
            tl.store(
                o7 + idx, fn(tl.load(a7 + idx, mask=m).to(OT), b).to(OT), mask=m
            )
        if NT > 8:
            tl.store(
                o8 + idx, fn(tl.load(a8 + idx, mask=m).to(OT), b).to(OT), mask=m
            )
        if NT > 9:
            tl.store(
                o9 + idx, fn(tl.load(a9 + idx, mask=m).to(OT), b).to(OT), mask=m
            )
        if NT > 10:
            tl.store(
                o10 + idx, fn(tl.load(a10 + idx, mask=m).to(OT), b).to(OT), mask=m
            )
        if NT > 11:
            tl.store(
                o11 + idx, fn(tl.load(a11 + idx, mask=m).to(OT), b).to(OT), mask=m
            )
        if NT > 12:
            tl.store(
                o12 + idx, fn(tl.load(a12 + idx, mask=m).to(OT), b).to(OT), mask=m
            )
        if NT > 13:
            tl.store(
                o13 + idx, fn(tl.load(a13 + idx, mask=m).to(OT), b).to(OT), mask=m
            )
        if NT > 14:
            tl.store(
                o14 + idx, fn(tl.load(a14 + idx, mask=m).to(OT), b).to(OT), mask=m
            )
        if NT > 15:
            tl.store(
                o15 + idx, fn(tl.load(a15 + idx, mask=m).to(OT), b).to(OT), mask=m
            )


@triton.jit
def _foreach1_tensor_kernel(
    a_ptr,
    b_ptr,
    o_ptr,
    num_tasks,
    tiles_per_cta,
    fn: tl.constexpr,
    ALPHA: tl.constexpr,
    OT: tl.constexpr,
    tile_size: tl.constexpr,
    EVEN: tl.constexpr,
):
    """One tensor per launch, one tile per CTA.

    The broadcast scalar is loaded once per program and reused for every tile.
    ``EVEN`` selects an unmasked whole-tile body: on this backend a masked
    load/store is lowered element-wise, which loses the block-DMA path.
    ``EVEN`` therefore implies that the tile count is an exact multiple of the
    CTA count, so no runtime tile-id guard is needed -- that form does not lower
    on this backend.
    """
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    b = tl.load(b_ptr)
    if ALPHA != 1:
        b = b * ALPHA
    b = b.to(OT)
    for j in range(0, tiles_per_cta):
        tid = (pid + j * num_ctas) * tile_size + tl.arange(0, tile_size)
        if EVEN:
            tl.store(o_ptr + tid, fn(tl.load(a_ptr + tid).to(OT), b).to(OT))
        else:
            m = tid < num_tasks
            tl.store(
                o_ptr + tid,
                fn(tl.load(a_ptr + tid, mask=m).to(OT), b).to(OT),
                mask=m,
            )


# ---------------------------------------------------------------------------
# launch helpers
# ---------------------------------------------------------------------------


def _pick_block(n: int):
    """``(block, even)``: the largest power-of-two block that divides ``n``."""
    for block in (8192, 4096, 2048, 1024):
        if n % block == 0:
            return block, True
    return 1024, False


def _launch16(tensors, dsts, other, alpha, fn, ot):
    n = tensors[0].numel()
    block, even = _pick_block(n)
    nt = len(tensors)
    nt16 = min(nt, _K16_SLOTS)
    # The dead ``if NT > i`` guards compile the padding slots away; their
    # argument values only have to be *valid* pointers, so the live slots are
    # reused instead of allocating a shadow buffer.
    a = list(tensors[:nt16]) + [tensors[0]] * (_K16_SLOTS - nt16)
    o = list(dsts[:nt16]) + [dsts[0]] * (_K16_SLOTS - nt16)
    _foreach16_tensor_kernel[(triton.cdiv(n, block),)](
        *a,
        *o,
        other,
        n,
        fn,
        nt16,
        float(alpha),
        ot,
        block,
        even,
        num_warps=8,
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
    """``(num_ctas, tiles_per_cta, tile_size, even)``.

    ``even`` requires both tile alignment and an exact CTA/tile multiple: the
    unmasked body has no runtime tile-id guard (that form does not lower here).
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
        num_ctas = want
        even = False
    tiles_per_cta = triton.cdiv(num_tiles, num_ctas)
    return num_ctas, tiles_per_cta, tile_size, even


def _launch1(src, dst, other, alpha, fn, ot):
    n = src.numel()
    num_ctas, tiles_per_cta, tile_size, even = _big_tile_plan(n)
    num_warps = 16 if tile_size >= 4096 else (8 if tile_size >= 2048 else 4)
    _foreach1_tensor_kernel[(num_ctas,)](
        src,
        other,
        dst,
        n,
        tiles_per_cta,
        fn,
        float(alpha),
        ot,
        tile_size,
        even,
        num_warps=num_warps,
        unroll_num=_UNROLL,
        buffer_size_limit=_BUF,
    )


def _alloc_dsts(tensors, out_dtype):
    """Outputs for the out-of-place path, from one allocation when possible.

    ``[torch.empty_like(t) for t in tensors]`` costs one allocator round trip
    per tensor, which dominates the small-shape cost.  A single flat buffer
    sliced into views has the same layout because every tensor here is
    contiguous, and it keeps the fast path allocation-free per tensor.
    """
    if all(t.is_contiguous() for t in tensors):
        n = tensors[0].numel()
        buf = torch.empty(
            len(tensors) * n, dtype=out_dtype, device=tensors[0].device
        )
        return [buf[i * n : (i + 1) * n].view(t.shape) for i, t in enumerate(tensors)]
    return [torch.empty_like(t, dtype=out_dtype) for t in tensors]


def _try_fast(name, self, other, inplace, alpha):
    """Run the fast path; return ``(handled, result)``.

    ``handled=False`` means the generic path must decide (it owns the ATen error
    messages and the promotion rules).  The dtype policy is deliberately
    *shared* with it: the output dtype is resolved with the same
    ``torch.result_type`` call and the same ``out_dtype_fn``, so a case that
    would promote (or raise) never enters here.
    """
    declined = (False, None)
    op = BINARY_OPS[name]
    if not isinstance(other, torch.Tensor) or other.numel() != 1:
        return declined
    tensors = check_tensor_list(self)
    if not tensors:
        return declined
    first = tensors[0]
    if first.dtype not in _FAST_DTYPES:
        return declined
    if other.device != first.device:
        return declined
    n = first.numel()
    if n == 0:
        return declined
    for t in tensors:
        if t.dtype != first.dtype or t.device != first.device:
            return declined
        if t.numel() != n:
            return declined
        if not is_flat_addressable(t):
            return declined
        if op.allowed is not None and t.dtype not in op.allowed:
            # Let the generic path raise ATen's own dtype error.
            return declined
    if alpha != 1:
        _check_alpha(alpha, first.dtype)
    # Same promotion rule as the generic path.  A 0-dim operand is a wrapped
    # number to ``result_type`` (fp16 list * fp32 0-dim stays fp16), while a
    # shape-(1,) operand really does promote -- both cases are handled by
    # simply taking the resolved dtype.
    out_dtypes = [torch.result_type(t, other) for t in tensors]
    if op.out_dtype_fn is not None:
        out_dtypes = [op.out_dtype_fn(dt) for dt in out_dtypes]
    if any(dt != out_dtypes[0] for dt in out_dtypes):
        return declined
    out_dtype = out_dtypes[0]
    if out_dtype not in _FAST_DTYPES:
        return declined
    if inplace and out_dtype != first.dtype:
        # ATen raises "result type can't be cast"; let it produce the message.
        return declined

    ot = tl_dtype(out_dtype)
    dsts = tensors if inplace else _alloc_dsts(tensors, out_dtype)
    _log(name, "Tensor", inplace)
    with torch_device_fn.device(first.device):
        if len(tensors) <= _K16_SLOTS and n <= _K16_MAX_NUMEL:
            _launch16(tensors, dsts, other, alpha, op.fn, ot)
        else:
            for a, o in zip(tensors, dsts):
                _launch1(a, o, other, alpha, op.fn, ot)
    return True, (None if inplace else dsts)


# ---------------------------------------------------------------------------
# wrappers, one per registered ATen overload
# ---------------------------------------------------------------------------


def _run(name, self, other, inplace, alpha=1):
    handled, result = _try_fast(name, self, other, inplace, alpha)
    if handled:
        return result
    return _apply_tensor(name, self, other, inplace, alpha)


def _make(name, inplace, with_alpha):
    suffix = "_" if inplace else ""
    attr = f"_foreach_{name}{suffix}_Tensor"

    if with_alpha:

        def wrapper(self, other, *, alpha=1):
            return _run(name, self, other, inplace, alpha)

    else:

        def wrapper(self, other):
            return _run(name, self, other, inplace)

    wrapper.__name__ = attr
    wrapper.__qualname__ = attr
    return wrapper


_foreach_add_Tensor = _make("add", False, True)
_foreach_add__Tensor = _make("add", True, True)
_foreach_mul__Tensor = _make("mul", True, False)

__all__ = [
    "_foreach_add_Tensor",
    "_foreach_add__Tensor",
    "_foreach_mul__Tensor",
]
