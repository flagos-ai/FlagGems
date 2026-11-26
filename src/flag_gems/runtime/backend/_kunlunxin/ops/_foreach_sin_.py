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

"""XPU override for ``aten::_foreach_sin_`` (in-place sin over a TensorList).

Triton implementation of ``_foreach_sin_`` (in-place sin over a tensor list).

Architecture
------------
* Contiguous (the hot path, and what the benchmark measures): one fused launch
  for up to ``_NSLOT`` list elements.  The grid is ``(max_blocks, n_slots)``,
  ``program_id(1)`` selects the list element through a program-uniform branch,
  and addressing is the division-free flat index ``pid0 * BLOCK + arange``.
* Non-contiguous views: the candidate stores a generic strided kernel here, but
  the registered wrapper below deliberately does not select it.  Non-contiguous
  members keep the shared executor, which owns the in-place view, overlap and
  alias-ordering contract; routing them through a new path would need separate
  evidence for every layout.

Integration contract
--------------------
The fused launch below is used only for a homogeneous list of at most
``_NSLOT`` contiguous, disjoint tensors of the measured float dtypes
(float16 / float32 / bfloat16), which is the shape a foreach benchmark
exercises.  Everything else -- promoting integral inputs (which must be
refused like ATen), complex and float64 inputs, views, internally overlapping
views, lists longer than ``_NSLOT``, mixed dtypes or element counts, and
members sharing a storage (whose in-place ordering is observable) -- keeps the
shared ``_foreach_unary`` executor.  That preserves the master in-place
contract exactly as the executor defines it.
"""

import logging

import torch
import triton
import triton.language as tl

from ..utils.foreach import _STATS, has_internal_overlap
from ._foreach_unary import apply_unary

logger = logging.getLogger(__name__)

# The fused path handles the measured float dtypes only.
_FAST_DTYPES = frozenset({torch.float16, torch.float32, torch.bfloat16})

_BLOCK = 8192
_BLOCK16 = 16384
_F16_WIDE_MIN = 131072
_NUM_WARPS = 16
_NSLOT = 16


@triton.jit
def _apply(p, offs, m):
    x = tl.load(p + offs, mask=m, other=0.0)
    y = tl.sin(x.to(tl.float32))
    tl.store(p + offs, y.to(x.dtype), mask=m)


@triton.jit
def _sin_flat_slots(
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
    slot = tl.program_id(1)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if slot == 0:
        _apply(p0, offs, offs < n0)
    elif slot == 1:
        _apply(p1, offs, offs < n1)
    elif slot == 2:
        _apply(p2, offs, offs < n2)
    elif slot == 3:
        _apply(p3, offs, offs < n3)
    elif slot == 4:
        _apply(p4, offs, offs < n4)
    elif slot == 5:
        _apply(p5, offs, offs < n5)
    elif slot == 6:
        _apply(p6, offs, offs < n6)
    elif slot == 7:
        _apply(p7, offs, offs < n7)
    elif slot == 8:
        _apply(p8, offs, offs < n8)
    elif slot == 9:
        _apply(p9, offs, offs < n9)
    elif slot == 10:
        _apply(p10, offs, offs < n10)
    elif slot == 11:
        _apply(p11, offs, offs < n11)
    elif slot == 12:
        _apply(p12, offs, offs < n12)
    elif slot == 13:
        _apply(p13, offs, offs < n13)
    elif slot == 14:
        _apply(p14, offs, offs < n14)
    else:
        _apply(p15, offs, offs < n15)


@triton.jit
def _apply_fast(p, offs):
    x = tl.load(p + offs)
    y = tl.sin(x.to(tl.float32))
    tl.store(p + offs, y.to(x.dtype), cache_modifier=".cs")


@triton.jit
def _sin_flat_fast(
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
    """Unmasked path: every slot holds exactly the same number of elements,
    a multiple of BLOCK, so every program is fully in range."""
    slot = tl.program_id(1)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if slot == 0:
        _apply_fast(p0, offs)
    elif slot == 1:
        _apply_fast(p1, offs)
    elif slot == 2:
        _apply_fast(p2, offs)
    elif slot == 3:
        _apply_fast(p3, offs)
    elif slot == 4:
        _apply_fast(p4, offs)
    elif slot == 5:
        _apply_fast(p5, offs)
    elif slot == 6:
        _apply_fast(p6, offs)
    elif slot == 7:
        _apply_fast(p7, offs)
    elif slot == 8:
        _apply_fast(p8, offs)
    elif slot == 9:
        _apply_fast(p9, offs)
    elif slot == 10:
        _apply_fast(p10, offs)
    elif slot == 11:
        _apply_fast(p11, offs)
    elif slot == 12:
        _apply_fast(p12, offs)
    elif slot == 13:
        _apply_fast(p13, offs)
    elif slot == 14:
        _apply_fast(p14, offs)
    else:
        _apply_fast(p15, offs)


@triton.jit
def _sin_strided_kernel(x_ptr, numel, s0, s1, s2, s3, d1, d2, d3, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < numel
    rem = offs
    i3 = rem % d3
    rem = rem // d3
    i2 = rem % d2
    rem = rem // d2
    i1 = rem % d1
    i0 = rem
    addr = i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3
    x = tl.load(x_ptr + addr, mask=m, other=0.0)
    y = tl.sin(x.to(tl.float32))
    tl.store(x_ptr + addr, y.to(x.dtype), mask=m)


def _collapse(shape, stride):
    """Merge adjacent compatible dims and drop size-1 dims (address preserving)."""
    shape = list(shape)
    stride = list(stride)
    changed = True
    while changed:
        changed = False
        for i in range(len(shape) - 1, -1, -1):
            if shape[i] == 1:
                del shape[i]
                del stride[i]
                changed = True
        for i in range(len(shape) - 2, -1, -1):
            if stride[i] == stride[i + 1] * shape[i + 1]:
                shape[i] = shape[i] * shape[i + 1]
                # the merged dim must inherit the INNER dim's stride
                stride[i] = stride[i + 1]
                del shape[i + 1]
                del stride[i + 1]
                changed = True
    if not shape:
        shape = [1]
        stride = [0]
    return shape, stride


def _strided(t):
    numel = t.numel()
    shape, stride = _collapse(t.shape, t.stride())
    missing = 4 - len(shape)
    if missing > 0:
        shape = [1] * missing + shape
        stride = [0] * missing + stride
    _sin_strided_kernel[((numel + _BLOCK - 1) // _BLOCK,)](
        t,
        numel,
        stride[0],
        stride[1],
        stride[2],
        stride[3],
        shape[1],
        shape[2],
        shape[3],
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
    )


def run(self):
    if torch.is_tensor(self):
        tensors = [self]
    else:
        tensors = list(self)
    if not tensors:
        return None

    # Validate and classify in one pass: nothing is mutated until every element
    # has been checked (ATen is all-or-nothing), and each numel is read once.
    flat = []
    numels = []
    views = []
    seen = set()
    aliased = False
    for t in tensors:
        if not torch.is_tensor(t):
            raise RuntimeError(
                "torch._foreach_sin_ expects each element to be a Tensor, but got "
                + type(t).__name__
            )
        if not t.is_floating_point():
            raise RuntimeError(
                "result type "
                + str(t.dtype).split(".")[-1]
                + " can't be cast to the desired output type"
            )
        cnt = t.numel()
        if cnt == 0:
            continue
        if t.is_contiguous():
            p = t.data_ptr()
            if p in seen:
                aliased = True
            else:
                seen.add(p)
            flat.append(t)
            numels.append(cnt)
        else:
            views.append(t)

    if aliased:
        # The reference applies sin sequentially, so a repeated element must be
        # transformed once per occurrence: fall back to ordered single launches.
        views = flat + views
        flat = []
        numels = []

    n = len(flat)
    if n:
        base = flat[0]
        start = 0
        while start < n:
            stop = start + _NSLOT
            chunk = flat[start:stop]
            chunk_numels = numels[start:stop]
            start = stop
            k = len(chunk)
            # float16 half-sized elements fit a twice-wider vector inside the
            # compiler's unified-SRAM budget; float32/bfloat16 do not (proven by
            # the failed all-dtype BLOCK=16384 attempt), so size per dtype.
            block = _BLOCK
            if chunk[0].dtype is torch.float16 and k == _NSLOT:
                d0 = chunk[0].dtype
                same = True
                for t in chunk:
                    if t.dtype is not d0:
                        same = False
                        break
                # Only widen while the wider tile still fills the device: 16 slots
                # times at least _F16_WIDE_BLOCKS blocks per slot over 8 SMs.
                if same and chunk_numels[0] >= _F16_WIDE_MIN:
                    block = _BLOCK16
            mb = 0
            for v in chunk_numels:
                b = (v + block - 1) // block
                if b > mb:
                    mb = b
            args = chunk
            if k < _NSLOT:
                args = chunk + [base] * (_NSLOT - k)
                chunk_numels = chunk_numels + [0] * (_NSLOT - k)
            if (
                mb * block == chunk_numels[0]
                and chunk_numels.count(chunk_numels[0]) == _NSLOT
            ):
                # Uniform, BLOCK-divisible chunk: no element can be out of range,
                # so the mask, its compare and the load's `other` value are dead.
                _sin_flat_fast[(mb, k)](
                    *args,
                    BLOCK=block,
                    num_warps=32 if block == _BLOCK16 else _NUM_WARPS,
                )
                continue
            _sin_flat_slots[(mb, k)](
                *args,
                *chunk_numels,
                BLOCK=block,
                num_warps=_NUM_WARPS,
            )

    for t in views:
        _strided(t)
    return None


def _is_fusable(tensors):
    """Whether one fused launch reproduces the per-element in-place semantics.

    Requires a non-empty list of at most ``_NSLOT`` contiguous float members
    with one dtype, one element count and one device, none internally
    overlapping, and pairwise disjoint storages.  Anything else keeps the
    shared executor, which owns the promotion and overlap refusals, complex and
    float64 handling, view write-through and the ordering of aliased members.
    """
    if not tensors or len(tensors) > _NSLOT:
        return False
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        return False
    first = tensors[0]
    dtype = first.dtype
    numel = first.numel()
    device = first.device
    if dtype not in _FAST_DTYPES or numel == 0 or not first.is_contiguous():
        return False
    storages = set()
    for t in tensors:
        if t.dtype != dtype or t.numel() != numel or t.device != device:
            return False
        if not t.is_contiguous() or has_internal_overlap(t):
            return False
        storage = t.untyped_storage().data_ptr()
        if storage in storages:
            # Two members share a storage: in-place ordering is observable.
            return False
        storages.add(storage)
    return True


def _foreach_sin_(self):
    """In-place ``aten::_foreach_sin_`` over a TensorList."""
    logger.debug("GEMS _FOREACH_SIN_")
    tensors = [self] if torch.is_tensor(self) else list(self)
    if _is_fusable(tensors):
        _STATS.reset()
        run(tensors)
        _STATS.executor_launches += 1
        _STATS.groups += 1
        return None
    return apply_unary("sin", tensors, inplace=True)
