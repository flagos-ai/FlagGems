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

"""XPU override for ``aten::_foreach_sub_.List`` (``self[i] = self[i] -
alpha * other[i]`` over two TensorLists, in-place).

The in-place twin of ``_foreach_sub_list.py``; the generic multi-tensor executor
feeds every element through a device-side metadata indirection (``loaded_ptr +
idx``) and masks every load/store with a runtime ``idx < n_elements`` predicate.
That defeats the block-DMA analysis and runs the masked-memory path even when
the mask is everywhere true (measured 3850 ms of 3958 ms for 16 x 16M fp16 /
16 x 1M fp32 / 16 x 16M fp32 across 60 iterations, 97.3% of the end-to-end
time).

This override mirrors the sibling ``_foreach_add_list_.py`` fast path:

1. **Exact arithmetic.**  ``x - alpha * e`` computed directly in the operand
   dtype (no ``.to(tl.float32)`` round-trip).  Unlike ``pow``/``sigmoid``,
   ``sub`` is not transcendental, and promoting fp16/bf16 operands to fp32
   before subtracting would introduce a second rounding step that the accuracy
   tests reject.  ``alpha`` is a ``tl.constexpr`` so the multiply compiles away
   for the default ``alpha == 1``.
2. **Unmasked fast path.**  A homogeneous list of equal-size float tensors is
   processed by a single unmasked 2D-grid launch (axis 0 selects the tensor,
   axis 1 the chunk) so the accesses lower to block-DMA.

In-place specifics (vs. the out-of-place sibling):

* **No output allocation.**  ``self[i]`` is read and written back in place; the
  metadata buffer carries only the ``2 * NT`` input pointers (``self`` then
  ``other``).  The in-place schema returns ``()``, so the wrapper returns
  ``None`` rather than a list.
* **Dense preconditions.**  Both operands must be contiguous: flat indexing
  pairs position by position, and a contiguous tensor is by construction dense
  and free of internal overlap (the generic executor stages non-contiguous
  operands to dense form).

Everything else (non-float dtypes, mixed dtypes/shapes, non-contiguous
operands, sizes not divisible by the block, empty lists, length mismatch)
falls back to the generic implementation verbatim, including all error
semantics (notably the bool refusal and the integral ``alpha`` check).
``has_internal_overlap`` and the dtype checks run on the host only, so a skip
costs nothing on the device.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_binary import _foreach_sub__List as _generic_sub_list
from flag_gems.utils.foreach import check_tensor_list, has_internal_overlap, tl_dtype

logger = logging.getLogger(__name__)

_FAST_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def _foreach_sub_inplace_kernel(
    meta_ptr,
    NT: tl.constexpr,
    DT: tl.constexpr,
    ALPHA: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One unmasked launch over ``NT`` equal-size tensors (in-place).

    ``meta_ptr`` layout, all int64::

        [0,      NT)     first input pointers   (a, also the output)
        [NT,   2*NT)     second input pointers  (b)

    The grid is ``(NT, n_elements // BLOCK)``: axis 0 selects the tensor, axis 1
    the chunk inside it.  No mask anywhere, which is what lets the backend lower
    the accesses to block-DMA.  Both operands are read before the store, so a
    ``self``/``other`` element-wise aliasing (``a is b``) stays correct.
    """
    t = tl.program_id(0)
    chunk = tl.program_id(1)
    a_ptr = tl.load(meta_ptr + t).to(tl.pointer_type(DT))
    b_ptr = tl.load(meta_ptr + NT + t).to(tl.pointer_type(DT))
    idx = chunk * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(a_ptr + idx)
    e = tl.load(b_ptr + idx)
    if ALPHA != 1:
        e = (e * ALPHA).to(DT)
    tl.store(a_ptr + idx, (x - e).to(DT))


def _pick_block(n_elements):
    """``(block_size, num_warps)``, mirroring the sibling add/pow/sigmoid sweep.

    Every returned block is a power of two, so the caller only takes the fast
    path when ``n_elements % block == 0``.
    """
    if n_elements >= 262_144:
        return 131072, 32
    if n_elements >= 65_536:
        return 32768, 16
    if n_elements >= 16_384:
        return 16384, 16
    if n_elements >= 4_096:
        return 4096, 8
    if n_elements >= 1_024:
        return 1024, 4
    return 128, 4


def _try_fast(tensors, others, alpha):
    if not tensors:
        return False
    first = tensors[0]
    dtype = first.dtype
    if dtype not in _FAST_DTYPES:
        return False
    device = first.device
    n_elements = first.numel()
    if n_elements == 0 or n_elements < 128:
        return False
    for a, b in zip(tensors, others):
        if a.dtype != dtype or b.dtype != dtype:
            return False
        if a.device != device or b.device != device:
            return False
        if a.numel() != n_elements or b.numel() != n_elements:
            return False
        # The paired kernel matches operands position by position; flat
        # indexing is only correct when both sides are contiguous (the
        # generic executor stages non-contiguous operands to dense form).
        # Contiguity also guarantees dense, overlap-free storage, which the
        # in-place schema requires before writing back.
        if not a.is_contiguous() or not b.is_contiguous():
            return False
        if has_internal_overlap(a):
            return False
    block, num_warps = _pick_block(n_elements)
    if n_elements % block != 0:
        return False
    nt = len(tensors)
    # One flat Python list -> one int64 tensor -> one H2D transfer.  Only the
    # input pointers are needed: the in-place schema writes back to the same
    # storage, so no output tensor is ever allocated.
    meta = torch.tensor(
        [t.data_ptr() for t in tensors] + [t.data_ptr() for t in others],
        dtype=torch.int64,
    ).to(device, non_blocking=True)
    grid = (nt, n_elements // block)
    _foreach_sub_inplace_kernel[grid](
        meta,
        nt,
        tl_dtype(dtype),
        alpha,
        block,
        num_warps=num_warps,
        unroll_num=16,
        buffer_size_limit=8192,
        isCloseMemoryAsync=False,
    )
    return True


def _foreach_sub__List(self, other, *, alpha=1):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_SUB_")
    tensors = check_tensor_list(self)
    others = check_tensor_list(other)
    if len(tensors) == len(others):
        if _try_fast(tensors, others, alpha):
            # In-place schemas return ``()``; returning the list would make the
            # dispatcher reject the kernel.
            return None
    return _generic_sub_list(self, other, alpha=alpha)


__all__ = ["_foreach_sub__List"]
