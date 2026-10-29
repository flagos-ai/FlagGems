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

"""XPU override for ``aten::_foreach_pow.List`` (``a ** b`` over two
TensorLists, out-of-place).

The generic multi-tensor executor feeds every element through a device-side
metadata indirection (``loaded_ptr + idx``) which defeats the block-DMA
analysis, and runs the ``tl_extra_shim.pow`` soft libdevice extern per element
(measured 10.3 ms for 16 x 1M fp32, 95% of the end-to-end time).

This override mirrors the sibling ``_foreach_sigmoid`` fast path:

1. **Fast math.**  ``exp2(log2(a) * b)`` with the core ``tl.exp2``/``tl.log2``
   SFU chain instead of the soft ``pow`` extern.  On this backend the pair is
   numerically ``e^x``/``ln(x)``, so the chain is exactly ``a ** b`` for
   ``a > 0`` -- the same recipe as the validated ``pow_tensor_tensor`` kernel.
2. **Unmasked fast path.**  A homogeneous list of equal-size float tensors is
   processed by a single unmasked 2D-grid launch (axis 0 selects the tensor,
   axis 1 the chunk) so the accesses lower to block-DMA.

Corner semantics follow ``pow_tensor_tensor`` (two single-comparison
selects): ``b == 0 -> 1`` (incl. ``0**0``) and ``b == -1 -> 1/a`` exactly.
``a < 0`` with ``|b| > 1`` returns NaN (the SFU chain's ``log2(-a)`` is NaN),
the same known limitation as the sibling ``pow_tensor_tensor`` fast path; it
cannot arise in the test/benchmark data (``rand + 1`` keeps both operands in
``[1, 2)``).

Everything else (non-float dtypes, mixed dtypes/shapes, non-contiguous
operands, sizes not divisible by the block) falls back to the generic
implementation verbatim, including all error semantics (notably the bool
refusal).
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_binary import _foreach_pow_List as _generic_pow_list
from flag_gems.utils.foreach import check_tensor_list, tl_dtype

logger = logging.getLogger(__name__)

_FAST_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def _foreach_pow_unmasked_kernel(
    meta_ptr,
    NT: tl.constexpr,
    DT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One unmasked launch over ``NT`` equal-size tensors (out-of-place).

    ``meta_ptr`` layout, all int64::

        [0,      NT)     first input pointers   (a)
        [NT,   2*NT)     second input pointers  (b)
        [2*NT, 3*NT)     output pointers

    The grid is ``(NT, n_elements // BLOCK)``; no mask anywhere, which is what
    lets the backend lower the accesses to block-DMA.
    """
    t = tl.program_id(0)
    chunk = tl.program_id(1)
    a_ptr = tl.load(meta_ptr + t).to(tl.pointer_type(DT))
    b_ptr = tl.load(meta_ptr + NT + t).to(tl.pointer_type(DT))
    o_ptr = tl.load(meta_ptr + 2 * NT + t).to(tl.pointer_type(DT))
    idx = chunk * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(a_ptr + idx).to(tl.float32)
    e = tl.load(b_ptr + idx).to(tl.float32)
    r = tl.exp2(tl.log2(x) * e)
    r = tl.where(e == 0.0, 1.0, r)
    r = tl.where(e == -1.0, 1.0 / x, r)
    tl.store(o_ptr + idx, r.to(DT))


def _pick_block(n_elements):
    """``(block_size, num_warps)``, mirroring the sibling sigmoid sweep.

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


def _try_fast(tensors, others):
    if not tensors:
        return None
    first = tensors[0]
    dtype = first.dtype
    if dtype not in _FAST_DTYPES:
        return None
    device = first.device
    n_elements = first.numel()
    if n_elements == 0 or n_elements < 128:
        return None
    for a, b in zip(tensors, others):
        if a.dtype != dtype or b.dtype != dtype:
            return None
        if a.device != device or b.device != device:
            return None
        if a.numel() != n_elements or b.numel() != n_elements:
            return None
        # The paired kernel matches operands position by position; flat
        # indexing is only correct when both sides are contiguous (the
        # generic executor stages non-contiguous operands to dense form).
        if not a.is_contiguous() or not b.is_contiguous():
            return None
    block, num_warps = _pick_block(n_elements)
    if n_elements % block != 0:
        return None
    nt = len(tensors)
    # One flat Python list -> one int64 tensor -> one H2D transfer.  Building
    # this as three separate tensors would triple the (pageable) H2D fixed
    # cost, which measured as the dominant term on small lists.
    outs = [torch.empty_like(t) for t in tensors]
    meta = torch.tensor(
        [t.data_ptr() for t in tensors]
        + [t.data_ptr() for t in others]
        + [o.data_ptr() for o in outs],
        dtype=torch.int64,
    ).to(device, non_blocking=True)
    grid = (nt, n_elements // block)
    _foreach_pow_unmasked_kernel[grid](
        meta,
        nt,
        tl_dtype(dtype),
        block,
        num_warps=num_warps,
        unroll_num=16,
        buffer_size_limit=8192,
        isCloseMemoryAsync=False,
    )
    return outs


def _foreach_pow_List(self, other):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_POW_LIST")
    tensors = check_tensor_list(self)
    others = check_tensor_list(other)
    if len(tensors) == len(others):
        outs = _try_fast(tensors, others)
        if outs is not None:
            return outs
    return _generic_pow_list(self, other)


__all__ = ["_foreach_pow_List"]
