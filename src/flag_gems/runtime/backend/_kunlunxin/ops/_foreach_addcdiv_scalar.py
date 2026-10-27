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

"""XPU override for ``aten::_foreach_addcdiv.Scalar`` (``self + value * (t1 / t2)``
over three TensorLists, out-of-place).

The generic ternary executor feeds every element through a device-side metadata
indirection and masks every load/store with a runtime ``idx < n_elements``
guard, which blocks the block-DMA lowering (KERNEL_OPT_EXPERIENCE.md #36) and
leaves large memory-bound lists ~40-130x slower than torch.

This override mirrors the sibling ``_foreach_pow_list`` fast path:

1. **Fast math.** ``a + value * (b / c)`` with a plain hardware ``/`` (not the
   soft libdevice extern) computed in fp32, then narrowed to the storage dtype.
2. **Unmasked fast path.** A homogeneous list of equal-size float tensors is
   processed by a single unmasked 2D-grid launch (axis 0 selects the tensor,
   axis 1 the chunk) so the accesses lower to block-DMA.

Everything else (non-float dtypes, mixed dtypes/shapes, non-contiguous
operands, sizes not divisible by the block) falls back to the generic
implementation verbatim, including all error semantics.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_binary import (
    _foreach_addcdiv_Scalar as _generic_addcdiv_scalar,
)
from flag_gems.utils.foreach import check_tensor_list, tl_dtype

logger = logging.getLogger(__name__)

_FAST_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def _foreach_addcdiv_unmasked_kernel(
    meta_ptr,
    NT: tl.constexpr,
    DT: tl.constexpr,
    VALUE,
    BLOCK: tl.constexpr,
):
    """One unmasked launch over ``NT`` equal-size tensors (out-of-place).

    ``meta_ptr`` layout, all int64::

        [0,      NT)     self pointers        (a)
        [NT,   2*NT)     tensor1 pointers     (b)
        [2*NT, 3*NT)     tensor2 pointers     (c)
        [3*NT, 4*NT)     output pointers

    The grid is ``(NT, n_elements // BLOCK)``; no mask anywhere, which is what
    lets the backend lower the accesses to block-DMA.
    """
    t = tl.program_id(0)
    chunk = tl.program_id(1)
    a_ptr = tl.load(meta_ptr + t).to(tl.pointer_type(DT))
    b_ptr = tl.load(meta_ptr + NT + t).to(tl.pointer_type(DT))
    c_ptr = tl.load(meta_ptr + 2 * NT + t).to(tl.pointer_type(DT))
    o_ptr = tl.load(meta_ptr + 3 * NT + t).to(tl.pointer_type(DT))
    idx = chunk * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(a_ptr + idx).to(tl.float32)
    b = tl.load(b_ptr + idx).to(tl.float32)
    c = tl.load(c_ptr + idx).to(tl.float32)
    r = a + VALUE * (b / c)
    tl.store(o_ptr + idx, r.to(DT))


def _pick_block(n_elements):
    """``(block_size, num_warps)``, mirroring the sibling sigmoid/pow sweep.

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


def _try_fast(a_list, b_list, c_list, value):
    if not a_list:
        return None
    first = a_list[0]
    dtype = first.dtype
    if dtype not in _FAST_DTYPES:
        return None
    device = first.device
    n_elements = first.numel()
    if n_elements < 128:
        return None
    for a, b, c in zip(a_list, b_list, c_list):
        if a.dtype != dtype or b.dtype != dtype or c.dtype != dtype:
            return None
        if a.device != device or b.device != device or c.device != device:
            return None
        if (
            a.numel() != n_elements
            or b.numel() != n_elements
            or c.numel() != n_elements
        ):
            return None
        if not a.is_contiguous() or not b.is_contiguous() or not c.is_contiguous():
            return None
    block, num_warps = _pick_block(n_elements)
    if n_elements % block != 0:
        return None
    nt = len(a_list)
    outs = [torch.empty_like(t) for t in a_list]
    meta = torch.tensor(
        [t.data_ptr() for t in a_list]
        + [t.data_ptr() for t in b_list]
        + [t.data_ptr() for t in c_list]
        + [o.data_ptr() for o in outs],
        dtype=torch.int64,
    ).to(device, non_blocking=True)
    grid = (nt, n_elements // block)
    _foreach_addcdiv_unmasked_kernel[grid](
        meta,
        nt,
        tl_dtype(dtype),
        value,
        block,
        num_warps=num_warps,
        unroll_num=16,
        buffer_size_limit=8192,
        isCloseMemoryAsync=False,
    )
    return outs


def _foreach_addcdiv_Scalar(self, tensor1, tensor2, value=1):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_ADDCDIV_SCALAR")
    tensors = check_tensor_list(self)
    t1 = check_tensor_list(tensor1)
    t2 = check_tensor_list(tensor2)
    if len(tensors) == len(t1) == len(t2):
        outs = _try_fast(tensors, t1, t2, float(value))
        if outs is not None:
            return outs
    return _generic_addcdiv_scalar(self, tensor1, tensor2, value=value)


__all__ = ["_foreach_addcdiv_Scalar"]
