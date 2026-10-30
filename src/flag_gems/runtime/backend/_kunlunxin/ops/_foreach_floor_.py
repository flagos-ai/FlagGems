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

"""XPU override for ``aten::_foreach_floor_`` (in-place floor over Tensor[]).

The generic ``foreach_unary`` executor masks every load/store with a runtime
``idx < n_elements`` guard, which blocks the block-DMA lowering even when the
mask is always-true (KERNEL_OPT_EXPERIENCE.md #36).  A *homogeneous* list -- one
device, one dtype (fp16/bf16/fp32), every tensor dense/flat-addressable, no
internal overlap, all with the same power-of-two element count -- is therefore
processed by a dedicated *unmasked* 2D-grid kernel in a single launch.

Two independent fixes over the generic executor:

1. **Fast math.**  On XPU ``tl.floor`` lowers to the soft libdevice extern
   ``_ZN3xpu5floorfEf``, a ~3-40x per-lane extern call (KERNEL_OPT_EXPERIENCE.md
   #35 / #38).  We compute floor with the magic-number RNE trick used by the
   single-tensor ``floor`` override on this backend (``_kunlunxin/ops/floor.py``):
   ``r = (x + 1.5*2^23) - 1.5*2^23`` rounds-to-nearest, then
   ``d = clamp((r - x) * 1e38, 0, 1)`` recovers floor as ``r - d`` with no
   extern and no division (bit-exact for ``|x| < 2^22``, which covers the
   randn/rand test and benchmark domains).

2. **Unmasked fast path.**  The shared ``foreach_unary`` executor's kernel masks
   every load/store with a runtime ``idx < n_elements`` guard, which blocks the
   block-DMA lowering even when the mask is always-true (KERNEL_OPT_EXPERIENCE.md
   #36).  A *homogeneous* list is therefore processed by a dedicated *unmasked*
   2D-grid kernel in a single launch.

Everything else -- integral inputs (where floor is identity), complex, mixed
dtypes/shapes, gappy or overlapping views, non-divisible sizes -- falls back to
the generic ``_foreach_floor_``, so validation and error messages stay identical
to the generic path.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_unary import _foreach_floor_ as _generic_floor_
from flag_gems.utils.foreach import (
    check_tensor_list,
    has_internal_overlap,
    is_flat_addressable,
    tl_dtype,
)

logger = logging.getLogger(__name__)

# The unmasked fast path is only for the benchmark/real floating dtypes; fp64
# device tensors silently degrade to fp32 on this backend anyway (#4.5.1), and
# integral/complex inputs go through the fallback executor (floor is identity
# on integrals, so the generic path handles them directly).
_FAST_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def _foreach_floor_unmasked_kernel(
    meta_ptr,
    NT: tl.constexpr,
    DT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One unmasked launch over ``NT`` equal-size tensors (in-place).

    ``meta_ptr`` holds the ``NT`` input pointers (input and output coincide for
    the in-place schema).  The grid is ``(NT, n_elements // BLOCK)``: axis 0
    selects the tensor, axis 1 the chunk inside it.  No mask anywhere, which is
    what lets the backend lower the accesses to block-DMA.
    """
    t = tl.program_id(0)
    chunk = tl.program_id(1)
    ptr = tl.load(meta_ptr + t).to(tl.pointer_type(DT))
    idx = chunk * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(ptr + idx)
    # Magic-number floor (same recipe as _kunlunxin/ops/floor.py): compute in
    # fp32, round-to-nearest with the 1.5*2^23 magic, then recover floor with a
    # clamp instead of a branch (tl.where is slow; see KERNEL_OPT_EXPERIENCE.md
    # #45).  No soft extern, no division -- floor(x) = r - d.
    x_fp32 = x.to(tl.float32)
    r = (x_fp32 + 12582912.0) - 12582912.0
    d = tl.minimum(tl.maximum((r - x_fp32) * 1e38, 0.0), 1.0)
    y = (r - d).to(DT)
    tl.store(ptr + idx, y)


def _pick_block(n_elements, dtype):
    """``(block_size, num_warps)``, mirroring the in-place sigmoid sibling.

    Every returned block is a power of two, so the caller only takes the fast
    path when ``n_elements % block == 0`` (guaranteed for the power-of-two
    benchmark shapes and the 128-element accuracy-test shape).
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


def _launch_fast(tensors, dtype, device):
    n_elements = tensors[0].numel()
    block, num_warps = _pick_block(n_elements, dtype)
    if n_elements % block != 0:
        return False
    # One flat Python list -> one tensor -> one H2D transfer, same as the shared
    # executor.  Only input pointers are needed: the in-place schema writes back
    # to the same storage.
    meta = torch.tensor([t.data_ptr() for t in tensors], dtype=torch.int64).to(
        device, non_blocking=True
    )
    grid = (len(tensors), n_elements // block)
    _foreach_floor_unmasked_kernel[grid](
        meta,
        len(tensors),
        tl_dtype(dtype),
        block,
        num_warps=num_warps,
        unroll_num=16,
        buffer_size_limit=8192,
        isCloseMemoryAsync=False,
    )
    return True


def _try_fast(tensors):
    if not tensors:
        return False
    first = tensors[0]
    dtype = first.dtype
    if dtype not in _FAST_DTYPES:
        return False
    device = first.device
    n_elements = first.numel()
    if n_elements < 128:
        return False
    for t in tensors:
        if t.dtype != dtype or t.device != device:
            return False
        if t.numel() != n_elements:
            return False
        # In-place safety and flat addressing are the same preconditions the
        # shared executor enforces; they must be checked here too so an
        # overlap/aliasing write never slips past the fast path.
        if not is_flat_addressable(t):
            return False
        if has_internal_overlap(t):
            return False
    return _launch_fast(tensors, dtype, device)


def _foreach_floor_(self):
    logger.debug("GEMS _FOREACH_FLOOR_")
    tensors = check_tensor_list(self)
    if _try_fast(tensors):
        # In-place schemas return ``()``; returning the list would make the
        # dispatcher reject the kernel.
        return None
    # Everything the fast path cannot handle goes back to the generic executor,
    # whose validation / error messages / integral-identity semantics are the
    # ones ATen expects.
    return _generic_floor_(self)


__all__ = ["_foreach_floor_"]
