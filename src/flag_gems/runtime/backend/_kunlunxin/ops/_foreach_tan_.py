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

"""XPU override for ``aten::_foreach_tan_`` (in-place tan over Tensor[]).

Mirrors the sibling ``_foreach_sigmoid_`` override: a *homogeneous* list of
floating-point tensors is processed by a dedicated *unmasked* 2D-grid kernel in
a single launch, so the loads/stores lower to block-DMA instead of the
masked-memory path the shared executor uses (the mask blocks the block-DMA
lowering; see KERNEL_OPT_EXPERIENCE.md #36).  The element-wise math is the same
``tan`` the generic executor uses (``tl_extra_shim.tan`` on XPU), so results
are bit-identical to the shared implementation -- the win is purely the
unmasked single-launch layout.

The in-place schema returns ``()`` and writes back into the input storage, so
the fast kernel reads and writes the same pointer.  The fast path is taken only
when every precondition the shared executor would enforce for an in-place call
holds: one device, one float dtype (fp16/bf16/fp32), identical power-of-two
element counts, every tensor dense/flat-addressable, and no internal overlap.

Everything else -- complex inputs, mixed dtypes/shapes, gappy or overlapping
views, non-divisible sizes, and integral promotion (which the in-place schema
rejects anyway) -- falls back to ``foreach_unary``, so validation and error
messages stay identical to the generic path.
"""

import logging

import torch
import triton
import triton.language as tl
from triton.language.extra.xpu import libdevice as xpu

from flag_gems.ops._foreach_complex_math import c_tan
from flag_gems.utils.foreach import (
    check_tensor_list,
    foreach_unary,
    has_internal_overlap,
    int_to_float,
    is_flat_addressable,
    tl_dtype,
)

logger = logging.getLogger(__name__)

# tan rejects complex32 but accepts complex64/complex128, all ints and bool
# (matching the generic ``UNARY_OPS["tan"].allowed``).
_NO_COMPLEX32 = frozenset(
    (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    + (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool)
    + (torch.complex64, torch.complex128)
)

# The unmasked fast path is only for the benchmark/real floating dtypes; fp64
# device tensors silently degrade to fp32 on this backend anyway (#4.5.1), and
# complex/integral inputs go through the fallback executor.
_FAST_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def _tan_fast(x):
    # Same math as flag_gems.ops.tan.tan_func (which the generic table row
    # reuses): the XPU libdevice tanf, computed in fp32.
    return xpu.tan(x.to(tl.float32))


@triton.jit
def _foreach_tan_unmasked_kernel(
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
    x = tl.load(ptr + idx).to(tl.float32)
    y = xpu.tan(x)
    tl.store(ptr + idx, y.to(DT))


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
    _foreach_tan_unmasked_kernel[grid](
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


def _foreach_tan_(self):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_TAN_")
    tensors = check_tensor_list(self)
    if _try_fast(tensors):
        # In-place schemas return ``()``; returning the list would make the
        # dispatcher reject the kernel.
        return None
    return foreach_unary(
        tensors,
        _tan_fast,
        complex_fn=c_tan,
        out_dtype_fn=int_to_float,
        inplace=True,
        allowed_dtypes=_NO_COMPLEX32,
    )
