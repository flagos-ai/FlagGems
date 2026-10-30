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

"""XPU override for ``aten::_foreach_tan`` (out-of-place tan over Tensor[]).

Two independent fixes over the generic executor, mirroring the sibling
``_foreach_sigmoid`` override:

1. **Fast math.**  The generic executor reuses
   ``flag_gems.ops.tan.tan_func`` whose math is ``tl_extra_shim.tan``, which on
   XPU lowers to the soft libdevice extern ``_ZN3xpu5tanfEf`` -- a ~15-40x
   per-lane extern call (see KERNEL_OPT_EXPERIENCE.md #35).  We compute
   ``tan(x) = sin(x) / cos(x)`` with the *core* ``tl.sin``/``tl.cos`` (native
   ``math.sin``/``math.cos`` in the LLVM IR, the same primitives already used
   by the single-tensor ``sin``/``cos`` overrides on this backend).

   The math is computed in fp32 even for fp16/bf16 inputs, matching the generic
   executor's ``.to(OUT_DTYPE)`` promotion before the transcendental.

2. **Unmasked fast path.**  The shared ``foreach_unary`` executor's kernel masks
   every load/store with a runtime ``idx < n_elements`` guard, which blocks the
   block-DMA lowering even when the mask is always-true (KERNEL_OPT_EXPERIENCE.md
   #36).  A *homogeneous* list -- one device, one dtype (fp16/bf16/fp32), every
   tensor flat-addressable, all with the same power-of-two element count -- is
   therefore processed by a dedicated *unmasked* 2D-grid kernel in a single
   launch, using the large-block recipe tuned by the sigmoid on-device sweep
   (see ``_pick_block``).

   Everything else -- complex inputs, mixed dtypes/shapes, gappy views,
   non-divisible sizes, integral promotion, and lists whose tensors are smaller
   than ``_FAST_MIN_ELEMENTS`` (where host work dominates and the gate keeps the
   leaner one-transfer fallback) -- falls back to ``foreach_unary``, so the
   validation and error messages stay identical to the generic path.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_complex_math import c_tan
from flag_gems.utils.foreach import (
    check_tensor_list,
    foreach_unary,
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
    # tan(x) = sin(x) / cos(x); core tl.sin/tl.cos, not tl_extra_shim.tan.
    xf = x.to(tl.float32)
    return tl.sin(xf) / tl.cos(xf)


@triton.jit
def _foreach_tan_unmasked_kernel(
    meta_ptr,
    NT: tl.constexpr,
    DT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One unmasked launch over ``NT`` equal-size tensors (out-of-place).

    ``meta_ptr`` layout, all int64::

        [0,   NT)   input pointers
        [NT, 2*NT)  output pointers

    The grid is ``(NT, n_elements // BLOCK)``; no mask anywhere, which is what
    lets the backend lower the accesses to block-DMA.  One combined H2D transfer
    (rather than a separate one per pointer array) keeps the host path as lean
    as the shared executor's.
    """
    t = tl.program_id(0)
    chunk = tl.program_id(1)
    in_ptr = tl.load(meta_ptr + t).to(tl.pointer_type(DT))
    out_ptr = tl.load(meta_ptr + NT + t).to(tl.pointer_type(DT))
    idx = chunk * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(in_ptr + idx).to(tl.float32)
    y = tl.sin(x) / tl.cos(x)
    tl.store(out_ptr + idx, y.to(DT))


def _pick_block(n_elements, dtype):
    """``(block_size, num_warps)``, mirroring the sibling's tuned sweep.

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


# Under this size the device time is small enough that the host work (16
# ``empty_like`` allocations plus the metadata H2D transfer) dominates, so the
# masked shared executor, whose one-transfer host path is leaner, wins; the
# sweep measured ``n < 16384`` at ~1.6-2.1x slower through the fast path.  The
# unmasked kernel pays off from 16384 elements up (1.25x at 16384, ~10x at
# 262144, ~25x at 16.7M), so the gate keeps every shape at or above the generic
# baseline.  The accuracy-test shape (128 elements) also falls back, which
# exercises the shared executor there; the unmasked kernel is covered by the
# benchmark shapes and the solution probes.
_FAST_MIN_ELEMENTS = 16_384


def _launch_fast(tensors, dtype, device):
    n_elements = tensors[0].numel()
    block, num_warps = _pick_block(n_elements, dtype)
    if n_elements % block != 0:
        return None
    outs = [torch.empty_like(t) for t in tensors]
    # One flat Python list -> one tensor -> one H2D transfer, same shaped
    # staging as the shared executor (only input + output pointers; the element
    # count is a compile-time power of two on this path).
    meta = torch.tensor(
        [t.data_ptr() for t in tensors] + [o.data_ptr() for o in outs],
        dtype=torch.int64,
    ).to(device, non_blocking=True)
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
    return outs


def _try_fast(tensors):
    if not tensors:
        return None
    first = tensors[0]
    dtype = first.dtype
    if dtype not in _FAST_DTYPES:
        return None
    device = first.device
    n_elements = first.numel()
    if n_elements < _FAST_MIN_ELEMENTS:
        return None
    for t in tensors:
        if t.dtype != dtype or t.device != device:
            return None
        if t.numel() != n_elements:
            return None
        if not is_flat_addressable(t):
            return None
    return _launch_fast(tensors, dtype, device)


def _foreach_tan(self):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_TAN")
    tensors = check_tensor_list(self)
    outs = _try_fast(tensors)
    if outs is not None:
        return outs
    return foreach_unary(
        tensors,
        _tan_fast,
        complex_fn=c_tan,
        out_dtype_fn=int_to_float,
        allowed_dtypes=_NO_COMPLEX32,
    )


__all__ = ["_foreach_tan"]
