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

"""XPU overrides for ``aten::_foreach_log`` / ``aten::_foreach_log_``.

The shared ``foreach_unary`` executor masks every load and store with a runtime
``idx < n_elements`` guard, which prevents wide contiguous accesses even when
the mask is always true.  A *homogeneous* list -- one device, one float dtype
(fp16/bf16/fp32), every tensor dense/flat-addressable, all with the same
element count and a count that divides the chosen tile -- is therefore handled
without any mask:

* the in-place form, and the out-of-place form below ``_LARGE`` elements, use a
  2D-grid kernel (axis 0 selects the tensor, axis 1 the tile within it) whose
  addresses come from a small metadata buffer -- one launch per call;
* the out-of-place form at or above ``_LARGE`` elements issues one launch per
  tensor with the pointers as kernel arguments, which is faster at that size
  (the per-call overhead is amortised against a multi-millisecond kernel).

Two details of the implementation:

1. Every launch uses the same vectorization setting as the single-tensor
   ``log``/``log1p`` overrides (``isCloseVectorization=True``).
2. The math runs in fp32 and the result is cast back to the tensor dtype, so
   fp16/bf16 inputs keep their dtype while the logarithm is evaluated at fp32
   precision.

Everything else -- integral inputs (which promote to fp32), complex, fp64,
mixed dtypes or shapes, gappy/overlapping views, non-divisible sizes -- falls
back to the generic ``foreach_unary`` executor, so the dtype policy, the
in-place promotion rejection and ATen's error messages are unchanged.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_unary import UNARY_OPS
from flag_gems.utils.foreach import (
    check_tensor_list,
    foreach_unary,
    has_internal_overlap,
    is_flat_addressable,
    tl_dtype,
)

logger = logging.getLogger(__name__)

_FAST_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# Above this element count the launch-per-tensor form is the faster one: the
# per-call launch overhead is amortised, while the wider tiles of a single
# metadata launch no longer pay off for every dtype.
_LARGE = 1 << 18
_LARGE_BLOCK = 65536
_LARGE_WARPS = 32


@triton.jit
def _foreach_log_unmasked_kernel(
    meta_ptr,
    NT: tl.constexpr,
    DT: tl.constexpr,
    BLOCK: tl.constexpr,
    INPLACE: tl.constexpr,
):
    # meta layout (int64): [0, NT) input pointers, [NT, 2*NT) output pointers.
    t = tl.program_id(0)
    chunk = tl.program_id(1)
    in_ptr = tl.load(meta_ptr + t).to(tl.pointer_type(DT))
    if INPLACE:
        out_ptr = in_ptr
    else:
        out_ptr = tl.load(meta_ptr + NT + t).to(tl.pointer_type(DT))
    idx = chunk * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(in_ptr + idx)
    y = tl.log(x.to(tl.float32))
    tl.store(out_ptr + idx, y.to(DT))


@triton.jit
def _foreach_log_direct_kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
    # One tensor per launch: the pointers travel as kernel arguments instead of
    # being read from a metadata buffer.
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + idx)
    y = tl.log(x.to(tl.float32))
    tl.store(y_ptr + idx, y.to(y_ptr.dtype.element_ty))


def _pick_block(n_elements):
    """``(block_size, num_warps)`` for one tensor element count."""
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


def _launch_direct(srcs, dsts):
    n_elements = srcs[0].numel()
    for src, dst in zip(srcs, dsts):
        _foreach_log_direct_kernel[(n_elements // _LARGE_BLOCK,)](
            src,
            dst,
            _LARGE_BLOCK,
            num_warps=_LARGE_WARPS,
            isCloseVectorization=True,
            unroll_num=16,
            buffer_size_limit=8192,
            isCloseMemoryAsync=False,
        )


def _launch_fast(srcs, dsts, dtype, device, block, num_warps):
    n_elements = srcs[0].numel()
    inplace = dsts is srcs
    ptrs = [t.data_ptr() for t in srcs]
    if not inplace:
        ptrs += [t.data_ptr() for t in dsts]
    meta = torch.tensor(ptrs, dtype=torch.int64).to(device, non_blocking=True)
    grid = (len(srcs), n_elements // block)
    _foreach_log_unmasked_kernel[grid](
        meta,
        len(srcs),
        tl_dtype(dtype),
        block,
        inplace,
        num_warps=num_warps,
        isCloseVectorization=True,
        unroll_num=16,
        buffer_size_limit=8192,
        isCloseMemoryAsync=False,
    )


def _fast_tiling(tensors, inplace):
    """``(block, num_warps)`` when the unmasked launch can serve the whole list."""
    if not tensors:
        return None
    first = tensors[0]
    dtype = first.dtype
    if dtype not in _FAST_DTYPES:
        return None
    device = first.device
    n_elements = first.numel()
    if n_elements < 128:
        return None
    block, num_warps = _pick_block(n_elements)
    if n_elements % block != 0:
        return None
    for t in tensors:
        if t.dtype != dtype or t.device != device:
            return None
        if t.numel() != n_elements:
            return None
        if not is_flat_addressable(t):
            return None
        if inplace and has_internal_overlap(t):
            return None
    return block, num_warps


def _foreach_log(self):
    logger.debug("GEMS _FOREACH_LOG")
    tensors = check_tensor_list(self)
    tiling = _fast_tiling(tensors, inplace=False)
    if tiling is not None:
        device = tensors[0].device
        outs = [torch.empty_like(t) for t in tensors]
        n_elements = tensors[0].numel()
        if n_elements >= _LARGE and n_elements % _LARGE_BLOCK == 0:
            _launch_direct(tensors, outs)
        else:
            _launch_fast(tensors, outs, tensors[0].dtype, device, *tiling)
        return outs
    op = UNARY_OPS["log"]
    return foreach_unary(
        tensors,
        op.fn,
        complex_fn=op.complex_fn,
        out_dtype_fn=op.out_dtype_fn,
        allowed_dtypes=op.allowed,
    )


def _foreach_log_(self):
    logger.debug("GEMS _FOREACH_LOG_")
    tensors = check_tensor_list(self)
    tiling = _fast_tiling(tensors, inplace=True)
    if tiling is not None:
        _launch_fast(tensors, tensors, tensors[0].dtype, tensors[0].device, *tiling)
        return None
    op = UNARY_OPS["log"]
    foreach_unary(
        tensors,
        op.fn,
        complex_fn=op.complex_fn,
        out_dtype_fn=op.out_dtype_fn,
        inplace=True,
        allowed_dtypes=op.allowed,
    )
    return None
