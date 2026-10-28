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

"""XPU override for ``aten::_foreach_add.List`` (out-of-place, two TensorLists).

Direct-pointer fast path: the tensor pointers are passed *directly* as scalar
kernel arguments and selected with a ``tl.where`` chain over
``tl.program_id(0)`` (see ``_foreach_direct.py``).  A runtime pointer loaded
from a metadata buffer cannot be block-DMA-vectorized, so the meta-buffer form
runs at roughly a quarter of native bandwidth; direct scalar pointers keep the
access compile-time analysable and the memory-bound add reaches full bandwidth.

Non-fast cases fall back to the generic executor verbatim.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_binary import _foreach_add_List as _generic_add_list
from flag_gems.utils.foreach import check_tensor_list, tl_dtype

from ._foreach_direct import _pick_block, _select_ptr

logger = logging.getLogger(__name__)

_FAST_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_MAX_DIRECT = 16


@triton.jit
def _foreach_add_list_kernel(
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
    b0,
    b1,
    b2,
    b3,
    b4,
    b5,
    b6,
    b7,
    b8,
    b9,
    b10,
    b11,
    b12,
    b13,
    b14,
    b15,
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
    t,
    ALPHA: tl.constexpr,
    DT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    a = _select_ptr(
        t, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15
    )
    b = _select_ptr(
        t, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15
    )
    o = _select_ptr(
        t, o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15
    )
    a = a.to(tl.pointer_type(DT))
    b = b.to(tl.pointer_type(DT))
    o = o.to(tl.pointer_type(DT))
    chunk = tl.program_id(1)
    idx = chunk * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(a + idx)
    e = tl.load(b + idx)
    if ALPHA != 1:
        e = (e * ALPHA).to(DT)
    tl.store(o + idx, (x + e).to(DT))


def _try_fast(tensors, others, alpha):
    if not tensors or len(tensors) > _MAX_DIRECT:
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
        if not a.is_contiguous() or not b.is_contiguous():
            return None
    block, num_warps = _pick_block(n_elements)
    if n_elements % block != 0:
        return None
    nt = len(tensors)
    out = [torch.empty_like(a) for a in tensors]
    a_ptrs = [t.data_ptr() for t in tensors] + [tensors[0].data_ptr()] * (
        _MAX_DIRECT - nt
    )
    b_ptrs = [t.data_ptr() for t in others] + [others[0].data_ptr()] * (
        _MAX_DIRECT - nt
    )
    o_ptrs = [t.data_ptr() for t in out] + [out[0].data_ptr()] * (_MAX_DIRECT - nt)
    grid = (nt, n_elements // block)
    _foreach_add_list_kernel[grid](
        *a_ptrs,
        *b_ptrs,
        *o_ptrs,
        nt,
        alpha,
        tl_dtype(dtype),
        block,
        num_warps=num_warps,
        unroll_num=16,
        buffer_size_limit=8192,
        isCloseMemoryAsync=False,
    )
    return out


def _foreach_add_List(self, other, *, alpha=1):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_ADD_LIST")
    tensors = check_tensor_list(self)
    others = check_tensor_list(other)
    if len(tensors) == len(others):
        out = _try_fast(tensors, others, alpha)
        if out is not None:
            return out
    return _generic_add_list(self, other, alpha=alpha)


__all__ = ["_foreach_add_List"]
