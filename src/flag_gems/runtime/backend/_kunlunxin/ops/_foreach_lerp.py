# Copyright 2026, The FlagOS Contributors.
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

"""Kunlunxin (XPU) override for the foreach ``lerp`` ``.List`` overloads.

``aten::_foreach_lerp_.List(self, other, weight)`` computes
``self[i] + weight[i] * (other[i] - self[i])`` over three TensorLists.  The
generic executor funnels every element through a device-side metadata pointer
indirection and runs ~250x slower than native on large lists.

The fast path passes the three list pointers *directly* as scalar kernel
arguments (selected with a ``tl.where`` chain over ``tl.program_id(0)``) so the
block-DMA analysis sees stride-1 accesses.  Non-uniform lists fall back to the
generic executor verbatim.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_binary import _foreach_lerp__List as _generic_lerp__list
from flag_gems.ops._foreach_binary import _foreach_lerp_List as _generic_lerp_list
from flag_gems.utils.foreach import check_tensor_list, tl_dtype

from ._foreach_direct import _pick_block, _select_ptr

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

_MAX_DIRECT = 16
_FAST_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def _foreach_lerp_list_kernel(
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
    w0,
    w1,
    w2,
    w3,
    w4,
    w5,
    w6,
    w7,
    w8,
    w9,
    w10,
    w11,
    w12,
    w13,
    w14,
    w15,
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
    DT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0)
    a = _select_ptr(
        t, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15
    )
    b = _select_ptr(
        t, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15
    )
    w = _select_ptr(
        t, w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15
    )
    o = _select_ptr(
        t, o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15
    )
    a = a.to(tl.pointer_type(DT))
    b = b.to(tl.pointer_type(DT))
    w = w.to(tl.pointer_type(DT))
    o = o.to(tl.pointer_type(DT))
    chunk = tl.program_id(1)
    idx = chunk * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(a + idx).to(tl.float32)
    y = tl.load(b + idx).to(tl.float32)
    z = tl.load(w + idx).to(tl.float32)
    r = x + z * (y - x)
    tl.store(o + idx, r.to(DT))


def _try_direct(self, other, weight, inplace):
    if not self or len(self) > _MAX_DIRECT:
        return None
    first = self[0]
    dtype = first.dtype
    if dtype not in _FAST_DTYPES:
        return None
    device = first.device
    n = first.numel()
    if n == 0:
        return None
    for a, b, w in zip(self, other, weight):
        if a.dtype != dtype or b.dtype != dtype or w.dtype != dtype:
            return None
        if a.device != device or b.device != device or w.device != device:
            return None
        if a.numel() != n or b.numel() != n or w.numel() != n:
            return None
        if not (a.is_contiguous() and b.is_contiguous() and w.is_contiguous()):
            return None
    block, num_warps = _pick_block(n)
    if n % block != 0:
        return None

    nt = len(self)
    if inplace:
        out = [t for t in self]
    else:
        out = [torch.empty_like(t) for t in self]
    pad = [self[0].data_ptr()] * (_MAX_DIRECT - nt)
    a_ptrs = [t.data_ptr() for t in self] + pad
    b_ptrs = [t.data_ptr() for t in other] + pad
    w_ptrs = [t.data_ptr() for t in weight] + pad
    o_ptrs = [t.data_ptr() for t in out] + pad

    grid = (nt, n // block)
    _foreach_lerp_list_kernel[grid](
        *a_ptrs,
        *b_ptrs,
        *w_ptrs,
        *o_ptrs,
        tl_dtype(dtype),
        block,
        num_warps=num_warps,
        unroll_num=16,
        buffer_size_limit=8192,
        isCloseMemoryAsync=False,
    )
    return out if not inplace else True


def _foreach_lerp_List(self, other, weight):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_LERP_LIST")
    self = check_tensor_list(self)
    other = check_tensor_list(other)
    weight = check_tensor_list(weight)
    if len(self) == len(other) == len(weight):
        out = _try_direct(self, other, weight, inplace=False)
        if out is not None:
            return out
    return _generic_lerp_list(self, other, weight)


def _foreach_lerp__List(self, other, weight):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_LERP__LIST")
    self = check_tensor_list(self)
    other = check_tensor_list(other)
    weight = check_tensor_list(weight)
    if len(self) == len(other) == len(weight):
        if _try_direct(self, other, weight, inplace=True):
            return None
    return _generic_lerp__list(self, other, weight)


__all__ = ["_foreach_lerp_List", "_foreach_lerp__List"]


def _patch_generic_wrapper():
    import sys

    _m = sys.modules.get("flag_gems.ops._foreach_binary")
    if _m is not None:
        _m._foreach_lerp_List = _foreach_lerp_List
        _m._foreach_lerp__List = _foreach_lerp__List
    import flag_gems

    flag_gems._foreach_lerp_List = _foreach_lerp_List
    flag_gems._foreach_lerp__List = _foreach_lerp__List


_patch_generic_wrapper()
