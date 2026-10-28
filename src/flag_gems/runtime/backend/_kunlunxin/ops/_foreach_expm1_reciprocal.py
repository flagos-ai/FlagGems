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

"""Kunlunxin (XPU) overrides for ``foreach_expm1`` / ``foreach_reciprocal``.

Both are unary foreach ops; the generic ``foreach_unary`` executor funnels each
element through a device-side metadata pointer indirection and runs ~250x
slower than native on large lists (~164 ms vs ~2 ms).  The fast path passes the
list pointers *directly* as scalar kernel arguments (selected with a
``tl.where`` chain over ``tl.program_id(0)``) so the access stays
block-DMA-analysable, and computes ``expm1``/``reciprocal`` in fp32.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_unary import NO_COMPLEX32, apply_unary
from flag_gems.utils.foreach import (
    check_dtype_supported,
    check_tensor_list,
    has_internal_overlap,
    tl_dtype,
)

from ._foreach_direct import _pick_block, _select_ptr

logger = logging.getLogger(__name__)

_MAX_DIRECT = 16
_FAST_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def _foreach_unary_direct_kernel(
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
    OUT_DT: tl.constexpr,
    BLOCK: tl.constexpr,
    OP: tl.constexpr,
):
    t = tl.program_id(0)
    a = _select_ptr(
        t, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15
    )
    o = _select_ptr(
        t, o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15
    )
    a = a.to(tl.pointer_type(DT))
    o = o.to(tl.pointer_type(OUT_DT))
    chunk = tl.program_id(1)
    idx = chunk * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(a + idx).to(tl.float32)
    if OP == 0:
        r = tl.exp(x) - 1.0
    else:
        r = 1.0 / x
    tl.store(o + idx, r.to(OUT_DT))


def _try_direct(tensors, op):
    if not tensors or len(tensors) > _MAX_DIRECT:
        return None
    first = tensors[0]
    dtype = first.dtype
    if dtype not in _FAST_DTYPES:
        return None
    device = first.device
    n = first.numel()
    if n == 0:
        return None
    for t in tensors:
        if t.dtype != dtype or t.device != device or t.numel() != n:
            return None
        if not t.is_contiguous():
            return None
    block, num_warps = _pick_block(n)
    if n % block != 0:
        return None

    nt = len(tensors)
    out = [torch.empty_like(t) for t in tensors]
    pad = [first.data_ptr()] * (_MAX_DIRECT - nt)
    a_ptrs = [t.data_ptr() for t in tensors] + pad
    o_ptrs = [t.data_ptr() for t in out] + pad

    grid = (nt, n // block)
    _foreach_unary_direct_kernel[grid](
        *a_ptrs,
        *o_ptrs,
        tl_dtype(dtype),
        tl_dtype(dtype),
        block,
        op,
        num_warps=num_warps,
        unroll_num=16,
        buffer_size_limit=8192,
        isCloseMemoryAsync=False,
    )
    return out


def _foreach_expm1(self):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_EXPM1")
    tensors = check_tensor_list(self)
    if any(t.dtype.is_complex for t in tensors):
        return apply_unary("expm1", self, inplace=False)
    for t in tensors:
        check_dtype_supported(t.dtype, NO_COMPLEX32)
    out = _try_direct(tensors, op=0)
    if out is not None:
        return out
    return apply_unary("expm1", self, inplace=False)


def _foreach_expm1_(self):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_EXPM1_")
    tensors = check_tensor_list(self)
    if any(t.dtype.is_complex for t in tensors):
        return apply_unary("expm1", self, inplace=True)
    for t in tensors:
        check_dtype_supported(t.dtype, NO_COMPLEX32)
        if has_internal_overlap(t):
            raise RuntimeError("internal overlap not supported")
    if all(t.is_contiguous() for t in tensors) and all(
        t.dtype in _FAST_DTYPES for t in tensors
    ):
        srcs = tensors
        if _launch_inplace(srcs, op=0):
            return None
    return apply_unary("expm1", self, inplace=True)


def _foreach_reciprocal(self):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_RECIPROCAL")
    tensors = check_tensor_list(self)
    if any(t.dtype.is_complex for t in tensors):
        return apply_unary("reciprocal", self, inplace=False)
    for t in tensors:
        check_dtype_supported(t.dtype, NO_COMPLEX32)
    out = _try_direct(tensors, op=1)
    if out is not None:
        return out
    return apply_unary("reciprocal", self, inplace=False)


def _foreach_reciprocal_(self):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_RECIPROCAL_")
    tensors = check_tensor_list(self)
    if any(t.dtype.is_complex for t in tensors):
        return apply_unary("reciprocal", self, inplace=True)
    for t in tensors:
        check_dtype_supported(t.dtype, NO_COMPLEX32)
        if has_internal_overlap(t):
            raise RuntimeError("internal overlap not supported")
    if all(t.is_contiguous() for t in tensors) and all(
        t.dtype in _FAST_DTYPES for t in tensors
    ):
        if _launch_inplace(tensors, op=1):
            return None
    return apply_unary("reciprocal", self, inplace=True)


def _launch_inplace(tensors, op):
    if not tensors or len(tensors) > _MAX_DIRECT:
        return False
    first = tensors[0]
    n = first.numel()
    if n == 0:
        return False
    if any(t.numel() != n for t in tensors):
        return False
    block, num_warps = _pick_block(n)
    if n % block != 0:
        return False
    nt = len(tensors)
    pad = [first.data_ptr()] * (_MAX_DIRECT - nt)
    ptrs = [t.data_ptr() for t in tensors] + pad
    grid = (nt, n // block)
    _foreach_unary_direct_kernel[grid](
        *ptrs,
        *ptrs,
        tl_dtype(first.dtype),
        tl_dtype(first.dtype),
        block,
        op,
        num_warps=num_warps,
        unroll_num=16,
        buffer_size_limit=8192,
        isCloseMemoryAsync=False,
    )
    return True


__all__ = [
    "_foreach_expm1",
    "_foreach_expm1_",
    "_foreach_reciprocal",
    "_foreach_reciprocal_",
]


def _patch_generic_wrapper():
    import sys

    _m = sys.modules.get("flag_gems.ops._foreach_unary")
    if _m is not None:
        _m._foreach_expm1 = _foreach_expm1
        _m._foreach_expm1_ = _foreach_expm1_
        _m._foreach_reciprocal = _foreach_reciprocal
        _m._foreach_reciprocal_ = _foreach_reciprocal_
    import flag_gems

    flag_gems._foreach_expm1 = _foreach_expm1
    flag_gems._foreach_expm1_ = _foreach_expm1_
    flag_gems._foreach_reciprocal = _foreach_reciprocal
    flag_gems._foreach_reciprocal_ = _foreach_reciprocal_


_patch_generic_wrapper()
