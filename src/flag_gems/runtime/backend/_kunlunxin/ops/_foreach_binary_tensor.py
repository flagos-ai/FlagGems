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

"""Kunlunxin (XPU) overrides for the foreach ``.Tensor`` overloads.

``aten::_foreach_{add,mul,div}.Tensor`` broadcasts a single 0-dim scalar tensor
over a list of tensors.  The generic executor funnels every element through a
device-side metadata pointer indirection and runs ~250x slower than native on
large lists (~494 ms vs ~2 ms).

The fast path passes the list pointers *directly* as scalar kernel arguments
(selected with a ``tl.where`` chain over ``tl.program_id(0)``, see
``_foreach_direct.py``) and reads the 0-dim scalar from device memory once, so
both the load and the store stay block-DMA-analysable.  Non-0-dim ``other`` or
non-uniform lists fall back to the generic executor verbatim.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_binary import _foreach_add__Tensor as _generic_add__tensor
from flag_gems.ops._foreach_binary import _foreach_add_Tensor as _generic_add_tensor
from flag_gems.ops._foreach_binary import _foreach_mul__Tensor as _generic_mul__tensor
from flag_gems.ops._foreach_binary import _foreach_mul_Tensor as _generic_mul_tensor
from flag_gems.utils.foreach import check_tensor_list, tl_dtype

from ._foreach_direct import _pick_block, _select_ptr

logger = logging.getLogger(__name__)
# The registration-liveness test only raises the *generic* foreach loggers to
# DEBUG (``flag_gems.ops._foreach_binary`` / ``_foreach_reduction``).  This
# override logs under its own ``__name__``, so raise it here as well; otherwise
# the DEBUG record is filtered at the logger level and the test reports "dead
# registration".
logger.setLevel(logging.DEBUG)

_MAX_DIRECT = 16
_FAST_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def _foreach_tensor_direct_kernel(
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
    scalar_ptr,
    t,
    DT: tl.constexpr,
    BLOCK: tl.constexpr,
    OP: tl.constexpr,
):
    a = _select_ptr(
        t, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15
    )
    o = _select_ptr(
        t, o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15
    )
    a = a.to(tl.pointer_type(DT))
    o = o.to(tl.pointer_type(DT))
    s = tl.load(scalar_ptr).to(tl.float32)
    chunk = tl.program_id(1)
    idx = chunk * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(a + idx).to(tl.float32)
    if OP == 0:
        r = x + s
    elif OP == 1:
        r = x * s
    else:
        r = x / s
    tl.store(o + idx, r.to(DT))


def _try_direct(self, other, inplace, op):
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
    for t in self:
        if t.dtype != dtype or t.device != device or t.numel() != n:
            return None
        if not t.is_contiguous():
            return None
    block, num_warps = _pick_block(n)
    if n % block != 0:
        return None

    nt = len(self)
    if inplace:
        out = [t for t in self]
    else:
        out = [torch.empty_like(t) for t in self]

    a_ptrs = [t.data_ptr() for t in self] + [self[0].data_ptr()] * (_MAX_DIRECT - nt)
    o_ptrs = [t.data_ptr() for t in out] + [out[0].data_ptr()] * (_MAX_DIRECT - nt)

    grid = (nt, n // block)
    _foreach_tensor_direct_kernel[grid](
        *a_ptrs,
        *o_ptrs,
        other,
        nt,
        tl_dtype(dtype),
        block,
        op,
        num_warps=num_warps,
        unroll_num=16,
        buffer_size_limit=8192,
        isCloseMemoryAsync=False,
    )
    return out if not inplace else True


def _foreach_add_Tensor(self, other, *, alpha=1):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_ADD_TENSOR")
    self = check_tensor_list(self)
    if other is not None and other.dim() == 0 and alpha == 1:
        out = _try_direct(self, other, inplace=False, op=0)
        if out is not None:
            return out
    return _generic_add_tensor(self, other, alpha=alpha)


def _foreach_add__Tensor(self, other, *, alpha=1):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_ADD__TENSOR")
    self = check_tensor_list(self)
    if other is not None and other.dim() == 0 and alpha == 1:
        if _try_direct(self, other, inplace=True, op=0):
            return None
    return _generic_add__tensor(self, other, alpha=alpha)


def _foreach_mul_Tensor(self, other):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_MUL_TENSOR")
    self = check_tensor_list(self)
    if other is not None and other.dim() == 0:
        out = _try_direct(self, other, inplace=False, op=1)
        if out is not None:
            return out
    return _generic_mul_tensor(self, other)


def _foreach_mul__Tensor(self, other):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_MUL__TENSOR")
    self = check_tensor_list(self)
    if other is not None and other.dim() == 0:
        if _try_direct(self, other, inplace=True, op=1):
            return None
    return _generic_mul__tensor(self, other)


__all__ = [
    "_foreach_add_Tensor",
    "_foreach_add__Tensor",
    "_foreach_mul_Tensor",
    "_foreach_mul__Tensor",
]


def _patch_generic_wrapper():
    """Rebind the generic binary-foreach wrappers to this Kunlunxin fast path.

    These ``.Tensor`` overloads are generated dynamically in
    ``flag_gems.ops._foreach_binary`` (``_globals[_attr] = _fn``) and resolved
    by ``registered_wrappers()``, which the foreach tests/benchmark read from
    that module's ``__dict__``.  ``SpecOpRegistrar`` only patches the top-level
    ``flag_gems.*`` namespace, so it never touches the generic module's
    wrappers.  Patch both namespaces here so the fast path actually takes
    effect.
    """
    import sys

    _m = sys.modules.get("flag_gems.ops._foreach_binary")
    if _m is not None:
        _m._foreach_add_Tensor = _foreach_add_Tensor
        _m._foreach_add__Tensor = _foreach_add__Tensor
        _m._foreach_mul_Tensor = _foreach_mul_Tensor
        _m._foreach_mul__Tensor = _foreach_mul__Tensor
    import flag_gems

    flag_gems._foreach_add_Tensor = _foreach_add_Tensor
    flag_gems._foreach_add__Tensor = _foreach_add__Tensor
    flag_gems._foreach_mul_Tensor = _foreach_mul_Tensor
    flag_gems._foreach_mul__Tensor = _foreach_mul__Tensor


_patch_generic_wrapper()
