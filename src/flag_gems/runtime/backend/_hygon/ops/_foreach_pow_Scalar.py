# Copyright (c) 2024 FlagGems. All rights reserved.
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

"""Native zero-copy result views; numerical work remains in Triton."""

import logging
import numbers

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_binary import registered_wrappers
from flag_gems.ops._foreach_binary import (
    registered_wrappers as _hygon_foreach_pow_ScalarList__registered_wrappers,
)
from flag_gems.utils import libentry as _hygon_foreach_pow_ScalarList__libentry
from flag_gems.utils import (
    tl_extra_shim as _hygon_foreach_pow_ScalarList__tl_extra_shim,
)
from flag_gems.utils.foreach import check_tensor_list
from flag_gems.utils.foreach import (
    check_tensor_list as _hygon_foreach_pow_ScalarList__check_tensor_list,
)

"Native zero-copy result views; numerical work remains in Triton."
_hygon_foreach_views__NATIVE_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)
_hygon_foreach_views__UNBIND = torch.ops.aten.unbind.int


def _hygon_foreach_views__unbind_views(tensor):
    return _hygon_foreach_views__UNBIND.redispatch(
        _hygon_foreach_views__NATIVE_KEYSET, tensor, 0
    )


_hygon_foreach_pow_ScalarList__unbind_views = _hygon_foreach_views__unbind_views
_hygon_foreach_pow_ScalarList__libdevice = _hygon_foreach_pow_ScalarList__tl_extra_shim
_hygon_foreach_pow_ScalarList__generic_pow = (
    _hygon_foreach_pow_ScalarList__registered_wrappers()["_foreach_pow.ScalarList"]
)
_hygon_foreach_pow_ScalarList__logger = logging.getLogger(__name__)
_hygon_foreach_pow_ScalarList__BLOCK = 2048
_hygon_foreach_pow_ScalarList__MAX_SLOTS = 16
_hygon_foreach_pow_ScalarList__M_GENERIC = 0
_hygon_foreach_pow_ScalarList__M_SQUARE = 1
_hygon_foreach_pow_ScalarList__M_CUBE = 2
_hygon_foreach_pow_ScalarList__M_SQRT = 3


@_hygon_foreach_pow_ScalarList__libentry()
@triton.jit
def _hygon_foreach_pow_ScalarList__pow_uniform_kernel(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8,
    p9,
    p10,
    p11,
    p12,
    p13,
    p14,
    p15,
    out_ptr,
    n,
    exp,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One launch for a list of same-shape, same-dtype tensors.

    program_id(1) selects the tensor slot, program_id(0) the tile inside it.
    All outputs live in one flat buffer, so the destination is out_ptr +
    slot * n + offs.
    """
    pid = tl.program_id(0)
    slot = tl.program_id(1)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    if slot == 0:
        x = tl.load(p0 + offs, mask=mask, other=0)
    elif slot == 1:
        x = tl.load(p1 + offs, mask=mask, other=0)
    elif slot == 2:
        x = tl.load(p2 + offs, mask=mask, other=0)
    elif slot == 3:
        x = tl.load(p3 + offs, mask=mask, other=0)
    elif slot == 4:
        x = tl.load(p4 + offs, mask=mask, other=0)
    elif slot == 5:
        x = tl.load(p5 + offs, mask=mask, other=0)
    elif slot == 6:
        x = tl.load(p6 + offs, mask=mask, other=0)
    elif slot == 7:
        x = tl.load(p7 + offs, mask=mask, other=0)
    elif slot == 8:
        x = tl.load(p8 + offs, mask=mask, other=0)
    elif slot == 9:
        x = tl.load(p9 + offs, mask=mask, other=0)
    elif slot == 10:
        x = tl.load(p10 + offs, mask=mask, other=0)
    elif slot == 11:
        x = tl.load(p11 + offs, mask=mask, other=0)
    elif slot == 12:
        x = tl.load(p12 + offs, mask=mask, other=0)
    elif slot == 13:
        x = tl.load(p13 + offs, mask=mask, other=0)
    elif slot == 14:
        x = tl.load(p14 + offs, mask=mask, other=0)
    else:
        x = tl.load(p15 + offs, mask=mask, other=0)
    xf = x.to(tl.float64 if out_ptr.dtype.element_ty == tl.float64 else tl.float32)
    if MODE == 3:
        y = tl.sqrt(xf)
    elif MODE == 1:
        y = xf * xf
    elif MODE == 2:
        y = xf * xf * xf
    else:
        y = _hygon_foreach_pow_ScalarList__libdevice.pow(
            xf, tl.full((BLOCK,), exp, xf.dtype)
        )
    dst = out_ptr + slot.to(tl.int64) * n + offs
    tl.store(dst, y.to(out_ptr.dtype.element_ty), mask=mask)


def _hygon_foreach_pow_ScalarList__mode_of(exp):
    if exp == 0.5:
        return _hygon_foreach_pow_ScalarList__M_SQRT
    if exp == 2.0:
        return _hygon_foreach_pow_ScalarList__M_SQUARE
    if exp == 3.0:
        return _hygon_foreach_pow_ScalarList__M_CUBE
    return _hygon_foreach_pow_ScalarList__M_GENERIC


def _hygon_foreach_pow_ScalarList__fallback(tensors, n, exponent):
    return _hygon_foreach_pow_ScalarList__generic_pow(tensors, exponent)


def _hygon_foreach_pow_ScalarList__foreach_pow_ScalarList(self, exponent):
    _hygon_foreach_pow_ScalarList__logger.debug("GEMS_HYGON _FOREACH_POW_SCALARLIST")
    tensors = _hygon_foreach_pow_ScalarList__check_tensor_list(self)
    n = len(tensors)
    t0 = tensors[0]
    dt = t0.dtype
    if (
        n <= _hygon_foreach_pow_ScalarList__MAX_SLOTS
        and dt.is_floating_point
        and t0.is_contiguous()
    ):
        ex = exponent
        tex = type(ex)
        if tex is list or tex is tuple:
            if len(ex) != n or ex.count(ex[0]) != n:
                return _hygon_foreach_pow_ScalarList__fallback(tensors, n, exponent)
            e = float(ex[0])
        elif isinstance(ex, torch.Tensor):
            return _hygon_foreach_pow_ScalarList__fallback(tensors, n, exponent)
        else:
            e = float(ex)
        shp = t0.shape
        for t in tensors:
            if (
                t.shape != shp
                or t.dtype is not dt
                or t.device != t0.device
                or (not t.is_contiguous())
            ):
                return _hygon_foreach_pow_ScalarList__fallback(tensors, n, exponent)
        numel = t0.numel()
        buf = torch.empty((n,) + shp, dtype=dt, device=t0.device)
        if numel:
            if e == 0.5:
                mode = _hygon_foreach_pow_ScalarList__M_SQRT
            elif e == 2.0:
                mode = _hygon_foreach_pow_ScalarList__M_SQUARE
            elif e == 3.0:
                mode = _hygon_foreach_pow_ScalarList__M_CUBE
            else:
                mode = _hygon_foreach_pow_ScalarList__M_GENERIC
            ptrs = (
                tensors
                if n == _hygon_foreach_pow_ScalarList__MAX_SLOTS
                else tensors + [t0] * (_hygon_foreach_pow_ScalarList__MAX_SLOTS - n)
            )
            _hygon_foreach_pow_ScalarList__pow_uniform_kernel[
                (numel + _hygon_foreach_pow_ScalarList__BLOCK - 1)
                // _hygon_foreach_pow_ScalarList__BLOCK,
                n,
            ](*ptrs, buf, numel, e, mode, _hygon_foreach_pow_ScalarList__BLOCK)
        return list(_hygon_foreach_pow_ScalarList__unbind_views(buf))
    return _hygon_foreach_pow_ScalarList__fallback(tensors, n, exponent)


_foreach_pow_ScalarList = _hygon_foreach_pow_ScalarList__foreach_pow_ScalarList
logger = logging.getLogger(__name__)
_generic_pow = registered_wrappers()["_foreach_pow.Scalar"]


def _foreach_pow_Scalar(self, exponent):
    logger.debug("GEMS_HYGON _FOREACH_POW_SCALAR")
    tensors = check_tensor_list(self)
    if not isinstance(exponent, numbers.Number):
        raise TypeError("argument 'exponent' must be Number")
    if tensors[0].dtype.is_floating_point and (not isinstance(exponent, complex)):
        return _foreach_pow_ScalarList(tensors, [exponent] * len(tensors))
    return _generic_pow(tensors, exponent)
