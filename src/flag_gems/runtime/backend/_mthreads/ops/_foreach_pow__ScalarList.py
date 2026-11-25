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
"""In-place foreach power with per-tensor scalar exponents.

Reproduces ``aten::_foreach_pow_.ScalarList``: for each pair (T_i, e_i) the
tensor T_i is overwritten element-wise with T_i ** e_i, in the exact same
storage (in place), returning ``None``.

When the whole list is contiguous, non-overlapping, of one size and uses one
exponent value (the common foreach case), the call becomes a single 2-D launch
``grid = (cdiv(n, BLOCK), len(list))`` whose second axis selects the tensor, so
the host pays for one launch instead of ``len(list)``.  Lists of uniform dtype
that are not fused use one masked launch per tensor.  Anything the kernel does
not cover (non-contiguous tensors, fp64/int/bool, non-MUSA devices) is handed
back to the generic implementation.

The exponent is a ``tl.constexpr`` so the common exact values compile to cheap
device code: ``e == 0.5`` becomes a single sqrt instead of a full powf, and
``e == 1.0`` is an in-place identity that needs no launch at all.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_binary import (
    _foreach_pow__ScalarList as default__foreach_pow__ScalarList,
)
from flag_gems.utils import tl_extra_shim

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_sqrt = tl_extra_shim.sqrt
_pow = tl_extra_shim.pow

_BLOCK = 1024
_MAX_FUSE = 16
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_ITEMSIZE = {
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.float32: 4,
}


@triton.jit
def _pow_eval(xf, EXP: tl.constexpr):
    if EXP == 0.5:
        return _sqrt(xf)
    if EXP == 2.0:
        return xf * xf
    if EXP == 3.0:
        return xf * xf * xf
    if EXP == -1.0:
        return 1.0 / xf
    return _pow(xf, EXP)


@triton.jit
def _pow_work(p, n, pid, EXP: tl.constexpr, BLOCK: tl.constexpr):
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(p + offs, mask=mask, other=0.0)
    y = _pow_eval(x.to(tl.float32), EXP)
    tl.store(p + offs, y.to(x.dtype), mask=mask)


@triton.jit
def _pow_single_kernel(ptr, n_elem, EXP: tl.constexpr, BLOCK: tl.constexpr):
    _pow_work(ptr, n_elem, tl.program_id(0), EXP, BLOCK)


@triton.jit
def _pow_uniform16_kernel(
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
    n_elem,
    EXP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    t = tl.program_id(1)
    if t == 0:
        _pow_work(p0, n_elem, pid, EXP, BLOCK)
    elif t == 1:
        _pow_work(p1, n_elem, pid, EXP, BLOCK)
    elif t == 2:
        _pow_work(p2, n_elem, pid, EXP, BLOCK)
    elif t == 3:
        _pow_work(p3, n_elem, pid, EXP, BLOCK)
    elif t == 4:
        _pow_work(p4, n_elem, pid, EXP, BLOCK)
    elif t == 5:
        _pow_work(p5, n_elem, pid, EXP, BLOCK)
    elif t == 6:
        _pow_work(p6, n_elem, pid, EXP, BLOCK)
    elif t == 7:
        _pow_work(p7, n_elem, pid, EXP, BLOCK)
    elif t == 8:
        _pow_work(p8, n_elem, pid, EXP, BLOCK)
    elif t == 9:
        _pow_work(p9, n_elem, pid, EXP, BLOCK)
    elif t == 10:
        _pow_work(p10, n_elem, pid, EXP, BLOCK)
    elif t == 11:
        _pow_work(p11, n_elem, pid, EXP, BLOCK)
    elif t == 12:
        _pow_work(p12, n_elem, pid, EXP, BLOCK)
    elif t == 13:
        _pow_work(p13, n_elem, pid, EXP, BLOCK)
    elif t == 14:
        _pow_work(p14, n_elem, pid, EXP, BLOCK)
    else:
        _pow_work(p15, n_elem, pid, EXP, BLOCK)


def _launch_single(t, e, n):
    grid = ((n + _BLOCK - 1) // _BLOCK,)
    _pow_single_kernel[grid](t, n, EXP=e, BLOCK=_BLOCK)


def _can_specialize(tensors, exponents):
    """Whether every (tensor, exponent) pair is handled by this kernel."""
    if not isinstance(tensors, (list, tuple)):
        return False
    if not isinstance(exponents, (list, tuple)):
        return False
    if len(tensors) != len(exponents):
        return False
    for t, e in zip(tensors, exponents):
        if not isinstance(t, torch.Tensor) or t.device.type != "musa":
            return False
        if t.dtype not in _SUPPORTED_DTYPES or not t.is_contiguous():
            return False
        if isinstance(e, torch.Tensor):
            if e.numel() != 1:
                return False
        elif isinstance(e, bool) or not isinstance(e, (int, float)):
            return False
    return True


def _foreach_pow__ScalarList(self, exponent, *args, **kwargs):
    logger.debug("GEMS_MTHREADS _FOREACH_POW__SCALARLIST")
    if not _can_specialize(self, exponent):
        return default__foreach_pow__ScalarList(self, exponent, *args, **kwargs)
    tensors = self
    if type(tensors) is torch.Tensor:
        tensors = [tensors]
    elif type(tensors) is not list and type(tensors) is not tuple:
        tensors = list(tensors)

    n_t = len(tensors)
    if n_t == 0:
        return None

    exps = exponent
    if type(exps) is not list and type(exps) is not tuple:
        if type(exps) is torch.Tensor or not hasattr(exps, "__len__"):
            exps = [exps]
        else:
            exps = list(exps)

    # Single metadata pass: (tensor, exponent, numel, contiguous, data_ptr)
    items = []
    add = items.append
    for i in range(n_t):
        t = tensors[i]
        n = t.numel()
        if n:
            e = exps[i]
            if type(e) is not float:
                if isinstance(e, torch.Tensor):
                    e = float(e.reshape(-1)[0])
                else:
                    e = float(e)
            add((t, e, n, t.is_contiguous(), t.data_ptr()))

    k = len(items)
    if k == 0:
        return None

    n0 = items[0][2]
    e0 = items[0][1]
    fast = k <= _MAX_FUSE
    if fast:
        for it in items:
            if it[1] != e0 or it[2] != n0 or not it[3]:
                fast = False
                break
    if fast and k > 1:
        spans = []
        for it in items:
            t = it[0]
            isz = _ITEMSIZE.get(t.dtype)
            if isz is None:
                isz = t.element_size()
            spans.append((it[4], it[4] + it[2] * isz))
        spans = sorted(spans)
        for i in range(k - 1):
            if spans[i + 1][0] < spans[i][1]:
                fast = False
                break

    if fast:
        if e0 == 1.0:
            # pow(x, 1) is the identity: nothing to write, no launch needed.
            return None
        ptrs = [it[0] for it in items]
        if k < _MAX_FUSE:
            ptrs += [ptrs[0]] * (_MAX_FUSE - k)
        _pow_uniform16_kernel[((n0 + _BLOCK - 1) // _BLOCK, k)](
            *ptrs, n0, EXP=e0, BLOCK=_BLOCK
        )
        return None

    for it in items:
        _launch_single(it[0], it[1], it[2])
    return None
