# Copyright 2026 FlagOS Contributors
# Copyright (c) 2024 FlagGems. All rights reserved.
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

"""Fused homogeneous floating-point pow.List with ATen result dtypes.

The generic Gems wrapper handles mixed types, strides, and list validation.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_binary import registered_wrappers
from flag_gems.utils import libentry, tl_extra_shim
from flag_gems.utils.foreach import check_tensor_list

"Native zero-copy result views; numerical work remains in Triton."
_hygon_foreach_views__NATIVE_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)
_hygon_foreach_views__UNBIND = torch.ops.aten.unbind.int


def _hygon_foreach_views__unbind_views(tensor):
    return _hygon_foreach_views__UNBIND.redispatch(
        _hygon_foreach_views__NATIVE_KEYSET, tensor, 0
    )


unbind_views = _hygon_foreach_views__unbind_views
logger = logging.getLogger(__name__)
_libdevice = tl_extra_shim
_generic_pow = registered_wrappers()["_foreach_pow.List"]
_MAX_SLOTS = 16
_BLOCK = 1024
_NUM_WARPS = 4


@libentry()
@triton.jit(do_not_specialize=["N"])
def _foreach_pow_flat_kernel(
    S0,
    E0,
    S1,
    E1,
    S2,
    E2,
    S3,
    E3,
    S4,
    E4,
    S5,
    E5,
    S6,
    E6,
    S7,
    E7,
    S8,
    E8,
    S9,
    E9,
    S10,
    E10,
    S11,
    E11,
    S12,
    E12,
    S13,
    E13,
    S14,
    E14,
    S15,
    E15,
    O,
    N,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    """Uniform-shape variant: one output base pointer, one numel."""
    k = tl.program_id(1)
    s = S0
    e = E0
    if k == 1:
        s = S1
        e = E1
    elif k == 2:
        s = S2
        e = E2
    elif k == 3:
        s = S3
        e = E3
    elif k == 4:
        s = S4
        e = E4
    elif k == 5:
        s = S5
        e = E5
    elif k == 6:
        s = S6
        e = E6
    elif k == 7:
        s = S7
        e = E7
    elif k == 8:
        s = S8
        e = E8
    elif k == 9:
        s = S9
        e = E9
    elif k == 10:
        s = S10
        e = E10
    elif k == 11:
        s = S11
        e = E11
    elif k == 12:
        s = S12
        e = E12
    elif k == 13:
        s = S13
        e = E13
    elif k == 14:
        s = S14
        e = E14
    elif k == 15:
        s = S15
        e = E15
    off = tl.program_id(1).to(tl.int64) * N
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        vidx = tl.max_contiguous(tl.multiple_of(idx, BLOCK), BLOCK)
        a = tl.load(s + vidx).to(tl.float32)
        b = tl.load(e + vidx).to(tl.float32)
        if tl.min(a, axis=0) > 0.0:
            y = tl.exp2(b * tl.log2(a))
        else:
            y = _libdevice.pow(a, b)
        tl.store(O + off + vidx, y.to(O.dtype.element_ty))
    else:
        mask = idx < N
        a = tl.load(s + idx, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(e + idx, mask=mask, other=1.0).to(tl.float32)
        if tl.min(a, axis=0) > 0.0:
            y = tl.exp2(b * tl.log2(a))
        else:
            y = _libdevice.pow(a, b)
        tl.store(O + off + idx, y.to(O.dtype.element_ty), mask=mask)


def _run_general(self, exponent):
    return _generic_pow(self, exponent)


def _foreach_pow_List(self, exponent):
    logger.debug("GEMS_HYGON _FOREACH_POW_LIST")
    "Out-of-place elementwise power over two equal-length tensor lists."
    self = check_tensor_list(self)
    n_pairs = len(self)
    if not isinstance(exponent, (list, tuple)) or len(exponent) != n_pairs:
        return _run_general(self, exponent)
    first = self[0]
    dt = first.dtype
    if dt not in (torch.float16, torch.bfloat16, torch.float32) or n_pairs > _MAX_SLOTS:
        return _run_general(self, exponent)
    shape = first.shape
    for a, b in zip(self, exponent):
        if (
            a.dtype != dt
            or b.dtype != dt
            or a.device != first.device
            or (b.device != first.device)
            or (a.shape != shape)
            or (b.shape != shape)
            or (not a.is_contiguous())
            or (not b.is_contiguous())
        ):
            return _run_general(self, exponent)
    n = first.numel()
    buf = first.new_empty((n_pairs,) + shape, dtype=dt)
    outs = list(unbind_views(buf))
    tiles = (n + _BLOCK - 1) // _BLOCK
    if tiles:
        even = n % _BLOCK == 0
        pad = _MAX_SLOTS - n_pairs
        bases = list(self) + [first] * pad
        exponents = list(exponent) + [exponent[0]] * pad
        pairs = [value for pair in zip(bases, exponents) for value in pair]
        _foreach_pow_flat_kernel[tiles, n_pairs](*pairs, buf, n, _BLOCK, even)
    return outs
