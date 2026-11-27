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

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_binary import (
    _foreach_addcdiv__ScalarList as default_foreach_addcdiv__ScalarList,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Moore Threads hardware does not support fp64.
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}

_BLOCK = 1024  # small-list fused kernel tile
_BLOCK_L = 4096  # large-list per-entry kernel tile
_WARPS_L = 8
_MAXS = 16  # list entries fused into one launch
_SMALL_MAX = 1 << 16  # per-entry numel up to which one fused launch is used

# Lazily compiled small-list kernels, one per dtype (no value specialization).
_CK = {}

_NONSPEC = []
for _i in range(_MAXS):
    _NONSPEC += ["s%d" % _i, "a%d" % _i, "b%d" % _i, "al%d" % _i, "n%d" % _i]


@triton.jit
def _tile(s_ptr, a_ptr, b_ptr, alpha, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    s = tl.load(s_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    tl.store(s_ptr + offs, (s + alpha * (a / b)).to(s_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _entry_kernel(s_ptr, a_ptr, b_ptr, alpha, n, BLOCK: tl.constexpr):
    _tile(s_ptr, a_ptr, b_ptr, alpha, n, BLOCK)


@triton.jit(do_not_specialize=_NONSPEC)
def _fused16(
    s0,
    a0,
    b0,
    al0,
    n0,
    s1,
    a1,
    b1,
    al1,
    n1,
    s2,
    a2,
    b2,
    al2,
    n2,
    s3,
    a3,
    b3,
    al3,
    n3,
    s4,
    a4,
    b4,
    al4,
    n4,
    s5,
    a5,
    b5,
    al5,
    n5,
    s6,
    a6,
    b6,
    al6,
    n6,
    s7,
    a7,
    b7,
    al7,
    n7,
    s8,
    a8,
    b8,
    al8,
    n8,
    s9,
    a9,
    b9,
    al9,
    n9,
    s10,
    a10,
    b10,
    al10,
    n10,
    s11,
    a11,
    b11,
    al11,
    n11,
    s12,
    a12,
    b12,
    al12,
    n12,
    s13,
    a13,
    b13,
    al13,
    n13,
    s14,
    a14,
    b14,
    al14,
    n14,
    s15,
    a15,
    b15,
    al15,
    n15,
    BLOCK: tl.constexpr,
):
    slot = tl.program_id(1)
    if slot == 0:
        _tile(s0, a0, b0, al0, n0, BLOCK)
    elif slot == 1:
        _tile(s1, a1, b1, al1, n1, BLOCK)
    elif slot == 2:
        _tile(s2, a2, b2, al2, n2, BLOCK)
    elif slot == 3:
        _tile(s3, a3, b3, al3, n3, BLOCK)
    elif slot == 4:
        _tile(s4, a4, b4, al4, n4, BLOCK)
    elif slot == 5:
        _tile(s5, a5, b5, al5, n5, BLOCK)
    elif slot == 6:
        _tile(s6, a6, b6, al6, n6, BLOCK)
    elif slot == 7:
        _tile(s7, a7, b7, al7, n7, BLOCK)
    elif slot == 8:
        _tile(s8, a8, b8, al8, n8, BLOCK)
    elif slot == 9:
        _tile(s9, a9, b9, al9, n9, BLOCK)
    elif slot == 10:
        _tile(s10, a10, b10, al10, n10, BLOCK)
    elif slot == 11:
        _tile(s11, a11, b11, al11, n11, BLOCK)
    elif slot == 12:
        _tile(s12, a12, b12, al12, n12, BLOCK)
    elif slot == 13:
        _tile(s13, a13, b13, al13, n13, BLOCK)
    elif slot == 14:
        _tile(s14, a14, b14, al14, n14, BLOCK)
    else:
        _tile(s15, a15, b15, al15, n15, BLOCK)


def _use_triton_kernel(x):
    """Whether ``x`` can be handled by the MUSA kernels."""
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if x.is_complex():
        return False
    return True


def _list_supported(seq):
    """A foreach call is specialized only if every tensor in it qualifies."""
    if seq is None:
        return True
    if isinstance(seq, torch.Tensor):
        return _use_triton_kernel(seq)
    for t in seq:
        if isinstance(t, torch.Tensor) and not _use_triton_kernel(t):
            return False
    return True


def _foreach_addcdiv__ScalarList(self, tensor1, tensor2, scalars):
    logger.debug("GEMS_MTHREADS _FOREACH_ADDCDIV__SCALARLIST")

    if not _list_supported(self):
        return default_foreach_addcdiv__ScalarList(self, tensor1, tensor2, scalars)

    with torch_device_fn.device(
        self[0].device
        if isinstance(self, (list, tuple)) and self
        else torch.device("musa")
    ):
        return _launch_foreach_addcdiv__ScalarList(self, tensor1, tensor2, scalars)


def _launch_foreach_addcdiv__ScalarList(self, tensor1, tensor2, scalars):
    if torch.is_tensor(self):
        self = [self]
        tensor1 = [tensor1]
        tensor2 = [tensor2]

    n = len(self)
    if n == 0:
        return None

    if not isinstance(scalars, (list, tuple)):
        scalars = [scalars] * n
    elif len(scalars) != n:
        raise ValueError("scalars must have the same length as the tensor lists")

    # One pass over the list: fill the fused launch argument block directly for a
    # short, homogeneous, small list; abandon the fast path as soon as it is not
    # eligible (the block is then rebuilt by the per-entry loop below).
    ok = n <= _MAXS
    n_act = 0
    args = []
    first = None
    dt = None
    max_n = 0
    for si, ai, bi, al in zip(self, tensor1, tensor2, scalars):
        numel = si.numel()
        if numel == 0:
            continue
        n_act += 1
        if not ok:
            continue
        if numel > _SMALL_MAX:
            ok = False
            continue
        d = si.dtype
        if ai.dtype is not d or bi.dtype is not d or (dt is not None and d is not dt):
            ok = False
            continue
        if first is None:
            dt = d
            first = si
        if numel > max_n:
            max_n = numel
        if not isinstance(al, float):
            al = float(al)
        args.extend((si, ai, bi, al, numel))

    if n_act == 0:
        return None

    if ok:
        while len(args) < _MAXS * 5:
            args.extend((first, first, first, 0.0, 0))
        ck = _CK.get(dt)
        if ck is None:
            ck = _fused16.warmup(*args, _BLOCK, grid=(1, 1, 1))
            _CK[dt] = ck
        ck[(triton.cdiv(max_n, _BLOCK), n_act, 1)](*args, _BLOCK)
        return None

    for si, ai, bi, al in zip(self, tensor1, tensor2, scalars):
        numel = si.numel()
        if numel == 0:
            continue
        if not isinstance(al, float):
            al = float(al)
        _entry_kernel[(triton.cdiv(numel, _BLOCK_L),)](
            si, ai, bi, al, numel, _BLOCK_L, num_warps=_WARPS_L
        )
    return None
