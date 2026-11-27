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
    _foreach_pow__Scalar as default__foreach_pow__Scalar,
)
from flag_gems.utils import tl_extra_shim

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_sqrt = tl_extra_shim.sqrt
_pow = tl_extra_shim.pow

_BLOCK = 1024
_SLOTS = 16
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# Math modes: ATen `pow_tensor_scalar` fast paths, evaluated in fp32 opmath.
_M_GENERIC = 0
_M_SQRT = 1
_M_SQUARE = 2
_M_CUBE = 3
_M_RECIP = 4
_M_RSQRT = 5
_M_RECIP_SQ = 6
_M_ONE = 7
_M_IDENTITY = 8


@triton.jit
def _apply(xf, exponent, MODE: tl.constexpr):
    if MODE == 1:
        y = _sqrt(xf)
    elif MODE == 2:
        y = xf * xf
    elif MODE == 3:
        y = xf * xf * xf
    elif MODE == 4:
        y = 1.0 / xf
    elif MODE == 5:
        y = 1.0 / _sqrt(xf)
    elif MODE == 6:
        y = 1.0 / (xf * xf)
    elif MODE == 7:
        y = xf * 0.0 + 1.0
    elif MODE == 8:
        y = xf
    else:
        y = _pow(xf, exponent)
    return y


@triton.jit
def _pow_fused16(
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
    n0,
    n1,
    n2,
    n3,
    n4,
    n5,
    n6,
    n7,
    n8,
    n9,
    n10,
    n11,
    n12,
    n13,
    n14,
    n15,
    nb0,
    nb1,
    nb2,
    nb3,
    nb4,
    nb5,
    nb6,
    nb7,
    nb8,
    nb9,
    nb10,
    nb11,
    nb12,
    nb13,
    nb14,
    exponent,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    c1 = nb0
    c2 = c1 + nb1
    c3 = c2 + nb2
    c4 = c3 + nb3
    c5 = c4 + nb4
    c6 = c5 + nb5
    c7 = c6 + nb6
    c8 = c7 + nb7
    c9 = c8 + nb8
    c10 = c9 + nb9
    c11 = c10 + nb10
    c12 = c11 + nb11
    c13 = c12 + nb12
    c14 = c13 + nb13
    c15 = c14 + nb14
    zero = pid - pid
    if pid < c1:
        ptr = p0
        base = zero
        n = n0
    elif pid < c2:
        ptr = p1
        base = c1
        n = n1
    elif pid < c3:
        ptr = p2
        base = c2
        n = n2
    elif pid < c4:
        ptr = p3
        base = c3
        n = n3
    elif pid < c5:
        ptr = p4
        base = c4
        n = n4
    elif pid < c6:
        ptr = p5
        base = c5
        n = n5
    elif pid < c7:
        ptr = p6
        base = c6
        n = n6
    elif pid < c8:
        ptr = p7
        base = c7
        n = n7
    elif pid < c9:
        ptr = p8
        base = c8
        n = n8
    elif pid < c10:
        ptr = p9
        base = c9
        n = n9
    elif pid < c11:
        ptr = p10
        base = c10
        n = n10
    elif pid < c12:
        ptr = p11
        base = c11
        n = n11
    elif pid < c13:
        ptr = p12
        base = c12
        n = n12
    elif pid < c14:
        ptr = p13
        base = c13
        n = n13
    elif pid < c15:
        ptr = p14
        base = c14
        n = n14
    else:
        ptr = p15
        base = c15
        n = n15
    offs = (pid - base) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(ptr + offs, mask=mask, other=0.0)
    y = _apply(x.to(tl.float32), exponent, MODE)
    tl.store(ptr + offs, y.to(x.dtype), mask=mask)


@triton.jit
def _pow_fused16_eq(
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
    n_elements,
    nb,
    exponent,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    m1 = nb
    m2 = m1 + nb
    m3 = m2 + nb
    m4 = m3 + nb
    m5 = m4 + nb
    m6 = m5 + nb
    m7 = m6 + nb
    m8 = m7 + nb
    m9 = m8 + nb
    m10 = m9 + nb
    m11 = m10 + nb
    m12 = m11 + nb
    m13 = m12 + nb
    m14 = m13 + nb
    m15 = m14 + nb
    zero = pid - pid
    if pid < m1:
        ptr = p0
        base = zero
    elif pid < m2:
        ptr = p1
        base = m1
    elif pid < m3:
        ptr = p2
        base = m2
    elif pid < m4:
        ptr = p3
        base = m3
    elif pid < m5:
        ptr = p4
        base = m4
    elif pid < m6:
        ptr = p5
        base = m5
    elif pid < m7:
        ptr = p6
        base = m6
    elif pid < m8:
        ptr = p7
        base = m7
    elif pid < m9:
        ptr = p8
        base = m8
    elif pid < m10:
        ptr = p9
        base = m9
    elif pid < m11:
        ptr = p10
        base = m10
    elif pid < m12:
        ptr = p11
        base = m11
    elif pid < m13:
        ptr = p12
        base = m12
    elif pid < m14:
        ptr = p13
        base = m13
    elif pid < m15:
        ptr = p14
        base = m14
    else:
        ptr = p15
        base = m15
    offs = (pid - base) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(ptr + offs, mask=mask, other=0.0)
    y = _apply(x.to(tl.float32), exponent, MODE)
    tl.store(ptr + offs, y.to(x.dtype), mask=mask)


@triton.jit
def _pow_kernel(x_ptr, n_elements, exponent, MODE: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = _apply(x.to(tl.float32), exponent, MODE)
    tl.store(x_ptr + offs, y.to(x.dtype), mask=mask)


@triton.jit
def _pow_strided(
    x_ptr,
    n_elements,
    exponent,
    SIZES: tl.constexpr,
    STRIDES: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    idx = offs.to(tl.int64)
    off = tl.zeros([BLOCK], dtype=tl.int64)
    for i in tl.static_range(len(SIZES)):
        off += (idx % SIZES[i]) * STRIDES[i]
        idx = idx // SIZES[i]
    x = tl.load(x_ptr + off, mask=mask, other=0.0)
    y = _apply(x.to(tl.float32), exponent, MODE)
    tl.store(x_ptr + off, y.to(x.dtype), mask=mask)


_KCACHE = {}


def _launch_eq(chunk, n, nb, e, mode):
    """Uniform-size fast path: one fused launch, reusing a cached compiled
    kernel to skip the per-call JIT dispatch work when the launch
    specialization (pointer alignment, no duplicated pointers) is unchanged."""
    k = len(chunk)
    total = nb * k
    ptrs = list(chunk) + [chunk[0]] * (_SLOTS - k)
    tag = (chunk[0].dtype, n, nb, mode, k)
    runner = _KCACHE.get(tag)
    if runner is None:
        dps = [t.data_ptr() for t in chunk]
        if len(set(dps)) == k and not any(p & 15 for p in dps):
            compiled = _pow_fused16_eq.warmup(
                *ptrs, n, nb, e, MODE=mode, BLOCK=_BLOCK, grid=(1, 1, 1)
            )
            runner = compiled[(total, 1, 1)]
            _KCACHE[tag] = runner
    if runner is not None:
        runner(*ptrs, n, nb, e, mode, _BLOCK)
    else:
        _pow_fused16_eq[(total,)](*ptrs, n, nb, e, MODE=mode, BLOCK=_BLOCK)


def _launch_contig(chunk, e, mode):
    k = len(chunk)
    if k == 1:
        t = chunk[0]
        n = t.numel()
        if n == 0:
            return
        _pow_kernel[(triton.cdiv(n, _BLOCK),)](t, n, e, MODE=mode, BLOCK=_BLOCK)
        return
    nums = [t.numel() for t in chunk]
    blocks = [(n + _BLOCK - 1) // _BLOCK for n in nums]
    if nums.count(nums[0]) == k:
        if nums[0] == 0:
            return
        _launch_eq(chunk, nums[0], blocks[0], e, mode)
        return
    total = sum(blocks)
    if total == 0:
        return
    fill = chunk[0]
    ptrs = list(chunk) + [fill] * (_SLOTS - k)
    ns = nums + [0] * (_SLOTS - k)
    nbs = blocks[:15] + [0] * (15 - (k if k < 15 else 15))
    _pow_fused16[(total,)](*ptrs, *ns, *nbs, e, MODE=mode, BLOCK=_BLOCK)


_MODES = {
    0.5: _M_SQRT,
    2.0: _M_SQUARE,
    3.0: _M_CUBE,
    -1.0: _M_RECIP,
    -0.5: _M_RSQRT,
    -2.0: _M_RECIP_SQ,
    0.0: _M_ONE,
    1.0: _M_IDENTITY,
}


def _foreach_pow__Scalar(self, exponent):
    logger.debug("GEMS_MTHREADS _FOREACH_POW__SCALAR")
    if not isinstance(self, (list, tuple)):
        return default__foreach_pow__Scalar(self, exponent)
    # The scalar overload takes a Python number; a device tensor or any other
    # type goes back to the generic implementation.  Moore Threads hardware has
    # no fp64/int64 path, so only fp16/bf16/fp32 on MUSA are specialized.
    if isinstance(exponent, bool) or not isinstance(exponent, (int, float)):
        return default__foreach_pow__Scalar(self, exponent)
    for t in self:
        if (
            not isinstance(t, torch.Tensor)
            or t.device.type != "musa"
            or t.dtype not in _SUPPORTED_DTYPES
        ):
            return default__foreach_pow__Scalar(self, exponent)
    if not self:
        return None
    e = float(exponent)
    mode = _MODES.get(e, _M_GENERIC)
    groups = {}
    for t in self:
        key = (t.dtype, t.is_contiguous())
        g = groups.get(key)
        if g is None:
            groups[key] = [t]
        else:
            g.append(t)
    for (dtype, contig), ts in groups.items():
        if contig:
            m = len(ts)
            for s in range(0, m, _SLOTS):
                _launch_contig(ts[s : s + _SLOTS], e, mode)
        else:
            for t in ts:
                n = t.numel()
                if n == 0:
                    continue
                sizes = tuple(int(d) for d in t.shape)
                strides = tuple(int(s) for s in t.stride())
                _pow_strided[(triton.cdiv(n, _BLOCK),)](
                    t, n, e, SIZES=sizes, STRIDES=strides, MODE=mode, BLOCK=_BLOCK
                )
    return None
