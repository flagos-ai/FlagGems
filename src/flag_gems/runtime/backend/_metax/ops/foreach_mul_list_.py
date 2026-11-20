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

import torch
import triton
import triton.language as tl

_BLOCK = 2048
_SBLOCK = 1024


@triton.jit
def _blk(a_ptr, b_ptr, offs, n, EVEN: tl.constexpr):
    if EVEN:
        x = tl.load(a_ptr + offs)
        y = tl.load(b_ptr + offs)
        tl.store(a_ptr + offs, x * y)
    else:
        m = offs < n
        x = tl.load(a_ptr + offs, mask=m, other=0.0)
        y = tl.load(b_ptr + offs, mask=m, other=0.0)
        tl.store(a_ptr + offs, x * y, mask=m)


@triton.jit
def _k_u4(a0, b0, a1, b1, a2, b2, a3, b3, n, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.program_id(1)
    if slot == 0:
        _blk(a0, b0, offs, n, EVEN)
    elif slot == 1:
        _blk(a1, b1, offs, n, EVEN)
    elif slot == 2:
        _blk(a2, b2, offs, n, EVEN)
    else:
        _blk(a3, b3, offs, n, EVEN)


@triton.jit
def _k_flat(a_ptr, b_ptr, n, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    _blk(a_ptr, b_ptr, offs, n, EVEN)


@triton.jit
def _k_bcast(a_ptr, b_ptr, n, bn, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(a_ptr + offs, mask=mask, other=0.0)
    y = tl.load(b_ptr + (offs % bn), mask=mask, other=0.0)
    tl.store(a_ptr + offs, x * y, mask=mask)


@triton.jit
def _k_scalar(a_ptr, s, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(a_ptr + offs, mask=mask, other=0.0)
    tl.store(a_ptr + offs, x * s, mask=mask)


@triton.jit
def _step(i, sh, ast, bst, aoff, boff):
    q = i % sh
    i = i // sh
    aoff += q * ast
    boff += q * bst
    return i, aoff, boff


@triton.jit
def _k_strided(
    a_ptr,
    b_ptr,
    n,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6,
    a7,
    b0,
    b1,
    b2,
    b3,
    b4,
    b5,
    b6,
    b7,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    i = tl.where(mask, offs, 0).to(tl.int64)
    aoff = tl.zeros((BLOCK,), dtype=tl.int64)
    boff = tl.zeros((BLOCK,), dtype=tl.int64)
    i, aoff, boff = _step(i, s7, a7, b7, aoff, boff)
    i, aoff, boff = _step(i, s6, a6, b6, aoff, boff)
    i, aoff, boff = _step(i, s5, a5, b5, aoff, boff)
    i, aoff, boff = _step(i, s4, a4, b4, aoff, boff)
    i, aoff, boff = _step(i, s3, a3, b3, aoff, boff)
    i, aoff, boff = _step(i, s2, a2, b2, aoff, boff)
    i, aoff, boff = _step(i, s1, a1, b1, aoff, boff)
    i, aoff, boff = _step(i, s0, a0, b0, aoff, boff)
    x = tl.load(a_ptr + aoff, mask=mask, other=0.0)
    y = tl.load(b_ptr + boff, mask=mask, other=0.0)
    tl.store(a_ptr + aoff, x * y, mask=mask)


def _uniform(a_list, b_list, lo, m, n, nb):
    a0 = a_list[lo]
    b0 = b_list[lo]
    even = (n & (_BLOCK - 1)) == 0
    if m == 1:
        _k_flat[(nb,)](a0, b0, n, _BLOCK, even)
        return
    a1 = a_list[lo + 1]
    b1 = b_list[lo + 1]
    if m == 2:
        _k_u4[(nb, 2)](a0, b0, a1, b1, a0, b0, a0, b0, n, _BLOCK, even)
        return
    a2 = a_list[lo + 2]
    b2 = b_list[lo + 2]
    if m == 3:
        _k_u4[(nb, 3)](a0, b0, a1, b1, a2, b2, a0, b0, n, _BLOCK, even)
        return
    _k_u4[(nb, 4)](
        a0, b0, a1, b1, a2, b2, a_list[lo + 3], b_list[lo + 3], n, _BLOCK, even
    )


def _valid_inplace_bcast(a, b):
    ash = a.shape
    bsh = b.shape
    la = len(ash)
    lb = len(bsh)
    if lb > la:
        return False
    for k in range(1, la + 1):
        sa = ash[-k]
        sb = bsh[-k] if k <= lb else 1
        if sa != sb and sa != 1 and sb != 1:
            return False
        if sa == 1 and sb != 1:
            return False
    return True


def _strided_pair(a, b, n):
    if not _valid_inplace_bcast(a, b):
        raise RuntimeError(
            "output with shape {} doesn't match the broadcast shape {}".format(
                list(a.shape), list(b.shape)
            )
        )
    ash = a.shape
    bsh = b.shape
    ast = a.stride()
    bst = b.stride()
    la = len(ash)
    lb = len(bsh)
    r = la if la > lb else lb
    if r > 8:
        raise RuntimeError("foreach_mul_: unsupported tensor rank")
    sh = [1, 1, 1, 1, 1, 1, 1, 1]
    as_ = [0, 0, 0, 0, 0, 0, 0, 0]
    bs = [0, 0, 0, 0, 0, 0, 0, 0]
    for k in range(r):
        ia = la - r + k
        ib = lb - r + k
        sa = ash[ia] if ia >= 0 else 1
        sb = bsh[ib] if ib >= 0 else 1
        p = 8 - r + k
        sh[p] = sa
        if ia >= 0:
            as_[p] = ast[ia]
        if ib >= 0 and sb == sa:
            bs[p] = bst[ib]
    _k_strided[((n + _SBLOCK - 1) // _SBLOCK,)](
        a,
        b,
        n,
        sh[0],
        sh[1],
        sh[2],
        sh[3],
        sh[4],
        sh[5],
        sh[6],
        sh[7],
        as_[0],
        as_[1],
        as_[2],
        as_[3],
        as_[4],
        as_[5],
        as_[6],
        as_[7],
        bs[0],
        bs[1],
        bs[2],
        bs[3],
        bs[4],
        bs[5],
        bs[6],
        bs[7],
        _SBLOCK,
    )


def _pair(a, b):
    n = a.numel()
    if n == 0:
        return
    bn = b.numel()
    if a.is_contiguous() and b.is_contiguous():
        if bn == n:
            if a.shape != b.shape and not _valid_inplace_bcast(a, b):
                raise RuntimeError(
                    "output with shape {} doesn't match the broadcast shape {}".format(
                        list(a.shape), list(b.shape)
                    )
                )
            _k_flat[((n + _BLOCK - 1) // _BLOCK,)](
                a, b, n, _BLOCK, (n & (_BLOCK - 1)) == 0
            )
            return
        if bn == 1:
            _k_bcast[((n + _BLOCK - 1) // _BLOCK,)](a, b, n, bn, _BLOCK)
            return
        if (
            b.dim() <= a.dim()
            and n % bn == 0
            and a.shape[a.dim() - b.dim() :] == b.shape
        ):
            _k_bcast[((n + _BLOCK - 1) // _BLOCK,)](a, b, n, bn, _BLOCK)
            return
    _strided_pair(a, b, n)


def _scalar_pair(a, s):
    n = a.numel()
    if n == 0:
        return
    _k_scalar[((n + _BLOCK - 1) // _BLOCK,)](a, s, n, _BLOCK)


def foreach_mul_list_(self, other):
    if isinstance(self, torch.Tensor):
        a_list = [self]
    else:
        a_list = self
        if len(a_list) == 0:
            raise RuntimeError("Tensor list must have at least one tensor.")
    if isinstance(other, torch.Tensor):
        if other.dim() != 0:
            raise RuntimeError(
                "scalar tensor expected to be 0 dim but has {} dimensions "
                "and {} elements.".format(other.dim(), other.numel())
            )
        bn = other.numel()
        for a in a_list:
            n = a.numel()
            if n:
                _k_bcast[((n + _BLOCK - 1) // _BLOCK,)](a, other, n, bn, _BLOCK)
        return None
    if isinstance(other, (list, tuple)):
        cnt = len(a_list)
        if len(other) != cnt:
            if len(other) and not isinstance(other[0], torch.Tensor):
                raise RuntimeError(
                    "Tensor list must have same number of elements as scalar list."
                )
            raise RuntimeError(
                "Tensor lists must have the same number of tensors, "
                "got {} and {}".format(cnt, len(other))
            )
        if len(other) and not isinstance(other[0], torch.Tensor):
            for k in range(cnt):
                _scalar_pair(a_list[k], float(other[k]))
            return None
        if cnt == 4:
            a0 = a_list[0]
            b0 = other[0]
            a1 = a_list[1]
            b1 = other[1]
            a2 = a_list[2]
            b2 = other[2]
            a3 = a_list[3]
            b3 = other[3]
            n = a0.numel()
            if (
                n
                and a0.is_contiguous()
                and b0.is_contiguous()
                and b0.numel() == n
                and a1.is_contiguous()
                and b1.is_contiguous()
                and a1.numel() == n
                and b1.numel() == n
                and a2.is_contiguous()
                and b2.is_contiguous()
                and a2.numel() == n
                and b2.numel() == n
                and a3.is_contiguous()
                and b3.is_contiguous()
                and a3.numel() == n
                and b3.numel() == n
                and a1.dtype is a0.dtype
                and a2.dtype is a0.dtype
                and a3.dtype is a0.dtype
                and b1.dtype is b0.dtype
                and b2.dtype is b0.dtype
                and b3.dtype is b0.dtype
                and a1.device == a0.device
                and a2.device == a0.device
                and a3.device == a0.device
                and b0.device == a0.device
                and b1.device == a0.device
                and b2.device == a0.device
                and b3.device == a0.device
            ):
                _k_u4[((n + _BLOCK - 1) // _BLOCK, 4)](
                    a0, b0, a1, b1, a2, b2, a3, b3, n, _BLOCK, (n & (_BLOCK - 1)) == 0
                )
                return None
            _pair(a0, b0)
            _pair(a1, b1)
            _pair(a2, b2)
            _pair(a3, b3)
            return None
        a0 = a_list[0]
        b0 = other[0]
        n = a0.numel()
        k = 0
        if n and a0.is_contiguous() and b0.is_contiguous() and b0.numel() == n:
            dt_a = a0.dtype
            dt_b = b0.dtype
            dev = a0.device
            k = 1
            while k < cnt:
                a = a_list[k]
                b = other[k]
                if (
                    a.numel() != n
                    or not a.is_contiguous()
                    or a.dtype is not dt_a
                    or b.numel() != n
                    or not b.is_contiguous()
                    or b.dtype is not dt_b
                    or a.device != dev
                    or b.device != dev
                ):
                    break
                k += 1
            if k == cnt:
                nb = (n + _BLOCK - 1) // _BLOCK
                i = 0
                while i < cnt:
                    m = cnt - i
                    if m > 4:
                        m = 4
                    _uniform(a_list, other, i, m, n, nb)
                    i += m
                return None
        for k in range(cnt):
            _pair(a_list[k], other[k])
        return None
    s = float(other)
    for a in a_list:
        _scalar_pair(a, s)
    return None
