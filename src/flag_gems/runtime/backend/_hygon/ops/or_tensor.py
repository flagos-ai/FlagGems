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

"""Triton implementation of aten::__or__.Tensor (elementwise bitwise OR).

out = self | other, with ATen broadcasting / dtype promotion, no aliasing of
inputs, and a freshly allocated contiguous output.  All element work happens in
the Triton kernels below; the host side only reads metadata (shapes, strides,
dtypes) and launches.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_MAXD = 8  # max broadcast rank handled by the strided kernel
_WARPS = 4
_INT32_MAX = 2147483647
_BIG_N = 1 << 20  # numel from which the autotuned kernel is used


# --------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------
@triton.jit
def _or_flat_exact(a_ptr, b_ptr, o_ptr, BLOCK: tl.constexpr):
    """Same dtype, dense, numel an exact multiple of BLOCK: no tail, no cast."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(a_ptr + offs)
    y = tl.load(b_ptr + offs)
    tl.store(o_ptr + offs, x | y)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 1024}, num_warps=2),
        triton.Config({"BLOCK": 1024}, num_warps=4),
        triton.Config({"BLOCK": 2048}, num_warps=2),
        triton.Config({"BLOCK": 2048}, num_warps=4),
        triton.Config({"BLOCK": 2048}, num_warps=8),
        triton.Config({"BLOCK": 4096}, num_warps=2),
        triton.Config({"BLOCK": 4096}, num_warps=4),
        triton.Config({"BLOCK": 4096}, num_warps=8),
        triton.Config({"BLOCK": 8192}, num_warps=2),
        triton.Config({"BLOCK": 8192}, num_warps=4),
        triton.Config({"BLOCK": 8192}, num_warps=8),
        triton.Config({"BLOCK": 16384}, num_warps=4),
        triton.Config({"BLOCK": 16384}, num_warps=8),
    ],
    key=["n"],
)
@triton.jit
def _or_flat_big(a_ptr, b_ptr, o_ptr, n, BLOCK: tl.constexpr):
    """Same dtype, dense, large numel.  Full blocks run mask-free; the tail
    block takes the masked path (uniform branch, no per-element cost)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if pid * BLOCK + BLOCK <= n:
        x = tl.load(a_ptr + offs)
        y = tl.load(b_ptr + offs)
        tl.store(o_ptr + offs, x | y)
    else:
        m = offs < n
        x = tl.load(a_ptr + offs, mask=m, other=0)
        y = tl.load(b_ptr + offs, mask=m, other=0)
        tl.store(o_ptr + offs, x | y, mask=m)


@triton.jit
def _or_flat(
    a_ptr,
    b_ptr,
    o_ptr,
    n,
    BLOCK: tl.constexpr,
    INT64: tl.constexpr,
    MIXED: tl.constexpr,
):
    """Dense over the (contiguous) output domain, with a tail mask."""
    pid = tl.program_id(0)
    if INT64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(a_ptr + offs, mask=m, other=0)
    y = tl.load(b_ptr + offs, mask=m, other=0)
    if MIXED:
        oty = o_ptr.dtype.element_ty
        x = x.to(oty)
        y = y.to(oty)
    tl.store(o_ptr + offs, x | y, mask=m)


@triton.jit
def _or_bcast(
    a_ptr,
    b_ptr,
    o_ptr,
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
    bval,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
    INT64: tl.constexpr,
    B_SCALAR: tl.constexpr,
    MIXED: tl.constexpr,
):
    """Broadcast/strided form: output is contiguous, operands use 0 strides on
    broadcast axes.  Dims are ordered outer -> inner in s0..s7."""
    pid = tl.program_id(0)
    if INT64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    rem = offs
    ao = offs * 0
    bo = offs * 0
    if RANK > 7:
        i = rem % s7
        rem = rem // s7
        ao += i * a7
        bo += i * b7
    if RANK > 6:
        i = rem % s6
        rem = rem // s6
        ao += i * a6
        bo += i * b6
    if RANK > 5:
        i = rem % s5
        rem = rem // s5
        ao += i * a5
        bo += i * b5
    if RANK > 4:
        i = rem % s4
        rem = rem // s4
        ao += i * a4
        bo += i * b4
    if RANK > 3:
        i = rem % s3
        rem = rem // s3
        ao += i * a3
        bo += i * b3
    if RANK > 2:
        i = rem % s2
        rem = rem // s2
        ao += i * a2
        bo += i * b2
    if RANK > 1:
        i = rem % s1
        rem = rem // s1
        ao += i * a1
        bo += i * b1
    if RANK > 0:
        i = rem % s0
        ao += i * a0
        bo += i * b0

    x = tl.load(a_ptr + ao, mask=m, other=0)
    if B_SCALAR:
        v = (x.to(tl.int64) | bval).to(o_ptr.dtype.element_ty)
    else:
        y = tl.load(b_ptr + bo, mask=m, other=0)
        if MIXED:
            oty = o_ptr.dtype.element_ty
            x = x.to(oty)
            y = y.to(oty)
        v = x | y
    tl.store(o_ptr + offs, v, mask=m)


# --------------------------------------------------------------------------
# host helpers (metadata only)
# --------------------------------------------------------------------------
def _flat_block(itemsize):
    """Elements per block so each thread moves ~16B per operand."""
    if itemsize <= 1:
        return 4096
    if itemsize == 2:
        return 2048
    if itemsize == 4:
        return 1024
    return 512


def _expand(shape, stride, rank):
    """Right-align an operand's strides to the output rank, 0 on broadcast axes."""
    st = [0] * (rank - len(shape))
    for j, sz in enumerate(shape):
        st.append(0 if sz == 1 else stride[j])
    return st


def _plan(oshape, sa, sb):
    """TensorIterator-style dimension merge, outer -> inner.

    Returns a list of (size, a_stride, b_stride) covering the whole output
    domain; adjacent dims merge when all operands are contiguous across them.
    """
    groups = []
    for d in range(len(oshape) - 1, -1, -1):
        s = oshape[d]
        if s == 1:
            continue
        aa = sa[d]
        bb = sb[d]
        if groups:
            gs, ga, gb = groups[-1]
            if aa == ga * gs and bb == gb * gs:
                groups[-1] = (gs * s, ga, gb)
                continue
        groups.append((s, aa, bb))
    groups.reverse()
    return groups


def _check_dtype(dt):
    if dt.is_floating_point or dt.is_complex:
        raise RuntimeError(
            "bitwise_or not implemented for '%s'" % str(dt).replace("torch.", "")
        )


def _launch_flat(a, b, out, n, mixed):
    """Dense launch for the general path (tail mask / mixed dtypes allowed)."""
    blk = _flat_block(min(a.element_size(), b.element_size(), out.element_size()))
    if not mixed and n % blk == 0 and n <= _INT32_MAX:
        _or_flat_exact[(n // blk,)](a, b, out, BLOCK=blk, num_warps=_WARPS)
        return
    _or_flat[((n + blk - 1) // blk,)](
        a,
        b,
        out,
        n,
        BLOCK=blk,
        INT64=n > _INT32_MAX,
        MIXED=mixed,
        num_warps=_WARPS,
    )


def _launch_plan(groups, a, b, out, n, mixed, b_scalar=False, bval=0):
    rank = len(groups)
    if rank == 0:
        # every output dim had size 1 -> a single element, both offsets are 0
        blk = _flat_block(min(a.element_size(), b.element_size(), out.element_size()))
        _or_flat[(1,)](
            a, b, out, 1, BLOCK=blk, INT64=False, MIXED=mixed, num_warps=_WARPS
        )
        return
    if (
        not b_scalar
        and rank == 1
        and groups[0][0] == n
        and groups[0][1] == 1
        and groups[0][2] == 1
    ):
        _launch_flat(a, b, out, n, mixed)
        return
    if rank > _MAXD:
        raise RuntimeError("bitwise_or: broadcast rank %d unsupported" % rank)
    blk = _flat_block(min(a.element_size(), b.element_size(), out.element_size()))
    s = [1] * _MAXD
    as_ = [0] * _MAXD
    bs = [0] * _MAXD
    for i, (sz, aa, bb) in enumerate(groups):
        s[i] = sz
        as_[i] = aa
        bs[i] = bb
    _or_bcast[((n + blk - 1) // blk,)](
        a,
        b,
        out,
        n,
        *s,
        *as_,
        *bs,
        bval,
        RANK=rank,
        BLOCK=blk,
        INT64=n > _INT32_MAX,
        B_SCALAR=b_scalar,
        MIXED=mixed,
        num_warps=_WARPS,
    )


def _or_same(a, b):
    """Hot path: identical dtype, identical shape, both dense."""
    dt = a.dtype
    if dt.is_floating_point or dt.is_complex:
        _check_dtype(dt)
    out = torch.empty_like(a)
    n = out.numel()
    if n == 0:
        return out
    if _BIG_N <= n <= _INT32_MAX:
        _or_flat_big[lambda meta: ((n + meta["BLOCK"] - 1) // meta["BLOCK"],)](
            a, b, out, n
        )
        return out
    blk = _flat_block(a.element_size())
    if n % blk == 0:
        _or_flat_exact[(n // blk,)](a, b, out, BLOCK=blk, num_warps=_WARPS)
    else:
        _or_flat[((n + blk - 1) // blk,)](
            a,
            b,
            out,
            n,
            BLOCK=blk,
            INT64=n > _INT32_MAX,
            MIXED=False,
            num_warps=_WARPS,
        )
    return out


def _or_tensor(a, b):
    dt = torch.result_type(a, b)
    _check_dtype(dt)
    sa = a.shape
    sb = b.shape
    if sa == sb:
        oshape = sa
    else:
        oshape = torch.broadcast_shapes(sa, sb)
    out = torch.empty(oshape, dtype=dt, device=a.device)
    n = out.numel()
    if n == 0:
        return out
    mixed = a.dtype is not dt or b.dtype is not dt
    if sa == oshape and sb == oshape and a.is_contiguous() and b.is_contiguous():
        _launch_flat(a, b, out, n, mixed)
        return out
    rank = len(oshape)
    groups = _plan(oshape, _expand(sa, a.stride(), rank), _expand(sb, b.stride(), rank))
    _launch_plan(groups, a, b, out, n, mixed)
    return out


def _or_scalar(t, val):
    dt = torch.result_type(t, val)
    _check_dtype(dt)
    out = torch.empty(t.shape, dtype=dt, device=t.device)
    n = out.numel()
    if n == 0:
        return out
    rank = t.dim()
    groups = _plan(list(t.shape), _expand(t.shape, t.stride(), rank), [0] * rank)
    mixed = t.dtype is not dt
    _launch_plan(groups, t, t, out, n, mixed, b_scalar=True, bval=int(val))
    return out


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def or_tensor(self, other):
    logger.debug("GEMS_HYGON OR_TENSOR")
    if isinstance(other, torch.Tensor):
        if isinstance(self, torch.Tensor):
            if (
                self.dtype is other.dtype
                and self.shape == other.shape
                and self.is_contiguous()
                and other.is_contiguous()
            ):
                return _or_same(self, other)
            return _or_tensor(self, other)
        return _or_scalar(other, self)
    if isinstance(self, torch.Tensor):
        return _or_scalar(self, other)
    raise TypeError("bitwise_or: at least one operand must be a tensor")
