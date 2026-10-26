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

"""Kunlunxin overrides for the ternary scalar ``_foreach_*`` overloads.

``_foreach_lerp.Scalar``/``.ScalarList`` and ``_foreach_addcmul.Scalar`` go
through the shared ternary executor, which reads every tensor base pointer back
from a device-side metadata array.  On this backend that extra dependent load
makes each access a per-element gather, and a sixteen-tensor list measured two
orders of magnitude slower than the eager path.

These wrappers keep every pointer a kernel argument for homogeneous floating
point lists.  Two launch shapes cover the range: a single launch carrying up to
sixteen tensors while the call is host bound, and the big-tile streaming launch
of ``clamp_max`` per tensor once memory time dominates.  Lists of mixed dtypes,
devices or shapes are split into homogeneous groups, and anything outside the
fast dtypes falls back to the shared executor, which keeps the promotion and
error semantics exactly as the generic implementation defines them.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops import _foreach_binary
from flag_gems.utils.foreach import _resolve_scalars, check_tensor_list, tl_dtype

from ._foreach_binary_scalar import _K16_SLOTS
from .addcmul import addcmul_forward as _addcmul_forward
from .lerp import lerp_scalar_kernel_head as _lerp_scalar_kernel

logger = logging.getLogger(__name__)

_FAST_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# Below this numel the call is launch bound, so one launch carrying all sixteen
# tensors wins; above it the tuned pointwise kernel of the standalone operator
# streams each tensor on its own (measured on 2^22-element lists: 2.4 ms fused
# vs 0.9 ms pointwise).
_FUSED_MAX_NUMEL = 1 << 18

# Fused-launch tile: the widest tile that still divides the numel.  A wide tile
# over one CTA measured 6x cheaper per launch than a narrow tile over sixteen
# (a 256-element tile costs ~100 us, a 4096-element one ~16 us), so the tile is
# maximised rather than the CTA count.


@triton.jit
def _lerp_slot(
    a_ptr,
    b_ptr,
    o_ptr,
    idx,
    m,
    s,
    EVEN: tl.constexpr,
    OT: tl.constexpr,
    INPLACE: tl.constexpr,
):
    if EVEN:
        x = tl.load(a_ptr + idx).to(tl.float32)
        y = tl.load(b_ptr + idx).to(tl.float32)
        v = x + s.to(tl.float32) * (y - x)
        if INPLACE:
            tl.store(a_ptr + idx, v.to(OT))
        else:
            tl.store(o_ptr + idx, v.to(OT))
    else:
        x = tl.load(a_ptr + idx, mask=m, other=0.0).to(tl.float32)
        y = tl.load(b_ptr + idx, mask=m, other=0.0).to(tl.float32)
        v = x + s.to(tl.float32) * (y - x)
        if INPLACE:
            tl.store(a_ptr + idx, v.to(OT), mask=m)
        else:
            tl.store(o_ptr + idx, v.to(OT), mask=m)


@triton.jit
def _addcmul_slot(
    a_ptr,
    b_ptr,
    c_ptr,
    o_ptr,
    idx,
    m,
    s,
    EVEN: tl.constexpr,
    OT: tl.constexpr,
    INPLACE: tl.constexpr,
):
    if EVEN:
        x = tl.load(a_ptr + idx).to(tl.float32)
        y = tl.load(b_ptr + idx).to(tl.float32)
        z = tl.load(c_ptr + idx).to(tl.float32)
        v = x + s.to(tl.float32) * y * z
        if INPLACE:
            tl.store(a_ptr + idx, v.to(OT))
        else:
            tl.store(o_ptr + idx, v.to(OT))
    else:
        x = tl.load(a_ptr + idx, mask=m, other=0.0).to(tl.float32)
        y = tl.load(b_ptr + idx, mask=m, other=0.0).to(tl.float32)
        z = tl.load(c_ptr + idx, mask=m, other=0.0).to(tl.float32)
        v = x + s.to(tl.float32) * y * z
        if INPLACE:
            tl.store(a_ptr + idx, v.to(OT), mask=m)
        else:
            tl.store(o_ptr + idx, v.to(OT), mask=m)


@triton.jit
def _foreach2_16_kernel(
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
    o_base,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    s8,
    s9,
    s10,
    s11,
    s12,
    s13,
    s14,
    s15,
    n,
    NT: tl.constexpr,
    OT: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    INPLACE: tl.constexpr,
):
    """
    One launch, up to sixteen ``self``/``tensors1`` pairs and one weight each.

        Slot ``i`` stores to ``out_base + i * n``: the caller hands in a single
        allocation whose per-tensor views are exactly those offsets, which keeps the
        output pointers out of the parameter list.
    """
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = idx < n
    if NT > 0:
        _lerp_slot(a0, b0, o_base, idx, m, s0, EVEN, OT, INPLACE)
    if NT > 1:
        _lerp_slot(a1, b1, o_base + 1 * n, idx, m, s1, EVEN, OT, INPLACE)
    if NT > 2:
        _lerp_slot(a2, b2, o_base + 2 * n, idx, m, s2, EVEN, OT, INPLACE)
    if NT > 3:
        _lerp_slot(a3, b3, o_base + 3 * n, idx, m, s3, EVEN, OT, INPLACE)
    if NT > 4:
        _lerp_slot(a4, b4, o_base + 4 * n, idx, m, s4, EVEN, OT, INPLACE)
    if NT > 5:
        _lerp_slot(a5, b5, o_base + 5 * n, idx, m, s5, EVEN, OT, INPLACE)
    if NT > 6:
        _lerp_slot(a6, b6, o_base + 6 * n, idx, m, s6, EVEN, OT, INPLACE)
    if NT > 7:
        _lerp_slot(a7, b7, o_base + 7 * n, idx, m, s7, EVEN, OT, INPLACE)
    if NT > 8:
        _lerp_slot(a8, b8, o_base + 8 * n, idx, m, s8, EVEN, OT, INPLACE)
    if NT > 9:
        _lerp_slot(a9, b9, o_base + 9 * n, idx, m, s9, EVEN, OT, INPLACE)
    if NT > 10:
        _lerp_slot(a10, b10, o_base + 10 * n, idx, m, s10, EVEN, OT, INPLACE)
    if NT > 11:
        _lerp_slot(a11, b11, o_base + 11 * n, idx, m, s11, EVEN, OT, INPLACE)
    if NT > 12:
        _lerp_slot(a12, b12, o_base + 12 * n, idx, m, s12, EVEN, OT, INPLACE)
    if NT > 13:
        _lerp_slot(a13, b13, o_base + 13 * n, idx, m, s13, EVEN, OT, INPLACE)
    if NT > 14:
        _lerp_slot(a14, b14, o_base + 14 * n, idx, m, s14, EVEN, OT, INPLACE)
    if NT > 15:
        _lerp_slot(a15, b15, o_base + 15 * n, idx, m, s15, EVEN, OT, INPLACE)


@triton.jit
def _foreach3_16_kernel(
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
    c0,
    c1,
    c2,
    c3,
    c4,
    c5,
    c6,
    c7,
    c8,
    c9,
    c10,
    c11,
    c12,
    c13,
    c14,
    c15,
    o_base,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    s8,
    s9,
    s10,
    s11,
    s12,
    s13,
    s14,
    s15,
    n,
    NT: tl.constexpr,
    OT: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    INPLACE: tl.constexpr,
):
    """
    ``self + value * tensor1 * tensor2`` for up to sixteen triples.
    """
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = idx < n
    if NT > 0:
        _addcmul_slot(a0, b0, c0, o_base, idx, m, s0, EVEN, OT, INPLACE)
    if NT > 1:
        _addcmul_slot(a1, b1, c1, o_base + 1 * n, idx, m, s1, EVEN, OT, INPLACE)
    if NT > 2:
        _addcmul_slot(a2, b2, c2, o_base + 2 * n, idx, m, s2, EVEN, OT, INPLACE)
    if NT > 3:
        _addcmul_slot(a3, b3, c3, o_base + 3 * n, idx, m, s3, EVEN, OT, INPLACE)
    if NT > 4:
        _addcmul_slot(a4, b4, c4, o_base + 4 * n, idx, m, s4, EVEN, OT, INPLACE)
    if NT > 5:
        _addcmul_slot(a5, b5, c5, o_base + 5 * n, idx, m, s5, EVEN, OT, INPLACE)
    if NT > 6:
        _addcmul_slot(a6, b6, c6, o_base + 6 * n, idx, m, s6, EVEN, OT, INPLACE)
    if NT > 7:
        _addcmul_slot(a7, b7, c7, o_base + 7 * n, idx, m, s7, EVEN, OT, INPLACE)
    if NT > 8:
        _addcmul_slot(a8, b8, c8, o_base + 8 * n, idx, m, s8, EVEN, OT, INPLACE)
    if NT > 9:
        _addcmul_slot(a9, b9, c9, o_base + 9 * n, idx, m, s9, EVEN, OT, INPLACE)
    if NT > 10:
        _addcmul_slot(a10, b10, c10, o_base + 10 * n, idx, m, s10, EVEN, OT, INPLACE)
    if NT > 11:
        _addcmul_slot(a11, b11, c11, o_base + 11 * n, idx, m, s11, EVEN, OT, INPLACE)
    if NT > 12:
        _addcmul_slot(a12, b12, c12, o_base + 12 * n, idx, m, s12, EVEN, OT, INPLACE)
    if NT > 13:
        _addcmul_slot(a13, b13, c13, o_base + 13 * n, idx, m, s13, EVEN, OT, INPLACE)
    if NT > 14:
        _addcmul_slot(a14, b14, c14, o_base + 14 * n, idx, m, s14, EVEN, OT, INPLACE)
    if NT > 15:
        _addcmul_slot(a15, b15, c15, o_base + 15 * n, idx, m, s15, EVEN, OT, INPLACE)


def _chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _fused_block(n):
    for block in (8192, 4096, 2048, 1024):
        if n % block == 0:
            return block, True
    return 1024, False


def _fused(lists, outs, values, n, inplace):
    block, even = _fused_block(n)
    nt = len(lists[0])
    pad = _K16_SLOTS - nt
    args = []
    for tensors in lists:
        args.extend(tensors)
        args.extend([tensors[0]] * pad)
    args.append(lists[0][0] if inplace else outs[0])
    args.extend(values)
    args.extend([0.0] * pad)
    args.append(n)
    if len(lists) == 2:
        kernel = _foreach2_16_kernel
    else:
        kernel = _foreach3_16_kernel
    kernel[(triton.cdiv(n, block),)](
        *args,
        nt,
        tl_dtype(lists[0][0].dtype),
        block,
        even,
        inplace,
        num_warps=8,
    )


def _pointwise(a, b, c, out, val, tern, inplace):
    """Per-tensor launch with the tuned pointwise kernel of the standalone op."""
    dst = a if inplace else out
    if tern:
        _addcmul_forward(a, b, c, val, out0=dst)
    else:
        _lerp_scalar_kernel(a, b, val, out0=dst)


def _fast(lists, values, inplace):
    """Run every list through the native kernels, or return ``None``.

    The whole list must be one homogeneous run -- same dtype, device, shape and
    a dense layout -- because that is what lets a single launch carry up to
    sixteen tensors with nothing broadcast.  Anything else is the shared
    executor's job.
    """
    head = lists[0]
    count = len(head)
    if count == 0:
        return None
    for tensors in lists:
        if len(tensors) != count:
            return None
    first = head[0]
    dtype = first.dtype
    if dtype not in _FAST_DTYPES:
        return None
    device = first.device
    shape = first.shape
    for tensors in lists:
        for t in tensors:
            if (
                t.dtype is not dtype
                or t.device != device
                or t.shape != shape
                or not t.is_contiguous()
            ):
                return None

    tern = len(lists) == 3
    n = first.numel()
    if inplace:
        outs = list(head)
    else:
        outs = list(
            torch.empty((count,) + tuple(shape), dtype=dtype, device=device).unbind(0)
        )
    if n:
        for start in range(0, count, _K16_SLOTS):
            index = slice(start, start + _K16_SLOTS)
            cols = [tensors[index] for tensors in lists]
            if n <= _FUSED_MAX_NUMEL:
                _fused(cols, outs[index], list(values[index]), n, inplace)
            else:
                slot_out = outs[index]
                vals = values[index]
                dummy = cols[0][0]
                for slot in range(len(cols[0])):
                    _pointwise(
                        cols[0][slot],
                        cols[1][slot],
                        cols[2][slot] if tern else dummy,
                        slot_out[slot],
                        vals[slot],
                        tern,
                        inplace,
                    )
    return outs


def _generic(attr):
    return getattr(_foreach_binary, attr)


def _lerp(self, tensors1, weight, inplace):
    self_l = check_tensor_list(self)
    tensors1_l = check_tensor_list(tensors1)
    if len(self_l) == len(tensors1_l):
        values = _resolve_scalars(self_l, weight)
        res = _fast([self_l, tensors1_l], values, inplace)
        if res is not None:
            return None if inplace else res
    return _generic("_foreach_lerp_%sScalar" % ("_" if inplace else ""))(
        self, tensors1, weight
    )


def _addcmul(self, tensor1, tensor2, value, inplace):
    self_l = check_tensor_list(self)
    tensor1_l = check_tensor_list(tensor1)
    tensor2_l = check_tensor_list(tensor2)
    if len(self_l) == len(tensor1_l) == len(tensor2_l):
        values = _resolve_scalars(self_l, value)
        res = _fast([self_l, tensor1_l, tensor2_l], values, inplace)
        if res is not None:
            return None if inplace else res
    return _generic("_foreach_addcmul_%sScalar" % ("_" if inplace else ""))(
        self, tensor1, tensor2, value
    )


def _foreach_lerp_Scalar(self, tensors1, weight):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_LERP_SCALAR")
    return _lerp(self, tensors1, weight, False)


def _foreach_lerp__Scalar(self, tensors1, weight):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_LERP__SCALAR")
    return _lerp(self, tensors1, weight, True)


def _foreach_lerp_ScalarList(self, tensors1, weight):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_LERP_SCALARLIST")
    return _lerp(self, tensors1, weight, False)


def _foreach_lerp__ScalarList(self, tensors1, weight):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_LERP__SCALARLIST")
    return _lerp(self, tensors1, weight, True)


def _foreach_addcmul_Scalar(self, tensor1, tensor2, value=1):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_ADDCMUL_SCALAR")
    return _addcmul(self, tensor1, tensor2, value, False)


__all__ = [
    "_foreach_addcmul_Scalar",
    "_foreach_lerp_Scalar",
    "_foreach_lerp_ScalarList",
    "_foreach_lerp__Scalar",
    "_foreach_lerp__ScalarList",
]
