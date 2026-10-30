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

"""Kunlunxin overrides for the out-of-place ``_foreach_*`` scalar overloads.

A uniform floating-point list keeps every tensor pointer a kernel argument and
picks the launch shape from the list size: one launch carrying the whole list
for small numels, a streaming tile per tensor once memory time dominates.
Anything else -- other dtypes, mixed numels/devices, non-contiguous inputs,
tensor-valued scalars -- is handed back to the shared executor, so promotion,
bool and layout semantics stay exactly as the generic implementation defines
them.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils.foreach import check_tensor_list, tl_dtype

from ._foreach_binary_scalar import _K16_SLOTS, _foreach16_kernel, _pick_block

logger = logging.getLogger(__name__)


# bf16 is computed in fp32 and converted back.  Keep the dtype branches
# exclusive: a branch ``return`` does not stop the rest of the body from being
# traced, so a trailing return would also count in the return-type check.
@triton.jit
def _fadd(a, s):
    if a.dtype == tl.bfloat16:
        return (a.to(tl.float32) + s.to(tl.float32)).to(tl.bfloat16)
    elif a.dtype == tl.int1:
        return a | s
    else:
        return a + s


@triton.jit
def _fsub(a, s):
    if a.dtype == tl.bfloat16:
        return (a.to(tl.float32) - s.to(tl.float32)).to(tl.bfloat16)
    else:
        return a - s


@triton.jit
def _fmul(a, s):
    if a.dtype == tl.bfloat16:
        return (a.to(tl.float32) * s.to(tl.float32)).to(tl.bfloat16)
    elif a.dtype == tl.int1:
        return a & s
    else:
        return a * s


@triton.jit
def _fdiv(a, s):
    if a.dtype == tl.bfloat16:
        return (a.to(tl.float32) / s.to(tl.float32)).to(tl.bfloat16)
    else:
        return a / s


@triton.jit
def _foreach16_flat_kernel(
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
    o_ptr,
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
    nt: tl.constexpr,
    fn: tl.constexpr,
    OT: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    """``_foreach16_kernel`` with the sixteen outputs addressed off one base.

    Slot ``i`` of the result sits at ``i * n`` in the single allocation, so the
    outputs cost one kernel argument instead of sixteen and the per-call launch
    time drops by a third at the small shapes, where it is the whole cost.
    """
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        if nt > 0:
            tl.store(
                o_ptr + 0 * n + idx, fn(tl.load(a0 + idx).to(OT), s0.to(OT)).to(OT)
            )
        if nt > 1:
            tl.store(
                o_ptr + 1 * n + idx, fn(tl.load(a1 + idx).to(OT), s1.to(OT)).to(OT)
            )
        if nt > 2:
            tl.store(
                o_ptr + 2 * n + idx, fn(tl.load(a2 + idx).to(OT), s2.to(OT)).to(OT)
            )
        if nt > 3:
            tl.store(
                o_ptr + 3 * n + idx, fn(tl.load(a3 + idx).to(OT), s3.to(OT)).to(OT)
            )
        if nt > 4:
            tl.store(
                o_ptr + 4 * n + idx, fn(tl.load(a4 + idx).to(OT), s4.to(OT)).to(OT)
            )
        if nt > 5:
            tl.store(
                o_ptr + 5 * n + idx, fn(tl.load(a5 + idx).to(OT), s5.to(OT)).to(OT)
            )
        if nt > 6:
            tl.store(
                o_ptr + 6 * n + idx, fn(tl.load(a6 + idx).to(OT), s6.to(OT)).to(OT)
            )
        if nt > 7:
            tl.store(
                o_ptr + 7 * n + idx, fn(tl.load(a7 + idx).to(OT), s7.to(OT)).to(OT)
            )
        if nt > 8:
            tl.store(
                o_ptr + 8 * n + idx, fn(tl.load(a8 + idx).to(OT), s8.to(OT)).to(OT)
            )
        if nt > 9:
            tl.store(
                o_ptr + 9 * n + idx, fn(tl.load(a9 + idx).to(OT), s9.to(OT)).to(OT)
            )
        if nt > 10:
            tl.store(
                o_ptr + 10 * n + idx, fn(tl.load(a10 + idx).to(OT), s10.to(OT)).to(OT)
            )
        if nt > 11:
            tl.store(
                o_ptr + 11 * n + idx, fn(tl.load(a11 + idx).to(OT), s11.to(OT)).to(OT)
            )
        if nt > 12:
            tl.store(
                o_ptr + 12 * n + idx, fn(tl.load(a12 + idx).to(OT), s12.to(OT)).to(OT)
            )
        if nt > 13:
            tl.store(
                o_ptr + 13 * n + idx, fn(tl.load(a13 + idx).to(OT), s13.to(OT)).to(OT)
            )
        if nt > 14:
            tl.store(
                o_ptr + 14 * n + idx, fn(tl.load(a14 + idx).to(OT), s14.to(OT)).to(OT)
            )
        if nt > 15:
            tl.store(
                o_ptr + 15 * n + idx, fn(tl.load(a15 + idx).to(OT), s15.to(OT)).to(OT)
            )
    else:
        m = idx < n
        if nt > 0:
            tl.store(
                o_ptr + 0 * n + idx,
                fn(tl.load(a0 + idx, mask=m).to(OT), s0.to(OT)).to(OT),
                mask=m,
            )
        if nt > 1:
            tl.store(
                o_ptr + 1 * n + idx,
                fn(tl.load(a1 + idx, mask=m).to(OT), s1.to(OT)).to(OT),
                mask=m,
            )
        if nt > 2:
            tl.store(
                o_ptr + 2 * n + idx,
                fn(tl.load(a2 + idx, mask=m).to(OT), s2.to(OT)).to(OT),
                mask=m,
            )
        if nt > 3:
            tl.store(
                o_ptr + 3 * n + idx,
                fn(tl.load(a3 + idx, mask=m).to(OT), s3.to(OT)).to(OT),
                mask=m,
            )
        if nt > 4:
            tl.store(
                o_ptr + 4 * n + idx,
                fn(tl.load(a4 + idx, mask=m).to(OT), s4.to(OT)).to(OT),
                mask=m,
            )
        if nt > 5:
            tl.store(
                o_ptr + 5 * n + idx,
                fn(tl.load(a5 + idx, mask=m).to(OT), s5.to(OT)).to(OT),
                mask=m,
            )
        if nt > 6:
            tl.store(
                o_ptr + 6 * n + idx,
                fn(tl.load(a6 + idx, mask=m).to(OT), s6.to(OT)).to(OT),
                mask=m,
            )
        if nt > 7:
            tl.store(
                o_ptr + 7 * n + idx,
                fn(tl.load(a7 + idx, mask=m).to(OT), s7.to(OT)).to(OT),
                mask=m,
            )
        if nt > 8:
            tl.store(
                o_ptr + 8 * n + idx,
                fn(tl.load(a8 + idx, mask=m).to(OT), s8.to(OT)).to(OT),
                mask=m,
            )
        if nt > 9:
            tl.store(
                o_ptr + 9 * n + idx,
                fn(tl.load(a9 + idx, mask=m).to(OT), s9.to(OT)).to(OT),
                mask=m,
            )
        if nt > 10:
            tl.store(
                o_ptr + 10 * n + idx,
                fn(tl.load(a10 + idx, mask=m).to(OT), s10.to(OT)).to(OT),
                mask=m,
            )
        if nt > 11:
            tl.store(
                o_ptr + 11 * n + idx,
                fn(tl.load(a11 + idx, mask=m).to(OT), s11.to(OT)).to(OT),
                mask=m,
            )
        if nt > 12:
            tl.store(
                o_ptr + 12 * n + idx,
                fn(tl.load(a12 + idx, mask=m).to(OT), s12.to(OT)).to(OT),
                mask=m,
            )
        if nt > 13:
            tl.store(
                o_ptr + 13 * n + idx,
                fn(tl.load(a13 + idx, mask=m).to(OT), s13.to(OT)).to(OT),
                mask=m,
            )
        if nt > 14:
            tl.store(
                o_ptr + 14 * n + idx,
                fn(tl.load(a14 + idx, mask=m).to(OT), s14.to(OT)).to(OT),
                mask=m,
            )
        if nt > 15:
            tl.store(
                o_ptr + 15 * n + idx,
                fn(tl.load(a15 + idx, mask=m).to(OT), s15.to(OT)).to(OT),
                mask=m,
            )


@triton.jit
def _scalar1_kernel(
    a_ptr,
    o_ptr,
    s,
    num_tasks,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    fn: tl.constexpr,
    OT: tl.constexpr,
):
    """One tensor per launch, streaming tile.

    ``BLOCK`` is the widest tile that still streams at full bandwidth here.
    """
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        tl.store(o_ptr + idx, fn(tl.load(a_ptr + idx).to(OT), s.to(OT)).to(OT))
    else:
        m = idx < num_tasks
        tl.store(
            o_ptr + idx,
            fn(tl.load(a_ptr + idx, mask=m).to(OT), s.to(OT)).to(OT),
            mask=m,
        )


# ``.Scalar`` / ``.ScalarList`` share one element-wise function per operator.
_OPS = {"add": _fadd, "sub": _fsub, "mul": _fmul, "div": _fdiv}

_FAST_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# Above this numel one launch carrying all 16 tensors loses to the streaming
# tile: the gain from skipping launches no longer covers the idle tiles.
_K16_MAX_NUMEL = 1 << 19

_BIG_BLOCK = 32768
_BIG_WARPS = 8
# A 128K tile with 16 warps streams fastest for fp16; fp32 and bf16 gain
# nothing from it.
_BIG_TILES = {torch.float16: (131072, 16, 16384)}


def _launch_big(src, dst, val, fn):
    n = src.numel()
    block, warps, limit = _BIG_TILES.get(src.dtype, (_BIG_BLOCK, _BIG_WARPS, 8192))
    _scalar1_kernel[(triton.cdiv(n, block),)](
        src,
        dst,
        float(val),
        n,
        block,
        n % block == 0,
        fn,
        tl_dtype(dst.dtype),
        num_warps=warps,
        unroll_num=8,
        buffer_size_limit=limit,
    )


def _launch_list(srcs, dsts, vals, fn, flat=None):
    n = srcs[0].numel()
    nt = len(srcs)
    if nt <= _K16_SLOTS and n <= _K16_MAX_NUMEL:
        pad = _K16_SLOTS - nt
        # Padding slots sit behind ``NT > i`` guards, so their pointers are
        # only required to be valid.
        block, even = _pick_block(n)
        grid = (triton.cdiv(n, block),)
        srcs = list(srcs) + [srcs[0]] * pad
        svals = list(vals) + [0.0] * pad
        if flat is not None:
            _foreach16_flat_kernel[grid](
                *srcs,
                flat,
                *svals,
                n,
                nt,
                fn,
                tl_dtype(dsts[0].dtype),
                block,
                even,
                num_warps=8,
            )
            return
        _foreach16_kernel[grid](
            *srcs,
            *(list(dsts) + [dsts[0]] * pad),
            *svals,
            n,
            fn,
            nt,
            tl_dtype(dsts[0].dtype),
            block,
            even,
            num_warps=8,
        )
        return
    for src, dst, val in zip(srcs, dsts, vals):
        _launch_big(src, dst, val, fn)


def _uniform_float(tensors) -> bool:
    """True when one launch can carry every pointer as a kernel argument."""
    if not tensors:
        return False
    dtype = tensors[0].dtype
    if dtype not in _FAST_DTYPES:
        return False
    device = tensors[0].device
    numel = tensors[0].numel()
    for t in tensors:
        if t.dtype is not dtype or t.device != device or t.numel() != numel:
            return False
        if not t.is_contiguous():
            return False
    return True


# One allocation plus one unbind replaces ``len(tensors)`` allocator calls, which
# is most of the host time at the small shapes.  Above this total the single
# block is large enough that per-tensor allocation is cheaper to place.
_FLAT_BYTES_LIMIT = 64 * 1024 * 1024


def _outputs(tensors):
    """Return ``(outputs, flat)``; ``flat`` is the one allocation behind them."""
    first = tensors[0]
    shape = tuple(first.shape)
    if len(tensors) * first.numel() * first.element_size() <= _FLAT_BYTES_LIMIT:
        try:
            flat = torch.empty(
                (len(tensors),) + shape, dtype=first.dtype, device=first.device
            )
            return flat.unbind(0), flat
        except Exception:
            pass
    return [torch.empty_like(t) for t in tensors], None


def _generic(attr):
    from flag_gems.ops import _foreach_binary

    return getattr(_foreach_binary, attr)


def _apply(name, tensors, scalars, scalar_list):
    tensors = check_tensor_list(tensors)
    values = None
    if scalar_list:
        if isinstance(scalars, (list, tuple)) and len(scalars) == len(tensors):
            if all(isinstance(v, (bool, int, float)) for v in scalars):
                values = [float(v) for v in scalars]
    elif isinstance(scalars, (bool, int, float)):
        values = [float(scalars)] * len(tensors)
    if values is not None and _uniform_float(tensors):
        out, flat = _outputs(tensors)
        _launch_list(tensors, out, values, _OPS[name], flat)
        return out
    attr = "_foreach_%s_%s" % (name, "ScalarList" if scalar_list else "Scalar")
    return _generic(attr)(tensors, scalars)


def _make(name, scalar_list):
    if scalar_list:

        def wrapper(self, scalar_list_, *, alpha=1):
            logger.debug("GEMS_KUNLUNXIN _FOREACH_%s_SCALARLIST", name.upper())
            if alpha != 1:
                return _generic("_foreach_%s_ScalarList" % name)(
                    self, scalar_list_, alpha=alpha
                )
            return _apply(name, self, scalar_list_, True)

    else:

        def wrapper(self, scalar, *, alpha=1):
            logger.debug("GEMS_KUNLUNXIN _FOREACH_%s_SCALAR", name.upper())
            if alpha != 1:
                return _generic("_foreach_%s_Scalar" % name)(self, scalar, alpha=alpha)
            return _apply(name, self, scalar, False)

    return wrapper


_foreach_add_Scalar = _make("add", False)
_foreach_add_ScalarList = _make("add", True)
_foreach_sub_Scalar = _make("sub", False)
_foreach_sub_ScalarList = _make("sub", True)
_foreach_mul_Scalar = _make("mul", False)
_foreach_mul_ScalarList = _make("mul", True)
_foreach_div_Scalar = _make("div", False)
_foreach_div_ScalarList = _make("div", True)

__all__ = [
    "_foreach_add_Scalar",
    "_foreach_add_ScalarList",
    "_foreach_sub_Scalar",
    "_foreach_sub_ScalarList",
    "_foreach_mul_Scalar",
    "_foreach_mul_ScalarList",
    "_foreach_div_Scalar",
    "_foreach_div_ScalarList",
]
