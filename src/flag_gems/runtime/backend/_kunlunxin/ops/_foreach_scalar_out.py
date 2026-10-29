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


# bf16 is computed in fp32 and converted back.
@triton.jit
def _fadd(a, s):
    if a.dtype == tl.bfloat16:
        return (a.to(tl.float32) + s.to(tl.float32)).to(tl.bfloat16)
    if a.dtype == tl.int1:
        return a | s
    return a + s


@triton.jit
def _fsub(a, s):
    if a.dtype == tl.bfloat16:
        return (a.to(tl.float32) - s.to(tl.float32)).to(tl.bfloat16)
    return a - s


@triton.jit
def _fmul(a, s):
    if a.dtype == tl.bfloat16:
        return (a.to(tl.float32) * s.to(tl.float32)).to(tl.bfloat16)
    if a.dtype == tl.int1:
        return a & s
    return a * s


@triton.jit
def _fdiv(a, s):
    if a.dtype == tl.bfloat16:
        return (a.to(tl.float32) / s.to(tl.float32)).to(tl.bfloat16)
    return a / s


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


def _launch_list(srcs, dsts, vals, fn):
    n = srcs[0].numel()
    nt = len(srcs)
    if nt <= _K16_SLOTS and n <= _K16_MAX_NUMEL:
        pad = _K16_SLOTS - nt
        # Padding slots sit behind ``NT > i`` guards, so their pointers are
        # only required to be valid.
        block, even = _pick_block(n)
        _foreach16_kernel[(triton.cdiv(n, block),)](
            *(list(srcs) + [srcs[0]] * pad),
            *(list(dsts) + [dsts[0]] * pad),
            *(list(vals) + [0.0] * pad),
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
    first = tensors[0]
    shape = tuple(first.shape)
    if len(tensors) * first.numel() * first.element_size() <= _FLAT_BYTES_LIMIT:
        try:
            return torch.empty(
                (len(tensors),) + shape, dtype=first.dtype, device=first.device
            ).unbind(0)
        except Exception:
            pass
    return [torch.empty_like(t) for t in tensors]


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
        out = _outputs(tensors)
        _launch_list(tensors, out, values, _OPS[name])
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
