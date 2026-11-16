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

"""Registration table for the unary ``aten::_foreach_*`` family.

Thirty operators share one multi-tensor executor.  They differ only in the
element-wise math, the output dtype policy, and which dtypes ATen accepts, so
each one is a table row rather than a copy of the wrapper.  The element-wise
math is *reused* from the existing single-tensor FlagGems implementations by
taking the ``triton.jit`` scalar function out of their ``pointwise_dynamic``
wrapper -- no formula is written twice.

The dtype policies below were measured on this backend rather than inferred
from operator names.  Notable measured facts:

* Transcendentals promote every integral and bool input to ``float32``, while
  ``ceil``/``floor``/``round``/``trunc``/``frac``/``neg``/``sign`` preserve the
  input dtype.
* The exclusions differ per operator: ``floor`` rejects ``bool`` but accepts
  all integers, ``frac`` rejects all integers, ``sign`` accepts ``bool`` but
  rejects complex, ``erf``/``erfc``/``lgamma`` reject complex.
* Only ``abs`` maps complex to real; it is the one operator whose output dtype
  differs from its input for a complex input.
"""

import logging
import os
from typing import Callable, Dict, Optional

import torch
import triton
import triton.language as tl
import triton.language.extra.xpu.libdevice as xpu

from flag_gems.ops._foreach_complex_math import (
    _hypot_fn,
    c_acos,
    c_asin,
    c_atan,
    c_cos,
    c_cosh,
    c_exp,
    c_expm1,
    c_log,
    c_log1p,
    c_log2,
    c_log10,
    c_neg,
    c_reciprocal,
    c_sigmoid,
    c_sin,
    c_sinh,
    c_sqrt,
    c_tan,
    c_tanh,
)
from flag_gems.ops._foreach_unary import c_rsqrt
from flag_gems.ops.atan import atan_kernel
from flag_gems.ops.ceil import ceil_func
from flag_gems.ops.cos import cos_func
from flag_gems.ops.cosh import cosh_func
from flag_gems.ops.erf import erf_func
from flag_gems.ops.exp import exp_func
from flag_gems.ops.expm1 import expm1_func
from flag_gems.ops.floor import floor_func
from flag_gems.ops.frac_ import frac_func
from flag_gems.ops.lgamma_ import lgamma_func
from flag_gems.ops.log import log_func
from flag_gems.ops.log1p import log1p_func
from flag_gems.ops.log10 import log10_func
from flag_gems.ops.neg import neg_func
from flag_gems.ops.round import round_half_to_even_impl
from flag_gems.ops.sigmoid import sigmoid_forward
from flag_gems.ops.sin import sin_func
from flag_gems.ops.sinh import sinh_kernel
from flag_gems.ops.special_erfc import _erfc_kernel
from flag_gems.ops.tan import tan_func
from flag_gems.ops.tanh import tanh_kernel
from flag_gems.ops.trunc_ import trunc_func
from flag_gems.utils import tl_extra_shim
from flag_gems.utils.foreach import _STATS, check_dtype_supported, check_tensor_list
from flag_gems.utils.foreach import foreach_unary as _generic_foreach_unary
from flag_gems.utils.foreach import has_internal_overlap, is_flat_addressable, tl_dtype

from ..utils.foreach import foreach_unary, int_to_float, real_dtype_of, same_dtype

# Shares the generic operator's logger name; only the GEMS prefix differs.
logger = logging.getLogger(f"flag_gems.ops.{__name__.rsplit('.', 1)[-1]}")


@triton.jit
def _sqrt_func(x):
    # x * rsqrt(x) is sqrt(x); the zero case would otherwise be 0 * inf.
    v = x.to(tl.float32)
    return tl.where(v == 0.0, v, v * tl_extra_shim.rsqrt(v))


@triton.jit
def _reciprocal_func(x):
    # rsqrt needs a non-negative argument, so the sign is reapplied afterwards.
    v = x.to(tl.float32)
    p = tl_extra_shim.rsqrt(tl.abs(v))
    p = p * p
    return tl.where(v < 0.0, -p, p)


@triton.jit
def _acos_func(x):
    # acos(x) = pi/2 - asin(x)
    v = x.to(tl.float32)
    return 1.5707963267948966 - tl_extra_shim.atan(v / tl.sqrt((1.0 - v) * (1.0 + v)))


@triton.jit
def _asin_func(x):
    # asin(x) = atan(x / sqrt((1 - x) * (1 + x)))
    v = x.to(tl.float32)
    return tl_extra_shim.atan(v / tl.sqrt((1.0 - v) * (1.0 + v)))


def _scalar_fn(pointwise_fn) -> Callable:
    """The bare ``triton.jit`` function inside a ``pointwise_dynamic`` wrapper.

    ``pointwise_dynamic`` keeps the user's scalar function so it can splice the
    source into a generated module; the foreach executor needs exactly the same
    object as a ``tl.constexpr`` callable.  Reaching for it is what lets thirty
    operators reuse the repository's existing math instead of restating it.
    """
    return pointwise_fn._scalar_fn


# ---------------------------------------------------------------------------
# math not available as a reusable pointwise_dynamic scalar function
# ---------------------------------------------------------------------------
#
# ``round`` and ``sign`` are implemented in FlagGems as hand-written kernels
# rather than ``pointwise_dynamic``, so there is no scalar function to borrow.
# ``round`` still reuses the repository's tie-breaking device function; only the
# dtype guard around it is restated here.


@triton.jit
def _round_func(x):
    # Integral inputs are already rounded; float ones go through the shared
    # half-to-even helper from ops/round.py, which requires fp32.
    if x.dtype.is_floating():
        return round_half_to_even_impl(x.to(tl.float32)).to(x.dtype)
    else:
        return x


@triton.jit
def _sign_func(x):
    # Mirrors ops/sign.py: NaN yields 0 because neither comparison holds.
    if x.dtype == tl.int1:
        return x
    else:
        return (x > 0).to(x.dtype) - (x < 0).to(x.dtype)


@triton.jit
def _abs_func(x):
    return tl.abs(x)


@triton.jit
def _log2_func(x):
    # Base-2 logarithm as a ratio of natural logarithms: the direct form
    # returns the natural logarithm on this backend.
    return tl.log(x.to(tl.float32)) / 0.6931471805599453


@triton.jit
def _abs_complex_func(re, im):
    # Modulus of a complex value.  The executor passes the real and imaginary
    # components separately because Triton has no complex dtype.
    return _hypot_fn(re, im)


# ---------------------------------------------------------------------------
# dtype sets, as measured
# ---------------------------------------------------------------------------

_FLOAT = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_INT = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
_COMPLEX = (torch.complex32, torch.complex64, torch.complex128)

# Everything except complex: erf / erfc / lgamma / sign.
NO_COMPLEX = frozenset(_FLOAT + _INT + (torch.bool,))
# Integral dtypes are fine, bool is not: ceil / floor / round / trunc.
NO_BOOL_NO_COMPLEX = frozenset(_FLOAT + _INT)
# Floating point only: frac.
FLOAT_ONLY = frozenset(_FLOAT)
# Everything except bool: neg.
NO_BOOL = frozenset(_FLOAT + _INT + (torch.complex64, torch.complex128))
# Transcendentals: complex64/128 work, complex32 does not.  Measured -- only
# ``abs`` accepts complex32 in this build, ComplexHalf being experimental.
NO_COMPLEX32 = frozenset(
    _FLOAT + _INT + (torch.bool, torch.complex64, torch.complex128)
)


class UnaryOp:
    """One row of the unary foreach table."""

    __slots__ = ("name", "fn", "out_dtype_fn", "allowed", "complex_fn")

    def __init__(
        self,
        name: str,
        fn: Callable,
        out_dtype_fn: Callable = int_to_float,
        allowed: Optional[frozenset] = None,
        complex_fn: Optional[Callable] = None,
    ) -> None:
        self.name = name
        self.fn = fn
        self.out_dtype_fn = out_dtype_fn
        self.allowed = allowed
        self.complex_fn = complex_fn


UNARY_OPS: Dict[str, UnaryOp] = {
    op.name: op
    for op in (
        UnaryOp("abs", _abs_func, real_dtype_of, None, _abs_complex_func),
        UnaryOp("acos", _acos_func, int_to_float, NO_COMPLEX32, c_acos),
        UnaryOp("asin", _asin_func, int_to_float, NO_COMPLEX32, c_asin),
        UnaryOp("atan", _scalar_fn(atan_kernel), int_to_float, NO_COMPLEX32, c_atan),
        UnaryOp("ceil", _scalar_fn(ceil_func), same_dtype, NO_BOOL_NO_COMPLEX),
        UnaryOp("cos", _scalar_fn(cos_func), int_to_float, NO_COMPLEX32, c_cos),
        UnaryOp("cosh", _scalar_fn(cosh_func), int_to_float, NO_COMPLEX32, c_cosh),
        UnaryOp("erf", _scalar_fn(erf_func), int_to_float, NO_COMPLEX),
        UnaryOp("erfc", _scalar_fn(_erfc_kernel), int_to_float, NO_COMPLEX),
        UnaryOp("exp", _scalar_fn(exp_func), int_to_float, NO_COMPLEX32, c_exp),
        UnaryOp("expm1", _scalar_fn(expm1_func), int_to_float, NO_COMPLEX32, c_expm1),
        UnaryOp("floor", _scalar_fn(floor_func), same_dtype, NO_BOOL_NO_COMPLEX),
        UnaryOp("frac", _scalar_fn(frac_func), same_dtype, FLOAT_ONLY),
        UnaryOp("lgamma", _scalar_fn(lgamma_func), int_to_float, NO_COMPLEX),
        UnaryOp("log", _scalar_fn(log_func), int_to_float, NO_COMPLEX32, c_log),
        UnaryOp("log10", _scalar_fn(log10_func), int_to_float, NO_COMPLEX32, c_log10),
        UnaryOp("log1p", _scalar_fn(log1p_func), int_to_float, NO_COMPLEX32, c_log1p),
        UnaryOp("log2", _log2_func, int_to_float, NO_COMPLEX32, c_log2),
        UnaryOp("neg", _scalar_fn(neg_func), same_dtype, NO_BOOL, c_neg),
        UnaryOp(
            "reciprocal",
            _reciprocal_func,
            int_to_float,
            NO_COMPLEX32,
            c_reciprocal,
        ),
        UnaryOp("round", _round_func, same_dtype, NO_BOOL_NO_COMPLEX),
        UnaryOp(
            "sigmoid",
            _scalar_fn(sigmoid_forward),
            int_to_float,
            NO_COMPLEX32,
            c_sigmoid,
        ),
        UnaryOp("sign", _sign_func, same_dtype, NO_COMPLEX),
        UnaryOp("sin", _scalar_fn(sin_func), int_to_float, NO_COMPLEX32, c_sin),
        UnaryOp("sinh", _scalar_fn(sinh_kernel), int_to_float, NO_COMPLEX32, c_sinh),
        UnaryOp("sqrt", _sqrt_func, int_to_float, NO_COMPLEX32, c_sqrt),
        UnaryOp("tan", _scalar_fn(tan_func), int_to_float, NO_COMPLEX32, c_tan),
        UnaryOp("tanh", _scalar_fn(tanh_kernel), int_to_float, NO_COMPLEX32, c_tanh),
        UnaryOp("trunc", _scalar_fn(trunc_func), same_dtype, NO_BOOL_NO_COMPLEX),
    )
}


def apply_unary(name: str, self, inplace: bool):
    """Dispatch one table row through the shared executor."""
    op = UNARY_OPS[name]
    logger.debug("GEMS_KUNLUNXIN _FOREACH_%s%s", name.upper(), "_" if inplace else "")
    if inplace:
        for t in self:
            if t.dtype.is_complex and op.out_dtype_fn is real_dtype_of:
                raise RuntimeError(
                    f"In-place {name} is not supported for complex tensors."
                )
        foreach_unary(
            self,
            op.fn,
            complex_fn=op.complex_fn,
            # The dtype policy has to be passed here too: it is what makes the
            # executor reject an integral in-place call on a promoting operator,
            # matching ATen's "result type Float can't be cast to the desired
            # output type Long".  Omitting it silently accepted int64 sin_.
            out_dtype_fn=op.out_dtype_fn,
            inplace=True,
            allowed_dtypes=op.allowed,
        )
        # The in-place schemas return ``()``; returning the list makes the
        # dispatcher reject the kernel.
        return None
    return foreach_unary(
        self,
        op.fn,
        complex_fn=op.complex_fn,
        out_dtype_fn=op.out_dtype_fn,
        allowed_dtypes=op.allowed,
    )


def _foreach_abs(self):
    return apply_unary("abs", self, inplace=False)


def _foreach_abs_(self):
    return apply_unary("abs", self, inplace=True)


def _foreach_acos(self):
    return apply_unary("acos", self, inplace=False)


def _foreach_acos_(self):
    return apply_unary("acos", self, inplace=True)


def _foreach_asin(self):
    return apply_unary("asin", self, inplace=False)


def _foreach_asin_(self):
    return apply_unary("asin", self, inplace=True)


def _foreach_atan(self):
    return apply_unary("atan", self, inplace=False)


def _foreach_atan_(self):
    return apply_unary("atan", self, inplace=True)


def _foreach_ceil(self):
    return apply_unary("ceil", self, inplace=False)


def _foreach_ceil_(self):
    return apply_unary("ceil", self, inplace=True)


def _foreach_cos(self):
    return apply_unary("cos", self, inplace=False)


def _foreach_cos_(self):
    return apply_unary("cos", self, inplace=True)


def _foreach_cosh(self):
    return apply_unary("cosh", self, inplace=False)


def _foreach_cosh_(self):
    return apply_unary("cosh", self, inplace=True)


def _foreach_erf(self):
    return apply_unary("erf", self, inplace=False)


def _foreach_erf_(self):
    return apply_unary("erf", self, inplace=True)


def _foreach_erfc(self):
    return apply_unary("erfc", self, inplace=False)


def _foreach_erfc_(self):
    return apply_unary("erfc", self, inplace=True)


def _foreach_exp(self):
    return apply_unary("exp", self, inplace=False)


def _foreach_exp_(self):
    return apply_unary("exp", self, inplace=True)


def _foreach_expm1(self):
    return apply_unary("expm1", self, inplace=False)


def _foreach_expm1_(self):
    return apply_unary("expm1", self, inplace=True)


def _foreach_floor(self):
    return apply_unary("floor", self, inplace=False)


def _foreach_floor_(self):
    return apply_unary("floor", self, inplace=True)


def _foreach_frac(self):
    return apply_unary("frac", self, inplace=False)


def _foreach_frac_(self):
    return apply_unary("frac", self, inplace=True)


def _foreach_lgamma(self):
    return apply_unary("lgamma", self, inplace=False)


def _foreach_lgamma_(self):
    return apply_unary("lgamma", self, inplace=True)


def _foreach_log(self):
    return apply_unary("log", self, inplace=False)


def _foreach_log_(self):
    return apply_unary("log", self, inplace=True)


def _foreach_log10(self):
    return apply_unary("log10", self, inplace=False)


def _foreach_log10_(self):
    return apply_unary("log10", self, inplace=True)


def _foreach_log1p(self):
    return apply_unary("log1p", self, inplace=False)


def _foreach_log1p_(self):
    return apply_unary("log1p", self, inplace=True)


def _foreach_log2(self):
    return apply_unary("log2", self, inplace=False)


def _foreach_log2_(self):
    return apply_unary("log2", self, inplace=True)


def _foreach_neg(self):
    return apply_unary("neg", self, inplace=False)


def _foreach_neg_(self):
    return apply_unary("neg", self, inplace=True)


def _foreach_reciprocal(self):
    return apply_unary("reciprocal", self, inplace=False)


def _foreach_reciprocal_(self):
    return apply_unary("reciprocal", self, inplace=True)


def _foreach_round(self):
    return apply_unary("round", self, inplace=False)


def _foreach_round_(self):
    return apply_unary("round", self, inplace=True)


def _foreach_sigmoid(self):
    return apply_unary("sigmoid", self, inplace=False)


def _foreach_sigmoid_(self):
    return apply_unary("sigmoid", self, inplace=True)


def _foreach_sign(self):
    return apply_unary("sign", self, inplace=False)


def _foreach_sign_(self):
    return apply_unary("sign", self, inplace=True)


def _foreach_sin(self):
    return apply_unary("sin", self, inplace=False)


def _foreach_sin_(self):
    return apply_unary("sin", self, inplace=True)


def _foreach_sinh(self):
    return apply_unary("sinh", self, inplace=False)


def _foreach_sinh_(self):
    return apply_unary("sinh", self, inplace=True)


def _foreach_sqrt(self):
    return apply_unary("sqrt", self, inplace=False)


def _foreach_sqrt_(self):
    return apply_unary("sqrt", self, inplace=True)


def _foreach_tan(self):
    return apply_unary("tan", self, inplace=False)


def _foreach_tan_(self):
    return apply_unary("tan", self, inplace=True)


def _foreach_tanh(self):
    return apply_unary("tanh", self, inplace=False)


def _foreach_tanh_(self):
    return apply_unary("tanh", self, inplace=True)


def _foreach_trunc(self):
    return apply_unary("trunc", self, inplace=False)


def _foreach_trunc_(self):
    return apply_unary("trunc", self, inplace=True)


@triton.jit
def _rsqrt_sfu(x):
    # Compute in fp32 through the SFU; the store casts back. ``x`` is already
    # promoted to the output dtype by the executor.
    return xpu.rsqrt(x.to(tl.float32))


@triton.jit
def _rsqrt_flat_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    OUT_DT: tl.constexpr,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """One tensor, pointers passed as direct kernel args.

    Direct base pointers (vs loading from a metadata buffer at runtime) let the
    compiler prove alignment and emit a wide coalesced block-DMA; the indirect
    form caps bandwidth on a pure-bandwidth op like rsqrt.
    """
    offset = tl.program_id(0) * BLOCK
    idx = offset + tl.arange(0, BLOCK)
    if NEED_MASK:
        mask = idx < n_elements
        x = tl.load(in_ptr + idx, mask=mask, other=0)
        tl.store(out_ptr + idx, xpu.rsqrt(x.to(tl.float32)).to(OUT_DT), mask=mask)
    else:
        x = tl.load(in_ptr + idx)
        tl.store(out_ptr + idx, xpu.rsqrt(x.to(tl.float32)).to(OUT_DT))


class _bf16_fast:
    """Enable the faster bf16 store path.

    The env is read at triton compile time, so wrapping the launch that first
    compiles a bf16 kernel is enough; it is restored afterward.
    """

    def __init__(self, on):
        self.on = on
        self.prev = None

    def __enter__(self):
        if self.on:
            self.prev = os.environ.get("TRITONXPU_BF16_FAST")
            os.environ["TRITONXPU_BF16_FAST"] = "1"

    def __exit__(self, *exc):
        if self.on:
            if self.prev is None:
                os.environ.pop("TRITONXPU_BF16_FAST", None)
            else:
                os.environ["TRITONXPU_BF16_FAST"] = self.prev
        return False


@triton.jit
def _foreach_unary_kernel_xpu(
    meta_ptr,
    fn: tl.constexpr,
    NT: tl.constexpr,
    IN_DT: tl.constexpr,
    OUT_DT: tl.constexpr,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """One launch, ``NT`` tensors; 2D grid ``(NT, chunks)``.

    ``meta_ptr`` (int64): ``[0,NT)`` input ptrs, ``[NT,2NT)`` output ptrs,
    ``[2NT,3NT)`` element counts. When every tensor shares one numel that
    divides ``BLOCK``, the grid covers each exactly, so ``NEED_MASK`` is False
    and load/store run mask-free (the masked path is slower even when the mask
    is trivially true).
    """
    t = tl.program_id(0)
    offset = tl.program_id(1) * BLOCK
    in_ptr = tl.load(meta_ptr + t).to(tl.pointer_type(IN_DT))
    out_ptr = tl.load(meta_ptr + NT + t).to(tl.pointer_type(OUT_DT))
    idx = offset + tl.arange(0, BLOCK)
    if NEED_MASK:
        n_elements = tl.load(meta_ptr + 2 * NT + t)
        mask = idx < n_elements
        x = tl.load(in_ptr + idx, mask=mask, other=0)
        tl.store(out_ptr + idx, fn(x.to(OUT_DT)).to(OUT_DT), mask=mask)
    else:
        x = tl.load(in_ptr + idx)
        tl.store(out_ptr + idx, fn(x.to(OUT_DT)).to(OUT_DT))


_BIG = 1 << 20  # per-tensor direct-arg launch at/above this numel


def _pick_block_small(max_numel):
    """(block, num_warps, buffer_size_limit) for the fused metadata path.

    Only mid/small groups reach here (big ones use the direct-arg path). Small
    tensors get one chunk each -- they are launch/host bound, so a wider tile
    buys nothing.
    """
    if max_numel >= (1 << 14):
        return 8192, 8, 4096
    block = min(8192, triton.next_power_of_2(max_numel))
    return block, 4, 0


def _launch_direct(srcs, dsts, out_dtype):
    """Big shapes: one direct-arg launch per tensor.

    rsqrt is pure-bandwidth and needs direct pointers to hit full block-DMA. The
    per-tensor launch host cost is negligible against a 16M-element transfer.
    """
    BLOCK, num_warps, bsl = 131072, 32, 8192
    out_tl = tl_dtype(out_dtype)
    with _bf16_fast(out_dtype == torch.bfloat16):
        for s, d in zip(srcs, dsts):
            n = s.numel()
            need_mask = (n % BLOCK) != 0
            grid = (triton.cdiv(n, BLOCK),)
            _rsqrt_flat_kernel[grid](
                s,
                d,
                n,
                out_tl,
                BLOCK,
                need_mask,
                num_warps=num_warps,
                buffer_size_limit=bsl,
            )


def _launch_fused(srcs, dsts, in_dtype, out_dtype, device):
    """Mid/small shapes: one fused launch over the whole group (NT tensors).

    Pointers come through a metadata buffer. The per-launch host cost of a
    per-tensor split dominates at these sizes, so fusing wins even though the
    indirect pointer caps bandwidth.
    """
    numels = [t.numel() for t in srcs]
    max_numel = max(numels)
    block, num_warps, bsl = _pick_block_small(max_numel)
    nt = len(srcs)
    uniform = min(numels) == max_numel
    need_mask = not (uniform and (max_numel % block == 0))
    meta = torch.tensor(
        [t.data_ptr() for t in srcs] + [t.data_ptr() for t in dsts] + numels,
        dtype=torch.int64,
    ).to(device, non_blocking=True)
    grid = (nt, triton.cdiv(max_numel, block))
    kwargs = {"num_warps": num_warps}
    if bsl:
        kwargs["buffer_size_limit"] = bsl
    with _bf16_fast(out_dtype == torch.bfloat16):
        _foreach_unary_kernel_xpu[grid](
            meta,
            _rsqrt_sfu,
            nt,
            tl_dtype(in_dtype),
            tl_dtype(out_dtype),
            block,
            need_mask,
            **kwargs,
        )


def _launch_group_xpu(srcs, dsts, in_dtype, device):
    numels = [t.numel() for t in srcs]
    max_numel = max(numels)
    if max_numel == 0:
        return
    out_dtype = dsts[0].dtype
    if max_numel >= _BIG:
        _launch_direct(srcs, dsts, out_dtype)
    else:
        _launch_fused(srcs, dsts, in_dtype, out_dtype, device)
    _STATS.executor_launches += 1
    _STATS.groups += 1


def _foreach_rsqrt_impl(tensors, inplace):
    tensors = check_tensor_list(tensors)
    # Complex is rare; defer to the reference executor so the complex kernel and
    # dtype policy stay bit-identical.
    if any(t.dtype.is_complex for t in tensors):
        return _generic_foreach_unary(
            tensors,
            _rsqrt_sfu,
            complex_fn=c_rsqrt,
            out_dtype_fn=int_to_float,
            inplace=inplace,
            allowed_dtypes=NO_COMPLEX32,
        )

    _STATS.reset()
    results = [None] * len(tensors)
    buckets = {}
    writebacks = []
    for i, t in enumerate(tensors):
        dtype = t.dtype
        check_dtype_supported(dtype, NO_COMPLEX32)
        out_dtype = int_to_float(dtype)
        key = (t.device, dtype)
        if inplace:
            if out_dtype != dtype:
                raise RuntimeError(
                    f"result type {str(out_dtype).replace('torch.', '').capitalize()}"
                    " can't be cast to the desired output type "
                    f"{str(dtype).replace('torch.', '').capitalize()}"
                )
            if has_internal_overlap(t):
                raise RuntimeError(
                    "unsupported operation: more than one element of the "
                    "written-to tensor refers to a single memory location. "
                    "Please clone() the tensor before performing the operation."
                )
            if is_flat_addressable(t):
                src = dst = t
            else:
                src = dst = t.contiguous()
                writebacks.append((t, src))
                _STATS.staged_inputs += 1
            results[i] = t
        else:
            src = t if is_flat_addressable(t) else t.contiguous()
            if src is not t:
                _STATS.staged_inputs += 1
            dst = torch.empty_like(src, dtype=out_dtype)
            results[i] = dst
        bucket = buckets.get(key)
        if bucket is None:
            bucket = buckets[key] = ([], [])
        bucket[0].append(src)
        bucket[1].append(dst)

    for (device, dtype), (srcs, dsts) in buckets.items():
        _launch_group_xpu(srcs, dsts, dtype, device)

    for dst_view, staged in writebacks:
        dst_view.copy_(staged)
        _STATS.writebacks += 1

    return results


def _foreach_rsqrt(self):
    logger.debug("GEMS _FOREACH_RSQRT")
    return _foreach_rsqrt_impl(self, inplace=False)


def _foreach_rsqrt_(self):
    logger.debug("GEMS _FOREACH_RSQRT_")
    _foreach_rsqrt_impl(self, inplace=True)
    # In-place schema returns ``()``; returning the list makes the dispatcher
    # reject the kernel.
    return None
