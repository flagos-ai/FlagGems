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

"""In-place ``aten::ldexp_`` for Kunlunxin/XPU3.

Same root cause as the out-of-place override (``_kunlunxin/ops/ldexp.py``): the
generic ``flag_gems.ops.ldexp_`` routes its integral-exponent kernels through
``tl_extra_shim.ldexp`` (libdevice ``ldexpf``), which has **no implementation**
on this FlagTree/XPU3 build, so every kernel fails at link time with
``ld.lld: error: undefined symbol: Unsupported``.  Its float / complex paths
call ``_exp2`` (``tl_extra_shim.exp2``), which links but is numerically wrong
here (``exp2(128) -> inf``).

This module keeps the generic host semantics (in-place alias validation,
internal-overlap rejection, broadcast-shape check, dtype castability, partial
overlap copy) and swaps in the exact exponent-field math already proven by the
out-of-place ``ldexp`` override:

* integral exponent  -> ``_ldexp_i32_exact`` / ``_ldexp_i32_fast`` from
  ``.ldexp`` (two normal-range ``2**e`` factors + FTZ denormal rescale +
  zero-sign fix);
* floating exponent  -> a local kernel with the same exact split, plus aten's
  single-``2**y`` overflow behaviour (``x * 2**128`` overflows even when the
  product would be finite);
* complex operand    -> ``_ldexp_complex_func`` / a local integral-complex
  kernel that reproduces aten's full complex multiply (including its NaN
  cross-terms).

In-place specifics:

* The fast flat kernel is the *same* ``_ldexp_flat_kernel``; it is launched
  with ``x_ptr == o_ptr == self``, so each lane loads before it stores and the
  storage / stride / ``data_ptr`` of ``self`` are untouched (the operator
  returns the very same tensor object).
* The generic pointwise kernels are called with ``out0=self`` (the
  Kunlunxin ``pointwise_dynamic`` supports in-place outputs) for the
  broadcast / non-contiguous / float-exponent shapes.
* Exact self/other aliasing is safe (element-wise dependency); *partial*
  storage overlap is resolved by copying ``other`` to a temporary first, the
  same rule the generic implementation uses.
"""

import importlib
import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.copy import copy_ as _gems_copy_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.shape_utils import MemOverlap, has_internal_overlapping

from ..utils.pointwise_dynamic import pointwise_dynamic
from .ldexp import (
    _CONFIG,
    _FLAT_OPTS,
    _PLAN_CACHE,
    _can_use_flat,
    _exponent_wide,
    _ldexp_complex_func,
    _ldexp_flat_kernel,
    _ldexp_int_func,
    _pick_geometry,
    _scale_parts,
    _split_float_exponent,
)

logger = logging.getLogger(__name__)

_generic = importlib.import_module("flag_gems.ops.ldexp_")

_INF = tl.constexpr(float("inf"))


# ---------------------------------------------------------------------------
# in-place flat kernel (same-shape, both contiguous, integral exponent)
# ---------------------------------------------------------------------------
def _run_flat_inplace(self, other):
    n = self.numel()
    exact = self.dtype == torch.float32
    wide = other.dtype == torch.int64
    key = (
        torch_device_fn.current_device(),
        self.dtype,
        other.dtype,
        n,
        exact,
        wide,
    )
    plan = _PLAN_CACHE.get(key)
    if plan is None:
        if len(_PLAN_CACHE) >= 256:
            _PLAN_CACHE.clear()
        plan = _pick_geometry(n)
        _PLAN_CACHE[key] = plan
    blk, grid, masked = plan
    # ``x_ptr`` and ``o_ptr`` are the same buffer: Triton emits one load then
    # one store per lane, so the in-place update is alias safe.
    _ldexp_flat_kernel[(grid,)](
        self,
        other,
        self,
        n,
        BLOCK=blk,
        MASKED=masked,
        EXACT=exact,
        WIDE_EXP=wide,
        **_FLAT_OPTS,
    )


# ---------------------------------------------------------------------------
# complex integral exponent (in-place ``_write_complex`` happens host-side)
# ---------------------------------------------------------------------------
@pointwise_dynamic(
    is_tensor=[True, True, True],
    num_outputs=2,
    promotion_methods=[
        (0, 1, 2, "INT_TO_FLOAT"),
        (0, 1, 2, "INT_TO_FLOAT"),
    ],
    config=_CONFIG,
)
@triton.jit
def _ldexp_complex_int_func(xr, xi, y):
    # aten scales a complex value by the *real* factor ``2**e`` through a full
    # complex multiply, so the zero cross-terms ``xi * 0`` / ``xr * 0`` produce
    # NaN whenever the other component is non-finite (e.g. 1+infj at e=0 ->
    # nan+infj).  Reproduce that formula with an exact single-fp32 ``2**e``
    # (the two normal-range factors are multiplied first, so ``e >= 128``
    # overflows to +inf exactly like ``exp2f``).
    e = _exponent_wide(y)
    s1, s2 = _scale_parts(e)
    scale = s1 * s2
    xr = xr.to(tl.float32)
    xi = xi.to(tl.float32)
    return xr * scale - xi * 0.0, xr * 0.0 + xi * scale


def _complex_integral_inplace(self, other):
    real_dtype = torch.float64 if self.dtype == torch.complex128 else torch.float32
    xr, xi = _generic._complex_parts(self, real_dtype)
    real, imag = _ldexp_complex_int_func(xr, xi, other)
    _generic._write_complex(self, real, imag)


def _complex_float_inplace(self, other):
    real_dtype = torch.float64 if self.dtype == torch.complex128 else torch.float32
    xr, xi = _generic._complex_parts(self, real_dtype)
    yr, yi = _generic._complex_parts(other, real_dtype)
    real, imag = _ldexp_complex_func(xr, xi, yr, yi)
    _generic._write_complex(self, real, imag)


# ---------------------------------------------------------------------------
# floating exponent (aten semantics: a single ``2**y`` factor)
# ---------------------------------------------------------------------------
@pointwise_dynamic(
    is_tensor=[True, True],
    promotion_methods=[(0, 1, "INT_TO_FLOAT")],
    config=_CONFIG,
)
@triton.jit
def _ldexp_float_inplace_func(x, y):
    xf = x.to(tl.float32)
    yf = y.to(tl.float32)
    bits = xf.to(tl.int32, bitcast=True)
    s1, s2, fexp = _split_float_exponent(yf)
    r = ((xf * s1) * s2) * fexp
    # aten computes ``x * 2**y`` with a single fp32 ``2**y`` factor, so the
    # intermediate overflows to +inf for ``y >= 128`` even when ``x * 2**y``
    # stays finite (2**-126 * 2**128 -> inf, 0 * 2**1024 -> NaN).  Reproduce
    # that overflow; keep the exact split everywhere else.
    tiny = (bits & 0x7FFFFFFF) < 0x00800000  # zero or denormal encoding
    denormal = tiny & ((bits & 0x7FFFFF) != 0)
    r_ovf = xf * _INF
    # A denormal ``x`` is flushed to zero by the multiplier; its true value is
    # nonzero, so x * inf must be a signed inf rather than NaN.
    r_ovf = tl.where(denormal, tl.where(bits < 0, -_INF, _INF), r_ovf)
    r = tl.where(yf >= 128.0, r_ovf, r)
    # ``2**-inf`` is 0, so ``x * 2**-inf`` is a signed zero (inf * 0 -> NaN).
    r = tl.where(yf == -_INF, xf * 0.0, r)
    # NaN exponent wins over everything.
    r = tl.where(yf != yf, yf, r)
    # A flushed-to-zero product loses its sign on XPU3; torch keeps the sign.
    return (r.to(tl.int32, bitcast=True) | (bits & -2147483648)).to(
        tl.float32, bitcast=True
    )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def ldexp_(self, other):
    logger.debug("GEMS_KUNLUNXIN LDEXP_")
    if has_internal_overlapping(self) == MemOverlap.Yes:
        raise RuntimeError(
            "unsupported operation: more than one element of the written-to tensor "
            "refers to a single memory location"
        )
    # Partial self/other storage overlap: materialise ``other`` first.  Exact
    # aliasing is fine (each output depends only on the same element).
    if (
        self.device == other.device
        and self.untyped_storage().nbytes() > 0
        and self.untyped_storage().data_ptr() == other.untyped_storage().data_ptr()
        and not _generic._is_exact_alias(self, other)
    ):
        temporary = torch.empty_like(other)
        if other.is_complex():
            real, imag = _generic._complex_parts(
                other,
                torch.float64 if other.dtype == torch.complex128 else torch.float32,
            )
            _generic._write_complex(temporary, real, imag)
        else:
            _gems_copy_(temporary, other)
        other = temporary

    result_dtype = _generic._result_dtype(self, other)
    if not torch.can_cast(result_dtype, self.dtype):
        raise RuntimeError(
            f"result type {result_dtype} can't be cast to the desired output "
            f"type {self.dtype}"
        )

    output_shape = torch.broadcast_shapes(self.shape, other.shape)
    if output_shape != self.shape:
        raise RuntimeError(
            f"output with shape {self.shape} doesn't match the broadcast shape "
            f"{output_shape}"
        )

    if self.numel() == 0:
        return self

    integral_exponent = not other.is_floating_point() and not other.is_complex()
    if self.is_complex():
        if integral_exponent:
            _complex_integral_inplace(self, other)
        else:
            _complex_float_inplace(self, other)
    elif integral_exponent:
        if _can_use_flat(self, other):
            _run_flat_inplace(self, other)
        else:
            _ldexp_int_func(self, other, out0=self)
    else:
        _ldexp_float_inplace_func(self, other, out0=self)
    return self
