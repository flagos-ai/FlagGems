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

"""Out-of-place ``aten::ldexp.Tensor`` for Kunlunxin/XPU3.

Why this override exists
------------------------
The generic implementation routes every non-complex call through
``flag_gems.ops.ldexp._ldexp``, which resolves to ``tl_extra_shim.ldexp`` --
the libdevice soft extern ``ldexpf``.  On this FlagTree/XPU3 build that extern
has **no implementation**: kernel compilation fails at link time with

    ld.lld: error: undefined symbol: Unsupported
    >>> referenced by ldexp.py:47 (src/flag_gems/ops/ldexp.py:47)

so the operator does not run at all (40/41 functional cells fail, the fp64
boundary cell is impossible on a device without float64 -- see
``hardware-xpu3-no-float64``).  The complex path additionally computes its
scale with ``tl.exp2``, which is numerically wrong on this backend.

Kernel formulation (integral exponent, the hot path)
----------------------------------------------------
``x * 2**e`` is built from the exponent field instead of a libdevice call:

    s1 = bitcast((clamp(e, -126, 126) + 127) << 23)
    s2 = bitcast((clamp(e, -252, 252) - e1 + 127) << 23)
    r  = (x * s1) * s2

Two factors keep every intermediate in the normal range, so the split is exact
for ``|e| <= 252``; beyond that the exponent is clamped, which is exact for
every representable fp32/fp16/bf16 input except ``|e| == 253`` with
``|x|`` in the two ulps just above ``2**-126`` (torch's own CPU/CUDA kernels
are further off there -- they compute ``x * exp2(e)``, i.e. NaN/inf).  Three
extra pieces are required for bit-exact IEEE behaviour on this hardware:

* **Subnormal inputs** - XPU3 runs with flush-to-zero (``FTZ``); a denormal
  *operand* is read as zero by the multiplier, so ``2**-127 * 2**128`` would
  return 0 instead of 2.  ``tests/test_ldexp.py::
  test_ldexp_integral_exponent_boundaries`` exercises exactly that.  The
  kernel detects the denormal encoding from the raw bits and rescales the
  mantissa integer (``m * 2**(e-149)``), which never feeds a denormal to the
  multiplier.  Only fp32 sources need this (an fp16/bf16 denormal is a normal
  fp32 number once promoted), so the branch is a ``tl.constexpr``.
* **Sign of a flushed-to-zero result** - the multiplier returns ``+0.0`` when
  a tiny product is flushed (``-0.0 * 2**-126 -> +0.0``), while torch keeps
  ``-0.0``.  The sign bit of the input is OR-ed onto the result bits.
* **Float exponents** - ``ldexp(x, y)`` with a floating ``y`` is ``x *
  2**y`` (aten semantics: ``2**inf`` is inf, ``0 * 2**inf`` is NaN).  The
  integer part is split into the same two exact factors, the fractional part
  goes through ``tl.exp(frac * ln2)`` (``tl.exp2`` is wrong on XPU3), and
  non-finite exponents select ``inf``/``0`` explicitly.

Per-dtype behaviour measured on the benchmark matrix (16.7M):

| dtype | torch | kernel | speedup |
|---|---|---|---|
| fp16  | 1266 us | 526 us | 2.41x |
| bf16  | 1336 us | 532 us | 2.51x |
| fp32  | 362 us | 886 us | 0.41x (denormal-exact body) |

The fp32 body carries the denormal rescale for every lane (it is the only
dtype where a denormal input is representable); fp16/bf16 run the same kernel
with that branch compiled out because their denormals are normal fp32 numbers
once promoted.  Dropping the rescale from the fp32 body would make it 444 us
(0.81x) but would return 0 instead of 2 for ``2**-127 * 2**128``.

Everything else (broadcast, non-contiguous, mixed dtypes, complex) keeps a
real XPU ``pointwise_dynamic`` kernel with the Kunlunxin ``CodeGenConfig`` --
never an ATen/torch fallback.
"""

import importlib
import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.copy import copy_ as _gems_copy_
from flag_gems.runtime import torch_device_fn

from ..utils.codegen_config_utils import CodeGenConfig
from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

_generic = importlib.import_module("flag_gems.ops.ldexp")

# --- kernel constants -------------------------------------------------------
_EXP_BIAS = tl.constexpr(127)
_LN2 = tl.constexpr(0.6931471805599453)
_INF = tl.constexpr(float("inf"))

_CONFIG = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    unroll_num=4,
)


# ---------------------------------------------------------------------------
# integral-exponent math (shared by the flat and the pointwise kernels)
# ---------------------------------------------------------------------------
@triton.jit
def _scale_parts(e):
    """Split an integer exponent into two normal-range powers of two."""
    eh = tl.maximum(tl.minimum(e, 252), -252)
    e1 = tl.maximum(tl.minimum(eh, 126), -126)
    s1 = ((e1 + _EXP_BIAS) << 23).to(tl.float32, bitcast=True)
    s2 = (((eh - e1) + _EXP_BIAS) << 23).to(tl.float32, bitcast=True)
    return s1, s2


@triton.jit
def _ldexp_i32_exact(x, e):
    """``x * 2**e`` for a float32 ``x`` and a clamped int32 ``e``.

    Handles denormal inputs: XPU3 flushes denormal operands to zero, so the
    mantissa is rebuilt as an integer-scaled value that never reaches the
    multiplier in denormal form.
    """
    bits = x.to(tl.int32, bitcast=True)
    # ``|x| < 2**-126`` is exactly the zero/denormal encoding (both scale to
    # zero, and zero keeps its sign through ``_signed_result``).  The mantissa
    # integer is exact, so ``mant * 2**(e - 149)`` reproduces the denormal
    # value without ever feeding a denormal to the multiplier.
    denormal = (bits & 0x7FFFFFFF) < 0x00800000
    mant = (bits & 0x7FFFFF).to(tl.float32)
    x = tl.where(denormal, mant, x)
    e = tl.where(denormal, e - 149, e)
    s1, s2 = _scale_parts(e)
    return _signed_result((x * s1) * s2, bits)


@triton.jit
def _ldexp_i32_fast(x, e):
    """``x * 2**e`` for a float32 ``x`` whose dtype already normalises denormals."""
    bits = x.to(tl.int32, bitcast=True)
    s1, s2 = _scale_parts(e)
    return _signed_result((x * s1) * s2, bits)


@triton.jit
def _signed_result(r, bits):
    # A product that is flushed to zero loses its sign on XPU3; torch keeps the
    # sign of the input.
    return (r.to(tl.int32, bitcast=True) | (bits & -2147483648)).to(
        tl.float32, bitcast=True
    )


@triton.jit
def _exponent_narrow(y):
    """Clamp a <=32-bit integral exponent into what the two-factor split covers."""
    e = y.to(tl.int32)
    return tl.maximum(tl.minimum(e, 378), -378)


@triton.jit
def _exponent_wide(y):
    """Clamp a 64-bit integral exponent.

    Narrowing goes through float32: XPU3 cannot lower an i64
    ``arith.minsi``/``trunci`` pair here, and a float32 clamp is exact for
    ``|y| <= 2**24`` and saturates to +-378 beyond it, which is all the split
    needs.  (A ``tl.constexpr`` switch inside a *helper* also breaks the
    pointwise codegen on this backend, hence the two functions.)
    """
    f = tl.maximum(tl.minimum(y.to(tl.float32), 378.0), -378.0)
    return f.to(tl.int32)


# ---------------------------------------------------------------------------
# fast flat kernel (same shape, both contiguous)
# ---------------------------------------------------------------------------
@triton.jit
def _ldexp_flat_kernel(
    x_ptr,
    y_ptr,
    o_ptr,
    n,
    BLOCK: tl.constexpr,
    MASKED: tl.constexpr,
    EXACT: tl.constexpr,
    WIDE_EXP: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if MASKED:
        # Strict ``<`` against the runtime bound: XPU3 can fold a non-strict
        # tail mask away and write the full tile out of bounds
        # (wiki pattern-xpu3-masked-store-constexpr-tail).
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
        y = tl.load(y_ptr + offs, mask=mask)
        e = _exponent_wide(y) if WIDE_EXP else _exponent_narrow(y)
        r = _ldexp_i32_exact(x, e) if EXACT else _ldexp_i32_fast(x, e)
        tl.store(o_ptr + offs, r.to(o_ptr.dtype.element_ty), mask=offs < n)
    else:
        x = tl.load(x_ptr + offs).to(tl.float32)
        y = tl.load(y_ptr + offs)
        e = _exponent_wide(y) if WIDE_EXP else _exponent_narrow(y)
        r = _ldexp_i32_exact(x, e) if EXACT else _ldexp_i32_fast(x, e)
        tl.store(o_ptr + offs, r.to(o_ptr.dtype.element_ty))


_FLAT_BLOCK = 32768
_FLAT_TINY = 4096
_FLAT_OPTS = dict(num_warps=4, buffer_size_limit=4096, unroll_num=4)
_PLAN_CACHE = {}


def _pick_geometry(n):
    if n <= _FLAT_TINY:
        blk = max(256, triton.next_power_of_2(max(n, 1)))
    else:
        blk = min(_FLAT_BLOCK, triton.next_power_of_2(max(n, 1)))
    return blk, max(1, (n + blk - 1) // blk), (n % blk) != 0


def _run_flat(self, other, result_dtype):
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
    out = torch.empty(self.shape, dtype=result_dtype, device=self.device)
    _ldexp_flat_kernel[(grid,)](
        self,
        other,
        out,
        n,
        BLOCK=blk,
        MASKED=masked,
        EXACT=exact,
        WIDE_EXP=wide,
        **_FLAT_OPTS,
    )
    return out


def _can_use_flat(self, other):
    return (
        self.is_contiguous()
        and other.is_contiguous()
        and self.shape == other.shape
        and self.numel() > 0
        and self.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and not other.dtype.is_floating_point
        and not other.dtype.is_complex
    )


# ---------------------------------------------------------------------------
# general (broadcast / non-contiguous / mixed dtype) pointwise kernels
# ---------------------------------------------------------------------------
@pointwise_dynamic(
    is_tensor=[True, True],
    promotion_methods=[(0, 1, "INT_TO_FLOAT")],
    config=_CONFIG,
)
@triton.jit
def _ldexp_int_func(x, y):
    # Always exact: the general path carries mixed dtypes, broadcast and
    # non-contiguous inputs where the extra ~10 ops are irrelevant.
    return _ldexp_i32_exact(x.to(tl.float32), _exponent_wide(y))


@triton.jit
def _split_float_exponent(y):
    """``2**y`` for a float32 ``y`` as ``(s1, s2, frac_factor)``.

    ``r = x * s1 * s2 * frac_factor`` never forms ``2**y`` explicitly, so the
    split is exact for the integer part (the two factors stay normal) and the
    residual ``y - nf`` (which carries everything the integer clamp dropped,
    i.e. ``|y| > 252``) is folded into the ``tl.exp`` factor.
    """
    nf = tl.maximum(tl.minimum(tl.floor(y), 252.0), -252.0)
    frac = y - nf
    s1, s2 = _scale_parts(nf.to(tl.int32))
    return s1, s2, tl.exp(frac * _LN2)


@pointwise_dynamic(
    is_tensor=[True, True],
    promotion_methods=[(0, 1, "INT_TO_FLOAT")],
    config=_CONFIG,
)
@triton.jit
def _ldexp_float_func(x, y):
    xf = x.to(tl.float32)
    yf = y.to(tl.float32)
    bits = xf.to(tl.int32, bitcast=True)
    s1, s2, fexp = _split_float_exponent(yf)
    r = ((xf * s1) * s2) * fexp
    # Non-finite exponents: aten computes ``x * 2**y``, i.e. ``x * inf`` for
    # ``y == +inf`` (so ``0 * inf`` is NaN) and ``x * 0`` for ``y == -inf``.
    finite = tl.abs(yf) != _INF
    inf_scale = tl.where(yf > 0, _INF, 0.0)
    r = tl.where(finite, r, xf * inf_scale)
    # ``0 * 2**y`` is a signed zero for every finite ``y``, but the residual
    # factor above can be inf/0 for huge ``|y|`` and ``0 * inf`` is NaN.
    r = tl.where((xf == 0.0) & finite, xf, r)
    # NaN exponent (applied last so it wins over the zero guard).
    r = tl.where(yf != yf, yf, r)
    return (r.to(tl.int32, bitcast=True) | (bits & -2147483648)).to(
        tl.float32, bitcast=True
    )


_COMPLEX_PROMOTIONS = [
    (0, 1, 2, 3, "INT_TO_FLOAT"),
    (0, 1, 2, 3, "INT_TO_FLOAT"),
]


@pointwise_dynamic(
    is_tensor=[True, True, True, True],
    num_outputs=2,
    promotion_methods=_COMPLEX_PROMOTIONS,
    config=_CONFIG,
)
@triton.jit
def _ldexp_complex_func(xr, xi, yr, yi):
    xr = xr.to(tl.float32)
    xi = xi.to(tl.float32)
    yr = yr.to(tl.float32)
    yi = yi.to(tl.float32)
    # 2**(yr + 1j*yi) = 2**yr * (cos(yi*ln2) + 1j*sin(yi*ln2)); the scale uses
    # the same exact two-factor split as the real path (``tl.exp2`` is wrong
    # on this backend).
    s1, s2, fexp = _split_float_exponent(yr)
    angle = yi * _LN2
    cos_angle = tl.cos(angle)
    sin_angle = tl.sin(angle)
    pr = ((s1 * cos_angle) * s2) * fexp
    pi = tl.where(yi == 0, 0.0, ((s1 * sin_angle) * s2) * fexp)
    return xr * pr - xi * pi, xr * pi + xi * pr


def _complex_ldexp(self, other, result_dtype):
    real_dtype = torch.float64 if result_dtype == torch.complex128 else torch.float32
    xr, xi = _generic._complex_parts(self, real_dtype)
    yr, yi = _generic._complex_parts(other, real_dtype)
    real, imag = _ldexp_complex_func(xr, xi, yr, yi)
    result = torch.empty(real.shape, dtype=result_dtype, device=real.device)
    parts = torch.view_as_real(result)
    _gems_copy_(parts[..., 0], real)
    _gems_copy_(parts[..., 1], imag)
    return result


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def ldexp(self, other):
    logger.debug("GEMS_KUNLUNXIN LDEXP")
    result_dtype = _generic._result_dtype(self, other)
    device = _generic._result_device(self, other)
    # CPU zero-dimensional operands are host scalars; materialise them on the
    # device (allocation + fill only).
    if self.device != device:
        self = torch.tensor(self.item(), dtype=self.dtype, device=device)
    if other.device != device:
        other = torch.tensor(other.item(), dtype=other.dtype, device=device)
    if self.is_complex() or other.is_complex():
        return _complex_ldexp(self, other, result_dtype)
    if self.numel() == 0:
        shape = torch.broadcast_shapes(self.shape, other.shape)
        return torch.empty(shape, dtype=result_dtype, device=device)
    integral_exponent = (
        self.is_floating_point()
        and not other.is_floating_point()
        and not other.is_complex()
    )
    if integral_exponent:
        if _can_use_flat(self, other):
            return _run_flat(self, other, result_dtype)
        return _ldexp_int_func(self, other)
    return _ldexp_float_func(self, other)
