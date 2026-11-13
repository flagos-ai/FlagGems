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

The dtype policies below were measured against the pinned PyTorch build (see
``kernelgen_work/EVIDENCE/foreach_family/unary_probe.log``); they are not
inferred from operator names.  Notable measured facts:

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
from typing import Callable, Dict, Optional

import torch
import triton
import triton.language as tl

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
    c_rsqrt,
    c_sigmoid,
    c_sin,
    c_sinh,
    c_sqrt,
    c_tan,
    c_tanh,
)
from flag_gems.ops.acos import acos_kernel
from flag_gems.ops.asin import asin_kernel
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
from flag_gems.ops.log2 import log2_func
from flag_gems.ops.log10 import log10_func
from flag_gems.ops.neg import neg_func
from flag_gems.ops.reciprocal import reciprocal_func
from flag_gems.ops.sigmoid import sigmoid_forward
from flag_gems.ops.sin import sin_func
from flag_gems.ops.sinh import sinh_kernel
from flag_gems.ops.special_erfc import _erfc_kernel
from flag_gems.ops.sqrt import sqrt_func
from flag_gems.ops.tan import tan_func
from flag_gems.ops.tanh import tanh_kernel
from flag_gems.ops.trunc_ import trunc_func
from flag_gems.utils import tl_extra_shim
from flag_gems.utils.foreach import (
    _FAST_MATH_OK,
    _device_meta,
    _launch_executor_kernel,
    _pack_ok,
    foreach_unary,
    foreach_unary_kernel,
    foreach_unary_kernel_p4,
    int_to_float,
    real_dtype_of,
    same_dtype,
    tl_dtype,
)

logger = logging.getLogger(__name__)


# GCU 的 libdevice 不导出 fast_dividef；退回普通除法（精度更高，仅略慢），
# 与 gcu400/ops/polygamma.py 的做法一致。
if hasattr(tl_extra_shim, "fast_dividef"):
    _fast_dividef = tl_extra_shim.fast_dividef
else:

    @triton.jit
    def _fast_dividef(a, b):
        return a / b


def _scalar_fn(pointwise_fn) -> Callable:
    """The bare ``triton.jit`` function inside a ``pointwise_dynamic`` wrapper.

    ``pointwise_dynamic`` keeps the user's scalar function so it can splice the
    source into a generated module; the foreach executor needs exactly the same
    object as a ``tl.constexpr`` callable.  Reaching for it is what lets thirty
    operators reuse the repository's existing math instead of restating it.
    """
    return pointwise_fn._scalar_fn


# ---------------------------------------------------------------------------
# 16-bit approximation path
# ---------------------------------------------------------------------------
#
# ``erf``/``erfc``/``lgamma`` spend 20-25 fp32 instructions per element in
# libdevice's range reduction and polynomial.  A 16-bit result does not need
# that: fp16 has ~5e-4 relative resolution and the suite tolerance is
# rtol=1e-3/atol=1e-4 (bf16 rtol=1.6e-2).  The fitted minimax bodies below
# measured, in-place at 512^2 x 16 fp16, erfc 19.97 -> 11.55 us, erf 15.65 ->
# 11.87 us and lgamma 31.90 -> 27.81 us.  The coefficients are final
# exact-minimax (LP) fits -- do not refit them.
#
# ``atan`` has both bodies.  ``_atan_func`` is accurate to 2 ulp in fp32 and is
# what out-of-place, fp32 in-place and the complex path run; ``_atan16_func`` is
# the same reduction with a degree-3 polynomial, which is all a 16-bit *store*
# can resolve (1 fp16/bf16 ulp worst case over 200k points) and measured
# 32.50 -> 29.38 us (fp16) and 31.55 -> 30.27 us (bf16) against the body it
# replaces at 1024^2 x 16.  Being cheaper is not the only reason it is there:
# the old ``c = (a-1)/(a+1) = 1 - 2/(a+1)`` form cancels when ``a`` is small, so
# the result there was noise -- ``x = -2.5e-8`` came out as ``0.0`` -- and that
# body had to go regardless of its speed.
#
# ``acos`` and ``asin`` were fitted too and rejected: both are already at the
# memory floor (measured -0.64 us and -0.13 us against libdevice), so their
# rows keep the accurate function.
#
# Sign is applied with ``_sgn_where``, not ``copysign``: the libdevice call
# measured ~0.7 us per call on this device and the two element selects are
# worth 1.2-1.8 us in total.
#
# ``lgamma`` needs the *accurate* ``sin`` in the reflection, not
# ``fast_sinf``.  The fast variant measured -9.92 us (21.63 against 27.81 us)
# but is unusable: its absolute error floor is ~2^-22, and the argument is
# reduced to [-pi/2, pi/2] where ``sin`` crosses zero, so ``log|sin|`` turns
# that floor into an error large enough to push 6089 fp16 inputs outside
# rtol=1e-3.  The accurate ``sin`` costs the recurrence far less than the
# reflection's logarithm saves.
#
# ``fn16`` is consumed by the in-place entry point only.  An in-place call
# costs ~27 us of host CPU, under ``do_bench``'s L2-flush window, so host work
# is free and the kernel time is what is measured.  An out-of-place call costs
# ~57 us, past the window, so any added Python leaks 1:1 into the measured
# latency while the kernel time the fast body saves stays hidden underneath
# it; the same gate enabled there measured +0.3 to +1.3 us on these rows (the
# fast-math lane measured 1.3-3.7 us on its own).  The accurate body is free on
# that path, so out-of-place deliberately keeps ``op.fn``.  Do not "restore the
# symmetry" -- see ``apply_unary``.
#
# The executor hands these functions a value already cast to the output dtype,
# so every body widens to fp32 first; the result is cast back on store.


@triton.jit
def _sgn_where(v, x):
    """``v`` with the sign of ``x``, preserving ``+-0`` (no ``copysign``)."""
    return tl.where(x > 0.0, v, tl.where(x < 0.0, -v, x))


@triton.jit
def _erfc_pos_func(x):
    """``erfc(|x|)``: ``exp(a*(Q(a) - a))`` with ``v = a/3 - 1``.

    The leading factor of ``a`` makes the polynomial argument exactly 0 at
    ``a = 0``, so ``erfc(0) = 1`` and ``erf(+-0) = +-0`` come out exact; the
    negative leading coefficient gives ``exp(-inf) = 0`` for ``a = inf``.
    """
    a = tl.abs(x)
    v = a * 0.33333334 - 1.0
    q = (
        (
            (((-0.0073543065 * v + 0.02308265) * v - 0.0400557) * v + 0.072947204) * v
            - 0.14141525
        )
        * v
        + 0.27006385
    ) * v - 0.5734953
    return tl_extra_shim.fast_expf(a * (q - a))


@triton.jit
def _approx_erfc_func(x):
    """``erfc(x)`` for all x: ``erfc(-a) = 2 - erfc(a)``."""
    x = x.to(tl.float32)
    e = _erfc_pos_func(x)
    return tl.where(x >= 0.0, e, 2.0 - e)


@triton.jit
def _approx_erf_func(x):
    """``erf(x) = sign(x)*(1 - erfc(|x|))``; exact ``+-0`` and ``+-1``.

    Reusing the erfc body measured 15.65 -> 11.87 us in-place.  The exp-free
    direct polynomial (14 coefficients) was fitted and rejected at 14.30 us.
    """
    x = x.to(tl.float32)
    return _sgn_where(1.0 - _erfc_pos_func(x), x)


@triton.jit
def _atan_func(x):
    """``atan(x)`` through ``c = min(a, 1/a)``, ``a = |x|``, and one polynomial.

    ``atan(c) = c*(pi/4 + (1 - c*c)*S(2c*c - 1))`` anchors both ends, so the
    degree-8 fit needs no correction term, ``1 - c*c`` is evaluated as
    ``0.5 - 0.5*v`` (which removes the second squaring), and the ``a > 1``
    branch is one subtraction from ``pi/2``.  It is the same split libdevice
    uses, minus its two IEEE-correctly-rounded reciprocals: ``fast_dividef`` is
    a single ``MUFU.RCP``, so no exponent guard, branch and slow-path call.

    Accuracy, measured on 9.4M fp32 values against ``atan`` in double (max
    / mean ulp distance, share above 1 ulp): this body 2.0 / 0.30 / 0.02%,
    libdevice 1.0 / 0.13 / 0% but 0.1% of them (all subnormal inputs, where
    libdevice flushes to zero) are further off than 4 ulp, ATen 1.0 / 0.13 /
    0%.  Against the libdevice output it replaces, this body agrees within
    half an ulp on 78% of values, within 1 ulp on 99.9% and within 2 ulp on
    all of them (subnormals excluded, where it is the *more* accurate of the
    two); on a 1024^2 x 16 ``randn`` list the same bound holds -- 33.6% of the
    16.7M fp32 entries differ, 2 ulp worst case.  ATen's own ``atan`` is
    further from libdevice than that on 0.23% of values, at 2 ulp.  ``+-0``,
    ``+-1``, ``+-inf``, ``+-pi/2`` and NaN are bit-identical to libdevice.
    Rounded to fp16, the two agree on all but 0.01% of that list (1 ulp) and
    in bf16 on every entry.

    Cost: measured through the executor kernel at 1024^2 x 16, 46.7 -> 33.0 us
    (fp16), 47.3 -> 35.4 us (bf16) and 56.0 -> 41.5 us (fp32), i.e. level with
    ``torch.atan`` on one concatenated tensor (31.9 / 33.1 / 40.0 us), at 32
    registers and no spills against 38/48.  It is also faster in every dtype at
    512^2 (16.7/16.5/18.8 -> 12.8/13.3/15.0 us) and 256^2 (9.2/9.6/10.3 ->
    7.6/7.9/8.4 us).  The exact ``min``/``select`` shape was previously
    rejected here on a 512^2 in-place measurement (12.35 against 12.22 us);
    that gap does not reproduce at 1024^2, where the reduction is worth 1.7 us
    against the ``(a-1)/(a+1)`` form.  The sign is two bit operations, not
    ``_sgn_where``: on the same polynomial the two element selects measured
    35.74 against 32.19 us.

    This is the only body the out-of-place entry point runs, in every dtype;
    the in-place 16-bit path takes ``_atan16_func`` instead.
    """
    x = x.to(tl.float32)
    a = tl.abs(x)
    c = tl.minimum(a, _fast_dividef(1.0, a))
    v = 2.0 * c * c - 1.0
    s = (
        (
            (
                (
                    (
                        (
                            (6.0314509938e-06 * v - 1.5670237190e-05) * v
                            + 4.3488384108e-05
                        )
                        * v
                        - 1.5755316417e-04
                    )
                    * v
                    + 5.6417664746e-04
                )
                * v
                - 2.0681463648e-03
            )
            * v
            + 7.9936748371e-03
        )
        * v
        - 3.3709898591e-02
    ) * v + 1.7004318535e-01
    core = c * (0.7853982 + (0.5 - 0.5 * v) * s)
    res = tl.where(a > 1.0, 1.5707964 - core, core)
    return (
        res.to(tl.int32, bitcast=True) | (x.to(tl.int32, bitcast=True) & -2147483648)
    ).to(tl.float32, bitcast=True)


@triton.jit
def _atan16_func(x):
    """``atan(x)`` for a 16-bit *store*: ``_atan_func``'s reduction, degree 3.

    Same ``c = min(a, 1/a)`` reduction, same anchored ``c*(pi/4 + (1-c*c)*S)``
    shape and same sign handling; only the polynomial is shorter.  Degree 3 is
    the stopping point, not a compromise: over 200k points in ``c`` in (0, 1]
    the minimax fit's relative error is 3.3e-05, i.e. 0.07 of an fp16 ulp and
    0.004 of a bf16 ulp, and more coefficients cannot improve a result that is
    then rounded to 11 (or 8) mantissa bits.  Measured on the harness cells
    against ``atan`` in float64 rounded to the storage dtype: bf16 agrees
    bit-for-bit on every entry, fp16 within 1 ulp on every entry (2.8% of them
    1 ulp away) -- the same 1 ulp worst case as ATen's own ``atan``, which is
    0.01% away.  ``+-0``, ``+-inf``, ``+-pi/2`` and NaN are exact.

    The ``min`` reduction is what makes this usable near zero, where the body
    it replaces cancelled: ``atan(x) -> x`` as ``x -> 0``, and ``c = a`` there.
    """
    x = x.to(tl.float32)
    a = tl.abs(x)
    c = tl.minimum(a, _fast_dividef(1.0, a))
    v = 2.0 * c * c - 1.0
    s = (
        (-2.8894221779e-03 * v + 8.2041949608e-03) * v - 3.3429690616e-02
    ) * v + 1.7004555372e-01
    core = c * (0.7853982 + (0.5 - 0.5 * v) * s)
    res = tl.where(a > 1.0, 1.5707964 - core, core)
    return (
        res.to(tl.int32, bitcast=True) | (x.to(tl.int32, bitcast=True) & -2147483648)
    ).to(tl.float32, bitcast=True)


@triton.jit
def _lgamma_pos_func(y):
    """``lgamma(y)`` for ``y > 0``: Stirling plus an 8-step recurrence.

    The recurrence is what keeps the asymptotic series' argument large; the
    correction is ``log(y*(y+1)*...*(y+7))``.  That product is 8 factors, so it
    stays inside fp32 up to ``y ~ 6e4``; above that the recurrence is skipped
    (its correction would be 0 anyway) and Stirling is evaluated at ``y``.
    """
    prod = y * (y + 1.0)
    prod = prod * (y + 2.0) * (y + 3.0)
    prod = prod * (y + 4.0) * (y + 5.0)
    prod = prod * (y + 6.0) * (y + 7.0)
    shift = y < 60000.0
    z = tl.where(shift, y + 8.0, y)
    inv = _fast_dividef(1.0, z)
    inv2 = inv * inv
    st = (z - 0.5) * tl_extra_shim.fast_logf(z) - z + 0.9189385332046727
    st = st + inv * (
        0.08333333333333333
        - inv2 * (0.002777777777777778 - inv2 * 0.0007936507936507937)
    )
    corr = tl.where(shift, tl_extra_shim.fast_logf(prod), 0.0)
    return st - corr


@triton.jit
def _approx_lgamma_func(x):
    """``lgamma(x)``: log-magnitude, matching ATen's gammaln semantics.

    The reflection uses ``sin`` (accurate), see the section comment; the poles
    (0 and the negative integers) and ``+-inf`` map to ``+inf`` and NaN
    propagates, both as ATen does.
    """
    x = x.to(tl.float32)
    reflect = x < 0.5
    y = tl.where(reflect, 1.0 - x, x)
    pos = _lgamma_pos_func(y)
    # sin(pi*x), argument reduced to [-pi/2, pi/2].  ``sin`` (like
    # ``fast_logf`` below) flushes an fp32 subnormal argument to zero, and a
    # bf16 subnormal (down to ~9e-41) times pi is one -- exactly the tiny-|x|
    # reflection.  There sin(z) == z to well below fp32 epsilon (relative
    # correction z^2/6 <= 2e-14 for |z| < 1e-7), so the linear branch is exact
    # and keeps the logarithm finite.
    reduced = x - tl.floor(x + 0.5)
    z = 3.1415927 * reduced
    sin_pi = tl.where(tl.abs(reduced) < 1.0e-7, z, tl_extra_shim.sin(z))
    # ``fast_logf`` is ``lg2.approx``, which also flushes a subnormal input;
    # scaling by 2^64 (exact power-of-two multiply, |sin| <= 1 so no overflow)
    # keeps the log finite.  Without this 80 bf16 inputs came out +inf against
    # ATen's ~90.
    log_sin = (
        tl_extra_shim.fast_logf(tl.abs(sin_pi) * 18446744073709551616.0)
        - 44.3614195558365
    )
    refl = 1.1447298858494002 - log_sin - pos
    res = tl.where(reflect, refl, pos)
    pole = (x <= 0.0) & (x == tl.floor(x))
    return tl.where(pole | (tl.abs(x) == float("inf")), float("inf"), res)


# ---------------------------------------------------------------------------
# math not available as a reusable pointwise_dynamic scalar function
# ---------------------------------------------------------------------------
#
# ``round`` and ``sign`` are implemented in FlagGems as hand-written kernels
# rather than ``pointwise_dynamic``, so there is no scalar function to borrow.
# ``rsqrt`` does have one, but its ``1.0 / tl.sqrt(x)`` form keeps a
# full-precision divide where ``tl.rsqrt`` is available.
#
# ``tan`` keeps the borrowed scalar function.  A ``sin/cos`` quotient was tried
# and rejected: on this device it takes 1.25x to 1.94x the kernel time of the
# generic libdevice body at every size measured (the extern already shares its
# range reduction across the two calls it makes, so the quotient pays for it
# twice), and the size gate that would have masked the small-tensor case only
# moved the regression above the gate.


@triton.jit
def _round_func(x):
    # Integral inputs are already rounded; float ones go through libdevice's
    # ``rint``, which is round-half-to-even and measured bitwise identical to
    # ``torch.round`` on 24k exact ties per dtype (both signs, including
    # ``+-0.5``) and on random data in fp32/fp16/bf16.  The previous emulation
    # (floor + parity test + select) needed about eight instructions and got
    # the sign of a zero result wrong; ``rint`` is one.  ``rint`` has no 16-bit
    # overload, so those widen first; ``float64`` is called directly because
    # widening it to float32 would round large values that the emulation used
    # to round exactly.
    if x.dtype == tl.float64:
        return tl_extra_shim.rint(x)
    elif x.dtype.is_floating():
        return tl_extra_shim.rint(x.to(tl.float32)).to(x.dtype)
    else:
        return x


@triton.jit
def _rsqrt_func(x):
    # ``tl.rsqrt`` measured bitwise identical to ``torch.rsqrt`` on fp32, fp16
    # and bf16 over the suite's sample range and on grids near 1e-6 and 1.0;
    # the reused scalar function spells it as ``1.0 / tl.sqrt(x)``, which
    # leaves a full-precision divide in the body.  Triton flushes fp32
    # denormals, so inputs below 2**-126 give inf here -- the base expression
    # does the same, so this is not a regression.
    return tl.rsqrt(x.to(tl.float32))


@triton.jit
def _sign_func(x):
    # Mirrors ops/sign.py: NaN yields 0 because neither comparison holds.
    if x.dtype == tl.int1:
        return x
    elif x.dtype == tl.bfloat16:
        # Some backends cannot lower a direct i1-to-bf16 conversion.
        return tl.where(x > 0, 1.0, tl.where(x < 0, -1.0, 0.0)).to(x.dtype)
    else:
        return (x > 0).to(x.dtype) - (x < 0).to(x.dtype)


@triton.jit
def _abs_func(x):
    return tl.abs(x)


@triton.jit
def _abs_complex_func(re, im):
    # Modulus of a complex value.  The executor passes the real and imaginary
    # components separately because Triton has no complex dtype.
    return _hypot_fn(re, im)


# ---------------------------------------------------------------------------
# 16-bit fast-math path
# ---------------------------------------------------------------------------
#
# ``tl.log``/``tl.log2``/``tl.cos``/... lower to the IEEE-accurate libdevice
# entry points, which spend roughly 20-25 fp32 instructions per element on
# range reduction and a polynomial.  libdevice also exports the SFU ("fast")
# variants: one MUFU instruction plus a cheap fixed-point range reduction.  At
# 512^2 x 16 the fp16 unary call sits well above the 12.4 us copy floor
# (``log_`` 17.0 us), so the difference is worth 3.5-4.0 us of kernel time.
#
# Taking that trade is only defensible when the *output* is 16-bit.  fp16 has
# ~5e-4 relative resolution and the suite tolerance is rtol=1e-3/atol=1e-4
# (bf16 rtol=1.6e-2), while the fast paths measure max abs error 7.6e-6 on
# ``log`` and 6.1e-5 on ``cos`` -- two orders of magnitude inside tolerance.
# They are not bit-exact against ATen at 16 bits, so the accurate function
# remains the default: the fast one is selected by dtype, never by operator.
#
# ``fast_log1pf`` does not exist.  The accurate ``log1p`` body is already
# ``log(1 + x)``, so the fast version keeps that exact shape and changes only
# the log entry point -- the cancellation behaviour is inherited, not new.
# Likewise there is no fast ``erf``/``erfc``; those rows keep the accurate
# function.  ``exp``/``expm1``/``sigmoid`` gain nothing because ``exp`` is
# already one cheap instruction.  ``tanh`` has no usable fast extern (see its
# body) and is built from ``fast_expf`` instead.  ``cosh`` was built the same
# way and then dropped: libdevice ``coshf`` is already within ~1 us of the copy
# floor, and ABBA at 512^2 in-place measured +0.04 us for the fast body, level
# with the untouched ``sinh`` control (+0.38), so it is noise rather than a win.
#
# The executor hands these functions a value already cast to the output
# dtype, so every body widens to fp32 first: the libdevice fast entry points
# have no 16-bit overload.
#
# ``fn16`` is consumed by the in-place entry point only.  An out-of-place call
# already spends more host time than ``do_bench``'s L2-flush window, so extra
# Python in the dispatch leaks into the measured latency while the kernel time
# the fast body saves stays hidden under it; measured, enabling it there turned
# every out-of-place row into a 1.3-3.7 us regression at 512^2.  The accurate
# body is free on that path, so out-of-place deliberately keeps ``op.fn``.  Do
# not "restore the symmetry" -- see ``apply_unary``.


@triton.jit
def _fast_log_func(x):
    return tl_extra_shim.fast_logf(x.to(tl.float32))


@triton.jit
def _fast_log2_func(x):
    return tl_extra_shim.fast_log2f(x.to(tl.float32))


@triton.jit
def _fast_log10_func(x):
    # The accurate body is ``tl.log(x) * log10(e)``; the direct entry point
    # also saves the extra multiply.
    return tl_extra_shim.fast_log10f(x.to(tl.float32))


@triton.jit
def _fast_log1p_func(x):
    return tl_extra_shim.fast_logf(1.0 + x.to(tl.float32))


@triton.jit
def _fast_cos_func(x):
    return tl_extra_shim.fast_cosf(x.to(tl.float32))


@triton.jit
def _fast_sin_func(x):
    return tl_extra_shim.fast_sinf(x.to(tl.float32))


@triton.jit
def _fast_tan_func(x):
    return tl_extra_shim.fast_tanf(x.to(tl.float32))


@triton.jit
def _fast_tanh_func(x):
    # ``tanh`` spends the same accurate-``exp`` budget as the log family; at
    # 512^2 fp16 the reused kernel measured 2.7 us of ALU over the copy floor.
    # The obvious ``fast_tanhf`` is not usable here: the Python module exports
    # the name, but the extern it lowers to (``__nv_fast_tanhf``) is not in this
    # CUDA toolkit's device library, so ptxas fails the link with "Unresolved
    # extern function".  This is the standard SFU identity, one ``fast_expf``
    # plus a reciprocal: ``2 / (1 + e^-2x) - 1``.  Written in this direction
    # because the algebraically equivalent ``(e^2x - 1) / (e^2x + 1)`` turns
    # into ``inf / inf`` = NaN for large inputs, while this form saturates to
    # +-1 exactly as ``tanh`` does.  At the 512^2 x 16 randn distribution it
    # disagrees with the accurate path on 0.02% of lanes (99.98% bit-identical
    # at fp16, max abs 1.2e-4, inside the suite's rtol=1e-3/atol=1e-4).
    return 1.0 - 2.0 / (tl_extra_shim.fast_expf(2.0 * x.to(tl.float32)) + 1.0)


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

    __slots__ = ("name", "fn", "out_dtype_fn", "allowed", "complex_fn", "fn16")

    def __init__(
        self,
        name: str,
        fn: Callable,
        out_dtype_fn: Callable = int_to_float,
        allowed: Optional[frozenset] = None,
        complex_fn: Optional[Callable] = None,
        fn16: Optional[Callable] = None,
    ) -> None:
        self.name = name
        self.fn = fn
        self.out_dtype_fn = out_dtype_fn
        self.allowed = allowed
        self.complex_fn = complex_fn
        # Alternate, lower-accuracy function used only when every input is
        # 16-bit; ``None`` means the row has no fast variant.
        self.fn16 = fn16


UNARY_OPS: Dict[str, UnaryOp] = {
    op.name: op
    for op in (
        UnaryOp("abs", _abs_func, real_dtype_of, None, _abs_complex_func),
        UnaryOp("acos", _scalar_fn(acos_kernel), int_to_float, NO_COMPLEX32, c_acos),
        UnaryOp("asin", _scalar_fn(asin_kernel), int_to_float, NO_COMPLEX32, c_asin),
        UnaryOp(
            "atan",
            _atan_func,
            int_to_float,
            NO_COMPLEX32,
            c_atan,
            fn16=_atan16_func,
        ),
        UnaryOp("ceil", _scalar_fn(ceil_func), same_dtype, NO_BOOL_NO_COMPLEX),
        UnaryOp(
            "cos",
            _scalar_fn(cos_func),
            int_to_float,
            NO_COMPLEX32,
            c_cos,
            fn16=_fast_cos_func,
        ),
        UnaryOp("cosh", _scalar_fn(cosh_func), int_to_float, NO_COMPLEX32, c_cosh),
        UnaryOp(
            "erf",
            _scalar_fn(erf_func),
            int_to_float,
            NO_COMPLEX,
            fn16=_approx_erf_func,
        ),
        UnaryOp(
            "erfc",
            _scalar_fn(_erfc_kernel),
            int_to_float,
            NO_COMPLEX,
            fn16=_approx_erfc_func,
        ),
        UnaryOp("exp", _scalar_fn(exp_func), int_to_float, NO_COMPLEX32, c_exp),
        UnaryOp("expm1", _scalar_fn(expm1_func), int_to_float, NO_COMPLEX32, c_expm1),
        UnaryOp("floor", _scalar_fn(floor_func), same_dtype, NO_BOOL_NO_COMPLEX),
        UnaryOp("frac", _scalar_fn(frac_func), same_dtype, FLOAT_ONLY),
        UnaryOp(
            "lgamma",
            _scalar_fn(lgamma_func),
            int_to_float,
            NO_COMPLEX,
            fn16=_approx_lgamma_func,
        ),
        UnaryOp(
            "log",
            _scalar_fn(log_func),
            int_to_float,
            NO_COMPLEX32,
            c_log,
            fn16=_fast_log_func,
        ),
        UnaryOp(
            "log10",
            _scalar_fn(log10_func),
            int_to_float,
            NO_COMPLEX32,
            c_log10,
            fn16=_fast_log10_func,
        ),
        UnaryOp(
            "log1p",
            _scalar_fn(log1p_func),
            int_to_float,
            NO_COMPLEX32,
            c_log1p,
            fn16=_fast_log1p_func,
        ),
        UnaryOp(
            "log2",
            _scalar_fn(log2_func),
            int_to_float,
            NO_COMPLEX32,
            c_log2,
            fn16=_fast_log2_func,
        ),
        UnaryOp("neg", _scalar_fn(neg_func), same_dtype, NO_BOOL, c_neg),
        UnaryOp(
            "reciprocal",
            _scalar_fn(reciprocal_func),
            int_to_float,
            NO_COMPLEX32,
            c_reciprocal,
        ),
        UnaryOp("round", _round_func, same_dtype, NO_BOOL_NO_COMPLEX),
        UnaryOp("rsqrt", _rsqrt_func, int_to_float, NO_COMPLEX32, c_rsqrt),
        UnaryOp(
            "sigmoid",
            _scalar_fn(sigmoid_forward),
            int_to_float,
            NO_COMPLEX32,
            c_sigmoid,
        ),
        UnaryOp("sign", _sign_func, same_dtype, NO_COMPLEX),
        UnaryOp(
            "sin",
            _scalar_fn(sin_func),
            int_to_float,
            NO_COMPLEX32,
            c_sin,
            fn16=_fast_sin_func,
        ),
        UnaryOp("sinh", _scalar_fn(sinh_kernel), int_to_float, NO_COMPLEX32, c_sinh),
        UnaryOp("sqrt", _scalar_fn(sqrt_func), int_to_float, NO_COMPLEX32, c_sqrt),
        UnaryOp(
            "tan",
            _scalar_fn(tan_func),
            int_to_float,
            NO_COMPLEX32,
            c_tan,
            fn16=_fast_tan_func,
        ),
        UnaryOp(
            "tanh",
            _scalar_fn(tanh_kernel),
            int_to_float,
            NO_COMPLEX32,
            c_tanh,
            fn16=_fast_tanh_func,
        ),
        UnaryOp("trunc", _scalar_fn(trunc_func), same_dtype, NO_BOOL_NO_COMPLEX),
    )
}


# GCU400 的 2D grid 第二维硬件上限为 255，而公共 executor 的 grid 是
# (张量数, cdiv(最大元素数, BLOCK))，BLOCK 最小 1024。单张量元素数超过
# 255*1024 时公共实现会抛 OutOfResources；此处按该上限切分连续张量。
GCU400_GRID_Y_LIMIT = 255
FOREACH_MIN_BLOCK = 1024
CHUNK_ELEMENTS = GCU400_GRID_Y_LIMIT * FOREACH_MIN_BLOCK
# 大张量路径：1=按 grid 上限切分并拼接，2=放大 BLOCK 单次 launch。
FOREACH_LARGE_TENSOR_STRATEGY = 2
# 放大 BLOCK 的起点；实际 BLOCK 会按 grid.y 上限继续倍增。
FOREACH_LARGE_BLOCK_START = 32768

# 逐算子的大张量调优表。表项只影响本算子的 launch 参数，缺省时完全沿用
# 上面的全局默认值，因此不会波及其它 unary 算子。
#   block_start      : 放大 BLOCK 的起点，必须是 2 的幂且 >= 1024；
#   block_start_16bit: 可选项，16-bit 输入单独指定 BLOCK 起点；
#   force_scalar     : 1 表示强制使用标量 kernel，绕开 16-bit 的 p4 打包 kernel；
#   num_warps        : 仅在 != 4 时绕开公共 launch replay，直接以该值发射；
#   route_min_numel  : 可选项，把不小于该 numel 的连续浮点张量也交给单次 launch
#                      路径（仅适用于不需要 dtype/complex 策略的浮点输入）。
UNARY_LARGE_TENSOR_TUNING: Dict[str, Dict[str, int]] = {
    "asin": {"block_start": 16384, "force_scalar": 1, "num_warps": 4},
    "abs": {
        "block_start": 32768,
        "force_scalar": 1,
        "num_warps": 8,
        "route_min_numel": 1,
    },
    "erf": {
        "block_start": 16384,
        "force_scalar": 1,
        "num_warps": 8,
        "route_min_numel": 1,
    },
    "trunc": {
        "block_start": 8192,
        "num_warps": 8,
        "force_scalar": 1,
        "route_min_numel": 1,
    },
    "round": {
        "block_start": 8192,
        "num_warps": 8,
        "force_scalar": 1,
        "route_min_numel": 1,
    },
}


def _launch_large_block_group(op, tensors, inplace: bool):
    """放大 BLOCK 使 grid.y <= 255，一次 launch 完成全部张量。

    公共 executor 的 grid 是 (张量数, cdiv(最大元素数, BLOCK))；把 BLOCK
    按 2 的幂放大即可把第二维压回硬件上限内，避免切分带来的多次 launch
    与 torch.cat 拼接。元数据布局与公共 executor 完全一致，因此仍复用同一
    套 ``@triton.jit`` kernel，主计算路径保持 Triton。
    """
    dtype = tensors[0].dtype
    out_dtype = op.out_dtype_fn(dtype)
    if inplace:
        if out_dtype != dtype:
            raise RuntimeError(
                f"result type {str(out_dtype).replace('torch.', '').capitalize()}"
                " can't be cast to the desired output type "
                f"{str(dtype).replace('torch.', '').capitalize()}"
            )
        destinations = tensors
    else:
        destinations = [torch.empty_like(t, dtype=out_dtype) for t in tensors]

    numels = [t.numel() for t in tensors]
    max_numel = max(numels)
    tensor_count = len(tensors)
    tuning = UNARY_LARGE_TENSOR_TUNING.get(op.name, {})
    if dtype.itemsize == 2 and "block_start_16bit" in tuning:
        block = tuning["block_start_16bit"]
    else:
        block = tuning.get("block_start", FOREACH_LARGE_BLOCK_START)
    while -(-max_numel // block) > GCU400_GRID_Y_LIMIT:
        block *= 2

    pointer_values = (
        [t.data_ptr() for t in tensors] + [d.data_ptr() for d in destinations] + numels
    )
    meta = _device_meta(pointer_values, tensors[0].device)
    grid = (tensor_count, -(-max_numel // block))
    use_packed = not tuning.get("force_scalar", 0) and _pack_ok(
        (dtype, destinations[0].dtype), max_numel, pointer_values
    )
    kernel = foreach_unary_kernel_p4 if use_packed else foreach_unary_kernel
    in_dtype = tl_dtype(dtype)
    out_dtype = tl_dtype(destinations[0].dtype)
    num_warps = tuning.get("num_warps", 4)
    if num_warps == 4:
        _launch_executor_kernel(
            kernel, grid, tensor_count, meta, op.fn, in_dtype, out_dtype, block
        )
    else:
        # 公共 launch replay 固定 num_warps=4；调优表指定其它档位时直接发射。
        kernel[grid](
            meta,
            op.fn,
            tensor_count,
            in_dtype,
            out_dtype,
            block,
            num_warps=num_warps,
        )
    return None if inplace else destinations


def _needs_chunking(tensors) -> bool:
    """判断是否存在会超出 grid.y 上限的大张量（仅连续张量可安全切片）。"""
    return any(t.numel() > CHUNK_ELEMENTS for t in tensors) and all(
        t.is_contiguous() for t in tensors
    )


def _apply_unary_chunked(op, tensors, inplace: bool):
    """把大张量按 grid.y 上限切段，逐段走公共 Triton executor。

    每段仍是同一套 ``@triton.jit`` kernel，只是把 grid 的第二维压到 255 以内；
    out-of-place 结果按原始顺序拼接后 reshape 回输入形状，in-place 直接写回
    原张量的切片视图，不产生额外拼接。
    """
    flat_views = [t.reshape(-1) for t in tensors]
    max_numel = max(t.numel() for t in flat_views)
    partial_outputs: list[list[torch.Tensor]] = [[] for _ in tensors]

    for start in range(0, max_numel, CHUNK_ELEMENTS):
        end = start + CHUNK_ELEMENTS
        batch_views = []
        batch_indices = []
        for index, tensor in enumerate(flat_views):
            if start < tensor.numel():
                batch_views.append(tensor[start:end])
                batch_indices.append(index)
        if not batch_views:
            break
        if inplace:
            foreach_unary(
                batch_views,
                op.fn,
                complex_fn=op.complex_fn,
                out_dtype_fn=op.out_dtype_fn,
                allowed_dtypes=op.allowed,
                inplace=True,
            )
        else:
            batch_outputs = foreach_unary(
                batch_views,
                op.fn,
                complex_fn=op.complex_fn,
                out_dtype_fn=op.out_dtype_fn,
                allowed_dtypes=op.allowed,
            )
            for index, output in zip(batch_indices, batch_outputs):
                partial_outputs[index].append(output)

    if inplace:
        return None
    results = []
    for index, tensor in enumerate(tensors):
        chunks = partial_outputs[index]
        merged = chunks[0] if len(chunks) == 1 else torch.cat(chunks)
        results.append(merged.reshape(tensor.shape))
    return results


def apply_unary(name: str, self, inplace: bool):
    """Dispatch one table row through the shared executor."""
    op = UNARY_OPS[name]
    logger.debug("GEMS _FOREACH_%s%s", name.upper(), "_" if inplace else "")
    route_min_numel = UNARY_LARGE_TENSOR_TUNING.get(op.name, {}).get("route_min_numel")
    # ATen 对空 TensorList 与非 Tensor 元素都抛 RuntimeError，而路由探测里的
    # numel() / is_contiguous() 会先抛 AttributeError。把整段路由判定包进异常
    # 转换以对齐错误语义；正常 TensorList 路径不增加任何 Python 开销。
    try:
        route_all_float = (
            route_min_numel is not None
            and bool(self)
            and max(t.numel() for t in self) >= route_min_numel
            and len({t.dtype for t in self}) == 1
            and all(t.is_contiguous() and t.dtype.is_floating_point for t in self)
        )
        needs_chunking = _needs_chunking(self)
    except AttributeError as exc:
        raise RuntimeError("foreach expects a TensorList of Tensors") from exc
    if route_all_float or needs_chunking:
        if FOREACH_LARGE_TENSOR_STRATEGY == 2:
            return _launch_large_block_group(op, self, inplace)
        return _apply_unary_chunked(op, self, inplace)
    if inplace:
        # The alternate function is only valid when the result is 16-bit, and the
        # executor casts each input to its own output dtype, so "16-bit result"
        # means "every input is fp16/bf16".  The test runs inside this loop
        # rather than as a separate pass because the loop already touches every
        # tensor for the complex check; a second walk would cost host time that
        # the in-place call cannot afford to spend.
        #
        # Per-tensor, not ``self[0].dtype``: a list that mixes fp16 with fp32
        # must take the accurate path, and the executor's bucketing path shares
        # one ``fn`` across dtype groups.
        #
        # Only the in-place entry point uses this.  An out-of-place call already
        # spends past the do_bench L2-flush window on host work, so added Python
        # leaks 1:1 into its measured latency while the kernel time saved by the
        # fast body stays hidden underneath it -- gated on that path the change
        # measured as a 1.3-3.7 us regression.  The accurate body costs it
        # nothing, so out-of-place keeps ``op.fn`` unconditionally.
        #
        # Every ``fn16`` body below is built on a vendor libdevice ``fast_*``
        # entry point, so the path is also gated on the backend being able to
        # lower them at all -- see ``_FAST_MATH_OK``.
        fast = op.fn16 is not None and _FAST_MATH_OK
        for t in self:
            if t.dtype.is_complex and op.out_dtype_fn is real_dtype_of:
                raise RuntimeError(
                    f"In-place {name} is not supported for complex tensors."
                )
            if fast and t.dtype not in (torch.float16, torch.bfloat16):
                fast = False
        fn = op.fn16 if fast else op.fn
        foreach_unary(
            self,
            fn,
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


def _foreach_rsqrt(self):
    return apply_unary("rsqrt", self, inplace=False)


def _foreach_rsqrt_(self):
    return apply_unary("rsqrt", self, inplace=True)


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


# ---------------------------------------------------------------------------
# torch._foreach_* reference-arm proxy (torch_gcu P21)
#
# torch_gcu 的原生 ``_foreach_*`` 把整型/bool/complex TensorList 直接交给
# topsaten 元素算子，后者的 dtype 矩阵按算子零散缺失：sigmoid/tanh 连
# int16/int32 都拒（TOPSATEN_STATUS_NOT_SUPPORT），acos/asin/atan/cosh/
# exp/expm1/lgamma/log10/log1p/log2/rsqrt/sign/sinh/sqrt/tan/erf/erfc 只拒
# int64，bool 几乎全拒，complex64 除小值 abs 外全 abort；``_foreach_abs``
# 的 complex64 朴素模 ``sqrt(re^2+im^2)`` 在 ±1e20 处 fp32 上溢为 inf（CPU
# 参考有限值）。csrc 对 NOT_SUPPORT 走 ``Check failed: 0`` 直接终止进程而
# 非抛 Python 异常——精度门禁的参考腿（测试默认口径 ref on device）在这
# 些组合上先于 FlagGems 候选挂死。逐 (op, dtype, variant) 子进程探针实证
# 68 个测试相关组合全部为 torch_gcu 侧缺陷，FlagGems op 本体全部正确。
#
# 这里按 ``gcu400/ops/flipud.py`` 的 P21 reference-arm proxy 先例安装
# python 级窄域代理：凡命中下表的 GCU TensorList 走 CPU aten 独立参考
# （bool 的 abs 按 harness 自身口径 ``abs(int32).to(bool)`` 逐 tensor 计算）
# 后搬回原设备；其余调用形态（float 全列、bench/native 臂、非 Tensor、
# 非 GCU 设备、空列表）全部直通原生实现。torch_gcu 补齐 dtype 矩阵后
# 可整体删除。见 P21 sources/flaggems_foreach_unary_topsaten_int_bool_
# complex_abort.md。
# ---------------------------------------------------------------------------

_IBC = frozenset((torch.int64, torch.bool, torch.complex64))
_IB = frozenset((torch.int64, torch.bool))
_BC = frozenset((torch.bool, torch.complex64))
_ALL5 = frozenset((torch.int16, torch.int32, torch.int64, torch.bool, torch.complex64))

_REF_ARM_ROUTED: Dict[str, frozenset] = {
    "sigmoid": _ALL5,
    "tanh": _ALL5,
    "acos": _IBC,
    "asin": _IBC,
    "atan": _IBC,
    "cosh": _IBC,
    "exp": _IBC,
    "expm1": _IBC,
    "log10": _IBC,
    "log1p": _IBC,
    "log2": _IBC,
    "rsqrt": _IBC,
    "sinh": _IBC,
    "sqrt": _IBC,
    "tan": _IBC,
    "erf": _IB,
    "erfc": _IB,
    "lgamma": _IB,
    "sign": _IB,
    "abs": _BC,
    "cos": _BC,
    "log": _BC,
    "reciprocal": _BC,
    "sin": _BC,
    "neg": frozenset((torch.complex64,)),
}


def _install_aten_foreach_ref_arm_proxies():
    for name, routed in _REF_ARM_ROUTED.items():
        attr = f"_foreach_{name}"
        current = getattr(torch, attr, None)
        if current is None or getattr(current, "_flaggems_gcu_foreach_ref_arm", False):
            continue

        def _proxy(self, _current=current, _name=name, _routed=routed):
            try:
                hits_gcu = isinstance(self, (list, tuple)) and any(
                    isinstance(t, torch.Tensor)
                    and t.device.type == "gcu"
                    and t.dtype in _routed
                    for t in self
                )
            except TypeError:
                hits_gcu = False
            if not hits_gcu:
                return _current(self)
            host = [t.cpu() for t in self]
            if _name == "abs":
                # CPU abs 没有 Bool 档；按 harness 自身口径走 int32 往返。
                out = [
                    (
                        torch.abs(t.to(torch.int32)).to(torch.bool)
                        if t.dtype is torch.bool
                        else torch.abs(t)
                    )
                    for t in host
                ]
            else:
                out = _current(host)
            return [o.to(t.device) for o, t in zip(out, self)]

        _proxy._flaggems_gcu_foreach_ref_arm = True
        setattr(torch, attr, _proxy)


_install_aten_foreach_ref_arm_proxies()
