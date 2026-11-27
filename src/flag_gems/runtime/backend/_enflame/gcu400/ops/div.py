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
import numbers

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry, tl_extra_shim

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# libdevice fma on gcu400; used to get the exact division residual.
fma = tl_extra_shim.fma


@triton.jit
def _rne_div(x, y):
    # gcu400's f32 `/` and libdevice div_rn are only accurate to ~1-2 ulp,
    # which flips trunc() when the exact quotient is within a few ulp of an
    # integer (e.g. 1.5639446 / 0.3909861 is exactly 4.0 but divides to
    # 3.9999998). Refine with an exact FMA residual.
    #
    # Original implemented Newton with three divides (div_rn + two `/y`),
    # which is ~8x slower than torch. Use a single reciprocal instead:
    #   inv = 1/y; q = x*inv; r = fma(-q, y, x); q' = q + r*inv
    # Division is far more expensive than multiply/FMA on gcu400.
    inv = 1.0 / y
    q = x * inv
    # Skip the residual for non-finite quotients (x/0 -> +-inf, 0/0 -> nan):
    # fma(-inf, 0, x) is nan and would clobber the correct inf/nan result.
    refined = q + fma(-q, y, x) * inv
    return tl.where(inv - inv == 0.0, refined, q)


@triton.jit
def _trunc_f32(q):
    """trunc(q) for an fp32 value via an int32 round-trip.

    ``tl.floor``/``tl.ceil`` lower to extern libm calls on this backend: a
    single extern floor costs ~+1813us of serial ALU time at 512M elements
    (tan probe E, 2026-11-16), and the two-call ``where(x >= 0, floor, ceil)``
    trunc chain was the dominant cost of both rounding bodies.  The int32
    round trip truncates toward zero at a fraction of that cost and is
    bit-exact against the extern body over randn, magnitude pairs spanning
    1e-30..1e30, exact multiples, +-1-ulp-off-integer quotients and the
    special-value cross product including +-0/inf/nan (probe 2026-11-18):
    ``|q| >= 2^31`` lanes are integral (fp32 has a 24-bit mantissa) and pass
    through, NaN fails every comparison and passes through, and ``q == +-0``
    keeps its signed zero because the int cast returns +0.0 for -0.0.
    """
    qi = q.to(tl.int32).to(tl.float32)
    k = tl.where(tl.abs(q) < 2147483648.0, qi, q)
    return tl.where(q == 0.0, q, k)


@triton.jit
def _floor_f32(q):
    """floor(q) from the int32 truncation (avoids the extern libm floor)."""
    t = _trunc_f32(q)
    return tl.where(q < t, t - 1.0, t)


@triton.jit
def _exact_fmod(x, y):
    """IEEE-exact fmod built from ``fma`` and the refined division.

    gcu400's libdevice ``fmod`` is not exact once x/y is large: for
    ``x=1.47, y=1.39e-7`` it returns -7.7e-7 instead of 1.09e-7 (and even
    violates ``|r| < |y|``), which moves ``floor(x/y)`` by several units.
    ``fma`` is exact on this device, so recover the integral quotient one step
    at a time and keep ``r = x - k*y`` exact via fma.  Each reduction removes
    an approximate quotient of a few ulp, so two passes plus a magnitude
    safety net leave ``|r| < |y|``.
    """
    k = _trunc_f32(_rne_div(x, y))
    r = fma(-k, y, x)
    k = _trunc_f32(_rne_div(r, y))
    r = fma(-k, y, r)
    k = _trunc_f32(_rne_div(r, y))
    r = fma(-k, y, r)

    ay = tl.abs(y)
    ar = tl.abs(r)
    adj = tl.where(ar >= ay, tl.where(r >= 0.0, ay, -ay), 0.0)
    r = r - adj
    ar = tl.abs(r)
    adj = tl.where(ar >= ay, tl.where(r >= 0.0, ay, -ay), 0.0)
    r = r - adj

    # fmod's result carries the dividend's sign.  When the true quotient sits
    # just below an integer, the rounded division can land on the integer and
    # leave a tiny r of the wrong sign; the canonical remainder is then
    # r + sign(x)*|y|.
    wrong_sign = (r != 0.0) & ((r < 0.0) != (x < 0.0))
    return tl.where(wrong_sign, r + tl.where(x < 0.0, -ay, ay), r)


@triton.jit
def _ulp_at(value):
    """One ulp at ``|value|`` for an integral ``value``, as a float.

    gcu400's division is only accurate to ~1 ulp and aten truncates the
    *rounded* quotient for its rounding-mode ops, so a residual within one ulp
    of the integer has to yield the integer itself rather than integer-1.
    """
    bits = value.to(tl.int32, bitcast=True)
    return tl.where(value == 0.0, 0.0, (bits - 0x0B800000).to(tl.float32, bitcast=True))


@triton.jit
def _trunc_div(x, y):
    """aten-style ``trunc(x / y)`` for floating point operands.

    aten truncates the **fp32-rounded quotient**, not the real quotient: when
    the exact quotient sits just below an integer, fp32 division can round it
    up to the integer and trunc then yields the integer (e.g.
    ``-0.700268 / -0.233423`` is exactly 2.9999994 but fp32 division gives
    3.0, so aten returns 3).  ``_rne_div`` refines the reciprocal quotient
    with an exact FMA residual and matches IEEE division rounding -- 0
    mismatches over 16.7M random pairs against the CPU reference -- so simply
    truncating it reproduces aten.  Do **not** correct ``k`` with an extra
    exact-residual pass: that switches to trunc(exact quotient) and differs
    from aten at every integer boundary (8/16.7M pairs).
    """
    orig_dtype = x.dtype
    x32 = x.to(tl.float32)
    y32 = y.to(tl.float32)
    result = _trunc_f32(_rne_div(x32, y32))
    quotient_negative = ((x32 < 0.0).to(tl.int32) ^ (y32 < 0.0).to(tl.int32)) != 0
    result = tl.where(x32 == 0.0, tl.where(quotient_negative, -0.0, 0.0), result)
    return result.to(orig_dtype)


@pointwise_dynamic(promotion_methods=[(0, 1, "INT_TO_FLOAT")])
@triton.jit
def true_div_func(x, y):
    return x / y


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "INT_TO_FLOAT")])
@triton.jit
def true_div_func_tensor_scalar(x, y):
    return x / y


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "INT_TO_FLOAT")])
@triton.jit
def true_div_func_scalar_tensor(x, y):
    return x / y


# 16-bit float dtypes where the flat true-division kernel beats the enflame
# pointwise launch shape (see the flat fast-path comment above).
_TRUE_DIV_FLAT_DTYPES = (torch.float16, torch.bfloat16)
# Float dtypes accepted by the trunc flat kernels; integer trunc keeps the
# dedicated int pointwise kernels.
_TRUNC_FLAT_DTYPES = (torch.float16, torch.float32, torch.bfloat16)


def true_divide(A, B):
    logger.debug("GEMS_ENFLAME TRUE_DIVIDE")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        flat = _try_flat_tensor(
            A,
            B,
            None,
            kernel=_true_div_flat_kernel,
            wide16=True,
            dtypes=_TRUE_DIV_FLAT_DTYPES,
        )
        if flat is not None:
            return flat
        return true_div_func(A, B)
    elif isinstance(A, torch.Tensor):
        return true_div_func_tensor_scalar(A, B)
    elif isinstance(B, torch.Tensor):
        return true_div_func_scalar_tensor(A, B)
    else:
        # Both scalar
        return torch.tensor(A / B)


def true_divide_(A, B):
    logger.debug("GEMS_ENFLAME TRUE_DIVIDE_")
    if isinstance(B, torch.Tensor):
        flat = _try_flat_tensor(
            A,
            B,
            A,
            kernel=_true_div_flat_kernel,
            wide16=True,
            dtypes=_TRUE_DIV_FLAT_DTYPES,
        )
        if flat is not None:
            return flat
        return true_div_func(A, B, out0=A)
    else:
        return true_div_func_tensor_scalar(A, B, out0=A)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_func(x, y):
    return _trunc_div(x, y)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_func_tensor_scalar(x, y):
    return _trunc_div(x, y)


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_func_scalar_tensor(x, y):
    return _trunc_div(x, y)


# Integer truncation division: Triton's // on integers is C-style (truncates
# toward zero), matching aten.
@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_int_func(x, y):
    return x // y


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_int_func_tensor_scalar(x, y):
    return x // y


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_int_func_scalar_tensor(x, y):
    return x // y


def trunc_divide(A, B):
    logger.debug("GEMS_ENFLAME TRUNC_DIVIDE")
    # Integer types: dedicated int kernels (Triton // is C-style truncation)
    if isinstance(A, torch.Tensor) and not A.is_floating_point():
        if isinstance(B, torch.Tensor):
            return trunc_div_int_func(A, B)
        else:
            return trunc_div_int_func_tensor_scalar(A, B)
    if isinstance(B, torch.Tensor) and not B.is_floating_point():
        return trunc_div_int_func_scalar_tensor(A, B)
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        flat = _try_flat_tensor(
            A,
            B,
            None,
            kernel=_trunc_div_flat_kernel,
            wide16=True,
            dtypes=_TRUNC_FLAT_DTYPES,
        )
        if flat is not None:
            return flat
        return trunc_div_func(A, B)
    elif isinstance(A, torch.Tensor):
        flat = _try_flat_scalar(
            A,
            B,
            None,
            kernel=_trunc_div_scalar_flat_kernel,
            wide16=True,
            dtypes=_TRUNC_FLAT_DTYPES,
        )
        if flat is not None:
            return flat
        return trunc_div_func_tensor_scalar(A, B)
    elif isinstance(B, torch.Tensor):
        return trunc_div_func_scalar_tensor(A, B)
    else:
        return _trunc_divide_scalars(A, B)


def _trunc_divide_scalars(A, B):
    """Both operands are Python/NumPy scalars: mirror aten's wrapped-number
    semantics for ``div(..., rounding_mode="trunc")``."""
    if isinstance(A, bool) or isinstance(B, bool):
        raise NotImplementedError("\"div_trunc_cpu\" not implemented for 'Bool'")
    if isinstance(A, numbers.Integral) and isinstance(B, numbers.Integral):
        if B == 0:
            raise RuntimeError("ZeroDivisionError")
        quot = abs(int(A)) // abs(int(B))
        return torch.tensor(-quot if (A < 0) != (B < 0) else quot)
    a = torch.tensor(float(A), dtype=torch.get_default_dtype())
    b = torch.tensor(float(B), dtype=torch.get_default_dtype())
    return torch.trunc(a / b)


def trunc_divide_(A, B):
    logger.debug("GEMS_ENFLAME TRUNC_DIVIDE_")
    # Integer types: dedicated int kernels (Triton // is C-style truncation)
    if not A.is_floating_point():
        if isinstance(B, torch.Tensor):
            return trunc_div_int_func(A, B, out0=A)
        else:
            return trunc_div_int_func_tensor_scalar(A, B, out0=A)
    if isinstance(B, torch.Tensor):
        flat = _try_flat_tensor(
            A,
            B,
            A,
            kernel=_trunc_div_flat_kernel,
            wide16=True,
            dtypes=_TRUNC_FLAT_DTYPES,
        )
        if flat is not None:
            return flat
        return trunc_div_func(A, B, out0=A)
    else:
        flat = _try_flat_scalar(
            A,
            B,
            A,
            kernel=_trunc_div_scalar_flat_kernel,
            wide16=True,
            dtypes=_TRUNC_FLAT_DTYPES,
        )
        if flat is not None:
            return flat
        return trunc_div_func_tensor_scalar(A, B, out0=A)


@triton.jit
def _int_floordiv(x, y):
    # TODO: request Triton to add an integer remainder builtin
    # The semantic of Triton floordiv differs from Pytorch/Numpy
    # Triton floordiv equates to
    #     (x - np.fmod(x, y)) / y
    # whereas Pytorch floordiv is
    #     (x - np.remainder(x, y)) y
    # The results show a one off difference when
    #     C1) x and y have opposite signs
    # and C2) x is not multiples of y.
    # Apart from the above, there's an erroneous case x // 0 returns -1
    # whereas in Pytorch x // 0 returns -1 if x >=0 and -2 if x < 0
    # but this special case is coalesced into the c1 and c2 check so
    # there's extra handling.
    orig_dtype = x.dtype
    if orig_dtype == tl.int8 or orig_dtype == tl.int16:
        # Narrow integer division is lowered through i32 and cast back with
        # saturation, so INT_MIN // -1 becomes INT_MAX on this backend.
        # PyTorch wraps that single overflowing quotient back to INT_MIN
        # (which is x itself); it is the only quotient above the dtype max.
        xw = x.to(tl.int32)
        yw = y.to(tl.int32)
    else:
        xw = x
        yw = y
    # Measured on gcu400: `x % y` and `x // y` are two separate integer divisions (rem does not reuse div's
    # quotient); on 16M elements the single-division version gives 4096x4096 int16 447us->414us, int32 342us->305us.
    # So do the division only once to get the quotient, then use the exact remainder r = x - q*y to decide whether -1 is
    # needed.
    q = xw // yw
    r = xw - q * yw
    c1 = r != 0
    c2 = (xw < 0) ^ (yw < 0)
    q = tl.where(c1 & c2, q - 1, q)
    if orig_dtype == tl.int8 or orig_dtype == tl.int16:
        if orig_dtype == tl.int8:
            limit = 128
        else:
            limit = 32768
        q = tl.where(q >= limit, xw, q)
        res = q.to(orig_dtype)
    else:
        res = q
    return res


# TO be consistent with python, numpy and torch, we have to implement it in the
# following way.
# CPython
# https://github.com/python/cpython/blob/ace008c531dd685a30c1dd68f9b5ba35f20171cf/Objects/floatobject.c#L636
# numpy
# https://github.com/numpy/numpy/blob/a4ad142aa1282a77bbb05acd706cb57c9cc29846/numpy/_core/src/npymath/npy_math_internal.h.src#L532
# torch
# https://github.com/pytorch/pytorch/blob/d6d9183456cd07ca0b361a194b98c2fb196e7c36/c10/util/generic_math.h#L23
@triton.jit
def _trunc_to_int(q):
    return tl.where(q >= 0.0, tl.math.floor(q), tl.math.ceil(q))


@triton.jit
def _float_floordiv(x, y):
    """floor(x / y) for float inputs (aten ``div(rounding_mode="floor")``).

    aten does **not** floor the fp32-rounded quotient: it follows
    ``c10::div_floor_floating``, which reduces by the exact fmod and divides
    the difference.  The two disagree at every integer boundary -- for
    ``x=1.123, y=0.001`` the fp32 quotient rounds to 1123.0 while the exact
    quotient is 1122.99994, so the reference returns 1122 (measured on CPU).
    Rebuild the same reduction: exact remainder, corrected division, and the
    same tie step the reference applies to the rescaled quotient.

    gcu400: triton_gcu only exposes same-dtype signatures for fmod/div_rn, while a
    tensor+scalar pair may arrive as (fp16, fp32); both are widened to fp32 first and the
    input dtype is restored at the end (aten division of half/bfloat16 also runs in fp32).
    """
    orig_dtype = x.dtype
    x32 = x.to(tl.float32)
    y32 = y.to(tl.float32)
    remainder = _exact_fmod(x32, y32)
    imperfect = remainder != 0.0
    different_sign = ((x32 < 0.0).to(tl.int32) ^ (y32 < 0.0).to(tl.int32)) != 0

    # (x - remainder) is an exact multiple of y in real arithmetic; the
    # refined division turns it into the correctly rounded quotient.
    q = _rne_div(x32 - remainder, y32)
    q = tl.where(imperfect & different_sign, q - 1.0, q)

    floor_q = _floor_f32(q)
    c = q - floor_q > 0.5
    floor_q = tl.where(c, floor_q + 1.0, floor_q)

    q_is_zero = q == 0.0
    floor_q = tl.where(q_is_zero, tl.where(different_sign, -0.0, 0.0), floor_q)

    is_div_by_zero = y32 == 0.0
    return tl.where(is_div_by_zero, x32 / y32, floor_q).to(orig_dtype)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def floor_div_func(x, y):
    if x.type.scalar.is_int() & y.type.scalar.is_int():
        return _int_floordiv(x, y)
    else:
        return _float_floordiv(x, y)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def floor_div_func_tensor_scalar(x, y):
    if x.type.scalar.is_int() & y.type.scalar.is_int():
        return _int_floordiv(x, y)
    else:
        return _float_floordiv(x, y)


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def floor_div_func_scalar_tensor(x, y):
    if x.type.scalar.is_int() & y.type.scalar.is_int():
        return _int_floordiv(x, y)
    else:
        return _float_floordiv(x, y)


# ---------------------------------------------------------------------------
# Flat floor-division fast path.
#
# ``pointwise_dynamic`` picks a 64K-element tile and caps the grid at 48 CTAs,
# which is close to optimal for the grid, but its per-call wrapping costs an
# extra pass and its tile is far too wide for the float kernel.  A dedicated
# 1-D launch over dense storage measured, on the benchmark shapes:
#
#   dtype / shape        pointwise   flat(48 CTAs)   native
#   int8   4096x4096       471us        284us         204us
#   int16  4096x4096       568us        173us         188us
#   int32  4096x4096       221us        156us         411us
#   int16  1G              24.0ms       12.3ms        11.0ms
#   int32  1G              10.7ms       11.1ms        25.0ms
#   fp32   4096x4096      6815us       4172us         432us
#
# The kernel bodies reuse ``_int_floordiv`` / ``_float_floordiv`` unchanged, so
# the output is bit-identical to the pointwise path -- verified over magnitude
# pairs from 1e-30 to 1e30 (including overflow to inf/nan) by comparing the
# raw bit patterns, plus randn, exact multiples, and quotients sitting 1 ulp
# off an integer.
#
# A correctly rounded ``div_rn`` costs ~1200us on 16M elements here (12x a
# plain ``/``), so the float kernels avoid it entirely: one reciprocal plus an
# exact ``fma`` residual recovers the integral quotient and the true remainder,
# which is all floor/trunc need.  The residual keeps the result bit-exact
# against the fp64 ground truth, and a one-ulp guard around the integer covers
# the lanes where aten's ~1 ulp division rounds onto the integer itself.
# ---------------------------------------------------------------------------

_FLAT_BLOCK_INT = 4096
# The *scalar* integer body has a different tile ladder from the tensor/tensor one that
# _FLAT_BLOCK_INT was tuned on: re-probing _floor_div_scalar_flat_kernel on GCU400
# (GRID=48, in-place out==in, do_bench median) makes 4096 the worst tile in every case:
#   int16 16.7M: B4096 282.5us, B16384 141.5us, B32768 135.7us, B65536 137.7us
#   int16 1G   : B4096 18106us, B16384 8048us,  B32768 7757us,  B65536 7742us
#   int32 16.7M: B4096 167.2us, B16384 129.0us, B32768 121.6us, B65536 123.7us
# 32768 is the joint optimum (65536 wins by only 0.2% at 1G int16 and loses at 16.7M).
_FLAT_BLOCK_INT_SCALAR = 32768
_FLAT_BLOCK_F32 = 32768
_FLAT_BLOCK_F16_WIDE = 65536
_FLAT_BLOCK_F16_MID = 16384
# BLOCK must adapt to the element count. GCU400 measurements (kernel-only, GRID=48):
#   n=4K     fp32 optimal 512-1024 (4.2us), 16384 takes 12.9us (3x slower)
#   n=64K    fp32 optimal 4096 (7.3us), 16384 takes 12.2us
#   n=1M     fp32 optimal 8192 (45.5us), 16384 is on par
#   n=4M/16M fp32 optimal 16384 (140.6/508.2us)
#   int8/16  capped at 4096 throughout (at 16384, n=16M rises from 398us to 730us)
#   int32    at 16M, 16384 is slightly better (299 vs 347us), but the tens-of-GB
#            int32 cases in the benchmark (aten single call on the order of 50s) show speedup 2.679->2.042 after
# switching to 16384,
#            so the integer path is always capped at 4096.
# The float path was re-tuned by probes on 2026-11-18 (int-cast trunc body, ablation B/nw sweep,
# trunc_i/floor_i @ 4096^2 and 1G flat):
#   fp32     B32768/nw2 is optimal across the board (16M trunc_i 94.9us, floor_i 176.4us;
#            1G trunc_i 5.33ms, floor_i 10.19ms), 16384/nw4 is 15-30% slower
#   fp16/bf16 use B16384/nw4 for n<=2^24 (16M floor_i 226.8us, 65536/nw2 takes
#            253.7us); switch to B65536/nw2 for n>2^24 (1G floor_i 13.98ms,
#            16384/nw4 takes 15.04ms)
# Rule: target at least _FLAT_MIN_TILES tiles, then cap by dtype tier.
_FLAT_MIN_TILES = 16
# 16-bit float kernels switch from B16384/nw4 to B65536/nw2 above this size
# (see the probe table above).
_FLAT_F16_WIDE_MIN = 1 << 24
# 48 CTAs measured fastest for every dtype and shape tried; larger grids lost
# 10-25%.  Kept under the 65535 grid.x limit by the grid-stride loop.
_FLAT_GRID = 48


def _flat_launch(n_elements, dtype, wide16=False, scalar_int=False):
    """Picks (BLOCK, num_warps) for the flat kernel by element count and dtype (see the table above).

    ``wide16`` lets 16-bit float kernels use the wide B65536/nw2 shape even
    below _FLAT_F16_WIDE_MIN: the lighter true-division / trunc bodies prefer
    it from 16.7M elements up (probe 2026-11-18: plain fp16 4096^2 49.9us
    vs 67.7us; trunc_i fp16 121.6us vs 124.4us), while the heavier floor
    body keeps the B16384/nw4 mid band there (floor_i 226.8us vs 253.7us).
    """
    if dtype.is_floating_point:
        if dtype == torch.float32:
            cap, num_warps = _FLAT_BLOCK_F32, 2
        elif n_elements > _FLAT_F16_WIDE_MIN or wide16:
            cap, num_warps = _FLAT_BLOCK_F16_WIDE, 2
        else:
            cap, num_warps = _FLAT_BLOCK_F16_MID, 4
    else:
        cap = _FLAT_BLOCK_INT_SCALAR if scalar_int else _FLAT_BLOCK_INT
        num_warps = 4
    wanted = triton.next_power_of_2(max(1, triton.cdiv(n_elements, _FLAT_MIN_TILES)))
    return max(1024, min(cap, wanted)), num_warps


@libentry()
@triton.jit(do_not_specialize=["n_elements"])
def _floor_div_flat_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
    GRID_SIZE: tl.constexpr,
):
    num_tiles = tl.cdiv(n_elements, BLOCK)
    for tile_id in tl.range(tl.program_id(0), num_tiles, GRID_SIZE):
        offsets = tile_id * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        if x.type.scalar.is_int() & y.type.scalar.is_int():
            result = _int_floordiv(x, y)
        else:
            result = _float_floordiv(x, y)
        tl.store(out_ptr + offsets, result, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["n_elements", "y"])
def _floor_div_scalar_flat_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    y,
    BLOCK: tl.constexpr,
    GRID_SIZE: tl.constexpr,
):
    num_tiles = tl.cdiv(n_elements, BLOCK)
    for tile_id in tl.range(tl.program_id(0), num_tiles, GRID_SIZE):
        offsets = tile_id * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        # The Python side only takes this path when the scalar is exactly
        # representable in the element type, so this cast never rounds.  It has
        # to happen: letting the scalar arrive as int64 forces gcu400's
        # emulated 64-bit integer division, which measured 366ms for an int16
        # 4096x4096 tensor against 0.17ms in the element width.
        scalar = y if x.type.scalar.is_int() else y.to(tl.float32)
        if x.type.scalar.is_int():
            result = _int_floordiv(x, tl.full(x.shape, 0, x.dtype) + scalar)
        else:
            result = _float_floordiv(x, scalar)
        tl.store(out_ptr + offsets, result, mask=mask)


# Flat fast paths for the other two div bodies.  The kernel bodies reuse the
# exact pointwise expressions (``x / y`` / ``_trunc_div``), so results are
# bit-identical to the pointwise path -- only the launch shape differs.  The
# same dense/same-shape guard machinery as floor applies; everything the
# guard rejects keeps the pointwise path unchanged.
#
# Probe 2026-11-18 (kernel-only, GRID=48, int-cast trunc):
#   true div fp16/bf16 flat B65536/nw2: 4096^2 49.9us vs pointwise 66.4us,
#     1G 2446us vs 2790us.  fp32 flat is within noise of the pointwise shape
#     (4754-5089us vs 4660us at 1G), so the flat arm is 16-bit only.
#   trunc flat fp32 B32768/nw2: 4096^2 94.9us vs pointwise-intcast 122.3us,
#     1G 5333us vs 5696us; trunc scalar flat est ~3.8ms vs 6.1ms pointwise
#     (the int-cast body is slower than the extern one inside the scalar
#     pointwise codegen; the flat arm bypasses that regression).


@libentry()
@triton.jit(do_not_specialize=["n_elements"])
def _true_div_flat_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
    GRID_SIZE: tl.constexpr,
):
    num_tiles = tl.cdiv(n_elements, BLOCK)
    for tile_id in tl.range(tl.program_id(0), num_tiles, GRID_SIZE):
        offsets = tile_id * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x / y, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["n_elements"])
def _trunc_div_flat_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
    GRID_SIZE: tl.constexpr,
):
    num_tiles = tl.cdiv(n_elements, BLOCK)
    for tile_id in tl.range(tl.program_id(0), num_tiles, GRID_SIZE):
        offsets = tile_id * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, _trunc_div(x, y), mask=mask)


@libentry()
@triton.jit(do_not_specialize=["n_elements", "y"])
def _trunc_div_scalar_flat_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    y,
    BLOCK: tl.constexpr,
    GRID_SIZE: tl.constexpr,
):
    num_tiles = tl.cdiv(n_elements, BLOCK)
    for tile_id in tl.range(tl.program_id(0), num_tiles, GRID_SIZE):
        offsets = tile_id * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, _trunc_div(x, y), mask=mask)


def _dense(result_dtype, *tensors):
    """Whether every listed operand is dense and already at ``result_dtype``."""
    for tensor in tensors:
        if tensor is None:
            continue
        if not tensor.is_contiguous() or tensor.dtype != result_dtype:
            return False
    return True


def _int_scalar_fits(value, dtype):
    """Whether a Python int survives an exact cast to ``dtype``."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        return False
    info = torch.iinfo(dtype)
    return info.min <= int(value) <= info.max


def _try_flat_tensor(A, B, out, kernel=None, wide16=False, dtypes=None):
    """Launch the flat kernel, or return None to keep the pointwise path."""
    dtype = torch.promote_types(A.dtype, B.dtype)
    if dtype.is_complex or dtype == torch.bool:
        return None
    if dtype.is_floating_point and dtype.itemsize > 4:
        # fp64 arrives silently downconverted on this backend anyway.
        return None
    if dtypes is not None and dtype not in dtypes:
        return None
    if not _dense(dtype, A, B, out):
        return None
    # The flat kernel indexes both operands linearly, so it cannot broadcast.
    # A 0-d operand against a larger tensor (floor_divide(scalar_tensor, x))
    # has to keep the pointwise path, which does.
    if A.shape != B.shape:
        return None
    n_elements = A.numel()
    buffer = out if out is not None else torch.empty_like(A)
    if n_elements == 0:
        return buffer
    BLOCK, num_warps = _flat_launch(n_elements, dtype, wide16)
    grid_size = min(triton.cdiv(n_elements, BLOCK), _FLAT_GRID)
    flat_kernel = _floor_div_flat_kernel if kernel is None else kernel
    flat_kernel[(grid_size,)](
        A,
        B,
        buffer,
        n_elements,
        BLOCK=BLOCK,
        GRID_SIZE=grid_size,
        num_warps=num_warps,
    )
    return buffer


def _try_flat_scalar(A, B, out, kernel=None, wide16=False, dtypes=None):
    """Launch the flat kernel for a dense tensor and a Python scalar."""
    dtype = A.dtype
    if dtype.is_complex or dtype == torch.bool:
        return None
    if dtype.is_floating_point:
        if dtype.itemsize > 4 or not isinstance(B, float):
            return None
        scalar = float(B)
    else:
        if not _int_scalar_fits(B, dtype):
            return None
        scalar = int(B)
    if dtypes is not None and dtype not in dtypes:
        return None
    if not _dense(dtype, A, out):
        return None
    n_elements = A.numel()
    buffer = out if out is not None else torch.empty_like(A)
    if n_elements == 0:
        return buffer
    block, num_warps = _flat_launch(n_elements, dtype, wide16, scalar_int=True)
    grid_size = min(triton.cdiv(n_elements, block), _FLAT_GRID)
    flat_kernel = _floor_div_scalar_flat_kernel if kernel is None else kernel
    flat_kernel[(grid_size,)](
        A,
        buffer,
        n_elements,
        scalar,
        BLOCK=block,
        GRID_SIZE=grid_size,
        num_warps=num_warps,
    )
    return buffer


def floor_divide(A, B):
    logger.debug("GEMS_ENFLAME FLOOR_DIVIDE")
    # gcu400: pointwise_dynamic allocates its output with ``empty_like`` of the
    # first input, so a non-contiguous input makes the store use a
    # non-contiguous block_ptr.  The backend's block_ptr store lowering writes
    # the wrong addresses for several stride orders (see
    # docs/bug_reports/gcu400_blockptr_store_nondense.md), which corrupts the
    # result on every pointwise op.  Copy non-contiguous operands to dense
    # storage first: the returned tensor is contiguous, but correct.
    if isinstance(A, torch.Tensor) and not A.is_contiguous():
        A = A.contiguous()
    if isinstance(B, torch.Tensor) and not B.is_contiguous():
        B = B.contiguous()
    # gcu400: pointwise_dynamic promotes two tensors but leaves a scalar or a
    # 0-d tensor at its own dtype.  A mixed int/float pair then reaches the
    # kernel as (int32, float32), where _int_floordiv's `x // y` raises
    # "unexpected type fp32" and _float_floordiv's fmod has no mixed-type
    # signature.  Normalise with torch's promotion rules before dispatch.
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        flat = _try_flat_tensor(A, B, None)
        if flat is not None:
            return flat
        if A.dtype != B.dtype:
            dtype = torch.promote_types(A.dtype, B.dtype)
            if A.dtype != dtype:
                A = A.to(dtype)
            if B.dtype != dtype:
                B = B.to(dtype)
        return floor_div_func(A, B)
    elif isinstance(A, torch.Tensor):
        # The scalar specialisation of _float_floordiv used to be wrong on this
        # backend for a Python-float divisor (measured: floor_divide(x, -2.5)
        # returned only {-1, 0, 1}); that was the tensor/scalar bool `^`
        # miscompile fixed in _float_floordiv.  Keep the scalar in the kernel
        # rather than materialising a 0-d device tensor: the operand comes
        # through as fp32 (or int64 for an integer scalar), so no widening of
        # the input is needed, and dropping the extra tensor plus the
        # broadcasted tensor/tensor kernel took the 4096x4096 int cases from
        # 4.8ms/11.5ms to 0.24ms/0.15ms.
        flat = _try_flat_scalar(A, B, None)
        if flat is not None:
            return flat
        return floor_div_func_tensor_scalar(A, B)
    elif isinstance(B, torch.Tensor):
        A = torch.as_tensor(A, dtype=B.dtype, device=B.device)
        return floor_div_func(A, B)
    else:
        # Both scalar
        return torch.tensor(A // B)


def floor_divide_(A, B):
    logger.debug("GEMS_ENFLAME FLOOR_DIVIDE_")
    if isinstance(A, torch.Tensor) and not A.is_contiguous():
        # Same block_ptr store issue as floor_divide: writing into a
        # non-contiguous destination corrupts the result, so compute into a
        # dense buffer and copy back into the view.
        A.copy_(floor_divide(A, B))
        return A
    if isinstance(B, torch.Tensor) and not B.is_contiguous():
        B = B.contiguous()
    if isinstance(B, torch.Tensor):
        flat = _try_flat_tensor(A, B, A)
        if flat is not None:
            return flat
        return floor_div_func(A, B, out0=A)
    flat = _try_flat_scalar(A, B, A)
    if flat is not None:
        return flat
    return floor_div_func_tensor_scalar(A, B, out0=A)


def _any_complex(A, B):
    return (isinstance(A, torch.Tensor) and A.is_complex()) or (
        isinstance(B, torch.Tensor) and B.is_complex()
    )


def div_mode(A, B, rounding_mode=None):
    # trunc and floor both need the gcu400-specific IEEE arithmetic above.
    # None (plain true division) also routes through this module's own
    # true_divide, the same implementation div.Tensor/div.Scalar already
    # dispatch to: the generic ops-module fallback runs the generic
    # pointwise_dynamic wrapper, which measured 3-5x slower than this
    # backend's launch shape on the same `x / y` body (1G fp32 14.7ms vs
    # 4.8ms, 1G fp16 13.9ms vs 2.8ms; probe 2026-11-18).  Complex inputs
    # keep the generic route for its registered complex kernels.
    if rounding_mode == "trunc":
        return trunc_divide(A, B)
    if rounding_mode == "floor":
        return floor_divide(A, B)
    if rounding_mode is None and not _any_complex(A, B):
        return true_divide(A, B)
    from flag_gems.ops.div import div_mode as _generic_div_mode

    return _generic_div_mode(A, B, rounding_mode)


def div_mode_(A, B, rounding_mode=None):
    # See div_mode for the None-mode routing rationale.
    if rounding_mode == "trunc":
        return trunc_divide_(A, B)
    if rounding_mode == "floor":
        return floor_divide_(A, B)
    if rounding_mode is None and not _any_complex(A, B):
        return true_divide_(A, B)
    from flag_gems.ops.div import div_mode_ as _generic_div_mode_

    return _generic_div_mode_(A, B, rounding_mode)


@triton.jit
def _remainder(x, y):
    r = x % y
    c1 = r != 0
    c2 = (x < 0) ^ (y < 0)
    return tl.where(c1 & c2, r + y, r)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def rem_tt(x, y):
    return _remainder(x, y)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def rem_ts(x, y):
    return _remainder(x, y)


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def rem_st(x, y):
    return _remainder(x, y)


def remainder(A, B):
    logger.debug("GEMS_ENFLAME FLOOR_DIVIDE")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return rem_tt(A, B)
    elif isinstance(A, torch.Tensor):
        return rem_ts(A, B)
    elif isinstance(B, torch.Tensor):
        return rem_st(A, B)
    else:
        # Both scalar
        return torch.tensor(A % B)


def remainder_(A, B):
    logger.debug("GEMS_ENFLAME REMAINDER_")
    if isinstance(B, torch.Tensor):
        return rem_tt(A, B, out0=A)
    else:
        return rem_ts(A, B, out0=A)


import sys as _sys  # noqa: E402

_generic_ops_module = _sys.modules.get("flag_gems.ops")
if _generic_ops_module is not None:
    setattr(_generic_ops_module, "div_mode", div_mode)
    setattr(_generic_ops_module, "div_mode_", div_mode_)
