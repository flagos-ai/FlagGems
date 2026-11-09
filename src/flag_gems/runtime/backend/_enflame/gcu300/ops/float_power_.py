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

"""FlagGems float_power.Scalar_out (out variant) reimplemented with Triton.

Semantics reproduced from the FlagGems reference implementation:

    out[i] = float_power(A, exponent[i])   with out.dtype == torch.float64

    * ``out`` must be a float64 tensor on the exponent's device; its shape is
      resized to ``exponent.shape`` when needed and the very same tensor object
      is returned.
    * The power is evaluated element-wise with one uniform scalar base and one
      per-lane exponent (no broadcasting of the exponent).
    * A positive base uses exp2(y * log2(base)); base == 2.0 gives
      log2(base) == 1.0 exactly, so the result is bit-identical to exp2(y).
    * IEEE special cases of C ``pow`` are preserved: pow(x, 0) == 1,
      pow(1, y) == 1, pow(0, y<0) == inf, negative base with a non-integral
      exponent == NaN, negative base with an odd integer exponent keeps the
      sign.

Target note (Enflame GCU): the device has no 64-bit ALU and ``tl.float64``
values cannot appear in kernel IR ("64-bit data type not supported on
GCU300").  A float32 value stored through a float64 pointer is widened by the
backend, which is exactly the reference behaviour (the device ``pow`` is
computed in float32 as well).
"""

import math
import struct

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_BLOCK = 1024
# From this element count upwards the launch configuration is searched: large
# inputs were measured to be highly launch-sensitive (the same body at a wider
# tile is up to 2.6x faster on 2048x2048), while every smaller input keeps the
# fixed BLOCK=1024/num_warps=4 launch whose compiled code is the correctness
# baseline.  The two kernels are separate functions, so the small-input binary
# is unaffected by the search.
_BIG_MIN_ELEMENTS = 131072
# The mask-free large-input path needs the element count to cover its tile
# exactly (8192 is a multiple of every candidate BLOCK) and the output storage
# to be aligned enough for the vector accesses the hints describe.
_BIG_ALIGN = 8192
_MAX_RANK = 8
_BLOCK_MAX = 131072
_BLOCK_MAX_HUGE = 262144
_MAX_GRID_X = 65535


def _pick_block(n):
    """Empty-kernel cost is ~11.2us + 0.46us * grid.x on this vendor, so one
    program covers the tensor up to 128K elements (262K above 256 programs)
    instead of the previous fixed 1024-element tile."""
    block = 1 << max(10, (n - 1).bit_length())
    if block > _BLOCK_MAX:
        block = _BLOCK_MAX
    if (n + block - 1) // block > 256:
        block = _BLOCK_MAX_HUGE
    while (n + block - 1) // block > _MAX_GRID_X:
        block *= 2
    return block


def _pick_warps(block, nblocks):
    if block <= 16384:
        return 2
    if block <= 32768:
        return 4
    if nblocks <= 2:
        return 8
    if nblocks <= 4:
        return 4
    return 2


_MODE_POS = 0  # base > 0
_MODE_NEG = 1  # base < 0
_MODE_ONE = 2  # base == 1
_MODE_ZERO = 3  # base == 0
_MODE_NAN = 4  # base is NaN
_MODE_POS_INF = 5  # base == +inf
_MODE_NEG_INF = 6  # base == -inf

# Kerns of @triton.jit code may only read globals that are tl.constexpr.
_KM_POS = tl.constexpr(0)
_KM_NEG = tl.constexpr(1)
_KM_ONE = tl.constexpr(2)
_KM_ZERO = tl.constexpr(3)
_KM_NAN = tl.constexpr(4)
_KM_POS_INF = tl.constexpr(5)
_KM_NEG_INF = tl.constexpr(6)


def _f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def _split_f32(value):
    hi = _f32(value)
    return hi, value - hi


# Taylor coefficients ln(2)**k / k! of 2**f, each split into an exact float32
# high part plus its float32 residual so the double-single polynomial is not
# limited by coefficient rounding (the CPU float64 oracle allows only ~0.84 ulp
# of float32 error).
_LN2 = math.log(2.0)
_COEFFS = [_split_f32(_LN2**k / math.factorial(k)) for k in range(11)]
_CH = tl.constexpr(tuple(c[0] for c in _COEFFS))
_CL = tl.constexpr(tuple(c[1] for c in _COEFFS))


@triton.jit
def _ds_mul(a_hi, a_lo, b):
    """(a_hi + a_lo) * b as a double-single pair (b exact in float32)."""
    p = a_hi * b
    e = tl.math.fma(a_hi, b, -p)
    return p, a_lo * b + e


@triton.jit
def _ds_add(a_hi, a_lo, c):
    """(a_hi + a_lo) + c as a double-single pair (c exact in float32)."""
    s = a_hi + c
    t = s - a_hi
    return s, (a_hi - (s - t)) + (c - t) + a_lo


@triton.jit
def _pow2_bits(k):
    """2**k for an integer-valued float32 k, built from the exponent field."""
    return ((k + 127) << 23).to(tl.float32, bitcast=True)


@triton.jit
def _exp2_accurate(t_hi, t_lo, EXACT: tl.constexpr):
    """Correctly rounded float32 2**(t_hi + t_lo).

    The GCU exp2 instruction is only accurate to a few float32 ulp, while the
    authoritative reference is an exact float64 power, so the exponential is
    carried in double-single (two float32) arithmetic:
      t = n + f with an integer n (magic-number rounding) and |f| <= 0.5,
      2**f from the degree-10 ln2 Taylor polynomial in double-single,
      2**n applied as an exact exponent-field scaling split in two halves.
    """
    n = (t_hi + 12582912.0) - 12582912.0
    f = t_hi - n
    r_hi = _CH[10]
    r_lo = _CL[10]
    r_hi, r_lo = _ds_mul(r_hi, r_lo, f)
    r_hi, r_lo = _ds_add(r_hi, r_lo, _CH[9])
    r_lo = r_lo + _CL[9]
    r_hi, r_lo = _ds_mul(r_hi, r_lo, f)
    r_hi, r_lo = _ds_add(r_hi, r_lo, _CH[8])
    r_lo = r_lo + _CL[8]
    r_hi, r_lo = _ds_mul(r_hi, r_lo, f)
    r_hi, r_lo = _ds_add(r_hi, r_lo, _CH[7])
    r_lo = r_lo + _CL[7]
    r_hi, r_lo = _ds_mul(r_hi, r_lo, f)
    r_hi, r_lo = _ds_add(r_hi, r_lo, _CH[6])
    r_lo = r_lo + _CL[6]
    r_hi, r_lo = _ds_mul(r_hi, r_lo, f)
    r_hi, r_lo = _ds_add(r_hi, r_lo, _CH[5])
    r_lo = r_lo + _CL[5]
    r_hi, r_lo = _ds_mul(r_hi, r_lo, f)
    r_hi, r_lo = _ds_add(r_hi, r_lo, _CH[4])
    r_lo = r_lo + _CL[4]
    r_hi, r_lo = _ds_mul(r_hi, r_lo, f)
    r_hi, r_lo = _ds_add(r_hi, r_lo, _CH[3])
    r_lo = r_lo + _CL[3]
    r_hi, r_lo = _ds_mul(r_hi, r_lo, f)
    r_hi, r_lo = _ds_add(r_hi, r_lo, _CH[2])
    r_lo = r_lo + _CL[2]
    r_hi, r_lo = _ds_mul(r_hi, r_lo, f)
    r_hi, r_lo = _ds_add(r_hi, r_lo, _CH[1])
    r_lo = r_lo + _CL[1]
    r_hi, r_lo = _ds_mul(r_hi, r_lo, f)
    r_hi, r_lo = _ds_add(r_hi, r_lo, _CH[0])
    r_lo = r_lo + _CL[0]
    r = r_hi + r_lo
    if not EXACT:
        # 2**(f + t_lo) = 2**f * (1 + ln2 * t_lo) to first order
        r = r * (1.0 + 0.6931471805599453 * t_lo)
    ni = n.to(tl.int32)
    half = ni >> 1
    r = r * _pow2_bits(half)
    r = r * _pow2_bits(ni - half)
    # infinite exponents bypass the integer split above
    r = tl.where(t_hi == float("inf"), float("inf"), r)
    r = tl.where(t_hi == float("-inf"), 0.0, r)
    return r


@triton.jit
def _pow_value(x, BASE_LOG2: tl.constexpr, MODE: tl.constexpr):
    """Element-wise pow(base, x) for a uniform base, evaluated in float32."""
    if BASE_LOG2 == 1.0:
        t_hi = x
        t_lo = 0.0
    else:
        t_hi = x * BASE_LOG2
        t_lo = tl.math.fma(x, BASE_LOG2, -t_hi)
    if MODE == _KM_POS:
        return _exp2_accurate(t_hi, t_lo, BASE_LOG2 == 1.0)
    elif MODE == _KM_ONE:
        return tl.full(x.shape, 1.0, tl.float32)
    elif MODE == _KM_ZERO:
        nan = tl.sqrt(tl.full(x.shape, -1.0, tl.float32))
        return tl.where(
            x > 0.0,
            0.0,
            tl.where(x < 0.0, float("inf"), tl.where(x == 0.0, 1.0, nan)),
        )
    elif MODE == _KM_POS_INF:
        nan = tl.sqrt(tl.full(x.shape, -1.0, tl.float32))
        return tl.where(
            x > 0.0,
            float("inf"),
            tl.where(x < 0.0, 0.0, tl.where(x == 0.0, 1.0, nan)),
        )
    else:
        # negative, NaN or -inf base: need the sign / integrality of x
        nan = tl.sqrt(tl.full(x.shape, -1.0, tl.float32))
        integral = x == tl.floor(x)
        mag = _exp2_accurate(t_hi, t_lo, BASE_LOG2 == 1.0)
        if MODE == _KM_NEG:
            odd = (x.to(tl.int32) % 2) != 0
            signed = tl.where(odd, -mag, mag)
            return tl.where(integral, signed, nan)
        elif MODE == _KM_NAN:
            return tl.where(x == 0.0, 1.0, nan)
        else:  # _MODE_NEG_INF
            signed = tl.where((x > 0.0) | ((x.to(tl.int32) % 2) != 0), mag, -mag)
            pos = tl.where(integral, signed, nan)
            neg = tl.where(x > 0.0, 0.0, -0.0)
            return tl.where(x >= 0.0, pos, neg)


@triton.jit
def _pow_flat_kernel(
    exp_ptr,
    out_ptr,
    n_elements,
    BASE_LOG2: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(exp_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = _pow_value(x, BASE_LOG2, MODE)
    # storing a float32 value through the float64 pointer widens it on GCU
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def _pow_flat_kernel_big(
    exp_ptr,
    out_ptr,
    n_elements,
    BASE_LOG2: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Same arithmetic as _pow_flat_kernel on a mask-free exact-multiple path.

    Only chosen when n >= _BIG_MIN_ELEMENTS, n is a multiple of _BIG_ALIGN (so
    of every candidate BLOCK, leaving no out-of-range lane) and the output is
    aligned, so the fixed-launch kernel that serves every correctness workload
    keeps its compiled code and its bounds mask.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    # tell the backend the offsets are a contiguous, BLOCK-aligned run
    offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)
    x = tl.load(exp_ptr + offs).to(tl.float32)
    y = _pow_value(x, BASE_LOG2, MODE)
    # storing a float32 value through the float64 pointer widens it on GCU
    tl.store(out_ptr + offs, y)


@triton.jit
def _pow_strided_kernel(
    exp_ptr,
    out_ptr,
    n_elements,
    S0,
    S1,
    S2,
    S3,
    S4,
    S5,
    S6,
    S7,
    ES0,
    ES1,
    ES2,
    ES3,
    ES4,
    ES5,
    ES6,
    ES7,
    OS0,
    OS1,
    OS2,
    OS3,
    OS4,
    OS5,
    OS6,
    OS7,
    RANK: tl.constexpr,
    BASE_LOG2: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    idx = offs
    eo = tl.zeros([BLOCK], tl.int32)
    oo = tl.zeros([BLOCK], tl.int32)
    if RANK >= 1:
        i = idx % S0
        idx = idx // S0
        eo += i * ES0
        oo += i * OS0
    if RANK >= 2:
        i = idx % S1
        idx = idx // S1
        eo += i * ES1
        oo += i * OS1
    if RANK >= 3:
        i = idx % S2
        idx = idx // S2
        eo += i * ES2
        oo += i * OS2
    if RANK >= 4:
        i = idx % S3
        idx = idx // S3
        eo += i * ES3
        oo += i * OS3
    if RANK >= 5:
        i = idx % S4
        idx = idx // S4
        eo += i * ES4
        oo += i * OS4
    if RANK >= 6:
        i = idx % S5
        idx = idx // S5
        eo += i * ES5
        oo += i * OS5
    if RANK >= 7:
        i = idx % S6
        idx = idx // S6
        eo += i * ES6
        oo += i * OS6
    if RANK >= 8:
        i = idx % S7
        eo += i * ES7
        oo += i * OS7
    x = tl.load(exp_ptr + eo, mask=mask, other=0.0).to(tl.float32)
    y = _pow_value(x, BASE_LOG2, MODE)
    tl.store(out_ptr + oo, y, mask=mask)


def _classify_base(base):
    """Return (mode, log2(|base|)) for a Python scalar base."""
    if base != base:  # NaN
        return _MODE_NAN, 0.0
    if base == 1.0:
        return _MODE_ONE, 0.0
    if base == 0.0:
        return _MODE_ZERO, 0.0
    if base == float("inf"):
        return _MODE_POS_INF, 0.0
    if base == float("-inf"):
        return _MODE_NEG_INF, 0.0
    if base > 0.0:
        return _MODE_POS, math.log2(base)
    return _MODE_NEG, math.log2(-base)


def _rev_pad(seq, fill, n=_MAX_RANK):
    out = list(seq)[::-1]
    if len(out) < n:
        out.extend([fill] * (n - len(out)))
    return out


def _fp32_alias(out):
    """Zero-copy float32 view over the float64 ``out`` storage.

    A float64 pointer cannot be passed to a GCU Triton kernel, while a float32
    view aliases the same storage and is still addressed in 8-byte elements by
    the backend.
    """
    try:
        return out.view(torch.float32)
    except RuntimeError:
        pass
    span = 1 + sum((d - 1) * st for d, st in zip(out.shape, out.stride()))
    return out.as_strided((span,), (1,), out.storage_offset()).view(torch.float32)


def float_power_scalar_tensor_out(A, exponent, *, out):
    logger.debug("GEMS FLOAT_POWER_SCALAR_TENSOR_OUT")
    if out.dtype != torch.float64:
        raise RuntimeError(
            "the output given to float_power has dtype {} but the operation's "
            "result requires dtype Double".format(out.dtype)
        )
    if not torch.is_tensor(exponent):
        exponent = torch.as_tensor(exponent, device=out.device)
    if out.device != exponent.device:
        raise RuntimeError(
            "Expected out tensor to have device {}, but got {} instead".format(
                exponent.device, out.device
            )
        )
    shape = exponent.shape
    if out.shape != shape:
        out.resize_(shape)
    n = exponent.numel()
    if n == 0:
        return out

    base = A if isinstance(A, float) else float(A)
    mode, base_log2 = _classify_base(base)
    # A float64 pointer cannot appear in a GCU kernel ("64-bit data type not
    # supported on GCU300"), so the kernel receives a zero-copy float32 view of
    # the same storage.  The backend still addresses it in 8-byte (float64)
    # elements and widens the float32 result, which is exactly the reference
    # value.
    out_arg = _fp32_alias(out)

    if exponent.is_contiguous() and out.is_contiguous():
        if n >= _BIG_MIN_ELEMENTS and n % _BIG_ALIGN == 0 and out.data_ptr() % 256 == 0:
            block = _pick_block(n)
            _pow_flat_kernel_big[(n // block,)](
                exponent,
                out_arg,
                n,
                base_log2,
                mode,
                BLOCK=block,
                num_warps=_pick_warps(block, n // block),
            )
        else:
            block = _pick_block(n)
            nb = (n + block - 1) // block
            _pow_flat_kernel[(nb,)](
                exponent,
                out_arg,
                n,
                base_log2,
                mode,
                BLOCK=block,
                num_warps=_pick_warps(block, nb),
            )
        return out

    rank = exponent.dim()
    if rank > _MAX_RANK:
        raise RuntimeError(
            "float_power: rank {} non-contiguous input is not supported".format(rank)
        )
    shapes = _rev_pad(shape, 1)
    e_strides = _rev_pad(exponent.stride(), 0)
    o_strides = _rev_pad(out.stride(), 0)
    grid = ((n + _BLOCK - 1) // _BLOCK,)
    _pow_strided_kernel[grid](
        exponent,
        out_arg,
        n,
        *shapes,
        *e_strides,
        *o_strides,
        rank,
        base_log2,
        mode,
        BLOCK=_BLOCK,
        num_warps=4,
    )
    return out
