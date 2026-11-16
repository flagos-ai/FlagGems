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

"""``aten::kaiser_window`` for Kunlunxin/XPU3.

Why this override exists
------------------------
The generic ``flag_gems.ops.kaiser_window`` kernel calls
``libdevice.cyl_bessel_i0`` (modified Bessel function of order 0).  On the
XPU3 Triton backend that symbol has no device implementation: the kernel does
not even link,

    ld.lld: error: undefined symbol: Unsupported
    >>> referenced by kaiser_window.py:70 (fp32 branch)

so *every* window_length >= 2 fails (61/73 cases of ``test_kaiser_window``);
only the ``window_length in {0, 1}`` fast returns work.  This is a hard
functional hole, not a performance regression.

Kernel formulation
------------------
I0 is evaluated inline with the Abramowitz & Stegun 9.8.1 / 9.8.2 rational
approximations (the same family already shipped in ``flag_gems.ops.i0`` /
``special_i0e``), which are FMA + ``sqrt`` + ``exp`` only -- no libdevice
special-function soft extern.  The *ratio*

    val = I0(z) / I0(beta),      z = beta * sqrt(max(1 - x^2, 0)) <= beta

is computed through the exponentially scaled form ``i0e(t) = I0(t) e^{-t}``
so the ``exp(beta)`` never appears on its own: the pipeline is

    large-z branch : (q/sqrt(z)) * exp(z - beta) / i0e(beta)   (exp arg <= 0)
    small-z branch : p(z) * exp(-beta) / i0e(beta)             (p == I0(z))

Both exponents are <= 0, so large beta (torch allows e.g. 1000, for which
I0(beta) overflows fp32) degrades to underflow-to-zero edge samples instead of
NaN/inf.  The single per-lane transcendental is ``tl.exp`` (fast on XPU3; the
broken ``tl.exp2`` is not used).  ``--ref cpu`` parity: torch computes the
window in double then rounds; our fp32 pipeline sits ~1e-6 from it, far inside
the CI tolerances (fp32 atol 1e-4 / rtol 1.3e-6, fp16 1e-3, bf16 0.016).

The fp16/bf16 index-rounding quirk of the generic kernel is preserved, but
computed explicitly (``_round_index``): torch rounds the element index into the
*output* dtype before the arithmetic, and XPU3 elides an in-register int ->
fp16/bf16 cast (both are carried as fp32), so the generic
``idx.to(out_dtype).to(tl.float32)`` silently leaves the index unrounded and
shifts whole-ulp bf16 samples.  A round-half-to-even bit trick on the fp32 index
reproduces torch exactly.

Geometry is a flat 1-D tile with a numel-adaptive BLOCK and a ``MASKED``
constexpr so exactly-dividing windows skip the masked-memory tax.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime

logger = logging.getLogger(__name__)


@triton.jit
def _round_index(v, MAN: tl.constexpr):
    """Round a non-negative integer-valued fp32 index to ``1 + MAN`` significand
    bits with IEEE round-half-to-even, i.e. what ``idx.to(fp16/bf16)`` means.

    On XPU3 a ``tl`` int -> bf16/fp16 cast is elided in registers (bf16/fp16 are
    carried as fp32), so the generic ``idx.to(out_dtype).to(tl.float32)`` never
    rounds the index -- observable for bf16 windows as a whole-ulp value shift
    once the index passes the dtype's exact-integer range.  Rounding the fp32
    bit pattern directly reproduces torch's CPU/ATen rounding exactly.
    """
    u = v.to(tl.int32, bitcast=True)
    shift: tl.constexpr = 23 - MAN
    lsb = (u >> shift) & 1
    u = u + ((1 << (shift - 1)) - 1) + lsb
    u = u & (-(1 << shift))
    return u.to(tl.float32, bitcast=True)


@triton.jit
def _i0e(ax):
    """Exponentially scaled I0: ``I0(ax) * exp(-ax)`` for ``ax >= 0``.

    Abramowitz & Stegun 9.8.1 (``ax <= 3.75``) and 9.8.2 (``ax > 3.75``),
    max relative error ~2e-7 in fp32.  ``ax`` may be a scalar or a vector.
    """
    # Small argument: I0(ax) = poly((ax/3.75)^2)
    t = ax / 3.75
    t2 = t * t
    p = 1.0 + t2 * (
        3.5156229
        + t2
        * (
            3.0899424
            + t2 * (1.2067492 + t2 * (0.2659732 + t2 * (0.0360768 + t2 * 0.0045813)))
        )
    )
    small = p * tl.exp(-ax)

    # Asymptotic branch; clamp the reciprocal argument away from 0 so the
    # (unused) large-branch values stay finite when ax == 0.
    ax_safe = tl.maximum(ax, 1.0)
    u = 3.75 / ax_safe
    q = 0.39894228 + u * (
        0.01328592
        + u
        * (
            0.00225319
            + u
            * (
                -0.00157565
                + u
                * (
                    0.00916281
                    + u
                    * (
                        -0.02057706
                        + u * (0.02635537 + u * (-0.01647633 + u * 0.00392377))
                    )
                )
            )
        )
    )
    large = q / tl.sqrt(ax_safe)
    return tl.where(ax <= 3.75, small, large)


@triton.jit
def _i0_ratio(z, beta):
    """``I0(|z|) / I0(|beta|)`` without ever materialising ``exp(beta)``."""
    az = tl.abs(z)
    ab = tl.abs(beta)

    # Scalar denominator terms (beta is uniform across lanes).
    inv_den_i0e = 1.0 / _i0e(ab)  # 1 / (I0(ab) e^{-ab})
    inv_den_i0 = tl.exp(-ab) * inv_den_i0e  # 1 / I0(ab)

    # Numerator small branch: I0(az) == poly(az), az <= 3.75.
    t = az / 3.75
    t2 = t * t
    p = 1.0 + t2 * (
        3.5156229
        + t2
        * (
            3.0899424
            + t2 * (1.2067492 + t2 * (0.2659732 + t2 * (0.0360768 + t2 * 0.0045813)))
        )
    )
    small = p * inv_den_i0

    # Numerator large branch: I0(az) = (q/sqrt(az)) e^{az}; fold the e^{az} into
    # the denominator scale, exp(az - ab) <= 1 because az <= ab.
    az_safe = tl.maximum(az, 1.0)
    u = 3.75 / az_safe
    q = 0.39894228 + u * (
        0.01328592
        + u
        * (
            0.00225319
            + u
            * (
                -0.00157565
                + u
                * (
                    0.00916281
                    + u
                    * (
                        -0.02057706
                        + u * (0.02635537 + u * (-0.01647633 + u * 0.00392377))
                    )
                )
            )
        )
    )
    large = (q / tl.sqrt(az_safe)) * (tl.exp(az - ab) * inv_den_i0e)

    return tl.where(az <= 3.75, small, large)


@triton.jit
def _kaiser_window_kernel(
    out_ptr,
    window_length,
    half_length,
    beta,
    IS_FP64: tl.constexpr,
    LOW_MAN: tl.constexpr,
    BLOCK: tl.constexpr,
    MASKED: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    out_dtype = out_ptr.dtype.element_ty

    if IS_FP64:
        n = idx.to(tl.float64)
        half = tl.full((), 0.0, tl.float64) + half_length
        b = tl.full((), 0.0, tl.float64) + beta
        x = (n - half) / half
        z = b * tl.sqrt(tl.maximum(1.0 - x * x, 0.0))
        val = _i0_ratio(z, b)
    else:
        if LOW_MAN >= 0:
            # torch rounds the element index into the *output* dtype before the
            # arithmetic, which is observable for fp16/bf16 once the index
            # exceeds the dtype's exact-integer range; reproduce it exactly.
            n = _round_index(idx.to(tl.float32), LOW_MAN)
        else:
            n = idx.to(tl.float32)
        x = (n - half_length) / half_length
        z = beta * tl.sqrt(tl.maximum(1.0 - x * x, 0.0))
        val = _i0_ratio(z, beta)

    if MASKED:
        tl.store(out_ptr + idx, val.to(out_dtype), mask=idx < window_length)
    else:
        tl.store(out_ptr + idx, val.to(out_dtype))


@triton.jit
def _fill_one_kernel(out_ptr, BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    out_dtype = out_ptr.dtype.element_ty
    tl.store(out_ptr + idx, tl.full((BLOCK,), 1.0, out_dtype), mask=idx < 1)


def _pick_block(n):
    # Flat 1-D tiling: one wide program for tiny windows, otherwise ~8 tiles of
    # at most 32K (bandwidth plateau) / 64K for very large windows.
    if n <= 4096:
        blk = max(1, triton.next_power_of_2(n))
    else:
        cap = 65536 if n >= (8 << 20) else 32768
        blk = min(cap, max(8192, triton.next_power_of_2((n + 7) // 8)))
    grid = max(1, (n + blk - 1) // blk)
    masked = (n % blk) != 0
    return blk, grid, masked


def kaiser_window(
    window_length,
    periodic=True,
    beta=12.0,
    *,
    dtype=None,
    layout=torch.strided,
    device=None,
    pin_memory=None,
):
    logger.debug("GEMS_KUNLUNXIN KAISER_WINDOW")
    assert window_length >= 0, "window_length must be non-negative"

    if dtype is None:
        dtype = torch.get_default_dtype()
    assert dtype.is_floating_point, "kaiser_window only supports floating point dtype"

    if layout != torch.strided:
        raise ValueError("Only strided layout is supported for kaiser_window.")

    if device is None:
        device = torch.device(runtime.device.name)

    out = torch.empty(
        (window_length,),
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
    )

    if window_length == 0:
        return out
    if window_length == 1:
        _fill_one_kernel[(1,)](out, BLOCK=1)
        return out

    # N is the effective full size: aten uses N = window_length for the
    # periodic sampling and N = window_length - 1 for the symmetric window.
    full_length = window_length if periodic else window_length - 1
    half_length = full_length / 2.0

    blk, grid, masked = _pick_block(window_length)
    if dtype == torch.float16:
        low_man = 10
    elif dtype == torch.bfloat16:
        low_man = 7
    else:
        low_man = -1
    _kaiser_window_kernel[(grid,)](
        out,
        window_length,
        half_length,
        float(beta),
        dtype == torch.float64,
        low_man,
        BLOCK=blk,
        MASKED=masked,
    )
    return out


def kaiser_window_periodic(
    window_length,
    periodic,
    *,
    dtype=None,
    layout=torch.strided,
    device=None,
    pin_memory=None,
):
    return kaiser_window(
        window_length,
        periodic,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
    )


def kaiser_window_beta(
    window_length,
    periodic,
    beta,
    *,
    dtype=None,
    layout=torch.strided,
    device=None,
    pin_memory=None,
):
    return kaiser_window(
        window_length,
        periodic,
        beta,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
    )
