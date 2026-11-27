"""Triton implementation of aten::special_digamma (torch.special.digamma).

Wiring note: this file must stay self-contained.  The previous version simply
re-exported ``flag_gems.ops.digamma_.digamma``, but ``SpecOpRegistrar`` only
replaces symbols in the top-level ``flag_gems`` namespace, so that import bound
the *generic* kernel (BLOCK_SIZE=1024, one program per block, uncapped grid).
The 512M-element benchmark core shape then needs grid.x = 524288 programs,
past the GCU400 hardware limit of 65535, and the whole benchmark aborts.

Fidelity target: ATen/cephes ``calc_digamma``.  ATen accumulates the recurrence
and asymptotic series in ``float`` for float16/bfloat16/float32 inputs but
still evaluates the negative-argument reflection term ``pi / tan(pi * r)`` in
``double``, so the fp32 lanes need double-tan accuracy.  Reaching it with an
fp64 transcendental is not an option on GCU400 (no fp64 arithmetic), so the
fp32 reflection is instead evaluated with a double-float (Dekker) correction:
the argument ``pi * w`` is split into a leading part and a compensated low
part, the truncated ``tan`` is corrected by that low part, and the reciprocal
``pi / t`` is corrected by its residual.  The fractional part keeps x's sign
(modf semantics) so a tiny input does not lose its magnitude, and the
compensated quotient falls back to the plain one when it is not finite,
covering the subnormal-underflow and overflow regimes ATen handles with its
fp64 division.  Integral and bool inputs are promoted to float32 for compute
and output (an in-kernel int64 -> float32 conversion produces NaNs on GCU).

float64/complex/fp8 inputs are rejected with a RuntimeError: GCU400 has no
fp64 arithmetic (see compiler_problems P04/P21), so failing fast is better
than a compiler crash or a silent precision downgrade.

GCU400 launch geometry: the device rejects grid.x > 65535, so every kernel
runs on a fixed resident grid (NUM_SIPS*2 programs) and tile-walks the
element space, either grid-stride or in contiguous SLAB segments.  EVEN drops
the bounds mask when n is an exact multiple of the tile, and WIDE widens the
tile offset arithmetic to int64 past the 2^31-element int32 domain (see
ops/randn.py for the same guard).  The tile tier is bounded by the DSM
budget: the digamma body keeps ~26 fp32 vectors live per element group, so
BLOCK=4096 with num_warps=4 and a single stage fits the per-program budget
(448 KiB at a 48-way launch), while BLOCK=8192 with deeper stages overflows
it (measured OutOfResources, see ops/special_psi.py).

The host wrapper only allocates outputs, reads metadata, computes the launch
grid and launches the device kernel.
"""

import logging

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# GCU400 topology: 4 DIE x 6 SIP.  The launcher caps grid.x at 65535 programs
# (hardware limit), so large launches run on a fixed resident grid of
# NUM_SIPS*2 programs and each program tile-walks the element space; every
# launch therefore stays inside the limit for any input size.
NUM_SIPS = 24
MAX_GRID = NUM_SIPS * 2

# Tile tiers for the flat kernel (tuned on GCU400, official bench shapes).
# Tiny inputs keep a single narrow tile so the launch is not the bottleneck.
_TINY_N = 4096
_TINY_BLOCK = 256
_TINY_WARPS = 4
_BLOCK = 4096
_NUM_WARPS = 4
_STAGES = 1

# Split of pi for the double-float (Dekker) corrected fp32 reflection.
_PI_HI = tl.constexpr(3.1415927410125732)  # float32(pi)
_PI_LO = tl.constexpr(-8.742277657347586e-08)  # pi - float32(pi)

_FP8_DTYPES = (
    torch.float8_e4m3fn,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2,
    torch.float8_e5m2fnuz,
)


@triton.jit
def _body32(x):
    """ATen calc_digamma for a float32 accumulator."""
    neg = x < 0.0
    isint = neg & (x == tl.floor(x))

    # modf-style fractional part of x, keeping x's sign:
    #   x >= 0 -> r in [0, 1)     x < 0 -> r in (-1, 0]
    # ceil() returns -0.0 for x in (-1, 0), so a tiny negative x keeps its full
    # magnitude instead of rounding away; floor() does the same for positives.
    # ATen relies on this to keep tan(pi * r) accurate for |x| > 1.
    r = tl.where(neg, x - tl.ceil(x), x - tl.floor(x))
    # Fold |r| > 0.5 into (-0.5, 0.5): tan has period pi in pi * w, so
    # -pi / tan(pi * w) is unchanged while |pi * w| stays at correction scale.
    w = tl.where(r > 0.5, r - 1.0, tl.where(r < -0.5, r + 1.0, r))
    ph = _PI_HI * w
    q = tl.math.fma(_PI_HI, w, -ph) + _PI_LO * w
    t0 = libdevice.tan(ph)
    t = t0 + q * (1.0 + t0 * t0)
    a = _PI_HI / t
    corr = a * ((q * (1.0 + t0 * t0)) / t) - _PI_LO / t
    val = -a - corr
    # At overflow/subnormal scales the compensated quotient can become NaN or
    # Inf on its own (a = pi/t overflowing while the correction cancels it, or
    # tan of a subnormal pi*w underflowing to signed zero).  Fall back to the
    # plain fp32 quotient when the compensated one is not finite; that also
    # covers t0 == 0, whose plain quotient is the correct signed infinity.
    refl = tl.where(val - val == 0.0, val, -_PI_HI / t0)

    base = tl.where(neg, 1.0 - x, x)

    # Recurrence: push the argument to >= 6 instead of ATen's >= 10.  Each step
    # is one reciprocal plus a masked add.  Stopping at 6 leaves a truncation
    # of ~1e-12 in the cephes tail (the next omitted term is 8.3e-2 * z^7,
    # z = 1/y^2), which measured identical worst-case error to the >= 10 form
    # over 44M dense reference points.  Six masked iterations cover every
    # input: base >= 0 (x >= 0) reaches >= 6 after at most 6 steps and
    # base > 1 (x < 0) after at most 5.
    result = tl.zeros_like(x)
    y = base
    for _ in range(6):
        m = y < 6.0
        result = result - tl.where(m, 1.0 / y, 0.0)
        y = tl.where(m, y + 1.0, y)

    # cephes asymptotic series: y = z * polevl(z, A, 6), z = 1 / y^2.
    z = 1.0 / (y * y)
    p = tl.zeros_like(x) + 8.33333333333333333333e-2
    p = p * z + (-2.10927960927960927961e-2)
    p = p * z + 7.57575757575757575758e-3
    p = p * z + (-4.16666666666666666667e-3)
    p = p * z + 3.96825396825396825397e-3
    p = p * z + (-8.33333333333333333333e-3)
    p = p * z + 8.33333333333333333333e-2
    yv = z * p

    logy = tl.log(y)
    asym = logy - (0.5 / y) - yv + result
    asym = tl.where(y < 1.0e17, asym, logy - (0.5 / y) + result)
    # y == 10 (only reachable when the input itself is 10) keeps ATen's exact
    # PSI_10 return; every other argument uses the series above.
    asym = tl.where(y == 10.0, result + 2.25175258906672110764, asym)

    out = tl.where(neg, asym + refl, asym)
    nanv = tl.zeros_like(x) * float("nan")
    out = tl.where(isint, nanv, out)
    # x == 0 (either sign) -> copysign(inf, -x); -1.0 / x is exactly that.
    return tl.where(x == 0.0, -1.0 / x, out)


# --------------------------------------------------------------------------- #
# kernel: flat (contiguous logical index space)
#
# Every launch runs on a capped resident grid and tile-walks the element
# space, so grid.x never exceeds the 65535 hardware limit for any input size.
# EVEN drops the bounds mask on exact-multiple tiles; WIDE widens the offset
# arithmetic to int64 past the int32 element domain (n > 2^31 - BLOCK); SLAB
# assigns each program a contiguous block range instead of striding, which
# flattens HBM channel hotspots on multi-hundred-MiB streams.
# --------------------------------------------------------------------------- #
@libentry()
@triton.jit
def _digamma_flat32(
    x_ptr,
    o_ptr,
    n,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    WIDE: tl.constexpr,
    SLAB: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_blocks = (n + BLOCK - 1) // BLOCK
    if SLAB:
        per_pid = (num_blocks + num_pids - 1) // num_pids
        begin = pid * per_pid
        end = tl.minimum(begin + per_pid, num_blocks)
        step = 1
    else:
        begin = pid
        end = num_blocks
        step = num_pids
    for block_id in tl.range(begin, end, step, num_stages=NUM_STAGES):
        if WIDE:
            offs = block_id.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        else:
            offs = block_id * BLOCK + tl.arange(0, BLOCK)
        if EVEN:
            x = tl.load(x_ptr + offs).to(tl.float32)
            r = _body32(x)
            tl.store(o_ptr + offs, r.to(o_ptr.dtype.element_ty))
        else:
            mask = offs < n
            x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
            r = _body32(x)
            tl.store(o_ptr + offs, r.to(o_ptr.dtype.element_ty), mask=mask)


def _flat_cfg(n):
    """Return (BLOCK, num_warps, num_stages) for a flat launch of ``n`` elements."""
    if n <= _TINY_N:
        return _TINY_BLOCK, _TINY_WARPS, 1
    return _BLOCK, _NUM_WARPS, _STAGES


def _flat_kwargs(n, block):
    even = n % block == 0
    wide = n > 2**31 - block
    return even, wide


def special_digamma(A):
    """Elementwise digamma, matching aten::special_digamma semantics."""
    logger.debug("GEMS_ENFLAME_GCU400 SPECIAL_DIGAMMA")
    if not isinstance(A, torch.Tensor):
        raise TypeError("special_digamma expects a torch.Tensor as the first argument")

    dt = A.dtype
    if dt.is_complex or dt in _FP8_DTYPES or dt == torch.float64:
        raise RuntimeError(f"special_digamma is not implemented for {dt} on GCU400")

    x = A
    if not x.is_floating_point():
        # Integral/bool: ATen computes in float32 and outputs float32, and a
        # GCU in-kernel int64 -> float32 conversion produces NaNs, so promote
        # on the host; numeric semantics then match the reference exactly.
        x = x.to(torch.float32)
    if not x.is_contiguous():
        x = x.contiguous()

    out = torch.empty_like(x)
    n = x.numel()
    if n == 0:
        return out

    block, nw, stages = _flat_cfg(n)
    grid = (min(triton.cdiv(n, block), MAX_GRID),)
    even, wide = _flat_kwargs(n, block)
    with torch_device_fn.device(x.device):
        _digamma_flat32[grid](
            x,
            out,
            n,
            BLOCK=block,
            EVEN=even,
            WIDE=wide,
            SLAB=False,
            NUM_STAGES=stages,
            num_warps=nw,
        )
    return out
