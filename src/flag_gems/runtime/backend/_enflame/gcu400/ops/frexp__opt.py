"""frexp GCU400-optimized: tiered flat/strided 1D kernels.

The previous path ran every shape through the pointwise_dynamic codegen
template whose ENFLAME config caps the grid at 12 CTAs
(max_grid_size=(12,1,1), max_tile_size=4096): 48 SPs host 12 CTAs of ~131k
elementwise tiles each, so 2^29 fp16 measured 11.84 ms (363 GB/s effective)
against torch native 4.77 ms (probe, 2026-11-13).

Contiguous fp16/bf16/fp32 tensors are dispatched to hand-written 1D
kernels instead (probe-validated tiers, do_bench w1000/r100 median):

* the body decomposes IEEE-754 bits directly: ``payload = bits & 0x7FFFFFFF``
  yields |x| in one pass, the subnormal normalization ``abs_x * 2**126``
  runs unconditionally (special lanes are overwritten by the final selects);
  versus the generic fp32-math body this drops three float compares, one
  select and the fp sign flip for -31% at 2^24 / -37..-56% at 2^29 (same
  launch config);
* special values are detected in the float domain
  (``x == 0 | (x - x) != 0``), which also fixes the generic body's NaN
  handling on this backend: ``x != x`` is folded to False in full kernel
  contexts (compiler_traps), so frexp(NaN) returned (~1.0, 129) instead of
  (NaN, 0); integer bit compares on NaN-derived bits are equally
  untrustworthy here. Verified bitwise against CPU/native on a full
  codepoint scan (NaN payload family, sNaN/qNaN, +/-inf, +/-0, subnormals,
  ULP-below-powers-of-two) for all three dtypes;
* tiers (fp16/bf16): n<8192 single CTA BLOCK=next_pow2(n) nw2;  n<2^19
  flat B16384 nw4;  n<2^23 flat B32768 nw2;  n<2^27 flat B65536 nw2;  else
  strided B131072 nw2 grid=48
  (2^24 98.9us vs torch 159.6; 2^29 2636us vs torch 4770 = 1.81x;
  Triton GCU 4.1.6 retest 2026-11-18: strided B131072 nw2 overtakes the
  flat launch at >=2^27 (2636 vs 2676us) while 2^26 still prefers flat
  (351 vs 354us), and mid 2^24 prefers flat B65536 (98.9 vs 99.8us));
* tiers (fp32): n<8192 single CTA BLOCK=next_pow2(n) nw2;  n<2^19 flat
  B16384 nw2;  n<2^22 flat B32768 nw2;  else strided B32768 nw2 grid=48
  (2^24 85.4us vs torch 145.4; 2^29 2436us vs torch 4306 = 1.77x,
  2.64TB/s at the 3-stream roof - the 4.1.6 toolchain shifted the
  strided optimum from B65536/nw4 (92.5/2651us) to B32768/nw2,
  -7.8/-8.1%, 3x resampled stable);
* n % BLOCK == 0 takes an unmasked EVEN fast path; non-eligible forms
  (non-contiguous / other device / n >= 2^31 / non-fp16-bf16-fp32) fall
  back to the generic pointwise_dynamic wrapper, zero behaviour change;
* DSM budget: single-input load BLOCK*itemsize dominates Required
  (fp16 B131072 = 256KB, fp32 B32768 = 128KB; fp32 flat B262144 probe hit
  OutOfResources 1048576 > 917504, no tier crosses the limit).
"""

import logging
import time

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device

from .frexp import _frexp_func

logger = logging.getLogger(__name__)
device = device.name

_FAST_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_SMALL_N = 8192
_F16_TIER1 = 1 << 19
_F16_TIER2 = 1 << 23
_F16_TIER3 = 1 << 27
_F32_TIER1 = 1 << 19
_F32_TIER2 = 1 << 22
_NPROGS = 48
_GRID_LIMIT = 65535
_INT32_SAFE = 0x7FFFFFFF


def _launch_with_retry(launch):
    # Flaky external toolchain (fresh-compile segv windows on the shared
    # node): retry compile-time failures only, re-raise the original error
    # once exhausted so real failures are never masked.
    for attempt in range(6):
        try:
            launch()
            return
        except Exception:
            if attempt == 5:
                raise
            time.sleep(1.0)


@triton.jit
def _frexp_body(x):
    x32 = x.to(tl.float32)
    bits = x32.to(tl.int32, bitcast=True)
    payload = bits & 0x7FFFFFFF
    biased = (bits >> 23) & 0xFF
    is_sub = biased == 0
    # Special values (zero / inf / NaN) via float-domain arithmetic only:
    # integer bit compares on NaN-derived bits are untrustworthy on this
    # backend (compiler_traps "integer-domain NaN bit predicate"), and `x != x` is folded
    # to False in full kernel contexts (compiler_traps retest regression,
    # generic frexp returned (~1.0, 129) for NaN on 2026-11-13).
    is_special = (x32 == 0.0) | ((x32 - x32) != 0.0)
    abs_x = payload.to(tl.float32, bitcast=True)
    scaled = abs_x * 85070591730234615865843651857942052864.0
    sbits = scaled.to(tl.int32, bitcast=True)
    s_biased = (sbits >> 23) & 0xFF
    s_mant = (sbits & 0x007FFFFF) | 0x3F000000
    mant_bits = tl.where(is_sub, s_mant, (bits & 0x007FFFFF) | 0x3F000000)
    abs_m = mant_bits.to(tl.float32, bitcast=True)
    mantissa = tl.where(bits < 0, -abs_m, abs_m)
    mantissa = tl.where(is_special, x32, mantissa)
    exponent = tl.where(is_sub, s_biased - 252, biased - 126)
    exponent = tl.where(is_special, 0, exponent)
    return mantissa.to(x.dtype), exponent


@triton.jit(do_not_specialize=["N"])
def _frexp_flat_kernel(X, M, E, N, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        x = tl.load(X + offs)
        m, e = _frexp_body(x)
        tl.store(M + offs, m.to(M.dtype.element_ty))
        tl.store(E + offs, e)
    else:
        mask = offs < N
        x = tl.load(X + offs, mask=mask, other=0.0)
        m, e = _frexp_body(x)
        tl.store(M + offs, m.to(M.dtype.element_ty), mask=mask)
        tl.store(E + offs, e, mask=mask)


@triton.jit(do_not_specialize=["N"])
def _frexp_strided_kernel(X, M, E, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    for tile in tl.range(pid, tl.cdiv(N, BLOCK), num_ctas):
        off = tile * BLOCK + tl.arange(0, BLOCK)
        mask = off < N
        x = tl.load(X + off, mask=mask, other=0.0)
        m, e = _frexp_body(x)
        tl.store(M + off, m.to(M.dtype.element_ty), mask=mask)
        tl.store(E + off, e, mask=mask)


def frexp(A):
    logger.debug("GEMS_ENFLAME FREXP")
    if not A.is_floating_point():
        raise RuntimeError(
            f"frexp(): expected a floating-point tensor, but got {A.dtype}"
        )
    if A.dtype == torch.float64:
        raise RuntimeError("FlagGems frexp currently does not support float64")

    mantissa = torch.empty_like(A)
    exponent = torch.empty_like(A, dtype=torch.int32)
    n = A.numel()
    if (
        A.device.type == device
        and A.dtype in _FAST_DTYPES
        and A.is_contiguous()
        and 0 < n < _INT32_SAFE
    ):
        if A.dtype == torch.float32:
            if n < _SMALL_N:
                block, nw, strided = max(16, triton.next_power_of_2(n)), 2, False
            elif n < _F32_TIER1:
                block, nw, strided = 16384, 2, False
            elif n < _F32_TIER2:
                block, nw, strided = 32768, 2, False
            else:
                block, nw, strided = 32768, 2, True
        else:
            if n < _SMALL_N:
                block, nw, strided = max(16, triton.next_power_of_2(n)), 2, False
            elif n < _F16_TIER1:
                block, nw, strided = 16384, 4, False
            elif n < _F16_TIER2:
                block, nw, strided = 32768, 2, False
            elif n < _F16_TIER3:
                block, nw, strided = 65536, 2, False
            else:
                block, nw, strided = 131072, 2, True
        if strided:
            _launch_with_retry(
                lambda: _frexp_strided_kernel[(_NPROGS,)](
                    A, mantissa, exponent, n, BLOCK=block, num_warps=nw
                )
            )
            return mantissa, exponent
        grid = (triton.cdiv(n, block),)
        if grid[0] <= _GRID_LIMIT:
            _launch_with_retry(
                lambda: _frexp_flat_kernel[grid](
                    A,
                    mantissa,
                    exponent,
                    n,
                    BLOCK=block,
                    EVEN=(n % block == 0),
                    num_warps=nw,
                )
            )
            return mantissa, exponent
    return _frexp_func(A, out0=mantissa, out1=exponent)
