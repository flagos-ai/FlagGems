"""poisson candidate v3: two independent philox chains per Knuth chunk.

Successor of the v2 kernel (block-max Knuth bound, 4 uniforms per philox
call, uint32 counters).  v3 keeps the sampler contract -- per-element philox
streams with ``STRIDE``-spaced counters, truncation probability <= 1e-5 on the
small path -- and spends the same op budget more efficiently:

* uniforms come from the top 24 bits of the philox word (``(r >> 8) * 2**-24``).
  Three cheap vector ops replace the five-op ``uint_to_uniform_float`` chain
  (bitcast + where + mul); 24 random mantissa bits are plenty for a product of
  at most 16 uniforms and the distribution stays exactly uniform on
  ``[0, 1)``.  Only the drawn values change, not the distribution;
* the ``max(u, 1e-10)`` clamp is gone.  If u == 0 then p == 0 <= L and k
  freezes -- which is precisely the Knuth semantics.  The clamp only made a
  difference for lam > 23 *and* u == 0 (probability ~1e-9), so results are
  unchanged for every realistic input;
* the loop bound is capped at the small-path ceiling ``min(bmax, 30)`` before
  the 3-sigma margin.  Lanes with lam > 30 discard their Knuth result in
  favour of the normal approximation anyway, so the loop only has to cover
  lanes that actually read it (nit 17 -> 13 for an all-lam-50 block);
* the loop processes two philox calls per iteration (8 uniforms).  The two
  chains depend only on their own counters, so the compiler can overlap them,
  and the consumption order is unchanged, so k is bit-identical to the
  sequential loop.  Over-drawing to a multiple of 8 is safe: p is monotone
  non-increasing, so once p <= L no later uniform can bump k.  Every element
  owns a 64-counter slot (``STRIDE``), and the widened loop consumes at most
  ``2 * ceil(17 / 2) + 1 = 19 < 64`` counters of it;
* one BLOCK bucket for every dtype: 512 for N <= 16384 and 4096 above.  The
  16-bit-specific 2048 bucket from v2 is gone -- with the 24-bit conversion
  the fp16/bf16 widen hazard no longer penalises 4096, and 4096 is faster
  across the board (1M fp16: 78.9 -> 66.4 us);
* the grid must be SIP-aligned.  24 CTAs on 24 SIPs beat 48 (one program per
  SIP gets the full DSM) and beat raw intermediate counts badly: 32 CTAs on
  24 SIPs leaves 8 SIPs with two programs and 16 with one, which measured
  ~97 us vs ~67 us at 1M fp32.  The launcher therefore uses ``cdiv`` when it
  fits on one CTA per SIP and clamps to ``NUM_SIPS`` otherwise.

Measured (op-level do_bench median, warmup 1000ms / rep 100ms, gcu400):

    fp32 1M     74.2 -> 67.7 us   (u24+noclamp+g24)
    fp16 1M     87.1 -> 66.4 us   (BLOCK 2048 -> 4096 dominates)
    bf16 1M                66.7 us
    fp32 65K     9.1 ->  8.8 us
    fp16 65K    12.2 ->  8.7 us   (BLOCK 2048 g32 -> 4096 g16)
    fp32 4K      4.8 ->  4.8 us   (launch bound)

The 64-bit-counter (WIDE) path, the increment accounting and the normal
approximation for lam > 30 are unchanged from v2; the normal approximation
now also draws 24-bit uniforms, which the mean/variance tolerance swallows.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._enflame.gcu400.ops.poisson import NUM_SIPS
from flag_gems.utils import libentry
from flag_gems.utils.random_utils import philox_backend_seed_offset
from flag_gems.utils.shape_utils import volume

logger = logging.getLogger(__name__)

__all__ = ["poisson"]

_BLOCK_SMALL = 512
_BLOCK_LARGE = 4096
_SMALL_N = 16384
_LAMBDA_THRESHOLD = 30.0
# Cap on uniforms per element (hard limit of the Knuth loop).  lam < 30 needs
# at most ~49 uniforms, so the cap never binds on the small-lambda path.
_MAX_UNIFORMS = 64
_PHILOX_STRIDE = 64
# off * STRIDE must stay inside uint32 for the fast counter path.
_U32_COUNTER_MAX_ELEMS = 2**32 // _PHILOX_STRIDE


@triton.jit
def _u24(r):
    """Uniform float in [0, 1): top 24 bits of a philox word."""
    return (r >> 8).to(tl.float32) * 5.960464477539063e-08


@triton.jit
def _poisson_large(lam, seed, c0, c1, z):
    """Normal approximation (Box-Muller), used only for lam > threshold."""
    r0, r1, r2, r3 = tl.philox(seed, c0, c1, z, z)
    u1 = tl.maximum(_u24(r0), 1e-10)
    u2 = _u24(r1)
    two_pi = 6.283185307179586
    radius = tl.sqrt(-2.0 * tl.log(u1))
    result = lam + tl.sqrt(lam) * (radius * tl.cos(two_pi * u2))
    return tl.floor(tl.maximum(result, 0.0) + 0.5)


@libentry()
@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "N"])
def poisson_kernel_dyn(
    inp_ptr,
    out_ptr,
    N,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
    LAMBDA_THRESHOLD: tl.constexpr,
    MAX_UNIFORMS: tl.constexpr,
    STRIDE: tl.constexpr,
    WIDE: tl.constexpr,
):
    seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0_base = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    pid = tl.program_id(0)
    npid = tl.num_programs(0)
    ar = tl.arange(0, BLOCK)
    for blk in tl.range(pid, (N + BLOCK - 1) // BLOCK, npid):
        off = blk * BLOCK + ar
        mask = off < N
        lam = tl.maximum(
            tl.load(inp_ptr + off, mask=mask, other=0.0).to(tl.float32), 0.0
        )
        bmax = tl.max(lam)
        # Knuth needs N ~ 1 + Poisson(lam) uniforms; three sigma covers the
        # tail.  Lanes above the threshold never read their Knuth result, so
        # the bound only has to cover the small-path ceiling of the block max.
        bl = tl.minimum(bmax, LAMBDA_THRESHOLD * 1.0)
        need = tl.minimum(bl + 3.0 * tl.sqrt(bl) + 3.0, MAX_UNIFORMS * 1.0)
        nit = (need * 0.25).to(tl.int32) + 1
        n2 = (nit + 1) // 2

        if WIDE:
            counter = c0_base.to(tl.int64) + off.to(tl.int64) * STRIDE
            c0 = (counter & 0xFFFFFFFF).to(tl.uint32)
            c1v = ((counter >> 32) & 0xFFFFFFFF).to(tl.uint32)
        else:
            # uint32 keeps the per-element counter arithmetic off the (emulated)
            # int64 path; off * STRIDE cannot wrap because WIDE is False.
            c0 = c0_base + off.to(tl.uint32) * STRIDE
            c1v = c1
        z = c0 * 0

        L = tl.exp(-lam)
        p = lam * 0.0 + 1.0
        k = (lam * 0).to(tl.int32)
        for _ in tl.range(0, n2):
            # Two independent philox chains, consumed in stream order, so k
            # is bit-identical to the sequential loop.  Over-draw to a
            # multiple of 8 uniforms is safe: p is monotone non-increasing,
            # once p <= L no later uniform can raise k.
            ra0, ra1, ra2, ra3 = tl.philox(seed, c0, c1v, z, z)
            rb0, rb1, rb2, rb3 = tl.philox(seed, c0 + 1, c1v, z, z)
            c0 = c0 + 2
            pp = p * _u24(ra0)
            k = tl.where(pp > L, k + 1, k)
            p = pp
            pp = p * _u24(ra1)
            k = tl.where(pp > L, k + 1, k)
            p = pp
            pp = p * _u24(ra2)
            k = tl.where(pp > L, k + 1, k)
            p = pp
            pp = p * _u24(ra3)
            k = tl.where(pp > L, k + 1, k)
            p = pp
            pp = p * _u24(rb0)
            k = tl.where(pp > L, k + 1, k)
            p = pp
            pp = p * _u24(rb1)
            k = tl.where(pp > L, k + 1, k)
            p = pp
            pp = p * _u24(rb2)
            k = tl.where(pp > L, k + 1, k)
            p = pp
            pp = p * _u24(rb3)
            k = tl.where(pp > L, k + 1, k)
            p = pp

        result = k.to(tl.float32)
        if bmax > LAMBDA_THRESHOLD:
            large = _poisson_large(lam, seed, c0, c1v, z)
            result = tl.where(lam <= LAMBDA_THRESHOLD, result, large)
        tl.store(out_ptr + off, result, mask=mask)


def poisson(input, generator=None):
    logger.debug("GEMS_ENFLAME POISSON")
    assert input.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ), f"Unsupported dtype: {input.dtype}"

    inp = input.contiguous()
    N = volume(inp.shape)
    out = torch.empty_like(inp)
    if N == 0:
        return out

    block = _BLOCK_SMALL if N <= _SMALL_N else _BLOCK_LARGE
    cdiv = (N + block - 1) // block
    # GCU400 wants SIP-aligned grids: one program per SIP beats two, and
    # intermediate counts leave some SIPs with two programs and others with
    # one (~45% slower at 1M fp32).  Small launches fit inside NUM_SIPS.
    grid = cdiv if cdiv <= NUM_SIPS else NUM_SIPS
    increment = triton.cdiv(N * _PHILOX_STRIDE, 4)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )
    with torch_device_fn.device(inp.device):
        poisson_kernel_dyn[(grid,)](
            inp,
            out,
            N,
            philox_seed,
            philox_offset,
            BLOCK=block,
            LAMBDA_THRESHOLD=_LAMBDA_THRESHOLD,
            MAX_UNIFORMS=_MAX_UNIFORMS,
            STRIDE=_PHILOX_STRIDE,
            WIDE=N > _U32_COUNTER_MAX_ELEMS,
            num_warps=4,
        )
    return out
