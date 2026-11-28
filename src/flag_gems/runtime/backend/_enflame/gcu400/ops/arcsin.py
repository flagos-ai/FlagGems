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

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, pointwise_dynamic, tl_extra_shim

_ASIN = tl_extra_shim.asin
logger = logging.getLogger(__name__)

NUM_SIPS = 24
FLAT_GRID = NUM_SIPS * 2

PI_OVER_2 = tl.constexpr(1.5707963705062866)

# asin(w) = w + w^3 * P(w^2), w = sqrt((1 - a) / 2) in [0, 1/sqrt2],
# u = w^2 = 0.5 * (1 - a) in [0, 0.5], a = |x|.
# Fold the coefficients of 2*asin(w) into the polynomial (q = 2*P, so F/H are twice the unfolded version):
#   arcsin(a) = pi/2 - w * (2 + u * q(u))
# u is also the Horner variable, saving one w*w; two tl.fma calls eliminate the multiply-subtract chain.
# fp32 uses a 4-term chain (max|err| = 1.6e-6, 62x margin at the atol=1e-4 gate);
# 16-bit uses a 3-term chain (max|err| ≈ 1.2e-5, far below the rounding of the 16-bit output itself).
F0 = tl.constexpr(0.3333333334)
F1 = tl.constexpr(0.14794641546)
F2 = tl.constexpr(0.11325906374)
F3 = tl.constexpr(-0.0335772589)
F4 = tl.constexpr(0.18328779898)

H0 = tl.constexpr(0.3333333332)
H1 = tl.constexpr(0.15667759974)
H2 = tl.constexpr(0.03776437846)
H3 = tl.constexpr(0.17391203516)


@libentry()
@triton.jit(do_not_specialize=["N"])
def _arcsin_flat_kernel(
    X,
    Y,
    N,
    BLOCK: tl.constexpr,
    HAS_TAIL: tl.constexpr,
    HIGH_PREC: tl.constexpr,
    SIGN16: tl.constexpr,
):
    """asin is solved with a unified half-angle identity: one Horner chain over the whole domain,

    The identity (holds for every a in [0, 1]):
        asin(a) = pi/2 - 2 * asin(w),  w = sqrt((1 - a) / 2) in [0, 1/sqrt2]
    a is |x|, and one sqrt plus one asin(w) Horner chain covers the entire domain. Folding 2*P into
    the coefficients gives r = pi/2 - w * (2 + u*q(u)), reusing u = w^2 directly (saving one
    square), with `tl.fma` removing the u multiply-subtract and the merge multiply-add. The minimax
    fit covers w in [0, 0.7071] (u in [0, 0.5]), see the coefficient comments above.

    Sign fix: arcsin is odd, so y = where(x<0, -r, r). For 16-bit output the select moves after
    the precision reduction (r.to(16bit) then where +-r16): the L600 ablation measured the
    f32-domain select double-live interval at ~590us on 512Mi fp16 versus ~237us in the 16-bit
    domain (fsel vs 16sel: 1777 vs 2133us); fp32 shows no difference (select ~37us), so it stays
    in place.

    For |x| > 1, u < 0 and sqrt of a negative propagates NaN naturally, so no branch is needed;
    NaN / +-inf inputs behave the same. Cancellation only happens as |x| -> 0, where the error is
    the polynomial's absolute error, an order of magnitude below the atol = 1e-4 gate.

    GCU400/L600 measurement (this board, grid cap 48, benchmark/test_arcsin.py official kernel
    mode --level core, two re-sampled rounds, do_bench w1000/r100 median, 45-case mean speedup
    2.24x -> 2.48x):
    - 512Mi elements: fp32 ~1855us (2.61x native, copy floor 1355us), 16-bit ~1776us (2.82x, floor
      671us);
    - 16Mi elements: fp32 ~67.3us (2.60x), 16-bit ~67.0us (2.62x);
    - (64,64) 4096 elements: ~3.7us (BLOCK 1024/nw1, the old 2048/nw2 ladder gave 4.1us).
    Per-case drift between rounds <=0.4% (16Mi and above). 16-bit large shapes are still 2.6x off
    the copy floor: the remainder is abs/fma/sqrt (247us) + Horner 3FMA + the merge ALU serial
    chain, and select / constant-multiply / integer-bit-or sign variants all measured worse than
    the 16-bit-domain select (see the case log).
    """
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    if HAS_TAIL:
        num_blocks = (N + BLOCK - 1) // BLOCK
    else:
        num_blocks = N // BLOCK
    for block in tl.range(pid, num_blocks, num_pids):
        offsets = block * BLOCK + tl.arange(0, BLOCK)
        if HAS_TAIL:
            mask = offsets < N
            x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
        else:
            x = tl.load(X + offsets).to(tl.float32)
        ax = tl.abs(x)
        # When |x| > 1, u < 0, sqrt produces NaN and propagates it along the chain
        u = tl.fma(-0.5, ax, 0.5)
        w = tl.sqrt(u)
        if HIGH_PREC:
            q = F4
            q = q * u + F3
            q = q * u + F2
            q = q * u + F1
            q = q * u + F0
        else:
            q = H3
            q = q * u + H2
            q = q * u + H1
            q = q * u + H0
        r = PI_OVER_2 - w * tl.fma(u, q, 2.0)
        if SIGN16:
            # Reduce precision first, then fix the sign: a dual-active-interval select in the f32 domain is 3x more
            # expensive at the 16-bit setting
            r16 = r.to(Y.dtype.element_ty)
            y = tl.where(x < 0.0, -r16, r16)
        else:
            y = tl.where(x < 0.0, -r, r)
        if HAS_TAIL:
            tl.store(Y + offsets, y.to(Y.dtype.element_ty), mask=mask)
        else:
            tl.store(Y + offsets, y.to(Y.dtype.element_ty))


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit()
def arcsin_kernel(x):
    # Match torch.arcsin by evaluating in float32 before casting back.
    return _ASIN(x.to(tl.float32))


def _flat_ready(A):
    return (
        A.is_contiguous()
        and 0 < A.numel() < 2**31
        and A.dtype in (torch.float32, torch.float16, torch.bfloat16)
    )


def _flat_config(n, itemsize):
    # Tiers measured on L600 (grid capped at 48, do_bench w1000/r100 median):
    # - Small shapes BLOCK 1024/nw1: 4096 elements 3.7us, old 2048/nw2 was 4.1us;
    # - Large tier: 16-bit B131072/nw2 and fp32 B65536/nw2 are ablation-optimal (nw1/nw4 are both worse),
    #   16-bit mid tier 32768/nw2 is 7-9% faster than nw1;
    # - The block cap is limited by dsm (both 16-bit 131072 and fp32 65536 are 256KB tiles,
    #   doubling again exceeds max_dsm and compilation fails outright).
    if itemsize == 4:
        if n <= 2**17:
            return 1024, 1
        if n <= 2**21:
            return 4096, 2
        if n <= 2**25:
            return 32768, 2
        return 65536, 2
    if n <= 2**13:
        return 1024, 1
    if n <= 2**21:
        return 4096, 2
    if n <= 2**25:
        return 32768, 2
    return 131072, 2


def _flat_launch(A, out):
    n = A.numel()
    block, num_warps = _flat_config(n, A.element_size())
    grid = (min(triton.cdiv(n, block), FLAT_GRID),)
    with torch_device_fn.device(A.device):
        _arcsin_flat_kernel[grid](
            A.view(-1),
            out.view(-1),
            n,
            BLOCK=block,
            HAS_TAIL=(n % block) != 0,
            HIGH_PREC=(A.dtype == torch.float32),
            SIGN16=(A.dtype != torch.float32),
            num_warps=num_warps,
        )


def arcsin(x, *, out=None):
    logger.debug("GEMS_ENFLAME ARCSIN_FORWARD")
    # ATen rejects an integer or bool out buffer (the float result cannot be
    # cast into it); mirror that instead of silently truncating.
    if out is not None and not out.dtype.is_floating_point:
        raise RuntimeError(
            "result type Float can't be cast to the desired output type " f"{out.dtype}"
        )
    if out is None:
        if _flat_ready(x):
            out = torch.empty_like(x)
            _flat_launch(x, out)
            return out
        return arcsin_kernel(x)
    if (
        _flat_ready(x)
        and out.shape == x.shape
        and out.dtype == x.dtype
        and out.is_contiguous()
    ):
        _flat_launch(x, out)
        return out
    arcsin_kernel(x, out0=out)
    return out


def arcsin_(x):
    logger.debug("GEMS_ENFLAME ARCSIN_INPLACE")
    if _flat_ready(x):
        _flat_launch(x, x)
        return x
    arcsin_kernel(x, out0=x)
    return x


def arcsin_out(x, *, out=None):
    logger.debug("GEMS_ENFLAME ARCSIN_OUT")
    return arcsin(x, out=out)
