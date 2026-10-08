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

# Kunlunxin( XPU ) erfcx(x) = exp(x^2) * erfc(x).
#
# Generic flag_gems/ops/special_erfcx.py relies on tl_extra_shim.erfcx, which
# fails to link on XPU ("ld.lld: error: undefined symbol: Unsupported",
# referenced from the pointwise kernel) -- same family as lgamma/log1p/pow.
# This override implements erfcx in pure fp32 with linkable primitives only
# (mul/div/add/exp/where), and, crucially, reproduces the torch CPU fp32
# reference bit-for-bit on the negative half axis (see note below).
#
# Math:
#   x >= 0:
#     x in [0, X1=1.5] : erfcx ~= P(x),  Chebyshev(u) deg-10, u = 2x/1.5 - 1,
#                        fp32 Horner max REL err 1.0e-7
#     x >= 1.5         : erfcx = g(t) / (x*sqrt(pi)), t = 1/x^2 in [0, 4/9],
#                        g Chebyshev(v) deg-13, v = 2t/(4/9) - 1,
#                        fp32 Horner max REL err 2.3e-8 (covers the whole
#                        positive tail incl. +inf via t -> 0)
#   x < 0 :  erfcx(x) = 2*exp(x^2) - erfcx(|x|).
#
#   Numerical note on the negative half.  torch's CPU fp32 reference computes
#   2*expf(x*x) - erfcx(|x|) with the *same* fp32-rounded x*x that is fed to
#   expf (verified: at x=-8.7 the torch CPU fp32 reference deviates ~1.8e-6
#   relative from the exact value -- exactly the exp argument-quantization
#   amplification x^2*2^-24).  Any implementation that is *more* accurate
#   than the reference here (e.g. fma-compensated exp(x^2)) breaks the test
#   tolerance (atol 1e-4 + rtol 1.3e-6) at |x| ~> 4.7 even though it is
#   closer to the exact math.  We therefore reproduce the reference chain
#   identically: y = 2*exp(x*x) - r(|x|) with the same fp32 x*x and the same
#   correctly rounded exp (probed: tl.exp is bit-exact with torch.exp across
#   the fp32 domain), making the negative half bit-for-bit equal to torch.
#   Corners: x=-inf -> +inf, x=+inf -> 0, NaN -> NaN, +-0 -> 1 (verified
#   on device against torch).

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# CodeGenConfig tuned on XPU for the erf-family pointwise kernels
MIN_BLOCK = 2048
MAX_BLOCK = 131072
UNROLL_NUM = 16
BUFFER_SIZE_LIMIT = 8192
IS_CLOSE_MEMORY_ASYNC = False


def _pick_block(n_elements):
    # Bucket the tile into one of 3 unmasked sizes + 1 masked fallback so the
    # kernel compiles at most ~4 times total. Unmasked runs when the shape
    # divides the tile exactly (masked memory path on XPU costs ~2x).
    if n_elements >= 1_048_576 and n_elements % MAX_BLOCK == 0:
        return MAX_BLOCK, 32, False
    if n_elements >= 262_144 and n_elements % 32768 == 0:
        return 32768, 8, False
    if n_elements >= 16384 and n_elements % 16384 == 0:
        return 16384, 8, False
    if n_elements <= 65536:
        return 2048, 4, True
    return 16384, 8, True


@triton.jit
def _erfcx(x):
    # |x| (x == +-0: |x| = 0 handled by region A)
    a = tl.where(x < 0, -x, x)
    # ---- Region A: x in [0, 1.5], Chebyshev-deg10 in u = 2x/1.5 - 1 ----
    u = a * 1.3333333730697632 - 1.0
    p = 2.2037075e-05
    p = p * u + -8.535864e-05
    p = p * u + 0.00024344234
    p = p * u + -0.0007935578
    p = p * u + 0.002542182
    p = p * u + -0.0075751324
    p = p * u + 0.02113351
    p = p * u + -0.05477366
    p = p * u + 0.12991387
    p = p * u + -0.27597955
    p = p * u + 0.5069376
    # ---- Region B: x >= 1.5, g(t) = x*sqrt(pi)*erfcx(x), t = 1/x^2 ----
    t = 1.0 / (a * a)
    v = t * 4.5 - 1.0
    g = -3.353409e-06
    g = g * v + 5.448586e-06
    g = g * v + 1.5268429e-06
    g = g * v + -4.3207035e-07
    g = g * v + -1.6099795e-05
    g = g * v + 3.10895e-05
    g = g * v + -5.732727e-05
    g = g * v + 0.00013798248
    g = g * v + -0.0003571754
    g = g * v + 0.0010035556
    g = g * v + -0.0032166068
    g = g * v + 0.012547941
    g = g * v + -0.06885448
    g = g * v + 0.9137709
    r_pos = tl.where(a <= 1.5, p, g * 0.5641895835477563 / a)
    # ---- x < 0: erfcx(x) = 2*exp(x^2) - erfcx(|x|); same fp32 chain
    #      (x*x rounded, correctly rounded exp) as the torch CPU reference
    e = tl.exp(x * x)
    return tl.where(x < 0, 2.0 * e - r_pos, r_pos)


@triton.jit
def special_erfcx_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    x = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    y = _erfcx(x)
    tl.store(out_ptr + offset, y.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def special_erfcx_kernel_unmasked(x_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offset).to(tl.float32)
    y = _erfcx(x)
    tl.store(out_ptr + offset, y.to(out_ptr.dtype.element_ty))


def _launch(x, out):
    n_elements = x.numel()
    if n_elements == 0:
        return
    block_size, num_warps, masked = _pick_block(n_elements)
    extra = dict(
        unroll_num=UNROLL_NUM,
        buffer_size_limit=BUFFER_SIZE_LIMIT,
        isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
    )
    if masked:
        grid = (triton.cdiv(n_elements, block_size),)
        special_erfcx_kernel[grid](
            x,
            out,
            n_elements,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            **extra,
        )
    else:
        grid = (n_elements // block_size,)
        special_erfcx_kernel_unmasked[grid](
            x,
            out,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            **extra,
        )


def special_erfcx(x):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_ERFCX")
    assert x.dtype == torch.float32, "special_erfcx only supports float32"
    x = x.contiguous()
    out = torch.empty_like(x)
    _launch(x, out)
    return out
