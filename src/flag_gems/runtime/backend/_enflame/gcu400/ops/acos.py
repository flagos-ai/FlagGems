# Copyright 2026, The FlagOS Contributors.
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
#
# Enflame GCU400 fast path for aten::acos (out-of-place). Generic reference:
# flag_gems/ops/acos.py (libdevice extern call); sibling template with the
# same half-angle machinery and the measured cost buckets:
# gcu400/ops/arcsin.py.
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, pointwise_dynamic, tl_extra_shim

_ACOS = tl_extra_shim.acos
logger = logging.getLogger(__name__)

NUM_SIPS = 24
FLAT_GRID = NUM_SIPS * 2

PI = tl.constexpr(3.141592653589793)
PI_OVER_2 = tl.constexpr(1.5707963705062866)  # fp32(π/2)
NEG_PI_OVER_2 = tl.constexpr(-1.5707963705062866)
PI16 = tl.constexpr(3.140625)  # fp16/bf16-representable π (err 9.7e-4)

# acos(|x|) = 2*asin(w),  w = sqrt((1-|x|)/2) in [0, 1/sqrt2],
# u = w^2 = 0.5*(1-|x|) in [0, 0.5].  With arcsin's 2*P coefficient folding
# (q = 2*P) the whole domain is one Horner chain on the tightest range:
#   acos(|x|) = w * (2 + u * q(u))   ->   r in [0, π/2],
# followed by the quadrant fixup  acos(x) = x >= 0 ? r : π - r  (acos is
# neither odd nor even, unlike the arcsin sibling).
#
# q chains (minimax on u in [0, 0.5]):
#   H3 (3 fma): max arm err 1.16e-5 -> fp32 (8.5x inside the atol=1e-4 gate)
#               and fp16 (far below one 16-bit ulp);
#   L2 (2 fma): max arm err 1.88e-3 -> bf16 only (rtol 0.016 gate, 13x
#               margin); blocked for fp16 (1.88e-3 > 1.67e-3 tol at π/2).
H0 = tl.constexpr(0.3333333332)
H1 = tl.constexpr(0.15667759974)
H2 = tl.constexpr(0.03776437846)
H3 = tl.constexpr(0.17391203516)

L0 = tl.constexpr(0.3280023162944349)
L1 = tl.constexpr(0.21911999999999998)

# Quadrant-fixup variants, picked per dtype (512M-element medians, GCU400):
#   4 "refloat" (fp32): d = fma(w, t, -π/2); s = where(x<0, -d, d); y = π/2+s.
#     The sign-flip select keeps a single live f32 vector; the plain
#     two-arm select(x<0, π-r, r) measured +250us slower (2222 -> 1973us
#     at F4; 2036 for the parallel-arm fold variant, both vetoed).  With
#     the H3 chain this lands at 1800us = 2.48x native (official bench).
#   2 "per-arm cvt" (fp16): where(x<0, (π-r).to16, r.to16).  Bit-identical
#     to the f32 select but the f32 vectors die at their cvts; the 16-bit
#     π16-r16 variant is 196us faster but violates the fp16 rtol gate
#     (measured 2.5% violations on a dense sweep: the π16 constant error
#     9.7e-4 is not recoverable in fp16 granularity).
#   1 "16-bit sub" (bf16): where(x<0, π16-r16, r16).  The π16 constant error
#     is harmless under bf16's 0.016 rtol (max err/tol 0.48); cheapest arm.
_QUAD_REFLOAT = 4
_QUAD_PERARM = 2
_QUAD16_SUB = 1

# dtype -> (CHAIN, QUAD16): fp32 = H3 + refloat, fp16 = H3 + per-arm cvt,
# bf16 = L2 + 16-bit sub.
_DTYPE_PLAN = {
    torch.float32: (1, _QUAD_REFLOAT),
    torch.float16: (1, _QUAD_PERARM),
    torch.bfloat16: (2, _QUAD16_SUB),
}


@libentry()
@triton.jit(do_not_specialize=["N"])
def acos_flat_kernel(
    X,
    Y,
    N,
    BLOCK: tl.constexpr,
    HAS_TAIL: tl.constexpr,
    CHAIN: tl.constexpr,
    QUAD16: tl.constexpr,
):
    """acos via the unified half-angle identity: one sqrt + one Horner chain
    over the whole domain, no domain split; NaN for |x|>1 falls out of
    sqrt(negative) and propagates through the quadrant fixup."""
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
        # |x| > 1 -> u < 0 -> sqrt yields NaN and flows through the chain
        u = tl.fma(-0.5, ax, 0.5)
        w = tl.sqrt(u)
        if CHAIN == 2:
            q = L1
            q = q * u + L0
        else:
            q = H3
            q = q * u + H2
            q = q * u + H1
            q = q * u + H0
        t = tl.fma(u, q, 2.0)
        if QUAD16 == 1:
            r16 = (w * t).to(Y.dtype.element_ty)
            y = tl.where(x < 0.0, PI16 - r16, r16)
        elif QUAD16 == 2:
            ty = Y.dtype.element_ty
            r = w * t
            y = tl.where(x < 0.0, (PI - r).to(ty), r.to(ty))
        else:
            d = tl.fma(w, t, NEG_PI_OVER_2)
            s = tl.where(x < 0.0, -d, d)
            y = PI_OVER_2 + s
        if HAS_TAIL:
            tl.store(Y + offsets, y.to(Y.dtype.element_ty), mask=mask)
        else:
            tl.store(Y + offsets, y.to(Y.dtype.element_ty))


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit()
def acos_kernel(x):
    # Strided / non-float / >2^31-element fallback: match torch.acos by
    # evaluating in float32 before casting back.
    return _ACOS(x.to(tl.float32))


def _flat_ready(A):
    return (
        A.is_contiguous()
        and 0 < A.numel() < 2**31
        and A.dtype in (torch.float32, torch.float16, torch.bfloat16)
    )


def _flat_config(n, itemsize):
    # Block caps are DSM-bound (256KB tiles; doubling exceeds max_dsm and
    # fails the compile).  Ladder re-scanned after the quadrant/chain
    # restructure (GCU400, official bench kernel medians, w1000/r100):
    # - fp32 tops at B65536/nw2 (1800us = 2.48x); nw1 matches in isolated
    #   sweeps but regresses +29% in the official bench context (2325us)
    #   -- nw2 is placement-stable, nw1 is not;
    # - 16-bit tops at B131072/nw2 (fp16 2009us = 2.32x, bf16 1331us =
    #   3.51x), 16M bucket B32768/nw2 (fp16 74us, bf16 52us);
    # - 4096-element shapes sit on the ~3.4-3.8us launch floor at B1024/nw1.
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
    chain, quad16 = _DTYPE_PLAN[A.dtype]
    grid = (min(triton.cdiv(n, block), FLAT_GRID),)
    with torch_device_fn.device(A.device):
        acos_flat_kernel[grid](
            A.view(-1),
            out.view(-1),
            n,
            BLOCK=block,
            HAS_TAIL=(n % block) != 0,
            CHAIN=chain,
            QUAD16=quad16,
            num_warps=num_warps,
        )


def acos(A):
    logger.debug("GEMS_ENFLAME ACOS")
    if _flat_ready(A):
        out = torch.empty_like(A)
        _flat_launch(A, out)
        return out
    return acos_kernel(A)
