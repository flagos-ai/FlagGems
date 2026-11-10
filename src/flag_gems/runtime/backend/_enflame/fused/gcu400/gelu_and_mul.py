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

"""gcu400 GELU-and-mul with flat linear kernels.

The previous implementation used the backend @pointwise_dynamic codegen.
Although it has no multi-dimensional index decomposition problem (2D and 3D
inputs of the same size cost the same), its generated wrapper still costs a
lot: 64x64 21.9us -> 4.6us and 16.7M elements 679us -> 201us with a flat
one-dimensional kernel.

The launch now follows the gcu400 "fixed grid + tl.range(num_stages)" family
(deg2rad/fmax/log1p): a small fixed CTA grid (48, or 24 for 1e9-element 2-byte
inputs) over a pipelined tile loop, with the tile size chosen per dtype and
size.  On top of that the exact-erf mode evaluates erf through a fitted
logistic polynomial instead of libdevice erff: libdevice erf costs ~8.4 ms of
the 12.3 ms for 1e9 bf16 elements here, the polynomial costs ~1.1 ms.  Max
|gelu error| of the approximation is 3.1e-6 over the whole real line (~3% of
the fp32 test tolerance); the backward pass keeps libdevice erf so the
gradients stay exact.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry, tl_extra_shim

erf = tl_extra_shim.erf
logger = logging.getLogger(__name__)

BLOCK_SIZE = 4096
MAX_GRID_SIZE = 96

# Tuned on GCU400 for the forward kernel (sweep in the commit message).  Two
# size buckets pick the tile and the CTA count:
#  - 16.7M elements: 8K tile, 48 CTAs.  Bigger tiles are ~7% faster in the best
#    case but alternate between 0.082 and 0.13 ms from one launch to the next;
#    8K is flat 0.088 across repeats.  That bimodality is in the memory
#    staging (a pure load/mul/store kernel shows it too), not the gelu math.
#  - 1e9 elements: 32K tile, 24 CTAs (one per SIP).  An 8K tile is much worse
#    here (7.3 ms vs 5.2 on the first big allocation of a process).
#  - fp32: 32K tile, 48 CTAs, num_warps=4, in both buckets.
_GRID_SIZE = 48
_GRID_BIG = 24
_TILE_SMALL = 8192
_TILE_FP32 = 32768
_TILE_BIG = 32768
_SMALL_N = 32768
_BIG_N = 1 << 28


# logistic sigmoid in the backend's fast-exp form (see _gelu_tanh_sigmoid)
@triton.jit
def _gelu_sigmoid(z):
    return tl.fdiv(1.0, 1.0 + tl.exp(-z))


# tanh-GELU in its sigmoid form: 0.5*x*(1+tanh(u)) == x*sigmoid(2u).  libdevice
# tanh is one of the most expensive ops in this kernel on GCU (~8 ms of the
# 11.8 ms for 1e9 elements), while tl.exp + tl.fdiv use the backend's fast exp
# path.  Saturation: exp(-2u) -> 0 gives s = 1 (x), exp(-2u) -> inf gives s = 0.
@triton.jit
def _gelu_tanh_sigmoid(u):
    return tl.fdiv(1.0, 1.0 + tl.exp(-2.0 * u))


# Exact-erf GELU through a logistic form of the normal CDF:
#   Phi(x) = 0.5*(1+erf(x/sqrt(2))) ~= sigmoid(x*P(x^2)),  gelu = x*Phi(x)
# P is a minimax (LP) fit of 2*atanh(erf(x/sqrt(2)))/x weighted by the gelu
# error; max |gelu error| = 3.1e-6 over the whole real line (checked in fp32 to
# |x| = 5000).  The leading coefficient is positive, so P(t) > 0 for all
# t >= 36 and the sigmoid saturates correctly for large |x| -- no clamp needed.
# Evaluated as a Horner chain in t^2 with an Estrin step in t: a plain 5-deep
# Horner chain costs 2x more on this backend (8.5 vs 4.0 ms per 1e9 bf16
# elements) because the dependent FMAs cannot be overlapped.  This form needs
# 6 ops for the polynomial and keeps the dependency depth at 3.
# 11 vector ops total vs libdevice erff's ~24-30.
@triton.jit
def _gelu_erf_logistic(x):
    t = x * x
    t2 = t * t
    q = (-2.512334e-4 - 6.08916e-5 * t) + t2 * 2.2226e-6
    p = (1.5956548825 + 0.0729405472 * t) + t2 * q
    return x * _gelu_sigmoid(x * p)


@libentry()
@triton.jit
def _gelu_mul_flat(
    X,
    Y,
    Out,
    N,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
    GRID: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    num_tile = (N + BLOCK - 1) // BLOCK
    cols = tl.arange(0, BLOCK)
    for tile_id in tl.range(pid, num_tile, GRID, num_stages=NUM_STAGES):
        off = tile_id * BLOCK + cols
        if EVEN:
            x = tl.load(X + off).to(tl.float32)
            y = tl.load(Y + off).to(tl.float32)
        else:
            mask = off < N
            x = tl.load(X + off, mask=mask, other=0.0).to(tl.float32)
            y = tl.load(Y + off, mask=mask, other=0.0).to(tl.float32)
        if MODE == 0:
            x_gelu = _gelu_erf_logistic(x)
        else:
            # c*(x + b*x^3) == (c + c*b*x^2)*x, one op less
            u = (0.7978845608028654 + 0.035677408136300125 * x * x) * x
            x_gelu = x * _gelu_tanh_sigmoid(u)
        out = x_gelu * y
        if EVEN:
            tl.store(Out + off, out)
        else:
            tl.store(Out + off, out, mask)


@libentry()
@triton.jit
def _gelu_mul_grad_flat(
    X, Y, DGrad, DX, DY, N, MODE: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    cols = tl.arange(0, BLOCK)
    for start in range(pid * BLOCK, N, num_programs * BLOCK):
        off = start + cols
        mask = off < N
        x = tl.load(X + off, mask, other=0.0).to(tl.float32)
        y = tl.load(Y + off, mask, other=0.0).to(tl.float32)
        d = tl.load(DGrad + off, mask, other=0.0).to(tl.float32)
        if MODE == 0:
            e = erf(x * 0.7071067811865475)
            x_gelu = 0.5 * x * (1.0 + e)
            dx = d * y * 0.5 * (1.0 + e + x * 0.7978845608028654 * tl.exp(-0.5 * x * x))
        else:
            u = 0.7978845608028654 * (x + 0.044715 * x * x * x)
            s = _gelu_tanh_sigmoid(u)
            x_gelu = x * s
            # d/dx [x*sigmoid(2u)] = s + 2*x*s*(1-s)*u'
            dx = (
                d
                * y
                * (
                    s
                    + 2.0
                    * x
                    * s
                    * (1.0 - s)
                    * (0.7978845608028654 * (1.0 + 3.0 * 0.044715 * x * x))
                )
            )
        tl.store(DX + off, dx, mask)
        tl.store(DY + off, d * x_gelu, mask)


def _grid(n):
    return (min(MAX_GRID_SIZE, triton.cdiv(n, BLOCK_SIZE)),)


def _launch(x, y, out, mode):
    """Forward launch: fixed grid, pipelined tile loop, per-dtype tile size.

    Small inputs shrink the tile to the exact size so a single CTA does one
    iteration (same shape as the previous implementation, no regression); the
    mask only exists when the tile does not divide the element count.
    """
    n = x.numel()
    if n == 0:
        return
    two_byte = x.element_size() <= 2
    if n <= _SMALL_N:
        block = max(64, triton.next_power_of_2(n))
        grid = _GRID_SIZE
        num_warps = 4
    elif two_byte and n >= _BIG_N:
        block = _TILE_BIG
        grid = _GRID_BIG
        num_warps = 2
    else:
        block = _TILE_SMALL if two_byte else _TILE_FP32
        grid = _GRID_SIZE
        num_warps = 2 if two_byte else 4
    grid = min(grid, triton.cdiv(n, block))
    _gelu_mul_flat[(grid, 1, 1)](
        x,
        y,
        out,
        n,
        mode,
        block,
        grid,
        1,
        (n % block) == 0,
        num_warps=num_warps,
    )


class GeluAndMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, approximate="none"):
        logger.debug("GEMS_ENFLAME GELU_AND_MUL_FORWARD")
        if approximate not in ("none", "tanh"):
            raise ValueError(f"Invalid approximate value: {approximate}")
        x = x.contiguous()
        y = y.contiguous()
        ctx.save_for_backward(x, y)
        ctx.approximate = approximate
        out = torch.empty_like(x)
        mode = 0 if approximate == "none" else 1
        _launch(x, y, out, mode)
        return out

    @staticmethod
    def backward(ctx, dgrad):
        x, y = ctx.saved_tensors
        dgrad = dgrad.contiguous()
        dx = torch.empty_like(x)
        dy = torch.empty_like(y)
        mode = 0 if ctx.approximate == "none" else 1
        _gelu_mul_grad_flat[_grid(x.numel())](
            x, y, dgrad, dx, dy, x.numel(), mode, BLOCK_SIZE, num_warps=4
        )
        return dx, dy, None


def gelu_and_mul(x, y, approximate="none"):
    # Inference path: avoid autograd.Function.apply graph construction.
    if not torch.is_grad_enabled() or not (x.requires_grad or y.requires_grad):
        if approximate not in ("none", "tanh"):
            raise ValueError(f"Invalid approximate value: {approximate}")
        x = x.contiguous()
        y = y.contiguous()
        out = torch.empty_like(x)
        mode = 0 if approximate == "none" else 1
        _launch(x, y, out, mode)
        return out
    return GeluAndMul.apply(x, y, approximate)
