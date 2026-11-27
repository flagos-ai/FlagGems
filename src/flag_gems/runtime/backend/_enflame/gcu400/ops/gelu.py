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
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)
# Forward launch parameters (measured on GCU400): fixed 48 CTAs, 32K tile for large shapes.
_GRID_SIZE = 48
_TILE_BIG = 32768
_SMALL_N = 32768


@triton.jit
def _sigmoid(z):
    """Backend fast-path sigmoid (tl.exp + tl.fdiv), an order of magnitude cheaper than libdevice tanh/erf."""
    return tl.fdiv(1.0, 1.0 + tl.exp(-z))


@libentry()
@triton.jit(do_not_specialize=["N"])
def _gelu_kernel(
    X,
    Out,
    N,
    APPROX: tl.constexpr,
    LOWPOLY: tl.constexpr,
    EVEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """gelu forward (inplace included; Out may alias X).

    erf takes the logistic form: Phi(x) = 0.5(1+erf(x/sqrt2)) ~ sigmoid(x*P(x^2)),
    gelu = x*Phi(x). P comes from a minimax fit (fp32 degree 4, |gelu error| <= 3.1e-6 over the
    whole real line; 16-bit uses degree 2 with 4.2e-5 error, far below the quantization noise).
    Against libdevice erff (~24-30 vector ops) this form needs only 11 vector ops, and tl.exp
    takes the backend fast exp path. The polynomial **must be inlined** in the loop body:
    extracting it into a `@triton.jit` subroutine makes the backend fuse one layer less, measured
    ~22% slower at the same shape (1M fp32 0.0125 -> 0.0151ms). The tanh approximation also uses
    the sigmoid form, since 0.5x(1+tanh(u)) == x*sigmoid(2u).
    """
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_blocks = (N + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + tl.arange(0, BLOCK)
        if EVEN:
            x = tl.load(X + off).to(tl.float32)
        else:
            mask = off < N
            x = tl.load(X + off, mask=mask, other=0.0).to(tl.float32)
        if APPROX == 0:
            t = x * x
            t2 = t * t
            if LOWPOLY:
                p = 1.59469875 + t * (0.0742564352 - 7.38255533e-4 * t)
            else:
                q = (-2.512334e-4 - 6.08916e-5 * t) + t2 * 2.2226e-6
                p = (1.5956548825 + 0.0729405472 * t) + t2 * q
            y = x * _sigmoid(x * p)
        else:
            # c*(x + b*x^3) == (c + c*b*x^2)*x, saving one multiplication
            u = (0.7978845608028654 + 0.035677408136300125 * x * x) * x
            y = x * tl.fdiv(1.0, 1.0 + tl.exp(-2.0 * u))
        if EVEN:
            tl.store(Out + off, y)
        else:
            tl.store(Out + off, y, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N"])
def _gelu_backward_kernel(
    DY,
    X,
    Out,
    N,
    APPROX: tl.constexpr,
    LOWPOLY: tl.constexpr,
    EVEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    inv_sqrt_2pi: tl.constexpr = 0.3989422804014327
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_blocks = (N + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + tl.arange(0, BLOCK)
        if EVEN:
            dy = tl.load(DY + off).to(tl.float32)
            x = tl.load(X + off).to(tl.float32)
        else:
            mask = off < N
            dy = tl.load(DY + off, mask=mask, other=0.0).to(tl.float32)
            x = tl.load(X + off, mask=mask, other=0.0).to(tl.float32)
        if APPROX == 0:
            # d/dx [x*Phi(x)] = Phi(x) + x*phi(x); Phi uses the same logistic form as the forward pass
            t = x * x
            t2 = t * t
            if LOWPOLY:
                p = 1.59469875 + t * (0.0742564352 - 7.38255533e-4 * t)
            else:
                q = (-2.512334e-4 - 6.08916e-5 * t) + t2 * 2.2226e-6
                p = (1.5956548825 + 0.0729405472 * t) + t2 * q
            dydx = tl.fdiv(1.0, 1.0 + tl.exp(-(x * p))) + x * (
                inv_sqrt_2pi * tl.exp(-0.5 * t)
            )
        else:
            u = (0.7978845608028654 + 0.035677408136300125 * x * x) * x
            s = tl.fdiv(1.0, 1.0 + tl.exp(-2.0 * u))
            # d/dx [x*sigmoid(2u)] = s + 2*x*s*(1-s)*u'
            dydx = s + 2.0 * x * s * (1.0 - s) * (
                0.7978845608028654 + 0.1070322243 * x * x
            )
        if EVEN:
            tl.store(Out + off, dydx * dy)
        else:
            tl.store(Out + off, dydx * dy, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N"])
def _gelu_backward_kernel_h(
    DY,
    X,
    Out,
    N,
    APPROX: tl.constexpr,
    LOWPOLY: tl.constexpr,
    EVEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Half-block staged gelu_backward (fp32 large tier only).

    Each outer BLOCK tile is processed internally as two static half-width stages: under fp32
    each stage gets a 128KB DMA descriptor (stable granularity) and the in-flight footprint
    halves (393KB -> below the full-tile simultaneous in-flight size); 256M elements measured
    1486 vs 1733us tiled B32768 (-14.3%). The math body is identical to _gelu_backward_kernel;
    EVEN is judged on the outer BLOCK only, with per-half masking when it does not divide.
    Do not use it for 16-bit (measured 28-70% worse).
    """
    inv_sqrt_2pi: tl.constexpr = 0.3989422804014327
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_blocks = (N + BLOCK - 1) // BLOCK
    HALF: tl.constexpr = BLOCK // 2
    for block_id in tl.range(pid, num_blocks, num_pids):
        base = block_id * BLOCK
        for h in tl.static_range(2):
            off = base + h * HALF + tl.arange(0, HALF)
            if EVEN:
                dy = tl.load(DY + off).to(tl.float32)
                x = tl.load(X + off).to(tl.float32)
            else:
                mask = off < N
                dy = tl.load(DY + off, mask=mask, other=0.0).to(tl.float32)
                x = tl.load(X + off, mask=mask, other=0.0).to(tl.float32)
            if APPROX == 0:
                # Same 4th-order Phi logistic form as the tiled version
                t = x * x
                t2 = t * t
                if LOWPOLY:
                    p = 1.59469875 + t * (0.0742564352 - 7.38255533e-4 * t)
                else:
                    q = (-2.512334e-4 - 6.08916e-5 * t) + t2 * 2.2226e-6
                    p = (1.5956548825 + 0.0729405472 * t) + t2 * q
                dydx = tl.fdiv(1.0, 1.0 + tl.exp(-(x * p))) + x * (
                    inv_sqrt_2pi * tl.exp(-0.5 * t)
                )
            else:
                u = (0.7978845608028654 + 0.035677408136300125 * x * x) * x
                s = tl.fdiv(1.0, 1.0 + tl.exp(-2.0 * u))
                dydx = s + 2.0 * x * s * (1.0 - s) * (
                    0.7978845608028654 + 0.1070322243 * x * x
                )
            if EVEN:
                tl.store(Out + off, dydx * dy)
            else:
                tl.store(Out + off, dydx * dy, mask=mask)


def _pick_tile(N):
    """Picks (tile width, CTA count) from the element count.

    GCU400 measurements: small shapes (<=32K elements) are cheapest with an exact next_pow2(N)
    tile in a single CTA doing one iteration; larger shapes split across 48 CTAs with a
    ceil(N/48) power-of-two tile capped at 32K. The original implementation gave fp32 65536,
    which left a 64K-element case with a single program (0.0206ms, 4x slower than torch) -- the
    origin of the "large tile, single CTA" trap.
    """
    if N <= _SMALL_N:
        return max(64, triton.next_power_of_2(N)), 1
    block = min(
        _TILE_BIG, max(2048, triton.next_power_of_2(triton.cdiv(N, _GRID_SIZE)))
    )
    return block, min(_GRID_SIZE, triton.cdiv(N, block))


# gelu_backward-specific tiers (measured 2026-11-18, using the official bench kernel convention w1000/r100,
# rechecked with a bench-style tensor layout):
# The 2R1W sweet spot differs from the forward 1R1W one; backward has its own tiers, do not mix with _pick_tile.
_BWD_TINY_2B = 4096
_BWD_MID1 = 1048576
_BWD_MID2 = 16777216
_BWD_STAGED = 16777216  # Starting point of the fp32 half-block staging
_BWD_STAGED_BLOCK = 65536


def _pick_tile_bwd(N, two_byte):
    """backward-specific (kernel, tile width, CTA count, num_warps) tiers.

    GCU400 probes (official bench shapes plus a bench-style tensor layout re-check):
    - 2-byte tiny (<=4096): B512/nw1 over many CTAs sits on the launch floor (fp16 (64,64)
      3.40 vs 4.11us for a single CTA).
    - 2-byte (4096, 32K]: keep an exact np2 tile in a single CTA (same as forward).
    - 2-byte (32K, 1M]: B8192/nw2/g48 (1M: 10.6 vs 12.4us).
    - 2-byte (1M, 16M]: B16384/nw2 (2^21..2^24: 16.2/24.5/40.3/71.6us, best at every tier).
    - 2-byte >16M: B32768/nw2 (256M: 976.5us; B65536 is 17% worse; the half-block staged
      S8@131072 is also 28% worse -- with a heavy double-exp math chain neither helps).
    - fp32 (32K, 16M): B8192/nw4 (2^16..2^23: 4.9/7.7/10.0/15.4/23.2/37.7/66.3us
      all tiers best; a 64KB-granularity tile keeps only 96KB in flight, far from the DSM line).
    - fp32 >=16M: the half-block staged kernel (outer 65536, two inner 32768 stages, 128KB
      descriptors): 16M 122.2 vs 125.7, 256M 1486 vs 1733us (-14.3%). Note that fp32 tiled
      B16384 (64KB descriptor) has an address-layout-sensitive defect: bench layout 142/2109us vs
      a clean layout 107.6/1550us, so do not use it; tiled B32768 is stable (126/1734) but loses to staging.
    One call returns kernel/num_warps directly: the bench kernel-mode timing closure includes the
    full Python wrapper, and a two-level conditional chain under heavy load punches through the
    do_bench 256MB cache-clear (~128us) shadow window (tiny tier +0.7us), so tiers converge in one frame.
    """
    if two_byte:
        if N <= _BWD_TINY_2B:
            return _gelu_backward_kernel, 512, triton.cdiv(N, 512), 1
        if N <= _SMALL_N:
            return _gelu_backward_kernel, max(64, triton.next_power_of_2(N)), 1, 4
        if N <= _BWD_MID1:
            return _gelu_backward_kernel, 8192, min(_GRID_SIZE, triton.cdiv(N, 8192)), 2
        if N <= _BWD_MID2:
            return (
                _gelu_backward_kernel,
                16384,
                min(_GRID_SIZE, triton.cdiv(N, 16384)),
                2,
            )
        return (
            _gelu_backward_kernel,
            _TILE_BIG,
            min(_GRID_SIZE, triton.cdiv(N, _TILE_BIG)),
            2,
        )
    if N <= _SMALL_N:
        return _gelu_backward_kernel, max(64, triton.next_power_of_2(N)), 1, 4
    if N < _BWD_STAGED:
        return _gelu_backward_kernel, 8192, min(_GRID_SIZE, triton.cdiv(N, 8192)), 4
    return (
        _gelu_backward_kernel_h,
        _BWD_STAGED_BLOCK,
        min(_GRID_SIZE, triton.cdiv(N, _BWD_STAGED_BLOCK)),
        4,
    )


def _approx_code(approximate):
    return 1 if approximate == "tanh" else 0


def _launch_forward(inp, out, approximate):
    """Forward launch: tiered tile + 48 CTA cap.

    Small shapes use an exact tile (one iteration in a single CTA); large shapes use a
    ceil(N/48) power-of-two tile capped at 32K so the CTA count fills up. All dtypes use
    num_warps=4: with 4 warps the 32K tile keeps 8x more independent lanes in flight than the
    previous 2-warp 2-byte tier, which measured +6% end to end (fp16 1.89 -> 2.03, and the
    512M-element cases 1.42 -> 2.26); 8 warps regressed to 1.85. The mask is only generated
    when the tile does not divide numel.
    """
    N = inp.numel()
    if N == 0:
        return out
    two_byte = inp.element_size() <= 2
    block, grid_size = _pick_tile(N)
    num_warps = 4
    with torch_device_fn.device(inp.device):
        _gelu_kernel[(grid_size,)](
            inp,
            out,
            N,
            _approx_code(approximate),
            two_byte,
            (N % block) == 0,
            BLOCK=block,
            num_warps=num_warps,
        )
    return out


def gelu(self, *, approximate="none"):
    logger.debug("GEMS_ENFLAME GELU")
    inp = self if self.is_contiguous() else self.contiguous()
    out = torch.empty_like(inp)
    return _launch_forward(inp, out, approximate)


def gelu_(A, *, approximate="none"):
    logger.debug("GEMS_ENFLAME GELU_")
    if not A.is_contiguous():
        A.copy_(gelu(A, approximate=approximate))
        return A
    return _launch_forward(A, A, approximate)


def gelu_backward(grad_output, self, *, approximate="none"):
    logger.debug("GEMS_ENFLAME GELU_BACKWARD")
    dy = grad_output if grad_output.is_contiguous() else grad_output.contiguous()
    inp = self if self.is_contiguous() else self.contiguous()
    out = torch.empty_like(dy)
    N = dy.numel()
    if N == 0:
        return out
    # 2R1W backward-specific tiers (see _pick_tile_bwd, a single call returns kernel + hyperparameters):
    # The 2-byte tiny tier with nw1 and more CTAs hits the launch floor; other 2-byte large tiers use nw2, and fp32
    # always uses nw4.
    kernel, block, grid_size, num_warps = _pick_tile_bwd(N, dy.element_size() <= 2)
    with torch_device_fn.device(dy.device):
        kernel[(grid_size,)](
            dy,
            inp,
            out,
            N,
            _approx_code(approximate),
            # Backward always uses 4th-order Phi (error 3.1e-6): the 16-bit 2nd-order arm's absolute error 4.2e-5 is
            # amplified by dy (~N(0,1)) and then combined with the tiny dydx~0 values at x~-4..-6, which in practice
            # causes
            # 0.1% of elements in test_gelu_backward to exceed the 1e-4 tolerance; the forward pass has no such
            # amplification but is still tiered.
            False,
            (N % block) == 0,
            BLOCK=block,
            num_warps=num_warps,
        )
    return out
