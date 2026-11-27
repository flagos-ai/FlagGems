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
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)

logger = logging.getLogger(__name__)

# Tile-shape constants for the single-pass kernel (gcu400 tuned, hash-RNG era):
#   * a [ROWS, BLOCK_K] program tile aims at <= TILE_LANES lanes. With the
#     fmix32 hash RNG (no philox intermediates, no int64 counter ladder) the
#     sampler's live set is small enough that 4096-lane tiles are the measured
#     optimum on every core shape; the philox-era 2048 cap (2048-lane tiles
#     spilled and 4096-lane tiles were 3x slower) no longer applies.
#   * each GCU warp drives LANES_PER_WARP=512 lanes, so num_warps follows the
#     tile width (measured: nw < lanes/512 starves the extern calls, nw above
#     it just idles warps).
TILE_LANES = 4096
ROWS_MAX = 64
LANES_PER_WARP = 512
# gcu400 grid.x is capped at 65535 programs; larger row counts fold into the
# y axis (GRID_X is the folded x size, 0 meaning a plain 1D launch).
GRID_X_MAX = 65535

# Golden-ratio and avalanche constants used to derive independent hash streams.
_GOLDEN = 0x9E3779B9
_MIX_C = 0x85EBCA77
_CORR_C = 0xC2B2AE3D ^ 0x27D4EB2F


@triton.jit
def _fmix32(x):
    # xxhash32-style avalanche finalizer: 8 cheap integer ops, full avalanche
    # (same helper as ops/binomial.py / ops/dropout.py on this backend).
    x = x ^ (x >> 15)
    x = tl.mul(x, 0x85EBCA77, sanitize_overflow=False)
    x = x ^ (x >> 13)
    x = tl.mul(x, 0xC2B2AE3D, sanitize_overflow=False)
    x = x ^ (x >> 16)
    return x


@triton.jit
def _cos_poly(phi):
    # cos(phi) for phi in [-pi, pi] via the 5-coefficient near-minimax
    # polynomial (same coefficient class as ops/randn.py::_sin_cos_reduced):
    # max abs err 8.3e-5, far below what a Monte-Carlo normal draw can resolve.
    x2 = phi * phi
    x4 = x2 * x2
    x6 = x2 * x4
    x8 = x4 * x4
    return (
        (1.87756032403289852e-05 * x8 + -1.33820223333839584e-03 * x6)
        + (4.14830130757504120e-02 * x4 + -4.99744345256876943e-01 * x2)
        + 9.99916607180738048e-01
    )


@triton.jit
def sample_gamma_marsaglia_tsang(
    alpha,
    key,
    MAX_ITER: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    USE_POLY_COS: tl.constexpr,
):
    """
    Sample from Gamma(alpha, 1) using Marsaglia-Tsang method.
    For alpha < 1, use the transformation:
        Gamma(alpha) = Gamma(alpha+1) * U^(1/alpha)

    All arithmetic runs in COMPUTE_DTYPE so an fp64 input keeps double
    precision through the rejection loop and the tail correction.

    Randomness comes from the fmix32 hash keyed by `key` (one uint32 per
    element, already mixed with the per-call avalanched state).  Iteration i
    draws from stream keys key ^ (i*golden) with three stream offsets, so
    draws are independent across elements, iterations and consumers.
    """
    alpha = alpha.to(COMPUTE_DTYPE)

    # For alpha < 1, boost to alpha + 1
    alpha_orig = alpha
    alpha_boosted = tl.where(alpha < 1.0, alpha + 1.0, alpha)

    d = alpha_boosted - 1.0 / 3.0
    c = 1.0 / tl.sqrt(9.0 * d)

    # Rejection sampling loop. Marsaglia-Tsang has a very high acceptance
    # rate (> 95% per iteration), so nearly every lane accepts within a
    # few draws. `result` is locked to the *first* acceptance (guarded by
    # `~done`), so later iterations leave an accepted lane untouched.
    # MAX_ITER only bounds the worst case. The upstream version exits early
    # with a block-wide `while`; see below for why the gcu400 copy runs the
    # bounded loop to the end instead.
    result = alpha * 0.0
    done = alpha < 0.0  # Initialize to all False

    # gcu400: a ``while`` loop whose exit test reduces the block through
    # ``tl.sum`` crashes the compiler when the body carries a random draw
    # (make_gcuir segfaults, no diagnostic).  The rejection loop is a
    # fixed-trip ``for`` instead: the ``~done`` guards freeze every lane at its
    # first acceptance, which is exactly the value the early exit produced, so
    # only the wasted tail iterations differ.
    for iter in range(MAX_ITER):
        # Three independent hash streams for this iteration.
        key_i = key ^ (iter.to(tl.uint32) * 0x9E3779B9)
        r0 = _fmix32(key_i)
        r1 = _fmix32(key_i ^ 0x9E3779B9)
        r2 = _fmix32(key_i ^ 0x85EBCA77)

        u1 = tl.maximum(uint_to_uniform_float(r0).to(COMPUTE_DTYPE), 1e-10)
        u2 = uint_to_uniform_float(r1).to(COMPUTE_DTYPE)
        v_uniform = uint_to_uniform_float(r2).to(COMPUTE_DTYPE)

        radius = tl.sqrt(-2.0 * tl.log(u1))
        # cos(2*pi*u2) for u2 in [0,1): fold to phi in [-pi,pi), then
        # cos(2*pi*u2) = cos(phi + pi) = -cos(phi).
        if USE_POLY_COS:
            phi = (u2 - 0.5) * 6.283185307179586
            x = radius * (-_cos_poly(phi))
        else:
            theta = 6.283185307179586 * u2
            x = radius * tl.cos(theta)

        v = 1.0 + c * x

        # Check if v > 0
        valid_v = v > 0.0

        v3 = v * v * v
        log_v = tl.log(v)

        # Acceptance condition: U < 1 - 0.0331*x^4 (squeeze)
        # or log(U) < 0.5*x^2 + d*(1 - v^3 + log(v^3))
        x2 = x * x
        squeeze = 1.0 - 0.0331 * x2 * x2
        accept_squeeze = v_uniform < squeeze

        log_condition = tl.log(v_uniform) < (0.5 * x2 + d * (1.0 - v3 + 3.0 * log_v))
        accept = valid_v & (accept_squeeze | log_condition)

        # Update result for accepted samples (only if not already done)
        gamma_sample = d * v3
        result = tl.where(~done & accept, gamma_sample, result)
        done = done | accept

    # For alpha < 1, apply correction: multiply by U^(1/alpha). The correction
    # draws from a fourth hash stream, independent of every rejection stream.
    rc = _fmix32(key ^ 0xE5664512)
    u_correct = tl.maximum(uint_to_uniform_float(rc).to(COMPUTE_DTYPE), 1e-10)
    correction = tl.exp(tl.log(u_correct) / alpha_orig)
    result = tl.where(alpha_orig < 1.0, result * correction, result)

    return result


@triton.jit
def _dirichlet_key(seed, offset, idx):
    """Per-element hash key: element index mixed with the per-call avalanched
    generator state. The state advances on every op call (philox_backend_
    seed_offset), so successive calls draw independent samples."""
    state = seed.to(tl.int64) ^ offset.to(tl.int64)
    s0 = _fmix32((state ^ (state >> 32)).to(tl.uint32))
    return idx.to(tl.uint32) ^ s0


@libentry()
@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "N", "K"])
def sample_dirichlet_kernel(
    alpha_ptr,
    out_ptr,
    N,
    K,
    philox_seed,
    philox_offset,
    tiny_ptr,
    BLOCK_K: tl.constexpr,
    MAX_ITER: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    ROWS: tl.constexpr,
    GRID_X: tl.constexpr,
):
    """
    Sample from Dirichlet distribution, single-pass over a [ROWS, BLOCK_K] tile.

    One program owns ROWS whole rows and BLOCK_K >= K columns, so the raw
    gammas stay in registers: sample once, reduce the row sum in-register,
    normalize, store.  This halves the gamma-sampling work of the two-pass
    scheme (needed only when K spans several BLOCK_K blocks) and the ROWS-batch
    amortizes the extern transcendental calls that dominate this kernel on
    gcu400 (measured 5-7x over ROWS=1 at equal total lanes).

    philox_seed/philox_offset advance the torch generator state exactly like
    the upstream philox scheme (so re-running with the same seed reproduces the
    same draws), but the per-element streams themselves come from fmix32 hash
    keys instead of philox counters: the draw sequence does not match the
    upstream/generic kernel element-for-element, only in distribution.
    """
    # GRID_X == 0 means the grid never folds into y (the overwhelmingly
    # common case); the fold keeps grid.x inside the 65535 hardware limit.
    if GRID_X == 0:
        pid = tl.program_id(0)
    else:
        pid = tl.program_id(0) + tl.program_id(1) * GRID_X

    # Load the clamp floor in COMPUTE_DTYPE. It must come through a pointer of
    # the input dtype: a Python-float kernel arg is coerced to fp32 and fp64's
    # tiny (2.2e-308) would underflow to zero, defeating the clamp.
    tiny = tl.load(tiny_ptr).to(COMPUTE_DTYPE)

    row_offs = pid * ROWS + tl.arange(0, ROWS)
    row_mask = row_offs < N
    k_offs = tl.arange(0, BLOCK_K)
    k_mask = k_offs < K
    alpha_idx = row_offs[:, None] * K + k_offs[None, :]
    mask = row_mask[:, None] & k_mask[None, :]

    alpha = tl.load(alpha_ptr + alpha_idx, mask=mask, other=1.0)
    key = _dirichlet_key(philox_seed, philox_offset, alpha_idx)

    gamma = sample_gamma_marsaglia_tsang(
        alpha,
        key,
        MAX_ITER,
        COMPUTE_DTYPE,
        USE_POLY_COS=COMPUTE_DTYPE != tl.float64,
    )
    # ATen clamps each gamma draw to the output dtype's smallest positive
    # value so tiny-alpha rows (whose boost factor U**(1/alpha) underflows
    # to zero) still sum to a well-defined 1 instead of 0/0.
    gamma = tl.maximum(gamma, tiny)
    # Every gamma was clamped to tiny, so each real row's sum is >= K*tiny > 0
    # and no epsilon floor is needed.  Fully masked rows reduce to 0, giving
    # inv_sum = inf there, but those lanes are never stored.
    gamma_sum = tl.sum(tl.where(mask, gamma, 0.0).to(COMPUTE_DTYPE), axis=1)
    inv_sum = 1.0 / gamma_sum
    normalized = gamma * inv_sum[:, None]
    tl.store(out_ptr + alpha_idx, normalized.to(out_ptr.dtype.element_ty), mask=mask)


@libentry()
@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "N", "K"])
def sample_dirichlet_kernel_multi_block(
    alpha_ptr,
    out_ptr,
    N,
    K,
    philox_seed,
    philox_offset,
    tiny_ptr,
    BLOCK_K: tl.constexpr,
    MAX_ITER: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    GRID_X: tl.constexpr,
):
    """
    Two-pass Dirichlet sampling for rows wider than one BLOCK_K block (K > 1024).

    A Dirichlet draw is Gamma(alpha_k) / sum_j Gamma(alpha_j), so the row must
    be normalized by the *full* row sum.  When K spans more than one BLOCK_K
    the per-element gammas cannot all stay in registers, so this kernel makes
    two passes over the row:

      1. sample each gamma and accumulate the full-row sum;
      2. re-sample each gamma (deterministic under the same hash key) and
         divide by the full-row sum.

    Re-sampling avoids stashing the raw gammas in the low-precision output
    buffer, so the normalization stays in COMPUTE_DTYPE.  GRID_X folds large
    row counts into grid.y exactly like sample_dirichlet_kernel.
    """
    if GRID_X == 0:
        row_id = tl.program_id(0)
    else:
        row_id = tl.program_id(0) + tl.program_id(1) * GRID_X

    if row_id >= N:
        return

    # Load the clamp floor in COMPUTE_DTYPE. It must come through a pointer of
    # the input dtype: a Python-float kernel arg is coerced to fp32 and fp64's
    # tiny (2.2e-308) would underflow to zero, defeating the clamp.
    tiny = tl.load(tiny_ptr).to(COMPUTE_DTYPE)

    # Pass 1: accumulate the full-row gamma sum in COMPUTE_DTYPE.
    gamma_sum = tl.zeros([], dtype=COMPUTE_DTYPE)
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        mask = k_offs < K

        alpha_idx = row_id * K + k_offs
        alpha = tl.load(alpha_ptr + alpha_idx, mask=mask, other=1.0)
        key = _dirichlet_key(philox_seed, philox_offset, alpha_idx)

        gamma = sample_gamma_marsaglia_tsang(
            alpha,
            key,
            MAX_ITER,
            COMPUTE_DTYPE,
            USE_POLY_COS=COMPUTE_DTYPE != tl.float64,
        )
        gamma = tl.maximum(gamma, tiny)
        gamma_sum += tl.sum(tl.where(mask, gamma, 0.0).to(COMPUTE_DTYPE))

    inv_sum = 1.0 / gamma_sum

    # Pass 2: re-sample each gamma with the identical key and normalize.
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        mask = k_offs < K

        alpha_idx = row_id * K + k_offs
        alpha = tl.load(alpha_ptr + alpha_idx, mask=mask, other=1.0)
        key = _dirichlet_key(philox_seed, philox_offset, alpha_idx)

        gamma = sample_gamma_marsaglia_tsang(
            alpha,
            key,
            MAX_ITER,
            COMPUTE_DTYPE,
            USE_POLY_COS=COMPUTE_DTYPE != tl.float64,
        )
        gamma = tl.maximum(gamma, tiny)
        normalized = gamma * inv_sum
        tl.store(
            out_ptr + alpha_idx, normalized.to(out_ptr.dtype.element_ty), mask=mask
        )


def _sample_dirichlet(input, generator=None):
    """
    Sample from Dirichlet distribution.

    Args:
        input: Tensor of shape [..., K] containing alpha parameters
        generator: Optional random generator

    Returns:
        Tensor of same shape as input, where each K-dimensional vector
        sums to 1
    """
    logger.debug("GEMS_ENFLAME _SAMPLE_DIRICHLET")

    assert input.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ), f"Unsupported dtype: {input.dtype}"

    # Flatten all but last dimension
    orig_shape = input.shape
    if input.ndim == 0:
        # Scalar input - unsqueeze to [1]
        input = input.unsqueeze(0)

    if input.ndim == 1:
        K = input.shape[0]
        N = 1
        inp = input.unsqueeze(0)
    else:
        inp = input.reshape(-1, input.shape[-1])
        N = inp.shape[0]
        K = inp.shape[1]

    inp = inp.contiguous()
    out = torch.empty_like(inp)

    if N == 0 or K == 0:
        return out.reshape(orig_shape)

    # Advance the generator state for this call's worth of randomness
    # (matches the upstream philox accounting; the hash streams keyed by
    # seed/offset then consume it, see _dirichlet_key).
    increment = triton.cdiv(N * K * 512, 4)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )

    # Worst-case rejection iterations. Marsaglia-Tsang accepts with >95%
    # probability per draw, so a lane still missing a sample after 6 draws
    # has probability < (0.05)^6 ~ 1.6e-8; such a lane degenerates to the
    # clamped tiny value and its row still sums to 1. The gcu400 fixed-trip
    # loop (see the sampler) makes every extra iteration cost full price on
    # every lane, so unlike the upstream early-exit version this bound is the
    # real trip count and is kept tight on purpose (100 -> 8 measured ~9.7x,
    # 8 -> 6 another -16%).
    MAX_ITER = 6

    # fp64 inputs keep double precision through sampling and normalization;
    # every other float dtype samples in fp32 (its uniforms have only fp32
    # granularity anyway) and rounds to the output dtype on store.
    compute_dtype = tl.float64 if inp.dtype == torch.float64 else tl.float32

    # Smallest positive value of the input dtype; each gamma draw is clamped to
    # it so tiny-alpha rows never collapse to an all-zero (0/0) normalization.
    # Passed as a 0-d tensor (not a Python float) to keep fp64 precision.
    tiny = torch.tensor(torch.finfo(inp.dtype).tiny, dtype=inp.dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        if K <= TILE_LANES:
            # Fast path: one program tile spans whole rows, gammas stay in
            # registers and each element is sampled exactly once.  ROWS is
            # kept a power of two (tl.arange requirement): the bit_length
            # form rounds the row batch down to the previous power of two so
            # the tail program masks its extra rows instead of requesting an
            # odd-shaped tile.
            BLOCK_K = triton.next_power_of_2(K)
            ROWS = min(
                ROWS_MAX,
                max(1, TILE_LANES // BLOCK_K),
                1 << (max(N, 1).bit_length() - 1),
            )
            num_warps = max(1, ROWS * BLOCK_K // LANES_PER_WARP)
            n_prog = triton.cdiv(N, ROWS)
            if n_prog <= GRID_X_MAX:
                grid = (n_prog,)
                grid_x = 0
            else:
                grid_x = GRID_X_MAX
                grid = (grid_x, triton.cdiv(n_prog, grid_x))
            sample_dirichlet_kernel[grid](
                inp,
                out,
                N,
                K,
                philox_seed,
                philox_offset,
                tiny,
                BLOCK_K=BLOCK_K,
                MAX_ITER=MAX_ITER,
                COMPUTE_DTYPE=compute_dtype,
                ROWS=ROWS,
                GRID_X=grid_x,
                num_warps=num_warps,
            )
        else:
            # Rows wider than one tile: two-pass multi-block sampling.
            BLOCK_K = 1024
            grid_x = min(N, GRID_X_MAX)
            grid = (grid_x, triton.cdiv(N, grid_x))
            sample_dirichlet_kernel_multi_block[grid](
                inp,
                out,
                N,
                K,
                philox_seed,
                philox_offset,
                tiny,
                BLOCK_K=BLOCK_K,
                MAX_ITER=MAX_ITER,
                COMPUTE_DTYPE=compute_dtype,
                GRID_X=0 if N <= GRID_X_MAX else GRID_X_MAX,
                num_warps=BLOCK_K // LANES_PER_WARP,
            )

    return out.reshape(orig_shape)
