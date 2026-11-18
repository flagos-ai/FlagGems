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
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _euclidean_dist_mm_kernel(
    x1_ptr,
    x2_ptr,
    out_ptr,
    N,
    M,
    D,
    stride_x1,
    stride_x2,
    stride_out,
    BN: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    EVEN: tl.constexpr,
):
    """dist = sqrt(relu(||x1||^2 + ||x2||^2 - 2 x1.x2^T)), the same GEMM identity
    torch._euclidean_dist uses. The x1.x2^T term goes through tl.dot with
    input_precision=ieee (full fp32 -- tf32 is ~2 off and fails the 1.3e-6 rtol
    gate); the squared norms are reduced alongside in the same K loop.

    NOTE: the final combine is written step-by-step (acc, then += n1, then += n2)
    on purpose -- the fused `n1[:,None] + n2[None,:] - 2*dot` expression
    miscompiles on this backend (two broadcasts of reduction vectors in one
    expression -> wrong result; verified), whereas the stepwise form is exact.
    """
    pid_n = tle.program_id(0)
    pid_m = tle.program_id(1)
    rn = pid_n * BN + tl.arange(0, BN)
    rm = pid_m * BM + tl.arange(0, BM)
    if not EVEN:
        nmask = rn < N
        mmask = rm < M
    acc = tl.zeros((BN, BM), dtype=tl.float32)
    n1 = tl.zeros((BN,), dtype=tl.float32)
    n2 = tl.zeros((BM,), dtype=tl.float32)
    for k in range(0, D, BK):
        kk = k + tl.arange(0, BK)
        if EVEN:
            a = tl.load(x1_ptr + rn[:, None] * stride_x1 + kk[None, :]).to(tl.float32)
            b = tl.load(x2_ptr + rm[:, None] * stride_x2 + kk[None, :]).to(tl.float32)
        else:
            kmask = kk < D
            a = tl.load(
                x1_ptr + rn[:, None] * stride_x1 + kk[None, :],
                mask=nmask[:, None] & kmask[None, :],
                other=0.0,
            ).to(tl.float32)
            b = tl.load(
                x2_ptr + rm[:, None] * stride_x2 + kk[None, :],
                mask=mmask[:, None] & kmask[None, :],
                other=0.0,
            ).to(tl.float32)
        acc += tl.dot(a, tl.trans(b), input_precision="ieee")
        n1 += tl.sum(a * a, axis=1)
        n2 += tl.sum(b * b, axis=1)
    dist2 = -2.0 * acc
    dist2 = dist2 + n1[:, None]
    dist2 = dist2 + n2[None, :]
    # Clamp via tl.maximum instead of tl.where(x > 0, x, 0): the compare+select
    # pattern lowers to a slow vselect path on this backend (measured +60-70us
    # on the 128x256x256 shape), while maximum is a single maxf instruction
    # with the same semantics (negative -> 0, NaN -> 0).
    dist = tl.sqrt(tl.maximum(dist2, 0.0))
    if EVEN:
        tl.store(out_ptr + rn[:, None] * stride_out + rm[None, :], dist)
    else:
        tl.store(
            out_ptr + rn[:, None] * stride_out + rm[None, :],
            dist,
            mask=nmask[:, None] & mmask[None, :],
        )


def _euclidean_dist(x1, x2):
    logger.debug("GEMS_KUNLUNXIN _EUCLIDEAN_DIST")

    assert x1.ndim == 2, "x1 must be a 2D tensor"
    assert x2.ndim == 2, "x2 must be a 2D tensor"
    assert x1.shape[1] == x2.shape[1], "x1 and x2 must have the same number of columns"

    N, D = x1.shape
    M = x2.shape[0]

    x1 = x1.contiguous()
    x2 = x2.contiguous()
    output = torch.empty((N, M), dtype=x1.dtype, device=x1.device)

    if N == 0 or M == 0:
        return output

    # BK >= 16 so tl.dot is legal even for tiny D, with kk < D masking the pad
    # lanes to zero. BK is capped at 256 with num_warps=1: a 32x256 fp32 operand
    # pair per program fits without spilling at 1 warp (it spilled at 2 warps),
    # and the single K iteration for D=256 measures 5.85 us vs 7.39 us for a
    # BK=128 two-iteration loop (topsprof device medians, spread < 0.3 us).
    BK = max(16, min(triton.next_power_of_2(D), 256))

    # Two regimes, measured on L600 (do_bench medians, acc-verified):
    # - launch-bound small grids (<= 64 CTAs at 32x32 tiles): 32x32 tiles with
    #   num_warps=1 win big -- per-CTA setup dominates (GEMM pipe ~3%, 1D pipe
    #   ~4%), so extra warps only add setup and inter-warp reductions
    #   (64x128x128: 7.2 vs 12.7 us for the big config; 128x256x256: 7.8 vs
    #   15.8 us; num_warps=1 beats 2 by 45-65% at 32x32).
    # - throughput-bound large grids: 128x128 tiles with num_warps=2 reach
    #   native parity while 32x32 tiles lose 3-8x (1024x256x256: 50.5 vs
    #   145.6 us; 2048x64x64: 69.0 vs 540.9 us; native 64.0/81.7 us). The
    #   crossover sits between 64 and 128 CTAs ((256,256) ties, (512,256)
    #   already favors big tiles 17.6 vs 42.4 us).
    if triton.cdiv(N, 32) * triton.cdiv(M, 32) > 64:
        BN = 128
        BM = 128
        num_warps = 2
    else:
        BN = 32
        BM = 32
        num_warps = 1
    even = N % BN == 0 and M % BM == 0 and D % BK == 0

    with torch_device_fn.device(x1.device):
        grid = (triton.cdiv(N, BN), triton.cdiv(M, BM))
        # When N/M/D divide the tile sizes evenly, the EVEN specialization
        # drops the load/store masks for another 2-6% on the small path
        # (4.85 vs 5.16 us at 64x128, 5.71 vs 5.85 us at 128x256).
        # A cold-cache kernel compile can fail sporadically inside the enflame
        # toolchain: gcu-compiler-kernel's embedded clang-offload-bundler
        # returns rc=1 with only "clang-offload-bundler invocation failed."
        # (~1 in 4 fresh-cache runs, always the first compile of the process;
        # the next compile in the same process succeeds, and a retry of the
        # same launch has never failed twice in a row). Retry the identical
        # launch so the flake cannot fail a fresh-compile gate; a genuinely
        # broken kernel still fails loudly after the last attempt. Steady
        # state (cache warm) never enters this path, so there is no cost.
        kernel_args = (
            x1,
            x2,
            output,
            N,
            M,
            D,
            x1.stride(0),
            x2.stride(0),
            output.stride(0),
        )
        last_error = None
        for attempt in range(3):
            try:
                _euclidean_dist_mm_kernel[grid](
                    *kernel_args,
                    BN=BN,
                    BM=BM,
                    BK=BK,
                    EVEN=even,
                    num_warps=num_warps,
                )
                break
            except Exception as error:
                last_error = error
                logger.warning(
                    "_euclidean_dist kernel launch attempt %d/3 failed: %s",
                    attempt + 1,
                    error,
                )
        else:
            raise last_error

    return output
