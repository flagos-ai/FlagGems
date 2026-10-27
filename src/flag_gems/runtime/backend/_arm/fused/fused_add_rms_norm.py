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

"""ARM-optimized fused residual-add + RMS normalization.

Replaces 5 separate Triton kernel launches (add, pow, mean, rsqrt, mul)
with a single kernel launch.  For decode shapes (M=1, N=896) this reduces
overhead from 5 × ~9μs ≈ 45μs to 1 × ~9μs ≈ 9μs per layer.

Uses a two-pass tiled approach with small BLOCK_SIZE (128) to avoid
extremely slow LLVM compilation with large vector widths on ARM.
"""

import logging
import math
import os

import torch
import triton
import triton.language as tl

from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_PREWARM_DONE = False
_PREWARM_ENABLED = os.environ.get("GEMS_ARM_FUSED_RMS_PREWARM", "1") == "1"

# Use small block size to keep LLVM compilation fast (~seconds not minutes)
_TILE_SIZE = 128


@triton.jit(do_not_specialize=["eps"])
def _fused_add_rms_norm_kernel(
    input_ptr,
    residual_ptr,
    weight_ptr,
    in_stride_r,
    r_stride_r,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused: residual += input; output = rms_norm(residual) * weight.

    Two-pass tiled approach:
      Pass 1: Load tiles, compute x=input+residual, store residual, accumulate x^2
      Pass 2: Load tiles of x (from residual), compute normalized output
    """
    pid = tle.program_id(0)
    in_row = input_ptr + pid * in_stride_r
    r_row = residual_ptr + pid * r_stride_r

    # Pass 1: fused add + store residual + accumulate variance
    sum_sq = tl.zeros([1], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N

        x = tl.load(in_row + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(r_row + cols, mask=mask, other=0.0).to(tl.float32)

        x = x + r

        # Store updated residual
        tl.store(r_row + cols, x.to(residual_ptr.dtype.element_ty), mask=mask)

        sum_sq += tl.sum(x * x, axis=0)

    # Compute rrms
    var = sum_sq / N
    rrms = 1.0 / tl.sqrt(var + eps)

    # Pass 2: load residual (=x+r), normalize, multiply by weight, store output
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N

        # Read back the updated residual (which is x+r in original dtype)
        x = tl.load(r_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + cols, mask=mask, other=0.0)

        y = (x * rrms).to(input_ptr.dtype.element_ty) * w
        tl.store(in_row + cols, y, mask=mask)


# Note: standalone _rms_norm_kernel (without residual add) was removed after
# A/B measurement showed zero E2E benefit vs ATen's Qwen3RMSNorm on BF16 M=1
# (see test_tle_phase1_plus.py ENABLE_RMSNORM_PATCH A/B, 3 rounds:
#  ON=9.93 tok/s, OFF=9.97 tok/s — within noise).
# The fused add+rmsnorm path is kept because it saves a residual-add memory
# roundtrip and is used by vLLM's forward_cpu when residual is present.


def _maybe_prewarm():
    global _PREWARM_DONE
    if _PREWARM_DONE or not _PREWARM_ENABLED:
        _PREWARM_DONE = True
        return
    try:
        for dt in (torch.float32,):
            x = torch.ones((1, _TILE_SIZE), dtype=dt, device="cpu")
            r = torch.ones((1, _TILE_SIZE), dtype=dt, device="cpu")
            w = torch.ones(_TILE_SIZE, dtype=dt, device="cpu")
            _fused_add_rms_norm_kernel[(1,)](
                x,
                r,
                w,
                _TILE_SIZE,
                _TILE_SIZE,
                _TILE_SIZE,
                1e-6,
                BLOCK_SIZE=_TILE_SIZE,
                num_warps=1,
                num_stages=1,
            )
    except Exception:
        logger.debug("GEMS_ARM fused RMSNorm prewarm failed", exc_info=True)
    _PREWARM_DONE = True


def fused_add_rms_norm(x, residual, normalized_shape, weight, eps=1e-5):
    """Fused residual-add + RMS normalization (in-place).

    Modifies both x and residual tensors in-place:
      residual = x + residual
      x = rms_norm(residual) * weight

    Returns: (x, residual) - both modified in-place.
    """
    _maybe_prewarm()

    dim = x.ndim - len(normalized_shape)
    M = math.prod(x.shape[:dim])
    N = math.prod(normalized_shape)

    x = x.contiguous()
    residual = residual.contiguous()
    weight = weight.contiguous()

    _fused_add_rms_norm_kernel[(M,)](
        x,
        residual,
        weight,
        N,  # in_stride_r (contiguous: stride = N)
        N,  # r_stride_r
        N,
        eps,
        BLOCK_SIZE=_TILE_SIZE,
        num_warps=1,
        num_stages=1,
    )
    return x, residual


# rms_norm_forward() (standalone RMSNorm without residual) removed: A/B
# measurement on Qwen3-1.7B INT8 decode showed no measurable benefit over
# ATen's native Qwen3RMSNorm.forward (9.93 vs 9.97 tok/s, within noise).
