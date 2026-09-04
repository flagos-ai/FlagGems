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

import builtins
import math

import torch
import triton

from flag_gems.runtime import torch_device_fn

from .rms_norm import (
    rms_norm_backward,
    rms_norm_flat_kernel,
    rms_norm_kerne_tile,
    rms_norm_kernel,
    rms_norm_multirow_kernel,
    rms_norm_tile2d_kernel,
)

# Launch-bound / 2D-tile dispatch for the fused forward. Same structural fixes as
# rms_norm (constexpr-N unmasked [TILE_M, N] tile -> block DMA; flat kernel for
# N == 1; kerne_tile for N > 8192), kept local to this file so the shared
# rms_norm.py dispatch is untouched. Measured on XPU (2026-08-13, official
# do_bench): for tile2d the preferred TILE_M is 32, not 16 -- (1024,1024) tm32
# 63-74us vs tm16 71-80us, (4096,4096) tm32 520-693us vs tm16 558-732us for all
# three dtypes.
MULTIROW_N = 256
MULTIROW_M = 4096
TILE_BUDGET = 8192


def _fused_tile_m(N, M):
    """TILE_M for the unmasked 2D tile kernel, or None if not applicable."""
    if N <= 256:
        for cand in (32, 16):
            if M % cand == 0:
                return cand
        return None
    tm = 32  # measured faster than 16 on XPU for every M % 32 == 0 shape
    while tm * N > 65536 * 2:  # [TILE_M, N] fp32 tile SRAM budget (tm32 x N=4096 OK)
        tm //= 2
    while tm >= 2:
        if M % tm == 0:
            return tm
        tm //= 2
    return None


def _fused_rms_norm_forward(x, normalized_shape, weight, eps=1e-5):
    """Forward of fused rms_norm (returns y, inv_rms), Kunlunxin backend local.

    Dispatch mirrors rms_norm_forward but keeps the tile selection in this file
    (TILE_M=32 preference) without touching the shared rms_norm.py kernels.
    """
    dim = x.ndim - len(normalized_shape)
    M = math.prod(x.shape[:dim])
    N = math.prod(normalized_shape)

    BLOCK_SIZE = builtins.min(64 * 128, triton.next_power_of_2(N))

    x = x.contiguous()
    weight = weight.contiguous()
    # Native empty_strided: gems `empty` is intercepted and JIT-recompiles per
    # call on this XPU (~95-100ms/call, see rms_norm fix); empty_strided is not.
    y = torch.empty_strided(x.size(), x.stride(), dtype=x.dtype, device=x.device)
    inv_rms = torch.empty_strided((M,), (1,), dtype=torch.float32, device=x.device)

    with torch_device_fn.device(x.device):
        if N > 64 * 128:
            need_mask = (N % BLOCK_SIZE) != 0
            rms_norm_kerne_tile[M,](
                y,
                inv_rms,
                x,
                weight,
                N,
                1,
                N,
                1,
                M,
                N,
                eps,
                BLOCK_SIZE,
                need_mask,
            )
        elif N == 1:
            FLAT_BLOCK = 4096
            rms_norm_flat_kernel[(triton.cdiv(M, FLAT_BLOCK),)](
                y, inv_rms, x, weight, M, eps, FLAT_BLOCK
            )
        else:
            TILE_M = _fused_tile_m(N, M)
            if TILE_M is not None:
                rms_norm_tile2d_kernel[(M // TILE_M,)](
                    y, inv_rms, x, weight, eps, TILE_M, N
                )
            elif N <= MULTIROW_N and M >= MULTIROW_M:
                TILE_M = builtins.max(1, TILE_BUDGET // N)
                rms_norm_multirow_kernel[(triton.cdiv(M, TILE_M),)](
                    y, inv_rms, x, weight, M, eps, TILE_M, N
                )
            else:
                need_mask = (N % BLOCK_SIZE) != 0
                rms_norm_kernel[M,](
                    y,
                    inv_rms,
                    x,
                    weight,
                    N,
                    1,
                    N,
                    1,
                    M,
                    N,
                    eps,
                    BLOCK_SIZE,
                    need_mask,
                )
    return y, inv_rms


class _FusedRmsNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, eps):
        y, inv_rms = _fused_rms_norm_forward(x, normalized_shape, weight, eps)
        ctx.save_for_backward(x, inv_rms, weight)
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        return y, inv_rms

    @staticmethod
    def backward(ctx, dy, d_inv_rms):
        x, inv_rms, weight = ctx.saved_tensors
        dx, dw = rms_norm_backward(
            dy, x, inv_rms, ctx.normalized_shape, weight, ctx.eps
        )
        return dx, None, dw, None


def _fused_rms_norm(x, normalized_shape, weight=None, eps=1e-5):
    if weight is not None:
        return _FusedRmsNorm.apply(x, normalized_shape, weight, eps)

    n = math.prod(normalized_shape)
    unit_weight = torch.ones(n, dtype=x.dtype, device=x.device)
    return _fused_rms_norm_forward(x, normalized_shape, unit_weight, eps)
