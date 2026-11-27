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
import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

from .rms_norm import (
    rms_norm_backward,
    rms_norm_flat_kernel,
    rms_norm_kerne_tile,
    rms_norm_kernel,
    rms_norm_multirow_kernel,
    rms_norm_tile2d_kernel,
)

logger = logging.getLogger(__name__)

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

# Backward fused-tile budget: each program holds several [TILE_M, N] fp32 tiles
# (x, dy, g, normalized, dx), so keep TILE_M * N <= 16384 elements (~64KB/tile).
BACKWARD_ROW_BLOCK = 8192
SMALL_SHAPE_LIMIT = 16384


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
    logger.debug("GEMS_KUNLUNXIN _FUSED_RMS_NORM_FORWARD")
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
        dx, dw = _fused_rms_norm_backward(
            dy,
            x,
            ctx.normalized_shape,
            inv_rms,
            weight,
            (True, weight is not None),
        )
        return dx, None, dw, None


def _fused_rms_norm(x, normalized_shape, weight=None, eps=1e-5):
    normalized_shape = list(normalized_shape)
    assert x.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ), f"_fused_rms_norm does not support dtype {x.dtype}"
    # The aten schema declares eps as ``float?``; the dispatcher may forward
    # ``None`` when the caller omits it, so fall back to the PyTorch default.
    if eps is None:
        eps = 1e-5
    if weight is not None:
        return _FusedRmsNorm.apply(x, normalized_shape, weight, eps)

    n = math.prod(normalized_shape)
    unit_weight = torch.ones(n, dtype=x.dtype, device=x.device)
    return _fused_rms_norm_forward(x, normalized_shape, unit_weight, eps)


# --- Fused backward kernel: dx and dw partials in ONE pass ---------------------
# The rms_norm_backward route runs two separate passes over x/dy: a per-row dx
# kernel (grid M) plus a [1, COL_BLOCK] dw kernel (grid M x cdiv(N, 256), i.e.
# up to 262144 programs for (1024,65536)) and a (M, N) fp32 partial; kernel
# profiling shows the dw kernel takes ~86% of the op time.  The kernel below
# fuses both gradients into a single per-row pass (grid M, 1D reductions only),
# so x/dy are read once and the launch count is just M; only the small
# cross-row dw reduction (torch.sum over the partial, the same accepted pattern
# as rms_norm_backward) remains.
#
# NOTE: a 2D [TILE_M, N] tile variant was measured first but is NOT usable on
# this XPU triton - axis=1 reductions silently miscompile for many (TILE_M, N)
# combinations (probe evidence in harness/perf_ir/debug_tile_kernel*.py), so the
# fused kernel is strictly per-row 1D (the same reduction shape as the proven
# rms_norm_grad_dx_kernel).
@libentry()
@triton.jit(do_not_specialize=["N"])
def _fused_rms_norm_grad_row_kernel(
    X,  # input
    DY,  # grad output
    INV_RMS,  # per-row inverse rms (fp32)
    W,  # weight
    DX,  # input gradient output
    PARTIAL,  # (M, N) fp32 dw partials (one row per program)
    N,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
    SINGLE_PHASE: tl.constexpr,
):
    pid = ext.program_id(0)
    x_row = X + pid * N
    dy_row = DY + pid * N
    dx_row = DX + pid * N
    inv_rms = tl.load(INV_RMS + pid).to(tl.float32)

    if SINGLE_PHASE:
        # N fits in ONE block: load x/dy/w once, reduce in-register, emit dx and
        # the dw partial from the same registers (no second pass over x/dy).
        cols = tl.arange(0, BLOCK_SIZE)
        if NEED_MASK:
            mask = cols < N
            x = tl.load(x_row + cols, mask, other=0.0).to(tl.float32)
            dy = tl.load(dy_row + cols, mask, other=0.0).to(tl.float32)
            w = tl.load(W + cols, mask, other=0.0).to(tl.float32)
        else:
            x = tl.load(x_row + cols).to(tl.float32)
            dy = tl.load(dy_row + cols).to(tl.float32)
            w = tl.load(W + cols).to(tl.float32)
        g = dy * w
        normalized = x * inv_rms
        row_sum_stats = tl.sum(normalized * g)
        dx = (g - normalized * (1.0 / N) * row_sum_stats) * inv_rms
        if NEED_MASK:
            tl.store(dx_row + cols, dx.to(DX.dtype.element_ty), mask=mask)
            tl.store(PARTIAL + pid * N + cols, x * inv_rms * dy, mask=mask)
        else:
            tl.store(dx_row + cols, dx.to(DX.dtype.element_ty))
            tl.store(PARTIAL + pid * N + cols, x * inv_rms * dy)
    else:
        # N spans multiple blocks: pass 1 accumulates the row sum over all
        # blocks, pass 2 emits dx and the dw partial.
        sum_base = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            if NEED_MASK:
                mask = cols < N
                x = tl.load(x_row + cols, mask, other=0.0).to(tl.float32)
                dy = tl.load(dy_row + cols, mask, other=0.0).to(tl.float32)
                w = tl.load(W + cols, mask, other=0.0).to(tl.float32)
            else:
                x = tl.load(x_row + cols).to(tl.float32)
                dy = tl.load(dy_row + cols).to(tl.float32)
                w = tl.load(W + cols).to(tl.float32)
            sum_base += (x * inv_rms) * (dy * w)
        row_sum_stats = tl.sum(sum_base)

        inv_n = 1.0 / N
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            if NEED_MASK:
                mask = cols < N
                x = tl.load(x_row + cols, mask, other=0.0).to(tl.float32)
                dy = tl.load(dy_row + cols, mask, other=0.0).to(tl.float32)
                w = tl.load(W + cols, mask, other=0.0).to(tl.float32)
            else:
                x = tl.load(x_row + cols).to(tl.float32)
                dy = tl.load(dy_row + cols).to(tl.float32)
                w = tl.load(W + cols).to(tl.float32)
            g = dy * w
            normalized = x * inv_rms
            dx = (g - normalized * inv_n * row_sum_stats) * inv_rms
            if NEED_MASK:
                tl.store(dx_row + cols, dx.to(DX.dtype.element_ty), mask=mask)
                tl.store(PARTIAL + pid * N + cols, x * inv_rms * dy, mask=mask)
            else:
                tl.store(dx_row + cols, dx.to(DX.dtype.element_ty))
                tl.store(PARTIAL + pid * N + cols, x * inv_rms * dy)


def _fused_rms_norm_backward(
    grad_out, input, normalized_shape, rstd, weight=None, output_mask=(True, True)
):
    """Kunlunxin override of aten::_fused_rms_norm_backward.

    The generic `flag_gems.ops._fused_rms_norm` backward kernels do not
    legalize on this XPU triton (`tl.sum(axis=0)` over a 2D tile and a
    `triton_xpu.reduce` type-inference failure), so the vendor implementation
    routes through the fused single-pass kernels below, with
    `.rms_norm.rms_norm_backward` retained as the correctness fallback for
    shapes that do not fit the fused fast path. For the no-weight case the dx
    formula equals the weight=ones result, so a unit weight is passed and the
    weight gradient is discarded.
    """
    logger.debug("GEMS_KUNLUNXIN _FUSED_RMS_NORM_BACKWARD")
    if len(output_mask) != 2:
        raise ValueError("output_mask must contain two booleans")
    if output_mask[1] and weight is None:
        raise RuntimeError("weight gradient requested without a weight tensor")

    if weight is None:
        if not output_mask[0]:
            return None, None
        n = math.prod(normalized_shape)
        unit_weight = torch.ones(n, dtype=input.dtype, device=input.device)
        dx, _ = rms_norm_backward(grad_out, input, rstd, normalized_shape, unit_weight)
        return dx, None

    dim = input.ndim - len(normalized_shape)
    M = math.prod(input.shape[:dim])
    N = math.prod(normalized_shape)

    x = input.contiguous()
    dy = grad_out.contiguous()
    weight = weight.contiguous()
    dx = torch.empty_like(x)

    # Tiny shapes (<= 16384 elements) stay on the shared rms_norm_backward path:
    # A/B measurement shows the fused single-pass kernel is 4-8% slower there
    # (launch/quantization bound at 1-2 programs-per-core work), so dispatching
    # them to the original path keeps B strictly no-regression vs A.
    if M * N <= SMALL_SHAPE_LIMIT:
        dx, dw = rms_norm_backward(dy, x, rstd, normalized_shape, weight)
    else:
        # Single fused per-row pass (1D reductions only): BLOCK_SIZE caps at
        # 8192 so large normalized dims use a loop, matching
        # rms_norm_grad_dx_kernel_tile; NEED_MASK only when N is not a multiple
        # of BLOCK_SIZE (unmasked = block DMA).
        BLOCK_SIZE = builtins.min(BACKWARD_ROW_BLOCK, triton.next_power_of_2(N))
        need_mask = (N % BLOCK_SIZE) != 0
        single_phase = N <= BACKWARD_ROW_BLOCK
        partial = torch.empty((M, N), dtype=torch.float32, device=x.device)
        with torch_device_fn.device(x.device):
            _fused_rms_norm_grad_row_kernel[(M,)](
                x,
                dy,
                rstd,
                weight,
                dx,
                partial,
                N,
                BLOCK_SIZE,
                need_mask,
                single_phase,
            )
            # Cross-row reduce of the (M, N) dw partial: dispatches to the
            # vendor-native XPU reduce kernel (profiled as reduce_tn), NOT a
            # CPU/CUDA fallback; same accepted production pattern as
            # rms_norm_backward.
            dw = torch.sum(  # allow-torch-compute
                partial, dim=0, dtype=x.dtype
            ).reshape(-1)

    return (dx if output_mask[0] else None), (dw if output_mask[1] else None)
