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

"""gcu400 te_rmsnorm_bwd.

Two changes on top of the generic implementation.

1. ``zero_centered_gamma`` is folded on the host.  ``rmsnorm_bwd_dx_kernel``
   applies ``w = w + 1.0`` inside a ``zero_centered_gamma`` constexpr branch and
   on this backend that branch miscompiles for several autotuned tile shapes
   (measured for BRS=4/8/16 with BCS=256/512: fp32 dx came back 1.35 off,
   matching neither the centred nor the un-centred formula; BRS=2 or
   num_warps=2 were correct).  Folding ``+1`` into the weight on the host keeps
   the fp32 dtype and leaves dgamma untouched, since dgamma does not depend on
   the weight.  Half-precision inputs keep the generic path (their tests pass
   and the host fold would change dgamma's dtype).

2. A single-pass dx kernel is used whenever a full row fits in one tile.  The
   generic dx kernel walks the columns twice - once to reduce
   ``c1 = mean(x_hat * dz * w)`` and once to emit dx - so it moves five arrays
   instead of three.  At [1024, 2048] fp32 that already measures 408 us against
   the ~200 GB/s wall of this device, i.e. the kernel is memory saturated and
   the only way down is dropping the second pass.  Keeping the row resident in
   registers removes the re-read of x and dz; dgamma still uses the generic
   kernels unchanged.
"""

import logging

import torch
import triton
import triton.language as tl

import flag_gems.ops.te_rmsnorm as _common_te_rmsnorm
from flag_gems.ops.te_rmsnorm import te_rmsnorm_bwd as _generic_te_rmsnorm_bwd
from flag_gems.runtime import torch_device_fn

__all__ = ["te_rmsnorm_bwd"]

# The single-pass tile must hold a whole row.  BLOCK_M * BLOCK_N * 4B * 3
# streaming buffers stays below ~100 KB of shared/register budget for the limits
# below.
_SINGLE_PASS_MAX_N = 8192
_SINGLE_PASS_MAX_ELEMS = 4096
_GENERIC_DGAMMA_M_THRESHOLD = 2048


@triton.jit(do_not_specialize=["M", "N"])
def _rmsnorm_bwd_dx_single_pass_kernel(
    dx_ptr,
    dz_ptr,
    x_ptr,
    weight_ptr,
    rsigma_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    row_mask = rows < M
    cols = tl.arange(0, BLOCK_N)[None, :]
    col_mask = cols < N
    mask = row_mask & col_mask

    if tl.constexpr(x_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        x_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = x_ptr.dtype.element_ty

    rsigma = tl.load(rsigma_ptr + rows, mask=row_mask, other=0.0).to(cdtype)
    # All three operands share the same row-major offsets, so the index math is
    # built once and only the base pointer differs.
    offsets = rows * N + cols
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(cdtype)
    dz = tl.load(dz_ptr + offsets, mask=mask, other=0.0).to(cdtype)
    w = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(cdtype)

    x_hat = x * rsigma
    # c1 per row; the row is already resident, so no second memory pass.
    c1 = tl.sum(tl.where(mask, x_hat * dz * w, 0.0), axis=1)[:, None] / N
    dx = rsigma * (dz * w - x_hat * c1)
    tl.store(dx_ptr + offsets, dx.to(dx_ptr.dtype.element_ty), mask=mask)


def _single_pass_dx_eligible(dz_2d, x_2d, gamma):
    return (
        x_2d.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and dz_2d.dtype == x_2d.dtype
        and gamma.dtype == x_2d.dtype
        and 0 < gamma.shape[0] <= _SINGLE_PASS_MAX_N
        and x_2d.is_contiguous()
        and dz_2d.is_contiguous()
    )


def _launch_single_pass_dx(dx, dz_2d, x_2d, gamma, rsigma, M, N):
    block_n = triton.next_power_of_2(N)
    block_m = max(1, min(64, _SINGLE_PASS_MAX_ELEMS // block_n))
    grid = (triton.cdiv(M, block_m),)
    _rmsnorm_bwd_dx_single_pass_kernel[grid](
        dx,
        dz_2d,
        x_2d,
        gamma,
        rsigma,
        M,
        N,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=1 if block_m * block_n <= 1024 else 4,
    )


def _compute_dgamma(dgamma, dz_2d, x_2d, rsigma, M, N):
    """Unchanged generic strategy: one kernel for small M, split-then-reduce for
    large M."""
    if M <= _GENERIC_DGAMMA_M_THRESHOLD:
        grid_dgamma = lambda meta: (
            triton.cdiv(N, meta["BLOCK_COL_SIZE"]),
        )  # noqa: E731
        _common_te_rmsnorm.rmsnorm_bwd_dgamma_kernel[grid_dgamma](
            dgamma, dz_2d, x_2d, rsigma, M, N
        )
        return

    num_row_groups = max(min(triton.cdiv(M, 128), 64), 1)
    dgamma_partial = torch.zeros(
        (num_row_groups, N), dtype=torch.float32, device=x_2d.device
    )
    grid_dgamma = lambda meta: (  # noqa: E731
        triton.cdiv(N, meta["BLOCK_COL_SIZE"]),
        num_row_groups,
    )
    _common_te_rmsnorm.rmsnorm_bwd_dgamma_parallel_kernel[grid_dgamma](
        dgamma_partial, dz_2d, x_2d, rsigma, M, N, N
    )
    reduce_block = 256
    _common_te_rmsnorm.rmsnorm_bwd_dgamma_reduce_kernel[
        (triton.cdiv(N, reduce_block),)
    ](dgamma, dgamma_partial, num_row_groups, N, N, BLOCK_COL_SIZE=reduce_block)


def te_rmsnorm_bwd(
    dz: torch.Tensor,
    x: torch.Tensor,
    rsigma: torch.Tensor,
    gamma: torch.Tensor,
    sm_margin: int = 0,
    zero_centered_gamma: bool = False,
):
    logger.debug("GEMS_ENFLAME TE_RMSNORM_BWD")
    if zero_centered_gamma and x.dtype == torch.float32:
        gamma = gamma + 1.0
        zero_centered_gamma = False
    if zero_centered_gamma:
        return _generic_te_rmsnorm_bwd(
            dz, x, rsigma, gamma, sm_margin, zero_centered_gamma
        )

    original_shape = x.shape
    N = gamma.shape[0]
    x_2d = x.reshape(-1, N).contiguous()
    dz_2d = dz.reshape(-1, N).contiguous()
    rsigma = rsigma.contiguous()
    M = x_2d.shape[0]

    if not _single_pass_dx_eligible(dz_2d, x_2d, gamma):
        return _generic_te_rmsnorm_bwd(dz_2d, x_2d, rsigma, gamma, sm_margin, False)

    dx = torch.empty_like(x_2d)
    dgamma = torch.zeros(N, dtype=gamma.dtype, device=gamma.device)
    with torch_device_fn.device(x.device):
        _launch_single_pass_dx(dx, dz_2d, x_2d, gamma, rsigma, M, N)
        _compute_dgamma(dgamma, dz_2d, x_2d, rsigma, M, N)
    return dx.view(*original_shape), dgamma


# Some callers (including tests) import ``te_rmsnorm_bwd`` from the shared
# module directly, which bypasses the backend registration.  Publish the
# wrapper there too; it is only imported on this backend.

logger = logging.getLogger(__name__)


_common_te_rmsnorm.te_rmsnorm_bwd = te_rmsnorm_bwd
