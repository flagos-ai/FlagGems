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

"""Register-vector triangular substitution for Kunlunxin."""

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _cholesky_solve_helper_kernel(
    b_ptr,
    factor_ptr,
    out_ptr,
    BATCH_SHAPE: tl.constexpr,
    B_STRIDES: tl.constexpr,
    F_STRIDES: tl.constexpr,
    B_ROW: tl.constexpr,
    B_COL: tl.constexpr,
    F_ROW: tl.constexpr,
    F_COL: tl.constexpr,
    N: tl.constexpr,
    NRHS: tl.constexpr,
    UPPER: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch = tl.program_id(0)
    rhs = tl.program_id(1)
    remaining = batch
    b_offset = rhs * B_COL
    f_offset = 0
    for dim in tl.static_range(len(BATCH_SHAPE) - 1, -1, -1):
        index = remaining % BATCH_SHAPE[dim]
        remaining = remaining // BATCH_SHAPE[dim]
        b_offset += index * B_STRIDES[dim]
        f_offset += index * F_STRIDES[dim]
    rows = tl.arange(0, BLOCK_N)
    safe_rows = tl.minimum(rows, N - 1)
    b_base = b_ptr + b_offset
    factor_base = factor_ptr + f_offset
    diagonal = tl.load(factor_base + safe_rows * (F_ROW + F_COL))
    inverse = 1.0 / diagonal
    value = tl.load(b_base + safe_rows * B_ROW) * inverse
    # Each pivot is broadcast from the register vector. The solved entries stay
    # in registers; only a factor column is loaded for each substitution step.
    for pivot in range(N):
        solved = tl.sum(tl.where(rows == pivot, value, 0.0), 0)
        if UPPER:
            column = tl.load(factor_base + pivot * F_ROW + safe_rows * F_COL)
        else:
            column = tl.load(factor_base + safe_rows * F_ROW + pivot * F_COL)
        coefficient = tl.where(rows > pivot, column * inverse, 0.0)
        value = value - coefficient * solved
    value = value * inverse
    for step in range(N):
        pivot = N - 1 - step
        solved = tl.sum(tl.where(rows == pivot, value, 0.0), 0)
        if UPPER:
            column = tl.load(factor_base + safe_rows * F_ROW + pivot * F_COL)
        else:
            column = tl.load(factor_base + pivot * F_ROW + safe_rows * F_COL)
        coefficient = tl.where(rows < pivot, column * inverse, 0.0)
        value = value - coefficient * solved
    tl.store(
        out_ptr + (batch * N + rows) * NRHS + rhs,
        value,
        mask=rows < N,
    )


def _cholesky_solve_helper(
    self: torch.Tensor, A: torch.Tensor, upper: bool
) -> torch.Tensor:
    """Solve a positive-definite system from its Cholesky factor."""
    logger.debug("GEMS_KUNLUNXIN _CHOLESKY_SOLVE_HELPER")
    if self.dtype not in (torch.float32, torch.float64):
        raise RuntimeError(
            "_cholesky_solve_helper currently supports float32 and float64 only"
        )
    if self.dtype != A.dtype or self.device != A.device:
        raise RuntimeError("self and A must have the same dtype and device")
    if self.ndim < 2 or A.ndim < 2 or A.shape[-1] != A.shape[-2]:
        raise RuntimeError("expected a matrix right-hand side and a square factor")
    if self.shape[-2] != A.shape[-1]:
        raise RuntimeError("self and A have incompatible matrix dimensions")

    n = A.shape[-1]
    nrhs = self.shape[-1]
    if self.shape[:-2] == A.shape[:-2]:
        batch_shape = self.shape[:-2]
        b, factor = self, A
    else:
        batch_shape = torch.broadcast_shapes(self.shape[:-2], A.shape[:-2])
        b = self.expand(*batch_shape, n, nrhs)
        factor = A.expand(*batch_shape, n, n)
    out = torch.empty((*batch_shape, n, nrhs), dtype=self.dtype, device=self.device)
    batch_size = math.prod(batch_shape)
    if out.numel() == 0:
        return out
    with torch_device_fn.device(self.device):
        _cholesky_solve_helper_kernel[(batch_size, nrhs)](
            b,
            factor,
            out,
            tuple(batch_shape),
            b.stride()[:-2],
            factor.stride()[:-2],
            b.stride(-2),
            b.stride(-1),
            factor.stride(-2),
            factor.stride(-1),
            n,
            nrhs,
            upper,
            triton.next_power_of_2(n),
            num_warps=1,
            num_stages=1,
        )
    return out
