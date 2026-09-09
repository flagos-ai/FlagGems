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

"""Ascend-specialized linalg.solve_triangular (v0.2).

Design (v0.2, see version log 2026_08_27_linalg_solve_triangular_v0_2_log.md):
  - Block-panel TRSM.  Diagonal blocks are solved by ONE kernel launch each
    (trsm_diag_single_kernel) using tl.debug_barrier() after every row store to
    order the within-kernel read-after-write (validated stable on triton-ascend
    3.2.0; see P1 experiment 1 in the log).
  - GEMM updates use the ascend mm (runtime/backend/_ascend/ops/mm.py), which
    works on NPU (avoids the generic mm's SPLIT_K issue) and is contribution
    compliant.  The update subtraction is a dedicated Triton kernel (sub2d).
  - Contribution guide (docs/contribution/overview.md): no PyTorch computation
    ops in host functions; only data-init functions and FlagGems/Triton ops.
    Forward and backward avoid torch.matmul / torch.linalg.* / torch.triu /
    torch.tril / torch.neg etc.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

from .mm import mm as _mm

logger = logging.getLogger(__name__)

BLOCK_SIZE: tl.constexpr = 64
SLIDE_SIZE: tl.constexpr = 64


@libentry()
@triton.jit
def trsm_diag_single_kernel(
    A_ptr,
    X_ptr,
    k,
    row_start,
    row_end,
    stride_batch_a,
    stride_a,
    stride_batch_x,
    stride_x,
    UNIT: tl.constexpr,
    UPPER: tl.constexpr,
    SLIDE_SIZE: tl.constexpr,
):
    """Solve one triangular diagonal block with a single kernel launch.

    One program per (column slice, batch).  Rows are solved serially inside the
    kernel; a tl.debug_barrier() after each row store orders the subsequent
    row loads (the plain store->load read-after-write was measured as an
    intermittent race on triton-ascend, the barrier makes it deterministic).
    """
    pid_col = tl.program_id(0)
    pid_batch = tl.program_id(1)

    A_ptr += pid_batch * stride_batch_a
    X_ptr += pid_batch * stride_batch_x

    col_start = pid_col * SLIDE_SIZE
    col_offs = col_start + tl.arange(0, SLIDE_SIZE)
    col_mask = col_offs < k

    block_rows = row_end - row_start

    for i_idx in range(block_rows):
        if not UPPER:
            actual_i = row_start + i_idx
        else:
            actual_i = row_end - 1 - i_idx

        x_vals = tl.load(
            X_ptr + actual_i * stride_x + col_offs, mask=col_mask, other=0.0
        )

        if UPPER:
            for p_idx in range(i_idx):
                actual_p = row_end - 1 - p_idx
                a_val = tl.load(A_ptr + actual_i * stride_a + actual_p)
                xp_vals = tl.load(
                    X_ptr + actual_p * stride_x + col_offs,
                    mask=col_mask,
                    other=0.0,
                )
                x_vals = x_vals - a_val * xp_vals
        else:
            for p_idx in range(i_idx):
                actual_p = row_start + p_idx
                a_val = tl.load(A_ptr + actual_i * stride_a + actual_p)
                xp_vals = tl.load(
                    X_ptr + actual_p * stride_x + col_offs,
                    mask=col_mask,
                    other=0.0,
                )
                x_vals = x_vals - a_val * xp_vals

        if not UNIT:
            a_diag = tl.load(A_ptr + actual_i * stride_a + actual_i)
            x_vals = x_vals / a_diag

        tl.store(X_ptr + actual_i * stride_x + col_offs, x_vals, mask=col_mask)
        tl.debug_barrier()


def _solve_diag_block(A, X, k, row_start, row_end, upper, unitriangular):
    batch = A.shape[0]
    grid = (triton.cdiv(k, SLIDE_SIZE), batch)
    trsm_diag_single_kernel[grid](
        A,
        X,
        k,
        row_start,
        row_end,
        A.stride(0),
        A.stride(1),
        X.stride(0),
        X.stride(1),
        unitriangular,
        upper,
        SLIDE_SIZE=SLIDE_SIZE,
    )


def _sub_view(X_view, update):
    """X_view = X_view - update via flag_gems.sub.

    A custom Triton elementwise sub kernel was measured to corrupt the
    top-left region for certain shapes on triton-ascend 3.2.0 (same class of
    backend issue as the general impl's first-columns bug); the library's
    pointwise op is correct and contribution compliant.
    """
    from flag_gems.ops import sub

    return sub(X_view, update)


def _mm_batched(a, b):
    """(batch, M, K) x (batch, K, N) -> (batch, M, N) via ascend mm."""
    batch = a.shape[0]
    M = a.shape[1]
    N = b.shape[2]
    out = torch.empty((batch, M, N), device=a.device, dtype=torch.float32)
    for i in range(batch):
        out[i] = _mm(a[i], b[i])
    return out


def _blocked_trsm(A, B, upper, left, unitriangular):
    if not left:
        result = _blocked_trsm(
            A.mT.contiguous(),
            B.mT.contiguous(),
            not upper,
            True,
            unitriangular,
        )
        return result.mT.contiguous()

    A = A.contiguous()
    B = B.contiguous()

    if A.numel() == 0 or B.numel() == 0:
        return B.clone()

    if A.dim() == 2:
        return _blocked_trsm_impl(
            A.unsqueeze(0), B.unsqueeze(0), upper, unitriangular
        ).squeeze(0)

    return _blocked_trsm_impl(A, B, upper, unitriangular)


def _blocked_trsm_impl(A, B, upper, unitriangular):
    """A: (..., n, n), B: (..., n, k).  Returns X with A X = B."""
    n = A.shape[-1]
    k = B.shape[-1]
    batch_shape = A.shape[:-2]
    batch = 1
    for s in batch_shape:
        batch *= s

    A = A.reshape(batch, n, n)
    B = B.reshape(batch, n, k)

    X = B.clone().contiguous()
    A = A.contiguous()

    if n == 0 or k == 0:
        return X.reshape(*batch_shape, n, k)

    if upper:
        blocks = [(i, min(i + BLOCK_SIZE, n)) for i in range(0, n, BLOCK_SIZE)]
        for i_start, i_end in reversed(blocks):
            _solve_diag_block(A, X, k, i_start, i_end, True, unitriangular)
            if i_start > 0:
                for b in range(batch):
                    update = _mm(
                        A[b, :i_start, i_start:i_end],
                        X[b, i_start:i_end, :],
                    )
                    X[b, :i_start, :] = _sub_view(X[b, :i_start, :], update)
    else:
        for i in range(0, n, BLOCK_SIZE):
            i_end = min(i + BLOCK_SIZE, n)

            _solve_diag_block(A, X, k, i, i_end, False, unitriangular)

            if i_end < n:
                for b in range(batch):
                    update = _mm(
                        A[b, i_end:, i:i_end],
                        X[b, i:i_end, :],
                    )
                    X[b, i_end:, :] = _sub_view(X[b, i_end:, :], update)

    return X.reshape(*batch_shape, n, k)


class SolveTriangularFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, upper, left, unitriangular):
        logger.debug("GEMS LINALG_SOLVE_TRIANGULAR FORWARD")
        ctx.upper = upper
        ctx.left = left
        ctx.unitriangular = unitriangular

        X = _blocked_trsm(A, B, upper, left, unitriangular)
        ctx.save_for_backward(A, X)
        return X

    @staticmethod
    def backward(ctx, grad_X):
        logger.debug("GEMS LINALG_SOLVE_TRIANGULAR BACKWARD")
        # deferred imports: flag_gems.ops is heavy and may cause circular import
        from flag_gems.ops import neg, tril, triu

        A, X = ctx.saved_tensors
        upper = ctx.upper
        left = ctx.left
        unitriangular = ctx.unitriangular

        grad_A = None
        grad_B = None

        if ctx.needs_input_grad[0]:
            if left:
                C = _mm_batched(grad_X, X.mT.contiguous())
            else:
                C = _mm_batched(X.mT.contiguous(), grad_X)

            if unitriangular:
                d = -1 if not upper else 1
            else:
                d = 0
            if upper:
                grad_A = neg(triu(C, diagonal=d))
            else:
                grad_A = neg(tril(C, diagonal=d))

        if ctx.needs_input_grad[1]:
            if left:
                grad_B = _blocked_trsm(
                    A.mT.contiguous(), grad_X, not upper, True, False
                )
            else:
                grad_B = _blocked_trsm(A.mT.contiguous(), grad_X, upper, False, False)

        return grad_A, grad_B, None, None, None


def linalg_solve_triangular(A, B, *, upper=False, left=True, unitriangular=False):
    logger.debug("GEMS LINALG_SOLVE_TRIANGULAR")
    assert A.dtype in (
        torch.float32,
        torch.float64,
    ), "linalg_solve_triangular only supports float32 and float64"
    assert A.shape[-1] == A.shape[-2], "A must be square in its last two dimensions"

    return SolveTriangularFunction.apply(A, B, upper, left, unitriangular)


def linalg_solve_triangular_out(
    A, B, *, upper=False, left=True, unitriangular=False, out=None
):
    logger.debug("GEMS LINALG_SOLVE_TRIANGULAR OUT")
    if out is None:
        return linalg_solve_triangular(
            A, B, upper=upper, left=left, unitriangular=unitriangular
        )

    result = SolveTriangularFunction.apply(A, B, upper, left, unitriangular)
    out.copy_(result)
    return out
