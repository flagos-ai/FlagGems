# Copyright 2026, The FlagOS Contributors.
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

"""Hygon specialization of the blocked fp32 cholesky_solve kernels.

The generic blocked kernels assume N % BLOCK_K == 0 and leave the forward
panel rows unmasked. With num_stages=3 the HCU software pipeliner peels one
extra iteration of ``for m in range(k + BLOCK_K, N, BLOCK_M)`` and issues the
``tail`` load for ``m == N`` (rows N .. N + BLOCK_M - 1). Those values are
discarded (the dot and the store are predicated on the iteration existing),
so results stay correct, but the loads still touch memory: when the output
tensor happens to be the last block of a 2 MB allocator segment, the peeled
access crosses into unmapped VA and raises a GPU VM fault that aborts the
process. See the cholesky_solve benchmark, whose repeated ``empty_like``
calls eventually place the output at a segment end.

This module re-declares only the two affected kernels with explicit row
masks and reuses the generic dispatcher for validation, layout normalization
and every other path; nothing else is forked.
"""

import importlib
import logging

import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils.triton_lang_extension import program_id

logger = logging.getLogger(__name__)

_generic = importlib.import_module("flag_gems.ops.cholesky_solve")


@libentry()
@triton.jit
def cholesky_solve_blocked_lower_kernel(
    L_ptr,
    B_ptr,
    X_ptr,
    N: tl.constexpr,
    nrhs: tl.constexpr,
    batch_stride_L,
    batch_stride_B,
    stride_L,
    stride_B,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_RHS: tl.constexpr,
):
    """Blocked lower-factor Cholesky solve.

    Solves L L^T X = B for one batch and one RHS tile using a row-major
    lower factor. Hygon copy of the generic kernel with masked panel rows.
    """
    batch_pid = program_id(0)
    rhs_tile_pid = program_id(1)

    L_base = batch_pid * batch_stride_L
    B_base = batch_pid * batch_stride_B
    rhs_cols = rhs_tile_pid * BLOCK_RHS + tl.arange(0, BLOCK_RHS)
    rhs_mask = rhs_cols < nrhs
    k_offsets = tl.arange(0, BLOCK_K)
    m_offsets = tl.arange(0, BLOCK_M)

    # Forward blocked TRSM: L * Y = B. The diagonal-block solve keeps the
    # running tile pre-scaled by the reciprocal diagonal so each serial row
    # step is a constant-index tl.gather (warp shuffle) plus a rank-1
    # update -- no per-row reduce or select (see the single-RHS kernels).
    for k in range(0, N, BLOCK_K):
        rows_k = k + k_offsets
        if k > 0:
            # Cross-warp handoff: rows of X below were written by tail
            # updates of earlier k-blocks, possibly by other warps. bar.sync
            # makes those global stores visible before we read them.
            tl.debug_barrier()
        if k == 0:
            y_block = tl.load(
                B_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
                mask=rhs_mask[None, :],
                other=0.0,
            )
        else:
            y_block = tl.load(
                X_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
                mask=rhs_mask[None, :],
                other=0.0,
            )

        diag_block = tl.load(L_ptr + L_base + rows_k * stride_L + rows_k)
        inv_diag = 1.0 / diag_block
        inv_diag = inv_diag * (2.0 - diag_block * inv_diag)

        y_block = y_block * inv_diag[:, None]
        for i in range(BLOCK_K):
            L_col = tl.load(
                L_ptr + L_base + rows_k * stride_L + (k + i),
                mask=k_offsets > i,
                other=0.0,
            )
            w_i = tl.gather(y_block, tl.full([1, BLOCK_RHS], i, tl.int32), 0)
            y_block = y_block - (L_col * inv_diag)[:, None] * w_i

        tl.store(
            X_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
            y_block,
            mask=rhs_mask[None, :],
        )

        for m in range(k + BLOCK_K, N, BLOCK_M):
            rows_m = m + m_offsets
            # The software pipeliner may peel one more iteration (m == N) for
            # this loop, so guard every tail-row access explicitly; loads are
            # predicated off instead of reading past B/L/X.
            rows_m_mask = rows_m < N
            L_tile = tl.load(
                L_ptr + L_base + rows_m[:, None] * stride_L + rows_k[None, :],
                mask=rows_m_mask[:, None],
                other=0.0,
            )
            if k == 0:
                tail = tl.load(
                    B_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                    mask=rows_m_mask[:, None] & rhs_mask[None, :],
                    other=0.0,
                )
            else:
                tail = tl.load(
                    X_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                    mask=rows_m_mask[:, None] & rhs_mask[None, :],
                    other=0.0,
                )
            tail = tail - tl.dot(L_tile, y_block, input_precision="ieee")
            tl.store(
                X_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                tail,
                mask=rows_m_mask[:, None] & rhs_mask[None, :],
            )

    # Backward blocked TRSM: L^T * X = Y.
    for k in range(N - BLOCK_K, -1, -BLOCK_K):
        rows_k = k + k_offsets
        tl.debug_barrier()
        x_block = tl.load(
            X_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
            mask=rhs_mask[None, :],
            other=0.0,
        )

        diag_block = tl.load(L_ptr + L_base + rows_k * stride_L + rows_k)
        inv_diag = 1.0 / diag_block
        inv_diag = inv_diag * (2.0 - diag_block * inv_diag)

        x_block = x_block * inv_diag[:, None]
        for ii in range(BLOCK_K - 1, -1, -1):
            L_row = tl.load(
                L_ptr + L_base + (k + ii) * stride_L + rows_k,
                mask=k_offsets < ii,
                other=0.0,
            )
            w_i = tl.gather(x_block, tl.full([1, BLOCK_RHS], ii, tl.int32), 0)
            x_block = x_block - (L_row * inv_diag)[:, None] * w_i

        tl.store(
            X_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
            x_block,
            mask=rhs_mask[None, :],
        )

        for m in range(0, k, BLOCK_M):
            rows_m = m + m_offsets
            L_tile = tl.load(
                L_ptr + L_base + rows_k[None, :] * stride_L + rows_m[:, None]
            )
            head = tl.load(
                X_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                mask=rhs_mask[None, :],
                other=0.0,
            )
            head = head - tl.dot(L_tile, x_block, input_precision="ieee")
            tl.store(
                X_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                head,
                mask=rhs_mask[None, :],
            )


@libentry()
@triton.jit
def cholesky_solve_blocked_upper_kernel(
    L_ptr,
    B_ptr,
    X_ptr,
    N: tl.constexpr,
    nrhs: tl.constexpr,
    batch_stride_L,
    batch_stride_B,
    stride_L,
    stride_B,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_RHS: tl.constexpr,
):
    """Blocked upper-factor Cholesky solve.

    Solves U^T U X = B for one batch and one RHS tile. Mirrors the lower
    blocked path while keeping the upper storage layout intact. Hygon copy of
    the generic kernel with masked panel rows.
    """
    batch_pid = program_id(0)
    rhs_tile_pid = program_id(1)

    L_base = batch_pid * batch_stride_L
    B_base = batch_pid * batch_stride_B
    rhs_cols = rhs_tile_pid * BLOCK_RHS + tl.arange(0, BLOCK_RHS)
    rhs_mask = rhs_cols < nrhs
    k_offsets = tl.arange(0, BLOCK_K)
    m_offsets = tl.arange(0, BLOCK_M)

    # Forward blocked TRSM: U^T * Y = B.
    for k in range(0, N, BLOCK_K):
        rows_k = k + k_offsets
        if k > 0:
            # Cross-warp handoff: rows of X below were written by tail
            # updates of earlier k-blocks, possibly by other warps. bar.sync
            # makes those global stores visible before we read them.
            tl.debug_barrier()
        if k == 0:
            y_block = tl.load(
                B_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
                mask=rhs_mask[None, :],
                other=0.0,
            )
        else:
            y_block = tl.load(
                X_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
                mask=rhs_mask[None, :],
                other=0.0,
            )

        diag_block = tl.load(L_ptr + L_base + rows_k * stride_L + rows_k)
        inv_diag = 1.0 / diag_block
        inv_diag = inv_diag * (2.0 - diag_block * inv_diag)

        y_block = y_block * inv_diag[:, None]
        for i in range(BLOCK_K):
            U_row = tl.load(
                L_ptr + L_base + (k + i) * stride_L + rows_k,
                mask=k_offsets > i,
                other=0.0,
            )
            w_i = tl.gather(y_block, tl.full([1, BLOCK_RHS], i, tl.int32), 0)
            y_block = y_block - (U_row * inv_diag)[:, None] * w_i

        tl.store(
            X_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
            y_block,
            mask=rhs_mask[None, :],
        )

        for m in range(k + BLOCK_K, N, BLOCK_M):
            rows_m = m + m_offsets
            # The software pipeliner may peel one more iteration (m == N) for
            # this loop, so guard every tail-row access explicitly; loads are
            # predicated off instead of reading past B/L/X.
            rows_m_mask = rows_m < N
            U_tile_km = tl.load(
                L_ptr + L_base + rows_k[:, None] * stride_L + rows_m[None, :],
                mask=rows_m_mask[None, :],
                other=0.0,
            )
            U_tile = tl.trans(U_tile_km)
            if k == 0:
                tail = tl.load(
                    B_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                    mask=rows_m_mask[:, None] & rhs_mask[None, :],
                    other=0.0,
                )
            else:
                tail = tl.load(
                    X_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                    mask=rows_m_mask[:, None] & rhs_mask[None, :],
                    other=0.0,
                )
            tail = tail - tl.dot(U_tile, y_block, input_precision="ieee")
            tl.store(
                X_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                tail,
                mask=rows_m_mask[:, None] & rhs_mask[None, :],
            )

    # Backward blocked TRSM: U * X = Y.
    for k in range(N - BLOCK_K, -1, -BLOCK_K):
        rows_k = k + k_offsets
        tl.debug_barrier()
        x_block = tl.load(
            X_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
            mask=rhs_mask[None, :],
            other=0.0,
        )

        diag_block = tl.load(L_ptr + L_base + rows_k * stride_L + rows_k)
        inv_diag = 1.0 / diag_block
        inv_diag = inv_diag * (2.0 - diag_block * inv_diag)

        x_block = x_block * inv_diag[:, None]
        for ii in range(BLOCK_K - 1, -1, -1):
            U_col = tl.load(
                L_ptr + L_base + rows_k * stride_L + (k + ii),
                mask=k_offsets < ii,
                other=0.0,
            )
            w_i = tl.gather(x_block, tl.full([1, BLOCK_RHS], ii, tl.int32), 0)
            x_block = x_block - (U_col * inv_diag)[:, None] * w_i

        tl.store(
            X_ptr + B_base + rows_k[:, None] * stride_B + rhs_cols[None, :],
            x_block,
            mask=rhs_mask[None, :],
        )

        for m in range(0, k, BLOCK_M):
            rows_m = m + m_offsets
            U_tile = tl.load(
                L_ptr + L_base + rows_m[:, None] * stride_L + rows_k[None, :]
            )
            head = tl.load(
                X_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                mask=rhs_mask[None, :],
                other=0.0,
            )
            head = head - tl.dot(U_tile, x_block, input_precision="ieee")
            tl.store(
                X_ptr + B_base + rows_m[:, None] * stride_B + rhs_cols[None, :],
                head,
                mask=rhs_mask[None, :],
            )


def cholesky_solve(B, L, upper=False, *, _out=None):
    """Run the generic dispatcher with the hardened blocked kernels.

    The generic implementation resolves the two blocked kernels through its
    module globals at call time, so substituting them for the duration of the
    call keeps its validation, layout normalization and every other dispatch
    path unchanged. The masked kernels are numerically identical for valid
    iterations, so a concurrent caller observing the swap is unaffected.
    """
    logger.debug("GEMS_HYGON CHOLESKY_SOLVE")
    prev_upper = _generic.cholesky_solve_blocked_upper_kernel
    prev_lower = _generic.cholesky_solve_blocked_lower_kernel
    _generic.cholesky_solve_blocked_upper_kernel = cholesky_solve_blocked_upper_kernel
    _generic.cholesky_solve_blocked_lower_kernel = cholesky_solve_blocked_lower_kernel
    try:
        return _generic.cholesky_solve(B, L, upper=upper, _out=_out)
    finally:
        _generic.cholesky_solve_blocked_upper_kernel = prev_upper
        _generic.cholesky_solve_blocked_lower_kernel = prev_lower


def cholesky_solve_out(B, L, upper=False, *, out):
    """Out variant, mirroring the generic direct-write/fallback split."""
    logger.debug("GEMS_HYGON CHOLESKY_SOLVE_OUT")
    _generic._check_cholesky_solve_out(B, out)
    if _generic._can_write_cholesky_solve_out_direct(B, L, out):
        return cholesky_solve(B, L, upper=upper, _out=out)
    result = cholesky_solve(B, L, upper=upper)
    return _generic._copy_cholesky_solve_out(result, out)
