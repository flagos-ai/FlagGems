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
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.linalg_solve_triangular import _kslice_trsm_kernel_notle
from flag_gems.ops.triangular_solve import _copy_matrix, _triangular_solve
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _resident_solve_kernel(
    A,
    B,
    X,
    COEFFICIENT,
    ORIGINAL,
    C_STRIDES: tl.constexpr,
    O_STRIDES: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BATCH: tl.constexpr,
    A_STRIDES: tl.constexpr,
    B_STRIDES: tl.constexpr,
    UPPER: tl.constexpr,
    UNIT: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    # One program owns complete RHS columns. Keep their dependent updates in
    # registers, reading the original input strides without materialization.
    # Select wide address arithmetic at compile time, before any product.
    BATCH_COUNT: tl.constexpr = math.prod(BATCH.value)
    A_STRIDES_SPAN: tl.constexpr = N * (
        A_STRIDES.value[-2] + A_STRIDES.value[-1]
    ) + BATCH_COUNT * math.fsum(A_STRIDES.value)
    B_STRIDES_SPAN: tl.constexpr = (
        N * B_STRIDES.value[-2]
        + K * B_STRIDES.value[-1]
        + BATCH_COUNT * math.fsum(B_STRIDES.value)
    )
    C_STRIDES_SPAN: tl.constexpr = N * (
        C_STRIDES.value[-2] + C_STRIDES.value[-1]
    ) + BATCH_COUNT * math.fsum(C_STRIDES.value)
    O_STRIDES_SPAN: tl.constexpr = N * (
        O_STRIDES.value[-2] + O_STRIDES.value[-1]
    ) + BATCH_COUNT * math.fsum(O_STRIDES.value)
    INDEX64: tl.constexpr = (
        BATCH_COUNT * N * K > 2147483647
        or A_STRIDES_SPAN > 2147483647
        or B_STRIDES_SPAN > 2147483647
        or C_STRIDES_SPAN > 2147483647
        or O_STRIDES_SPAN > 2147483647
    )
    rows = tl.arange(0, ROWS)
    if INDEX64:
        rows = rows.to(tl.int64)
    cols = tl.arange(0, COLS)
    batch = tl.program_id(0)
    col_tile = tl.program_id(1)
    if INDEX64:
        batch = batch.to(tl.int64)
        col_tile = col_tile.to(tl.int64)
    remainder = batch
    a_offset = 0
    b_offset = 0
    c_offset = 0
    o_offset = 0
    for dim in tl.static_range(len(BATCH) - 1, -1, -1):
        coord = remainder - (remainder // BATCH[dim]) * BATCH[dim]
        remainder = remainder // BATCH[dim]
        a_offset += coord * A_STRIDES[dim]
        b_offset += coord * B_STRIDES[dim]
        c_offset += coord * C_STRIDES[dim]
        o_offset += coord * O_STRIDES[dim]
    a_ptr = A + a_offset
    b_ptr = B + b_offset
    solve_tiles = tl.cdiv(K, COLS)
    if col_tile >= solve_tiles:
        # Copy the complete original coefficient matrix in the same launch.
        # Copy programs and solve programs write disjoint output allocations.
        tile = col_tile - solve_tiles
        tiles = tl.cdiv(N, 32)
        cr = (tile // tiles) * 32 + tl.arange(0, 32)
        cc = (tile % tiles) * 32 + tl.arange(0, 32)
        mask = (cr[:, None] < N) & (cc[None, :] < N)
        value = tl.load(
            ORIGINAL
            + o_offset
            + cr[:, None] * O_STRIDES[-2]
            + cc[None, :] * O_STRIDES[-1],
            mask=mask,
            other=0.0,
        )
        tl.store(
            COEFFICIENT
            + c_offset
            + cr[:, None] * C_STRIDES[-2]
            + cc[None, :] * C_STRIDES[-1],
            value,
            mask,
        )
    else:
        rhs = col_tile * COLS + cols
        valid = (rows[None, :] < N) & (rhs[:, None] < K)
        x = tl.load(
            b_ptr + rows[None, :] * B_STRIDES[-2] + rhs[:, None] * B_STRIDES[-1],
            mask=valid,
            other=0.0,
        )
        if not UNIT:
            diagonal = tl.load(
                a_ptr + rows * (A_STRIDES[-2] + A_STRIDES[-1]),
                mask=rows < N,
                other=1.0,
            )
            inverse = 1.0 / diagonal
        for step in range(N):
            row = N - 1 - step if UPPER else step
            if INDEX64:
                row = row.to(tl.int64)
            selected = rows == row
            value = tl.sum(tl.where(selected[None, :], x, 0.0), axis=1)
            if not UNIT:
                value *= tl.sum(tl.where(selected, inverse, 0.0), axis=0)
            remaining = rows < row if UPPER else rows > row
            # Mask before loading so a poisoned unused triangle or implicit
            # unit diagonal can never enter the arithmetic.
            column = tl.load(
                a_ptr + rows * A_STRIDES[-2] + row * A_STRIDES[-1],
                mask=(rows < N) & remaining,
                other=0.0,
            )
            x = tl.where(remaining[None, :], x - value[:, None] * column[None, :], x)
            x = tl.where(selected[None, :], value[:, None], x)
        tl.store(X + batch * N * K + rhs[:, None] * N + rows[None, :], x, valid)


def _copy_wide(dst, src):
    # MUSA wide vector addressing can fail despite i64 LLVM GEPs.
    # Tensor views carry the high bits in their base pointers instead.
    limit = 2147483647 // src.element_size()
    shape = src.shape
    src_strides, dst_strides = src.stride(), dst.stride()
    if (
        max(
            sum((size - 1) * stride for size, stride in zip(shape, src_strides)),
            sum((size - 1) * stride for size, stride in zip(shape, dst_strides)),
        )
        > limit
    ):
        axis = max(
            (i for i, size in enumerate(shape) if size > 1),
            key=lambda i: max(src_strides[i], dst_strides[i]),
        )
        for index in range(shape[axis]):
            _copy_wide(dst.select(axis, index), src.select(axis, index))
    else:
        while src.ndim < 2:
            src, dst = src.unsqueeze(0), dst.unsqueeze(0)
        _copy_matrix(dst, src)


def _solve_wide(A, B, *, upper, unitriangular, out, coefficient, original):
    dense_original = torch.empty(original.shape, dtype=A.dtype, device=A.device)
    dense_b = torch.empty(B.shape, dtype=B.dtype, device=B.device)
    _copy_wide(dense_original, original)
    _copy_wide(dense_b, B)
    dense_a = dense_original if A.stride() == original.stride() else dense_original.mT
    dense_coefficient = torch.empty_like(dense_original)
    result = _resident_solve(
        dense_a,
        dense_b,
        upper=upper,
        unitriangular=unitriangular,
        out=out,
        coefficient=dense_coefficient,
        original=dense_original,
    )
    _copy_wide(coefficient, dense_coefficient)
    return result


def _resident_solve(A, B, *, upper, unitriangular, out=None, coefficient, original):
    n, k = B.shape[-2:]
    batch = tuple(B.shape[:-2])
    # Keep wide address components in tensor-view base pointers, bypassing
    # the vendor compiler's failing vector-offset path for strided inputs.
    limit = 2147483647 // B.element_size()
    if max(A.numel(), B.numel()) < limit and any(
        sum((size - 1) * stride for size, stride in zip(t.shape, t.stride())) > limit
        for t in (A, B, coefficient, original)
    ):
        return _solve_wide(
            A,
            B,
            upper=upper,
            unitriangular=unitriangular,
            out=out,
            coefficient=coefficient,
            original=original,
        )
    result = (
        torch.empty((*batch, k, n), dtype=B.dtype, device=B.device).mT
        if out is None
        else out
    )
    if n > 128:
        _copy_matrix(coefficient, original)
        count = math.prod(batch)
        coefficients = A.contiguous().reshape(count, n, n)
        # The established right-looking kernel uses a row-major RHS. A
        # single RHS already has that physical layout in the final output.
        workspace = (
            result if k == 1 else torch.empty(B.shape, dtype=B.dtype, device=B.device)
        )
        _copy_matrix(workspace, B)
        rhs = workspace.reshape(count, n, k)
        # Keep the pure Triton path: MUSA's TLE allocator rejects the
        # nv_mma_shared_layout used by the alternate generic kernel.
        for index in range(count):
            _kslice_trsm_kernel_notle[(triton.cdiv(k, 16),)](
                coefficients[index],
                rhs[index],
                n,
                k,
                n,
                k,
                32,
                16,
                128,
                False,
                False,
                upper,
                unitriangular,
                num_warps=4,
            )
        if workspace is not result:
            _copy_matrix(result, workspace)
        return result
    cols = min(4, triton.next_power_of_2(k))
    _resident_solve_kernel[
        (math.prod(batch), triton.cdiv(k, cols) + triton.cdiv(n, 32) ** 2)
    ](
        A,
        B,
        result,
        coefficient,
        original,
        tuple(coefficient.stride()),
        tuple(original.stride()),
        n,
        k,
        batch,
        tuple(A.stride()),
        tuple(B.stride()),
        upper,
        unitriangular,
        triton.next_power_of_2(n),
        cols,
        num_warps=4,
    )
    return result


def triangular_solve(B, A, upper=True, transpose=False, unitriangular=False):
    logger.debug("GEMS_MTHREADS TRIANGULAR_SOLVE")
    return _triangular_solve(
        B,
        A,
        upper,
        transpose,
        unitriangular,
        _resident_solve,
        solver_out=True,
        solver_coeff=True,
        copy_fn=_copy_wide,
    )


def triangular_solve_out(
    B, A, upper=True, transpose=False, unitriangular=False, *, X, M
):
    logger.debug("GEMS_MTHREADS TRIANGULAR_SOLVE_OUT")
    return _triangular_solve(
        B,
        A,
        upper,
        transpose,
        unitriangular,
        _resident_solve,
        X,
        M,
        solver_out=True,
        solver_coeff=True,
        copy_fn=_copy_wide,
    )
