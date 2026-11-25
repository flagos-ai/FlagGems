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

from flag_gems.ops.triangular_solve import _copy_matrix, _triangular_solve
from flag_gems.utils import libentry

from .linalg_solve_triangular import linalg_solve_triangular

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
    FLAT_GRID: tl.constexpr,
    INDEX64: tl.constexpr,
):
    rows = tl.arange(0, ROWS)
    if INDEX64:
        rows = rows.to(tl.int64)
    batch = tl.program_id(0).to(tl.int64)
    if FLAT_GRID:
        tiles = tl.cdiv(K, COLS) + tl.cdiv(N, 32) * tl.cdiv(N, 32)
        col_tile = batch % tiles
        batch //= tiles
    else:
        col_tile = tl.program_id(1)
    if INDEX64:
        col_tile = col_tile.to(tl.int64)
    cols = col_tile * COLS + tl.arange(0, COLS)
    if INDEX64:
        cols = col_tile * COLS + tl.arange(0, COLS).to(tl.int64)
    remainder = batch
    a_offset = tl.full((), 0, tl.int64)
    b_offset = tl.full((), 0, tl.int64)
    c_offset = tl.full((), 0, tl.int64)
    o_offset = tl.full((), 0, tl.int64)
    for dim in tl.static_range(len(BATCH) - 1, -1, -1):
        coordinate = remainder % BATCH.value[dim]
        remainder //= BATCH.value[dim]
        a_offset += coordinate * A_STRIDES.value[dim]
        b_offset += coordinate * B_STRIDES.value[dim]
        c_offset += coordinate * C_STRIDES.value[dim]
        o_offset += coordinate * O_STRIDES.value[dim]
    A += a_offset
    B += b_offset
    solve_tiles = tl.cdiv(K, COLS)
    if col_tile >= solve_tiles:
        # Independent copy programs preserve the complete original A, including
        # its unused triangle, while solve programs compute the other output.
        tile = col_tile - solve_tiles
        tiles = tl.cdiv(N, 32)
        cr = (tile // tiles) * 32 + tl.arange(0, 32)
        cc = (tile % tiles) * 32 + tl.arange(0, 32)
        if INDEX64:
            cr = cr.to(tl.int64)
            cc = cc.to(tl.int64)
        mask = (cr[:, None] < N) & (cc[None, :] < N)
        value = tl.load(
            ORIGINAL
            + o_offset
            + cr[:, None] * O_STRIDES.value[-2]
            + cc[None, :] * O_STRIDES.value[-1],
            mask=mask,
            other=0.0,
        )
        tl.store(
            COEFFICIENT
            + c_offset
            + cr[:, None] * C_STRIDES.value[-2]
            + cc[None, :] * C_STRIDES.value[-1],
            value,
            mask,
        )
    else:
        valid = (rows[None, :] < N) & (cols[:, None] < K)
        x = tl.load(
            B
            + rows[None, :] * B_STRIDES.value[-2]
            + cols[:, None] * B_STRIDES.value[-1],
            mask=valid,
            other=0.0,
        )
        # Each program retains complete RHS columns across every substitution step,
        # avoiding input materialization and a separate launch for each batch.
        for step in range(N):
            row = N - 1 - step if UPPER else step
            if INDEX64:
                row = row.to(tl.int64)
            selected = rows == row
            value = tl.sum(tl.where(selected[None, :], x, 0.0), axis=1)
            if not UNIT:
                diagonal = tl.load(
                    A + row * (A_STRIDES.value[-2] + A_STRIDES.value[-1])
                )
                value = value / diagonal
            remaining = rows < row if UPPER else rows > row
            column = tl.load(
                A + rows * A_STRIDES.value[-2] + row * A_STRIDES.value[-1],
                mask=(rows < N) & remaining,
                other=0.0,
            )
            x = tl.where(remaining[None, :], x - value[:, None] * column[None, :], x)
            x = tl.where(selected[None, :], value[:, None], x)
        tl.store(X + batch * N * K + cols[:, None] * N + rows[None, :], x, valid)


def _safe_copy_matrix(dst, src):
    # CoreX 3.1 can truncate wide vector offsets despite i64 LLVM GEPs.
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
            _safe_copy_matrix(dst.select(axis, index), src.select(axis, index))
    else:
        while src.ndim < 2:
            src, dst = src.unsqueeze(0), dst.unsqueeze(0)
        _copy_matrix(dst, src)


def _solve_wide(A, B, *, upper, unitriangular, out, coefficient, original):
    dense_original = torch.empty(original.shape, dtype=A.dtype, device=A.device)
    dense_b = torch.empty(B.shape, dtype=B.dtype, device=B.device)
    _safe_copy_matrix(dense_original, original)
    _safe_copy_matrix(dense_b, B)
    dense_a = dense_original if A.stride() == original.stride() else dense_original.mT
    dense_coefficient = torch.empty_like(dense_original)
    result = _solve(
        dense_a,
        dense_b,
        upper=upper,
        unitriangular=unitriangular,
        out=out,
        coefficient=dense_coefficient,
        original=dense_original,
    )
    _safe_copy_matrix(coefficient, dense_coefficient)
    return result


def _solve(A, B, *, upper, unitriangular, out=None, coefficient, original):
    n, k = B.shape[-2:]
    batch = tuple(B.shape[:-2])
    # CoreX 3.1 cannot evaluate Python math helpers inside a JIT function.
    # Reuse these stride tuples for both the bound and the kernel arguments.
    a_strides = tuple(A.stride())
    b_strides = tuple(B.stride())
    c_strides = tuple(coefficient.stride())
    o_strides = tuple(original.stride())
    batches = math.prod(batch)
    index64 = (
        max(
            batches * n * k,
            n * (a_strides[-2] + a_strides[-1]) + batches * sum(a_strides),
            n * b_strides[-2] + k * b_strides[-1] + batches * sum(b_strides),
            n * (c_strides[-2] + c_strides[-1]) + batches * sum(c_strides),
            n * (o_strides[-2] + o_strides[-1]) + batches * sum(o_strides),
        )
        > 2147483647
    )
    if index64 and max(A.numel(), B.numel()) < 536870912:
        return _solve_wide(
            A,
            B,
            upper=upper,
            unitriangular=unitriangular,
            out=out,
            coefficient=coefficient,
            original=original,
        )
    if n > 128:
        _copy_matrix(coefficient, original)
        result = linalg_solve_triangular(A, B, upper=upper, unitriangular=unitriangular)
        if out is not None:
            _copy_matrix(out, result)
            return out
        return result
    result = (
        torch.empty((*batch, k, n), dtype=B.dtype, device=B.device).mT
        if out is None
        else out
    )
    cols = min(4, triton.next_power_of_2(k))
    tiles = triton.cdiv(k, cols) + triton.cdiv(n, 32) ** 2
    flat_grid = tiles > 65535
    # CoreX's CUDA-compatible launcher limits grid.y to 65535. Keep the
    # normal launch geometry, and flatten only unusually wide RHS matrices.
    grid = (batches * tiles,) if flat_grid else (batches, tiles)
    _resident_solve_kernel[grid](
        A,
        B,
        result,
        coefficient,
        original,
        c_strides,
        o_strides,
        n,
        k,
        batch,
        a_strides,
        b_strides,
        upper,
        unitriangular,
        triton.next_power_of_2(n),
        cols,
        flat_grid,
        index64,
        num_warps=4,
    )
    return result


def triangular_solve(B, A, upper=True, transpose=False, unitriangular=False):
    logger.debug("GEMS_ILUVATAR TRIANGULAR_SOLVE")
    return _triangular_solve(
        B,
        A,
        upper,
        transpose,
        unitriangular,
        _solve,
        solver_out=True,
        solver_coeff=True,
        copy_fn=_safe_copy_matrix,
    )


def triangular_solve_out(
    B, A, upper=True, transpose=False, unitriangular=False, *, X, M
):
    logger.debug("GEMS_ILUVATAR TRIANGULAR_SOLVE_OUT")
    return _triangular_solve(
        B,
        A,
        upper,
        transpose,
        unitriangular,
        _solve,
        X,
        M,
        solver_out=True,
        solver_coeff=True,
        copy_fn=_safe_copy_matrix,
    )
