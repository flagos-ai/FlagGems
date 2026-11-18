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

from flag_gems.ops.triangular_solve import _triangular_solve

logger = logging.getLogger(__name__)


@triton.jit
def _vector_solve_kernel(
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
    COPY: tl.constexpr,
    FLAT: tl.constexpr = False,
):
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
    if FLAT:
        width = K + tl.cdiv(N, 32) * tl.cdiv(N, 32)
        col = tl.program_id(0) % width
        batch = (tl.program_id(0) // width).to(tl.int64)
    else:
        col = tl.program_id(1)
        batch = tl.program_id(0).to(tl.int64)
    if INDEX64:
        col = col.to(tl.int64)
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
    solve_tiles = K
    if COPY and col >= solve_tiles:
        # Independent copy programs preserve the complete original A, including
        # its unused triangle, while solve programs compute the other output.
        tile = col - solve_tiles
        tiles = tl.cdiv(N, 32)
        cr = (tile // tiles) * 32 + tl.arange(0, 32)
        cc = (tile % tiles) * 32 + tl.arange(0, 32)
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
        # A single warp retains the solved vector; no global row writeback or
        # cross-warp layout conversion is needed between substitutions.
        rr = N - 1 - rows if UPPER else rows
        x = tl.full((ROWS,), 0.0, X.dtype.element_ty)
        if not UNIT:
            diagonal = tl.load(
                A + rr * (A_STRIDES.value[-2] + A_STRIDES.value[-1]),
                rows < N,
                other=1.0,
            )
            inverses = 1.0 / diagonal
            regular = (
                tl.sum(
                    (
                        (tl.abs(inverses) == float("inf"))
                        | (inverses == 0)
                        | (inverses != inverses)
                    ).to(tl.int32),
                    0,
                )
                == 0
            )
        for j in range(N):
            row = N - 1 - j if UPPER else j
            a = tl.load(
                A + row * A_STRIDES.value[-2] + rr * A_STRIDES.value[-1],
                rows < j,
                other=0.0,
            )
            rhs = tl.load(B + row * B_STRIDES.value[-2] + col * B_STRIDES.value[-1])
            value = rhs - tl.sum(a * x, 0)
            if not UNIT:
                if regular:
                    value *= tl.sum(tl.where(rows == j, inverses, 0.0), 0)
                else:
                    d = tl.load(A + row * (A_STRIDES.value[-2] + A_STRIDES.value[-1]))
                    value /= d
            x = tl.where(rows == j, value, x)
        tl.store(X + batch * N * K + col * N + rr, x, rows < N)


@triton.jit
def _prepare_panels_kernel(
    A,
    I,
    NORMALIZED,
    GOOD,
    K: tl.constexpr,
    COEFFICIENT,
    ORIGINAL,
    C_STRIDES: tl.constexpr,
    O_STRIDES: tl.constexpr,
    N: tl.constexpr,
    BATCH: tl.constexpr,
    A_STRIDES: tl.constexpr,
    UPPER: tl.constexpr,
    UNIT: tl.constexpr,
    PANEL: tl.constexpr,
    FLAT: tl.constexpr = False,
):
    # Select wide address arithmetic at compile time, before any product.
    BATCH_COUNT: tl.constexpr = math.prod(BATCH.value)
    A_STRIDES_SPAN: tl.constexpr = N * (
        A_STRIDES.value[-2] + A_STRIDES.value[-1]
    ) + BATCH_COUNT * math.fsum(A_STRIDES.value)
    C_STRIDES_SPAN: tl.constexpr = N * (
        C_STRIDES.value[-2] + C_STRIDES.value[-1]
    ) + BATCH_COUNT * math.fsum(C_STRIDES.value)
    O_STRIDES_SPAN: tl.constexpr = N * (
        O_STRIDES.value[-2] + O_STRIDES.value[-1]
    ) + BATCH_COUNT * math.fsum(O_STRIDES.value)
    INDEX64: tl.constexpr = (
        BATCH_COUNT * N * K > 2147483647
        or A_STRIDES_SPAN > 2147483647
        or C_STRIDES_SPAN > 2147483647
        or O_STRIDES_SPAN > 2147483647
        or BATCH_COUNT * ((N + PANEL - 1) // PANEL) * PANEL * PANEL > 2147483647
        or BATCH_COUNT * K * (((N + PANEL - 1) // PANEL) + 1) > 2147483647
    )
    if FLAT:
        panels = tl.cdiv(N, PANEL)
        if INDEX64:
            panels = panels.to(tl.int64)
        width = panels + tl.cdiv(N, 32) * tl.cdiv(N, 32)
        block = tl.program_id(0) % width
        batch = (tl.program_id(0) // width).to(tl.int64)
    else:
        block = tl.program_id(1)
        batch = tl.program_id(0).to(tl.int64)
    if INDEX64:
        block = block.to(tl.int64)
    remainder = batch
    a_offset = tl.full((), 0, tl.int64)
    c_offset = tl.full((), 0, tl.int64)
    o_offset = tl.full((), 0, tl.int64)
    for dim in tl.static_range(len(BATCH) - 1, -1, -1):
        coordinate = remainder % BATCH.value[dim]
        remainder //= BATCH.value[dim]
        a_offset += coordinate * A_STRIDES.value[dim]
        c_offset += coordinate * C_STRIDES.value[dim]
        o_offset += coordinate * O_STRIDES.value[dim]
    A += a_offset
    solve_tiles = tl.cdiv(N, PANEL)
    if INDEX64:
        solve_tiles = solve_tiles.to(tl.int64)
    if block >= solve_tiles:
        # Independent copy programs preserve the complete original A, including
        # its unused triangle, while solve programs compute the other output.
        tile = block - solve_tiles
        tiles = tl.cdiv(N, 32)
        cr = (tile // tiles) * 32 + tl.arange(0, 32)
        cc = (tile % tiles) * 32 + tl.arange(0, 32)
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
        if UPPER:
            # Normalize the solve matrix while preserving the untouched original
            # in COEFFICIENT. Both writes finish before the solve launch.
            value = tl.load(
                A
                + (N - 1 - cr[:, None]) * A_STRIDES.value[-2]
                + (N - 1 - cc[None, :]) * A_STRIDES.value[-1],
                mask,
                other=0,
            )
            tl.store(
                NORMALIZED + batch * N * N + cr[:, None] * N + cc[None, :], value, mask
            )
    else:
        r = tl.arange(0, PANEL)
        row = block * PANEL + r
        ar = N - 1 - row if UPPER else row
        a = tl.load(
            A + ar[:, None] * A_STRIDES.value[-2] + ar[None, :] * A_STRIDES.value[-1],
            (row[:, None] < N) & (row[None, :] < N),
            other=0,
        )
        if UNIT:
            inv = tl.full(
                (PANEL,),
                1.0,
                tl.float64 if A.dtype.element_ty == tl.float64 else tl.float32,
            )
        else:
            diagonal = tl.load(
                A + ar * (A_STRIDES.value[-2] + A_STRIDES.value[-1]), row < N, other=1
            )
            inv = 1.0 / diagonal
        # Forward elimination of the identity builds a full-precision inverse
        # for each independent diagonal panel using the input precision.
        v = tl.where(r[:, None] == r[None, :], 1.0, 0.0).to(a.dtype)
        for j in range(PANEL):
            selected = tl.sum(tl.where(r[:, None] == j, v, 0.0), 0)
            reciprocal = tl.sum(tl.where(r == j, inv, 0.0), 0)
            selected *= reciprocal
            column = tl.sum(tl.where(r[None, :] == j, a, 0.0), 1)
            v = tl.where(r[:, None] > j, v - column[:, None] * selected[None, :], v)
            v = tl.where(r[:, None] == j, selected[None, :], v)
        valid = (
            tl.sum(tl.sum(((tl.abs(v) == float("inf")) | (v != v)).to(tl.int32), 0), 0)
            == 0
        )
        valid = valid & (tl.sum((inv == 0).to(tl.int32), 0) == 0)
        tl.store(GOOD + batch * tl.cdiv(N, PANEL) + block, valid)
        tl.store(
            I
            + (batch * tl.cdiv(N, PANEL) + block) * PANEL * PANEL
            + r[:, None] * PANEL
            + r[None, :],
            v,
        )


@triton.jit
def _serial_panels_kernel(
    A,
    B,
    X,
    I,
    GOOD,
    N: tl.constexpr,
    K: tl.constexpr,
    BATCH: tl.constexpr,
    A_STRIDES: tl.constexpr,
    B_STRIDES: tl.constexpr,
    UPPER: tl.constexpr,
    UNIT: tl.constexpr,
    ROWS: tl.constexpr,
    PANEL: tl.constexpr,
    FLAT: tl.constexpr = False,
):
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
    INDEX64: tl.constexpr = (
        BATCH_COUNT * N * K > 2147483647
        or A_STRIDES_SPAN > 2147483647
        or B_STRIDES_SPAN > 2147483647
    )
    rows = tl.arange(0, ROWS)
    if INDEX64:
        rows = rows.to(tl.int64)
    if FLAT:
        width = K
        col = tl.program_id(0) % width
        batch = (tl.program_id(0) // width).to(tl.int64)
    else:
        col = tl.program_id(1)
        batch = tl.program_id(0).to(tl.int64)
    if INDEX64:
        col = col.to(tl.int64)
    remainder = batch
    a_offset = tl.full((), 0, tl.int64)
    b_offset = tl.full((), 0, tl.int64)
    for dim in tl.static_range(len(BATCH) - 1, -1, -1):
        coordinate = remainder % BATCH.value[dim]
        remainder //= BATCH.value[dim]
        a_offset += coordinate * A_STRIDES.value[dim]
        b_offset += coordinate * B_STRIDES.value[dim]
    A += a_offset
    B += b_offset
    p = tl.arange(0, PANEL)
    if INDEX64:
        p = p.to(tl.int64)
    # Remap both axes so all triangular substitutions run forward.
    rr = N - 1 - rows if UPPER else rows
    for start in range(0, N, PANEL):
        pr = start + p
        ar = N - 1 - pr if UPPER else pr
        aa = tl.load(
            A + pr[:, None] * A_STRIDES.value[-2] + pr[None, :] * A_STRIDES.value[-1],
            (pr[:, None] < N) & (pr[None, :] < N),
            other=0,
        )
        xp = tl.load(
            B + ar * B_STRIDES.value[-2] + col * B_STRIDES.value[-1],
            pr < N,
            other=0,
        )
        if N <= ROWS:
            if start > 0:
                previous = tl.load(
                    X + batch * N * K + col * N + rr, rows < start, other=0
                )
                left = tl.load(
                    A
                    + pr[:, None] * A_STRIDES.value[-2]
                    + rows[None, :] * A_STRIDES.value[-1],
                    (pr[:, None] < N) & (rows[None, :] < start),
                    other=0,
                )
                xp = xp - tl.sum(left * previous[None, :], 1)
        else:
            for previous_start in range(0, start, ROWS):
                previous_rows = previous_start + rows
                previous_ar = N - 1 - previous_rows if UPPER else previous_rows
                previous = tl.load(
                    X + batch * N * K + col * N + previous_ar,
                    previous_rows < start,
                    other=0,
                )
                left = tl.load(
                    A
                    + pr[:, None] * A_STRIDES.value[-2]
                    + previous_rows[None, :] * A_STRIDES.value[-1],
                    (pr[:, None] < N) & (previous_rows[None, :] < start),
                    other=0,
                )
                xp = xp - tl.sum(left * previous[None, :], 1)
        block = start // PANEL
        good = tl.load(GOOD + batch * tl.cdiv(N, PANEL) + block) != 0
        if good:
            inv = tl.load(
                I
                + (batch * tl.cdiv(N, PANEL) + block) * PANEL * PANEL
                + p[:, None] * PANEL
                + p[None, :]
            )
            candidate = tl.sum(inv * xp[None, :], 1)
            good = (
                tl.sum(
                    ((tl.abs(candidate) == float("inf")) | (candidate != candidate)).to(
                        tl.int32
                    ),
                    0,
                )
                == 0
            )
            if good:
                xp = candidate
        if not good:
            if not UNIT:
                diag = tl.load(
                    A + pr * (A_STRIDES.value[-2] + A_STRIDES.value[-1]),
                    pr < N,
                    other=1,
                )
                inverses = 1.0 / diag
                regular = (
                    tl.sum(
                        (
                            (tl.abs(inverses) == float("inf"))
                            | (inverses == 0)
                            | (inverses != inverses)
                        ).to(tl.int32),
                        0,
                    )
                    == 0
                )
            for j in range(PANEL):
                v = tl.gather(xp, tl.full((1,), j, tl.int32), 0).reshape(())
                if not UNIT:
                    if regular:
                        iv = tl.gather(inverses, tl.full((1,), j, tl.int32), 0).reshape(
                            ()
                        )
                        v = v * iv
                    else:
                        d = tl.sum(
                            tl.where((p[:, None] == j) & (p[None, :] == j), aa, 0),
                            0,
                        )
                        d = tl.sum(d, 0)
                        v = v / tl.where(start + j < N, d, 1)
                av = tl.gather(aa, tl.full((PANEL, 1), j, tl.int32), 1).reshape(
                    (PANEL,)
                )
                xp = tl.where(p > j, xp - av * v, xp)
                xp = tl.where(p == j, v, xp)
        tl.store(X + batch * N * K + col * N + ar, xp, pr < N)
        tl.debug_barrier()


@triton.jit
def _reverse_rhs_kernel(
    S,
    D,
    N: tl.constexpr,
    K: tl.constexpr,
    BATCH: tl.constexpr,
    STRIDES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    batch = offset // (N * K)
    col = offset // N % K
    row = offset % N
    remainder = batch
    source = tl.full((BLOCK,), 0, tl.int64)
    for dim in tl.static_range(len(BATCH) - 1, -1, -1):
        source += (remainder % BATCH.value[dim]) * STRIDES.value[dim]
        remainder //= BATCH.value[dim]
    source += (N - 1 - row) * STRIDES.value[-2] + col * STRIDES.value[-1]
    mask = offset < math.prod(BATCH.value) * N * K
    value = tl.load(S + source, mask, other=0)
    tl.store(D + offset, value, mask)


def _solve_thead(A, B, *, upper, unitriangular, out=None, coefficient, original):
    n, k = B.shape[-2:]
    batch = tuple(B.shape[:-2])
    result = (
        torch.empty((*batch, k, n), dtype=B.dtype, device=B.device).mT
        if out is None
        else out
    )
    rows = min(1024, triton.next_power_of_2(n))
    if n <= 32:
        width = k + triton.cdiv(n, 32) ** 2
        flat = width > 65535
        grid = (math.prod(batch) * width,) if flat else (math.prod(batch), width)
        _vector_solve_kernel[grid](
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
            rows,
            True,
            flat,
            num_warps=1,
        )
    else:
        panel = 32 if n >= 512 and B.dtype == torch.float32 else 16
        panels = triton.cdiv(n, panel)
        inverse = torch.empty(
            (*batch, panels, panel, panel), dtype=A.dtype, device=A.device
        )
        good = torch.empty((*batch, panels), dtype=torch.int32, device=A.device)
        width = panels + triton.cdiv(n, 32) ** 2
        flat = width > 65535
        grid = (math.prod(batch) * width,) if flat else (math.prod(batch), width)
        normalized = (
            torch.empty_like(A, memory_format=torch.contiguous_format) if upper else A
        )
        _prepare_panels_kernel[grid](
            A,
            inverse,
            normalized,
            good,
            k,
            coefficient,
            original,
            tuple(coefficient.stride()),
            tuple(original.stride()),
            n,
            batch,
            tuple(A.stride()),
            upper,
            unitriangular,
            panel,
            flat,
            num_warps=1,
        )
        # The full clone is ready after preparation. Its transpose gives the
        # transposed system a row-contiguous view without another copy kernel.
        solve_a = (
            normalized
            if upper
            else (
                coefficient.mT
                if A.stride() == original.mT.stride() and coefficient.stride(-2) == 1
                else A
            )
        )
        # Kernel launch ordering completes all inverses before substitution.
        # A single program owns each RHS; there is no cross-program spin wait.
        flat = k > 65535
        grid = (math.prod(batch) * k,) if flat else (math.prod(batch), k)
        solve_b = B
        solve_x = result
        if upper:
            solve_b = torch.empty((*batch, k, n), dtype=B.dtype, device=B.device).mT
            solve_x = torch.empty_like(solve_b)
            _reverse_rhs_kernel[(triton.cdiv(B.numel(), 256),)](
                B,
                solve_b,
                n,
                k,
                batch,
                tuple(B.stride()),
                256,
            )
        _serial_panels_kernel[grid](
            solve_a,
            solve_b,
            solve_x,
            inverse,
            good,
            n,
            k,
            batch,
            tuple(solve_a.stride()),
            tuple(solve_b.stride()),
            False,
            unitriangular,
            rows,
            panel,
            flat,
            num_warps=(
                min(4, max(1, rows // 128))
                if B.dtype == torch.float64
                else max(1, rows * panel // 4096)
            ),
        )
        if upper:
            _reverse_rhs_kernel[(triton.cdiv(B.numel(), 256),)](
                solve_x,
                result,
                n,
                k,
                batch,
                tuple(solve_x.stride()),
                256,
            )
    return result


def triangular_solve(B, A, upper=True, transpose=False, unitriangular=False):
    logger.debug("GEMS_THEAD TRIANGULAR_SOLVE")
    return _triangular_solve(
        B,
        A,
        upper,
        transpose,
        unitriangular,
        _solve_thead,
        solver_out=True,
        solver_coeff=True,
    )


def triangular_solve_out(
    B, A, upper=True, transpose=False, unitriangular=False, *, X, M
):
    logger.debug("GEMS_THEAD TRIANGULAR_SOLVE_OUT")
    return _triangular_solve(
        B,
        A,
        upper,
        transpose,
        unitriangular,
        _solve_thead,
        X,
        M,
        solver_out=True,
        solver_coeff=True,
    )
