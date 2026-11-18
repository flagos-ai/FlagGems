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
def _metax_vector_kernel(
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
        # A single wave retains the solved vector; no global row writeback or
        # inter-wave layout conversion is needed between substitutions.
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
def _metax_prepare_inverse(
    A,
    I,
    GOOD,
    STATE,
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
        width = (
            panels + tl.cdiv(N, 32) * tl.cdiv(N, 32) + tl.cdiv(K * (panels + 1), 256)
        )
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
    copy_tiles = tl.cdiv(N, 32) * tl.cdiv(N, 32)
    if block >= solve_tiles + copy_tiles:
        # Fixed-size reset tiles keep initialization bounded for any RHS count.
        state_size = K * (solve_tiles + 1)
        offsets = (block - solve_tiles - copy_tiles) * 256 + tl.arange(0, 256)
        tl.store(STATE + batch * state_size + offsets, 0, offsets < state_size)
    elif block >= solve_tiles:
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
        # for each independent diagonal panel. Never use MetaX FP64 tl.dot.
        v = tl.where(r[:, None] == r[None, :], 1.0, 0.0).to(a.dtype)
        for j in tl.static_range(PANEL):
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
def _metax_inverse_solve(
    A,
    B,
    X,
    I,
    GOOD,
    STATE,
    N: tl.constexpr,
    K: tl.constexpr,
    BATCH: tl.constexpr,
    A_STRIDES: tl.constexpr,
    B_STRIDES: tl.constexpr,
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
    B_STRIDES_SPAN: tl.constexpr = (
        N * B_STRIDES.value[-2]
        + K * B_STRIDES.value[-1]
        + BATCH_COUNT * math.fsum(B_STRIDES.value)
    )
    INDEX64: tl.constexpr = (
        BATCH_COUNT * N * K > 2147483647
        or A_STRIDES_SPAN > 2147483647
        or B_STRIDES_SPAN > 2147483647
        or BATCH_COUNT * ((N + PANEL - 1) // PANEL) * PANEL * PANEL > 2147483647
        or BATCH_COUNT * K * (((N + PANEL - 1) // PANEL) + 1) > 2147483647
    )
    if FLAT:
        panels = tl.cdiv(N, PANEL)
        if INDEX64:
            panels = panels.to(tl.int64)
        batch = (tl.program_id(0) // (K * panels)).to(tl.int64)
        col = (tl.program_id(0) // panels) % K
    else:
        batch = tl.program_id(0).to(tl.int64)
        col = tl.program_id(1)
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
    X += batch * N * K
    I += batch * tl.cdiv(N, PANEL) * PANEL * PANEL
    GOOD += batch * tl.cdiv(N, PANEL)
    r = tl.arange(0, PANEL)
    # Assign panels only after their CTA is running. The smallest unfinished
    # ticket therefore belongs to a resident CTA and has no unfinished earlier
    # dependency. Progress does not depend on launch order or the SM count.
    state = STATE + (batch * K + col) * (tl.cdiv(N, PANEL) + 1)
    block = tl.atomic_add(state + tl.cdiv(N, PANEL), 1, sem="relaxed")
    if INDEX64:
        block = block.to(tl.int64)
    start = block * PANEL
    row = start + r
    ar = N - 1 - row if UPPER else row
    b = tl.load(
        B + ar * B_STRIDES.value[-2] + col * B_STRIDES.value[-1], row < N, other=0
    )
    for prior in range(block):
        pr = prior * PANEL + r
        ap = N - 1 - pr if UPPER else pr
        a = tl.load(
            A + ar[:, None] * A_STRIDES.value[-2] + ap[None, :] * A_STRIDES.value[-1],
            (row[:, None] < N) & (pr[None, :] < N),
            other=0,
        )
        while tl.atomic_add(state + prior, 0, sem="acquire") == 0:
            pass
        previous = tl.load(X + col * N + ap, pr < N, other=0)
        b = b - tl.sum(a * previous[None, :], 1)
    inv = tl.load(
        I + (start // PANEL) * PANEL * PANEL + r[:, None] * PANEL + r[None, :]
    )
    regular = tl.load(GOOD + start // PANEL)
    x = tl.full((PANEL,), 0.0, B.dtype.element_ty)
    if regular:
        x = tl.sum(tl.where(r[:, None] >= r[None, :], inv * b[None, :], 0.0), 1)
        # A finite inverse can still overflow against the panel RHS before
        # cancellation. Keep b intact for full-precision direct substitution.
        regular = tl.sum(((tl.abs(x) == float("inf")) | (x != x)).to(tl.int32), 0) == 0
    if not regular:
        aa = tl.load(
            A + ar[:, None] * A_STRIDES.value[-2] + ar[None, :] * A_STRIDES.value[-1],
            (row[:, None] < N) & (row[None, :] < N),
            other=0,
        )
        x = b
        for j in tl.static_range(PANEL):
            value = tl.sum(tl.where(r == j, x, 0.0), 0)
            if not UNIT:
                diag = tl.load(
                    A
                    + (N - 1 - start - j if UPPER else start + j)
                    * (A_STRIDES.value[-2] + A_STRIDES.value[-1]),
                    start + j < N,
                    other=1.0,
                )
                value = value / diag
            column = tl.sum(tl.where(r[None, :] == j, aa, 0.0), 1)
            x = tl.where(r > j, x - column * value, x)
            x = tl.where(r == j, value, x)
    tl.store(X + col * N + ar, x, row < N)
    tl.debug_barrier()
    # The CTA barrier above joins every output store before publishing readiness.
    # An acquiring consumer may then load the complete solved panel.
    tl.atomic_xchg(state + block, 1, sem="release")


def _solve_panel(A, B, *, upper, unitriangular, coefficient, original, out=None):
    n, k = B.shape[-2:]
    batch = tuple(B.shape[:-2])
    batches = math.prod(batch)
    result = (
        torch.empty((*batch, k, n), dtype=B.dtype, device=B.device).mT
        if out is None
        else out
    )
    if n <= 32:
        width = k + triton.cdiv(n, 32) ** 2
        flat = width > 65535
        grid = (batches * width,) if flat else (batches, width)
        _metax_vector_kernel[grid](
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
            True,
            flat,
            num_warps=1,
        )
    else:
        panel = 16
        panels = triton.cdiv(n, panel)
        inverse = torch.empty(
            (batches, panels, panel, panel), dtype=B.dtype, device=B.device
        )
        good = torch.empty((batches, panels), dtype=torch.bool, device=B.device)
        state = torch.empty(
            (batches, k, panels + 1), dtype=torch.int32, device=B.device
        )
        width = panels + triton.cdiv(n, 32) ** 2 + triton.cdiv(k * (panels + 1), 256)
        flat = width > 65535
        grid = (batches * width,) if flat else (batches, width)
        _metax_prepare_inverse[grid](
            A,
            inverse,
            good,
            state,
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
        flat = k > 65535 or panels > 65535
        grid = (batches * k * panels,) if flat else (batches, k, panels)
        _metax_inverse_solve[grid](
            A,
            B,
            result,
            inverse,
            good,
            state,
            n,
            k,
            batch,
            tuple(A.stride()),
            tuple(B.stride()),
            upper,
            unitriangular,
            panel,
            flat,
            num_warps=1,
        )
    return result


def triangular_solve(B, A, upper=True, transpose=False, unitriangular=False):
    logger.debug("GEMS_METAX TRIANGULAR_SOLVE")
    return _triangular_solve(
        B,
        A,
        upper,
        transpose,
        unitriangular,
        _solve_panel,
        solver_out=True,
        solver_coeff=True,
    )


def triangular_solve_out(
    B, A, upper=True, transpose=False, unitriangular=False, *, X, M
):
    logger.debug("GEMS_METAX TRIANGULAR_SOLVE_OUT")
    return _triangular_solve(
        B,
        A,
        upper,
        transpose,
        unitriangular,
        _solve_panel,
        X,
        M,
        solver_out=True,
        solver_coeff=True,
    )
