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
import warnings

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)
_warned = False


@libentry()
@triton.jit
def _copy_matrix_kernel(
    SRC,
    DST,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    BATCH: tl.constexpr,
    SRC_STRIDES: tl.constexpr,
    DST_STRIDES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile = tl.program_id(0).to(tl.int64)
    col_tiles = tl.cdiv(COLS, BLOCK)
    row_tiles = tl.cdiv(ROWS, BLOCK)
    col_tile = tile % col_tiles
    row_tile = (tile // col_tiles) % row_tiles
    batch = tile // (col_tiles * row_tiles)
    src_offset = tl.full((), 0, tl.int64)
    dst_offset = tl.full((), 0, tl.int64)
    for dim in tl.static_range(len(BATCH) - 1, -1, -1):
        coordinate = batch % BATCH.value[dim]
        batch //= BATCH.value[dim]
        src_offset += coordinate * SRC_STRIDES.value[dim]
        dst_offset += coordinate * DST_STRIDES.value[dim]
    rows = row_tile * BLOCK + tl.arange(0, BLOCK)
    cols = col_tile * BLOCK + tl.arange(0, BLOCK)
    mask = (rows[:, None] < ROWS) & (cols[None, :] < COLS)
    value = tl.load(
        SRC
        + src_offset
        + rows[:, None] * SRC_STRIDES.value[-2]
        + cols[None, :] * SRC_STRIDES.value[-1],
        mask=mask,
        other=0,
    )
    tl.store(
        DST
        + dst_offset
        + rows[:, None] * DST_STRIDES.value[-2]
        + cols[None, :] * DST_STRIDES.value[-1],
        value,
        mask=mask,
    )


def _copy_matrix(dst, src):
    # Inputs have already been broadcast and checked by the legacy wrapper.
    # A dedicated copy avoids repeating generic dtype/broadcast dispatch for
    # each of the two outputs, and never redispatches to an ATen copy kernel.
    rows, cols = src.shape[-2:]
    block = 32
    grid = (
        math.prod(src.shape[:-2]) * triton.cdiv(rows, block) * triton.cdiv(cols, block),
    )
    _copy_matrix_kernel[grid](
        src,
        dst,
        rows,
        cols,
        tuple(src.shape[:-2]),
        tuple(src.stride()),
        tuple(dst.stride()),
        block,
    )


def _triangular_solve(
    B,
    A,
    upper,
    transpose,
    unitriangular,
    solver,
    X=None,
    M=None,
    *,
    solver_out=False,
    solver_coeff=False,
    copy_fn=_copy_matrix,
):
    logger.debug("GEMS TRIANGULAR_SOLVE_IMPL")
    global _warned
    if B.ndim < 2 or A.ndim < 2:
        raise RuntimeError("triangular_solve: A and B must have at least 2 dimensions")
    if A.device != B.device:
        raise RuntimeError("triangular_solve: A and B must be on the same device")
    if A.dtype != B.dtype:
        raise RuntimeError("triangular_solve: A and B must have the same dtype")
    if A.layout != torch.strided or B.layout != torch.strided:
        raise RuntimeError("triangular_solve: only dense strided tensors are supported")
    if A.shape[-2] != A.shape[-1]:
        raise RuntimeError("triangular_solve: A must be a square matrix")
    if B.shape[-2] != A.shape[-1]:
        raise RuntimeError("triangular_solve: incompatible matrix shapes for A X = B")
    a_batch, b_batch = A.shape[:-2], B.shape[:-2]
    batch = a_batch if a_batch == b_batch else torch.broadcast_shapes(a_batch, b_batch)
    if B.dtype not in (torch.float32, torch.float64) or (
        B.dtype == torch.float64 and not runtime.device.support_fp64
    ):
        raise RuntimeError(f"triangular_solve: not implemented for {B.dtype}")
    if X is not None:
        for output, matrix in ((X, B), (M, A)):
            if output.device != B.device or output.dtype != B.dtype:
                raise RuntimeError(
                    "triangular_solve: outputs must have the input dtype and device"
                )
            if output.layout != torch.strided:
                raise RuntimeError("triangular_solve: outputs must be strided")
            # Expanded outputs cannot represent independent matrix elements.
            # A mismatched shape is resized below, which also resets its strides.
            if 0 in output.stride():
                expected = (*batch, *matrix.shape[-2:])
                if (
                    output.shape == expected
                    and output.numel()
                    and any(
                        size > 1 and stride == 0
                        for size, stride in zip(output.shape, output.stride())
                    )
                ):
                    raise RuntimeError(
                        "triangular_solve: more than one element of the written-to "
                        "tensor refers to a single memory location"
                    )
        if torch.is_grad_enabled() and any(t.requires_grad for t in (B, A, X, M)):
            raise RuntimeError(
                "triangular_solve: functions with out= arguments don't support autograd"
            )
    if not _warned:
        warnings.warn(
            "torch.triangular_solve is deprecated in favor of "
            "torch.linalg.solve_triangular and will be removed in a future release. "
            "torch.linalg.solve_triangular has its arguments reversed and does not "
            "return a copy of the coefficient matrix.",
            UserWarning,
            stacklevel=3,
        )
        _warned = True

    with runtime.torch_device_fn.device(B.device):
        n, k = B.shape[-2:]
        if A.shape[:-2] != batch:
            A = A.expand(*batch, n, n)
        if B.shape[:-2] != batch:
            B = B.expand(*batch, n, k)
        # Keep the original orientation and the entire matrix, including the unused
        # triangle and diagonal. Both functional results use the legacy BLAS layout.
        # A correctly sized coefficient output can be filled directly, provided
        # that doing so cannot overwrite either input or the solution output.
        direct_coefficient = (
            M is not None
            and M.shape == A.shape
            and not any(torch._C._overlaps(M, tensor) for tensor in (A, B, X))
        )
        coefficient = (
            M
            if direct_coefficient
            else torch.empty((*batch, n, n), dtype=A.dtype, device=A.device).mT
        )
        if coefficient.numel() and not (solver_coeff and B.numel()):
            copy_fn(coefficient, A)
        if B.numel():
            strides = [n, 1]
            stride = n * k
            for size in reversed(batch):
                strides.append(stride)
                stride *= size
            strides = tuple(reversed(strides))
            options = {}
            if solver_coeff:
                options.update(coefficient=coefficient, original=A)
            if solver_out:
                direct_solution = (
                    X is not None
                    and X.shape == B.shape
                    and X.stride() == strides
                    and not any(torch._C._overlaps(X, tensor) for tensor in (A, B, M))
                )
                options["out"] = X if direct_solution else None
            solution = solver(
                A.mT if transpose else A,
                B,
                upper=not upper if transpose else upper,
                unitriangular=unitriangular,
                **options,
            )
            # Internal solvers return fresh storage. Preserve a BLAS-layout result
            # directly instead of launching a redundant copy into another buffer.
            if solution.stride() != strides:
                result = torch.empty((*batch, k, n), dtype=B.dtype, device=B.device).mT
                copy_fn(result, solution)
                solution = result
        else:
            solution = torch.empty((*batch, k, n), dtype=B.dtype, device=B.device).mT
        if X is None:
            return solution, coefficient
        # Compute both temporaries before writing: an output may alias an input.
        for output, result in ((X, solution), (M, coefficient)):
            if output.shape != result.shape:
                if output.numel():
                    warnings.warn(
                        "An output with one or more elements was resized since it had "
                        f"shape {list(output.shape)}, which does not match the required "
                        f"output shape {list(result.shape)}. This behavior is deprecated.",
                        UserWarning,
                        stacklevel=3,
                    )
                output.resize_(result.mT.shape)
                output.transpose_(-2, -1)
        # ATen's direct BLAS path also normalizes empty output strides. Existing
        # non-column-major outputs use temporaries and retain their supplied layout.
        if X.mT.is_contiguous() and M.mT.is_contiguous():
            for output, result in ((X, solution), (M, coefficient)):
                if not output.numel():
                    output.resize_(result.mT.shape)
                    output.transpose_(-2, -1)
        for output, result in ((X, solution), (M, coefficient)):
            if output is not result and result.numel():
                copy_fn(output, result)
        return X, M


@triton.jit
def _panel_solve_kernel(
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
    PANEL: tl.constexpr,
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
        p = tl.arange(0, PANEL)
        if INDEX64:
            p = p.to(tl.int64)
        # Remap both axes so all triangular substitutions run forward.
        rr = N - 1 - rows if UPPER else rows
        for start in range(0, N, PANEL):
            pr = start + p
            ar = N - 1 - pr if UPPER else pr
            aa = tl.load(
                A
                + ar[:, None] * A_STRIDES.value[-2]
                + ar[None, :] * A_STRIDES.value[-1],
                (pr[:, None] < N) & (pr[None, :] < N),
                other=0,
            )
            xp = tl.load(
                B + ar * B_STRIDES.value[-2] + col * B_STRIDES.value[-1],
                pr < N,
                other=0,
            )
            if start > 0:
                previous = tl.load(
                    X + batch * N * K + col * N + rr, rows < start, other=0
                )
                left = tl.load(
                    A
                    + ar[:, None] * A_STRIDES.value[-2]
                    + rr[None, :] * A_STRIDES.value[-1],
                    (pr[:, None] < N) & (rows[None, :] < start),
                    other=0,
                )
                xp = xp - tl.sum(left * previous[None, :], 1)
            if not UNIT:
                diag = tl.load(
                    A + ar * (A_STRIDES.value[-2] + A_STRIDES.value[-1]),
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
            for j in tl.static_range(PANEL):
                v = tl.gather(xp, tl.full((1,), j, tl.int32), 0).reshape(())
                if not UNIT:
                    if regular:
                        iv = tl.gather(inverses, tl.full((1,), j, tl.int32), 0).reshape(
                            ()
                        )
                        v = v * iv
                    else:
                        d = tl.sum(
                            tl.where((p[:, None] == j) & (p[None, :] == j), aa, 0), 0
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
def _prepare_inverse_kernel(
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
        # The strictly triangular normalized panel is nilpotent. Repeated
        # squaring evaluates its finite inverse polynomial without truncation.
        q = tl.where(r[:, None] > r[None, :], -a * inv[:, None], 0.0)
        v = q + tl.where(r[:, None] == r[None, :], 1.0, 0.0)
        for _ in tl.static_range(1, 5 if PANEL == 32 else 4):
            q = tl.dot(q, q, allow_tf32=False)
            v = v + tl.dot(q, v, allow_tf32=False)
        v = v * inv[None, :]
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
def _inverse_solve_kernel(
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
    if tl.load(GOOD + start // PANEL):
        x = tl.sum(tl.where(r[:, None] >= r[None, :], inv * b[None, :], 0.0), 1)
    else:
        aa = tl.load(
            A + ar[:, None] * A_STRIDES.value[-2] + ar[None, :] * A_STRIDES.value[-1],
            (row[:, None] < N) & (row[None, :] < N),
            other=0,
        )
        x = b
        for j in tl.static_range(PANEL):
            value = tl.gather(x, tl.full((1,), j, tl.int32), 0).reshape(())
            if not UNIT:
                diag = tl.load(
                    A
                    + (N - 1 - start - j if UPPER else start + j)
                    * (A_STRIDES.value[-2] + A_STRIDES.value[-1]),
                    start + j < N,
                    other=1.0,
                )
                value = value / diag
            column = tl.gather(aa, tl.full((PANEL, 1), j, tl.int32), 1).reshape(
                (PANEL,)
            )
            x = tl.where(r > j, x - column * value, x)
            x = tl.where(r == j, value, x)
    tl.store(X + col * N + ar, x, row < N)
    tl.debug_barrier()
    # The CTA barrier above joins every output store before publishing readiness.
    # An acquiring consumer may then load the complete solved panel.
    tl.atomic_xchg(state + block, 1, sem="release")


_panel_launches = {}


def _launch_panel(grid, args):
    kernel = _panel_solve_kernel
    knobs = getattr(triton, "knobs", None)
    # Older Triton installations and other vendors retain ordinary JIT dispatch.
    if (
        runtime.device.vendor_name != "nvidia"
        or knobs is None
        or not hasattr(kernel, "device_caches")
        or not hasattr(knobs, "nvidia")
        or kernel.pre_run_hooks
    ):
        kernel[grid](*args, num_warps=4)
        return
    device = args[0].device.index
    abi = kernel.device_caches[device]
    if len(abi) != 5 or not callable(abi[4]) or abi[2].backend != "cuda":
        kernel[grid](*args, num_warps=4)
        return
    # NVIDIA specializes ordinary tensor arguments by dtype and 16-byte
    # alignment. Everything after the five pointers is an explicit constexpr.
    # Cold entries additionally verify the compiler's actual signature/attrs.
    aligned = tuple(tensor.data_ptr() % 16 == 0 for tensor in args[:5])
    key = (
        device,
        abi[2],
        args[0].dtype,
        aligned,
        args[5:],
        grid,
        kernel.cache_key,
        4,
        kernel.debug or knobs.runtime.debug,
        knobs.compilation.instrumentation_mode,
        knobs.runtime.override_arch,
        knobs.language.default_fp_fusion,
        knobs.nvidia.libdevice_path,
        knobs.nvidia.ptxas_options,
    )
    runner = _panel_launches.get(key)
    if runner is not None:
        for (name, _), (value, globals_dict) in kernel.used_global_vals.items():
            if name not in globals_dict or globals_dict[name] != value:
                raise RuntimeError(f"Kernel global {name} changed after compilation")
        # CompiledKernel's launcher resolves the current stream and preserves
        # launch hooks. Neither inputs nor stream handles are cached here.
        runner(*args)
        return
    compiled = kernel[grid](*args, num_warps=4)
    if compiled is None or not hasattr(compiled, "src"):
        return
    source = compiled.src
    expected_dtype = "*fp64" if args[0].dtype == torch.float64 else "*fp32"
    if (
        not hasattr(source, "attrs")
        or not hasattr(source, "signature")
        or not hasattr(source, "constants")
        or any(
            source.signature.get(name) != expected_dtype
            for name in kernel.arg_names[:5]
        )
        or source.constants != {(i,): value for i, value in enumerate(args) if i >= 5}
        or any(path not in {(i,) for i in range(5)} for path in source.attrs)
        or any(
            source.attrs.get((i,), []) != ([["tt.divisibility", 16]] if flag else [])
            for i, flag in enumerate(aligned)
        )
    ):
        return
    if len(_panel_launches) >= 256:
        _panel_launches.clear()
    _panel_launches[key] = compiled[grid + (1,) * (3 - len(grid))]


def _solve_dense(A, B, *, upper, unitriangular, out=None, coefficient, original):
    n, k = B.shape[-2:]
    batch = tuple(B.shape[:-2])
    batches = math.prod(batch)
    result = (
        torch.empty((*batch, k, n), dtype=B.dtype, device=B.device).mT
        if out is None
        else out
    )
    if n <= 128:
        panel = min(16 if B.dtype == torch.float64 else 32, triton.next_power_of_2(n))
        width = k + triton.cdiv(n, 32) ** 2
        flat = width > 65535
        grid = (batches * width,) if flat else (batches, width)
        _launch_panel(
            grid,
            (
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
                panel,
                True,
                flat,
            ),
        )
    else:
        panel = 16 if B.dtype == torch.float64 else 32
        inverse = torch.empty(
            (batches, triton.cdiv(n, panel), panel, panel),
            dtype=B.dtype,
            device=B.device,
        )
        good = torch.empty(
            (batches, triton.cdiv(n, panel)), dtype=torch.int32, device=B.device
        )
        panels = triton.cdiv(n, panel)
        state = torch.empty(
            (batches, k, panels + 1), dtype=torch.int32, device=B.device
        )
        width = panels + triton.cdiv(n, 32) ** 2 + triton.cdiv(k * (panels + 1), 256)
        flat = width > 65535
        grid = (batches * width,) if flat else (batches, width)
        _prepare_inverse_kernel[grid](
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
            FLAT=flat,
            num_warps=4,
        )
        clone = coefficient if A.stride() == original.stride() else coefficient.mT
        if clone.stride(-1) < A.stride(-1):
            A = clone
        flat = k > 65535 or panels > 65535
        grid = (batches * k * panels,) if flat else (batches, k, panels)
        _inverse_solve_kernel[grid](
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
            triton.next_power_of_2(n),
            panel,
            FLAT=flat,
            num_warps=4,
        )
    return result


def triangular_solve(B, A, upper=True, transpose=False, unitriangular=False):
    logger.debug("GEMS TRIANGULAR_SOLVE")
    return _triangular_solve(
        B,
        A,
        upper,
        transpose,
        unitriangular,
        _solve_dense,
        solver_out=True,
        solver_coeff=True,
    )


def triangular_solve_out(
    B, A, upper=True, transpose=False, unitriangular=False, *, X, M
):
    logger.debug("GEMS TRIANGULAR_SOLVE_OUT")
    return _triangular_solve(
        B,
        A,
        upper,
        transpose,
        unitriangular,
        _solve_dense,
        X,
        M,
        solver_out=True,
        solver_coeff=True,
    )
