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

import itertools
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

# gcu400 lowers ``tl.gather`` only for a single warp: with num_warps > 1 the
# triton-gpu -> triton-gcu pass marks the op illegal and the compiler fails
# ("failed to legalize operation 'tt.gather'", sometimes as a segfault).  Both
# the panel-solve and the inverse-solve kernels gather, so every launch of them
# uses one warp.
_PANEL_NUM_WARPS = 1

# gcu400 hardware grid limits (CUDA allows more than this on both axes).
_MAX_GRID_X = 65535

# Orders above 128 normally take the inverse-panel path.  On gcu400 that path
# hangs: _inverse_solve_kernel spins on `tl.atomic_add(state + prior, 0,
# sem="acquire")` in a cross-CTA wait that never observes the prior panel's
# release, and the kernel then holds the device until the process is killed
# (measured at n=257: no progress for 15+ minutes, health still OK).  Every
# order goes through the sequential panel path instead; its previous-panel
# contribution is walked in PANEL-wide chunks so the tiles stay register-sized.
USE_INVERSE_PATH = 0


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
    #
    # gcu400's masked 2-D tile store only lands correctly when the destination
    # is innermost-contiguous; with a column-major destination (the BLAS output
    # layout used here, stride[-1] == rows) lanes are dropped (measured: n=4 ->
    # 12/16 wrong, n=33 -> 31 wrong).  Flip both operands when that makes the
    # store contiguous -- the strided load is exact.
    if dst.stride(-1) != 1 and dst.stride(-2) == 1:
        dst = dst.transpose(-2, -1)
        src = src.transpose(-2, -1)
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
    logger.debug("GEMS_ENFLAME TRIANGULAR_SOLVE_IMPL")
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
        width = K
        if COPY:
            width = width + tl.cdiv(N, 32) * tl.cdiv(N, 32)
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
                # Walk the already-solved rows in PANEL-wide chunks.  A single
                # (PANEL x ROWS) tile only fits while N is small; chunking keeps
                # the tiles PANEL x PANEL so the same kernel can serve the large
                # orders the inverse path is no longer used for (see
                # USE_INVERSE_PATH in _solve_dense).
                for base in range(0, start, PANEL):
                    q = base + p
                    qmask = q < start
                    prev_idx = N - 1 - q if UPPER else q
                    previous = tl.load(
                        X + batch * N * K + col * N + prev_idx, qmask, other=0
                    )
                    left = tl.load(
                        A
                        + ar[:, None] * A_STRIDES.value[-2]
                        + prev_idx[None, :] * A_STRIDES.value[-1],
                        (pr[:, None] < N) & qmask[None, :],
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
        kernel[grid](*args, num_warps=_PANEL_NUM_WARPS)
        return
    device = args[0].device.index
    abi = kernel.device_caches[device]
    if len(abi) != 5 or not callable(abi[4]) or abi[2].backend != "cuda":
        kernel[grid](*args, num_warps=_PANEL_NUM_WARPS)
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
        _PANEL_NUM_WARPS,
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
    compiled = kernel[grid](*args, num_warps=_PANEL_NUM_WARPS)
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


@triton.jit
def _coefficient_copy_kernel(
    SRC,
    DST,
    S_BASE,
    D_BASE,
    N,
    S_S0,
    S_S1,
    D_S0,
    D_S1,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Copy one (N, N) matrix (per batch) between arbitrary 2-D strides.

    Flat 1-D tiles with clamped addresses: gcu400's masked 32x32 tile copies
    only stored their first row/column whenever the tile straddled the matrix
    edge, so the coefficient came out mostly zero for N not a multiple of 32.
    """
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < TOTAL
    safe = tl.where(m, off, 0)
    i = safe % N
    j = safe // N
    value = tl.load(SRC + S_BASE + i * S_S0 + j * S_S1, mask=m, other=0.0)
    tl.store(DST + D_BASE + i * D_S0 + j * D_S1, value, mask=m)


def _copy_coefficient(dst, src):
    """Copy ``src`` into the coefficient output ``dst`` (both (..., N, N))."""
    if dst is None or src is None or not dst.numel() or not src.numel():
        return
    n = int(src.shape[-1])
    total = n * n
    block = 1024
    chunks = triton.cdiv(total, block)
    s_stride, d_stride = src.stride(), dst.stride()
    batch = tuple(src.shape[:-2])
    with runtime.torch_device_fn.device(dst.device):
        for coords in itertools.product(*(range(size) for size in batch)):
            s_base = 0
            d_base = 0
            for dim, coordinate in enumerate(coords):
                s_base += coordinate * s_stride[dim]
                d_base += coordinate * d_stride[dim]
            _coefficient_copy_kernel[(chunks,)](
                src,
                dst,
                s_base,
                d_base,
                n,
                s_stride[-2],
                s_stride[-1],
                d_stride[-2],
                d_stride[-1],
                TOTAL=total,
                BLOCK=block,
                num_warps=4,
            )


def _copy_coefficient_batched(dst, src):
    """Copy the full coefficient matrix for every batch element in one launch.

    The 2-D tile copy lands when the destination is innermost-contiguous or
    can be flipped into that form (the BLAS column-major outputs).  Exotic
    destinations (e.g. sliced outputs) keep the address-clamped flat kernel.
    """
    if dst is None or src is None or not dst.numel() or not src.numel():
        return
    inner, outer = dst.stride(-1), dst.stride(-2)
    if inner == 1 or (outer == 1 and inner != 1):
        _copy_matrix(dst, src)
    else:
        _copy_coefficient(dst, src)


# ══════════════════════════════════════════════════════════════════════════
# fp32 fast path (2026-11-17): column-scaled Neumann kernels, num_warps=1.
#
# Same measured design as the gcu400 linalg_solve_triangular fast path
# (whole-block Neumann solves as tl.dot chains; serial per-row substitution
# is reduction/barrier-bound here, and tl.gather forces num_warps=1 anyway).
# Deviations required by this op's contract:
#   * A = (I + M) D with M[i, j] = A[i, j]/A[j, j] (COLUMN scaling), so
#     X = D^{-1}(I + M)^{-1} B: the diagonal reciprocal multiplies the
#     result, after the series' cancellation.  The row-scaled form
#     (D^{-1} first) overflows to NaN on finite-inverses whose scaled RHS
#     exceeds the exponent range before cancelling (n=33 test case).
#   * X is written directly in the legacy BLAS column-major layout
#     (X[r, c] at c*N + r), out-of-place: read B, write X.
#   * UPPER uses natural ascending coordinates everywhere: only the strict
#     mask orientation flips and the block ORDER reverses.  Descending
#     address vectors degrade the load lowering on this backend, and the
#     host-side flip mirror costs extra full-matrix copies.
#   * The (I+M)^{-1} factor chain is complete up to M^(BLOCK-1) for every
#     BLOCK, including the M^64 factor of the BLOCK=128 inverse builder.
# ══════════════════════════════════════════════════════════════════════════


@libentry()
@triton.jit
def _tri_solve_1blk(
    A,
    B,
    X,
    N: tl.constexpr,
    K: tl.constexpr,
    BATCH: tl.constexpr,
    A_STRIDES: tl.constexpr,
    B_STRIDES: tl.constexpr,
    XB: tl.constexpr,
    BN: tl.constexpr,
    KS: tl.constexpr,
    UPPER: tl.constexpr,
    UNIT: tl.constexpr,
):
    """Whole-matrix Neumann solve for N <= BN <= 64.  Grid: (batch*k-slices,)."""
    BATCH_COUNT: tl.constexpr = math.prod(BATCH.value)
    A_SPAN: tl.constexpr = N * (A_STRIDES.value[-2] + A_STRIDES.value[-1]) + (
        BATCH_COUNT * math.fsum(A_STRIDES.value)
    )
    B_SPAN: tl.constexpr = (
        N * B_STRIDES.value[-2]
        + K * B_STRIDES.value[-1]
        + (BATCH_COUNT * math.fsum(B_STRIDES.value))
    )
    INDEX64: tl.constexpr = (
        A_SPAN > 2147483647 or B_SPAN > 2147483647 or BATCH_COUNT * N * K > 2147483647
    )
    NUM_K: tl.constexpr = (K + KS - 1) // KS
    EVEN_K: tl.constexpr = K % KS == 0
    pid = tl.program_id(0)
    # Fold the k-slice index into constants whenever the compiler can prove
    # the column mask: with one slice kcols is a constant vector, and with
    # full slices the mask is all-true.  Runtime column masks keep every
    # load/store of the walk on the expensive predicated path (measured:
    # same kernel 135 us runtime-masked vs 40 us constant-masked at n=128).
    if NUM_K == 1:
        kcols = tl.arange(0, KS)
        cmask = kcols < K
    elif EVEN_K:
        kcols = (pid % NUM_K) * KS + tl.arange(0, KS)
        cmask = tl.full((KS,), 1, tl.int1)
    else:
        kcols = (pid % NUM_K) * KS + tl.arange(0, KS)
        cmask = kcols < K
    flat_batch = pid // NUM_K
    if INDEX64:
        flat_batch = flat_batch.to(tl.int64)
        kcols = kcols.to(tl.int64)
    a_off = tl.full((), 0, tl.int64)
    b_off = tl.full((), 0, tl.int64)
    remainder = flat_batch
    for dim in tl.static_range(len(BATCH) - 1, -1, -1):
        coordinate = remainder % BATCH.value[dim]
        remainder //= BATCH.value[dim]
        a_off += coordinate * A_STRIDES.value[dim]
        b_off += coordinate * B_STRIDES.value[dim]
    A += a_off
    B += b_off
    X += flat_batch * XB
    rows = tl.arange(0, BN)
    if INDEX64:
        rows = rows.to(tl.int64)
    r2 = rows[:, None]
    c2 = rows[None, :]
    in_n = (rows[:, None] < N) & (rows[None, :] < N)
    # gcu400's masked loads do not always honor `other` for lanes whose
    # address falls outside the tensor: the DMA returns neighbor memory, and
    # 0 * NaN inside the dots poisons valid outputs (measured: X[64, 2] read
    # the NaN-prefilled output of the next allocation).  Clamp every masked
    # lane's address to element 0, as _coefficient_copy_kernel does.
    a_blk = tl.load(
        A + tl.where(in_n, r2 * A_STRIDES.value[-2] + c2 * A_STRIDES.value[-1], 0),
        mask=in_n,
        other=0.0,
    )
    if UNIT:
        dinv = tl.full((BN,), 1.0, a_blk.dtype)
    else:
        in_d = rows < N
        dinv = 1.0 / tl.load(
            A + tl.where(in_d, rows * (A_STRIDES.value[-2] + A_STRIDES.value[-1]), 0),
            mask=in_d,
            other=1.0,
        )
    if UPPER:
        m = tl.where(c2 > r2, a_blk, 0.0) * dinv[None, :]
    else:
        m = tl.where(c2 < r2, a_blk, 0.0) * dinv[None, :]
    rmask = (rows[:, None] < N) & cmask[None, :]
    v = tl.load(
        B
        + tl.where(
            rmask,
            r2 * B_STRIDES.value[-2] + kcols[None, :] * B_STRIDES.value[-1],
            0,
        ),
        mask=rmask,
        other=0.0,
    )
    m2 = tl.dot(m, m, allow_tf32=False)
    m4 = tl.dot(m2, m2, allow_tf32=False)
    m8 = tl.dot(m4, m4, allow_tf32=False)
    v = v - tl.dot(m, v, allow_tf32=False)
    v = v + tl.dot(m2, v, allow_tf32=False)
    v = v + tl.dot(m4, v, allow_tf32=False)
    v = v + tl.dot(m8, v, allow_tf32=False)
    if BN >= 32:
        m16 = tl.dot(m8, m8, allow_tf32=False)
        v = v + tl.dot(m16, v, allow_tf32=False)
    if BN >= 64:
        m32 = tl.dot(m16, m16, allow_tf32=False)
        v = v + tl.dot(m32, v, allow_tf32=False)
    # The solution buffer is column-major (X[r, c] at c*N + r): the tile's
    # innermost lane must walk the contiguous row axis, or gcu400's masked
    # 2-D store drops every lane but the first row (see _copy_matrix).
    tl.store(
        X + kcols[:, None] * N + rows[None, :],
        tl.trans(v * dinv[:, None]),
        mask=cmask[:, None] & (rows[None, :] < N),
    )


@libentry()
@triton.jit
def _tri_solve_walk(
    A,
    B,
    X,
    N: tl.constexpr,
    K: tl.constexpr,
    BATCH: tl.constexpr,
    A_STRIDES: tl.constexpr,
    B_STRIDES: tl.constexpr,
    XB: tl.constexpr,
    BLOCK: tl.constexpr,
    KS: tl.constexpr,
    BM: tl.constexpr,
    UPPER: tl.constexpr,
    UNIT: tl.constexpr,
):
    """Serial block walk with in-register Neumann solves, 64 < N <= 192.

    Grid: (batch*k-slices,).  Out-of-place: the first processed block reads
    B, later reads hit rows this CTA wrote to X (updates cover exactly the
    next block's rows, in either walk direction).  No X initialization.
    """
    BATCH_COUNT: tl.constexpr = math.prod(BATCH.value)
    A_SPAN: tl.constexpr = N * (A_STRIDES.value[-2] + A_STRIDES.value[-1]) + (
        BATCH_COUNT * math.fsum(A_STRIDES.value)
    )
    B_SPAN: tl.constexpr = (
        N * B_STRIDES.value[-2]
        + K * B_STRIDES.value[-1]
        + (BATCH_COUNT * math.fsum(B_STRIDES.value))
    )
    INDEX64: tl.constexpr = (
        A_SPAN > 2147483647 or B_SPAN > 2147483647 or BATCH_COUNT * N * K > 2147483647
    )
    NUM_K: tl.constexpr = (K + KS - 1) // KS
    NUM_BLOCKS: tl.constexpr = (N + BLOCK - 1) // BLOCK
    EVEN_K: tl.constexpr = K % KS == 0
    pid = tl.program_id(0)
    # Constant-fold the column mask whenever provable (see _tri_solve_1blk):
    # runtime column masks keep every load/store on the predicated path.
    if NUM_K == 1:
        kcols = tl.arange(0, KS)
        cmask = kcols < K
    elif EVEN_K:
        kcols = (pid % NUM_K) * KS + tl.arange(0, KS)
        cmask = tl.full((KS,), 1, tl.int1)
    else:
        kcols = (pid % NUM_K) * KS + tl.arange(0, KS)
        cmask = kcols < K
    flat_batch = pid // NUM_K
    if INDEX64:
        flat_batch = flat_batch.to(tl.int64)
        kcols = kcols.to(tl.int64)
    a_off = tl.full((), 0, tl.int64)
    b_off = tl.full((), 0, tl.int64)
    remainder = flat_batch
    for dim in tl.static_range(len(BATCH) - 1, -1, -1):
        coordinate = remainder % BATCH.value[dim]
        remainder //= BATCH.value[dim]
        a_off += coordinate * A_STRIDES.value[dim]
        b_off += coordinate * B_STRIDES.value[dim]
    A += a_off
    B += b_off
    X += flat_batch * XB
    cm2 = cmask[None, :]
    rows = tl.arange(0, BLOCK)
    rr = tl.arange(0, BM)
    if INDEX64:
        rows = rows.to(tl.int64)
        rr = rr.to(tl.int64)
    # Static unroll: on this backend the dot chains inside a dynamic block
    # loop lower ~2.5x slower than the same dots in straight-line code
    # (measured n=128: 100.7 -> 40.3 us).  NUM_BLOCKS is constexpr, so every
    # block-level scalar below folds at compile time.
    for block_idx in tl.static_range(NUM_BLOCKS):
        if not UPPER:
            bk = block_idx
        else:
            bk = NUM_BLOCKS - 1 - block_idx
        blk_start = bk * BLOCK
        # Trace-time bounds.  min() and if-statements lower to tensor ops
        # (tl.minimum / scf.if value merge) and deopt every static_range
        # below; the ternary keeps everything constexpr.
        blk_end = blk_start + BLOCK if blk_start + BLOCK < N else N
        blk_sz = blk_end - blk_start
        row1 = blk_start + rows
        r2 = row1[:, None]
        c2 = row1[None, :]
        in_blk = (rows[:, None] < blk_sz) & (rows[None, :] < blk_sz)
        # Clamp masked lanes' addresses in-bounds (see _tri_solve_1blk): a
        # ragged tail block addresses rows/cols past A and past X/B.
        a_blk = tl.load(
            A
            + tl.where(in_blk, r2 * A_STRIDES.value[-2] + c2 * A_STRIDES.value[-1], 0),
            mask=in_blk,
            other=0.0,
        )
        if UNIT:
            dinv = tl.full((BLOCK,), 1.0, a_blk.dtype)
        else:
            in_d = rows < blk_sz
            dinv = 1.0 / tl.load(
                A
                + tl.where(in_d, row1 * (A_STRIDES.value[-2] + A_STRIDES.value[-1]), 0),
                mask=in_d,
                other=1.0,
            )
        if UPPER:
            m = tl.where(c2 > r2, a_blk, 0.0) * dinv[None, :]
        else:
            m = tl.where(c2 < r2, a_blk, 0.0) * dinv[None, :]
        rmask = (rows[:, None] < blk_sz) & cm2
        if block_idx == 0:
            v = tl.load(
                B
                + tl.where(
                    rmask,
                    r2 * B_STRIDES.value[-2] + kcols[None, :] * B_STRIDES.value[-1],
                    0,
                ),
                mask=rmask,
                other=0.0,
            )
        else:
            v = tl.load(
                X + tl.where(rmask, r2 + kcols[None, :] * N, 0),
                mask=rmask,
                other=0.0,
            )
        m2 = tl.dot(m, m, allow_tf32=False)
        m4 = tl.dot(m2, m2, allow_tf32=False)
        m8 = tl.dot(m4, m4, allow_tf32=False)
        m16 = tl.dot(m8, m8, allow_tf32=False)
        v = v - tl.dot(m, v, allow_tf32=False)
        v = v + tl.dot(m2, v, allow_tf32=False)
        v = v + tl.dot(m4, v, allow_tf32=False)
        v = v + tl.dot(m8, v, allow_tf32=False)
        v = v + tl.dot(m16, v, allow_tf32=False)
        if BLOCK >= 64:
            m32 = tl.dot(m16, m16, allow_tf32=False)
            v = v + tl.dot(m32, v, allow_tf32=False)
        x = v * dinv[:, None]
        # Column-major X: store with the contiguous row axis innermost (see
        # _tri_solve_1blk).
        tl.store(
            X + kcols[:, None] * N + row1[None, :],
            tl.trans(x),
            mask=cmask[:, None] & (rows[None, :] < blk_sz),
        )
        if (bk > 0) if UPPER else (blk_end < N):
            if UPPER:
                m_rem = blk_start
                rem_s = 0
                bound = blk_start
            else:
                m_rem = N - blk_end
                rem_s = blk_end
                bound = N
            blk_nat = blk_start + rows
            x_panel = x
            for m_start in range(0, m_rem, BM):
                rm = rem_s + m_start + rr
                in_m = rm < bound
                in_a = in_m[:, None] & (rows[None, :] < blk_sz)
                a_sub = tl.load(
                    A
                    + tl.where(
                        in_a,
                        rm[:, None] * A_STRIDES.value[-2]
                        + blk_nat[None, :] * A_STRIDES.value[-1],
                        0,
                    ),
                    mask=in_a,
                    other=0.0,
                )
                acc = tl.dot(a_sub, x_panel, allow_tf32=False)
                mask_m = in_m[:, None] & cm2
                if block_idx == 0:
                    b_curr = tl.load(
                        B
                        + tl.where(
                            mask_m,
                            rm[:, None] * B_STRIDES.value[-2]
                            + kcols[None, :] * B_STRIDES.value[-1],
                            0,
                        ),
                        mask=mask_m,
                        other=0.0,
                    )
                else:
                    b_curr = tl.load(
                        X
                        + tl.where(
                            mask_m,
                            rm[:, None] + kcols[None, :] * N,
                            0,
                        ),
                        mask=mask_m,
                        other=0.0,
                    )
                tl.store(
                    X + kcols[:, None] * N + rm[None, :],
                    tl.trans(b_curr - acc),
                    mask=cmask[:, None] & in_m[None, :],
                )
            tl.debug_barrier()


@libentry()
@triton.jit
def _tri_ainv_build(
    A,
    AINV,
    N: tl.constexpr,
    BATCH: tl.constexpr,
    A_STRIDES: tl.constexpr,
    BLOCK: tl.constexpr,
    UPPER: tl.constexpr,
    UNIT: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    """K1: per-diagonal-block inverse, computed once (k-independent).

    Grid: (batch*blocks,).  Stores Ainv = D^{-1}(I + M)^{-1} tiles at
    AINV[(batch * NUM_BLOCKS + blk) * BLOCK * BLOCK].
    """
    BATCH_COUNT: tl.constexpr = math.prod(BATCH.value)
    A_SPAN: tl.constexpr = N * (A_STRIDES.value[-2] + A_STRIDES.value[-1]) + (
        BATCH_COUNT * math.fsum(A_STRIDES.value)
    )
    INDEX64: tl.constexpr = (
        A_SPAN > 2147483647
        or BATCH_COUNT * N * N > 2147483647
        or BATCH_COUNT * NUM_BLOCKS * BLOCK * BLOCK > 2147483647
    )
    pid = tl.program_id(0)
    flat_batch = pid // NUM_BLOCKS
    blk = pid % NUM_BLOCKS
    if INDEX64:
        flat_batch = flat_batch.to(tl.int64)
    a_off = tl.full((), 0, tl.int64)
    remainder = flat_batch
    for dim in tl.static_range(len(BATCH) - 1, -1, -1):
        coordinate = remainder % BATCH.value[dim]
        remainder //= BATCH.value[dim]
        a_off += coordinate * A_STRIDES.value[dim]
    A += a_off
    blk_start = blk * BLOCK
    blk_end = tl.minimum(blk_start + BLOCK, N)
    blk_sz = blk_end - blk_start
    rows = tl.arange(0, BLOCK)
    if INDEX64:
        rows = rows.to(tl.int64)
    row1 = blk_start + rows
    r2 = row1[:, None]
    c2 = row1[None, :]
    # The host hands this kernel a zero-padded A (padded to a BLOCK multiple),
    # so the whole square tile is in-bounds and the ragged tail block simply
    # reads padding zeros.  An unmasked load is the only addressing form that
    # stays exact at every tile size on this backend: masked loads and
    # clamped-offset loads miscompile on square tiles >= 128x128 (measured:
    # wrong elements / transposed triangle), while mask=/other= forms are only
    # conditionally stable depending on the surrounding expression graph.
    # Zero padding keeps every masked lane finite, so the dots below never
    # meet 0 * NaN poison from out-of-bounds lanes.
    a_blk = tl.load(
        A + row1[:, None] * A_STRIDES.value[-2] + row1[None, :] * A_STRIDES.value[-1]
    )
    if UNIT:
        dinv = tl.full((BLOCK,), 1.0, a_blk.dtype)
    else:
        in_d = rows < blk_sz
        dinv = 1.0 / tl.load(
            A + tl.where(in_d, row1 * (A_STRIDES.value[-2] + A_STRIDES.value[-1]), 0),
            mask=in_d,
            other=1.0,
        )
    if UPPER:
        m = tl.where(c2 > r2, a_blk, 0.0) * dinv[None, :]
    else:
        m = tl.where(c2 < r2, a_blk, 0.0) * dinv[None, :]
    p = tl.where(rows[:, None] == rows[None, :], 1.0, 0.0)
    m2 = tl.dot(m, m, allow_tf32=False)
    m4 = tl.dot(m2, m2, allow_tf32=False)
    m8 = tl.dot(m4, m4, allow_tf32=False)
    p = p - tl.dot(m, p, allow_tf32=False)
    p = p + tl.dot(m2, p, allow_tf32=False)
    p = p + tl.dot(m4, p, allow_tf32=False)
    p = p + tl.dot(m8, p, allow_tf32=False)
    if BLOCK >= 32:
        m16 = tl.dot(m8, m8, allow_tf32=False)
        p = p + tl.dot(m16, p, allow_tf32=False)
    if BLOCK >= 64:
        m32 = tl.dot(m16, m16, allow_tf32=False)
        p = p + tl.dot(m32, p, allow_tf32=False)
    if BLOCK >= 128:
        m64 = tl.dot(m32, m32, allow_tf32=False)
        p = p + tl.dot(m64, p, allow_tf32=False)
    ainv = p * dinv[:, None]
    base = (flat_batch * NUM_BLOCKS + blk) * (BLOCK * BLOCK)
    tl.store(AINV + base + rows[:, None] * BLOCK + rows[None, :], ainv)


@libentry()
@triton.jit
def _tri_solve_ainv(
    A,
    B,
    X,
    AINV,
    N: tl.constexpr,
    K: tl.constexpr,
    BATCH: tl.constexpr,
    A_STRIDES: tl.constexpr,
    B_STRIDES: tl.constexpr,
    XB: tl.constexpr,
    BLOCK: tl.constexpr,
    KS: tl.constexpr,
    BM: tl.constexpr,
    UPPER: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    """K2: block walk with precomputed inverses (one diag dot per block).

    Grid: (batch*k-slices,).  Reads B, writes X out-of-place (see
    _tri_solve_walk).  The update phase still needs the raw A columns.
    """
    BATCH_COUNT: tl.constexpr = math.prod(BATCH.value)
    A_SPAN: tl.constexpr = N * (A_STRIDES.value[-2] + A_STRIDES.value[-1]) + (
        BATCH_COUNT * math.fsum(A_STRIDES.value)
    )
    B_SPAN: tl.constexpr = (
        N * B_STRIDES.value[-2]
        + K * B_STRIDES.value[-1]
        + (BATCH_COUNT * math.fsum(B_STRIDES.value))
    )
    INDEX64: tl.constexpr = (
        A_SPAN > 2147483647
        or B_SPAN > 2147483647
        or BATCH_COUNT * N * K > 2147483647
        or BATCH_COUNT * NUM_BLOCKS * BLOCK * BLOCK > 2147483647
    )
    NUM_K: tl.constexpr = (K + KS - 1) // KS
    EVEN_K: tl.constexpr = K % KS == 0
    pid = tl.program_id(0)
    # Constant-fold the column mask whenever provable (see _tri_solve_1blk):
    # runtime column masks keep every load/store on the predicated path.
    if NUM_K == 1:
        kcols = tl.arange(0, KS)
        cmask = kcols < K
    elif EVEN_K:
        kcols = (pid % NUM_K) * KS + tl.arange(0, KS)
        cmask = tl.full((KS,), 1, tl.int1)
    else:
        kcols = (pid % NUM_K) * KS + tl.arange(0, KS)
        cmask = kcols < K
    flat_batch = pid // NUM_K
    if INDEX64:
        flat_batch = flat_batch.to(tl.int64)
        kcols = kcols.to(tl.int64)
    a_off = tl.full((), 0, tl.int64)
    b_off = tl.full((), 0, tl.int64)
    remainder = flat_batch
    for dim in tl.static_range(len(BATCH) - 1, -1, -1):
        coordinate = remainder % BATCH.value[dim]
        remainder //= BATCH.value[dim]
        a_off += coordinate * A_STRIDES.value[dim]
        b_off += coordinate * B_STRIDES.value[dim]
    A += a_off
    B += b_off
    X += flat_batch * XB
    AINV += flat_batch * NUM_BLOCKS * (BLOCK * BLOCK)
    cm2 = cmask[None, :]
    rows = tl.arange(0, BLOCK)
    rr = tl.arange(0, BM)
    if INDEX64:
        rows = rows.to(tl.int64)
        rr = rr.to(tl.int64)
    # Static unroll (see _tri_solve_walk): the per-block dot chain lowers
    # ~2.5x faster in straight-line code on this backend.
    for block_idx in tl.static_range(NUM_BLOCKS):
        if not UPPER:
            bk = block_idx
        else:
            bk = NUM_BLOCKS - 1 - block_idx
        blk_start = bk * BLOCK
        # Trace-time bounds.  min() and if-statements lower to tensor ops
        # (tl.minimum / scf.if value merge) and deopt every static_range
        # below; the ternary keeps everything constexpr.
        blk_end = blk_start + BLOCK if blk_start + BLOCK < N else N
        blk_sz = blk_end - blk_start
        row1 = blk_start + rows
        r2 = row1[:, None]
        ainv = tl.load(
            AINV + bk * (BLOCK * BLOCK) + rows[:, None] * BLOCK + rows[None, :]
        )
        rmask = (rows[:, None] < blk_sz) & cm2
        # Clamp masked lanes' addresses in-bounds (see _tri_solve_1blk).
        if block_idx == 0:
            v = tl.load(
                B
                + tl.where(
                    rmask,
                    r2 * B_STRIDES.value[-2] + kcols[None, :] * B_STRIDES.value[-1],
                    0,
                ),
                mask=rmask,
                other=0.0,
            )
        else:
            v = tl.load(
                X + tl.where(rmask, r2 + kcols[None, :] * N, 0),
                mask=rmask,
                other=0.0,
            )
        x = tl.dot(ainv, v, allow_tf32=False)
        # Column-major X: store with the contiguous row axis innermost (see
        # _tri_solve_1blk).
        tl.store(
            X + kcols[:, None] * N + row1[None, :],
            tl.trans(x),
            mask=cmask[:, None] & (rows[None, :] < blk_sz),
        )
        if (bk > 0) if UPPER else (blk_end < N):
            if UPPER:
                m_rem = blk_start
                rem_s = 0
                bound = blk_start
            else:
                m_rem = N - blk_end
                rem_s = blk_end
                bound = N
            blk_nat = blk_start + rows
            x_panel = x
            for m_start in range(0, m_rem, BM):
                rm = rem_s + m_start + rr
                in_m = rm < bound
                in_a = in_m[:, None] & (rows[None, :] < blk_sz)
                a_sub = tl.load(
                    A
                    + tl.where(
                        in_a,
                        rm[:, None] * A_STRIDES.value[-2]
                        + blk_nat[None, :] * A_STRIDES.value[-1],
                        0,
                    ),
                    mask=in_a,
                    other=0.0,
                )
                acc = tl.dot(a_sub, x_panel, allow_tf32=False)
                mask_m = in_m[:, None] & cm2
                if block_idx == 0:
                    b_curr = tl.load(
                        B
                        + tl.where(
                            mask_m,
                            rm[:, None] * B_STRIDES.value[-2]
                            + kcols[None, :] * B_STRIDES.value[-1],
                            0,
                        ),
                        mask=mask_m,
                        other=0.0,
                    )
                else:
                    b_curr = tl.load(
                        X
                        + tl.where(
                            mask_m,
                            rm[:, None] + kcols[None, :] * N,
                            0,
                        ),
                        mask=mask_m,
                        other=0.0,
                    )
                tl.store(
                    X + kcols[:, None] * N + rm[None, :],
                    tl.trans(b_curr - acc),
                    mask=cmask[:, None] & in_m[None, :],
                )
            tl.debug_barrier()


def _solve_columns(
    A, B, result, n, width, batch, batches, upper, unitriangular, ainv, xb
):
    # One flat grid per launch: batch rides the high digits, k-slices the low
    # (gcu400 caps grid.y at 255, so the panel kernels used the same shape).
    # Measured on this stack (Triton GCU 4.1.6): the inverse-walk kernel wants
    # narrow k-slices (KS=8) and small update tiles (BM=64); wider update
    # tiles (BM=256, the previous family default) spill at num_warps=1 and
    # run 3x slower, and num_warps>1 regresses 4-9x.
    ks = 8 if ainv is not None else 16
    num_k = (width + ks - 1) // ks
    if batches * num_k > _MAX_GRID_X:
        # Solve wide systems in column chunks; each chunk's flat grid stays
        # inside grid.x.  Slices keep the stride tuples, so kernels launch
        # unchanged on the views (xb stays the full-width batch stride).
        chunk = max(ks, (_MAX_GRID_X // max(batches, 1)) * ks)
        for start in range(0, width, chunk):
            end = min(start + chunk, width)
            _solve_columns(
                A,
                B[..., start:end],
                result[..., start:end],
                n,
                end - start,
                batch,
                batches,
                upper,
                unitriangular,
                ainv,
                xb,
            )
        return
    grid = (batches * num_k,)
    a_strides = tuple(A.stride())
    b_strides = tuple(B.stride())
    if n <= 64:
        bn = 16 if n <= 16 else (32 if n <= 32 else 64)
        _tri_solve_1blk[grid](
            A,
            B,
            result,
            n,
            width,
            batch,
            a_strides,
            b_strides,
            xb,
            bn,
            ks,
            upper,
            unitriangular,
            num_warps=1,
        )
    elif ainv is None:
        _tri_solve_walk[grid](
            A,
            B,
            result,
            n,
            width,
            batch,
            a_strides,
            b_strides,
            xb,
            64,
            ks,
            128,
            upper,
            unitriangular,
            num_warps=1,
        )
    else:
        block = 128
        _tri_solve_ainv[grid](
            A,
            B,
            result,
            ainv,
            n,
            width,
            batch,
            a_strides,
            b_strides,
            xb,
            block,
            ks,
            64,
            upper,
            (n + block - 1) // block,
            num_warps=1,
        )


def _solve_dense_legacy(A, B, *, upper, unitriangular, out=None, coefficient, original):
    if upper:
        # gcu400 masked loads with a reversed (negative-stride) index return
        # garbage in every lane but the first, so the kernels' mirror-index
        # upper path silently produces wrong values (measured: only the last
        # solution element is right).  Mirror both axes instead: A' = flip(A)
        # is lower triangular, so solve A' x' = flip(B) with the (verified)
        # lower path and flip the solution back.
        mirrored = _solve_dense(
            A.flip(-2).flip(-1),
            B.flip(-2),
            upper=False,
            unitriangular=unitriangular,
            out=None,
            coefficient=coefficient,
            original=original,
        )
        return mirrored.flip(-2)
    n, k = B.shape[-2:]
    batch = tuple(B.shape[:-2])
    batches = math.prod(batch)
    if batches * k > _MAX_GRID_X:
        # The panel solver runs one program per RHS column and gcu400 caps
        # grid.x at 65535 programs, so wide systems are solved in column
        # chunks (each chunk's flat grid stays inside the limit) and the
        # pieces are assembled into one BLAS-layout result.
        result = (
            torch.empty((*batch, k, n), dtype=B.dtype, device=B.device).mT
            if out is None
            else out
        )
        step = max(1, _MAX_GRID_X // max(batches, 1))
        for start in range(0, k, step):
            end = min(start + step, k)
            piece = _solve_dense(
                A,
                B[..., start:end],
                upper=False,
                unitriangular=unitriangular,
                out=None,
                coefficient=coefficient,
                original=original,
            )
            _copy_matrix(result[..., start:end], piece)
        return result
    result = (
        torch.empty((*batch, k, n), dtype=B.dtype, device=B.device).mT
        if out is None
        else out
    )
    if n <= 128 or not USE_INVERSE_PATH:
        panel = min(16 if B.dtype == torch.float64 else 32, triton.next_power_of_2(n))
        # The coefficient copy runs as its own flat kernel: the masked 32x32
        # tiles the panel kernel used for it only stored their first lane on
        # gcu400 (measured: rows/columns 1..N-1 of the copy left zero whenever
        # N is not a multiple of 32).
        width = k
        # gcu400 caps grid.y at 255 (CUDA allows 65535), so the flat 1-D grid
        # is needed far earlier than the CUDA threshold.
        flat = width > 255
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
                False,
                flat,
            ),
        )
        _copy_coefficient(coefficient, original)
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
        # grid.y is capped at 255 on gcu400; n=512 already needs width > 255.
        flat = width > 255
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
        # gcu400 caps grid.y at 255, so split into the flat grid once k (the
        # y extent here) outgrows it.
        flat = k > 255 or panels > 65535
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
            num_warps=_PANEL_NUM_WARPS,
        )
    return result


def _solve_dense(A, B, *, upper, unitriangular, out=None, coefficient, original):
    if B.dtype == torch.float64:
        # gcu400 raises fp64 before reaching here; keep the verified legacy
        # path for backends that report fp64 support.
        return _solve_dense_legacy(
            A,
            B,
            upper=upper,
            unitriangular=unitriangular,
            out=out,
            coefficient=coefficient,
            original=original,
        )
    n, k = B.shape[-2:]
    batch = tuple(B.shape[:-2])
    batches = math.prod(batch)
    result = (
        out
        if out is not None
        else torch.empty((*batch, k, n), dtype=B.dtype, device=B.device).mT
    )
    xb = n * k
    ainv = None
    if n > 192:
        # Above the walk crossover the per-k-slice CTAs would each redo the
        # k-independent Neumann chain per block; precompute the block inverses
        # once (measured on this family: see linalg_solve_triangular).
        block = 128
        num_blocks = (n + block - 1) // block
        # K1 reads each block as an unmasked square tile (the only addressing
        # form that stays exact at 128x128 on this backend), so ragged orders
        # get a zero-padded coefficient copy; full-block orders skip the copy.
        a_build = A
        if n % block:
            padded = block * num_blocks
            a_build = torch.zeros(
                (*batch, padded, padded), dtype=A.dtype, device=A.device
            )
            _copy_matrix(a_build, A)
        ainv = torch.empty(
            batches * num_blocks * block * block, dtype=B.dtype, device=B.device
        )
        _tri_ainv_build[(batches * num_blocks,)](
            a_build,
            ainv,
            n,
            batch,
            tuple(a_build.stride()),
            block,
            upper,
            unitriangular,
            num_blocks,
            num_warps=1,
        )
    _solve_columns(
        A,
        B,
        result,
        n,
        k,
        batch,
        batches,
        upper,
        unitriangular,
        ainv,
        xb,
    )
    if coefficient is not None and coefficient.numel():
        _copy_coefficient_batched(coefficient, original)
    return result


def triangular_solve(B, A, upper=True, transpose=False, unitriangular=False):
    logger.debug("GEMS_ENFLAME TRIANGULAR_SOLVE")
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
    logger.debug("GEMS_ENFLAME TRIANGULAR_SOLVE_OUT")
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
