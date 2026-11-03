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
import threading
from importlib.metadata import PackageNotFoundError, version

import torch
import triton
import triton.language as tl

from flag_gems.ops.triangular_solve import _copy_matrix, _triangular_solve
from flag_gems.utils import libentry

from .linalg_solve_triangular import _num_cores, linalg_solve_triangular

logger = logging.getLogger(__name__)
_launch_cache = {}
_default_plan_cache = {}
_launch_lock = threading.Lock()


def _launch_resident(entry, pointers, constants, grid):
    # This Ascend compiler specializes tensor arguments by dtype and 16-byte
    # alignment. All remaining arguments are constexpr, and the configuration
    # is fixed. Keep device/debug identity and every specialization in the key.
    if not _USE_LAUNCH_CACHE:
        entry[grid](*pointers, *constants)
        return
    debug = triton.knobs.runtime.debug
    key = (
        entry,
        pointers[0].device,
        tuple((tensor.dtype, tensor.data_ptr() % 16 == 0) for tensor in pointers),
        constants,
        debug,
    )
    kernel = _launch_cache.get(key)
    if kernel is None:
        with _launch_lock:
            kernel = _launch_cache.get(key)
            if kernel is None:
                kernel = entry.jit_function[grid](*pointers, *constants, debug=debug)
                if len(_launch_cache) >= 256:
                    _launch_cache.clear()
                _launch_cache[key] = kernel
                return
    # Resolve the current stream for the validated input device on every launch.
    stream = triton.runtime.driver.active.get_current_stream(pointers[0].device.index)
    kernel[(grid[0], 1, 1)](*pointers, *constants, stream=stream)


# These compiler closures cannot lower the resident substitution loop.
# Use streamed substitution or the existing pure-Triton panel solver instead.
try:
    _USE_PANEL_SOLVER = (triton.__version__, version("flagtree")) in (
        ("3.2.0", "0.6.0+ascend3.2"),
        ("3.5.1", "0.7.0+ascend3.5"),
    )
    _USE_STREAMED_SOLVER = (
        triton.__version__ == "3.5.1" and version("flagtree") == "0.7.0+ascend3.5"
    )
    _USE_LAUNCH_CACHE = triton.__version__ == "3.5.1" and version("flagtree") in (
        "0.6.0+ascend3.5",
        "0.7.0+ascend3.5",
    )
except PackageNotFoundError:
    _USE_PANEL_SOLVER = False
    _USE_STREAMED_SOLVER = False
    _USE_LAUNCH_CACHE = False


def _wide_matrix_offsets(batch, strides, rows, cols):
    span = sum((size - 1) * stride for size, stride in zip(batch, strides))
    span += (rows - 1) * strides[-2] + (cols - 1) * strides[-1]
    return span >= 2**31


# Evaluated only while compiling the resident kernels. The older compiler uses
# the panel solver and must still be able to import this module.
if hasattr(triton, "constexpr_function"):
    _wide_matrix_offsets = triton.constexpr_function(_wide_matrix_offsets)


@triton.jit
def _resident_solve_kernel(
    A,
    B,
    X,
    N: tl.constexpr,
    K: tl.constexpr,
    BATCH: tl.constexpr,
    A_STRIDES: tl.constexpr,
    B_STRIDES: tl.constexpr,
    UPPER: tl.constexpr,
    UNIT: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    ITEMS: tl.constexpr,
):
    # Keep the RHS tile in UB for the entire substitution. Each program owns
    # complete RHS columns, so there are no inter-program dependencies or
    # global store/load round trips between dependent rows.
    WIDE: tl.constexpr = (
        _wide_matrix_offsets(BATCH, A_STRIDES, ROWS, ROWS)
        or _wide_matrix_offsets(BATCH, B_STRIDES, ROWS, ((K + COLS - 1) // COLS) * COLS)
        or ITEMS * COLS * N >= 2**31
        or 2 * ITEMS >= 2**31
    )
    rows = tl.arange(0, ROWS)
    cols = tl.arange(0, COLS)
    if WIDE:
        rows = rows.to(tl.int64)
        cols = cols.to(tl.int64)
        per_core = tl.cdiv(tl.full((), ITEMS, tl.int64), tl.num_programs(0))
        start = tl.program_id(0).to(tl.int64) * per_core
    else:
        per_core = tl.cdiv(ITEMS, tl.num_programs(0))
        start = tl.program_id(0) * per_core
    for item in range(start, tl.minimum(start + per_core, ITEMS)):
        batch = item // tl.cdiv(K, COLS)
        col_tile = item - batch * tl.cdiv(K, COLS)
        remainder = batch
        a_offset = 0
        b_offset = 0
        for dim in tl.static_range(len(BATCH) - 1, -1, -1):
            coord = remainder - (remainder // BATCH[dim]) * BATCH[dim]
            remainder = remainder // BATCH[dim]
            a_offset += coord * A_STRIDES[dim]
            b_offset += coord * B_STRIDES[dim]
        a_ptr = A + a_offset
        b_ptr = B + b_offset
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
        for step in range(N):
            row = N - 1 - step if UPPER else step
            selected = rows == row
            # Select the pivot directly; a full one-hot reduction per row
            # wastes work and expands the loop-carried UB live range.
            indices = tl.full((COLS, 1), row, tl.int32)
            value = tl.gather(x, indices, axis=1)
            if not UNIT:
                pivot = tl.gather(diagonal, tl.full((1,), row, tl.int32), axis=0)
                value = value / pivot[None, :]
            remaining = rows < row if UPPER else rows > row
            # Mask before loading so a poisoned unused triangle or implicit
            # unit diagonal can never enter the arithmetic.
            address_row = row.to(tl.int64) if WIDE else row
            column = tl.load(
                a_ptr + rows * A_STRIDES[-2] + address_row * A_STRIDES[-1],
                mask=(rows < N) & remaining,
                other=0.0,
            )
            x = tl.where(remaining[None, :], x - value * column[None, :], x)
            x = tl.where(selected[None, :], value, x)
            # Order the loop-carried UB update before the next iteration or
            # final store on Ascend, including a specialization's first launch.
            tl.debug_barrier()
        tl.store(X + batch * N * K + rhs[:, None] * N + rows[None, :], x, valid)


@triton.jit
def _streamed_solve_kernel(
    A,
    B,
    X,
    N: tl.constexpr,
    K: tl.constexpr,
    BATCH: tl.constexpr,
    A_STRIDES: tl.constexpr,
    B_STRIDES: tl.constexpr,
    UPPER: tl.constexpr,
    UNIT: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    ITEMS: tl.constexpr,
):
    # Store completed rows in the column-major output. This avoids a
    # loop-carried RHS tensor, which the newer compiler cannot plan in UB.
    WIDE: tl.constexpr = (
        _wide_matrix_offsets(BATCH, A_STRIDES, ROWS, ROWS)
        or _wide_matrix_offsets(BATCH, B_STRIDES, ROWS, ((K + COLS - 1) // COLS) * COLS)
        or ITEMS * COLS * N >= 2**31
        or 2 * ITEMS >= 2**31
    )
    rows = tl.arange(0, ROWS)
    cols = tl.arange(0, COLS)
    if WIDE:
        rows = rows.to(tl.int64)
        cols = cols.to(tl.int64)
        per_core = tl.cdiv(tl.full((), ITEMS, tl.int64), tl.num_programs(0))
        start = tl.program_id(0).to(tl.int64) * per_core
    else:
        per_core = tl.cdiv(ITEMS, tl.num_programs(0))
        start = tl.program_id(0) * per_core
    for item in range(start, tl.minimum(start + per_core, ITEMS)):
        batch = item // tl.cdiv(K, COLS)
        col_tile = item - batch * tl.cdiv(K, COLS)
        remainder = batch
        a_offset = 0
        b_offset = 0
        for dim in tl.static_range(len(BATCH) - 1, -1, -1):
            coord = remainder - (remainder // BATCH[dim]) * BATCH[dim]
            remainder = remainder // BATCH[dim]
            a_offset += coord * A_STRIDES[dim]
            b_offset += coord * B_STRIDES[dim]
        a_ptr = A + a_offset
        b_ptr = B + b_offset
        rhs = col_tile * COLS + cols
        x_ptr = X + batch * N * K
        for step in range(N):
            row = N - 1 - step if UPPER else step
            address_row = row.to(tl.int64) if WIDE else row
            solved = rows > row if UPPER else rows < row
            weights = tl.load(
                a_ptr + address_row * A_STRIDES[-2] + rows * A_STRIDES[-1],
                mask=(rows < N) & solved,
                other=0.0,
            )
            previous = tl.load(
                x_ptr + rhs[:, None] * N + rows[None, :],
                mask=(rhs[:, None] < K) & (rows[None, :] < N) & solved[None, :],
                other=0.0,
            )
            value = tl.load(
                b_ptr + address_row * B_STRIDES[-2] + rhs * B_STRIDES[-1],
                mask=rhs < K,
                other=0.0,
            ) - tl.sum(previous * weights[None, :], axis=1)
            if not UNIT:
                diagonal = tl.load(
                    a_ptr + address_row * (A_STRIDES[-2] + A_STRIDES[-1])
                )
                value = value / diagonal
            tl.store(x_ptr + rhs * N + address_row, value, rhs < K)
            tl.debug_barrier()


_streamed_solve_entry = libentry()(_streamed_solve_kernel)

_resident_solve_entry = libentry()(_resident_solve_kernel)


@libentry()
@triton.jit
def _resident_solve_fused_kernel(
    A,
    B,
    X,
    ORIGINAL,
    COEFFICIENT,
    N: tl.constexpr,
    K: tl.constexpr,
    BATCH: tl.constexpr,
    A_STRIDES: tl.constexpr,
    B_STRIDES: tl.constexpr,
    ORIGINAL_STRIDES: tl.constexpr,
    COEFFICIENT_STRIDES: tl.constexpr,
    UPPER: tl.constexpr,
    UNIT: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    ITEMS: tl.constexpr,
    STREAMED: tl.constexpr = False,
):
    # Small systems share one launch for the solve and full original-A copy.
    # Copy tiles are distributed over the same programs as the RHS columns.
    WIDE: tl.constexpr = (
        _wide_matrix_offsets(
            BATCH, ORIGINAL_STRIDES, ((N + 31) // 32) * 32, ((N + 31) // 32) * 32
        )
        or _wide_matrix_offsets(
            BATCH, COEFFICIENT_STRIDES, ((N + 31) // 32) * 32, ((N + 31) // 32) * 32
        )
        or 2 * (ITEMS // ((K + COLS - 1) // COLS)) * ((N + 31) // 32) * ((N + 31) // 32)
        >= 2**31
    )
    tiles = tl.cdiv(N, 32)
    if WIDE:
        batches = tl.full((), ITEMS // ((K + COLS - 1) // COLS), tl.int64)
        start = tl.program_id(0).to(tl.int64)
    else:
        batches = ITEMS // tl.cdiv(K, COLS)
        start = tl.program_id(0)
    for tile in range(start, batches * tiles * tiles, tl.num_programs(0)):
        col_tile = tile % tiles
        row_tile = (tile // tiles) % tiles
        batch = tile // (tiles * tiles)
        source_offset = 0
        destination_offset = 0
        for dim in tl.static_range(len(BATCH) - 1, -1, -1):
            coordinate = batch % BATCH[dim]
            batch //= BATCH[dim]
            source_offset += coordinate * ORIGINAL_STRIDES[dim]
            destination_offset += coordinate * COEFFICIENT_STRIDES[dim]
        rows = row_tile * 32 + tl.arange(0, 32)
        cols = col_tile * 32 + tl.arange(0, 32)
        valid = (rows[:, None] < N) & (cols[None, :] < N)
        original = tl.load(
            ORIGINAL
            + source_offset
            + rows[:, None] * ORIGINAL_STRIDES[-2]
            + cols[None, :] * ORIGINAL_STRIDES[-1],
            valid,
            other=0.0,
        )
        tl.store(
            COEFFICIENT
            + destination_offset
            + rows[:, None] * COEFFICIENT_STRIDES[-2]
            + cols[None, :] * COEFFICIENT_STRIDES[-1],
            original,
            valid,
        )
    if STREAMED:
        _streamed_solve_kernel(
            A,
            B,
            X,
            N,
            K,
            BATCH,
            A_STRIDES,
            B_STRIDES,
            UPPER,
            UNIT,
            ROWS,
            COLS,
            ITEMS,
        )
    else:
        _resident_solve_kernel(
            A,
            B,
            X,
            N,
            K,
            BATCH,
            A_STRIDES,
            B_STRIDES,
            UPPER,
            UNIT,
            ROWS,
            COLS,
            ITEMS,
        )


def _resident_solve(
    A, B, *, upper, unitriangular, out=None, coefficient=None, original=None
):
    n, k = B.shape[-2:]
    streamed = _USE_STREAMED_SOLVER and n <= 1024
    if _USE_PANEL_SOLVER and not streamed:
        if coefficient is not None:
            _copy_matrix(coefficient, original)
        result = linalg_solve_triangular(A, B, upper=upper, unitriangular=unitriangular)
        if out is not None:
            _copy_matrix(out, result)
            return out
        return result
    batch = tuple(B.shape[:-2])
    result = (
        out
        if out is not None
        else torch.empty((*batch, k, n), dtype=B.dtype, device=B.device).mT
    )
    # Limit the live RHS tile for large orders: four columns corrupt the
    # loop-carried state at n=1024 on the supported Ascend 3.5 compiler.
    cols = 1 if n > 512 and not streamed else min(4, 1 << (k - 1).bit_length())
    items = math.prod(batch) * ((k + cols - 1) // cols)
    grid = (min(items, _num_cores()),)
    args = (
        n,
        k,
        batch,
        tuple(A.stride()),
        tuple(B.stride()),
    )
    solve_args = (upper, unitriangular, 1 << (n - 1).bit_length(), cols, items)
    if coefficient is not None and n <= 128:
        solve_args = (*solve_args, streamed)
        _launch_resident(
            _resident_solve_fused_kernel,
            (A, B, result, original, coefficient),
            (*args, tuple(original.stride()), tuple(coefficient.stride()), *solve_args),
            grid,
        )
    else:
        if coefficient is not None:
            _copy_matrix(coefficient, original)
        _launch_resident(
            _streamed_solve_entry if streamed else _resident_solve_entry,
            (A, B, result),
            (*args, *solve_args),
            grid,
        )
    return result


def triangular_solve(B, A, upper=True, transpose=False, unitriangular=False):
    logger.debug("GEMS_ASCEND TRIANGULAR_SOLVE")
    key = None
    plan = None
    if (
        _USE_LAUNCH_CACHE
        and type(A) is torch.Tensor
        and type(B) is torch.Tensor
        and type(upper) is bool
        and type(transpose) is bool
        and type(unitriangular) is bool
        and A.layout == B.layout == torch.strided
        and not (A.requires_grad or B.requires_grad)
    ):
        key = (
            tuple(A.shape),
            tuple(B.shape),
            tuple(A.stride()),
            tuple(B.stride()),
            A.dtype,
            B.dtype,
            A.device,
            B.device,
            A.layout,
            B.layout,
            upper,
            transpose,
            unitriangular,
        )
        plan = _default_plan_cache.get(key)
    if plan is None:
        outputs = _triangular_solve(
            B,
            A,
            upper,
            transpose,
            unitriangular,
            _resident_solve,
            solver_out=True,
            solver_coeff=True,
        )
        # A nonempty B can broadcast to an empty output batch.
        if key is not None and outputs[0].numel():
            solution, coefficient = outputs
            plan = (
                tuple(solution.shape),
                tuple(solution.stride()),
                tuple(coefficient.shape),
                tuple(coefficient.stride()),
                A.shape != coefficient.shape,
                B.shape != solution.shape,
            )
            if len(_default_plan_cache) >= 256:
                _default_plan_cache.clear()
            _default_plan_cache[key] = plan
        return outputs
    # A metadata-only plan is published only after the shared helper has
    # validated this exact signature. Rebuild views from the current inputs and
    # allocate both outputs afresh; no tensor, storage, or stream is retained.
    x_shape, x_strides, a_shape, a_strides, expand_a, expand_b = plan
    with torch.npu.device(B.device):
        if expand_a:
            A = A.expand(a_shape)
        if expand_b:
            B = B.expand(x_shape)
        solution = torch.empty_strided(
            x_shape, x_strides, dtype=B.dtype, device=B.device
        )
        coefficient = torch.empty_strided(
            a_shape, a_strides, dtype=A.dtype, device=A.device
        )
        _resident_solve(
            A.mT if transpose else A,
            B,
            upper=not upper if transpose else upper,
            unitriangular=unitriangular,
            out=solution,
            coefficient=coefficient,
            original=A,
        )
        return solution, coefficient


def triangular_solve_out(
    B, A, upper=True, transpose=False, unitriangular=False, *, X, M
):
    logger.debug("GEMS_ASCEND TRIANGULAR_SOLVE_OUT")
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
    )
