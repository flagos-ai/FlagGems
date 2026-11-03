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
from collections import OrderedDict

import torch
import triton
import triton.language as tl
from triton.backends.hcu.compiler import HIPBackend
from triton.runtime import driver

from flag_gems import runtime
from flag_gems.ops.triangular_solve import _triangular_solve

logger = logging.getLogger(__name__)

# Small kernels compile without pointer attributes. Large kernels retain the
# exact HCU alignment and storage-range specialization in the launch-cache key.
# Bounded caches store compiled code and scalar metadata, never tensors or streams.
_launch_cache = OrderedDict()
_launch_lock = threading.RLock()


def _launch(kernel, grid, args, tensor_count, num_warps=4):
    # User hooks run on every ordinary JIT call and may invoke other kernels.
    # Keep that path outside our cache lock.
    if kernel.pre_run_hooks:
        kernel[grid](*args, num_warps=num_warps)
        return
    key = (
        id(kernel),
        driver.active.get_current_device(),
        grid,
        num_warps,
        (
            kernel.debug or triton.knobs.runtime.debug,
            triton.knobs.compilation.instrumentation_mode,
        ),
        (
            tuple(
                (t.dtype, HIPBackend.get_tensor_specialization(t, align=True))
                for t in args[:tensor_count]
            )
            if kernel is not _resident_solve_kernel
            else tuple(t.dtype for t in args[:tensor_count])
        ),
        args[tensor_count:],
    )
    with _launch_lock:
        compiled = _launch_cache.get(key)
        if compiled is None:
            compiled = kernel[grid](*args, num_warps=num_warps)
            _launch_cache[key] = compiled
            if len(_launch_cache) > 128:
                _launch_cache.popitem(last=False)
            return
        _launch_cache.move_to_end(key)
    # CompiledKernel's public launcher queries the active stream on every call;
    # streams and tensor addresses are never captured by this cache.
    compiled[grid + (1,) * (3 - len(grid))](*args)


@triton.constexpr_function
def _requires_int64(
    batch, n, k, a_strides, b_strides=(), c_strides=(), o_strides=(), panel=0
):
    # Evaluated only by the compiler: no tensor inspection or per-call host work.
    # Include batch displacement before testing any matrix-stride product.
    spans = [math.prod(batch) * n * k]
    for strides, columns in (
        (a_strides, n),
        (b_strides, k),
        (c_strides, n),
        (o_strides, n),
    ):
        if strides:
            spans.append(
                sum(
                    size * stride for size, stride in zip((*batch, n, columns), strides)
                )
            )
    if panel:
        panels = (n + panel - 1) // panel
        spans.extend(
            (
                math.prod(batch) * panels * panel * panel,
                math.prod(batch) * k * (panels + 1),
            )
        )
    return max(spans) > 2147483647


@triton.jit(do_not_specialize=["A", "B", "X", "COEFFICIENT", "ORIGINAL"])
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
    INDEX64: tl.constexpr = _requires_int64(
        BATCH.value,
        N,
        K,
        A_STRIDES.value,
        B_STRIDES.value,
        C_STRIDES.value,
        O_STRIDES.value,
    )
    # One program owns complete RHS columns. Keep their dependent updates in
    # registers, reading the original input strides without materialization.
    rows = tl.arange(0, ROWS)
    cols = tl.arange(0, COLS)
    batch = tl.program_id(0)
    col_tile = tl.program_id(1)
    if INDEX64:
        rows = rows.to(tl.int64)
        cols = cols.to(tl.int64)
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
        for step in range(N):
            row = N - 1 - step if UPPER else step
            if INDEX64:
                row = row.to(tl.int64)
            selected = rows == row
            value = tl.sum(tl.where(selected[None, :], x, 0.0), axis=1)
            if not UNIT:
                diagonal = tl.load(a_ptr + row * (A_STRIDES[-2] + A_STRIDES[-1]))
                value = value / diagonal
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


@triton.jit
def _prepare_inverse_kernel(
    A,
    I,
    GOOD,
    STATE,
    COEFFICIENT,
    ORIGINAL,
    K: tl.constexpr,
    C_STRIDES: tl.constexpr,
    O_STRIDES: tl.constexpr,
    N: tl.constexpr,
    BATCH: tl.constexpr,
    A_STRIDES: tl.constexpr,
    UPPER: tl.constexpr,
    UNIT: tl.constexpr,
    PANEL: tl.constexpr,
):
    INDEX64: tl.constexpr = _requires_int64(
        BATCH.value,
        N,
        K,
        A_STRIDES.value,
        c_strides=C_STRIDES.value,
        o_strides=O_STRIDES.value,
        panel=PANEL,
    )
    block = tl.program_id(1)
    if INDEX64:
        block = block.to(tl.int64)
    batch = tl.program_id(0).to(tl.int64)
    remainder = batch
    a_offset = tl.full((), 0, tl.int64)
    c_offset = tl.full((), 0, tl.int64)
    o_offset = tl.full((), 0, tl.int64)
    for dim in tl.static_range(len(BATCH) - 1, -1, -1):
        coordinate = remainder % BATCH[dim]
        remainder //= BATCH[dim]
        a_offset += coordinate * A_STRIDES[dim]
        c_offset += coordinate * C_STRIDES[dim]
        o_offset += coordinate * O_STRIDES[dim]
    A += a_offset
    solve_tiles = tl.cdiv(N, PANEL)
    copy_axis = tl.cdiv(N, 32)
    if INDEX64:
        solve_tiles = solve_tiles.to(tl.int64)
        copy_axis = copy_axis.to(tl.int64)
    copy_tiles = copy_axis * copy_axis
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
        r = tl.arange(0, PANEL)
        if INDEX64:
            r = r.to(tl.int64)
        row = block * PANEL + r
        ar = N - 1 - row if UPPER else row
        a = tl.load(
            A + ar[:, None] * A_STRIDES[-2] + ar[None, :] * A_STRIDES[-1],
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
                A + ar * (A_STRIDES[-2] + A_STRIDES[-1]), row < N, other=1
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
):
    INDEX64: tl.constexpr = _requires_int64(
        BATCH.value, N, K, A_STRIDES.value, B_STRIDES.value, panel=PANEL
    )
    batch = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1)
    if INDEX64:
        col = col.to(tl.int64)
    remainder = batch
    a_offset = tl.full((), 0, tl.int64)
    b_offset = tl.full((), 0, tl.int64)
    for dim in tl.static_range(len(BATCH) - 1, -1, -1):
        coordinate = remainder % BATCH[dim]
        remainder //= BATCH[dim]
        a_offset += coordinate * A_STRIDES[dim]
        b_offset += coordinate * B_STRIDES[dim]
    A += a_offset
    B += b_offset
    X += batch * N * K
    I += batch * tl.cdiv(N, PANEL) * PANEL * PANEL
    GOOD += batch * tl.cdiv(N, PANEL)
    r = tl.arange(0, PANEL)
    if INDEX64:
        r = r.to(tl.int64)
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
    b = tl.load(B + ar * B_STRIDES[-2] + col * B_STRIDES[-1], row < N, other=0)
    for prior in range(block):
        pr = prior * PANEL + r
        ap = N - 1 - pr if UPPER else pr
        a = tl.load(
            A + ar[:, None] * A_STRIDES[-2] + ap[None, :] * A_STRIDES[-1],
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
            A + ar[:, None] * A_STRIDES[-2] + ar[None, :] * A_STRIDES[-1],
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
                    * (A_STRIDES[-2] + A_STRIDES[-1]),
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


def _solve(A, B, *, upper, unitriangular, out=None, coefficient, original):
    n, k = B.shape[-2:]
    batch = tuple(B.shape[:-2])
    result = (
        torch.empty((*batch, k, n), dtype=B.dtype, device=B.device).mT
        if out is None
        else out
    )
    if n <= 128:
        cols = min(4, 1 << (k - 1).bit_length())
        _launch(
            _resident_solve_kernel,
            (math.prod(batch), ((k + cols - 1) // cols) + ((n + 31) // 32) ** 2),
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
                1 << (n - 1).bit_length(),
                cols,
            ),
            5,
        )
    else:
        batches = math.prod(batch)
        panel = 16 if B.dtype == torch.float64 else 32
        panels = (n + panel - 1) // panel
        inverse = torch.empty(
            (batches, panels, panel, panel), dtype=B.dtype, device=B.device
        )
        good = torch.empty((batches, panels), dtype=torch.int32, device=B.device)
        state = torch.empty(
            (batches, k, panels + 1), dtype=torch.int32, device=B.device
        )
        _launch(
            _prepare_inverse_kernel,
            (batches, panels + ((n + 31) // 32) ** 2 + (k * (panels + 1) + 255) // 256),
            (
                A,
                inverse,
                good,
                state,
                coefficient,
                original,
                k,
                tuple(coefficient.stride()),
                tuple(original.stride()),
                n,
                batch,
                tuple(A.stride()),
                upper,
                unitriangular,
                panel,
            ),
            6,
        )
        clone = coefficient if A.stride() == original.stride() else coefficient.mT
        if clone.stride(-1) < A.stride(-1):
            A = clone
        _launch(
            _inverse_solve_kernel,
            (batches, k, panels),
            (
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
                1 << (n - 1).bit_length(),
                panel,
            ),
            6,
            num_warps=1,
        )
    return result


def _uncached_triangular_solve(B, A, upper=True, transpose=False, unitriangular=False):
    logger.debug("GEMS_HYGON TRIANGULAR_SOLVE")
    return _triangular_solve(
        B,
        A,
        upper,
        transpose,
        unitriangular,
        _solve,
        solver_out=True,
        solver_coeff=True,
    )


def _uncached_triangular_solve_out(
    B, A, upper=True, transpose=False, unitriangular=False, *, X, M
):
    logger.debug("GEMS_HYGON TRIANGULAR_SOLVE_OUT")
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
    )


_default_plans = OrderedDict()


def triangular_solve(B, A, upper=True, transpose=False, unitriangular=False):
    logger.debug("GEMS_HYGON TRIANGULAR_SOLVE")
    if (
        type(A) is not torch.Tensor
        or type(B) is not torch.Tensor
        or A.layout != torch.strided
        or B.layout != torch.strided
        or not B.numel()
        or not all(type(v) is bool for v in (upper, transpose, unitriangular))
    ):
        return _uncached_triangular_solve(B, A, upper, transpose, unitriangular)
    key = (
        A.shape,
        A.stride(),
        A.dtype,
        A.device,
        A.layout,
        B.shape,
        B.stride(),
        B.dtype,
        B.device,
        B.layout,
        upper,
        transpose,
        unitriangular,
        torch.is_grad_enabled(),
        A.requires_grad,
        B.requires_grad,
    )
    with _launch_lock:
        plan = _default_plans.get(key)
        if plan is not None:
            _default_plans.move_to_end(key)
    if plan is None:
        result = _uncached_triangular_solve(B, A, upper, transpose, unitriangular)
        x, m = result
        if not x.numel():
            return result
        plan = (
            tuple(x.shape),
            tuple(x.stride()),
            tuple(m.shape),
            tuple(m.stride()),
            B.shape != x.shape,
            A.shape != m.shape,
        )
        with _launch_lock:
            _default_plans[key] = plan
            if len(_default_plans) > 128:
                _default_plans.popitem(last=False)
        return result
    x_shape, x_stride, m_shape, m_stride, expand_b, expand_a = plan
    with runtime.torch_device_fn.device(B.device):
        x = torch.empty_strided(x_shape, x_stride, dtype=B.dtype, device=B.device)
        m = torch.empty_strided(m_shape, m_stride, dtype=A.dtype, device=A.device)
        if expand_a:
            A = A.expand(m_shape)
        if expand_b:
            B = B.expand(x_shape)
        _solve(
            A.mT if transpose else A,
            B,
            upper=not upper if transpose else upper,
            unitriangular=unitriangular,
            out=x,
            coefficient=m,
            original=A,
        )
        return x, m


_out_plans = OrderedDict()


def _output_key(B, A, X, M, upper, transpose, unitriangular):
    return tuple(
        (
            tuple(t.shape),
            tuple(t.stride()),
            t.dtype,
            t.device,
            t.layout,
            t.requires_grad,
        )
        for t in (B, A, X, M)
    ) + (upper, transpose, unitriangular, torch.is_grad_enabled())


def triangular_solve_out(
    B, A, upper=True, transpose=False, unitriangular=False, *, X, M
):
    logger.debug("GEMS_HYGON TRIANGULAR_SOLVE_OUT")
    if (
        any(
            type(t) is not torch.Tensor or t.layout != torch.strided
            for t in (B, A, X, M)
        )
        or not B.numel()
        or not all(type(v) is bool for v in (upper, transpose, unitriangular))
        or any(
            torch._C._overlaps(a, b)
            for a, b in ((X, A), (X, B), (M, A), (M, B), (X, M))
        )
    ):
        return _uncached_triangular_solve_out(
            B, A, upper, transpose, unitriangular, X=X, M=M
        )
    key = _output_key(B, A, X, M, upper, transpose, unitriangular)
    with _launch_lock:
        plan = _out_plans.get(key)
        if plan is not None:
            _out_plans.move_to_end(key)
    if plan is None:
        result = _uncached_triangular_solve_out(
            B, A, upper, transpose, unitriangular, X=X, M=M
        )
        if not X.numel():
            return result
        n, k = X.shape[-2:]
        sx = [n, 1]
        sm = [n, 1]
        bx, bm = n * k, n * n
        for size in reversed(X.shape[:-2]):
            sx.append(bx)
            sm.append(bm)
            bx *= size
            bm *= size
        if X.stride() == tuple(reversed(sx)) and M.stride() == tuple(reversed(sm)):
            # Key the validated shapes after any native-compatible resize.
            key = _output_key(B, A, X, M, upper, transpose, unitriangular)
            plan = (
                tuple(X.shape),
                tuple(M.shape),
                B.shape != X.shape,
                A.shape != M.shape,
            )
            with _launch_lock:
                _out_plans[key] = plan
                if len(_out_plans) > 128:
                    _out_plans.popitem(last=False)
        return result
    x_shape, m_shape, expand_b, expand_a = plan
    with runtime.torch_device_fn.device(B.device):
        if expand_a:
            A = A.expand(m_shape)
        if expand_b:
            B = B.expand(x_shape)
        _solve(
            A.mT if transpose else A,
            B,
            upper=not upper if transpose else upper,
            unitriangular=unitriangular,
            out=X,
            coefficient=M,
            original=A,
        )
        return X, M
