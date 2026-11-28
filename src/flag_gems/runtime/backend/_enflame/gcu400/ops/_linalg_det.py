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
import sys

import torch
import triton
import triton.language as tl

import flag_gems.ops
from flag_gems.runtime import device as runtime_device
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_DET_BLOCK_MAX = 64

# gcu400 has no fp64 datapath: the elimination kernels cannot legalize f64
# arithmetic (the pivot search fails on `math.absf` with a PassManager error).
# Follow the torch_gcu convention (same as ``cholesky_inverse``): when the
# device has no native fp64, carry the computation in float32 and restore the
# requested dtype on the outputs.
_HAS_FP64 = bool(getattr(runtime_device, "support_fp64", True))


@triton.jit
def _reduce_mul(a, b):
    return a * b


# The elimination kernels follow the vectorized design of
# ``flag_gems/ops/linalg_det.py`` (blocked / panel paths, float64 tuned to
# PANEL=16/BLOCK=32). They are vendored here rather than imported because
# ``aten::_linalg_det`` must also return the factorization and pivots, which
# requires storing the in-place working matrix (the combined L/U form) and
# the per-step pivot rows — outputs the upstream ``det``-only kernels do not
# produce. The register-tile kernels from the earlier dispatch are retired:
# the single-panel LU-working kernel below replaces them for float32 n <= 128
# (measured faster at every probed size; see the ``_det_luwork_kernel``
# comment and the dispatch note in ``_linalg_det``).


@libentry()
@triton.jit
def _det_blocked_kernel(
    A,
    out,
    LU_out,
    pivots_out,
    N,
    BLOCK: tl.constexpr,
):
    pid = tle.program_id(0)
    base = pid * N * N
    swap_count = tl.zeros((), dtype=tl.int32)

    for k in range(N):
        best_val = tl.full((), -1.0, dtype=A.dtype.element_ty)
        best_row = tl.full((), k, dtype=tl.int32)
        for i0 in range(k, N, BLOCK):
            rows = i0 + tl.arange(0, BLOCK)
            col = tl.load(A + base + rows * N + k, mask=rows < N, other=0.0)
            abs_col = tl.where((rows >= k) & (rows < N), tl.abs(col), -1.0)
            tile_max = tl.max(abs_col, axis=0)
            tile_row = tl.min(tl.where(abs_col == tile_max, rows, N), axis=0)
            is_better = tile_max > best_val
            best_row = tl.where(is_better, tile_row, best_row)
            best_val = tl.where(is_better, tile_max, best_val)
        # See the register kernel: skip the swap for an all-zero pivot column.
        best_row = tl.where(best_val <= 0.0, k, best_row)

        for j0 in range(k, N, BLOCK):
            cols = j0 + tl.arange(0, BLOCK)
            cmask = cols < N
            row_k = tl.load(A + base + k * N + cols, mask=cmask, other=0.0)
            row_p = tl.load(A + base + best_row * N + cols, mask=cmask, other=0.0)
            tl.store(A + base + k * N + cols, row_p, mask=cmask)
            tl.store(A + base + best_row * N + cols, row_k, mask=cmask)
        swap_count = tl.where(best_row != k, swap_count + 1, swap_count)
        tl.store(pivots_out + pid * N + k, best_row + 1)

        tl.debug_barrier()

        pivot = tl.load(A + base + k * N + k)
        safe_pivot = tl.where(pivot == 0.0, 1.0, pivot)

        for i0 in range(k + 1, N, BLOCK):
            rows = i0 + tl.arange(0, BLOCK)
            rmask = rows < N
            col = tl.load(A + base + rows * N + k, mask=rmask, other=0.0)
            tl.store(A + base + rows * N + k, col / safe_pivot, mask=rmask)

        tl.debug_barrier()

        for i0 in range(k + 1, N, BLOCK):
            rows = i0 + tl.arange(0, BLOCK)
            rmask = rows < N
            l_col = tl.load(A + base + rows * N + k, mask=rmask, other=0.0)
            for j0 in range(k + 1, N, BLOCK):
                cols = j0 + tl.arange(0, BLOCK)
                cmask = cols < N
                u_row = tl.load(A + base + k * N + cols, mask=cmask, other=0.0)
                tmask = rmask[:, None] & cmask[None, :]
                tile = tl.load(
                    A + base + rows[:, None] * N + cols[None, :], mask=tmask, other=0.0
                )
                tile = tile - l_col[:, None] * u_row[None, :]
                tl.store(A + base + rows[:, None] * N + cols[None, :], tile, mask=tmask)

        tl.debug_barrier()

    # Export the factorized matrix (combined L/U, LAPACK getrf layout).
    for i0 in range(0, N, BLOCK):
        rows = i0 + tl.arange(0, BLOCK)
        for j0 in range(0, N, BLOCK):
            cols = j0 + tl.arange(0, BLOCK)
            m = (rows[:, None] < N) & (cols[None, :] < N)
            v = tl.load(A + base + rows[:, None] * N + cols[None, :], mask=m, other=0.0)
            tl.store(LU_out + base + rows[:, None] * N + cols[None, :], v, mask=m)

    det = tl.full((), 1.0, dtype=A.dtype.element_ty)
    for i0 in range(0, N, BLOCK):
        d = i0 + tl.arange(0, BLOCK)
        diag = tl.load(A + base + d * N + d, mask=d < N, other=1.0)
        det = det * tl.reduce(diag, 0, combine_fn=_reduce_mul)
    det = tl.where(swap_count % 2 == 0, det, -det)
    tl.store(out + pid, det)


@libentry()
@triton.jit
def _det_panel_kernel(
    A,
    out,
    LU_out,
    pivots_out,
    N,
    PANEL: tl.constexpr,
    BLOCK: tl.constexpr,
    SINGLE: tl.constexpr,
):
    pid = tle.program_id(0)
    base = pid * N * N
    swap_count = tl.zeros((), dtype=tl.int32)

    for k0 in range(0, N, PANEL):
        kend = tl.minimum(k0 + PANEL, N)

        if SINGLE:
            # Single-block fast path (host guarantees BLOCK >= N): every inner
            # chunk loop of the elimination step has exactly one iteration, and
            # each range() costs an integer division on this backend. Flatten
            # the step to straight-line code and reuse the scaled multiplier
            # column from registers. The reload was the only read of column k
            # between the scaling store and the rank-1 update, so the barrier
            # separating them goes away too (the u-row/tile reads touch the
            # swap-written rows, already ordered by the post-swap barrier).
            for k in range(k0, kend):
                rows = k + tl.arange(0, BLOCK)
                # The load fills rows >= N with 0, and abs(0) = 0 never wins the
                # max against a nonzero pivot column; an all-zero (singular)
                # column falls through to the best_val <= 0 guard below.
                col = tl.load(A + base + rows * N + k, mask=rows < N, other=0.0)
                abs_col = tl.abs(col)
                best_val = tl.max(abs_col, axis=0)
                best_row = tl.min(tl.where(abs_col == best_val, rows, N), axis=0)
                best_row = tl.where(best_val <= 0.0, k, best_row)

                cols = k0 + tl.arange(0, BLOCK)
                cmask = cols < N
                row_k = tl.load(A + base + k * N + cols, mask=cmask, other=0.0)
                row_p = tl.load(A + base + best_row * N + cols, mask=cmask, other=0.0)
                tl.store(A + base + k * N + cols, row_p, mask=cmask)
                tl.store(A + base + best_row * N + cols, row_k, mask=cmask)
                swap_count = tl.where(best_row != k, swap_count + 1, swap_count)
                tl.store(pivots_out + pid * N + k, best_row + 1)

                tl.debug_barrier()

                pivot = tl.load(A + base + k * N + k)
                safe_pivot = tl.where(pivot == 0.0, 1.0, pivot)
                rows = k + 1 + tl.arange(0, BLOCK)
                rmask = rows < N
                l_col = tl.load(A + base + rows * N + k, mask=rmask, other=0.0)
                l_col = l_col / safe_pivot
                tl.store(A + base + rows * N + k, l_col, mask=rmask)

                pcols = k + 1 + tl.arange(0, PANEL)
                pcmask = pcols < kend
                u_row = tl.load(A + base + k * N + pcols, mask=pcmask, other=0.0)
                tmask = rmask[:, None] & pcmask[None, :]
                tile = tl.load(
                    A + base + rows[:, None] * N + pcols[None, :],
                    mask=tmask,
                    other=0.0,
                )
                tile = tile - l_col[:, None] * u_row[None, :]
                tl.store(
                    A + base + rows[:, None] * N + pcols[None, :], tile, mask=tmask
                )

                tl.debug_barrier()
        else:
            for k in range(k0, kend):
                best_val = tl.full((), -1.0, dtype=A.dtype.element_ty)
                best_row = tl.full((), k, dtype=tl.int32)
                for i0 in range(k, N, BLOCK):
                    rows = i0 + tl.arange(0, BLOCK)
                    col = tl.load(A + base + rows * N + k, mask=rows < N, other=0.0)
                    abs_col = tl.where((rows >= k) & (rows < N), tl.abs(col), -1.0)
                    tile_max = tl.max(abs_col, axis=0)
                    tile_row = tl.min(tl.where(abs_col == tile_max, rows, N), axis=0)
                    is_better = tile_max > best_val
                    best_row = tl.where(is_better, tile_row, best_row)
                    best_val = tl.where(is_better, tile_max, best_val)
                best_row = tl.where(best_val <= 0.0, k, best_row)

                for j0 in range(k0, N, BLOCK):
                    cols = j0 + tl.arange(0, BLOCK)
                    cmask = cols < N
                    row_k = tl.load(A + base + k * N + cols, mask=cmask, other=0.0)
                    row_p = tl.load(
                        A + base + best_row * N + cols, mask=cmask, other=0.0
                    )
                    tl.store(A + base + k * N + cols, row_p, mask=cmask)
                    tl.store(A + base + best_row * N + cols, row_k, mask=cmask)
                swap_count = tl.where(best_row != k, swap_count + 1, swap_count)
                tl.store(pivots_out + pid * N + k, best_row + 1)

                tl.debug_barrier()

                pivot = tl.load(A + base + k * N + k)
                safe_pivot = tl.where(pivot == 0.0, 1.0, pivot)
                for i0 in range(k + 1, N, BLOCK):
                    rows = i0 + tl.arange(0, BLOCK)
                    rmask = rows < N
                    col = tl.load(A + base + rows * N + k, mask=rmask, other=0.0)
                    tl.store(A + base + rows * N + k, col / safe_pivot, mask=rmask)

                tl.debug_barrier()

                pcols = k + 1 + tl.arange(0, PANEL)
                pcmask = pcols < kend
                u_row = tl.load(A + base + k * N + pcols, mask=pcmask, other=0.0)
                for i0 in range(k + 1, N, BLOCK):
                    rows = i0 + tl.arange(0, BLOCK)
                    rmask = rows < N
                    l_col = tl.load(A + base + rows * N + k, mask=rmask, other=0.0)
                    tmask = rmask[:, None] & pcmask[None, :]
                    tile = tl.load(
                        A + base + rows[:, None] * N + pcols[None, :],
                        mask=tmask,
                        other=0.0,
                    )
                    tile = tile - l_col[:, None] * u_row[None, :]
                    tl.store(
                        A + base + rows[:, None] * N + pcols[None, :], tile, mask=tmask
                    )

                tl.debug_barrier()

        # When the panel reaches the end of the matrix (kend == N) there is no
        # trailing submatrix and every j-loop of the strip phase below is empty,
        # leaving only a strip load + barrier per column. Skip the phase
        # entirely: for n <= PANEL (the whole matrix in one panel, e.g. the
        # f32 n in (64, 128] dispatch) this removes N dead barrier iterations
        # from the serial path.
        if kend < N:
            for c in range(k0, kend):
                prows = c + 1 + tl.arange(0, PANEL)
                prmask = prows < kend
                l_strip = tl.load(A + base + prows * N + c, mask=prmask, other=0.0)
                for j0 in range(kend, N, BLOCK):
                    cols = j0 + tl.arange(0, BLOCK)
                    cmask = cols < N
                    u_strip = tl.load(A + base + c * N + cols, mask=cmask, other=0.0)
                    tmask = prmask[:, None] & cmask[None, :]
                    tile = tl.load(
                        A + base + prows[:, None] * N + cols[None, :],
                        mask=tmask,
                        other=0.0,
                    )
                    tile = tile - l_strip[:, None] * u_strip[None, :]
                    tl.store(
                        A + base + prows[:, None] * N + cols[None, :], tile, mask=tmask
                    )

                tl.debug_barrier()

        pcols = k0 + tl.arange(0, PANEL)
        pmask = pcols < kend
        for i0 in range(kend, N, BLOCK):
            rows = i0 + tl.arange(0, BLOCK)
            rmask = rows < N
            l_tile = tl.load(
                A + base + rows[:, None] * N + pcols[None, :],
                mask=rmask[:, None] & pmask[None, :],
                other=0.0,
            )
            for j0 in range(kend, N, BLOCK):
                cols = j0 + tl.arange(0, BLOCK)
                cmask = cols < N
                u_tile = tl.load(
                    A + base + pcols[:, None] * N + cols[None, :],
                    mask=pmask[:, None] & cmask[None, :],
                    other=0.0,
                )
                tmask = rmask[:, None] & cmask[None, :]
                tile = tl.load(
                    A + base + rows[:, None] * N + cols[None, :], mask=tmask, other=0.0
                )
                tile = tile - tl.dot(l_tile, u_tile, input_precision="ieee")
                tl.store(A + base + rows[:, None] * N + cols[None, :], tile, mask=tmask)

        tl.debug_barrier()

    # Export the factorized matrix (combined L/U, LAPACK getrf layout).
    for i0 in range(0, N, BLOCK):
        rows = i0 + tl.arange(0, BLOCK)
        for j0 in range(0, N, BLOCK):
            cols = j0 + tl.arange(0, BLOCK)
            m = (rows[:, None] < N) & (cols[None, :] < N)
            v = tl.load(A + base + rows[:, None] * N + cols[None, :], mask=m, other=0.0)
            tl.store(LU_out + base + rows[:, None] * N + cols[None, :], v, mask=m)

    det = tl.full((), 1.0, dtype=A.dtype.element_ty)
    for i0 in range(0, N, BLOCK):
        d = i0 + tl.arange(0, BLOCK)
        diag = tl.load(A + base + d * N + d, mask=d < N, other=1.0)
        det = det * tl.reduce(diag, 0, combine_fn=_reduce_mul)
    det = tl.where(swap_count % 2 == 0, det, -det)
    tl.store(out + pid, det)


# GCU400 single-panel LU-working kernel for float32 _linalg_det (n <= 128).
#
# Probe attribution on this backend (kernel-only do_bench median, alternating
# A/B runs): the serial elimination step is dominated by fixed per-op round
# trips — the 2D tile load/FMA/store triple costs ~1.9-2.5us, the pivot search
# ~0.9us, the row swap ~0.3us and both debug barriers together only ~0.25us —
# so the per-step cost is roughly flat in n and the real levers are the tile
# shape, the warp count and the number of kernels on the op path. This variant:
#   * works directly in the LU output buffer: one masked tile copy A -> LU
#     upfront replaces both the host-side scratch clone (an extra aten copy
#     kernel launch per op call, ~10-20us) and the end-of-kernel export loop,
#     and the input tensor is never written (a contiguous input is passed as
#     a plain view);
#   * narrows the update tile to [BLOCK, PANEL] with PANEL = next_pow2(n)
#     (the previous dispatch padded the panel to 128, so n <= 64 carried a
#     doubly-wide masked tile: n=64 measured 3.57 vs 4.28 us/step);
#   * runs num_warps=1, measured fastest for every BLOCK <= 128 here
#     (n=64: nw1 3.57 / nw2 3.95 / nw4 5.06 us/step; n=128: nw1 3.48 /
#     nw2 4.61 / nw4 5.02 us/step — re-scan after any tile change, the
#     optimum is coupled to the tile width).
# Elimination order, masks and arithmetic are identical to the SINGLE branch
# of ``_det_panel_kernel`` (bit-identical outputs, probe-verified), so the
# combined L/U layout and the 1-indexed LAPACK ipiv semantics carry over
# unchanged. Kernel-only medians: n=16 53us / n=32 107us / n=64 219us /
# n=128 398us, vs the retired dispatch (register / register2 / panel-128 +
# host clone) 67 / 172 / 295 / 639us.
@libentry()
@triton.jit
def _det_luwork_kernel(
    A,
    out,
    LU_out,
    pivots_out,
    N,
    PANEL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tle.program_id(0)
    base = pid * N * N

    # Materialize the working matrix in the LU output buffer: the elimination
    # below mutates LU_out in place, A stays read-only.
    rows0 = tl.arange(0, BLOCK)
    cols0 = tl.arange(0, BLOCK)
    m0 = (rows0[:, None] < N) & (cols0[None, :] < N)
    v = tl.load(A + base + rows0[:, None] * N + cols0[None, :], mask=m0, other=0.0)
    tl.store(LU_out + base + rows0[:, None] * N + cols0[None, :], v, mask=m0)

    swap_count = tl.zeros((), dtype=tl.int32)
    for k in range(0, N):
        rows = k + tl.arange(0, BLOCK)
        col = tl.load(LU_out + base + rows * N + k, mask=rows < N, other=0.0)
        abs_col = tl.abs(col)
        best_val = tl.max(abs_col, axis=0)
        best_row = tl.min(tl.where(abs_col == best_val, rows, N), axis=0)
        # An all-zero (singular) pivot column falls through to row k: the
        # masked load fills padding rows with 0, which never wins the max
        # against a nonzero candidate.
        best_row = tl.where(best_val <= 0.0, k, best_row)

        cols = tl.arange(0, BLOCK)
        cmask = cols < N
        row_k = tl.load(LU_out + base + k * N + cols, mask=cmask, other=0.0)
        row_p = tl.load(LU_out + base + best_row * N + cols, mask=cmask, other=0.0)
        tl.store(LU_out + base + k * N + cols, row_p, mask=cmask)
        tl.store(LU_out + base + best_row * N + cols, row_k, mask=cmask)
        swap_count = tl.where(best_row != k, swap_count + 1, swap_count)
        # 1-indexed pivot row for this elimination step (LAPACK ipiv convention).
        tl.store(pivots_out + pid * N + k, best_row + 1)

        tl.debug_barrier()

        pivot = tl.load(LU_out + base + k * N + k)
        safe_pivot = tl.where(pivot == 0.0, 1.0, pivot)
        rows2 = k + 1 + tl.arange(0, BLOCK)
        rmask = rows2 < N
        l_col = tl.load(LU_out + base + rows2 * N + k, mask=rmask, other=0.0)
        l_col = l_col / safe_pivot
        tl.store(LU_out + base + rows2 * N + k, l_col, mask=rmask)

        pcols = k + 1 + tl.arange(0, PANEL)
        pcmask = pcols < N
        u_row = tl.load(LU_out + base + k * N + pcols, mask=pcmask, other=0.0)
        tmask = rmask[:, None] & pcmask[None, :]
        tile = tl.load(
            LU_out + base + rows2[:, None] * N + pcols[None, :], mask=tmask, other=0.0
        )
        tile = tile - l_col[:, None] * u_row[None, :]
        tl.store(LU_out + base + rows2[:, None] * N + pcols[None, :], tile, mask=tmask)

        tl.debug_barrier()

    # The working matrix already IS the LU output: only the determinant
    # remains (product of the diagonal, sign from the swap parity).
    det = tl.full((), 1.0, dtype=LU_out.dtype.element_ty)
    for i0 in range(0, N, BLOCK):
        d = i0 + tl.arange(0, BLOCK)
        diag = tl.load(LU_out + base + d * N + d, mask=d < N, other=1.0)
        det = det * tl.reduce(diag, 0, combine_fn=_reduce_mul)
    det = tl.where(swap_count % 2 == 0, det, -det)
    tl.store(out + pid, det)


def _linalg_det(A):
    """FlagGems implementation of aten::_linalg_det.

    Computes the determinant via LU decomposition with partial pivoting, using
    the vectorized elimination kernels shared with ``linalg_det`` (single-panel
    LU-working for float32 n <= 128, blocked / panel beyond and for float64),
    and additionally exports the factorized matrix and pivot indices.

    On devices without a native fp64 datapath (gcu400), float64 inputs follow
    the torch_gcu convention: the elimination runs in float32 and the outputs
    are returned in the requested dtype.

    Args:
        A: tensor of shape (*, n, n) where * is zero or more batch dimensions.

    Returns:
        A tuple (result, LU, pivots) where:
        - result is the determinant with the same dtype as A
        - LU is the factorized matrix in the LAPACK getrf combined layout
        - pivots is the int32 pivot buffer containing 1-indexed pivot rows
    """
    logger.debug("GEMS_ENFLAME _LINALG_DET")
    if A.dtype not in (torch.float32, torch.float64):
        raise ValueError(
            f"_linalg_det only supports float32 and float64, got {A.dtype}"
        )
    if A.dim() < 2:
        raise ValueError(
            f"_linalg_det: input tensor must be at least 2D, got {A.dim()}D"
        )
    m, n = A.shape[-2], A.shape[-1]
    if m != n:
        raise ValueError(f"_linalg_det: input tensor must be square, got {m}x{n}")

    orig_dtype = A.dtype
    downcast = orig_dtype == torch.float64 and not _HAS_FP64
    if downcast:
        A = A.detach().to(torch.float32)

    batch_shape = A.shape[:-2]
    batch_count = math.prod(batch_shape)

    result = torch.empty(batch_shape, dtype=A.dtype, device=A.device)
    LU = torch.empty(A.shape, dtype=A.dtype, device=A.device)
    # pivots follows the aten schema shape (*, n); stores 1-indexed pivot rows
    pivots = torch.empty(batch_shape + (n,), dtype=torch.int32, device=A.device)

    if n == 0:
        return (
            torch.ones(batch_shape, dtype=orig_dtype, device=A.device),
            LU.to(orig_dtype),
            pivots,
        )
    if batch_count == 0:
        return result.to(orig_dtype), LU.to(orig_dtype), pivots

    is_f32 = A.dtype == torch.float32
    # GCU400 fast path (f32, n <= 128): single-panel elimination working
    # directly in the LU output buffer (see ``_det_luwork_kernel``). Kernel-only
    # medians vs the retired register/register2/panel dispatch: n=16 53 vs 67us,
    # n=32 107 vs 172us, n=64 219 vs 295us, n=128 398 vs 639us; the old panel
    # numbers additionally carried the host-side scratch clone that this path
    # removes. The kernel never writes the input tensor, so a contiguous input
    # is passed as a plain view.
    use_luwork = is_f32 and n <= 128

    out = torch.empty(batch_count, dtype=A.dtype, device=A.device)
    grid = (batch_count,)

    with torch_device_fn.device(A.device):
        if use_luwork:
            if A.is_contiguous():
                A_pass = A.reshape(batch_count, n, n)
            else:
                A_pass = A.clone(memory_format=torch.contiguous_format).reshape(
                    batch_count, n, n
                )
            # Tile shape scan (alternating A/B runs; see the kernel comment):
            # the panel width tracks next_pow2(n), and BLOCK keeps a 64-wide
            # row vector down to n = 17 -- BLOCK=64/PANEL=32 beats BLOCK=32 at
            # n in (16, 32] (107 vs 111us) -- with num_warps=1 throughout.
            if n <= 16:
                lu_block = 16
            elif n <= 32:
                lu_block = 64
            else:
                lu_block = min(128, triton.next_power_of_2(n))
            lu_panel = max(16, triton.next_power_of_2(n))
            _det_luwork_kernel[grid](
                A_pass,
                out,
                LU,
                pivots,
                n,
                PANEL=lu_panel,
                BLOCK=lu_block,
                num_warps=1,
            )
        else:
            # Remaining paths (f32 n > 128 multi-panel, native f64): they
            # eliminate in place in a private scratch copy of the input.
            A_work = A.clone(memory_format=torch.contiguous_format).reshape(
                batch_count, n, n
            )
            # LU shape tuning is dtype dependent (see the det-only kernels
            # this follows): float64 wants a narrower panel than float32
            # because each element is twice as wide and the trailing update is
            # arithmetic-intensity bound. The metax float64 special case from
            # upstream is preserved.
            is_f64 = A.dtype == torch.float64
            use_panel = n > 32 and not (
                is_f64 and runtime_device.vendor_name == "metax"
            )
            if is_f64:
                # f64 panels are always multi-block (use_panel requires
                # n > 32 > BLOCK).
                panel_kwargs = {"PANEL": 16, "BLOCK": 32, "SINGLE": 0, "num_warps": 4}
            else:
                # GCU400: wider chunks amortize the per-op round-trip tax of
                # the serial panel phase (measured on L600, same tuning as
                # linalg_det): PANEL=128 with BLOCK equal to the matrix size
                # (<= 256), n <= BLOCK flattens every inner chunk loop.
                block_val = min(256, triton.next_power_of_2(max(n, 64)))
                panel_kwargs = {
                    "PANEL": 128,
                    "BLOCK": block_val,
                    "SINGLE": 1 if n <= block_val else 0,
                    "num_warps": 2 if block_val <= 64 else 4,
                }
            if use_panel:
                _det_panel_kernel[grid](A_work, out, LU, pivots, n, **panel_kwargs)
            else:
                block = min(64, max(8, triton.next_power_of_2(n)))
                num_warps = min(4, max(1, (block * block * 8) // 4096))
                _det_blocked_kernel[grid](
                    A_work, out, LU, pivots, n, BLOCK=block, num_warps=num_warps
                )

    return (
        out.reshape(batch_shape).to(orig_dtype),
        LU.reshape(A.shape).to(orig_dtype),
        pivots.reshape(batch_shape + (n,)),
    )


# ---------------------------------------------------------------------------
# Generic-path module alias (vendor injection).
#
# SpecOpRegistrar imports this file as ``gcu400.ops._linalg_det`` and wires
# ``flag_gems._linalg_det`` / the aten-registered ``torch.ops.aten._linalg_det``
# to the implementation above, but the KernelGen accuracy test and the perf
# benchmark resolve the op through the generic package attribute
# (``flag_gems.ops._linalg_det(...)``), which keeps pointing at the unused
# generic implementation -- a float64 input then reaches the fp64-less Triton
# kernels and fails to compile, even though the dispatcher path already runs
# this backend copy.  Publish this module under the generic path (the same
# pattern as ``quantized_max_pool2d`` / ``sparse_coo_tensor``) and rebind the
# package attribute so the module under test IS the implementation that runs.
# ---------------------------------------------------------------------------
sys.modules["flag_gems.ops._linalg_det"] = sys.modules[__name__]
flag_gems.ops._linalg_det = _linalg_det
