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

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

# Number of compute units on gcu400.  A launch with fewer programs than this
# leaves the device partly idle.  The original single-kernel path used one
# program per 32 rows and walked the whole reduced dimension inside it, so a
# 1-D input (one row, hence exactly one program) ran at ~1 GB/s: 11.5 ms for
# 2^20 elements against 71 us for the native path.
_MAX_GRID = 48
# Rows per program.  The [BLOCK_M, BLOCK_N] tile is held in registers/DSM, so
# the row block caps here.
_BLOCK_M = 32
# Element budget for the [BLOCK_M, BLOCK_N] load tile; it lives in
# registers/DSM and 65536 elements exceeds that budget.  BLOCK_N takes whatever
# is left after the row block takes its share, so a single-row (1-D) launch can
# use a much wider column tile than a 32-row one.
#   BLOCK_M = 32 -> BLOCK_N = 1024 is optimal (measured on (4096,4096),
#                  (1024,1024,1024) and (10000,65536); 512 loses 1.4~1.6x)
#   BLOCK_M = 1  -> BLOCK_N = 16384 is optimal; on a 2^28-element 1-D input this
#                  took 13.9 ms down to 1.28 ms (10.8x) because each iteration
#                  then moves a tile large enough to hide the load latency
#                  behind the serial accumulation.
_TILE_ELEMENTS = 32768
_BLOCK_N_MAX = 16384
# Shortest reduced-dimension chunk a split launch will create.  Below two
# BLOCK_N iterations the extra launch costs more than the added parallelism
# returns.
_MIN_CHUNK_N = 2048
# Shortest reduced dimension worth splitting at all.  A launch costs ~33 us
# here regardless of size, so the split only pays once the serial walk of the
# reduced dimension would dominate it: measured on gcu400 the crossover sits
# between N = 8192 (plain wins by 1.8x) and N = 65536 (split wins by 2.7x).
_SPLIT_MIN_N = 32768

# Hardware grid limit on gcu400: grid.x must fit in 16 bits.
_GRID_X_LIMIT = 65535


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("trapezoid_kernel"),
    key=["M", "N"],
)
@triton.jit
def trapezoid_kernel_dx(
    y_ptr,
    out_ptr,
    M,
    N,
    dx,
    M_STRIDE_Y,
    M_STRIDE_OUT,
    ROW_BLOCKS,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Trapezoid integration with constant spacing dx.

    One program per BLOCK_M rows, walking the reduced dimension serially.  Used
    when there are already enough row blocks to fill the device.

    ``ROW_BLOCKS`` is the width of the first grid dimension; the row-block index
    is recovered as ``pid0 + pid1 * ROW_BLOCKS`` so that a row count above the
    hardware grid.x limit can be spread over two dimensions.
    """
    # float32 accumulation only: gcu400 reports support_fp64 == False, so the
    # fp64 accumulator the generic implementation selects is unavailable.
    acc_dtype = tl.float32
    pid_m = tle.program_id(0) + tle.program_id(1) * ROW_BLOCKS

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = m_offsets < M

    acc = tl.zeros((BLOCK_M,), dtype=acc_dtype)

    # Single-pass reformulation of the trapezoidal rule:
    #   dx * (sum(y) - (y[0] + y[N-1]) / 2)
    # The literal form sums (y[i] + y[i+1]) / 2, which loads every element
    # twice, so this variant halves the read traffic.  Only the two endpoints
    # need a separate scalar load per row.
    for n_start in range(0, N, BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        mask_n = n_offsets < N

        y_offsets = m_offsets[:, None] * M_STRIDE_Y + n_offsets[None, :]
        mask = mask_m[:, None] & mask_n[None, :]

        y_i = tl.load(y_ptr + y_offsets, mask=mask, other=0.0).to(acc_dtype)
        acc += tl.sum(y_i, axis=1)

    row_base = m_offsets * M_STRIDE_Y
    y_first = tl.load(y_ptr + row_base, mask=mask_m, other=0.0).to(acc_dtype)
    y_last = tl.load(y_ptr + row_base + (N - 1), mask=mask_m, other=0.0).to(acc_dtype)

    result = (acc - (y_first + y_last) * 0.5) * dx
    out_offsets = m_offsets * M_STRIDE_OUT
    tl.store(out_ptr + out_offsets, result, mask=mask_m)


@libentry()
@triton.jit
def trapezoid_dx_partial_kernel(
    y_ptr,
    part_ptr,
    M,
    N,
    chunk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """First stage of the split reduction: one (chunk, row block) per program.

    Program ``(c, r)`` sums the plain elements of the disjoint range
    ``[c * chunk, min((c + 1) * chunk, N))`` for the rows of block ``r`` and
    writes one partial per row into ``part[c, m]``.  The endpoints of the
    trapezoidal rule are applied once by the merge stage, so this stage only
    needs a plain (single-load) element sum.
    """
    pid_c = tle.program_id(0)
    pid_m = tle.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = m_offsets < M

    lo = pid_c * chunk
    hi = tl.minimum(lo + chunk, N)

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for n_start in range(lo, hi, BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        mask_n = n_offsets < hi

        y_offsets = m_offsets[:, None] * N + n_offsets[None, :]
        mask = mask_m[:, None] & mask_n[None, :]

        y_i = tl.load(y_ptr + y_offsets, mask=mask, other=0.0).to(tl.float32)

        acc += tl.sum(y_i, axis=1)

    tl.store(part_ptr + pid_c * M + m_offsets, acc, mask=mask_m)


@libentry()
@triton.jit
def trapezoid_dx_merge_kernel(
    part_ptr,
    y_ptr,
    out_ptr,
    M,
    N,
    n_chunk,
    dx,
    M_STRIDE_OUT,
    BLOCK_M: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    """Second stage: sum the ``n_chunk`` partials of each row, apply the
    trapezoidal endpoint correction and scale by dx."""
    pid_m = tle.program_id(0)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = m_offsets < M

    p_offsets = tl.arange(0, BLOCK_P)
    mask = (p_offsets[:, None] < n_chunk) & mask_m[None, :]
    part = tl.load(
        part_ptr + p_offsets[:, None] * M + m_offsets[None, :], mask=mask, other=0.0
    )

    row_base = m_offsets * N
    y_first = tl.load(y_ptr + row_base, mask=mask_m, other=0.0)
    y_last = tl.load(y_ptr + row_base + (N - 1), mask=mask_m, other=0.0)

    total = tl.sum(part, axis=0)
    result = (total - (y_first + y_last) * 0.5) * dx
    tl.store(out_ptr + m_offsets * M_STRIDE_OUT, result, mask=mask_m)


def _normalize_dim(dim, ndim):
    if ndim == 0:
        raise IndexError(f"Dimension specified as {dim} but tensor has no dimensions")
    if not (-ndim <= dim < ndim):
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-ndim}, {ndim - 1}], but got {dim})"
        )
    return dim % ndim


def _launch(y, out, M, N, dx):
    """Launch the trapezoid kernel, splitting the reduced dimension if needed.

    ``y`` is the dim-compressed, contiguous ``[M, N]`` view and ``out`` the
    ``[M, 1]`` float32 destination.

    The plain kernel parallelises over rows only, so ``cdiv(M, BLOCK_M)``
    programs is all the device ever sees: a 1-D input runs a single program and
    leaves 47 of the 48 compute units idle.  When the reduced dimension is long
    enough for that serial walk to dominate the launch cost, it is cut into
    chunks, each program reducing its own disjoint interval range before a
    second kernel merges the partials.  The merge is a plain sum over at most
    ``_MAX_GRID`` terms, so it costs one short launch.

    Whether to split is decided by ``N`` alone, not by the row count: a long
    reduced dimension makes even the many-row case slower than the split (at
    N = 2^20 the plain kernel needs 2.1 ms for 8 rows where the split needs
    83 us), and a short one leaves the split paying for a launch it cannot
    amortise.  The row blocks still run concurrently alongside the chunks, so
    the two grids compose rather than compete.

    The row block shrinks to a power of two when ``M`` is small: a partial row
    tile would otherwise leave most of the ``[BLOCK_M, BLOCK_N]`` accumulator
    lanes idle.
    """
    block_m = min(_BLOCK_M, triton.next_power_of_2(M))
    # Rows and columns share one element budget; a narrow row block therefore
    # buys a wider column tile, and a column tile never exceeds the reduced
    # dimension (a wider one would only mask lanes away: N = 64 with BLOCK_N =
    # 1024 spends 94% of the lanes on padding).
    # The 128 lower bound matters: shrinking the column tile all the way to
    # next_pow2(N) for very short rows (N = 64) measured slower than a padded
    # tile, because the extra iterations cost more than the masked lanes.
    block_n = max(
        128,
        min(_BLOCK_N_MAX, _TILE_ELEMENTS // block_m, triton.next_power_of_2(N)),
    )
    # A narrow column tile (short reduced dimension) leaves most of the element
    # budget idle, and with many rows every program would then reduce a tiny
    # tile: the launch/retire cost of the programs, not the memory traffic,
    # becomes the bottleneck (M = 6.5M rows with N = 100 spends 43 ms in 204800
    # tiny programs).  Spend the leftover budget on more rows per program
    # instead, which cuts the program count by the same factor.
    if block_n < _TILE_ELEMENTS // block_m:
        block_m = min(triton.next_power_of_2(M), _TILE_ELEMENTS // block_n)
    rows = triton.cdiv(M, block_m)

    # Aim the chunk grid at one program per compute unit, but never make a
    # chunk shorter than _MIN_CHUNK_N intervals.
    chunks = 0
    if N >= _SPLIT_MIN_N:
        chunks = min(
            triton.cdiv(N, _MIN_CHUNK_N),
            max(1, triton.cdiv(_MAX_GRID, rows)),
        )

    if chunks <= 1:
        # gcu400 rejects grid.x > 65535 (a 1-D 2^28 input asks for 8M row
        # blocks), so fold the row-block index into two grid dimensions once it
        # exceeds the limit. The 1-D case keeps grid = (rows, 1).
        row_blocks_x = min(rows, _GRID_X_LIMIT)
        row_blocks_y = triton.cdiv(rows, _GRID_X_LIMIT)
        trapezoid_kernel_dx[(row_blocks_x, row_blocks_y)](
            y,
            out,
            M,
            N,
            dx,
            N,
            1,
            row_blocks_x,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
        )
        return

    chunk = triton.cdiv(N, chunks)
    # [chunks, M] float32: partial sums are accumulated in fp32 whatever the
    # input dtype, so the merge reads and writes one dtype.
    part = torch.empty((chunks, M), dtype=torch.float32, device=y.device)
    trapezoid_dx_partial_kernel[(chunks, rows)](
        y, part, M, N, chunk, BLOCK_M=block_m, BLOCK_N=block_n
    )
    trapezoid_dx_merge_kernel[(rows,)](
        part,
        y,
        out,
        M,
        N,
        chunks,
        dx,
        1,
        BLOCK_M=block_m,
        BLOCK_P=triton.next_power_of_2(chunks),
    )


class TrapzOp(torch.autograd.Function):
    """Custom autograd function for the trapezoidal rule.

    The forward pass runs a Triton kernel; the backward pass multiplies the
    upstream gradient by the trapezoidal weights ``dx * [0.5, 1, ..., 1, 0.5]``
    broadcast along the reduced dimension.
    """

    @staticmethod
    def forward(ctx, y, dx, dim):
        logger.debug("GEMS_ENFLAME TRAPEZOID_DX")

        dim = _normalize_dim(dim, y.ndim)
        shape = y.shape
        N = shape[dim]
        out_shape = shape[:dim] + shape[dim + 1 :]

        ctx.dx = dx
        ctx.dim = dim
        ctx.N = N
        ctx.y_shape = shape
        ctx.y_dtype = y.dtype

        # Empty tensors or a degenerate reduction dimension (N <= 1) integrate
        # to zero; the reduction dimension is always removed from the shape.
        if y.numel() == 0 or N <= 1:
            return torch.zeros(out_shape, dtype=y.dtype, device=y.device)

        y_compressed = dim_compress(y, dim)
        M = y_compressed.numel() // N

        # float32 accumulation only: the device reports support_fp64 == False,
        # so the fp64 accumulator the generic implementation selects does not
        # exist here.  The result is cast back to the input dtype below.
        compute_dtype = torch.float32
        out_compressed_shape = list(y_compressed.shape)
        out_compressed_shape[-1] = 1
        output = torch.empty(out_compressed_shape, dtype=compute_dtype, device=y.device)

        with torch_device_fn.device(y.device):
            _launch(y_compressed, output, M, N, dx)

        result = output.reshape(out_shape)
        if compute_dtype != y.dtype:
            result = result.to(y.dtype)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        dx = ctx.dx
        dim = ctx.dim
        N = ctx.N
        y_shape = ctx.y_shape
        y_dtype = ctx.y_dtype

        if N <= 1:
            return (
                torch.zeros(y_shape, dtype=y_dtype, device=grad_output.device),
                None,
                None,
            )

        # float32 only, for the same reason as the forward pass.
        acc_dtype = torch.float32
        w = torch.ones(N, dtype=acc_dtype, device=grad_output.device)
        w[0] = 0.5
        w[-1] = 0.5
        w_shape = [1] * len(y_shape)
        w_shape[dim] = N
        w = w.view(w_shape)

        grad_y = grad_output.to(acc_dtype).unsqueeze(dim) * w * dx
        return grad_y.to(y_dtype), None, None


def _normalize_dx(dx):
    # Mirror ATen: dx must be a real number. bool and complex scalars (including
    # 0-d tensors, which ATen inspects before conversion) are rejected, while a
    # plain python float()/item() would silently coerce True -> 1.0 and drop a
    # complex imaginary part.
    dx_val = dx.item() if isinstance(dx, torch.Tensor) else dx
    if isinstance(dx_val, (bool, complex)) or not isinstance(dx_val, (int, float)):
        raise RuntimeError("trapezoid: Currently, we only support dx as a real number.")
    return float(dx_val)


def trapz(y, dx=1.0, dim=-1):
    """Compute the trapezoidal rule along a dimension with constant spacing.

    Args:
        y: Input tensor.
        dx: Constant spacing between samples.
        dim: Dimension along which to integrate.

    Returns:
        The integral of ``y`` along ``dim``, with the reduced dimension removed.
    """
    # Match ATen's ordering: the dimension is validated first, then an *empty*
    # reduction dimension integrates to zero of the input dtype and returns
    # before the bool/complex/dx checks run at all. A degenerate N == 1 still
    # goes through those checks (ATen validates dx and rejects bool y for it).
    logger.debug("GEMS_ENFLAME TRAPZ")
    dim = _normalize_dim(dim, y.ndim)
    if y.shape[dim] == 0:
        out_shape = y.shape[:dim] + y.shape[dim + 1 :]
        return torch.zeros(out_shape, dtype=y.dtype, device=y.device)

    if y.dtype == torch.bool:
        raise RuntimeError(
            "trapezoid: received a bool input for `y`, but bool is not supported"
        )

    # Triton kernels are real-valued only, so complex inputs are unsupported.
    # Reject them explicitly instead of silently falling back to native ATen.
    if y.is_complex():
        raise RuntimeError(
            "trapz: complex inputs are not supported by the FlagGems Triton kernel"
        )

    dx = _normalize_dx(dx)

    # Integer inputs are promoted to float32, matching PyTorch.
    if not y.is_floating_point():
        y = y.to(torch.float32)

    return TrapzOp.apply(y, dx, dim)
