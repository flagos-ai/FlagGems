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
# Element width of the load tile along the reduced dimension.  The previous
# value of 128 issues 512 B loads and saturates near 0.5 TB/s; 1024 keeps the
# tile inside the DSM budget (2048 still fits, 4096 exceeds it) and reaches
# ~1.7 TB/s.  It also accumulates more accurately, because fewer fp32 partial
# sums are chained into the running total.
_BLOCK_N = 1024
# Shortest reduced-dimension chunk a split launch will create.  Below two
# BLOCK_N iterations the extra launch costs more than the added parallelism
# returns.
_MIN_CHUNK_N = 2048
# Shortest reduced dimension worth splitting at all.  A launch costs ~33 us
# here regardless of size, so the split only pays once the serial walk of the
# reduced dimension would dominate it: measured on gcu400 the crossover sits
# between N = 8192 (plain wins by 1.8x) and N = 65536 (split wins by 2.7x).
_SPLIT_MIN_N = 32768


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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Trapezoid integration with constant spacing dx.

    One program per BLOCK_M rows, walking the reduced dimension serially.  Used
    when there are already enough row blocks to fill the device.
    """
    # float32 accumulation only: gcu400 reports support_fp64 == False, so the
    # fp64 accumulator the generic implementation selects is unavailable.
    acc_dtype = tl.float32
    pid_m = tle.program_id(0)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = m_offsets < M

    acc = tl.zeros((BLOCK_M,), dtype=acc_dtype)

    for n_start in range(0, N - 1, BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        mask_n = n_offsets < N - 1

        y_offsets = m_offsets[:, None] * M_STRIDE_Y + n_offsets[None, :]
        mask = mask_m[:, None] & mask_n[None, :]

        y_i = tl.load(y_ptr + y_offsets, mask=mask, other=0.0).to(acc_dtype)
        y_next = tl.load(y_ptr + y_offsets + 1, mask=mask, other=0.0).to(acc_dtype)

        # Trapezoidal rule: (y[i] + y[i+1]) / 2
        trapezoid_sum = (y_i + y_next) * 0.5

        acc += tl.sum(trapezoid_sum, axis=1)

    result = acc * dx
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

    Interval ``i`` of the trapezoidal rule spans ``y[i]`` and ``y[i + 1]``, so
    there are ``N - 1`` of them.  Program ``(c, r)`` sums the disjoint range
    ``[c * chunk, min((c + 1) * chunk, N - 1))`` for the rows of block ``r`` and
    writes one partial per row into ``part[c, m]``.
    """
    pid_c = tle.program_id(0)
    pid_m = tle.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = m_offsets < M

    lo = pid_c * chunk
    hi = tl.minimum(lo + chunk, N - 1)

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for n_start in range(lo, hi, BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        mask_n = n_offsets < hi

        y_offsets = m_offsets[:, None] * N + n_offsets[None, :]
        mask = mask_m[:, None] & mask_n[None, :]

        y_i = tl.load(y_ptr + y_offsets, mask=mask, other=0.0).to(tl.float32)
        y_next = tl.load(y_ptr + y_offsets + 1, mask=mask, other=0.0).to(tl.float32)

        acc += tl.sum((y_i + y_next) * 0.5, axis=1)

    tl.store(part_ptr + pid_c * M + m_offsets, acc, mask=mask_m)


@libentry()
@triton.jit
def trapezoid_dx_merge_kernel(
    part_ptr,
    out_ptr,
    M,
    n_chunk,
    dx,
    M_STRIDE_OUT,
    BLOCK_M: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    """Second stage: sum the ``n_chunk`` partials of each row and scale by dx."""
    pid_m = tle.program_id(0)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = m_offsets < M

    p_offsets = tl.arange(0, BLOCK_P)
    mask = (p_offsets[:, None] < n_chunk) & mask_m[None, :]
    part = tl.load(
        part_ptr + p_offsets[:, None] * M + m_offsets[None, :], mask=mask, other=0.0
    )

    result = tl.sum(part, axis=0) * dx
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
    rows = triton.cdiv(M, block_m)

    # Aim the chunk grid at one program per compute unit, but never make a
    # chunk shorter than _MIN_CHUNK_N intervals.
    chunks = 0
    if N >= _SPLIT_MIN_N:
        chunks = min(
            triton.cdiv(N - 1, _MIN_CHUNK_N),
            max(1, triton.cdiv(_MAX_GRID, rows)),
        )

    if chunks <= 1:
        trapezoid_kernel_dx[(rows,)](
            y, out, M, N, dx, N, 1, BLOCK_M=block_m, BLOCK_N=_BLOCK_N
        )
        return

    chunk = triton.cdiv(N - 1, chunks)
    # [chunks, M] float32: partial sums are accumulated in fp32 whatever the
    # input dtype, so the merge reads and writes one dtype.
    part = torch.empty((chunks, M), dtype=torch.float32, device=y.device)
    trapezoid_dx_partial_kernel[(chunks, rows)](
        y, part, M, N, chunk, BLOCK_M=block_m, BLOCK_N=_BLOCK_N
    )
    trapezoid_dx_merge_kernel[(rows,)](
        part,
        out,
        M,
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
        logger.debug("GEMS_ENFLAME_GCU400 TRAPEZOID_DX")

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
    logger.debug("GEMS_ENFLAME_GCU400 TRAPZ")
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
