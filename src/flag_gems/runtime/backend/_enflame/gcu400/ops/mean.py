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

from flag_gems import runtime  # noqa: F401
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry

logger = logging.getLogger(__name__)

# DTE hardware register bitwidth limit: each tile dimension must fit in 24 bits.
DTE_DIM_MAX = (1 << 24) - 1
# GCU400 has 48 compute units: a fixed grid cap plus an in-kernel strided loop keeps the SPs saturated.
MAX_GRID = 48
# Per-tile byte budget and column-width cap for row reduction (last dimension). Measured on (4096,4096), BLOCK_N=8192
# is about 2x slower than 4096 (an overly wide contiguous dimension is counterproductive); 4096 and 2048 are on par.
ROW_TILE_BYTES = 131072
ROW_MAX_WIDTH = 4096
# Contiguous-dimension width cap for non-last-dimension reduction (task kernel): the axis reduction tree has a known
# slow path for contiguous dimension <=256 (same pathology as sum/prod), so pin it at 512; larger widths bring no
# benefit.
TASK_MAX_WIDTH = 512


@libentry()
@triton.jit
def mean_kernel_1(inp, mid, M, BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr = 1):
    pid = tl.program_id(0)
    num_prog = tl.num_programs(0)
    num_tile = (M + BLOCK_SIZE - 1) // BLOCK_SIZE
    for tile_id in tl.range(pid, num_tile, num_prog, num_stages=num_stages):
        offset = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        inp_ptrs = inp + offset
        mask = offset < M
        inp_val = tl.load(inp_ptrs, mask=mask, other=0.0).to(tl.float32)
        sum_val = tl.sum(inp_val, axis=0)
        mid_ptr = mid + tile_id
        tl.store(mid_ptr, sum_val)


@libentry()
@triton.jit
def mean_kernel_2(mid, out, M, MID_SIZE, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < MID_SIZE
    mid_val = tl.load(mid_ptrs, mask=mask, other=0.0)
    sum_val = tl.sum(mid_val, axis=0) / M
    tl.store(out, sum_val)


def _mean_dtype(inp, dtype):
    """Infer/validate the output dtype with CPU aten semantics.

    Mirrors the generic ``_mean_dtype``: mean over a non-floating input without
    an explicit floating ``dtype`` (and vice versa a non-floating explicit
    ``dtype``) must raise RuntimeError instead of computing garbage.
    """
    if dtype is None:
        dtype = inp.dtype
        if not (dtype.is_floating_point or dtype.is_complex):
            raise RuntimeError(
                "mean(): could not infer output dtype. Input dtype must be either "
                f"a floating point or complex dtype. Got: {dtype}"
            )
    elif not (dtype.is_floating_point or dtype.is_complex):
        raise RuntimeError(
            "mean(): could not infer output dtype. Optional dtype must be either "
            f"a floating point or complex dtype. Got: {dtype}"
        )
    return dtype


def mean(inp, *, dtype=None):
    logger.debug("GEMS_ENFLAME MEAN")
    if not inp.is_contiguous():
        inp = inp.contiguous()
    M = inp.numel()
    dtype = _mean_dtype(inp, dtype)
    # Full-mean of an empty tensor is NaN in aten; short-circuit to avoid launching a 0-element kernel.
    if M == 0:
        return torch.full([], float("nan"), dtype=dtype, device=inp.device)

    # 1D streaming partials + merge, same recipe as sum's global path (measured 2026-11-17):
    # grid 48 + nw4; tile of 131072 elements for 2-byte types, 65536 for the rest. Compared with the old parameters
    # (grid 24 / nw1 / tile<=64K), 1Mi is ~10% faster and 256Mi ~44% faster.
    # num_stages measured no different from 1 in the deep-loop regime (256Mi, 40+ tiles per program).
    element_size = inp.element_size()
    block_size = min(
        triton.next_power_of_2(M),
        131072 if element_size == 2 else 65536,
    )
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)
    # Cross-block partial sums accumulate in fp32 for 16-bit inputs: an fp16
    # partial sum overflows to inf once the running sum passes 65504
    # (measured: 96000 fp16 values around 3.0 -> torch.mean returns inf).
    acc_dtype = dtype if dtype in (torch.float32, torch.float64) else torch.float32
    mid = torch.empty((mid_size,), dtype=acc_dtype, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        mean_kernel_1[(min(mid_size, MAX_GRID), 1, 1)](
            inp, mid, M, block_size, num_warps=4
        )
        mean_kernel_2[(1, 1, 1)](mid, out, M, mid_size, block_mid, num_warps=1)
    return out


@libentry()
@triton.jit
def mean_kernel_dim_low(inp, Mean, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Last-dim reduction: each tile covers BLOCK_M rows and accumulates over BLOCK_N column blocks.

    Each block runs an axis=1 tree reduction immediately after loading, so the only live register
    across column blocks is acc[BLOCK_M] (the old version kept the whole (BLOCK_M, BLOCK_N) tile
    live, which was a main cause of bandwidth loss on large shapes).
    """
    pid = tl.program_id(0)
    step = tl.num_programs(0)
    num_tile = (M + BLOCK_M - 1) // BLOCK_M
    for tile_id in tl.range(pid, num_tile, step):
        m_offset = tile_id * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = m_offset < M
        n_offset = tl.arange(0, BLOCK_N)
        acc = tl.zeros([BLOCK_M], dtype=tl.float32)
        for col_start in tl.range(0, N, BLOCK_N):
            cols = col_start + n_offset
            mask = row_mask[:, None] & (cols[None, :] < N)
            inp_ptrs = inp + m_offset[:, None] * N + cols[None, :]
            x = tl.load(inp_ptrs, mask=mask, other=0.0).to(tl.float32)
            acc += tl.sum(x, axis=1)
        tl.store(Mean + m_offset, acc / N, row_mask)


@libentry()
@triton.jit
def mean_kernel_dim_mid(
    inpIn,
    out_value_in,
    B,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Reduces the M dim of (B, M, N) (leading-dim when B == 1; replaces the old dim_high kernel).

    A task is (batch, column tile) and the grid strides over B * cdiv(N, BLOCK_N) tasks; each task
    keeps acc[BLOCK_N] live, splits M into BLOCK_M chunks and tree-reduces along axis=0 per chunk.
    The old version pinned grid to 24 and looped serially over batch; on (1024,1024,512) 16-bit it
    814 -> 441 us (-46%).
    """
    pid = tl.program_id(0)
    step = tl.num_programs(0)
    num_k_tile = (N + BLOCK_N - 1) // BLOCK_N
    num_task = B * num_k_tile
    for task_id in tl.range(pid, num_task, step):
        b_index = task_id // num_k_tile
        k_tile = task_id % num_k_tile
        n_offset = k_tile * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = n_offset < N
        base = b_index * M * N
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        for i_start in tl.range(0, M, BLOCK_M):
            m_offset = i_start + tl.arange(0, BLOCK_M)
            mask = (m_offset[:, None] < M) & n_mask[None, :]
            inp_ptrs = inpIn + base + m_offset[:, None] * N + n_offset[None, :]
            x = tl.load(inp_ptrs, mask=mask, other=0.0).to(tl.float32)
            acc += tl.sum(x, axis=0)
        out_ptrs = out_value_in + b_index * N + n_offset
        tl.store(out_ptrs, acc / M, n_mask)


def _row_params(inp, M, N):
    """Row-reduction hyperparameters: column width pinned to min(np2(N), 4096),
    rows from the byte budget and 48-way spreading.

    Measured (2026-11-17, official do_bench median w1000/r100):
    (4096,4096) fp16 58.9 -> 22.4us, fp32 66.0 -> 34.7us;
    (1024,1024) fp16 9.2 -> 6.3us; (64,64) sits on the same ~4.8us launch floor as the old version.
    """
    element_size = inp.element_size()
    block_n = min(triton.next_power_of_2(N), ROW_MAX_WIDTH)
    tile_rows = max(1024, ROW_TILE_BYTES // element_size) // block_n
    batch_rows = triton.next_power_of_2(triton.cdiv(M, MAX_GRID))
    block_m = max(1, min(tile_rows, batch_rows))
    grid = min(MAX_GRID, triton.cdiv(M, block_m))
    # Tiny tiles (lane count <=512) are faster with a single warp (same criterion as sum).
    num_warps = 1 if block_m * block_n <= 512 else 2
    return block_m, block_n, grid, num_warps


def _task_params(inp, B, M, N):
    """Non-last-dim reduction hyperparameters: contiguous dim pinned to min(np2(N), 512), nw tiered by dtype.

    Measured (2026-11-17): 16-bit br128/nw4 is best (br64 regresses 56%); fp32 br256/nw4 is best;
    (64,512,512) fp32 35.6us, (1024,1024,512) fp32 761us.
    DSM: an fp32 256x512 tile is 512KiB < the 917504 cap (512x512 goes OOR, do not enlarge).
    """
    element_size = inp.element_size()
    block_n = min(triton.next_power_of_2(N), TASK_MAX_WIDTH)
    block_m_cap = 256 if element_size >= 4 else 128
    block_m = max(1, min(triton.next_power_of_2(M), block_m_cap))
    num_warps = 4 if block_n >= 512 else 1
    grid = min(MAX_GRID, B * triton.cdiv(N, block_n))
    return block_m, block_n, grid, num_warps


def _launch_kernel_dim_low(inp, out, M, N):
    """Reduce along rows (last-dim reduction): each row has N elements, M rows."""
    block_m, block_n, grid, num_warps = _row_params(inp, M, N)
    with torch_device_fn.device(inp.device):
        mean_kernel_dim_low[(grid,)](
            inp, out, M, N, BLOCK_M=block_m, BLOCK_N=block_n, num_warps=num_warps
        )


def _launch_kernel_dim_high(inp, out, M, N):
    """Reduce along columns (first-dim reduction): M rows reduced to 1, N columns.

    Reuses the (B, M, N) task kernel with B == 1: a task is then a column tile of N.
    """
    block_m, block_n, grid, num_warps = _task_params(inp, 1, M, N)
    with torch_device_fn.device(inp.device):
        mean_kernel_dim_mid[(grid,)](
            inp, out, 1, M, N, BLOCK_M=block_m, BLOCK_N=block_n, num_warps=num_warps
        )


def _launch_kernel_dim_mid(inp, out, B, M, N):
    """Reduce middle dim: B batches, M reduction length, N columns per batch."""
    block_m, block_n, grid, num_warps = _task_params(inp, B, M, N)
    with torch_device_fn.device(inp.device):
        mean_kernel_dim_mid[(grid,)](
            inp, out, B, M, N, BLOCK_M=block_m, BLOCK_N=block_n, num_warps=num_warps
        )


def _safe_launch_dim_low(inp, out, M, N, device):
    """Launch mean_kernel_dim_low ensuring N <= DTE_DIM_MAX.

    When N exceeds the DTE limit, split N = N_outer * N_inner (both <= DTE_DIM_MAX)
    and do a two-pass mean reduction.  This works because:
        mean(x_flat) = mean_j( mean_i( x[j, i] ) )
    when all inner chunks have equal size (guaranteed since N_inner divides N).

    Pass 1: reshape (M, N) -> (M * N_outer, N_inner), reduce last dim -> (M * N_outer,)
    Pass 2: reshape to (M, N_outer), reduce last dim -> (M,)
    """
    if N <= DTE_DIM_MAX:
        _launch_kernel_dim_low(inp, out, M, N)
        return

    N_inner = _largest_divisor_le(N, DTE_DIM_MAX)
    N_outer = N // N_inner

    # Pass 1: reduce N_inner (innermost chunk)
    mid = torch.empty(M * N_outer, dtype=inp.dtype, device=device)
    _launch_kernel_dim_low(inp, mid, M * N_outer, N_inner)

    # Pass 2: reduce N_outer — recurse in case N_outer > DTE_DIM_MAX
    _safe_launch_dim_low(mid, out, M, N_outer, device)


def _safe_launch_dim_high(inp, out, M, N, device):
    """Launch first-dim reduction ensuring N <= DTE_DIM_MAX.

    The kernel accesses memory as m_offset * N + n_offset, so N is the inner
    stride dimension seen by DTE and must not exceed DTE_DIM_MAX.

    When N > DTE_DIM_MAX, transpose to (N, M), reduce the last dim M using
    mean_kernel_dim_low (which has better performance for row reductions),
    then the result is already the column means.  If M also exceeds the limit,
    fall back to chunking along N.
    """
    if N <= DTE_DIM_MAX:
        _launch_kernel_dim_high(inp, out, M, N)
        return

    if M <= DTE_DIM_MAX:
        # Transpose (M, N) -> (N, M): now reduce last dim M (which is small).
        inp_t = inp.reshape(M, N).t().contiguous()
        out_flat = out.reshape(N)
        _safe_launch_dim_low(inp_t, out_flat, N, M, device)
    else:
        # Both M and N exceed the limit — chunk along N
        inp_2d = inp.reshape(M, N)
        out_flat = out.reshape(N)
        N_inner = _largest_divisor_le(N, DTE_DIM_MAX)
        for start in range(0, N, N_inner):
            end = min(start + N_inner, N)
            _launch_kernel_dim_high(
                inp_2d[:, start:end].contiguous(),
                out_flat[start:end],
                M,
                end - start,
            )


def _safe_launch_dim_mid(inp, out, B, M, N, device):
    """Launch mean_kernel_dim_mid ensuring N <= DTE_DIM_MAX.

    The kernel accesses inp as (B, M, N) with offset = b*M*N + m*N + n.
    DTE sees N as the inner stride and it must fit in 24 bits.

    When N > DTE_DIM_MAX, we split N = N_outer * N_inner where N_inner <=
    DTE_DIM_MAX.  Then reshape+permute the input so the kernel sees
    (B * N_outer) batches of (M, N_inner).

    Special case: when inp.numel() != B*M*N (e.g. the caller passed N as the
    total element count instead of the true column count), we detect and
    reroute to _safe_launch_dim_low which works on the actual flat layout.
    """
    if N <= DTE_DIM_MAX:
        _launch_kernel_dim_mid(inp, out, B, M, N)
        return

    total = inp.numel()
    expected = B * M * N

    if total != expected:
        # The caller's (B, M, N) doesn't match the physical element count.
        # Compute the real per-row column count from the physical layout and
        # reroute to dim_low: (num_rows, cols_per_row) -> reduce last dim.
        num_rows = total // M if M > 0 else total
        cols_per_row = M
        _safe_launch_dim_low(inp, out, num_rows, cols_per_row, device)
        return

    # Normal case: inp has B*M*N elements, truly (B, M, N) layout.
    N_inner = _largest_divisor_le(N, DTE_DIM_MAX)
    N_outer = N // N_inner

    # (B, M, N) -> (B, M, N_outer, N_inner)
    # -> permute to (B, N_outer, M, N_inner) -> contiguous
    # -> view as (B*N_outer, M, N_inner)
    inp_4d = inp.reshape(B, M, N_outer, N_inner)
    inp_perm = inp_4d.permute(0, 2, 1, 3).contiguous()

    B_new = B * N_outer
    out_new = torch.empty((B_new, N_inner), dtype=out.dtype, device=device)
    _launch_kernel_dim_mid(inp_perm, out_new, B_new, M, N_inner)

    out.copy_(out_new.reshape(out.shape))


def _largest_divisor_le(n, limit):
    """Find a large divisor of n that is <= limit.

    Uses a two-pronged strategy:
    1. Try descending powers of 2 (fast for typical tensor sizes).
    2. Try limit // k for small k to find divisors close to the limit itself.
    Returns the largest candidate found.
    """
    if n <= limit:
        return n

    best = 1

    # Strategy 1: descending powers of 2 — O(log(limit))
    p2 = 1 << (limit.bit_length() - 1) if limit > 0 else 1
    while p2 >= 1:
        if n % p2 == 0 and p2 <= limit:
            best = max(best, p2)
            break
        p2 >>= 1

    # Strategy 2: try n // k for small k — finds large divisors near limit
    # e.g. if n = 33554433 (odd), n//2 won't work but n//3 might
    k = max(1, n // limit)
    for k in range(k, k + 64):
        d = n // k
        if d <= 0:
            break
        if d <= limit and n % d == 0:
            best = max(best, d)
            break

    return best


def _empty_mean_dim_output(x, dim, keepdim, dtype):
    """Build the output of mean over dim for empty inputs.

    PyTorch semantics: when the reduced axis has length 0, all output elements are NaN, and keepdim
    decides whether the reduced axis is retained. Construct the result directly here, without launching a 0-element
    Triton kernel,
    and avoiding division by zero in ``numel // N`` when N is 0.
    """
    if dim is None:
        reduced_dims = list(range(x.ndim))
    elif isinstance(dim, int):
        reduced_dims = [dim % x.ndim]
    else:
        reduced_dims = [d % x.ndim for d in dim]

    out_shape = list(x.shape)
    for axis in reduced_dims:
        out_shape[axis] = 1
    if not keepdim:
        kept_axes = set(range(x.ndim)) - set(reduced_dims)
        out_shape = [size for axis, size in enumerate(out_shape) if axis in kept_axes]

    return torch.full(out_shape, float("nan"), dtype=dtype, device=x.device)


def mean_dim(x, dim, keepdim=False, *, dtype=None):
    logger.debug("GEMS_ENFLAME MEAN_DIM")
    if not x.is_contiguous():
        x = x.contiguous()
    dtype = _mean_dtype(x, dtype)
    # Empty inputs uniformly take the NaN-output short circuit, covering single-axis / multi-axis / dim=None branches.
    if x.numel() == 0:
        return _empty_mean_dim_output(x, dim, keepdim, dtype)
    if dim is None:
        out = mean(x, dtype=dtype)
        if not keepdim:
            out = out.reshape([1] * x.ndim)
        return out

    if isinstance(dim, int):
        dim = [dim]
    if len(dim) == 1:
        inp = x
        mean_dim_idx = dim[0] % inp.ndim
        shape = list(x.shape)
        if shape[mean_dim_idx] == 1:
            if not keepdim:
                inp = inp.squeeze(dim)
            return inp
        shape[mean_dim_idx] = 1
        out = torch.empty(shape, dtype=dtype, device=x.device)
        if mean_dim_idx == 0:
            M = inp.shape[0]
            N = inp.numel() // M
            with torch_device_fn.device(inp.device):
                _safe_launch_dim_high(inp, out, M, N, inp.device)
        elif mean_dim_idx == inp.ndim - 1:
            N = inp.shape[inp.ndim - 1]
            M = inp.numel() // N
            with torch_device_fn.device(inp.device):
                _safe_launch_dim_low(inp, out, M, N, inp.device)
        else:
            B = 1
            for i in range(0, mean_dim_idx):
                B *= inp.shape[i]
            M = inp.shape[mean_dim_idx]
            N = 1
            for i in range(mean_dim_idx + 1, inp.ndim):
                N *= inp.shape[i]
            if B <= N * 128:
                with torch_device_fn.device(inp.device):
                    _safe_launch_dim_mid(inp, out, B, M, N, inp.device)
            else:
                in_reshape = inp.reshape((B, M, N))
                inp_new = dim_compress(in_reshape, {0, 2})
                M_new = inp_new.shape[0]
                N_new = inp_new.numel() // M_new
                with torch_device_fn.device(inp.device):
                    _safe_launch_dim_high(inp_new, out, M_new, N_new, inp.device)
        if not keepdim:
            out = out.squeeze(dim)
        return out
    else:
        shape = list(x.shape)
        dim = [d % x.ndim for d in dim]
        x = dim_compress(x, dim)
        N = 1
        for i in dim:
            N *= shape[i]
            shape[i] = 1
        M = x.numel() // N
        if M == 1:
            out = mean(x)
            for i in range(0, x.ndim):
                out = out.unsqueeze(0)
            if not keepdim:
                out = out.squeeze(dim)
            return out
        else:
            out = torch.empty(shape, dtype=dtype, device=x.device)
            with torch_device_fn.device(x.device):
                _safe_launch_dim_low(x, out, M, N, x.device)
            if not keepdim:
                out = out.squeeze(dim)
            return out


# Shared registration state is attached to torch: this file is loaded once each under ``gcu400.ops.mean``,
# ``_enflame.gcu400.ops.mean``, and ``flag_gems.runtime.backend....mean`` — three
# module paths; the handle attached to torch ensures registration happens only once per process.
_ATEN_STATE_ATTR = "_flag_gems_gcu400_mean_aten_state"


# Module-level handle keeps the Library (and its kernels) alive for the
# process; the registration itself must never break the flag_gems import.
