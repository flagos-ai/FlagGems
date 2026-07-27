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

logger = logging.getLogger(__name__)

# DTE hardware register bitwidth limit: each tile dimension must fit in 24 bits.
DTE_DIM_MAX = (1 << 24) - 1


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
        inp_val = tl.load(inp_ptrs, mask=mask, other=0.0)
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


def mean(inp, *, dtype=None):
    M = inp.numel()
    if dtype is None:
        dtype = inp.dtype

    block_size = 32 * 64
    if M < 24 * 16 * 1024:
        block_size = 16 * 1024
    elif M >= 24 * 32 * 1024 and M < 24 * 64 * 1024:
        block_size = 32 * 1024
    elif M >= 24 * 64 * 1024:
        block_size = 64 * 1024
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)
    num_stages = 1
    if mid_size > 4 * 24:
        num_stages = 3
    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        mean_kernel_1[(min(triton.cdiv(M, block_size), 24), 1, 1)](
            inp, mid, M, block_size, num_stages=num_stages, num_warps=1
        )
        mean_kernel_2[(1, 1, 1)](mid, out, M, mid_size, block_mid, num_warps=1)
    return out


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 2048:
        return False
    if BLOCK_M * BLOCK_N >= 256 * 1024:
        return False
    return True


@libentry()
@libtuner(
    configs=list(filter(keep, runtime.get_tuned_config("naive_reduction"))),
    key=["M", "N"],
)
@triton.jit
def mean_kernel_dim_low(
    inp,
    Mean,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    num_stages: tl.constexpr = 1,
):
    # Map the program id to the row of X it should compute.
    step = tl.num_programs(0)
    pid_m = tl.program_id(0)
    num_tile = (M + BLOCK_M - 1) // BLOCK_M
    for tile_id in tl.range(pid_m, num_tile, step):
        m_offset = tile_id * BLOCK_M + tl.arange(0, BLOCK_M)
        # # Compute mean
        # _mean = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        n_offset_0 = tl.arange(0, BLOCK_N)
        offset_0 = m_offset[:, None] * N + n_offset_0[None, :]
        # set mask
        mask_0 = m_offset[:, None] < M and n_offset_0[None, :] < N
        inp_ptrs_0 = inp + offset_0
        _mean = tl.load(inp_ptrs_0, mask_0, other=0.0).to(tl.float32)
        if N > BLOCK_N:
            for i in tl.range(BLOCK_N, N, BLOCK_N, num_stages=num_stages):
                n_offset = i + tl.arange(0, BLOCK_N)
                offset = m_offset[:, None] * N + n_offset[None, :]
                # set mask
                mask = m_offset[:, None] < M and n_offset[None, :] < N
                inp_ptrs = inp + offset
                a = tl.load(inp_ptrs, mask, other=0.0).to(tl.float32)
                _mean = a + _mean
        _mean /= N
        mean_row = tl.sum(_mean, axis=1)
        Mean_ptr = Mean + m_offset
        mask_m = m_offset < M
        tl.store(Mean_ptr, mean_row, mask_m)


@libentry()
@libtuner(
    configs=list(filter(keep, runtime.get_tuned_config("naive_reduction"))),
    key=["M", "N"],
)
@triton.jit
def mean_kernel_dim_high(
    inp,
    Mean,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    num_stages: tl.constexpr = 1,
):
    # Map the program id to the row of X it should compute.
    pid_n = tl.program_id(0)
    step = tl.num_programs(0)
    num_tile = (N + BLOCK_N - 1) // BLOCK_N
    for tile_id_n in tl.range(pid_n, num_tile, step):
        n_offset = tile_id_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_offset_0 = tl.arange(0, BLOCK_M)
        offset_0 = m_offset_0[:, None] * N + n_offset[None, :]
        mask_0 = m_offset_0[:, None] < M and n_offset[None, :] < N
        inp_ptrs_0 = inp + offset_0
        _mean = tl.load(inp_ptrs_0, mask_0, other=0.0).to(tl.float32)
        if M > BLOCK_M:
            for i in tl.range(BLOCK_M, M, BLOCK_M, num_stages=num_stages):
                m_offset = i + tl.arange(0, BLOCK_M)
                offset = m_offset[:, None] * N + n_offset[None, :]
                # set mask
                mask = m_offset[:, None] < M and n_offset[None, :] < N
                inp_ptrs = inp + offset
                a = tl.load(inp_ptrs, mask, other=0.0).to(tl.float32)
                _mean += a
        _mean /= M
        mean_col = tl.sum(_mean, axis=0)
        Mean_ptr = Mean + n_offset
        n_mask = n_offset < N
        tl.store(Mean_ptr, mean_col, n_mask)


@libentry()
@libtuner(
    configs=list(filter(keep, runtime.get_tuned_config("naive_reduction"))),
    key=["M", "N"],
)
@triton.jit
def mean_kernel_dim_mid(
    inpIn,
    out_value_in,
    B,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    num_stages: tl.constexpr = 1,
):
    # Map the program id to the row of X it should compute.
    pid_b = tl.program_id(1)
    pid_n = tl.program_id(0)
    step = tl.num_programs(1)
    for tile_id_b in tl.range(pid_b, B, step):
        b_offset = tile_id_b * M * N
        inp = inpIn + b_offset
        out_value = out_value_in + tile_id_b * N
        n_offset = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_offset_0 = tl.arange(0, BLOCK_M)
        offset_0 = m_offset_0[:, None] * N + n_offset[None, :]
        mask_0 = m_offset_0[:, None] < M and n_offset[None, :] < N
        inp_ptrs_0 = inp + offset_0
        _mean = tl.load(inp_ptrs_0, mask_0, other=0.0).to(tl.float32)
        if M > BLOCK_M:
            for i in tl.range(BLOCK_M, M, BLOCK_M, num_stages=num_stages):
                m_offset = i + tl.arange(0, BLOCK_M)
                offset = m_offset[:, None] * N + n_offset[None, :]
                # set mask
                mask = m_offset[:, None] < M and n_offset[None, :] < N
                inp_ptrs = inp + offset
                a = tl.load(inp_ptrs, mask, other=0.0).to(tl.float32)
                _mean += a
        _mean /= M
        mean = tl.sum(_mean, axis=0)
        out_value_ptrs = out_value + n_offset
        n_mask = n_offset < N
        tl.store(out_value_ptrs, mean, n_mask)


def _launch_kernel_dim_low(inp, out, M, N):
    """Reduce along rows (last-dim reduction): each row has N elements, M rows."""
    grid = lambda meta: (min(triton.cdiv(M, meta["BLOCK_M"]), 24),)
    with torch_device_fn.device(inp.device):
        mean_kernel_dim_low[grid](inp, out, M, N)


def _launch_kernel_dim_high(inp, out, M, N):
    """Reduce along columns (first-dim reduction): M rows reduced to 1, N columns."""
    grid = lambda meta: (min(triton.cdiv(N, meta["BLOCK_N"]), 24),)
    with torch_device_fn.device(inp.device):
        mean_kernel_dim_high[grid](inp, out, M, N)


def _launch_kernel_dim_mid(inp, out, B, M, N):
    """Reduce middle dim: B batches, M reduction length, N columns per batch."""
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), min(B, 24), 1)
    with torch_device_fn.device(inp.device):
        mean_kernel_dim_mid[grid](inp, out, B, M, N)


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
    """Launch mean_kernel_dim_high ensuring N <= DTE_DIM_MAX.

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


def mean_dim(x, dim, keepdim=False, *, dtype=None):
    if dtype is None:
        dtype = x.dtype
    if dim is None:
        out = mean(x, dtype=dtype)
        if not keepdim:
            out = out.reshape([1] * x.ndim)
        return out

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
