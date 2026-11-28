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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry

from ..utils.config_utils import MAX_GRID_DIM

logger = logging.getLogger(__name__)

# Split-N trigger threshold: only worth the extra launch + temporary buffer when the number of N-direction slices
# reaches this value.
SPLIT_MIN_N_TILES = 16


@libentry()
@triton.jit(do_not_specialize=["correction"])
def var_kernel(
    X,
    Var,
    M,
    N,
    correction,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    num_stages: tl.constexpr = 1,
):
    # Persistent grid-stride over row tiles keeps grid.x within the GCU
    # hardware limit (P19: grid.x <= 65535) for large M.
    pid_m = tl.program_id(0)
    step = tl.num_programs(0)
    num_tile = (M + BLOCK_M - 1) // BLOCK_M
    for tile_id in tl.range(pid_m, num_tile, step, num_stages=num_stages):
        m_offset = tile_id * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = m_offset < M

        _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        _sum2 = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for col_off in tl.range(0, N, BLOCK_N):
            n_offset = col_off + tl.arange(0, BLOCK_N)
            offset = m_offset[:, None] * N + n_offset[None, :]
            mask = (m_offset[:, None] < M) & (n_offset[None, :] < N)
            x = tl.load(X + offset, mask, other=0.0).to(tl.float32)
            _sum += x
            _sum2 += x * x

        total_sum = tl.sum(_sum, axis=1)
        total_sum2 = tl.sum(_sum2, axis=1)
        var = (total_sum2 - total_sum * total_sum / N) / (N - correction)
        var = tl.maximum(var, 0.0)

        tl.store(Var + m_offset, var, row_mask)


@libentry()
@triton.jit
def var_global_kernel_1(
    X,
    Mid_sum,
    Mid_sum2,
    M,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr = 1,
):
    # Stage 1 of the global (all-dims) reduction: each program walks several
    # BLOCK_SIZE tiles with a grid-stride loop, so the launch grid stays small
    # and never exceeds the 65535 grid.x hardware limit (P19).
    pid = tl.program_id(0)
    num_prog = tl.num_programs(0)
    num_tile = (M + BLOCK_SIZE - 1) // BLOCK_SIZE
    for tile_id in tl.range(pid, num_tile, num_prog, num_stages=num_stages):
        offset = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offset < M
        x = tl.load(X + offset, mask=mask, other=0.0).to(tl.float32)
        sum_val = tl.sum(x, axis=0)
        sum2_val = tl.sum(x * x, axis=0)
        tl.store(Mid_sum + tile_id, sum_val)
        tl.store(Mid_sum2 + tile_id, sum2_val)


@libentry()
@triton.jit(do_not_specialize=["correction"])
def var_global_kernel_2(
    Mid_sum,
    Mid_sum2,
    Var,
    N,
    correction,
    MID_SIZE,
    BLOCK_MID: tl.constexpr,
):
    # Stage 2 folds the per-tile partial sums in a single program. The block
    # ladder in the host keeps MID_SIZE (hence BLOCK_MID) bounded.
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < MID_SIZE
    partial_sum = tl.load(Mid_sum + offset, mask=mask, other=0.0).to(tl.float32)
    partial_sum2 = tl.load(Mid_sum2 + offset, mask=mask, other=0.0).to(tl.float32)

    total_sum = tl.sum(partial_sum)
    total_sum2 = tl.sum(partial_sum2)

    var = (total_sum2 - total_sum * total_sum / N) / (N - correction)
    var = tl.maximum(var, 0.0)

    tl.store(Var, var)


@libentry()
@triton.jit(do_not_specialize=["M", "N", "NUM_N_TILES"])
def var_split_kernel_1(
    X,
    Partial_sum,
    Partial_sum2,
    M,
    N,
    NUM_N_TILES,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Split-N stage 1: split along N into NUM_N_TILES slices and compute partial sums in parallel.
    # The original var_kernel, with "a single M tile + very large N", ran the entire N loop serially in one program
    # (measured only ~4.8GB/s at 24.6M elements); here it is replaced with N slices across programs,
    # reducing the accumulation-chain length from N/BLOCK_N to the BLOCK_T level, with accuracy improving as well.
    pid = tl.program_id(0)
    step = tl.num_programs(0)
    m_offset = tl.arange(0, BLOCK_M)
    row_mask = m_offset < M

    for tile_id in tl.range(pid, NUM_N_TILES, step):
        n_offset = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N + n_offset[None, :]
        mask = row_mask[:, None] & (n_offset[None, :] < N)
        x = tl.load(X + offset, mask, other=0.0).to(tl.float32)
        # Partial-sum layout [BLOCK_M, NUM_N_TILES]: stage 2 reads contiguously along N.
        tl.store(
            Partial_sum + m_offset * NUM_N_TILES + tile_id,
            tl.sum(x, axis=1),
            row_mask,
        )
        tl.store(
            Partial_sum2 + m_offset * NUM_N_TILES + tile_id,
            tl.sum(x * x, axis=1),
            row_mask,
        )


@libentry()
@triton.jit(do_not_specialize=["N", "NUM_N_TILES", "correction"])
def var_split_kernel_2(
    Partial_sum,
    Partial_sum2,
    Var,
    M,
    N,
    NUM_N_TILES,
    correction,
    BLOCK_M: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    # Split-N stage 2: fold the partial sums from all slices into the final variance.
    m_offset = tl.arange(0, BLOCK_M)
    row_mask = m_offset < M
    acc = tl.zeros([BLOCK_M, BLOCK_T], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_M, BLOCK_T], dtype=tl.float32)

    for start in tl.range(0, NUM_N_TILES, BLOCK_T):
        t_offset = start + tl.arange(0, BLOCK_T)
        off = m_offset[:, None] * NUM_N_TILES + t_offset[None, :]
        mask = row_mask[:, None] & (t_offset[None, :] < NUM_N_TILES)
        acc += tl.load(Partial_sum + off, mask, other=0.0)
        acc2 += tl.load(Partial_sum2 + off, mask, other=0.0)

    total_sum = tl.sum(acc, axis=1)
    total_sum2 = tl.sum(acc2, axis=1)
    var = (total_sum2 - total_sum * total_sum / N) / (N - correction)
    var = tl.maximum(var, 0.0)
    tl.store(Var + m_offset, var, row_mask)


def var(x, dim=None, *, correction=None, keepdim=False):
    logger.debug("GEMS_ENFLAME VAR")
    if correction is None:
        correction = 1.0

    if dim is None or len(dim) == x.ndim:
        dim = list(range(x.ndim))
        shape = [1] * x.ndim
        N = x.numel()
        var = torch.empty(shape, dtype=x.dtype, device=x.device)

        block_size = 32 * 64
        if N < 24 * 16 * 1024:
            block_size = 16 * 1024
        elif N >= 24 * 32 * 1024 and N < 24 * 64 * 1024:
            block_size = 32 * 1024
        elif N >= 24 * 64 * 1024:
            block_size = 64 * 1024

        mid_size = triton.cdiv(N, block_size)
        block_mid = triton.next_power_of_2(mid_size)
        num_stages = 1
        if mid_size > 4 * 24:
            num_stages = 3

        mid_sum = torch.empty([mid_size], dtype=torch.float32, device=x.device)
        mid_sum2 = torch.empty([mid_size], dtype=torch.float32, device=x.device)

        grid_size = min(mid_size, 8)

        with torch_device_fn.device(x.device):
            var_global_kernel_1[(grid_size, 1, 1)](
                x,
                mid_sum,
                mid_sum2,
                N,
                BLOCK_SIZE=block_size,
                num_stages=num_stages,
                num_warps=1,
            )
            var_global_kernel_2[(1, 1, 1)](
                mid_sum,
                mid_sum2,
                var,
                N,
                correction,
                mid_size,
                BLOCK_MID=block_mid,
                num_warps=1,
            )
    else:
        shape = list(x.shape)
        dim = [d % x.ndim for d in dim]
        x = dim_compress(x, dim)
        N = 1
        for i in dim:
            N *= shape[i]
            shape[i] = 1
        M = x.numel() // N
        var = torch.empty(shape, dtype=x.dtype, device=x.device)

        BLOCK_N = min(triton.next_power_of_2(N), 2048)
        BLOCK_M = max(1, min(128, 32768 // BLOCK_N))

        num_stages = 1

        num_m_tiles = triton.cdiv(M, BLOCK_M)
        num_n_tiles = triton.cdiv(N, BLOCK_N)
        grid_m = min(num_m_tiles, MAX_GRID_DIM)

        # With a single M tile and enough N-direction slices, the original implementation's entire N loop lands in just
        # one
        # program (grid_m == 1 when M <= BLOCK_M), giving extremely low bandwidth utilization;
        # in that case, switch to two-stage split-N: compute partial sums over N slices in parallel, then fold them.
        use_split = num_m_tiles == 1 and num_n_tiles >= SPLIT_MIN_N_TILES

        with torch_device_fn.device(x.device):
            if use_split:
                partial_sum = torch.empty(
                    [BLOCK_M, num_n_tiles], dtype=torch.float32, device=x.device
                )
                partial_sum2 = torch.empty(
                    [BLOCK_M, num_n_tiles], dtype=torch.float32, device=x.device
                )
                var_split_kernel_1[(min(num_n_tiles, MAX_GRID_DIM),)](
                    x,
                    partial_sum,
                    partial_sum2,
                    M,
                    N,
                    num_n_tiles,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    num_stages=1,
                    num_warps=1,
                )
                block_t = min(triton.next_power_of_2(num_n_tiles), 256)
                var_split_kernel_2[(1,)](
                    partial_sum,
                    partial_sum2,
                    var,
                    M,
                    N,
                    num_n_tiles,
                    correction,
                    BLOCK_M=BLOCK_M,
                    BLOCK_T=block_t,
                    num_warps=1,
                )
            else:
                var_kernel[(grid_m,)](
                    x,
                    var,
                    M,
                    N,
                    correction,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    num_stages=num_stages,
                    num_warps=1,
                )

    if not keepdim:
        var = var.squeeze(dim=dim)
    return var


def var_dim(x, dim=None, *, correction=None, keepdim=False):
    logger.debug("GEMS_ENFLAME VAR_DIM")
    return var(x, dim=dim, correction=correction, keepdim=keepdim)


def var_correction(x, dim=None, *, correction=None, keepdim=False):
    logger.debug("GEMS_ENFLAME VAR_CORRECTION")
    return var(x, dim=dim, correction=correction, keepdim=keepdim)
