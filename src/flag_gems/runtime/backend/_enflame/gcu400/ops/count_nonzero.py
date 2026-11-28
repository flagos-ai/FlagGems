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

"""GCU400-specialized count_nonzero implementation (L600 measured tiers).

Global counting: small inputs run a single program; the rest use 1-D streaming segmented counting
plus an atomic merge (Out pre-zeroed, relaxed without fence), within one kernel body.

Single-dim counting expands on the original (M, N, K) layout, so no dim needs a dim_compress
transpose copy:
- K == 1 (reduced dim innermost): row-reduction 2-D tile, tiles tiered by width (65536 elements
  for 2B/1B, 32768 for 4B, 16384 for 8B; measured still faster past the 448KB two-program line);
  tiles >= 8192 elements use num_warps=4 and small tiles use 2.
- K > 1 (reduced dim trailing): tasks become (m, tail block k) with the tail fixed at 512 wide to
  guarantee >= 256B contiguous runs; when enough tasks exist the reduced dim is chunked by 128
  rows, trading two stages (partial int32 + merge) for parallelism, measured 3-7x over the
  single-stage shrink scheme; small inputs fall back to single-stage with a wider tail.
All kernels write the int64 output directly, saving a host-side .to(int64) launch.
"""

import builtins
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

_min = builtins.min
_max = builtins.max
_NP2 = triton.next_power_of_2
_CDIV = triton.cdiv

# GCU400 has 48 compute units; a fixed grid + in-kernel striding can keep SP fully occupied.
MAX_GRID = 48

# Streaming tile for global counting (element count, bucketed by dtype width; measured dsm ~8B/elt + miscellaneous).
_FLAT_BLOCK = {1: 65536, 2: 65536, 4: 32768, 8: 16384}

# Upper limit for one-shot handling in a single program (a single program can use the full 896KB hard limit).
_FLAT_SINGLE = {1: 65536, 2: 65536, 4: 65536, 8: 32768}

# Element count of the row-reduction tile and upper limit on reduction width (cross-lane reduction cost grows with
# width).
_ROW_TILE = {1: 65536, 2: 65536, 4: 32768, 8: 16384}
_ROW_MAX_BN = {1: 16384, 2: 16384, 4: 8192, 8: 4096}

# Tail-block width and number of blocked rows along the reduced dimension for the two-stage non-row-reduction path (K >
# 1), bucketed by dtype:
# a large tile with es=8 silently exceeds the DSM domain and yields wrong values; falling back to 16K elements is
# empirically safe).
_TAIL_BK = 512
_CHUNK_BI = {1: 128, 2: 128, 4: 128, 8: 32}
_CHUNK_BC = 128

# Byte range for using a single program with a large tile for extremely small inputs (streaming with 1 SP is fastest).
_SMALL_SINGLE_MIN = 131072
_SMALL_SINGLE_MAX = 262144

# Threshold for enabling the two-stage path: if there are too few tasks, first shrink the chunk row count to increase
# them; if still insufficient, fall back to the single-stage path.
_MIN_TASKS_2P = 16


@triton.jit(do_not_specialize=["n_elem"])
def count_global_direct_kernel(
    X, Out, n_elem, BLOCK: tl.constexpr, num_stages: tl.constexpr = 1
):
    """After Out is pre-zeroed, each program atomically accumulates its segmented count (relaxed, no fence)."""
    pid = tl.program_id(0)
    step = tl.num_programs(0)
    num_tile = (n_elem + BLOCK - 1) // BLOCK
    acc = tl.zeros([], dtype=tl.int64)
    for tile_id in tl.range(pid, num_tile, step, num_stages=num_stages):
        offset = tile_id * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + offset, mask=offset < n_elem, other=0)
        acc += tl.sum((x != 0).to(tl.int32), axis=0).to(tl.int64)
    tl.atomic_add(Out, acc, sem="relaxed")


@triton.jit(do_not_specialize=["n_elem"])
def count_global_single_kernel(X, Out, n_elem, BLOCK: tl.constexpr):
    """Small inputs: one program counts to the end, saving the zeroing and atomic overhead."""
    offset = tl.arange(0, BLOCK)
    x = tl.load(X + offset, mask=offset < n_elem, other=0)
    cnt = tl.sum((x != 0).to(tl.int32), axis=0).to(tl.int64)
    tl.store(Out, cnt)


@triton.jit(do_not_specialize=["M", "N"])
def count_dim_inner_kernel(
    X,
    Out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    num_stages: tl.constexpr = 1,
):
    """Reduced dim innermost: a BLOCK_M x BLOCK_N tile, counted row by row."""
    pid = tl.program_id(0)
    step = tl.num_programs(0)
    num_tile = (M + BLOCK_M - 1) // BLOCK_M
    m_offset = tl.arange(0, BLOCK_M)
    for tile_id in tl.range(pid, num_tile, step, num_stages=num_stages):
        rows = tile_id * BLOCK_M + m_offset
        row_mask = rows < M
        cnt = tl.zeros([BLOCK_M], dtype=tl.int64)
        for col_start in tl.range(0, N, BLOCK_N):
            n_offset = col_start + tl.arange(0, BLOCK_N)
            mask = row_mask[:, None] & (n_offset[None, :] < N)
            x = tl.load(X + rows[:, None] * N + n_offset[None, :], mask=mask, other=0)
            cnt += tl.sum((x != 0).to(tl.int32), axis=1).to(tl.int64)
        tl.store(Out + rows, cnt, row_mask)


@triton.jit(do_not_specialize=["M", "N", "K"])
def count_dim_non_inner_kernel(
    X,
    Out,
    M,
    N,
    K,
    BLOCK_I: tl.constexpr,
    BLOCK_K: tl.constexpr,
    num_stages: tl.constexpr = 1,
):
    """K elements follow the reduced dim: a task is (row m, tail block k), accumulated along the reduced dim.

    Small-input path: the tail takes BLOCK_K columns along the contiguous dim, and when there are
    too few tasks the grid count is doubled and the tail narrowed, avoiding tail effects on 48 units.
    """
    pid = tl.program_id(0)
    step = tl.num_programs(0)
    num_k_tile = (K + BLOCK_K - 1) // BLOCK_K
    num_task = M * num_k_tile
    for task_id in tl.range(pid, num_task, step, num_stages=num_stages):
        m_index = task_id // num_k_tile
        k_tile = task_id % num_k_tile
        k_offset = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = k_offset < K
        base = m_index * N * K
        cnt = tl.zeros([BLOCK_K], dtype=tl.int64)
        for i_start in tl.range(0, N, BLOCK_I):
            i_offset = i_start + tl.arange(0, BLOCK_I)
            mask = (i_offset[:, None] < N) & k_mask[None, :]
            x = tl.load(
                X + base + i_offset[:, None] * K + k_offset[None, :],
                mask=mask,
                other=0,
            )
            cnt += tl.sum((x != 0).to(tl.int32), axis=0).to(tl.int64)
        tl.store(Out + m_index * K + k_offset, cnt, k_mask)


@triton.jit(do_not_specialize=["M", "N", "K", "num_chunk"])
def count_dim_non_inner_partial_kernel(
    X,
    Part,
    M,
    N,
    K,
    num_chunk,
    BLOCK_I: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Two-stage phase 1: a task is (row m, tail block k, reduced-dim chunk).

    The tail is fixed at 512 wide (>= 256B runs) and the reduced dim is chunked by BLOCK_I rows for
    parallelism; the partial layout is [M * num_chunk, K] as int32.
    """
    pid = tl.program_id(0)
    step = tl.num_programs(0)
    num_k_tile = (K + BLOCK_K - 1) // BLOCK_K
    num_task = M * num_k_tile * num_chunk
    for task_id in tl.range(pid, num_task, step):
        m_index = task_id // (num_k_tile * num_chunk)
        rem = task_id % (num_k_tile * num_chunk)
        k_tile = rem // num_chunk
        chunk = rem % num_chunk
        k_offset = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = k_offset < K
        i_offset = chunk * BLOCK_I + tl.arange(0, BLOCK_I)
        mask = (i_offset[:, None] < N) & k_mask[None, :]
        base = m_index * N * K
        x = tl.load(
            X + base + i_offset[:, None] * K + k_offset[None, :], mask=mask, other=0
        )
        cnt = tl.sum((x != 0).to(tl.int32), axis=0)
        tl.store(Part + (m_index * num_chunk + chunk) * K + k_offset, cnt, k_mask)


@triton.jit(do_not_specialize=["M", "K", "num_chunk"])
def count_dim_non_inner_final_kernel(
    Part,
    Out,
    M,
    K,
    num_chunk,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Two-stage phase 2: merges the [num_chunk, K] partials per tail block into int64."""
    pid = tl.program_id(0)
    step = tl.num_programs(0)
    num_k_tile = (K + BLOCK_K - 1) // BLOCK_K
    num_task = M * num_k_tile
    for task_id in tl.range(pid, num_task, step):
        m_index = task_id // num_k_tile
        k_tile = task_id % num_k_tile
        k_offset = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = k_offset < K
        part_base = Part + m_index * num_chunk * K + k_offset
        acc = tl.zeros([BLOCK_K], dtype=tl.int32)
        for c_start in tl.range(0, num_chunk, BLOCK_C):
            c_offset = c_start + tl.arange(0, BLOCK_C)
            c_mask = c_offset < num_chunk
            part = tl.load(
                part_base + c_offset[:, None] * K,
                mask=c_mask[:, None] & k_mask[None, :],
                other=0,
            )
            acc += tl.sum(part, axis=0)
        tl.store(Out + m_index * K + k_offset, acc.to(tl.int64), k_mask)


def _launch_global_count(inp, out, n_elem):
    """Global counting: small inputs run a single program; the rest 1-D streaming + atomic merge."""
    element_size = inp.element_size()
    single_block = _min(_NP2(_max(n_elem, 1)), _FLAT_SINGLE.get(element_size, 32768))
    block = _min(_NP2(_max(n_elem, 1)), _FLAT_BLOCK.get(element_size, 32768))

    with torch_device_fn.device(inp.device):
        if n_elem <= single_block:
            count_global_single_kernel[(1,)](
                inp, out, n_elem, BLOCK=single_block, num_warps=2
            )
            return

        grid = _min(MAX_GRID, _CDIV(n_elem, block))
        count_global_direct_kernel[(grid,)](
            inp, out, n_elem, BLOCK=block, num_stages=1, num_warps=2
        )


def _launch_dim_count(inp, out, M, N, K):
    """Single-dim counting: K == 1 takes row reduction, K > 1 takes task splitting (two or one stage)."""
    element_size = inp.element_size()

    with torch_device_fn.device(inp.device):
        if K == 1:
            tile = _ROW_TILE.get(element_size, 32768)
            max_bn = _ROW_MAX_BN.get(element_size, 8192)
            block_n = _min(_NP2(N), max_bn)
            batch_rows = _NP2(_CDIV(M, MAX_GRID))
            block_m = _max(1, _min(tile // block_n, batch_rows))
            grid = _min(MAX_GRID, _CDIV(M, block_m))
            num_warps = 4 if block_m * block_n >= 8192 else 2
            count_dim_inner_kernel[(grid,)](
                inp,
                out,
                M,
                N,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                num_stages=1,
                num_warps=num_warps,
            )
            return

        block_k = _min(_NP2(K), _TAIL_BK)
        num_k_tile = _CDIV(K, block_k)
        total_bytes = M * N * K * element_size

        # For extremely small inputs of 128-256KB: handle everything in one shot with a single program and a large tile
        # (streaming with 1 SP
        # is faster than fragmented accesses across multiple programs; measured (256,256) 6.4us vs 14.8us for the
        # shrunken path).
        if _SMALL_SINGLE_MIN <= total_bytes <= _SMALL_SINGLE_MAX:
            elem_cap = 16384 if element_size >= 8 else 65536
            block_i = _min(_NP2(N), _max(1, elem_cap // block_k))
            grid = _min(MAX_GRID, M * num_k_tile)
            count_dim_non_inner_kernel[(grid,)](
                inp,
                out,
                M,
                N,
                K,
                BLOCK_I=block_i,
                BLOCK_K=block_k,
                num_stages=1,
                num_warps=2,
            )
            return

        # Two-stage: fixed 512-wide tail block (>= 256B contiguous segment), chunking the reduced dimension to trade for
        # parallelism; for small inputs, shrink the chunk row count to increase the task count.
        block_i_chunk = _CHUNK_BI.get(element_size, 128)
        num_chunk = _CDIV(N, block_i_chunk)
        num_task = M * num_k_tile * num_chunk
        if num_chunk >= 2:
            chunk_floor = 32 if element_size >= 8 else 16
            while num_task < _MIN_TASKS_2P and block_i_chunk > chunk_floor:
                block_i_chunk = _max(chunk_floor, block_i_chunk // 2)
                num_chunk = _CDIV(N, block_i_chunk)
                num_task = M * num_k_tile * num_chunk
        if num_chunk >= 2 and num_task >= _MIN_TASKS_2P:
            part = torch.empty([M * num_chunk, K], dtype=torch.int32, device=inp.device)
            grid = _min(MAX_GRID, num_task)
            count_dim_non_inner_partial_kernel[(grid,)](
                inp,
                part,
                M,
                N,
                K,
                num_chunk,
                BLOCK_I=block_i_chunk,
                BLOCK_K=block_k,
                num_warps=4,
            )
            final_task = M * num_k_tile
            count_dim_non_inner_final_kernel[(_min(MAX_GRID, final_task),)](
                part,
                out,
                M,
                K,
                num_chunk,
                BLOCK_K=block_k,
                BLOCK_C=_CHUNK_BC,
                num_warps=2,
            )
            return

        # Smaller inputs: single-stage, reducing the tail-block width until there are 2x the grid count.
        if element_size <= 2:
            budget_elems = 65536 // element_size
        elif element_size == 4:
            budget_elems = 32768
        else:
            budget_elems = 16384
        while block_k > 16 and M * num_k_tile < 2 * MAX_GRID:
            block_k //= 2
            num_k_tile = _CDIV(K, block_k)
        block_i = _min(_NP2(N), _max(1, budget_elems // block_k))
        block_i = _max(1, _min(block_i, 1024))
        grid = _min(MAX_GRID, M * num_k_tile)
        count_dim_non_inner_kernel[(grid,)](
            inp,
            out,
            M,
            N,
            K,
            BLOCK_I=block_i,
            BLOCK_K=block_k,
            num_stages=1,
            num_warps=1,
        )


def count_nonzero(x, dim=None):
    logger.debug("GEMS_ENFLAME COUNT_NONZERO")
    if dim is not None:
        dim = dim % x.ndim
        shape = list(x.shape)
        N = shape[dim]
        out_shape = shape[:dim] + shape[dim + 1 :]
        if N == 0:
            return torch.zeros(out_shape, dtype=torch.int64, device=x.device)
        M = 1
        for s in shape[:dim]:
            M *= s
        K = x.numel() // M // N
        out = torch.empty(out_shape, dtype=torch.int64, device=x.device)
        if M == 0:
            return out
        x = x.contiguous()
        _launch_dim_count(x, out, M, N, K)
        return out

    x = x.contiguous().flatten()
    n_elem = x.numel()
    out = torch.zeros((), dtype=torch.int64, device=x.device)
    if n_elem == 0:
        return out
    _launch_global_count(x, out, n_elem)
    return out
