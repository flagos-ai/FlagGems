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

"""GCU400-specific sum family implementation.

Global sum (dim=None / dim=[]) has three paths by data size:
- less than one tile: one program reduces in one pass;
- medium scale with a wide last dimension: 2D block tiling, two stages (segmented partial + merge);
- others: 1D streaming single launch, each program accumulates a segment then atomically merges.

Single-dimension sum is routed by an (M, N, K) expansion: N==1 uses copy, K==1 uses row reduction, K>1 uses (row,
tail-block) splitting.
Multiple dims are first compressed by dim_compress into the (M, N) K == 1 case.
"""

import builtins
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry

logger = logging.getLogger(__name__)

_min = builtins.min
_max = builtins.max

# GCU400 has 48 compute units; a fixed grid plus in-kernel striding keeps the SPs fully utilized.
MAX_GRID = 48

# Global reduction: 4-byte types achieve the best bandwidth with 65536 elements (fp32 256 KiB).
GLOBAL_TILE_ELEMS = 65536

# Streaming tile for 2-byte types: 131072 elements (fp16/bf16 256 KiB). Measured (2026-11-16)
# With grid 48 + nw4, it is 10~11% faster than 65536 elements + 96 ways across the entire 32MiB~2GiB range.
GLOBAL_TILE_ELEMS_2B = 131072

# Byte budget for a single tile in row reduction: sum has only one read pass, so the tile can be as large as 256 KiB.
INNER_TILE_BYTES = 131072

# Byte budget for a single tile in non-row reduction (K > 1).
NON_INNER_TILE_BYTES = 131072

# 2D block tiling for 2D global reduction is enabled only at medium sizes: at larger data sizes, the contiguous
# access of 1D streaming is more cost-effective (measured at 2 GiB: 1D 1.66x vs 2D 1.46x), while at smaller sizes launch
# overhead dominates.
GLOBAL_2D_MIN_BYTES = 64 * 1024 * 1024
GLOBAL_2D_MAX_BYTES = 512 * 1024 * 1024

# Upper limit on reduction width for 2D global reduction (fp16/fp32 measurements are both optimal near this value).
GLOBAL_2D_WIDTH = 8192

# Upper limit on intra-row reduction width: cross-lane reduction cost grows with width,
# measurements show bandwidth drops beyond 16384.
MAX_REDUCE_WIDTH = 16384

# When there are very few rows but extremely long rows, row reduction has only 1 tile, so it must be split once more
# along the N dimension to distribute across 48 programs.
ROW_SPLIT_MAX_ROWS = 8
ROW_SPLIT_MIN_N = 65536


def _tile_elems(element_size, budget):
    """Convert a byte budget into the number of elements one tile can hold."""
    return _max(1024, budget // element_size)


@libentry()
@triton.jit(do_not_specialize=["n_elem"])
def sum_global_direct_kernel(
    X,
    Out,
    n_elem,
    BLOCK: tl.constexpr,
    num_stages: tl.constexpr = 1,
):
    """After Out is pre-zeroed, each program atomically accumulates its partial directly into it.

    The cheapest path within one launch: no Part / Counter and no second merge.
    Partials always accumulate in fp32 (including fp16/bf16 inputs), and only the final atomic merge lands in
    Out's dtype.
    """
    if tl.constexpr(X.dtype.element_ty == tl.float16) | tl.constexpr(
        X.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = X.dtype.element_ty

    pid = tl.program_id(0)
    step = tl.num_programs(0)
    num_tile = (n_elem + BLOCK - 1) // BLOCK
    acc = tl.zeros([], dtype=cdtype)
    for tile_id in tl.range(pid, num_tile, step, num_stages=num_stages):
        offset = tile_id * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + offset, mask=offset < n_elem, other=0.0)
        acc += tl.sum(x).to(cdtype)
    # Programs have no data dependency, only atomicity is needed (no ordering/visibility),
    # so relaxed avoids the extra fence of acq_rel (which is the default).
    tl.atomic_add(Out, acc.to(Out.dtype.element_ty), sem="relaxed")


@libentry()
@triton.jit(do_not_specialize=["n_elem"])
def sum_global_single_kernel(X, Out, n_elem, BLOCK: tl.constexpr):
    """Small inputs: one program reduces to the end, saving the second stage's launch overhead."""
    offset = tl.arange(0, BLOCK)
    x = tl.load(X + offset, mask=offset < n_elem, other=0.0)
    tl.store(Out, tl.sum(x, axis=0).to(Out.dtype.element_ty))


@libentry()
@triton.jit(do_not_specialize=["n_elem"])
def sum_copy_kernel(Out, X, n_elem, BLOCK: tl.constexpr):
    """Reduced dimension length is 1: output is the input itself, one elementwise copy."""
    pid = tl.program_id(0)
    step = tl.num_programs(0)
    num_tile = (n_elem + BLOCK - 1) // BLOCK
    for tile_id in tl.range(pid, num_tile, step):
        offset = tile_id * BLOCK + tl.arange(0, BLOCK)
        mask = offset < n_elem
        x = tl.load(X + offset, mask=mask, other=0.0)
        tl.store(Out + offset, x.to(Out.dtype.element_ty), mask)


@libentry()
@triton.jit(do_not_specialize=["M", "N"])
def sum_dim_row_split_partial_kernel(
    X,
    Part,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    num_stages: tl.constexpr = 1,
):
    """When rows are few: split the in-row N dimension across programs, each producing one partial.

    partial layout is [n_part, BLOCK_M], program i writes row i.
    """
    if tl.constexpr(X.dtype.element_ty == tl.float16) | tl.constexpr(
        X.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = X.dtype.element_ty

    pid = tl.program_id(0)
    step = tl.num_programs(0)
    m_offset = tl.arange(0, BLOCK_M)
    row_mask = m_offset < M
    acc = tl.zeros([BLOCK_M], dtype=cdtype)
    for col_start in tl.range(pid * BLOCK_N, N, step * BLOCK_N, num_stages=num_stages):
        n_offset = col_start + tl.arange(0, BLOCK_N)
        mask = row_mask[:, None] & (n_offset[None, :] < N)
        x = tl.load(
            X + m_offset[:, None] * N + n_offset[None, :], mask=mask, other=0.0
        ).to(cdtype)
        acc += tl.sum(x, axis=1)
    tl.store(Part + pid * BLOCK_M + m_offset, acc.to(Part.dtype.element_ty), row_mask)


@libentry()
@triton.jit(do_not_specialize=["M", "n_part"])
def sum_dim_row_split_final_kernel(
    Part,
    Out,
    M,
    n_part,
    BLOCK_P: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Merge [n_part, BLOCK_M] partials by row."""
    if tl.constexpr(Part.dtype.element_ty == tl.float16) | tl.constexpr(
        Part.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = Part.dtype.element_ty

    m_offset = tl.arange(0, BLOCK_M)
    p_offset = tl.arange(0, BLOCK_P)
    mask = (p_offset[:, None] < n_part) & (m_offset[None, :] < M)
    part = tl.load(
        Part + p_offset[:, None] * BLOCK_M + m_offset[None, :],
        mask=mask,
        other=0.0,
    ).to(cdtype)
    tl.store(
        Out + m_offset, tl.sum(part, axis=0).to(Out.dtype.element_ty), m_offset < M
    )


@libentry()
@triton.jit(do_not_specialize=["M", "N"])
def sum_dim_inner_kernel(
    X,
    Out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    num_stages: tl.constexpr = 1,
):
    """Reduced dimension is innermost: one tile covers BLOCK_M rows x BLOCK_N columns, reducing by row.

    Reduction is done in fp32: low-precision accumulation on rows of length 4096 already has 5e-3 relative error,
    which fails sum_dim's accuracy gate (global sum can use it because the normalization length is huge and tolerance is
    relaxed by reduce_dim).
    """
    if tl.constexpr(X.dtype.element_ty == tl.float16) | tl.constexpr(
        X.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = X.dtype.element_ty

    pid = tl.program_id(0)
    step = tl.num_programs(0)
    num_tile = (M + BLOCK_M - 1) // BLOCK_M
    for tile_id in tl.range(pid, num_tile, step, num_stages=num_stages):
        m_offset = tile_id * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = m_offset < M
        acc = tl.zeros([BLOCK_M], dtype=cdtype)
        for col_start in tl.range(0, N, BLOCK_N):
            n_offset = col_start + tl.arange(0, BLOCK_N)
            mask = row_mask[:, None] & (n_offset[None, :] < N)
            x = tl.load(
                X + m_offset[:, None] * N + n_offset[None, :], mask=mask, other=0.0
            ).to(cdtype)
            acc += tl.sum(x, axis=1)
        tl.store(Out + m_offset, acc.to(Out.dtype.element_ty), row_mask)


@libentry()
@triton.jit(do_not_specialize=["M", "N", "K"])
def sum_dim_non_inner_kernel(
    X,
    Out,
    M,
    N,
    K,
    BLOCK_I: tl.constexpr,
    BLOCK_K: tl.constexpr,
    num_stages: tl.constexpr = 1,
):
    """K elements follow the reduced dimension: task = (row m, tail block k), accumulating along the reduced dimension.

    Tile is (BLOCK_I, BLOCK_K), with BLOCK_K along the tail (memory-contiguous) dimension,
    so each load is several contiguous segments; acc is retained along the tail dimension and written once at the end.
    """
    if tl.constexpr(X.dtype.element_ty == tl.float16) | tl.constexpr(
        X.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = X.dtype.element_ty

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
        acc = tl.zeros([BLOCK_K], dtype=cdtype)
        for i_start in tl.range(0, N, BLOCK_I):
            i_offset = i_start + tl.arange(0, BLOCK_I)
            mask = (i_offset[:, None] < N) & k_mask[None, :]
            x = tl.load(
                X + base + i_offset[:, None] * K + k_offset[None, :],
                mask=mask,
                other=0.0,
            ).to(cdtype)
            acc += tl.sum(x, axis=0)
        tl.store(Out + m_index * K + k_offset, acc.to(Out.dtype.element_ty), k_mask)


@libentry()
@triton.jit(do_not_specialize=["M", "N"])
def sum_global_block_partial_kernel(
    X,
    Part,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    num_stages: tl.constexpr = 1,
):
    """Stage one of 2D block tiling for medium-scale global sum.

    Input is laid out as (M, N) along the last dimension, tile is BLOCK_M rows x BLOCK_N columns, each row has one
    fp32 accumulator across tiles. Reduction width is fixed at BLOCK_N, so cross-lane reduction cost does not grow with
    overall tile size (the key benefit of 2D over a whole-block 1D reduction).
    """
    if tl.constexpr(X.dtype.element_ty == tl.float16) | tl.constexpr(
        X.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = X.dtype.element_ty

    pid = tl.program_id(0)
    step = tl.num_programs(0)
    num_col_tile = (N + BLOCK_N - 1) // BLOCK_N
    num_row_tile = (M + BLOCK_M - 1) // BLOCK_M
    m_offset = tl.arange(0, BLOCK_M)
    n_offset = tl.arange(0, BLOCK_N)
    acc = tl.zeros([BLOCK_M], dtype=cdtype)
    for tile_id in tl.range(
        pid, num_row_tile * num_col_tile, step, num_stages=num_stages
    ):
        row_tile = tile_id // num_col_tile
        col_tile = tile_id - row_tile * num_col_tile
        rows = row_tile * BLOCK_M + m_offset
        cols = col_tile * BLOCK_N + n_offset
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        x = tl.load(X + rows[:, None] * N + cols[None, :], mask=mask, other=0.0).to(
            cdtype
        )
        acc += tl.sum(x, axis=1)
    tl.store(Part + pid * BLOCK_M + m_offset, acc.to(Part.dtype.element_ty))


@libentry()
@triton.jit(do_not_specialize=["n_part"])
def sum_global_block_final_kernel(
    Part,
    Out,
    n_part,
    BLOCK_P: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Compress [n_part, BLOCK_M] partials into one scalar."""
    if tl.constexpr(Part.dtype.element_ty == tl.float16) | tl.constexpr(
        Part.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = Part.dtype.element_ty

    p_offset = tl.arange(0, BLOCK_P)[:, None]
    m_offset = tl.arange(0, BLOCK_M)[None, :]
    part = tl.load(
        Part + p_offset * BLOCK_M + m_offset, mask=p_offset < n_part, other=0.0
    ).to(cdtype)
    tl.store(
        Out,
        tl.sum(tl.sum(part, axis=1), axis=0).to(Out.dtype.element_ty),
    )


def _launch_global_sum(inp, out, n_elem):
    """Global sum: choose among three paths by data size.

    - less than one tile: one kernel produces the result directly;
    - medium scale with a wide last dimension: 2D block tiling (reduction width compressed to BLOCK_N);
    - others (including very large data): 1D streaming single launch, segmented accumulation + atomic merge.
    """
    element_size = inp.element_size()
    block = _min(triton.next_power_of_2(n_elem), GLOBAL_TILE_ELEMS)
    num_bytes = n_elem * element_size

    with torch_device_fn.device(inp.device):
        if n_elem <= block:
            sum_global_single_kernel[(1,)](inp, out, n_elem, BLOCK=block, num_warps=2)
            return

        if (
            inp.dim() >= 2
            and inp.shape[-1] >= 1024
            and GLOBAL_2D_MIN_BYTES <= num_bytes <= GLOBAL_2D_MAX_BYTES
        ):
            num_cols = inp.shape[-1]
            num_rows = n_elem // num_cols
            block_n = _min(triton.next_power_of_2(num_cols), GLOBAL_2D_WIDTH)
            block_m = _max(1, _tile_elems(element_size, INNER_TILE_BYTES) // block_n)
            num_tile = triton.cdiv(num_rows, block_m) * triton.cdiv(num_cols, block_n)
            grid = _min(MAX_GRID * 2, num_tile)
            part_dtype = torch.float32 if inp.is_floating_point() else out.dtype
            part = torch.empty([grid * block_m], dtype=part_dtype, device=inp.device)
            sum_global_block_partial_kernel[(grid,)](
                inp,
                part,
                num_rows,
                num_cols,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                num_stages=1,
                num_warps=2,
            )
            sum_global_block_final_kernel[(1,)](
                part,
                out,
                grid,
                BLOCK_P=triton.next_power_of_2(grid),
                BLOCK_M=block_m,
                num_warps=1,
            )
            return

        # 1D streaming: 2-byte types use 2x tile + 48 ways (fp16/bf16 measured 32MiB~2GiB
        # across the full range it is 10~11% faster than 65536 elements + 96 ways); 4-byte types keep the 65536-element
        # tile.
        # Small sizes are dominated by launch overhead and are insensitive to either configuration.
        block = _min(
            triton.next_power_of_2(n_elem),
            GLOBAL_TILE_ELEMS_2B if element_size == 2 else GLOBAL_TILE_ELEMS,
        )
        grid = _min(MAX_GRID, triton.cdiv(n_elem, block))

        # Each program atomically accumulates its partial directly into the pre-zeroed Out, single launch, no temporary
        # buffer. partials are accumulated in fp32 (even for fp16/bf16 inputs), and only the final atomic
        # merge lands on the dtype of Out.
        out.zero_()
        sum_global_direct_kernel[(grid,)](
            inp,
            out,
            n_elem,
            BLOCK=block,
            num_stages=1,
            num_warps=4,
        )


def _launch_dim_sum(inp, out, M, N, K):
    """Single-dimension sum: routed by (M, N, K) shape.

    N == 1           -> output is input, elementwise copy;
    K == 1 and M == 1 -> degenerates to global sum (reuse the single-launch global path);
    K == 1 with very few rows -> split once more along N, otherwise 1 program serializes the whole row;
    K == 1           -> row reduction (BLOCK_M x BLOCK_N tile);
    K > 1            -> (row, tail-block) partitioning, accumulating along the reduced dimension.
    """
    element_size = inp.element_size()

    with torch_device_fn.device(inp.device):
        if N == 1:
            n_elem = inp.numel()
            block = _min(
                triton.next_power_of_2(n_elem),
                _tile_elems(element_size, INNER_TILE_BYTES),
            )
            grid = _min(MAX_GRID, triton.cdiv(n_elem, block))
            sum_copy_kernel[(grid,)](
                out,
                inp,
                n_elem,
                BLOCK=block,
                num_warps=1 if block < 4096 else 4,
            )
            return

        if K == 1:
            if M == 1:
                _launch_global_sum(inp, out, inp.numel())
                return
            if M <= ROW_SPLIT_MAX_ROWS and N >= ROW_SPLIT_MIN_N:
                block_m = triton.next_power_of_2(M)
                block_n = MAX_REDUCE_WIDTH
                grid = _min(MAX_GRID, triton.cdiv(N, block_n))
                part = torch.empty(
                    [grid * block_m], dtype=torch.float32, device=inp.device
                )
                sum_dim_row_split_partial_kernel[(grid,)](
                    inp,
                    part,
                    M,
                    N,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    num_stages=1,
                    num_warps=2,
                )
                sum_dim_row_split_final_kernel[(1,)](
                    part,
                    out,
                    M,
                    grid,
                    BLOCK_P=triton.next_power_of_2(grid),
                    BLOCK_M=block_m,
                    num_warps=1,
                )
                return

            # fp32 cross-lane reduction is cheaper, so extremely long rows can double the reduction width
            max_width = MAX_REDUCE_WIDTH * 2 if element_size >= 4 else MAX_REDUCE_WIDTH
            block_n = _min(triton.next_power_of_2(N), max_width)
            batch_rows = triton.next_power_of_2(triton.cdiv(M, MAX_GRID))
            block_m = _tile_elems(element_size, INNER_TILE_BYTES) // block_n
            block_m = _max(1, _min(block_m, batch_rows))
            grid = _min(MAX_GRID, triton.cdiv(M, block_m))
            sum_dim_inner_kernel[(grid,)](
                inp,
                out,
                M,
                N,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                num_stages=1,
                num_warps=1 if block_m * block_n <= 512 else 2,
            )
            return

        # Tail-block width: prefer a larger value (longer memory access segments). For 4-byte types, when task count is
        # insufficient, fall back to 2x the grid count
        # to avoid the tail effect idling the 48 compute units; 16-bit is not halved — the axis=0 tree
        # reduction has a known slow path when the contiguous dimension width is <=256 (same pathology family as prod).
        block_k = _min(triton.next_power_of_2(K), 512)
        if element_size == 2:
            # For 16-bit when the tree width is sufficient, raise nw to 4: measured (1024,1024,512) fp16
            # 616 -> 440us (-28%), (64,512,512) 31.9 -> 26.0us. When K < 512
            # the tree is too narrow for the fast path, so keep the original nw=1 setting.
            num_warps = 4 if block_k >= 512 else 1
        else:
            while block_k > 16 and M * triton.cdiv(K, block_k) < 2 * MAX_GRID:
                block_k //= 2
            num_warps = 1
        block_i = _min(
            triton.next_power_of_2(N),
            _tile_elems(element_size, NON_INNER_TILE_BYTES) // block_k,
        )
        block_i = _max(1, _min(block_i, 1024))
        num_task = M * triton.cdiv(K, block_k)
        grid = _min(MAX_GRID, num_task)
        sum_dim_non_inner_kernel[(grid,)](
            inp,
            out,
            M,
            N,
            K,
            BLOCK_I=block_i,
            BLOCK_K=block_k,
            num_stages=1,
            num_warps=num_warps,
        )


def _normalize_dtype(inp, dtype):
    """Keep dtype inference consistent with the common implementation (bool accumulates as integer)."""
    if dtype is None:
        dtype = inp.dtype
        if dtype is torch.bool:
            inp = inp.to(torch.int32)
            dtype = torch.int32
    return inp, dtype


def sum(inp, *, dtype=None):
    logger.debug("GEMS_ENFLAME SUM")

    inp, out_dtype = _normalize_dtype(inp, dtype)
    inp = inp.contiguous()
    n_elem = inp.numel()

    out = torch.empty([], dtype=out_dtype, device=inp.device)
    if n_elem == 0:
        out.zero_()
        return out

    _launch_global_sum(inp, out, n_elem)
    return out


def sum_out(inp, *, dtype=None, out):
    logger.debug("GEMS_ENFLAME SUM_OUT")

    inp, _ = _normalize_dtype(inp, dtype)
    inp = inp.contiguous()
    n_elem = inp.numel()
    if n_elem == 0:
        out.zero_()
        return out

    _launch_global_sum(inp, out, n_elem)
    return out


def _empty_out_shape(shape, dim, keepdim):
    """Output shape for empty inputs (same as aten)."""
    if dim is None:
        return [1] * len(shape) if keepdim else []
    dims_to_reduce = dim if isinstance(dim, (list, tuple)) else [dim]
    if len(dims_to_reduce) == 0:
        return [1] * len(shape) if keepdim else []
    out_shape = list(shape)
    if keepdim:
        for d in dims_to_reduce:
            out_shape[d % len(shape)] = 1
    else:
        for d in sorted([d % len(shape) for d in dims_to_reduce], reverse=True):
            out_shape.pop(d)
    return out_shape


def _sum_dim_impl(inp, dim, keepdim, dtype, out):
    """Unified implementation of single-dimension / multi-dimension sum, semantics consistent with the common
    implementation."""
    if dim is None or (isinstance(dim, (list, tuple)) and len(dim) == 0):
        result = _sum_dim_global(inp, keepdim, dtype, out)
        return result

    shape = list(inp.shape)
    ndim = inp.ndim
    dims = [d % ndim for d in dim]

    if len(dims) == 1:
        # Single dim: (M, N, K) 3D expansion, K is the element count after the reduced dimension.
        reduce_dim = dims[0]
        N = shape[reduce_dim]
        M = 1
        for s in shape[:reduce_dim]:
            M *= s
        K = inp.numel() // M // N
        inp = inp.contiguous()
        out_shape_keepdim = list(shape)
        out_shape_keepdim[reduce_dim] = 1
        out_shape = (
            out_shape_keepdim
            if keepdim
            else shape[:reduce_dim] + shape[reduce_dim + 1 :]
        )
        if out is None:
            out = torch.empty(out_shape, dtype=dtype, device=inp.device)
        else:
            out.resize_(out_shape)
        with torch_device_fn.device(inp.device):
            _launch_dim_sum(inp, out, M, N, K)
        return out

    # Multiple dims: compress the reduced dims to the end, degenerating into an (M, N) row reduction.
    inp = dim_compress(inp, dims)
    N = 1
    for d in dims:
        N *= shape[d]
        shape[d] = 1
    M = inp.numel() // N
    out_shape = (
        shape if keepdim else [s for i, s in enumerate(shape) if i not in set(dims)]
    )
    if out is None:
        out = torch.empty(out_shape, dtype=dtype, device=inp.device)
    else:
        out.resize_(out_shape)
    with torch_device_fn.device(inp.device):
        _launch_dim_sum(inp, out, M, N, 1)
    return out


def _sum_dim_global(inp, keepdim, dtype, out):
    """dim=None or dim=[]: degenerates to global sum."""
    if out is None:
        result = sum(inp, dtype=dtype)
        return result.reshape([1] * inp.ndim) if keepdim else result
    out.resize_([1] * inp.ndim if keepdim else [])
    return sum_out(inp, dtype=dtype, out=out)


def sum_dim(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS_ENFLAME SUM_DIM")

    inp, dtype = _normalize_dtype(inp, dtype)
    if inp.numel() == 0:
        return torch.zeros(
            _empty_out_shape(list(inp.shape), dim, keepdim),
            dtype=dtype,
            device=inp.device,
        )
    return _sum_dim_impl(inp, dim, keepdim, dtype, None)


def sum_dim_out(inp, dim=None, keepdim=False, *, dtype=None, out):
    logger.debug("GEMS_ENFLAME SUM_DIM_OUT")

    inp, dtype = _normalize_dtype(inp, dtype)
    if inp.numel() == 0:
        out.resize_(_empty_out_shape(list(inp.shape), dim, keepdim))
        out.zero_()
        return out
    return _sum_dim_impl(inp, dim, keepdim, dtype, out)
