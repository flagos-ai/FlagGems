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

"""GCU400-specific prod family implementation.

Structurally aligned with sum.py's persistent fixed grid + tile byte budget in the same directory, with three key
changes for multiplicative reduction:

1. Multiplication has no atomic primitive (atomic_add cannot merge partial products), so cross-program
   merging uses a "last-block finish" single-launch mode: each program writes a partial then
   increments an acq_rel counter, and the last program merges all partials within the same launch
   (following the barrier precedent in masked_scatter_backward).
2. A multiplicative reduction tree costs far more than elementwise vector multiply (~an order of magnitude), so main
paths
   use 2D elementwise accumulators: the reduced dimension is expanded into vector multiply acc *= x by chunk, and the
   tree
   runs only once at each program's finish (inner/global), or degenerates into a narrow per-chunk tree
   (non-inner, tree depth = BI along the reduced dimension).
3. The main loop is mask-free (tail block separated by constexpr), so power-of-2 shapes have zero predicate cost.

Measured (L600/48SP, official bench core shapes kernel mode median w1000/r100,
job j20261111113840_b395bdf9, two fresh rounds geomean 1.355/1.309x, before change
0.405x):
- inner (4096,4096): 233->37us(fp16)/233->42us(fp32);
- non-inner (1024,1024,512): 2224->440us(fp16)/2393->965us(fp32);
- global (1048576,): 61->15us (single launch + per-call zeros counter).

Single-dimension reduction is routed by an (M, N, K) expansion: N==1 uses copy; K==1 and M==1 degenerates to global
reduction; K==1 with very few rows splits once more along N; K==1 uses row reduction; K>1 uses (row, tail-block)
task partitioning.
"""

import builtins
import logging
from functools import reduce as _reduce

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger("flag_gems.ops.prod")

_min = builtins.min
_max = builtins.max

# GCU400 has 48 compute units; a fixed grid plus in-kernel striding can keep the SPs fully occupied.
MAX_GRID = 48

# Global reduction: for small scales, a single program produces the result in one pass (aligned with the measured
# threshold in sum.py).
GLOBAL_TILE_ELEMS = 65536

# Hyperparameters for the two-stage (partial + merge) global reduction, bucketed by element width:
# For 2-byte (fp16/bf16) partials, use [1,W] element-wise accumulation; W=8192 is measured optimal
# (8.9µs@1M fp16); for 4-byte (fp32 and above) partials, use a per-tile 1D tree +
# scalar accumulation, W=16384 is measured optimal (9.8µs@1M fp32; under fp32 the DSM
# RMW of [1,W] is actually more expensive than the tree).
GLOBAL_PART_W_2B = 8192
GLOBAL_PART_GRID_2B = 48
GLOBAL_PART_W_4B = 16384
GLOBAL_PART_GRID_4B = 32

# Element budget for acc2d in row reduction: 128KB under fp32, to control DSM usage (measured optimal setting).
INNER_ACC_ELEMS = 32768
INNER_MAX_BN = 4096

# Wide-row tree path (N > 2048): the mul tree is on the fast path only when width >= 4096
# (BN=2048 measured +40%, and 1024 or below degrades by orders of magnitude), so BN is pinned to 4096;
# BM is bucketed by element width — for 2-byte tiles after fp32 promotion, BM=16 sits exactly within the DSM concurrency
# line
# and is 8% faster than BM=8; 4-byte tiles are already fp32, so BM=8 (128KB) is actually optimal.
INNER_TREE_BN = 4096
INNER_TREE_BM_2B = 16
INNER_TREE_BM_4B = 8
INNER_TREE_MIN_N = 2048

# Non-row-reduction tiles: BI runs along the reduced dimension (tree depth), BK along the memory-contiguous dimension.
NON_INNER_BI = 128
NON_INNER_BK = 512
NON_INNER_ACC_ELEMS = 65536

# When rows are few but rows are very long, split once more along N across 48 programs.
ROW_SPLIT_MAX_ROWS = 8
ROW_SPLIT_MIN_N = 65536
ROW_SPLIT_BN = 4096


@triton.jit
def _reduce_mul(a, b):
    return a * b


@libentry()
@triton.jit(do_not_specialize=["n_elem"])
def prod_copy_kernel(Out, X, n_elem, BLOCK: tl.constexpr):
    """Reduced dimension length is 1: output is the input itself, done with one elementwise copy."""
    pid = tl.program_id(0)
    step = tl.num_programs(0)
    num_tile = (n_elem + BLOCK - 1) // BLOCK
    for tile_id in tl.range(pid, num_tile, step, num_stages=1):
        offset = tile_id * BLOCK + tl.arange(0, BLOCK)
        mask = offset < n_elem
        x = tl.load(X + offset, mask=mask, other=0)
        tl.store(Out + offset, x.to(Out.dtype.element_ty), mask)


@libentry()
@triton.jit(do_not_specialize=["n_elem"])
def prod_global_single_kernel(X, Out, n_elem, BLOCK: tl.constexpr):
    """Small-scale global product: one program reduces in one pass, no partials and no counter."""
    if (
        tl.constexpr(X.dtype.element_ty == tl.float16)
        | tl.constexpr(X.dtype.element_ty == tl.bfloat16)
        | tl.constexpr(X.dtype.element_ty == tl.float64)
    ):
        cdtype = tl.float32
    else:
        cdtype = X.dtype.element_ty

    offset = tl.arange(0, BLOCK)
    x = tl.load(X + offset, mask=offset < n_elem, other=1).to(cdtype)
    tl.store(Out, tl.reduce(x, 0, _reduce_mul).to(Out.dtype.element_ty))


@libentry()
@triton.jit(do_not_specialize=["n_elem"])
def prod_global_partial_kernel(
    X,
    Part,
    n_elem,
    W: tl.constexpr,
):
    """Partial stage of the persistent streaming global product (measured best form for 2-byte types):

    Element-level acc [1, W] multiplies across tiles (reshape after a flat load, no cross-lane
    movement), with a single closing tree; each program writes one fp32 partial and the merge is
    done by a second launch (prod_global_merge_kernel), needing no counters and no per-call zeros.
    """
    if (
        tl.constexpr(X.dtype.element_ty == tl.float16)
        | tl.constexpr(X.dtype.element_ty == tl.bfloat16)
        | tl.constexpr(X.dtype.element_ty == tl.float64)
    ):
        cdtype = tl.float32
    else:
        cdtype = X.dtype.element_ty

    pid = tl.program_id(0)
    step = tl.num_programs(0)
    num_tile = (n_elem + W - 1) // W
    acc = tl.full([1, W], 1, cdtype)
    for tile_id in tl.range(pid, num_tile, step, num_stages=1):
        offset = tile_id * W + tl.arange(0, W)
        x = tl.load(X + offset, mask=offset < n_elem, other=1).to(cdtype)
        acc *= tl.reshape(x, (1, W))
    tl.store(Part + pid, tl.reduce(tl.reduce(acc, 1, _reduce_mul), 0, _reduce_mul))


@libentry()
@triton.jit(do_not_specialize=["n_elem"])
def prod_global_scalar_partial_kernel(
    X,
    Part,
    n_elem,
    W: tl.constexpr,
):
    """Partial stage of the persistent streaming global product (measured best form for 4-byte types):

    Each tile is collapsed to a scalar by a 1-D mul tree and the scalar acc multiplies across
    tiles -- for fp32 input the [1, W] element-level acc DSM RMW costs more than a per-tile tree.
    """
    if (
        tl.constexpr(X.dtype.element_ty == tl.float16)
        | tl.constexpr(X.dtype.element_ty == tl.bfloat16)
        | tl.constexpr(X.dtype.element_ty == tl.float64)
    ):
        cdtype = tl.float32
    else:
        cdtype = X.dtype.element_ty

    pid = tl.program_id(0)
    step = tl.num_programs(0)
    num_tile = (n_elem + W - 1) // W
    acc = tl.full([], 1, cdtype)
    for tile_id in tl.range(pid, num_tile, step, num_stages=1):
        offset = tile_id * W + tl.arange(0, W)
        x = tl.load(X + offset, mask=offset < n_elem, other=1).to(cdtype)
        acc *= tl.reduce(x, 0, _reduce_mul)
    tl.store(Part + pid, acc.to(Part.dtype.element_ty))


@libentry()
@triton.jit(do_not_specialize=["n_part"])
def prod_global_merge_kernel(
    Part,
    Out,
    n_part,
    NP_BLOCK: tl.constexpr,
):
    """Merge stage of the global product: one program multiplies the [n_part] fp32 partials into one value."""
    offs = tl.arange(0, NP_BLOCK)
    part = tl.load(Part + offs, mask=offs < n_part, other=1.0).to(tl.float32)
    tl.store(Out, tl.reduce(part, 0, _reduce_mul).to(Out.dtype.element_ty))


@libentry()
@triton.jit(do_not_specialize=["M", "N"])
def prod_dim_inner_kernel(
    X,
    Out,
    M,
    N,
    BM: tl.constexpr,
    BN: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    """Reduced dimension is innermost: persistent row-batch tile [BM, BN], acc2d elementwise multiply,
    one tree at each row tile's finish; main loop is mask-free (tail block separated by constexpr).
    """
    if (
        tl.constexpr(X.dtype.element_ty == tl.float16)
        | tl.constexpr(X.dtype.element_ty == tl.bfloat16)
        | tl.constexpr(X.dtype.element_ty == tl.float64)
    ):
        cdtype = tl.float32
    else:
        cdtype = X.dtype.element_ty

    pid = tl.program_id(0)
    step = tl.num_programs(0)
    num_tile = (M + BM - 1) // BM
    for tile_id in tl.range(pid, num_tile, step, num_stages=1):
        m_offset = tile_id * BM + tl.arange(0, BM)
        acc = tl.full([BM, BN], 1, cdtype)
        n_floor = (N // BN) * BN
        for n_start in tl.range(0, n_floor, BN, num_stages=1):
            n_offset = n_start + tl.arange(0, BN)
            if EVEN_M:
                x = tl.load(X + m_offset[:, None] * N + n_offset[None, :])
            else:
                x = tl.load(
                    X + m_offset[:, None] * N + n_offset[None, :],
                    mask=m_offset[:, None] < M,
                    other=1,
                )
            acc *= x.to(cdtype)
        if not EVEN_N:
            n_offset = n_floor + tl.arange(0, BN)
            tail_mask = n_offset[None, :] < N
            if EVEN_M:
                x = tl.load(
                    X + m_offset[:, None] * N + n_offset[None, :],
                    mask=tail_mask,
                    other=1,
                )
            else:
                x = tl.load(
                    X + m_offset[:, None] * N + n_offset[None, :],
                    mask=tail_mask & (m_offset[:, None] < M),
                    other=1,
                )
            acc *= x.to(cdtype)
        out = tl.reduce(acc, 1, _reduce_mul)
        if EVEN_M:
            tl.store(Out + m_offset, out.to(Out.dtype.element_ty))
        else:
            tl.store(Out + m_offset, out.to(Out.dtype.element_ty), mask=m_offset < M)


@libentry()
@triton.jit(do_not_specialize=["M", "N"])
def prod_dim_inner_tree_kernel(
    X,
    Out,
    M,
    N,
    BM: tl.constexpr,
    BN: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    """Row reduction for wide rows (N > 2048): each tile [BM, BN] runs a mul tree per reduction chunk,
    so acc degrades to scalar-level [BM].

    Key gain over the acc2d form: with a single block (N <= BN) acc2d's first multiply is a
    full-tile no-op multiply (acc filled with 1 then multiplied by x), while the tree form
    collapses each block into [BM] directly. A mul tree only lands on the fast path when the width
    is >= 4096 (2048 is +40%, 1024 and below are an order of magnitude worse), so BN is pinned at
    4096 and narrow rows skip this path. Tile fp32 footprint = BM*BN*4B, with BM tiered by
    element width to stay inside the DSM concurrency line (448KB)."""
    if (
        tl.constexpr(X.dtype.element_ty == tl.float16)
        | tl.constexpr(X.dtype.element_ty == tl.bfloat16)
        | tl.constexpr(X.dtype.element_ty == tl.float64)
    ):
        cdtype = tl.float32
    else:
        cdtype = X.dtype.element_ty

    pid = tl.program_id(0)
    step = tl.num_programs(0)
    num_tile = (M + BM - 1) // BM
    for tile_id in tl.range(pid, num_tile, step, num_stages=1):
        m_offset = tile_id * BM + tl.arange(0, BM)
        acc = tl.full([BM], 1, cdtype)
        n_floor = (N // BN) * BN
        for n_start in tl.range(0, n_floor, BN, num_stages=1):
            n_offset = n_start + tl.arange(0, BN)
            if EVEN_M:
                x = tl.load(X + m_offset[:, None] * N + n_offset[None, :])
            else:
                x = tl.load(
                    X + m_offset[:, None] * N + n_offset[None, :],
                    mask=m_offset[:, None] < M,
                    other=1,
                )
            acc *= tl.reduce(x.to(cdtype), 1, _reduce_mul)
        if not EVEN_N:
            n_offset = n_floor + tl.arange(0, BN)
            tail_mask = n_offset[None, :] < N
            if EVEN_M:
                x = tl.load(
                    X + m_offset[:, None] * N + n_offset[None, :],
                    mask=tail_mask,
                    other=1,
                )
            else:
                x = tl.load(
                    X + m_offset[:, None] * N + n_offset[None, :],
                    mask=tail_mask & (m_offset[:, None] < M),
                    other=1,
                )
            acc *= tl.reduce(x.to(cdtype), 1, _reduce_mul)
        if EVEN_M:
            tl.store(Out + m_offset, acc.to(Out.dtype.element_ty))
        else:
            tl.store(Out + m_offset, acc.to(Out.dtype.element_ty), mask=m_offset < M)


@libentry()
@triton.jit(do_not_specialize=["M", "N"])
def prod_dim_row_split_partial_kernel(
    X,
    Part,
    M,
    N,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    """When rows are few: split the in-row N dimension across programs, each producing one partial.

    acc2d [BM, BN] multiplies elementwise across chunks, one tree at each program's finish;
    partial layout is [n_part, BM], program i writes row i.
    """
    if (
        tl.constexpr(X.dtype.element_ty == tl.float16)
        | tl.constexpr(X.dtype.element_ty == tl.bfloat16)
        | tl.constexpr(X.dtype.element_ty == tl.float64)
    ):
        cdtype = tl.float32
    else:
        cdtype = X.dtype.element_ty

    pid = tl.program_id(0)
    step = tl.num_programs(0)
    m_offset = tl.arange(0, BM)
    row_mask = m_offset < M
    acc = tl.full([BM, BN], 1, cdtype)
    for col_start in tl.range(pid * BN, N, step * BN, num_stages=1):
        n_offset = col_start + tl.arange(0, BN)
        mask = row_mask[:, None] & (n_offset[None, :] < N)
        x = tl.load(
            X + m_offset[:, None] * N + n_offset[None, :], mask=mask, other=1
        ).to(cdtype)
        acc *= x
    out = tl.reduce(acc, 1, _reduce_mul)
    tl.store(Part + pid * BM + m_offset, out.to(Part.dtype.element_ty), row_mask)


@libentry()
@triton.jit(do_not_specialize=["M", "n_part"])
def prod_dim_row_split_final_kernel(
    Part,
    Out,
    M,
    n_part,
    BLOCK_P: tl.constexpr,
    BM: tl.constexpr,
):
    """Merge [n_part, BM] partials by row (multiplication)."""
    if tl.constexpr(Part.dtype.element_ty == tl.float16) | tl.constexpr(
        Part.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = Part.dtype.element_ty

    p_offset = tl.arange(0, BLOCK_P)[:, None]
    m_offset = tl.arange(0, BM)[None, :]
    mask = (p_offset < n_part) & (m_offset < M)
    part = tl.load(Part + p_offset * BM + m_offset, mask=mask, other=1).to(cdtype)
    out = tl.reduce(part, 0, _reduce_mul)
    m_flat = tl.arange(0, BM)
    tl.store(Out + m_flat, out.to(Out.dtype.element_ty), m_flat < M)


@libentry()
@triton.jit(do_not_specialize=["M", "N", "K"])
def prod_dim_non_inner_kernel(
    X,
    Out,
    M,
    N,
    K,
    BI: tl.constexpr,
    BK: tl.constexpr,
    EVEN_I: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """K elements follow the reduced dimension: task = (row m, tail block k), multiplying along the reduced dimension.

    Tile is (BI, BK), with BK along the tail (memory-contiguous) dimension so loads are several contiguous segments;
    each chunk runs one BI-deep narrow tree (columns fully parallel), acc [BK] retained along the tail dimension.
    Note: the tree must be in fp32 (or a wide integer type), so cast before reduce."""
    if (
        tl.constexpr(X.dtype.element_ty == tl.float16)
        | tl.constexpr(X.dtype.element_ty == tl.bfloat16)
        | tl.constexpr(X.dtype.element_ty == tl.float64)
    ):
        cdtype = tl.float32
    else:
        cdtype = X.dtype.element_ty

    pid = tl.program_id(0)
    step = tl.num_programs(0)
    num_k_tile = (K + BK - 1) // BK
    num_task = M * num_k_tile
    for task_id in tl.range(pid, num_task, step, num_stages=1):
        m_index = task_id // num_k_tile
        k_tile = task_id % num_k_tile
        k_offset = k_tile * BK + tl.arange(0, BK)
        base = m_index * N * K
        acc = tl.full([BK], 1, cdtype)
        i_floor = (N // BI) * BI
        for i_start in tl.range(0, i_floor, BI, num_stages=1):
            i_offset = i_start + tl.arange(0, BI)[:, None]
            if EVEN_K:
                x = tl.load(X + base + i_offset * K + k_offset[None, :])
            else:
                x = tl.load(
                    X + base + i_offset * K + k_offset[None, :],
                    mask=k_offset[None, :] < K,
                    other=1,
                )
            acc *= tl.reduce(x.to(cdtype), 0, _reduce_mul)
        if not EVEN_I:
            i_offset = i_floor + tl.arange(0, BI)[:, None]
            tail_mask = i_offset < N
            if EVEN_K:
                x = tl.load(
                    X + base + i_offset * K + k_offset[None, :], mask=tail_mask, other=1
                )
            else:
                x = tl.load(
                    X + base + i_offset * K + k_offset[None, :],
                    mask=tail_mask & (k_offset[None, :] < K),
                    other=1,
                )
            acc *= tl.reduce(x.to(cdtype), 0, _reduce_mul)
        if EVEN_K:
            tl.store(Out + m_index * K + k_offset, acc.to(Out.dtype.element_ty))
        else:
            tl.store(
                Out + m_index * K + k_offset,
                acc.to(Out.dtype.element_ty),
                mask=k_offset < K,
            )


def _launch_global_prod(inp, out, n_elem):
    """Global product: small scale uses one program; otherwise persistent streaming + last-block single launch.

    Counter and Part share one zeros buffer (counter occupies the tail element), so one
    memset clears both partials and the counter."""
    with torch_device_fn.device(inp.device):
        if n_elem <= GLOBAL_TILE_ELEMS:
            block = triton.next_power_of_2(n_elem)
            prod_global_single_kernel[(1,)](inp, out, n_elem, BLOCK=block, num_warps=2)
            return

        if inp.element_size() <= 2:
            part_w = GLOBAL_PART_W_2B
            grid_cap = GLOBAL_PART_GRID_2B
            partial_kernel = prod_global_partial_kernel
        else:
            part_w = GLOBAL_PART_W_4B
            grid_cap = GLOBAL_PART_GRID_4B
            partial_kernel = prod_global_scalar_partial_kernel
        num_tile = triton.cdiv(n_elem, part_w)
        grid = _min(grid_cap, num_tile)
        part = torch.empty([grid], dtype=torch.float32, device=inp.device)
        partial_kernel[(grid,)](inp, part, n_elem, W=part_w, num_warps=2)
        prod_global_merge_kernel[(1,)](
            part,
            out,
            grid,
            NP_BLOCK=triton.next_power_of_2(grid),
            num_warps=1,
        )


def _launch_dim_prod(inp, out, M, N, K):
    """Single-dimension product: routed by (M, N, K) shape.

    N == 1           -> output is input, elementwise copy;
    K == 1 and M == 1 -> degenerates to global product (reuse the single-launch global path);
    K == 1 with very few rows -> split once more along N, otherwise 1 program serializes the whole row;
    K == 1           -> row reduction (persistent [BM, BN] acc2d tile);
    K > 1            -> (row, tail-block) partitioning, narrow-tree multiply along the reduced dimension.
    """
    with torch_device_fn.device(inp.device):
        if N == 1:
            n_elem = inp.numel()
            block = _min(
                triton.next_power_of_2(n_elem),
                _max(1024, 262144 // inp.element_size()),
            )
            grid = _min(MAX_GRID, triton.cdiv(n_elem, block))
            prod_copy_kernel[(grid,)](
                out, inp, n_elem, BLOCK=block, num_warps=1 if block < 4096 else 4
            )
            return

        if K == 1:
            if M == 1:
                _launch_global_prod(inp, out, inp.numel())
                return
            if M <= ROW_SPLIT_MAX_ROWS and N >= ROW_SPLIT_MIN_N:
                block_m = triton.next_power_of_2(M)
                block_n = ROW_SPLIT_BN
                grid = _min(MAX_GRID, triton.cdiv(N, block_n))
                part = torch.empty(
                    [grid * block_m], dtype=torch.float32, device=inp.device
                )
                prod_dim_row_split_partial_kernel[(grid,)](
                    inp,
                    part,
                    M,
                    N,
                    BM=block_m,
                    BN=block_n,
                    num_warps=2,
                )
                prod_dim_row_split_final_kernel[(1,)](
                    part,
                    out,
                    M,
                    grid,
                    BLOCK_P=triton.next_power_of_2(grid),
                    BM=block_m,
                    num_warps=1,
                )
                return

            if N > INNER_TREE_MIN_N:
                # Wide-row tree path: the mul tree is on the fast path only when width >= 4096, BN pinned to 4096;
                # BM is bucketed by element width (for 2-byte after fp32 promotion, BM=16 sits exactly within the DSM
                # line,
                # 4-byte tiles are already fp32, so BM=8 is better).
                if inp.element_size() <= 2:
                    tree_bm = INNER_TREE_BM_2B
                else:
                    tree_bm = INNER_TREE_BM_4B
                block_m = _max(
                    1,
                    _min(
                        tree_bm,
                        triton.next_power_of_2(triton.cdiv(M, MAX_GRID)),
                    ),
                )
                grid = _min(MAX_GRID, triton.cdiv(M, block_m))
                prod_dim_inner_tree_kernel[(grid,)](
                    inp,
                    out,
                    M,
                    N,
                    BM=block_m,
                    BN=INNER_TREE_BN,
                    EVEN_M=(M % block_m == 0),
                    EVEN_N=(N % INNER_TREE_BN == 0),
                    num_warps=4,
                )
                return

            # Row reduction: cap the acc2d element budget, and spread row batches across 48 programs according to the
            # number of rows.
            block_n = _min(triton.next_power_of_2(N), INNER_MAX_BN)
            block_m = _max(
                1,
                _min(
                    INNER_ACC_ELEMS // block_n,
                    triton.next_power_of_2(triton.cdiv(M, MAX_GRID)),
                ),
            )
            grid = _min(MAX_GRID, triton.cdiv(M, block_m))
            prod_dim_inner_kernel[(grid,)](
                inp,
                out,
                M,
                N,
                BM=block_m,
                BN=block_n,
                EVEN_M=(M % block_m == 0),
                EVEN_N=(N % block_n == 0),
                num_warps=4 if block_m * block_n >= 8192 else 2,
            )
            return

        # Tail-block width fixed at 512: measured [BI, BK] tile tl.reduce(axis=0) falls into a ~55 Melem/s
        # slow path when BK <= 128 (30x), BK = 256 is intermittently
        # pathological, and only BK >= 512 stays on the fast path; prefer fewer than 48 tasks over
        # halving again (halving only creates pathological tiers).
        block_k = _min(triton.next_power_of_2(K), NON_INNER_BK)
        block_i = _max(
            1,
            _min(
                triton.next_power_of_2(N),
                NON_INNER_ACC_ELEMS // block_k,
                NON_INNER_BI,
            ),
        )
        num_task = M * triton.cdiv(K, block_k)
        grid = _min(MAX_GRID, num_task)
        prod_dim_non_inner_kernel[(grid,)](
            inp,
            out,
            M,
            N,
            K,
            BI=block_i,
            BK=block_k,
            EVEN_I=(N % block_i == 0),
            EVEN_K=(K % block_k == 0),
            num_warps=4,
        )


def _normalize_dtype(inp, dtype):
    """Keep dtype normalization consistent with sum.py (bool goes through integer)."""
    if dtype is None:
        dtype = inp.dtype
        if dtype == torch.bool:
            inp = inp.to(torch.int32)
            dtype = torch.int32
    return inp, dtype


def prod(inp, *, dtype=None):
    logger.debug("GEMS_ENFLAME PROD")

    inp, dtype = _normalize_dtype(inp, dtype)
    inp = inp.contiguous()
    n_elem = inp.numel()

    out = torch.empty([], dtype=dtype, device=inp.device)
    if n_elem == 0:
        out.fill_(1)
        return out

    _launch_global_prod(inp, out, n_elem)
    return out


def prod_dim(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS_ENFLAME PROD_DIM")

    if not (-inp.ndim <= dim < inp.ndim):
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-inp.ndim}, {inp.ndim - 1}])"
        )
    inp, dtype = _normalize_dtype(inp, dtype)

    shape = list(inp.shape)
    d = dim % inp.ndim
    N = shape[d]
    M = _reduce(lambda x, y: x * y, shape[:d], 1)
    K = _reduce(lambda x, y: x * y, shape[d + 1 :], 1)
    shape[d] = 1
    out = torch.empty(shape, dtype=dtype, device=inp.device)

    if M == 0 or K == 0:
        # A spectator dimension is empty: the output is a valid empty tensor.
        if not keepdim:
            out = torch.squeeze(out, d)
        return out
    if N == 0:
        # The product over an empty dimension is the identity, 1 (unlike max/min
        # which have no identity). Fill the output rather than launching.
        out.fill_(1)
        if not keepdim:
            out = torch.squeeze(out, d)
        return out

    inp = inp.contiguous()
    with torch_device_fn.device(inp.device):
        _launch_dim_prod(inp, out, M, N, K)
    if not keepdim:
        out = torch.squeeze(out, d)
    return out
