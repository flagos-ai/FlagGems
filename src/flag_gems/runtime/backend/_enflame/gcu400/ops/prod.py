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

# GCU400 has 48 compute units; a fixed grid + in-kernel stride keeps SPs saturated.
MAX_GRID = 48

# Global reduction: small scale uses one program in one pass (matching sum.py's measured threshold).
GLOBAL_TILE_ELEMS = 65536

# Tile / accumulator shape of persistent global reduction: elementwise acc multiplies across tiles,
# with the reduction tree run only once at the finish.
GLOBAL_R = 8
GLOBAL_W = 1024

# Element budget of row-reduction acc2d: 128KB under fp32, controlling DSM usage (measured optimum).
INNER_ACC_ELEMS = 32768
INNER_MAX_BN = 4096

# Non-row-reduction tile: BI along the reduced dimension (tree depth), BK along the contiguous memory dimension.
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
def prod_global_kernel(
    X,
    Part,
    Counter,
    Out,
    n_elem,
    R: tl.constexpr,
    W: tl.constexpr,
    NP_BLOCK: tl.constexpr,
):
    """Persistent streaming global product: elementwise acc [R, W] multiplies across tiles, one final tree;
    last-block (acq_rel barrier) merges all partials within the same launch.

    Counter is an fp32 scalar (the tail element of the same buffer as Part), exact in fp32 up to 48;
    Out is written exclusively by the last program.
    """
    BLOCK: tl.constexpr = R * W
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
    num_tile = (n_elem + BLOCK - 1) // BLOCK
    acc = tl.full([R, W], 1, cdtype)
    for tile_id in tl.range(pid, num_tile, step, num_stages=1):
        offset = tile_id * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + offset, mask=offset < n_elem, other=1).to(cdtype)
        acc *= tl.reshape(x, (R, W))
    tl.store(Part + pid, tl.reduce(tl.reduce(acc, 1, _reduce_mul), 0, _reduce_mul))
    barrier = tl.atomic_add(Counter, 1.0, sem="acq_rel")
    if barrier == (tl.num_programs(0) - 1).to(tl.float32):
        offs = tl.arange(0, NP_BLOCK)
        part = tl.load(Part + offs, mask=offs < tl.num_programs(0), other=1.0).to(
            tl.float32
        )
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

        block = GLOBAL_R * GLOBAL_W
        num_tile = triton.cdiv(n_elem, block)
        grid = _min(MAX_GRID, num_tile)
        buf = torch.zeros(grid + 1, dtype=torch.float32, device=inp.device)
        prod_global_kernel[(grid,)](
            inp,
            buf[:grid],
            buf[grid:],
            out,
            n_elem,
            R=GLOBAL_R,
            W=GLOBAL_W,
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

            # Row reduction: acc2d element budget capped, row batches spread across 48 programs by row count.
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
