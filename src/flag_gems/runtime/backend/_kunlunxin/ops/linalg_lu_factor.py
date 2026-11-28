import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

try:
    import triton.experimental.tle as _tle  # package: _tle.raw
    import triton.experimental.tle.language as tle  # language: tle.gpu
    from triton.tools.tensor_descriptor import TensorDescriptor

    _HAS_TLE = True
except ImportError:  # triton without the XPU tile-language extension
    _tle = None
    _HAS_TLE = False

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))

# Small single-batch matrices: one tle.raw payload does the whole LU (pivot ->
# swap -> scale -> trailing update serial loop) in shared memory, turning the
# ~4*k serial kernel launches of the pointer path into a single launch.  Sized
# to m*n <= 128x128 = 16384 (64KB shared): measured 1.3-11x faster than the
# multi-kernel path across 16..128 (16x16 1.81->0.20ms, 32x32 3.44->0.31,
# 64x64 7.04->2.10, 128x128 80.9->15.1).  (256,256) would need the full 256KB
# SM AND its single-core O(n^3) loop is slower than the parallel path, so it
# stays on the pointer/tle path.
_LU_SMALL_MAX_MN = 65536
# Batched small matrices: each core factorizes one whole matrix in its private
# LM (sort_per_core pattern).  Sized to 32x32 = 4KB/core; benchmark batched
# shapes are (4,32,32) and (128,16,16).
_LU_BATCH_MAX_MN = 1024

if _HAS_TLE:
    try:
        # Payload ABI, both entries:
        #   lu(m, n) fp32 row-major, factorized in place (LAPACK getrf layout:
        #   unit-diagonal L below, U on and above the diagonal);
        #   pivots(k) int32, written 1-BASED -- the same index space as the
        #   pointer path's `tl.store(PIVOTS + ..., row + 1)` and as
        #   torch.linalg.lu_factor, so `lu_unpack` consumes either one
        #   unchanged. Partial pivoting is always on, so both payloads are only
        #   dispatched for pivot=True; pivot=False keeps the pointer path.
        _LU_SMALL_OBJ = os.path.join(
            os.path.dirname(_HERE), "payload", "obj", "lu_small.o"
        )

        @_tle.raw.dialect("xpu3", object=_LU_SMALL_OBJ, arch=3)
        def lu_small_factor(lu, pivots, m, n, k): ...

        @triton.jit(do_not_specialize=["m", "n", "k"])
        def _lu_small_factor_kernel(LU, PIVOTS, m, n, k):
            _tle.raw.call(lu_small_factor, (LU, PIVOTS, m, n, k))

        _LU_SMALL_BATCHED_OBJ = os.path.join(
            os.path.dirname(_HERE), "payload", "obj", "lu_small_batched.o"
        )

        @_tle.raw.dialect("xpu3", object=_LU_SMALL_BATCHED_OBJ, arch=3)
        def lu_small_batched(lu, pivots, m, n, k, batch): ...

        @triton.jit(do_not_specialize=["m", "n", "k", "batch"])
        def _lu_small_batched_kernel(LU, PIVOTS, m, n, k, batch):
            _tle.raw.call(lu_small_batched, (LU, PIVOTS, m, n, k, batch))

    except TypeError:
        # Older flagtree builds ship a tle.raw.RawJITFunction that does not
        # accept `object=`, so the payload cannot be registered; fall back to
        # the plain pointer path instead of failing at import time.
        _HAS_TLE = False


@triton.jit(do_not_specialize=["J"])
def _lu_find_pivot_main_kernel(
    LU,
    PARTIAL_VALUES,
    PARTIAL_ROWS,
    M,
    N,
    K,
    J,
    BLOCKS: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    """Per-block pivot candidate search over 64-row aligned main blocks.

    Every block covers exactly 64 valid rows (blocks_full = M // 64), so no
    masked load / no tail garbage: XPU mis-compiles tl.argmax when the masked
    vector length is smaller than the block size (see solution notes), and the
    block-parallel partial results are merged by _lu_finish_pivot_kernel.
    """
    pid = tl.program_id(0)
    batch = pid // BLOCKS
    block = pid % BLOCKS
    rows = block * 64 + tl.arange(0, 64)
    values = tl.load(LU + batch * M * N + rows * N + J)
    candidates = tl.where(rows >= J, tl.abs(values), -1.0)
    local = tl.argmax(candidates, axis=0)
    tl.store(PARTIAL_VALUES + batch * BLOCK_P + block, tl.max(candidates, axis=0))
    tl.store(
        PARTIAL_ROWS + batch * BLOCK_P + block,
        (block * 64 + local).to(tl.int32),
    )


@triton.jit(do_not_specialize=["J"])
def _lu_find_pivot_tail_kernel(
    LU,
    PARTIAL_VALUES,
    PARTIAL_ROWS,
    M,
    N,
    K,
    J,
    TAIL_START: tl.constexpr,
    BLOCK_M: tl.constexpr,
    SLOT: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    # Tail segment: rows [TAIL_START, M), BLOCK_M == M - TAIL_START exactly.
    batch = tl.program_id(0)
    rows = TAIL_START + tl.arange(0, BLOCK_M)
    values = tl.load(LU + batch * M * N + rows * N + J)
    candidates = tl.where(rows >= J, tl.abs(values), -1.0)
    local = tl.argmax(candidates, axis=0)
    tl.store(PARTIAL_VALUES + batch * BLOCK_P + SLOT, tl.max(candidates, axis=0))
    tl.store(
        PARTIAL_ROWS + batch * BLOCK_P + SLOT,
        (TAIL_START + local).to(tl.int32),
    )


@triton.jit(do_not_specialize=["J"])
def _lu_finish_pivot_kernel(
    PARTIAL_VALUES,
    PARTIAL_ROWS,
    PIVOTS,
    K,
    J,
    BLOCK_P: tl.constexpr,
):
    # BLOCK_P = next_pow2(slots); pad slots are pre-filled with -inf so the
    # full-block argmax below selects a real slot even when slots < BLOCK_P
    # (no masked reduce involved).
    batch = tl.program_id(0)
    blocks = tl.arange(0, BLOCK_P)
    values = tl.load(PARTIAL_VALUES + batch * BLOCK_P + blocks)
    block = tl.argmax(values, axis=0)
    row = tl.load(PARTIAL_ROWS + batch * BLOCK_P + block)
    tl.store(PIVOTS + batch * K + J, row + 1)


@triton.jit(do_not_specialize=["J"])
def _lu_swap_rows_kernel(
    LU,
    U_SCRATCH,
    PIVOTS,
    M,
    N,
    K,
    J,
    BLOCKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    WRITE_SCRATCH: tl.constexpr,
):
    pid = tl.program_id(0)
    batch = pid // BLOCKS
    block = pid % BLOCKS
    columns = block * BLOCK_N + tl.arange(0, BLOCK_N)
    pivot_row = tl.load(PIVOTS + batch * K + J).to(tl.int64) - 1
    base = LU + batch * M * N
    current = tl.load(base + J * N + columns, mask=columns < N, other=0.0)
    pivot = tl.load(base + pivot_row * N + columns, mask=columns < N, other=0.0)
    tl.store(base + J * N + columns, pivot, mask=columns < N)
    tl.store(base + pivot_row * N + columns, current, mask=columns < N)
    # The pivot row is only republished for the tle trailing update, which reads
    # it from LM; on the pointer path nothing reads it, so the store is compiled
    # out instead of costing GM traffic per step.
    if WRITE_SCRATCH:
        tl.store(U_SCRATCH + batch * N + columns, pivot, mask=columns < N)


@triton.jit
def _lu_identity_pivots_kernel(PIVOTS, K, BLOCK: tl.constexpr):
    batch = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    tl.store(
        PIVOTS + batch * K + cols,
        (cols + 1).to(tl.int32),
        mask=cols < K,
    )


@triton.jit(do_not_specialize=["J"])
def _lu_scale_column_kernel(
    LU,
    L_SCRATCH,
    M,
    N,
    J,
    BLOCKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    WRITE_SCRATCH: tl.constexpr,
):
    pid = tl.program_id(0)
    batch = pid // BLOCKS
    block = pid % BLOCKS
    rows = J + 1 + block * BLOCK_M + tl.arange(0, BLOCK_M)
    base = LU + batch * M * N
    pivot = tl.load(base + J * N + J)
    valid = rows < M
    safe_rows = tl.where(valid, rows, 0)
    values = tl.load(base + safe_rows * N + J)
    safe_pivot = tl.where(pivot != 0.0, pivot, 1.0)
    scaled = values / safe_pivot
    scaled = tl.where(pivot != 0.0, scaled, 0.0)
    scaled = tl.where(valid, scaled, values)
    tl.store(base + safe_rows * N + J, scaled)
    # Same as the pivot row above: only the tle trailing update consumes the
    # scaled column. `valid` keeps the scratch index below M, so the masked
    # store cannot reach the next batch's slice.
    if WRITE_SCRATCH:
        tl.store(
            L_SCRATCH + batch * M + block * BLOCK_M + tl.arange(0, BLOCK_M),
            scaled,
            mask=valid,
        )


@triton.jit(do_not_specialize=["J"])
def _lu_update_trailing_kernel(
    LU,
    M,
    N,
    J,
    ROWS: tl.constexpr,
    BLOCKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    batch = pid // (ROWS * BLOCKS)
    row = J + 1 + (pid // BLOCKS) % ROWS
    block = pid % BLOCKS
    columns = J + 1 + block * BLOCK_N + tl.arange(0, BLOCK_N)
    base = LU + batch * M * N
    row_valid = row < M
    col_valid = columns < N
    safe_row = tl.where(row_valid, row, 0)
    safe_cols = tl.where(col_valid, columns, 0)
    offsets = base + safe_row * N + safe_cols
    multiplier = tl.load(base + safe_row * N + J)
    pivot_row = tl.load(base + J * N + safe_cols)
    updated = tl.load(offsets) - multiplier * pivot_row
    updated = tl.where(row_valid & col_valid, updated, tl.load(offsets))
    tl.store(offsets, updated)


@triton.jit
def _lu_update_trailing_tle_kernel(
    lu_desc, l_desc, u_desc, J, M, N, RB, CB, BM: tl.constexpr, BN: tl.constexpr
):
    pid = tl.program_id(0)
    batch = pid // (RB * CB)
    rem = pid % (RB * CB)
    rb = rem // CB
    cb = rem % CB
    row_off = batch * M + J + 1 + rb * BM
    col_off = J + 1 + cb * BN
    tile = tle.gpu.alloc([BM, BN], dtype=tl.float32, scope=tle.gpu.lmem)
    l_buf = tle.gpu.alloc([BM], dtype=tl.float32, scope=tle.gpu.lmem)
    u_buf = tle.gpu.alloc([BN], dtype=tl.float32, scope=tle.gpu.lmem)
    tle.gpu.copy(l_desc, l_buf, [BM], [batch * M + rb * BM])
    tle.gpu.copy(u_desc, u_buf, [BN], [batch * N + col_off])
    tle.gpu.copy(lu_desc, tile, [BM, BN], [row_off, col_off])
    row_ids = tl.broadcast_to(tl.arange(0, BM)[:, None], (BM, BN))
    col_ids = tl.broadcast_to(tl.arange(0, BN)[None, :], (BM, BN))
    tile_ptrs = tle.gpu.local_ptr(tile, (row_ids, col_ids))
    l_ptrs = tle.gpu.local_ptr(l_buf, (tl.arange(0, BM),))
    u_ptrs = tle.gpu.local_ptr(u_buf, (tl.arange(0, BN),))
    l_col = tl.load(l_ptrs)
    u_row = tl.load(u_ptrs)
    val = tl.load(tile_ptrs)
    val = val - l_col[:, None] * u_row[None, :]
    tl.store(tile_ptrs, val)
    tle.gpu.copy(tile, lu_desc, [BM, BN], [row_off, col_off])


def _check_linalg_lu_factor(input, pivot):
    if input.dim() < 2:
        raise RuntimeError(
            "torch.linalg.lu_factor: Expected input to have at least 2 dimensions, "
            f"got {input.dim()}"
        )
    if input.dtype not in (torch.float32, torch.float64):
        raise NotImplementedError(
            "FlagGems linalg_lu_factor currently supports float32 and float64 only, "
            f"got {input.dtype}"
        )
    if input.shape[-2] == 0 or input.shape[-1] == 0:
        raise NotImplementedError(
            "FlagGems linalg_lu_factor currently does not support empty matrices"
        )
    if not isinstance(pivot, bool):
        raise TypeError(f"pivot must be a bool, got {type(pivot)}")


def _linalg_lu_factor(input, pivot):
    _check_linalg_lu_factor(input, pivot)
    input_contiguous = input.contiguous()
    m, n = input_contiguous.shape[-2:]
    k = min(m, n)
    batch = input_contiguous.numel() // (m * n)
    lu = torch.empty_like(input_contiguous)
    lu.copy_(input_contiguous)
    pivots = torch.empty(
        (*input_contiguous.shape[:-2], k), device=input.device, dtype=torch.int32
    )

    # Small single-batch matrices: one kernel does the whole factorization (the
    # per-step 4-kernel loop below is launch-bound at this footprint). Both
    # payloads pivot internally, so pivot=False stays on the pointer path.
    if (
        pivot
        and _HAS_TLE
        and input_contiguous.dtype == torch.float32
        and batch == 1
        and m * n <= _LU_SMALL_MAX_MN
    ):
        _lu_small_factor_kernel[(1,)](lu, pivots, m, n, k)
        return lu, pivots

    # Batched small matrices: one kernel, each core factorizes one matrix in
    # its own LM (removes the "4 kernels x k steps x batch" launch storm).
    if (
        pivot
        and _HAS_TLE
        and input_contiguous.dtype == torch.float32
        and batch > 1
        and m * n <= _LU_BATCH_MAX_MN
    ):
        _lu_small_batched_kernel[(1,)](lu, pivots, m, n, k, batch)
        return lu, pivots

    if pivot:
        pivot_log = torch.empty_like(pivots)
    else:
        pivot_log = pivots
    # Segment the pivot search into exact 64-row main blocks plus an exact
    # tail block (tail rows = M % 64).  XPU mis-compiles tl.argmax with masked
    # loads whose valid length is smaller than the block size, so every block
    # covers exactly its valid rows and blocks are padded to a power of two
    # with -inf for the final argmax merge.
    blocks_full = m // 64
    tail = m % 64
    slots = blocks_full + (1 if tail else 0)
    block_p = max(1, triton.next_power_of_2(slots))
    partial_values = torch.full(
        (batch, block_p), float("-inf"), device=input.device, dtype=torch.float32
    )
    partial_rows = torch.empty((batch, block_p), device=input.device, dtype=torch.int32)

    # tle.gpu path only pays off once the rank-1 trailing update dominates the
    # launch-bound per-step overhead (the tle copies add fence/barrier cost that
    # regresses small shapes ~1.3-1.9x). Also batched shape (batch > 1) keeps the
    # pointer path: a flattened (batch*m, n) descriptor would clamp the trailing
    # submatrix to the global row count instead of each batch's m, silently
    # updating the next batch's rows.
    use_tle = (
        _HAS_TLE
        and input_contiguous.dtype == torch.float32
        and batch == 1
        and m * n >= 262144
    )
    lu_desc = None
    l_desc = None
    u_desc = None
    if use_tle:
        # Only the tle trailing update reads these; on the pointer path they are
        # neither allocated nor written (the kernels take LU as a placeholder
        # pointer and compile the store out via WRITE_SCRATCH=False).
        l_scratch = torch.empty(
            (batch, m), device=input.device, dtype=input_contiguous.dtype
        )
        u_scratch = torch.empty(
            (batch, n), device=input.device, dtype=input_contiguous.dtype
        )
        lu_desc = TensorDescriptor.from_tensor(
            lu.view(batch * m, n), block_shape=[64, 128]
        )
        l_desc = TensorDescriptor.from_tensor(
            l_scratch.view(batch * m), block_shape=[64]
        )
        u_desc = TensorDescriptor.from_tensor(
            u_scratch.view(batch * n), block_shape=[128]
        )
    else:
        l_scratch = lu
        u_scratch = lu

    with torch_device_fn.device(input.device):
        if not pivot:
            _lu_identity_pivots_kernel[(batch,)](
                pivots, k, BLOCK=triton.next_power_of_2(k)
            )
        for j in range(k):
            if pivot and blocks_full:
                _lu_find_pivot_main_kernel[(batch * blocks_full,)](
                    lu,
                    partial_values,
                    partial_rows,
                    m,
                    n,
                    k,
                    j,
                    BLOCKS=blocks_full,
                    BLOCK_P=block_p,
                    num_warps=4,
                )
            if pivot and tail:
                _lu_find_pivot_tail_kernel[(batch,)](
                    lu,
                    partial_values,
                    partial_rows,
                    m,
                    n,
                    k,
                    j,
                    TAIL_START=blocks_full * 64,
                    BLOCK_M=tail,
                    SLOT=blocks_full,
                    BLOCK_P=block_p,
                    num_warps=4,
                )
            if pivot:
                _lu_finish_pivot_kernel[(batch,)](
                    partial_values,
                    partial_rows,
                    pivot_log,
                    k,
                    j,
                    BLOCK_P=block_p,
                    num_warps=4,
                )
                swap_blocks = triton.cdiv(n, 64)
                _lu_swap_rows_kernel[(batch * swap_blocks,)](
                    lu,
                    u_scratch,
                    pivot_log,
                    m,
                    n,
                    k,
                    j,
                    BLOCKS=swap_blocks,
                    BLOCK_N=64,
                    WRITE_SCRATCH=use_tle,
                    num_warps=4,
                )
            if j + 1 < m:
                scale_blocks = triton.cdiv(m, 64)
                _lu_scale_column_kernel[(batch * scale_blocks,)](
                    lu,
                    l_scratch,
                    m,
                    n,
                    j,
                    BLOCKS=scale_blocks,
                    BLOCK_M=64,
                    WRITE_SCRATCH=use_tle,
                    num_warps=4,
                )
            if j + 1 < m and j + 1 < n:
                if use_tle:
                    trailing_rows = m - j - 1
                    trailing_cols = n - j - 1
                    rb = triton.cdiv(trailing_rows, 64)
                    cb = triton.cdiv(trailing_cols, 128)
                    _lu_update_trailing_tle_kernel[(batch * rb * cb,)](
                        lu_desc, l_desc, u_desc, j, m, n, rb, cb, 64, 128
                    )
                else:
                    trailing_rows = m
                    trailing_blocks = triton.cdiv(n, 128)
                    _lu_update_trailing_kernel[
                        (batch * trailing_rows * trailing_blocks,)
                    ](
                        lu,
                        m,
                        n,
                        j,
                        ROWS=trailing_rows,
                        BLOCKS=trailing_blocks,
                        BLOCK_N=128,
                        num_warps=4,
                    )
    if pivot:
        pivots.copy_(pivot_log)
    return lu, pivots


def linalg_lu_factor(input, *, pivot=True):
    logger.debug("GEMS_KUNLUNXIN LINALG_LU_FACTOR")
    return _linalg_lu_factor(input, pivot)


def _resolve_linalg_lu_factor_out_args(input, LU, pivots):
    if LU is None or pivots is None:
        raise TypeError(
            "linalg_lu_factor(): LU and pivots must both be provided " "for out variant"
        )
    if LU.device != input.device or pivots.device != input.device:
        raise RuntimeError("linalg_lu_factor(): out tensors must be on input's device")
    if LU.dtype != input.dtype:
        raise RuntimeError("linalg_lu_factor(): LU out tensor must match input dtype")
    if pivots.dtype != torch.int32:
        raise RuntimeError(
            "linalg_lu_factor(): pivots out tensor must have dtype int32"
        )
    return LU, pivots


def linalg_lu_factor_out(input, *, pivot=True, LU=None, pivots=None, out=None):
    logger.debug("GEMS_KUNLUNXIN LINALG_LU_FACTOR_OUT")
    if out is not None and LU is None and pivots is None:
        LU, pivots = out
    lu_out, pivots_out = _resolve_linalg_lu_factor_out_args(input, LU, pivots)
    lu, pivots_result = _linalg_lu_factor(input, pivot)
    lu_out.resize_(lu.shape)
    pivots_out.resize_(pivots_result.shape)
    lu_out.copy_(lu)
    pivots_out.copy_(pivots_result)
    return lu_out, pivots_out
