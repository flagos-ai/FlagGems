# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

BIG = tl.constexpr(1073741824)
INF = tl.constexpr(float("inf"))


@libentry()
@triton.jit
def _kthvalue_packed_kernel(
    input_ptr,
    value_ptr,
    index_ptr,
    N,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MAXC: tl.constexpr,
):
    """K-th smallest selection, one row per program (grid = M).

    Fast path: the fp32 value is turned into an IEEE-order-preserving 32-bit
    key and packed together with the column index into a single int64 lane,
    so one int64 tl.reduce(min) returns both the min value and its index at
    once.  Ranks are peeled sequentially inside the kernel (exclude the
    previous winner with a saturated key, see below), so k rounds reuse the
    same loaded tile.

    Elimination must not use ``tl.where(cond, big, keys)``: a second int64
    ``tl.min`` after an int64 ``tl.where`` is mis-compiled by this backend
    (returns a value identical for every program, see harness records).  The
    winner is instead raised to the saturated maximum arithmetically::

        keys + (col == idx) * (MAXKEY - keys)

    which is exact for every input (no int64 overflow: the result is at most
    MAXKEY) and compiles to compare/subtract/multiply/add, not a select.

    Loading: the reduction dim is padded to a power of two with ``+inf`` on
    the host, so the whole tile is always a contiguous, affine load
    (``base + col_offs``).  A wrapped ``col_offs % N`` index would scalarize
    the load and is ~1000x slower on this backend (measured: 96.9s vs 0.087s
    for a (200, 40999, 3) dim=0 reduction); masked tail loads are silently
    dropped / unreliable on this backend.  The padding lanes carry ``+inf``
    and are excluded from every rank by the logical ``col_offs < N`` masks
    below, so the peeled order statistics are exactly those of the logical
    row.

    Ties: the final index torch reports for a repeated k-th value is
    algorithm-dependent (quick_select_template in
    aten/src/ATen/native/Sorting.cpp) and NOT the smallest index of a tied
    value.  ``cnt`` (occurrences of the k-th value in the logical row) is
    therefore computed; rows with ``cnt > 1`` additionally run an exact
    register-resident port of quick_select (below), which reproduces torch's
    index.  The simulation is data-dependent: ``sim_bound`` is 0 for rows
    whose k-th value is unique, so the fast path above is all an untied row
    does.  The exact sim must live at kernel top level (wrapping the
    tt.reduce ops in an ``if`` is illegal on this backend), hence the
    dynamic-bound ``tl.range`` with ``isCloseUnrollControl``.
    """
    pid = ext.program_id(0)
    col_offs = tl.arange(0, BLOCK_N)
    base = input_ptr + pid * BLOCK_N
    pos = col_offs
    v = tl.load(base + col_offs)
    # IEEE fp32 -> total-order int32 key (bitcast + sign flip trick)
    b = v.to(tl.int32, bitcast=True)
    neg = b >> 31  # -1 if negative else 0
    u = b ^ (neg | -2147483648)
    # pack (key, index) into int64: shift key into signed range so the
    # int64 comparison keeps value ordering and breaks ties by index
    keys = (((u.to(tl.int64) & 0xFFFFFFFF) - 2147483648) << 32) | pos.to(tl.int64)
    MAXKEY = (1 << 63) - 1  # saturated int64 maximum (elimination target)
    for ki in tl.static_range(K):
        m = tl.min(keys, axis=0)
        if ki != K - 1:
            idx = m.to(tl.int32)  # low 32 bits carry the index
            sel = ((pos.to(tl.int64) & 0xFFFFFFFF) == idx.to(tl.int64)).to(tl.int64)
            # arithmetic select: raise the winner to the saturated maximum
            keys = keys + sel * (MAXKEY - keys)
    idx = m.to(tl.int32)
    # unpack value: high 32 bits -> key -> IEEE bits -> fp32
    key = ((m >> 32) + 2147483648).to(tl.int32)
    neg = ~(key >> 31)
    bits = key ^ (neg | -2147483648)
    kth_val = bits.to(tl.float32, bitcast=True)
    cnt = tl.sum(((v == kth_val) & (col_offs < N)).to(tl.int32), axis=0)
    kth = K - 1
    idxv = col_offs
    sim_bound = tl.where(cnt > 1, MAXC, 0)
    # ---------- exact quick_select_port (register-resident) ----------
    L = tl.full((), 0, dtype=tl.int32)
    R = tl.full((), N - 1, dtype=tl.int32)
    i = tl.full((), 0, dtype=tl.int32)
    j = tl.full((), 0, dtype=tl.int32)
    stage = tl.full((), 0, dtype=tl.int32)
    done = tl.full((), 0, dtype=tl.int32)
    piv = tl.full((), 0.0, dtype=tl.float32)
    i_at_L = tl.full((), 0, dtype=tl.int32)
    for _ in tl.range(0, sim_bound):
        s0 = ((stage == 0) & (done == 0)).to(tl.int32)
        s1 = ((stage == 1) & (done == 0)).to(tl.int32)
        P = L + (R - L) // 2
        mskL = (col_offs == L) & (col_offs < N)
        mskR = (col_offs == R) & (col_offs < N)
        mskL1 = (col_offs == L + 1) & (col_offs < N)
        mskP = (col_offs == P) & (col_offs < N)
        vL = tl.min(tl.where(mskL, v, INF), axis=0)
        vR = tl.min(tl.where(mskR, v, INF), axis=0)
        vL1 = tl.min(tl.where(mskL1, v, INF), axis=0)
        vP = tl.min(tl.where(mskP, v, INF), axis=0)
        iL = tl.min(tl.where(mskL, idxv, BIG), axis=0)
        iR = tl.min(tl.where(mskR, idxv, BIG), axis=0)
        iL1 = tl.min(tl.where(mskL1, idxv, BIG), axis=0)
        iP = tl.min(tl.where(mskP, idxv, BIG), axis=0)
        c1 = vP > vR
        vL1s = tl.where(c1, vR, vP)
        iL1s = tl.where(c1, iR, iP)
        vRs = tl.where(c1, vP, vR)
        iRs = tl.where(c1, iP, iR)
        c2 = vL > vRs
        vLs = tl.where(c2, vRs, vL)
        iLs = tl.where(c2, iRs, iL)
        vRs2 = tl.where(c2, vL, vRs)
        iRs2 = tl.where(c2, iL, iRs)
        c3 = vL1s > vLs
        vL1s2 = tl.where(c3, vLs, vL1s)
        iL1s2 = tl.where(c3, iLs, iL1s)
        vLs2 = tl.where(c3, vL1s, vLs)
        iLs2 = tl.where(c3, iL1s, iLs)
        p0c = s0 & (R >= L + 2)
        v = tl.where(mskP & (p0c != 0), vL1, v)
        v = tl.where(mskL1 & (p0c != 0), vL1s2, v)
        v = tl.where(mskL & (p0c != 0), vLs2, v)
        v = tl.where(mskR & (p0c != 0), vRs2, v)
        idxv = tl.where(mskP & (p0c != 0), iL1, idxv)
        idxv = tl.where(mskL1 & (p0c != 0), iL1s2, idxv)
        idxv = tl.where(mskL & (p0c != 0), iLs2, idxv)
        idxv = tl.where(mskR & (p0c != 0), iRs2, idxv)
        i = tl.where(p0c != 0, L + 1, i)
        j = tl.where(p0c != 0, R, j)
        piv = tl.where(p0c != 0, vLs2, piv)
        i_at_L = tl.where(p0c != 0, iLs2, i_at_L)
        stage = tl.where(p0c != 0, 1, stage)
        # R == L + 1: swap the pair if out of order, then done
        p0b = s0 & (R == L + 1)
        sw = (p0b != 0) & (vL > vR)
        v = tl.where(mskL & sw, vR, v)
        v = tl.where(mskR & sw, vL, v)
        idxv = tl.where(mskL & sw, iR, idxv)
        idxv = tl.where(mskR & sw, iL, idxv)
        done = tl.where(p0b != 0, 1, done)
        done = tl.where(s0 & (R <= L), 1, done)
        # phase 1: Hoare scans (min/max over masked positions)
        mskI = (col_offs > i) & (col_offs <= R) & (col_offs < N) & (v >= piv)
        mskJ = (col_offs < j) & (col_offs >= L) & (col_offs < N) & (v <= piv)
        i_s = tl.min(tl.where(mskI, col_offs, BIG), axis=0)
        j_s = tl.max(tl.where(mskJ, col_offs, -1), axis=0)
        mskj = (col_offs == j_s) & (col_offs < N)
        vJ = tl.min(tl.where(mskj, v, INF), axis=0)
        iJ = tl.min(tl.where(mskj, idxv, BIG), axis=0)
        less = s1 & (j_s < i_s)
        # partition done: drop pivot (L <-> j_s), shrink [L, R]
        v = tl.where(mskL & (less != 0), vJ, v)
        v = tl.where(mskj & (less != 0), piv, v)
        idxv = tl.where(mskL & (less != 0), iJ, idxv)
        idxv = tl.where(mskj & (less != 0), i_at_L, idxv)
        L = tl.where((less != 0) & (j_s <= kth), i_s, L)
        R = tl.where((less != 0) & (j_s >= kth), j_s - 1, R)
        stage = tl.where(less != 0, 0, stage)
        # partition continues: swap (i_s, j_s) and advance
        cont = s1 & (j_s >= i_s)
        mski = (col_offs == i_s) & (col_offs < N)
        vI = tl.min(tl.where(mski, v, INF), axis=0)
        iI = tl.min(tl.where(mski, idxv, BIG), axis=0)
        v = tl.where(mski & (cont != 0), vJ, v)
        v = tl.where(mskj & (cont != 0), vI, v)
        idxv = tl.where(mski & (cont != 0), iJ, idxv)
        idxv = tl.where(mskj & (cont != 0), iI, idxv)
        i = tl.where(cont != 0, i_s, i)
        j = tl.where(cont != 0, j_s, j)
    fin_i = tl.min(tl.where((col_offs == kth) & (col_offs < N), idxv, BIG), axis=0)
    tl.store(value_ptr + pid, kth_val)
    tl.store(index_ptr + pid, tl.where(cnt > 1, fin_i, idx).to(tl.int64))


@libentry()
@triton.jit
def _kthvalue_stage_kernel(
    input_ptr,
    selected_ptr,
    partial_value_ptr,
    partial_index_ptr,
    M,
    N,
    CHUNKS,
    CHUNK_OFFSET,
    CHUNK_ID,
    ROW_OFFSET,
    BLOCK_N: tl.constexpr,
):
    pid_m = ROW_OFFSET + ext.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    indices = CHUNK_OFFSET + offsets
    valid = indices < N
    input_base = input_ptr + pid_m * N

    previous0 = tl.load(selected_ptr + pid_m)
    previous1 = tl.load(selected_ptr + M + pid_m)
    previous2 = tl.load(selected_ptr + 2 * M + pid_m)
    values = tl.load(
        input_base + indices,
        mask=valid,
        other=float("inf"),
    ).to(tl.float32)
    values = tl.where(indices == previous0, float("inf"), values)
    values = tl.where(indices == previous1, float("inf"), values)
    values = tl.where(indices == previous2, float("inf"), values)

    chunk_value = tl.min(values, axis=0)
    chunk_index = tl.min(tl.where(values == chunk_value, indices, N), axis=0)
    partial_offset = pid_m * CHUNKS + CHUNK_ID
    tl.store(partial_value_ptr + partial_offset, chunk_value)
    tl.store(partial_index_ptr + partial_offset, chunk_index)


@libentry()
@triton.jit
def _kthvalue_finalize_kernel(
    partial_value_ptr,
    partial_index_ptr,
    selected_output_ptr,
    value_ptr,
    index_ptr,
    M,
    CHUNKS,
    ROW_OFFSET,
    BLOCK_C: tl.constexpr,
):
    pid = ROW_OFFSET + ext.program_id(0)
    offsets = tl.arange(0, BLOCK_C)
    valid = offsets < CHUNKS
    base = pid * CHUNKS
    values = tl.load(
        partial_value_ptr + base + offsets,
        mask=valid,
        other=float("inf"),
    )
    indices = tl.load(
        partial_index_ptr + base + offsets,
        mask=valid,
        other=-1,
    )
    best_value = tl.min(values, axis=0)
    best_index = tl.min(tl.where(values == best_value, indices, 2147483647), axis=0)
    tl.store(selected_output_ptr + pid, best_index)
    tl.store(value_ptr + pid, best_value)
    tl.store(index_ptr + pid, best_index)


def kthvalue(inp, k, dim=-1, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN KTHVALUE")

    ndim = inp.ndim
    if dim < -ndim or dim >= ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{-ndim}, {ndim - 1}], but got {dim})"
        )
    dim %= ndim
    dim_size = inp.shape[dim]
    if dim_size == 0:
        raise IndexError(
            f"kthvalue(): Expected reduction dim {dim} to have non-zero size."
        )
    if k < 1 or k > dim_size:
        raise RuntimeError(
            f"kthvalue(): selected number k out of range for dimension {dim}"
        )

    if inp.numel() == 0:
        out_shape = list(inp.shape)
        if keepdim:
            out_shape[dim] = 1
        else:
            del out_shape[dim]
        return (
            torch.empty(out_shape, dtype=inp.dtype, device=inp.device),
            torch.empty(out_shape, dtype=torch.int64, device=inp.device),
        )

    perm = [axis for axis in range(ndim) if axis != dim] + [dim]
    src = inp.permute(perm)
    M = src.numel() // dim_size
    values = torch.empty((M,), dtype=inp.dtype, device=inp.device)
    indices = torch.empty((M,), dtype=torch.int64, device=inp.device)

    max_programs = 16384
    with torch_device_fn.device(inp.device):
        if dim_size <= 512:
            block_n = triton.next_power_of_2(dim_size)
            if block_n == dim_size:
                transposed = src.contiguous()
            else:
                transposed = torch.full(
                    (M, block_n),
                    float("inf"),
                    dtype=inp.dtype,
                    device=inp.device,
                )
                transposed[:, :dim_size].copy_(src.reshape(M, dim_size))
            _kthvalue_packed_kernel[(M,)](
                transposed,
                values,
                indices,
                dim_size,
                K=k,
                BLOCK_N=block_n,
                MAXC=6 * dim_size + 128,
                num_warps=4,
                isCloseUnrollControl=True,
                isCloseVectorization=True,
            )
        else:
            transposed = src.contiguous()
            selected = torch.full((4, M), -1, dtype=torch.int32, device=inp.device)
            block_n = 512
            chunks = triton.cdiv(dim_size, block_n)
            block_c = triton.next_power_of_2(chunks)
            partial_values = torch.empty(
                (M, chunks), dtype=torch.float32, device=inp.device
            )
            partial_indices = torch.empty(
                (M, chunks), dtype=torch.int32, device=inp.device
            )
            for rank in range(k):
                for chunk_id in range(chunks):
                    for row_offset in range(0, M, max_programs):
                        row_count = min(max_programs, M - row_offset)
                        _kthvalue_stage_kernel[(row_count,)](
                            transposed,
                            selected,
                            partial_values,
                            partial_indices,
                            M,
                            dim_size,
                            chunks,
                            chunk_id * block_n,
                            chunk_id,
                            row_offset,
                            BLOCK_N=block_n,
                            num_warps=4,
                            buffer_size_limit=2048,
                            isCloseVectorization=True,
                        )
                        torch_device_fn.synchronize()
                for row_offset in range(0, M, max_programs):
                    row_count = min(max_programs, M - row_offset)
                    _kthvalue_finalize_kernel[(row_count,)](
                        partial_values,
                        partial_indices,
                        selected[rank],
                        values,
                        indices,
                        M,
                        chunks,
                        row_offset,
                        BLOCK_C=block_c,
                        num_warps=4,
                        buffer_size_limit=2048,
                        isCloseVectorization=True,
                    )
                    torch_device_fn.synchronize()

    out_shape = list(inp.shape)
    if keepdim:
        out_shape[dim] = 1
    else:
        del out_shape[dim]
    return values.reshape(out_shape), indices.reshape(out_shape)
