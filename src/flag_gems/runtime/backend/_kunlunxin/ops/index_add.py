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
from flag_gems.utils import dim_compress, libentry

logger = logging.getLogger(__name__)
# def cfggen():
#     block_m = [1, 2, 4]
#     block_n = [128, 1024, 2048, 4096]
#     configs = [
#         triton.Config({"BLOCK_M": m, "BLOCK_N": n}, num_warps=4)
#         for m in block_m
#         for n in block_n
#     ]
#     return configs


@libentry()
# @triton.autotune(configs=cfggen(), key=["M", "N"])
@triton.heuristics(runtime.get_heuristic_config("index_add"))
# @triton.autotune(
#     configs=[], generate_configs="index_add", op_affiliation="cluster", row_sign="M", col_sign="N",
#     key=["M", "N"],
# )
@triton.jit
def index_add_kernel(
    index,
    src,
    inp_cont,
    M: tl.constexpr,
    N: tl.constexpr,
    alpha,
    inp_len,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Fast path: index is unique. Each destination cell is written by exactly
    # one program, so plain (non-atomic) load-add-store is safe and fast.
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)
    rows_offsets = pid_x * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols_offsets = pid_y * BLOCK_N + tl.arange(0, BLOCK_N)

    rows_mask = rows_offsets < M
    index_mask = cols_offsets < N
    block_mask = rows_mask and index_mask

    cur_indices = tl.load(index + cols_offsets, mask=index_mask, other=0)
    inp_off = rows_offsets * inp_len + cur_indices[None, :]
    cur_inp = tl.load(inp_cont + inp_off, mask=block_mask, other=0.0)
    src_off = rows_offsets * N + cols_offsets[None, :]
    cur_src = tl.load(src + src_off, mask=block_mask, other=0.0)
    cur_inp += alpha * cur_src

    tl.store(inp_cont + inp_off, cur_inp, mask=block_mask)


@libentry()
@triton.jit
def index_add_seg_kernel(
    src_ptr,
    perm_ptr,
    v_ptr,
    head_ptr,
    len_ptr,
    out_ptr,
    M,
    N,
    out_len,
    U,
    alpha,
    BLOCK_S: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Deterministic duplicate-safe path. The index is sorted once (host side);
    # equal values form adjacent runs ("segments"). One program handles one row
    # and BLOCK_S segments; each segment's source columns are summed with a
    # bounded 1D column-tile reduction and accumulated into the unique
    # destination column of that segment. There are no atomics and no
    # cross-program writes to the same cell. (1D-only structure: the 2D
    # gather+reduce variant fails the XPU uni_sram lowering pass.)
    pid_m = tl.program_id(axis=0)
    pid_s = tl.program_id(axis=1)
    r = pid_m
    cc = tl.arange(0, BLOCK_C)
    for s in tl.static_range(BLOCK_S):
        gs = pid_s * BLOCK_S + s
        ok_s = gs < U
        c0_t = tl.load(head_ptr + gs)
        l_s = tl.load(len_ptr + gs)
        j_s = tl.load(v_ptr + c0_t)
        m_s = (cc < l_s) & ok_s
        # XPU masked loads may ignore `other=0` for runtime masks; read within
        # the padded buffers, then zero the out-of-segment lanes explicitly.
        cols = tl.load(perm_ptr + c0_t + cc)  # padded, always in-bounds
        col_clamped = tl.minimum(tl.where(m_s, cols, 0), N - 1)
        vals = tl.load(src_ptr + r.to(tl.int64) * N + col_clamped)
        vals = tl.where(m_s, vals, 0.0)
        seg_sum = tl.sum(vals, axis=0)
        dst = r.to(tl.int64) * out_len + j_s
        cur = tl.load(out_ptr + dst, mask=ok_s, other=0.0)
        tl.store(out_ptr + dst, cur + alpha * seg_sum, mask=ok_s)


@libentry()
@triton.jit
def index_add_seg_loop_kernel(
    src_ptr,
    perm_ptr,
    v_ptr,
    head_ptr,
    len_ptr,
    out_ptr,
    M,
    N,
    out_len,
    U,
    alpha,
    BLOCK_S: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Same deterministic segment reduction for segments longer than BLOCK_C:
    # each segment's source columns are accumulated in column tiles.
    pid_m = tl.program_id(axis=0)
    pid_s = tl.program_id(axis=1)
    r = pid_m
    cc = tl.arange(0, BLOCK_C)
    for s in tl.static_range(BLOCK_S):
        gs = pid_s * BLOCK_S + s
        ok_s = gs < U
        c0_t = tl.load(head_ptr + gs)
        l_s = tl.load(len_ptr + gs)
        j_s = tl.load(v_ptr + c0_t)
        acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
        for k in range(0, l_s, BLOCK_C):
            m_s = ((k + cc) < l_s) & ok_s
            cols = tl.load(perm_ptr + c0_t + k + cc)  # padded, in-bounds
            col_clamped = tl.minimum(tl.where(m_s, cols, 0), N - 1)
            vals = tl.load(src_ptr + r.to(tl.int64) * N + col_clamped)
            vals = tl.where(m_s, vals, 0.0)
            acc += vals
        seg_sum = tl.sum(acc, axis=0)
        dst = r.to(tl.int64) * out_len + j_s
        cur = tl.load(out_ptr + dst, mask=ok_s, other=0.0)
        tl.store(out_ptr + dst, cur + alpha * seg_sum, mask=ok_s)


# Duplicate-safe segment-reduction tile sizes (bounded: BLOCK_S * BLOCK_C
# stays well under the XPU tl.sum safety ceiling of 8192 elements).
index_add_seg_block_s = 64
index_add_seg_block_c = 64

# Fixed tiles reused by the non-atomic fast kernel when invoked without
# heuristics (bounded, mirrors the index_add heuristics caps).
index_add_fast_block_m = 8
index_add_fast_block_n = 256

def _index_add_sort(index):
    """Deterministic sort of an index returning (sorted_values, perm).

    Reuses the backend-local radix sort (deterministic, works inside the gems
    dispatch context without the tl.argsort lowering that fails on XPU).
    """
    N = index.numel()
    if N <= 1:
        return index, torch.arange(N, dtype=torch.int64, device=index.device)
    from .sort import radix_sort_low_mem  # backend-local reuse, no dispatch

    flat = index.reshape(1, -1)
    v, perm = radix_sort_low_mem(flat, k_bits=4)
    return v.reshape(-1), perm.reshape(-1)


@libentry()
@triton.jit
def index_mark_unique_kernel(index_ptr, mark_ptr, N, BLOCK: tl.constexpr):
    # Writes 1 at mark[index[i]] for every index element. Concurrent writes of
    # the same value 1 to one slot are benign (no atomicity requirement), so
    # after this kernel sum(mark) == number of distinct index values.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    v = tl.load(index_ptr + offs, mask=mask, other=0).to(tl.int64)
    tl.store(mark_ptr + v, tl.full((BLOCK,), 1, dtype=tl.int32), mask=mask)


def _index_unique_gate(index, inp_len):
    """Sound duplicate check in O(N): distinct_count == N  <=>  no duplicates.

    A per-value marker buffer is set to 1 (same-value concurrent writes are
    deterministic); sum(markers) equals the number of distinct index values,
    so sum == N iff the index contains no duplicate. No sorting required and
    the path stays entirely on XPU.
    """
    N = index.numel()
    if N <= 1:
        return True
    block = 256
    grid = (triton.cdiv(N, block),)
    mark = torch.zeros((inp_len,), dtype=torch.int32, device=index.device)
    index_mark_unique_kernel[grid](index, mark, N, BLOCK=block)
    return bool(mark.sum().item() == N)


@libentry()
@triton.jit
def index_seg_flags_kernel(v_ptr, flags_ptr, N, BLOCK: tl.constexpr):
    # flags[i] = 1 when sorted position i starts a new segment (v[i] != v[i-1]).
    # Written as a device kernel so the comparison never goes through the
    # pointwise codegen path (which fails to lower large N tiles on XPU).
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    cur = tl.load(v_ptr + offs, mask=mask, other=-1)
    prev = tl.load(v_ptr + offs - 1, mask=mask & (offs > 0))
    is_head = tl.where(offs == 0, True, cur != prev)
    tl.store(flags_ptr + offs, is_head.to(tl.int64), mask=mask)


def _index_add_segments(index, block_s, block_c):
    """Deterministic duplicate classification of an index tensor.

    Returns sorted values ``v``, the sort permutation ``perm`` and padded
    per-segment tables (head column offset / length) used by the duplicate-safe
    kernel. See ``index_add_seg_kernel``.
    """
    N = index.numel()
    v, perm = _index_add_sort(index)
    # int64 marker channel (device kernel, avoids the XPU pointwise ne crash).
    flag_blk = 256
    flags = torch.zeros((N,), dtype=torch.int64, device=index.device)
    index_seg_flags_kernel[(triton.cdiv(N, flag_blk),)](
        v, flags, N, BLOCK=flag_blk
    )
    heads = torch.nonzero(flags).reshape(-1)  # (U,) start column of each segment
    U = heads.numel()
    lens = torch.cat(
        (heads[1:] - heads[:-1], (N - heads[-1]).reshape(1))
    ).to(torch.int32)
    # Padded copies so masked lanes never read past the physical buffers.
    n_pad = ((N + block_c - 1) // block_c) * block_c
    u_pad = ((U + block_s - 1) // block_s) * block_s
    perm_pad = torch.zeros((n_pad,), dtype=torch.int64, device=index.device)
    perm_pad[:N] = perm
    v_pad = torch.zeros((n_pad,), dtype=torch.int64, device=index.device)
    v_pad[:N] = v
    head_pad = torch.zeros((u_pad,), dtype=torch.int64, device=index.device)
    head_pad[:U] = heads
    len_pad = torch.zeros((u_pad,), dtype=torch.int32, device=index.device)
    len_pad[:U] = lens
    return v_pad, perm_pad, head_pad, len_pad, U, int(lens.max().item())


def index_add(inp, dim, index, src, alpha=1):
    logger.debug("GEMS_KUNLUNXIN INDEX_ADD")
    assert ((0 <= index) * (index < inp.size(dim))).equal(
        torch.ones(tuple(index.shape), dtype=torch.bool, device="cuda")
    ), "0 <= index < self.size(dim)"
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index.numel() == src.size(
        dim
    ), "The dimth dimension of source must have the same size as the length of index"
    assert (
        inp.ndim == src.ndim
    ), "Self and source should have the same number of dimensions"
    assert (
        ((inp.size(i) == src.size(i)) or i == dim) for i in range(0, inp.ndim)
    ), "src.size(d) == self.size(d) for all dimensions d != dim"

    inp = inp.contiguous()
    index = index.contiguous()
    src = src.contiguous()

    dim = dim % inp.ndim
    inp_len = inp.size(dim)
    N = index.numel()
    M = src.numel() // N if N > 0 else 0
    fine_dim = inp.ndim - 1
    if dim != fine_dim:
        inp = dim_compress(inp, dim)
        src = dim_compress(src, dim)
    inp_cont = inp.clone()

    if N > 0 and M > 0:
        if _index_unique_gate(index, inp_len):
            # Index has no duplicates: exactly one writer per destination
            # cell, so the plain non-atomic load-add-store kernel is safe
            # (heuristics supply the bounded BLOCK_M/BLOCK_N at launch).
            grid = lambda meta: (
                triton.cdiv(M, min(index_add_fast_block_m, meta["BLOCK_M"])),
                triton.cdiv(N, min(index_add_fast_block_n, meta["BLOCK_N"])),
            )
            index_add_kernel[grid](
                index, src, inp_cont, M, N, alpha, inp_len
            )
        else:
            # Duplicate index values: deterministic segment reduction, no
            # atomics, no cross-program writes to the same cell.
            v_pad, perm_pad, head_pad, len_pad, U, len_max = _index_add_segments(
                index, index_add_seg_block_s, index_add_seg_block_c
            )
            grid = (M, triton.cdiv(U, index_add_seg_block_s))
            seg_args = (
                src,
                perm_pad,
                v_pad,
                head_pad,
                len_pad,
                inp_cont,
                M,
                N,
                inp_len,
                U,
                alpha,
            )
            if len_max <= index_add_seg_block_c:
                index_add_seg_kernel[grid](
                    *seg_args,
                    BLOCK_S=index_add_seg_block_s,
                    BLOCK_C=index_add_seg_block_c,
                )
            else:
                index_add_seg_loop_kernel[grid](
                    *seg_args,
                    BLOCK_S=index_add_seg_block_s,
                    BLOCK_C=index_add_seg_block_c,
                )
    if dim != fine_dim:
        order = [i for i in range(inp_cont.ndim - 1)]
        order.insert(dim, fine_dim)
        return inp_cont.permute(order).contiguous()
    else:
        return inp_cont


def index_add_(inp, dim, index, src, alpha=1):
    logger.debug("GEMS_KUNLUNXIN INDEX_ADD_")
    assert ((0 <= index) * (index < inp.size(dim))).equal(
        torch.ones(tuple(index.shape), dtype=torch.bool, device="cuda")
    ), "0 <= index < self.size(dim)"
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index.numel() == src.size(
        dim
    ), "The dimth dimension of source must have the same size as the length of index"
    assert (
        inp.ndim == src.ndim
    ), "Self and source should have the same number of dimensions"
    assert (
        ((inp.size(i) == src.size(i)) or i == dim) for i in range(0, inp.ndim)
    ), "src.size(d) == self.size(d) for all dimensions d != dim"

    inp_cont = inp.clone()
    inp_cont = inp_cont.contiguous()
    index = index.contiguous()
    src = src.contiguous()

    dim = dim % inp_cont.ndim
    inp_len = inp_cont.size(dim)
    N = index.numel()
    M = src.numel() // N
    fine_dim = inp_cont.ndim - 1
    if dim != fine_dim:
        inp_cont = dim_compress(inp_cont, dim)
        src = dim_compress(src, dim)

    if N > 0 and M > 0:
        if _index_unique_gate(index, inp_len):
            # Index has no duplicates: exactly one writer per destination
            # cell, so the plain non-atomic load-add-store kernel is safe
            # (heuristics supply the bounded BLOCK_M/BLOCK_N at launch).
            grid = lambda meta: (
                triton.cdiv(M, min(index_add_fast_block_m, meta["BLOCK_M"])),
                triton.cdiv(N, min(index_add_fast_block_n, meta["BLOCK_N"])),
            )
            index_add_kernel[grid](
                index, src, inp_cont, M, N, alpha, inp_len
            )
        else:
            # Duplicate index values: deterministic segment reduction.
            v_pad, perm_pad, head_pad, len_pad, U, len_max = _index_add_segments(
                index, index_add_seg_block_s, index_add_seg_block_c
            )
            grid = (M, triton.cdiv(U, index_add_seg_block_s))
            seg_args = (
                src,
                perm_pad,
                v_pad,
                head_pad,
                len_pad,
                inp_cont,
                M,
                N,
                inp_len,
                U,
                alpha,
            )
            if len_max <= index_add_seg_block_c:
                index_add_seg_kernel[grid](
                    *seg_args,
                    BLOCK_S=index_add_seg_block_s,
                    BLOCK_C=index_add_seg_block_c,
                )
            else:
                index_add_seg_loop_kernel[grid](
                    *seg_args,
                    BLOCK_S=index_add_seg_block_s,
                    BLOCK_C=index_add_seg_block_c,
                )
    if dim != fine_dim:
        order = [i for i in range(inp_cont.ndim - 1)]
        order.insert(dim, fine_dim)
        inp_cont = inp_cont.permute(order).contiguous()
        torch.ops.aten._copy_from(inp_cont, inp, False)
        return inp
    else:
        torch.ops.aten._copy_from(inp_cont, inp, False)
        return inp
