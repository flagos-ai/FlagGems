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

"""Shared Triton/TLE primitives for the active Hopper FA3 kernels."""

import triton
import triton.language as tl

from flag_gems.utils import tl_extra_shim

from .validation import tle


@triton.jit
def _apply_softcap_v3(S, softcap, IS_SOFTCAP: tl.constexpr):
    """Apply the compile-time soft-cap transform to attention scores."""

    if IS_SOFTCAP:
        S = tl_extra_shim.tanh(S * softcap)
    return S


@triton.jit
def _apply_alibi_v3(
    S,
    col_idx,
    row_idx,
    max_seqlen_q,
    max_seqlen_k,
    IS_CAUSAL: tl.constexpr,
    IS_ALIBI: tl.constexpr,
    alibi_slope,
):
    """Add the ALiBi positional bias to an attention-score tile."""

    if IS_ALIBI:
        if IS_CAUSAL:
            # Future columns are masked separately, so the signed distance is
            # non-positive here.  Keep the row-dependent term: dropping that
            # softmax-invariant constant would make the returned LSE incorrect.
            bias = alibi_slope * (
                col_idx[None, :] - max_seqlen_k + max_seqlen_q - row_idx[:, None]
            ).to(tl.float32)
            S += bias
        else:
            bias = -alibi_slope * tl.abs(
                col_idx[None, :] - max_seqlen_k + max_seqlen_q - row_idx[:, None]
            ).to(tl.float32)
            S += bias
    return S


@triton.jit
def _apply_mask_v3(
    S,
    col_idx,
    row_idx,
    max_seqlen_q,
    max_seqlen_k,
    window_size_left,
    window_size_right,
    IS_EVEN_MN: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    IS_LOCAL: tl.constexpr,
):
    """Mask invisible and out-of-bounds entries in an attention-score tile."""

    if IS_CAUSAL or IS_LOCAL or (not IS_EVEN_MN):
        col_lb = tl.maximum(0, row_idx + max_seqlen_k - max_seqlen_q - window_size_left)
        col_rb = tl.minimum(
            max_seqlen_k - 1,
            row_idx + max_seqlen_k - max_seqlen_q + window_size_right,
        )
        if IS_CAUSAL:
            S = tl.where(col_idx[None, :] > col_rb[:, None], float("-inf"), S)
        if IS_LOCAL:
            S = tl.where(
                (col_idx[None, :] > col_rb[:, None])
                | (col_idx[None, :] < col_lb[:, None]),
                float("-inf"),
                S,
            )
        if (not IS_LOCAL) and (not IS_CAUSAL) and (not IS_EVEN_MN):
            S = tl.where(col_idx[None, :] >= max_seqlen_k, float("-inf"), S)
    return S


@triton.jit
def _softmax_online_deferred(
    S,
    m_prev,
    l_prev,
    softmax_scale_log2e: tl.constexpr,
    IS_BORDER: tl.constexpr,
):
    """Merge one score tile into the deferred-normalization online softmax state."""

    m_new = tl.maximum(m_prev, tl.max(S, 1))
    if IS_BORDER:
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
    else:
        m_safe = m_new

    alpha = tl.math.exp2((m_prev - m_safe) * softmax_scale_log2e)
    if IS_BORDER:
        m_scaled = tl.where(m_new == float("-inf"), 0.0, m_safe * softmax_scale_log2e)
    else:
        m_scaled = m_safe * softmax_scale_log2e
    P = tl.math.exp2(S * softmax_scale_log2e - m_scaled[:, None])
    l_new = l_prev * alpha + tl.sum(P, 1)
    return alpha, P, m_new, l_new


@triton.jit
def _merge_attention_sink(
    rowmax,
    rowsum,
    sink,
    softmax_scale_log2e: tl.constexpr,
):
    """Merge a zero-value attention sink into the final softmax state."""

    rowmax = tl.where(rowmax == float("-inf"), 0.0, rowmax)
    rowsum += tl.math.exp2(
        sink.to(tl.float32) * 1.4426950408889634
        - rowmax * softmax_scale_log2e
    )
    return rowmax, rowsum


@triton.jit
def _paged_blockwise_cache_indices(
    n_start,
    offsets,
    max_virtual_index,
    page_table_ptr,
    block_size: tl.constexpr,
    page_stride_rows,
    BLOCK_N: tl.constexpr,
    PAGED_GATHER_MODE: tl.constexpr,
    BOUNDARY_CHECK: tl.constexpr = True,
):
    """Map one logical N tile to physical rows in a paged KV cache."""

    logical_idx = n_start + offsets
    # Keep FA3 paged-cache address arithmetic in the 64-bit domain before
    # multiplying a physical page id by its page stride.
    page_stride_rows = page_stride_rows.to(tl.int64)
    if PAGED_GATHER_MODE == 0:
        # Resolve each logical token through its virtual page and in-page offset.
        virtual_page_idx = logical_idx // block_size
        page_offset = logical_idx % block_size
        if BOUNDARY_CHECK:
            page_block_idx = tl.load(
                page_table_ptr + virtual_page_idx,
                mask=logical_idx < max_virtual_index,
                other=0,
            ).to(tl.int32)
        else:
            page_block_idx = tl.load(page_table_ptr + virtual_page_idx).to(tl.int32)
        cache_idx = page_block_idx * page_stride_rows + page_offset
    else:
        if block_size <= BLOCK_N:
            NUM_PAGE_ENTRIES: tl.constexpr = BLOCK_N // block_size
            ROWS_PER_ENTRY: tl.constexpr = block_size
        else:
            NUM_PAGE_ENTRIES: tl.constexpr = 1
            ROWS_PER_ENTRY: tl.constexpr = BLOCK_N

        page_offsets = tl.arange(0, NUM_PAGE_ENTRIES)
        first_idx = n_start + page_offsets * block_size
        page_idx = first_idx // block_size
        if BOUNDARY_CHECK:
            page_blocks = tl.load(
                page_table_ptr + page_idx,
                mask=first_idx < max_virtual_index,
                other=0,
            ).to(tl.int32)
        else:
            page_blocks = tl.load(page_table_ptr + page_idx).to(tl.int32)
        # PAGE_SIZE and BLOCK_N are powers of two, so one divides the other.  A
        # tile therefore spans whole pages or stays inside one page.  Broadcast
        # each page-table result over only the token rows that share that entry.
        page_blocks = tl.reshape(
            tl.broadcast_to(page_blocks[:, None], (NUM_PAGE_ENTRIES, ROWS_PER_ENTRY)),
            (BLOCK_N,),
            can_reorder=True,
        )
        cache_idx = page_blocks * page_stride_rows + logical_idx % block_size
    return cache_idx


@triton.jit
def _buf_phase_tle(count, num_buffers: tl.constexpr):
    """Map a pipeline iteration to its ring-buffer slot and phase."""

    buf = count % num_buffers
    phase_idx = count // num_buffers
    return buf, phase_idx


@triton.jit
def _persistent_tile_coords(
    tile_idx,
    num_pid_m,
    batch_size,
    NUM_HEADS: tl.constexpr = 1,
    HEADS_IN_L2: tl.constexpr = 0,
):
    """Map a rectangular persistent work id to M block, batch, and head."""

    if HEADS_IN_L2 > 1:
        # Uniform-sequence counterpart of the varlen LPT mapping below.  Keep
        # each batch contiguous and traverse an L2-sized group of heads inside
        # each M tile, matching FA3 without paying for the ragged prefix scan.
        batch_work = num_pid_m * NUM_HEADS
        bid = tile_idx // batch_work
        within_batch = tile_idx - bid * batch_work
        work_per_full_section: tl.constexpr = HEADS_IN_L2
        section_work = num_pid_m * work_per_full_section
        section_idx = within_batch // section_work
        section_start_head = section_idx * work_per_full_section
        heads_this_section = tl.minimum(
            work_per_full_section, NUM_HEADS - section_start_head
        )
        within_section = within_batch - section_idx * section_work
        m_block = within_section // heads_this_section
        hid = section_start_head + within_section - m_block * heads_this_section
        m_block = num_pid_m - 1 - m_block
        return m_block, bid, hid
    m_block = tile_idx % num_pid_m
    if HEADS_IN_L2 > 0:
        m_block = num_pid_m - 1 - m_block
    hb = tile_idx // num_pid_m
    bid = hb % batch_size
    hid = hb // batch_size
    return m_block, bid, hid


@triton.jit
def _ragged_persistent_tile_coords(
    tile_idx,
    cu_seqlens_q_ptr,
    batch_size,
    num_heads: tl.constexpr,
    BLOCK_M: tl.constexpr,
    PACK_FACTOR: tl.constexpr = 1,
    HEADS_IN_L2: tl.constexpr = 0,
):
    """Map a compact ragged work id to M block, batch, and head."""

    lane = tl.arange(0, 32)
    group_bid = 0
    group_start = 0
    found = False
    selected_bid = 0
    selected_m_blocks = 1
    selected_within = 0

    while (group_bid < batch_size) & (~found):
        bids = group_bid + lane
        valid = (lane < 31) & (bids < batch_size)
        q_bos = tl.load(cu_seqlens_q_ptr + bids, mask=valid, other=0)
        q_eos = tl.load(cu_seqlens_q_ptr + bids + 1, mask=valid, other=0)
        q_len = q_eos - q_bos
        m_blocks = tl.cdiv(q_len * PACK_FACTOR, BLOCK_M)
        batch_work = tl.where(valid, m_blocks * num_heads, 0)
        work_prefix = tl.cumsum(batch_work, axis=0)
        work_before = work_prefix - batch_work
        group_work = tl.sum(batch_work, axis=0)
        local_idx = tile_idx - group_start
        in_group = local_idx < group_work
        hit = valid & (local_idx >= work_before) & (local_idx < work_prefix)

        hit_bid = tl.sum(tl.where(hit, bids, 0), axis=0)
        hit_m_blocks = tl.sum(tl.where(hit, m_blocks, 0), axis=0)
        hit_within = tl.sum(tl.where(hit, local_idx - work_before, 0), axis=0)
        selected_bid = tl.where(in_group, hit_bid, selected_bid)
        selected_m_blocks = tl.where(in_group, hit_m_blocks, selected_m_blocks)
        selected_within = tl.where(in_group, hit_within, selected_within)
        found = found | in_group
        group_start += group_work
        group_bid += 31

    safe_m_blocks = tl.maximum(selected_m_blocks, 1)
    if HEADS_IN_L2 > 1:
        # Match FA3's LPT head swizzle: keep a power-of-two group of KV
        # heads resident in L2 and traverse heads inside each M tile.  A tail
        # group may contain fewer heads, so its divisor must be adjusted.
        section_work: tl.constexpr = HEADS_IN_L2
        work_per_full_section = safe_m_blocks * section_work
        section_idx = selected_within // work_per_full_section
        section_start_head = section_idx * section_work
        heads_this_section = tl.minimum(section_work, num_heads - section_start_head)
        within_section = selected_within - section_idx * work_per_full_section
        m_block = within_section // heads_this_section
        hid = section_start_head + within_section - m_block * heads_this_section
    else:
        hid = selected_within // safe_m_blocks
        m_block = selected_within - hid * safe_m_blocks
    if HEADS_IN_L2 > 0:
        m_block = safe_m_blocks - 1 - m_block
    return m_block, selected_bid, hid, found


@triton.jit
def _split_kv_count(k_len, MAX_SPLITS: tl.constexpr):
    """Choose the Split-KV count from the key sequence length."""

    # Preserve both the values and generated comparison-only mapper used by the
    # production auto-s3 path.  Larger explicit caps need the generalized
    # quotient so they can continue scaling past three work items.
    if MAX_SPLITS <= 3:
        split_count = 1 + (k_len > 12 * 128) + (k_len > 24 * 128)
    else:
        split_count = tl.maximum(1, tl.cdiv(k_len, 12 * 128))
    return tl.minimum(split_count, MAX_SPLITS)


@triton.jit
def _ragged_persistent_split_tile_coords(
    tile_idx,
    cu_seqlens_q_ptr,
    seqused_k_ptr,
    batch_size,
    num_heads: tl.constexpr,
    BLOCK_M: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
    PACK_FACTOR: tl.constexpr = 1,
    HEADS_IN_L2: tl.constexpr = 0,
):
    """Map a compact ragged Split-KV work id to its work unit."""

    lane = tl.arange(0, 32)
    group_bid = 0
    group_start = 0
    found = False
    selected_bid = 0
    selected_m_blocks = 1
    selected_splits = 1
    selected_within = 0

    while group_bid < batch_size:
        bids = group_bid + lane
        valid = (lane < 31) & (bids < batch_size)
        q_bos = tl.load(cu_seqlens_q_ptr + bids, mask=valid, other=0)
        q_eos = tl.load(cu_seqlens_q_ptr + bids + 1, mask=valid, other=0)
        q_len = q_eos - q_bos
        m_blocks = tl.cdiv(q_len * PACK_FACTOR, BLOCK_M)
        k_len = tl.load(seqused_k_ptr + bids, mask=valid, other=0)
        split_counts = _split_kv_count(k_len, MAX_SPLITS)
        batch_work = tl.where(valid, m_blocks * num_heads * split_counts, 0)
        work_prefix = tl.cumsum(batch_work, axis=0)
        work_before = work_prefix - batch_work
        group_work = tl.sum(batch_work, axis=0)
        local_idx = tile_idx - group_start
        in_group = (~found) & (local_idx >= 0) & (local_idx < group_work)
        hit = valid & (local_idx >= work_before) & (local_idx < work_prefix)

        hit_bid = tl.sum(tl.where(hit, bids, 0), axis=0)
        hit_m_blocks = tl.sum(tl.where(hit, m_blocks, 0), axis=0)
        hit_splits = tl.sum(tl.where(hit, split_counts, 0), axis=0)
        hit_within = tl.sum(tl.where(hit, local_idx - work_before, 0), axis=0)
        selected_bid = tl.where(in_group, hit_bid, selected_bid)
        selected_m_blocks = tl.where(in_group, hit_m_blocks, selected_m_blocks)
        selected_splits = tl.where(in_group, hit_splits, selected_splits)
        selected_within = tl.where(in_group, hit_within, selected_within)
        found = found | in_group
        group_start += group_work
        group_bid += 31

    safe_m_blocks = tl.maximum(selected_m_blocks, 1)
    safe_splits = tl.maximum(selected_splits, 1)
    head_split = selected_within // safe_m_blocks
    m_block = selected_within - head_split * safe_m_blocks
    hid = head_split // safe_splits
    split_id = head_split - hid * safe_splits
    if HEADS_IN_L2 > 0:
        m_block = safe_m_blocks - 1 - m_block
    return (
        m_block,
        selected_bid,
        hid,
        split_id,
        safe_splits,
        group_start,
        found,
    )


@triton.jit
def _split_n_block_range(n_block_min, n_block_max, split_id, split_count):
    """Return the half-open N-block range assigned to one KV split."""

    span = tl.maximum(n_block_max - n_block_min, 0)
    blocks_per_split = tl.cdiv(span, split_count)
    split_min = n_block_min + split_id * blocks_per_split
    split_max = tl.minimum(n_block_max, split_min + blocks_per_split)
    return split_min, split_max


@triton.jit
def _fence_async_shared_cta():
    """Publish generic-proxy shared-memory writes to the async proxy."""

    # mov.u32 here because you must declare at least one output.
    tl.inline_asm_elementwise(
        "mov.u32 $0, 0x0; fence.proxy.async.shared::cta;",
        constraints="=r",
        args=(),
        dtype=(tl.int32,),
        is_pure=False,
        pack=1,
    )


@triton.jit
def _copy_paged_kv_tile_to_pipe(
    writer,
    iteration,
    ASYNC_COMMIT: tl.constexpr,
    src_base,
    row_stride,
    page_table_ptr_b,
    n_offset,
    k_len,
    d: tl.constexpr,
    block_size: tl.constexpr,
    page_stride_rows,
    BLOCK_N: tl.constexpr,
    HEAD_DIM_PADDED: tl.constexpr,
    PAGED_GATHER_MODE: tl.constexpr = 2,
    BOUNDARY_CHECK: tl.constexpr = True,
):
    """Publish one paged KV tile through a cp.async-capable TLE pipe."""

    slot = writer.acquire(iteration)
    if ASYNC_COMMIT:
        rows = tl.arange(0, BLOCK_N)
        cols = tl.arange(0, HEAD_DIM_PADDED)
        logical_idx = n_offset + rows
        cache_idx = _paged_blockwise_cache_indices(
            n_offset,
            rows,
            k_len,
            page_table_ptr_b,
            block_size,
            page_stride_rows,
            BLOCK_N,
            PAGED_GATHER_MODE,
            BOUNDARY_CHECK=BOUNDARY_CHECK,
        )
        src_ptrs = src_base + cache_idx[:, None] * row_stride + cols[None, :]
        if BOUNDARY_CHECK:
            load_mask = (logical_idx[:, None] < k_len) & (cols[None, :] < d)
            vals = tl.load(src_ptrs, mask=load_mask, other=0.0)
        elif d == HEAD_DIM_PADDED:
            vals = tl.load(src_ptrs)
        else:
            vals = tl.load(src_ptrs, mask=cols[None, :] < d, other=0.0)
        tl.store(tle.gpu.local_ptr(slot.kv), vals)
    else:
        # Copy the boundary-checked paged KV tile into the pipe's shared-memory slot.
        rows = tl.arange(0, BLOCK_N)
        cols = tl.arange(0, HEAD_DIM_PADDED)
        logical_idx = n_offset + rows
        cache_idx = _paged_blockwise_cache_indices(
            n_offset,
            rows,
            k_len,
            page_table_ptr_b,
            block_size,
            page_stride_rows,
            BLOCK_N,
            PAGED_GATHER_MODE,
            BOUNDARY_CHECK=True,
        )
        src_ptrs = src_base + cache_idx[:, None] * row_stride + cols[None, :]
        load_mask = (logical_idx[:, None] < k_len) & (cols[None, :] < d)
        # A single full-slot store lets the TLE async-store pass associate this
        # copy with writer.commit and lower it to cp.async.mbarrier.arrive.
        vals = tl.load(src_ptrs, mask=load_mask, other=0.0)
        tl.store(tle.gpu.local_ptr(slot.kv), vals)
    writer.commit(iteration)


@triton.jit
def _copy_dense_kv_tile_to_pipe(
    writer,
    iteration,
    src_base,
    row_stride,
    row_offset,
    row_count,
    d: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM_PADDED: tl.constexpr,
):
    """Publish one contiguous K/V tile through a TLE pipe."""

    slot = writer.acquire(iteration)
    rows = tl.arange(0, BLOCK_N)
    cols = tl.arange(0, HEAD_DIM_PADDED)
    logical_rows = row_offset + rows
    vals = tl.load(
        src_base + logical_rows[:, None] * row_stride + cols[None, :],
        mask=(logical_rows[:, None] < row_count) & (cols[None, :] < d),
        other=0.0,
    )
    tl.store(tle.gpu.local_ptr(slot.kv), vals)
    writer.commit(iteration)


@triton.jit
def _copy_dense_kv_tma_tile_to_pipe(
    writer,
    iteration,
    desc,
    row_offset,
    BLOCK_N: tl.constexpr,
    HEAD_DIM_PADDED: tl.constexpr,
):
    """Publish one contiguous K/V tile through a TMA descriptor and TLE pipe."""

    slot = writer.acquire(iteration)
    tle.gpu.copy(
        desc,
        slot.kv,
        [BLOCK_N, HEAD_DIM_PADDED],
        [row_offset, 0],
    )
    writer.commit(iteration)


@triton.jit
def _copy_paged_kv_tma_tile_to_pipe(
    writer,
    iteration,
    desc,
    page_table_ptr_b,
    n_offset,
    block_size: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM_PADDED: tl.constexpr,
):
    """Publish one page-contained K/V tile through TMA and a TLE pipe."""

    slot = writer.acquire(iteration)
    virtual_page = n_offset // block_size
    page_offset = n_offset % block_size
    physical_page = tl.load(page_table_ptr_b + virtual_page).to(tl.int32)
    tle.gpu.copy(
        desc,
        slot.kv,
        [1, BLOCK_N, HEAD_DIM_PADDED],
        [physical_page, page_offset, 0],
    )
    writer.commit(iteration)


@triton.jit
def _load_paged_kv_tma_tile(
    k_desc,
    v_desc,
    page_table_ptr_b,
    n_offset,
    block_size: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM_PADDED: tl.constexpr,
):
    """Load one page-contained K/V tile through Hopper TMA descriptors."""

    virtual_page = n_offset // block_size
    page_offset = n_offset % block_size
    physical_page = tl.load(page_table_ptr_b + virtual_page).to(tl.int32)
    k_tile = k_desc.load([physical_page, page_offset, 0])
    v_tile = v_desc.load([physical_page, page_offset, 0])
    return (
        tl.reshape(k_tile, [BLOCK_N, HEAD_DIM_PADDED]),
        tl.reshape(v_tile, [BLOCK_N, HEAD_DIM_PADDED]),
    )


@triton.jit
def _copy_dense_tile_to_smem(
    src_base,
    row_stride,
    smem_tile,
    row_offset,
    row_count,
    d: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    HEAD_DIM_PADDED: tl.constexpr,
):
    """Copy a contiguous two-dimensional tile into shared memory."""

    rows = tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, HEAD_DIM_PADDED)
    logical_rows = row_offset + rows
    src_ptrs = src_base + logical_rows[:, None] * row_stride + cols[None, :]
    mask = (logical_rows[:, None] < row_count) & (cols[None, :] < d)
    tle.gpu.copy(
        src_ptrs,
        smem_tile,
        [BLOCK_ROWS, HEAD_DIM_PADDED],
        mask=mask,
        other=0.0,
    )


@triton.jit
def _copy_packed_gqa_tile_to_smem(
    q_ptr,
    q_offset,
    q_row_stride,
    q_head_stride,
    kv_head,
    smem_tile,
    packed_row_offset,
    q_len,
    d: tl.constexpr,
    GQA_RATIO: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    HEAD_DIM_PADDED: tl.constexpr,
):
    """Pack query rows and GQA heads into a WGMMA M tile in shared memory."""

    rows = tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, HEAD_DIM_PADDED)
    packed_rows = packed_row_offset + rows
    query_rows = packed_rows // GQA_RATIO
    query_heads = kv_head * GQA_RATIO + packed_rows % GQA_RATIO
    src_ptrs = (
        q_ptr
        + q_offset
        + query_rows[:, None] * q_row_stride
        + query_heads[:, None] * q_head_stride
        + cols[None, :]
    )
    mask = (packed_rows[:, None] < q_len * GQA_RATIO) & (cols[None, :] < d)
    tle.gpu.copy(
        src_ptrs,
        smem_tile,
        [BLOCK_ROWS, HEAD_DIM_PADDED],
        mask=mask,
        other=0.0,
        # api args,
    )


@triton.jit
def _copy_dense_q_tile_to_pipe(
    writer,
    iteration,
    src_base,
    row_stride,
    row_offset,
    row_count,
    d: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    HEAD_DIM_PADDED: tl.constexpr,
):
    """Acquire, copy, and publish one contiguous query tile asynchronously."""

    slot = writer.acquire(iteration)
    rows = tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, HEAD_DIM_PADDED)
    logical_rows = row_offset + rows
    src_ptrs = src_base + logical_rows[:, None] * row_stride + cols[None, :]
    mask = (logical_rows[:, None] < row_count) & (cols[None, :] < d)
    tle.gpu.copy(
        src_ptrs,
        slot.q,
        [BLOCK_ROWS, HEAD_DIM_PADDED],
        mask=mask,
        other=0.0,
    )
    writer.commit(iteration)


@triton.jit
def _copy_packed_gqa_q_tile_to_pipe(
    writer,
    iteration,
    q_ptr,
    q_offset,
    q_row_stride,
    q_head_stride,
    kv_head,
    packed_row_offset,
    q_len,
    d: tl.constexpr,
    GQA_RATIO: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    HEAD_DIM_PADDED: tl.constexpr,
):
    """Acquire, pack, and publish one GQA query tile asynchronously."""

    slot = writer.acquire(iteration)
    rows = tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, HEAD_DIM_PADDED)
    packed_rows = packed_row_offset + rows
    query_rows = packed_rows // GQA_RATIO
    query_heads = kv_head * GQA_RATIO + packed_rows % GQA_RATIO
    src_ptrs = (
        q_ptr
        + q_offset
        + query_rows[:, None] * q_row_stride
        + query_heads[:, None] * q_head_stride
        + cols[None, :]
    )
    mask = (packed_rows[:, None] < q_len * GQA_RATIO) & (cols[None, :] < d)
    tle.gpu.copy(
        src_ptrs,
        slot.q,
        [BLOCK_ROWS, HEAD_DIM_PADDED],
        mask=mask,
        other=0.0,
    )
    writer.commit(iteration)


@triton.jit
def _store_packed_gqa_tile_from_regs(
    o_ptr,
    o_offset,
    o_row_stride,
    o_head_stride,
    kv_head,
    vals,
    packed_row_offset,
    q_len,
    d: tl.constexpr,
    GQA_RATIO: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    HEAD_DIM_PADDED: tl.constexpr,
):
    """Unpack a register GQA tile and store valid values to the output tensor."""

    packed_rows = packed_row_offset + tl.arange(0, BLOCK_ROWS)
    query_rows = packed_rows // GQA_RATIO
    query_heads = kv_head * GQA_RATIO + packed_rows % GQA_RATIO
    cols = tl.arange(0, HEAD_DIM_PADDED)
    ptrs = (
        o_ptr
        + o_offset
        + query_rows[:, None] * o_row_stride
        + query_heads[:, None] * o_head_stride
        + cols[None, :]
    )
    mask = (packed_rows[:, None] < q_len * GQA_RATIO) & (cols[None, :] < d)
    tl.store(ptrs, vals, mask=mask)
