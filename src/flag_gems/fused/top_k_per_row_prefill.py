"""Pure Triton top_k_per_row_prefill - 4-way merge via sort(4096).

NO torch.topk or torch.argsort.

Strategy: Merge 4 top-k lists at once using tl.sort(4096).
  - Stage 1: 32 chunks × sort(4096) → extract top-1024 each
  - Stage 2: 8 groups × sort(4096) (merge 4×1024) → 8 top-1024 lists
  - Stage 3: 2 groups × sort(4096) (merge 4×1024) → 2 top-1024 lists
  - Stage 4: 1 group × sort(2048) (merge 2×1024) → final top-1024
Total: 4 serial stages instead of 5.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _mask_invalid_kernel(
    logits_ptr,
    row_starts_ptr,
    row_ends_ptr,
    stride0,
    BLOCK_SIZE: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
):
    """Mask logits outside [row_starts[i], row_ends[i]) to -inf."""
    pid = tl.program_id(0)
    num_blocks_per_row = tl.cdiv(VOCAB_SIZE, BLOCK_SIZE)
    row_id = pid // num_blocks_per_row
    block_id = pid % num_blocks_per_row

    start = tl.load(row_starts_ptr + row_id)
    end = tl.load(row_ends_ptr + row_id)

    if start == 0 and end >= VOCAB_SIZE:
        return

    offs = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    out_of_range = (offs < start) | (offs >= end)
    mask = (offs < VOCAB_SIZE) & out_of_range

    tl.store(logits_ptr + row_id * stride0 + offs, float("-inf"), mask=mask)


@triton.jit
def _float_to_sortkey(values):
    """Convert float32 values to int64 sort keys (descending order)."""
    bits = values.to(tl.int32, bitcast=True)
    bits64 = bits.to(tl.int64)
    is_neg = bits64 < 0
    unsigned_ord = tl.where(is_neg, bits64 ^ 0xFFFFFFFF, bits64 ^ 0x80000000)
    signed_ord = unsigned_ord - 0x80000000
    return -signed_ord


@triton.jit
def _sort_chunk_extract_topk(
    logits_ptr,
    output_indices_ptr,
    stride0,
    output_stride0,
    output_stride1,
    vocab_size: tl.constexpr,
    top_k: tl.constexpr,
    num_chunks: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    """Sort a chunk and extract top-k indices."""
    pid = tl.program_id(0)
    row_id = pid // num_chunks
    chunk_id = pid % num_chunks
    chunk_start = chunk_id * CHUNK_SIZE
    row_offset = row_id * stride0

    offs = chunk_start + tl.arange(0, CHUNK_SIZE)
    mask = offs < vocab_size
    values = tl.load(logits_ptr + row_offset + offs, mask=mask, other=float("-inf"))

    sort_key = _float_to_sortkey(values)
    packed = (sort_key << 32) | offs.to(tl.int64)
    sorted_packed = tl.sort(packed)

    all_indices = (sorted_packed & 0xFFFFFFFF).to(tl.int32)
    out_offs = tl.arange(0, CHUNK_SIZE)
    out_mask = out_offs < top_k
    out_base = row_id * output_stride0 + chunk_id * output_stride1
    tl.store(output_indices_ptr + out_base + out_offs, all_indices, mask=out_mask)


@triton.jit
def _merge_4way_topk(
    input_indices_ptr,
    logits_ptr,
    output_indices_ptr,
    stride0,
    in_stride0,
    in_stride1,
    out_stride0,
    out_stride1,
    num_groups: tl.constexpr,
    top_k: tl.constexpr,
    SORT_SIZE: tl.constexpr,
):
    """4-way merge: combine 4 top-k lists into 1 via sort(4*top_k).

    Grid: (num_rows * num_groups,)
    SORT_SIZE must be next_power_of_2(4 * top_k).
    """
    pid = tl.program_id(0)
    row_id = pid // num_groups
    group_id = pid % num_groups
    row_offset = row_id * stride0

    sort_offs = tl.arange(0, SORT_SIZE)
    quarter = SORT_SIZE // 4

    q0 = sort_offs < quarter
    q1 = (sort_offs >= quarter) & (sort_offs < 2 * quarter)
    q2 = (sort_offs >= 2 * quarter) & (sort_offs < 3 * quarter)
    q3 = sort_offs >= 3 * quarter

    local_offs = sort_offs % quarter
    valid0 = q0 & (local_offs < top_k)
    valid1 = q1 & (local_offs < top_k)
    valid2 = q2 & (local_offs < top_k)
    valid3 = q3 & (local_offs < top_k)

    base0 = row_id * in_stride0 + (group_id * 4 + 0) * in_stride1
    base1 = row_id * in_stride0 + (group_id * 4 + 1) * in_stride1
    base2 = row_id * in_stride0 + (group_id * 4 + 2) * in_stride1
    base3 = row_id * in_stride0 + (group_id * 4 + 3) * in_stride1

    idx0 = tl.load(input_indices_ptr + base0 + local_offs, mask=valid0, other=0)
    idx1 = tl.load(input_indices_ptr + base1 + local_offs, mask=valid1, other=0)
    idx2 = tl.load(input_indices_ptr + base2 + local_offs, mask=valid2, other=0)
    idx3 = tl.load(input_indices_ptr + base3 + local_offs, mask=valid3, other=0)

    combined_idx = tl.where(q0, idx0, tl.where(q1, idx1, tl.where(q2, idx2, idx3)))
    valid = valid0 | valid1 | valid2 | valid3

    combined_val = tl.load(
        logits_ptr + row_offset + combined_idx, mask=valid, other=float("-inf")
    )

    sort_key = _float_to_sortkey(combined_val)
    packed = (sort_key << 32) | combined_idx.to(tl.int64)
    sorted_packed = tl.sort(packed)

    sorted_indices = (sorted_packed & 0xFFFFFFFF).to(tl.int32)
    out_base = row_id * out_stride0 + group_id * out_stride1
    out_mask = sort_offs < top_k
    tl.store(output_indices_ptr + out_base + sort_offs, sorted_indices, mask=out_mask)


@triton.jit
def _merge_2way_topk(
    input_indices_ptr,
    logits_ptr,
    output_indices_ptr,
    stride0,
    in_stride0,
    in_stride1,
    out_stride0,
    out_stride1,
    num_pairs: tl.constexpr,
    top_k: tl.constexpr,
    MERGE_SIZE: tl.constexpr,
):
    """2-way merge: combine 2 top-k lists into 1."""
    pid = tl.program_id(0)
    row_id = pid // num_pairs
    pair_id = pid % num_pairs
    row_offset = row_id * stride0

    merge_offs = tl.arange(0, MERGE_SIZE)
    half = MERGE_SIZE // 2
    first_half = merge_offs < half
    local_offs = merge_offs % half

    valid_first = first_half & (local_offs < top_k)
    valid_second = ~first_half & (local_offs < top_k)

    a_base = row_id * in_stride0 + (pair_id * 2) * in_stride1
    b_base = row_id * in_stride0 + (pair_id * 2 + 1) * in_stride1

    idx_a = tl.load(input_indices_ptr + a_base + local_offs, mask=valid_first, other=0)
    idx_b = tl.load(input_indices_ptr + b_base + local_offs, mask=valid_second, other=0)

    combined_idx = tl.where(first_half, idx_a, idx_b)
    valid = valid_first | valid_second

    combined_val = tl.load(
        logits_ptr + row_offset + combined_idx, mask=valid, other=float("-inf")
    )

    sort_key = _float_to_sortkey(combined_val)
    packed = (sort_key << 32) | combined_idx.to(tl.int64)
    sorted_packed = tl.sort(packed)

    sorted_indices = (sorted_packed & 0xFFFFFFFF).to(tl.int32)
    out_base = row_id * out_stride0 + pair_id * out_stride1
    out_mask = merge_offs < top_k
    tl.store(output_indices_ptr + out_base + merge_offs, sorted_indices, mask=out_mask)


@triton.jit
def _fused_postprocess_kernel(
    src_ptr,
    dst_ptr,
    row_starts_ptr,
    num_rows: tl.constexpr,
    top_k: tl.constexpr,
    src_stride0: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Convert absolute indices to row-relative indices."""
    row_id = tl.program_id(0)
    if row_id >= num_rows:
        return

    row_start = tl.load(row_starts_ptr + row_id)

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < top_k

    src_idx = row_id * src_stride0 + offs
    src_vals = tl.load(src_ptr + src_idx, mask=mask, other=0)

    dst_vals = (src_vals - row_start).to(tl.int32)

    dst_idx = row_id * top_k + offs
    tl.store(dst_ptr + dst_idx, dst_vals, mask=mask)


def top_k_per_row_prefill(
    logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
):
    """Pure Triton top-k with 4-way merge using sort(4096)."""
    vocab_size = logits.shape[1]

    if top_k > vocab_size:
        raise ValueError(f"top_k ({top_k}) must not exceed vocab_size ({vocab_size})")

    # Phase 1: Mask invalid ranges
    MASK_BS = 8192
    num_mask_blocks = triton.cdiv(vocab_size, MASK_BS)
    _mask_invalid_kernel[(num_rows * num_mask_blocks,)](
        logits,
        row_starts,
        row_ends,
        stride0,
        BLOCK_SIZE=MASK_BS,
        VOCAB_SIZE=vocab_size,
        num_warps=2,
    )

    # Phase 2: Sort-based top-k with 4-way merge
    CHUNK_SIZE = 4096
    num_chunks = triton.cdiv(vocab_size, CHUNK_SIZE)  # 32
    K_BLOCK = triton.next_power_of_2(top_k)  # 1024
    MERGE_2WAY = 2 * K_BLOCK  # 2048
    SORT_4WAY = 4 * K_BLOCK  # 4096

    # Pre-allocate buffers (use fixed buffer_stride = num_chunks * top_k)
    buffer_stride0 = num_chunks * top_k
    buffer_a = torch.empty(
        (num_rows, num_chunks, top_k), dtype=torch.int32, device=logits.device
    )
    buffer_b = torch.empty(
        (num_rows, num_chunks, top_k), dtype=torch.int32, device=logits.device
    )

    # Stage 1: Sort each chunk, extract top-k
    _sort_chunk_extract_topk[(num_rows * num_chunks,)](
        logits,
        buffer_a,
        stride0=stride0,
        output_stride0=buffer_stride0,
        output_stride1=top_k,
        vocab_size=vocab_size,
        top_k=top_k,
        num_chunks=num_chunks,
        CHUNK_SIZE=CHUNK_SIZE,
        num_warps=8,
    )

    # Stage 2: 4-way merge (32→8)
    num_groups_4 = num_chunks // 4  # 8
    _merge_4way_topk[(num_rows * num_groups_4,)](
        buffer_a,
        logits,
        buffer_b,
        stride0=stride0,
        in_stride0=buffer_stride0,
        in_stride1=top_k,
        out_stride0=buffer_stride0,
        out_stride1=top_k,
        num_groups=num_groups_4,
        top_k=top_k,
        SORT_SIZE=SORT_4WAY,
        num_warps=8,
    )

    # Stage 3: 4-way merge (8→2)
    num_groups_4b = num_groups_4 // 4  # 2
    _merge_4way_topk[(num_rows * num_groups_4b,)](
        buffer_b,
        logits,
        buffer_a,
        stride0=stride0,
        in_stride0=buffer_stride0,
        in_stride1=top_k,
        out_stride0=buffer_stride0,
        out_stride1=top_k,
        num_groups=num_groups_4b,
        top_k=top_k,
        SORT_SIZE=SORT_4WAY,
        num_warps=8,
    )

    # Stage 4: 2-way merge (2→1)
    _merge_2way_topk[(num_rows * 1,)](
        buffer_a,
        logits,
        buffer_b,
        stride0=stride0,
        in_stride0=buffer_stride0,
        in_stride1=top_k,
        out_stride0=buffer_stride0,
        out_stride1=top_k,
        num_pairs=1,
        top_k=top_k,
        MERGE_SIZE=MERGE_2WAY,
        num_warps=8,
    )

    # Copy final result
    indices.copy_(buffer_b[:, 0, :])

    # Phase 3: Postprocess
    POSTPROC_BLOCK = triton.next_power_of_2(top_k)
    _fused_postprocess_kernel[(num_rows,)](
        indices,
        indices,
        row_starts,
        num_rows=num_rows,
        top_k=top_k,
        src_stride0=top_k,
        BLOCK_SIZE=POSTPROC_BLOCK,
        num_warps=4,
    )
