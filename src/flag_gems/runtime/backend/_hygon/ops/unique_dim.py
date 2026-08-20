import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.libentry import libentry

logger = logging.getLogger(__name__)

_UNIQUE_DIM_COMPARE_BLOCK_SIZE = 1024
_UNIQUE_DIM_GATHER_BLOCK_SIZE = 1024
# Largest row count handled by the single-launch group-id scan kernel.
_UNIQUE_DIM_GROUP_SCAN_BLOCK_SIZE = 4096
# Largest key count sorted by the single-launch rank-sort kernel. Above this we
# delegate to ``torch.sort`` which, under FlagGems op interception, dispatches to
# the backend's Triton radix sort. Rank-sort is O(N^2) but a single launch, so it
# is much cheaper than a 16-pass int64 radix sort for tiny shapes.
_UNIQUE_DIM_RANK_SORT_MAX_KEYS = 2048
_UNIQUE_DIM_TILED_RANK_SORT_MAX_KEYS = 4096
_UNIQUE_DIM_RANK_SORT_TILE_SIZE = 1024
_UNIQUE_DIM_SEGMENT_RANK_MAX_GROUP = 64
_UNIQUE_DIM_SMALL_LEX2_MAX_KEYS = 1024
_UNIQUE_DIM_HASH_MIN_ROW_LEN = 1024
# Smaller tile for the fused key kernel's float branches: their int64 bit-twiddle
# temporaries overflow the Ascend unified buffer at the default tile size.
_UNIQUE_DIM_BUILD_KEY_FLOAT_BLOCK_SIZE = 256

# Per-column bit budgets and to-int64 conversions that preserve the original
# value ordering. The encodings let us pack a per-row ``group_id`` together
# with a single column's key into one int64 that, when compared as signed
# int64, matches the lex order over ``(group_id, signed_value)``.
_INT_DTYPE_BITS = {
    torch.bool: 1,
    torch.int8: 8,
    torch.uint8: 8,
    torch.int16: 16,
    torch.int32: 32,
    torch.float16: 16,
    torch.bfloat16: 16,
    torch.float32: 32,
}


@libentry()
@triton.jit
def _unique_dim_argsort_rank_kernel(
    keys_ptr: tl.tensor,
    indices_ptr: tl.tensor,
    sorted_keys_ptr: tl.tensor,
    num_keys: int,
    BLOCK_SIZE: tl.constexpr,
    STORE_SORTED_KEYS: tl.constexpr,
):
    row = ext.program_id(0)
    candidates = tl.arange(0, BLOCK_SIZE)
    mask = candidates < num_keys

    cur = tl.load(keys_ptr + row)
    vals = tl.load(keys_ptr + candidates, mask=mask, other=cur)
    before = ((vals < cur) | ((vals == cur) & (candidates < row))) & mask
    rank = tl.sum(before.to(tl.int32), axis=0)
    tl.store(indices_ptr + rank, row)
    if STORE_SORTED_KEYS:
        tl.store(sorted_keys_ptr + rank, cur)


@libentry()
@triton.jit
def _unique_dim_argsort_rank_tiled_count_kernel(
    keys_ptr: tl.tensor,
    partial_rank_ptr: tl.tensor,
    num_keys: int,
    num_chunks: int,
    BLOCK_SIZE: tl.constexpr,
):
    row = ext.program_id(0)
    chunk = ext.program_id(1)
    candidates = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = candidates < num_keys

    cur = tl.load(keys_ptr + row)
    vals = tl.load(keys_ptr + candidates, mask=mask, other=cur)
    before = ((vals < cur) | ((vals == cur) & (candidates < row))) & mask
    rank_part = tl.sum(before.to(tl.int32), axis=0)
    tl.store(partial_rank_ptr + row * num_chunks + chunk, rank_part)


@libentry()
@triton.jit
def _unique_dim_argsort_rank_tiled_scatter_kernel(
    keys_ptr: tl.tensor,
    partial_rank_ptr: tl.tensor,
    indices_ptr: tl.tensor,
    sorted_keys_ptr: tl.tensor,
    num_chunks: int,
    BLOCK_CHUNKS: tl.constexpr,
    STORE_SORTED_KEYS: tl.constexpr,
):
    row = ext.program_id(0)
    chunks = tl.arange(0, BLOCK_CHUNKS)
    mask = chunks < num_chunks
    parts = tl.load(partial_rank_ptr + row * num_chunks + chunks, mask=mask, other=0)
    rank = tl.sum(parts, axis=0)
    tl.store(indices_ptr + rank, row)
    if STORE_SORTED_KEYS:
        cur = tl.load(keys_ptr + row)
        tl.store(sorted_keys_ptr + rank, cur)


@libentry()
@triton.jit
def _unique_dim_argsort_lex2_rank_kernel(
    flat_ptr: tl.tensor,
    indices_ptr: tl.tensor,
    duplicate_flags_ptr: tl.tensor,
    num_rows: int,
    row_stride: int,
    KEY_OFFSET: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_COL1: tl.constexpr,
):
    row = ext.program_id(0)
    candidates = tl.arange(0, BLOCK_SIZE)
    mask = candidates < num_rows

    cur0 = tl.load(flat_ptr + row * row_stride).to(tl.int64) + KEY_OFFSET
    vals0 = tl.load(flat_ptr + candidates * row_stride, mask=mask, other=0).to(tl.int64)
    vals0 = vals0 + KEY_OFFSET

    if HAS_COL1:
        cur1 = tl.load(flat_ptr + row * row_stride + 1).to(tl.int64) + KEY_OFFSET
        vals1 = tl.load(flat_ptr + candidates * row_stride + 1, mask=mask, other=0).to(
            tl.int64
        )
        vals1 = vals1 + KEY_OFFSET
        key_less = (vals0 < cur0) | ((vals0 == cur0) & (vals1 < cur1))
        key_equal = (vals0 == cur0) & (vals1 == cur1)
    else:
        key_less = vals0 < cur0
        key_equal = vals0 == cur0

    before = (key_less | (key_equal & (candidates < row))) & mask
    duplicate = key_equal & (candidates != row) & mask
    rank = tl.sum(before.to(tl.int32), axis=0)
    has_duplicate = tl.sum(duplicate.to(tl.int32), axis=0) != 0
    tl.store(indices_ptr + rank, row)
    tl.store(duplicate_flags_ptr + row, has_duplicate)


@libentry()
@triton.jit
def _unique_dim_argsort_dim1_2d_lex2_rank_kernel(
    input_ptr: tl.tensor,
    indices_ptr: tl.tensor,
    duplicate_flags_ptr: tl.tensor,
    num_cols: int,
    outer_rows: int,
    KEY_OFFSET: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_ROW1: tl.constexpr,
):
    col = ext.program_id(0)
    candidates = tl.arange(0, BLOCK_SIZE)
    mask = candidates < num_cols

    cur0 = tl.load(input_ptr + col).to(tl.int64) + KEY_OFFSET
    vals0 = tl.load(input_ptr + candidates, mask=mask, other=0).to(tl.int64)
    vals0 = vals0 + KEY_OFFSET

    if HAS_ROW1:
        cur1 = tl.load(input_ptr + num_cols + col).to(tl.int64) + KEY_OFFSET
        vals1 = tl.load(input_ptr + num_cols + candidates, mask=mask, other=0).to(
            tl.int64
        )
        vals1 = vals1 + KEY_OFFSET
        key_less = (vals0 < cur0) | ((vals0 == cur0) & (vals1 < cur1))
        key_equal = (vals0 == cur0) & (vals1 == cur1)
    else:
        key_less = vals0 < cur0
        key_equal = vals0 == cur0

    before = (key_less | (key_equal & (candidates < col))) & mask
    duplicate = key_equal & (candidates != col) & mask
    rank = tl.sum(before.to(tl.int32), axis=0)
    has_duplicate = tl.sum(duplicate.to(tl.int32), axis=0) != 0
    tl.store(indices_ptr + rank, col)
    tl.store(duplicate_flags_ptr + col, has_duplicate)


@libentry()
@triton.jit
def _unique_dim_extract_col_kernel(
    flat_ptr: tl.tensor,
    col_ptr: tl.tensor,
    num_rows: int,
    row_stride: int,
    col: int,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_rows
    values = tl.load(flat_ptr + offsets * row_stride + col, mask=mask, other=0)
    tl.store(col_ptr + offsets, values, mask=mask)


@libentry()
@triton.jit
def _unique_dim_segmented_rank_refine_kernel(
    flat_ptr: tl.tensor,
    sorted_indices_ptr: tl.tensor,
    group_id_ptr: tl.tensor,
    refined_indices_ptr: tl.tensor,
    refined_keys_ptr: tl.tensor,
    status_ptr: tl.tensor,
    num_rows: int,
    row_stride: int,
    col: int,
    KEY_OFFSET: tl.constexpr,
    KEY_SCALE: tl.constexpr,
    REMAP_KIND: tl.constexpr,
    BLOCK_GROUP: tl.constexpr,
):
    pos = ext.program_id(0)
    gid = tl.load(group_id_ptr + pos)
    cur_row = tl.load(sorted_indices_ptr + pos)
    cur_base = cur_row * row_stride + col

    if REMAP_KIND == 0:  # _REMAP_INT
        cur_x = tl.load(flat_ptr + cur_base).to(tl.int64)
        cur_val = cur_x + KEY_OFFSET
    elif REMAP_KIND == 1:  # _REMAP_FP16
        cur_bits = tl.load(flat_ptr + cur_base).to(tl.int64) & 0xFFFF
        cur_sign = (cur_bits & 0x8000) != 0
        cur_val = tl.where(cur_sign, cur_bits ^ 0xFFFF, cur_bits ^ 0x8000)
    else:  # _REMAP_FP32
        cur_bits = tl.load(flat_ptr + cur_base).to(tl.int64) & 0xFFFFFFFF
        cur_sign = (cur_bits & 0x80000000) != 0
        cur_val = tl.where(cur_sign, cur_bits ^ 0xFFFFFFFF, cur_bits ^ 0x80000000)

    start = pos
    for step in tl.static_range(1, BLOCK_GROUP):
        prev = pos - step
        same = prev >= 0
        same = same & (tl.load(group_id_ptr + prev, mask=prev >= 0, other=gid) == gid)
        start = tl.where(same, prev, start)
    overflow = (pos >= BLOCK_GROUP) & (
        tl.load(group_id_ptr + pos - BLOCK_GROUP, mask=pos >= BLOCK_GROUP, other=gid)
        == gid
    )

    cand = start + tl.arange(0, BLOCK_GROUP)
    cand_mask = (cand < num_rows) & (
        tl.load(group_id_ptr + cand, mask=cand < num_rows, other=-1) == gid
    )
    cand_rows = tl.load(sorted_indices_ptr + cand, mask=cand_mask, other=0)
    cand_base = cand_rows * row_stride + col

    if REMAP_KIND == 0:  # _REMAP_INT
        cand_x = tl.load(flat_ptr + cand_base, mask=cand_mask, other=0).to(tl.int64)
        cand_val = cand_x + KEY_OFFSET
    elif REMAP_KIND == 1:  # _REMAP_FP16
        cand_bits = (
            tl.load(flat_ptr + cand_base, mask=cand_mask, other=0).to(tl.int64) & 0xFFFF
        )
        cand_sign = (cand_bits & 0x8000) != 0
        cand_val = tl.where(cand_sign, cand_bits ^ 0xFFFF, cand_bits ^ 0x8000)
    else:  # _REMAP_FP32
        cand_bits = (
            tl.load(flat_ptr + cand_base, mask=cand_mask, other=0).to(tl.int64)
            & 0xFFFFFFFF
        )
        cand_sign = (cand_bits & 0x80000000) != 0
        cand_val = tl.where(cand_sign, cand_bits ^ 0xFFFFFFFF, cand_bits ^ 0x80000000)

    before = ((cand_val < cur_val) | ((cand_val == cur_val) & (cand < pos))) & cand_mask
    same_value = (cand_val == cur_val) & (cand != pos) & cand_mask
    has_duplicate = tl.sum(same_value.to(tl.int32), axis=0) != 0
    status = tl.where(overflow, 2, has_duplicate.to(tl.int32))
    tl.atomic_max(status_ptr, status)

    rank = tl.sum(before.to(tl.int32), axis=0)
    dest = start + rank
    tl.store(refined_indices_ptr + dest, cur_row)
    tl.store(refined_keys_ptr + dest, gid * KEY_SCALE + cur_val)


@libentry()
@triton.jit
def _unique_dim_gather_1d_kernel(
    input_ptr: tl.tensor,
    index_ptr: tl.tensor,
    output_ptr: tl.tensor,
    num_elements: int,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    indices = tl.load(index_ptr + offsets, mask=mask, other=0)
    values = tl.load(input_ptr + indices, mask=mask)
    tl.store(output_ptr + offsets, values, mask=mask)


@libentry()
@triton.jit
def _unique_dim_group_id_kernel(
    composite_ptr: tl.tensor,
    group_id_ptr: tl.tensor,
    last_group_id_ptr: tl.tensor,
    num_rows: int,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_rows
    cur = tl.load(composite_ptr + offsets, mask=mask, other=0)
    prev_offsets = tl.where(offsets == 0, 0, offsets - 1)
    prev = tl.load(composite_ptr + prev_offsets, mask=offsets > 0, other=cur)
    diff = ((cur - prev) != 0) & mask
    diff = tl.where(offsets == 0, False, diff)
    group_id = tl.cumsum(diff.to(tl.int64), axis=0)
    tl.store(group_id_ptr + offsets, group_id, mask=mask)
    last = tl.sum(tl.where(offsets == num_rows - 1, group_id, 0), axis=0)
    tl.store(last_group_id_ptr, last)


@libentry()
@triton.jit
def _unique_dim_row_hash_chunk_kernel(
    flat_ptr: tl.tensor,
    chunk_hash_ptr: tl.tensor,
    num_rows: int,
    row_len: int,
    num_chunks: int,
    BLOCK_SIZE: tl.constexpr,
):
    row = ext.program_id(0)
    chunk = ext.program_id(1)
    offsets = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < row_len

    vals = tl.load(flat_ptr + row * row_len + offsets, mask=mask, other=0)
    vals_i64 = vals.to(tl.int64)
    offsets_i64 = offsets.to(tl.int64)
    mix = (vals_i64 + (offsets_i64 + 1) * 1009 + 9176) * 131071
    mix = tl.where(mask, mix, 0)
    tl.store(chunk_hash_ptr + row * num_chunks + chunk, tl.sum(mix, axis=0))


@libentry()
@triton.jit
def _unique_dim_row_hash_reduce_kernel(
    chunk_hash_ptr: tl.tensor,
    row_hash_ptr: tl.tensor,
    num_chunks: int,
    BLOCK_CHUNKS: tl.constexpr,
):
    row = ext.program_id(0)
    chunks = tl.arange(0, BLOCK_CHUNKS)
    mask = chunks < num_chunks
    vals = tl.load(chunk_hash_ptr + row * num_chunks + chunks, mask=mask, other=0)
    tl.store(row_hash_ptr + row, tl.sum(vals, axis=0))


@libentry()
@triton.jit
def _unique_dim_row_chunk_diff_kernel(
    flat_ptr: tl.tensor,
    sorted_indices_ptr: tl.tensor,
    row_chunk_diff_ptr: tl.tensor,
    num_rows: int,
    row_len: int,
    num_chunks: int,
    BLOCK_SIZE: tl.constexpr,
):
    row = ext.program_id(0)
    chunk = ext.program_id(1)
    offsets = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < row_len

    out = tl.full((), 0, dtype=tl.int32)
    if row == 0:
        out = tl.where(chunk == 0, 1, 0)
    else:
        cur_row = tl.load(sorted_indices_ptr + row)
        prev_row = tl.load(sorted_indices_ptr + row - 1)
        cur = tl.load(flat_ptr + cur_row * row_len + offsets, mask=mask)
        prev = tl.load(flat_ptr + prev_row * row_len + offsets, mask=mask)
        neq = (cur != prev) & mask
        has_diff = tl.sum(neq.to(tl.int32), axis=0) != 0
        out = has_diff.to(tl.int32)
    tl.store(row_chunk_diff_ptr + row * num_chunks + chunk, out)


@libentry()
@triton.jit
def _unique_dim_row_chunk_diff_hash_kernel(
    flat_ptr: tl.tensor,
    sorted_indices_ptr: tl.tensor,
    row_hash_ptr: tl.tensor,
    row_chunk_diff_ptr: tl.tensor,
    num_rows: int,
    row_len: int,
    num_chunks: int,
    BLOCK_SIZE: tl.constexpr,
):
    row = ext.program_id(0)
    chunk = ext.program_id(1)
    offsets = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < row_len

    out = tl.full((), 0, dtype=tl.int32)
    if row == 0:
        out = tl.where(chunk == 0, 1, 0)
    else:
        cur_row = tl.load(sorted_indices_ptr + row)
        prev_row = tl.load(sorted_indices_ptr + row - 1)
        cur_hash = tl.load(row_hash_ptr + cur_row)
        prev_hash = tl.load(row_hash_ptr + prev_row)
        if cur_hash != prev_hash:
            out = tl.where(chunk == 0, 1, 0)
        else:
            cur = tl.load(flat_ptr + cur_row * row_len + offsets, mask=mask)
            prev = tl.load(flat_ptr + prev_row * row_len + offsets, mask=mask)
            neq = (cur != prev) & mask
            has_diff = tl.sum(neq.to(tl.int32), axis=0) != 0
            out = has_diff.to(tl.int32)
    tl.store(row_chunk_diff_ptr + row * num_chunks + chunk, out)


@libentry()
@triton.jit
def _unique_dim_row_diff_reduce_kernel(
    row_chunk_diff_ptr: tl.tensor,
    is_first_ptr: tl.tensor,
    num_chunks: int,
    BLOCK_CHUNKS: tl.constexpr,
):
    row = ext.program_id(0)
    chunks = tl.arange(0, BLOCK_CHUNKS)
    mask = chunks < num_chunks
    vals = tl.load(row_chunk_diff_ptr + row * num_chunks + chunks, mask=mask, other=0)
    tl.store(is_first_ptr + row, tl.sum(vals, axis=0) != 0)


@libentry()
@triton.jit
def _unique_dim_row_single_chunk_first_kernel(
    flat_ptr: tl.tensor,
    sorted_indices_ptr: tl.tensor,
    is_first_ptr: tl.tensor,
    num_rows: int,
    row_len: int,
    BLOCK_SIZE: tl.constexpr,
):
    row = ext.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < row_len

    out = tl.full((), True, dtype=tl.int1)
    if row != 0:
        cur_row = tl.load(sorted_indices_ptr + row)
        prev_row = tl.load(sorted_indices_ptr + row - 1)
        cur = tl.load(flat_ptr + cur_row * row_len + offsets, mask=mask)
        prev = tl.load(flat_ptr + prev_row * row_len + offsets, mask=mask)
        neq = (cur != prev) & mask
        out = tl.sum(neq.to(tl.int32), axis=0) != 0
    tl.store(is_first_ptr + row, out)


@libentry()
@triton.jit
def _unique_dim_gather_moved_kernel(
    flat_ptr: tl.tensor,
    unique_indices_ptr: tl.tensor,
    output_ptr: tl.tensor,
    num_unique: int,
    row_len: int,
    BLOCK_SIZE: tl.constexpr,
):
    # One program per (output row, column chunk). Copies a contiguous span of
    # the source row selected through ``unique_indices`` into the matching span
    # of the output row. Loading ``src_row`` once per program (scalar) and using
    # contiguous column offsets avoids the per-element integer divide/modulo and
    # scattered indexing of a flat-offset gather, which dominate NPU time.
    row = ext.program_id(0)
    chunk = ext.program_id(1)
    col = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col < row_len

    src_row = tl.load(unique_indices_ptr + row)
    values = tl.load(flat_ptr + src_row * row_len + col, mask=mask)
    tl.store(output_ptr + row * row_len + col, values, mask=mask)


@libentry()
@triton.jit
def _unique_dim_gather_moved_inverse_kernel(
    flat_ptr: tl.tensor,
    unique_indices_ptr: tl.tensor,
    output_ptr: tl.tensor,
    inverse_ptr: tl.tensor,
    num_unique: int,
    row_len: int,
    BLOCK_SIZE: tl.constexpr,
):
    row = ext.program_id(0)
    chunk = ext.program_id(1)
    col = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col < row_len

    src_row = tl.load(unique_indices_ptr + row)
    values = tl.load(flat_ptr + src_row * row_len + col, mask=mask)
    tl.store(output_ptr + row * row_len + col, values, mask=mask)
    if chunk == 0:
        tl.store(inverse_ptr + src_row, row)


@libentry()
@triton.jit
def _unique_dim_gather_dim1_2d_inverse_kernel(
    input_ptr: tl.tensor,
    unique_indices_ptr: tl.tensor,
    output_ptr: tl.tensor,
    inverse_ptr: tl.tensor,
    num_unique: int,
    outer_rows: int,
    input_cols: int,
    STORE_INVERSE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    out_col = ext.program_id(0)
    chunk = ext.program_id(1)
    rows = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = rows < outer_rows

    src_col = tl.load(unique_indices_ptr + out_col)
    values = tl.load(input_ptr + rows * input_cols + src_col, mask=mask)
    tl.store(output_ptr + rows * num_unique + out_col, values, mask=mask)
    if STORE_INVERSE and chunk == 0:
        tl.store(inverse_ptr + src_col, out_col)


@libentry()
@triton.jit
def _unique_dim_inverse_permutation_kernel(
    sorted_indices_ptr: tl.tensor,
    inverse_ptr: tl.tensor,
    num_rows: int,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_rows
    sorted_indices = tl.load(sorted_indices_ptr + offsets, mask=mask, other=0)
    tl.store(inverse_ptr + sorted_indices, offsets.to(tl.int64), mask=mask)


@libentry()
@triton.jit
def _unique_dim_inverse_kernel(
    sorted_indices_ptr: tl.tensor,
    inverse_sorted_ptr: tl.tensor,
    inverse_ptr: tl.tensor,
    num_rows: int,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_rows
    sorted_indices = tl.load(sorted_indices_ptr + offsets, mask=mask)
    inverse_sorted = tl.load(inverse_sorted_ptr + offsets, mask=mask)
    tl.store(inverse_ptr + sorted_indices, inverse_sorted, mask=mask)


@libentry()
@triton.jit
def _unique_dim_counts_kernel(
    first_positions_ptr: tl.tensor,
    counts_ptr: tl.tensor,
    num_rows: int,
    num_unique: int,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_unique
    positions = tl.load(first_positions_ptr + offsets, mask=mask)
    next_positions = tl.load(
        first_positions_ptr + offsets + 1,
        mask=(offsets + 1) < num_unique,
        other=num_rows,
    )
    tl.store(counts_ptr + offsets, next_positions - positions, mask=mask)


def _triton_num_warps(block_size: int) -> int:
    if block_size >= 8192:
        return 8
    if block_size >= 2048:
        return 4
    return 1


def _monotonic_key_bits(dtype: torch.dtype):
    """Return the per-element key width for ``dtype`` if it can be mapped
    into a monotonic int64 view, else ``None``."""
    return _INT_DTYPE_BITS.get(dtype)


# Monotonic-remap kinds for the fused key-build kernel.
_REMAP_INT = 0  # signed/unsigned int: value + KEY_OFFSET
_REMAP_FP16 = 1  # 16-bit float: order-preserving bit twiddle
_REMAP_FP32 = 2  # 32-bit float: order-preserving bit twiddle


def _remap_info(flat: torch.Tensor):
    """Return ``(int_view, remap_kind, key_offset)`` describing how to map this
    dtype to an order-preserving non-negative int64 in the fused key kernel.

    ``int_view`` reinterprets the buffer as an integer type the kernel can load
    directly (floats are bit-cast); the remap itself happens on-device.
    """
    dt = flat.dtype
    if dt == torch.bool:
        return flat.view(torch.uint8), _REMAP_INT, 0
    if dt == torch.uint8:
        return flat, _REMAP_INT, 0
    if dt == torch.int8:
        return flat, _REMAP_INT, 1 << 7
    if dt == torch.int16:
        return flat, _REMAP_INT, 1 << 15
    if dt == torch.int32:
        return flat, _REMAP_INT, 1 << 31
    if dt in (torch.float16, torch.bfloat16):
        return flat.view(torch.int16), _REMAP_FP16, 0
    if dt == torch.float32:
        return flat.view(torch.int32), _REMAP_FP32, 0
    raise NotImplementedError(dt)


@libentry()
@triton.jit
def _unique_dim_build_key_kernel(
    flat_ptr: tl.tensor,
    indices_ptr: tl.tensor,
    group_id_ptr: tl.tensor,
    out_ptr: tl.tensor,
    num_rows: int,
    row_stride: int,
    col: int,
    KEY_OFFSET: tl.constexpr,
    KEY_SCALE: tl.constexpr,
    REMAP_KIND: tl.constexpr,
    FIRST: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Build one cascade pass' composite key in a single launch.

    For ``FIRST`` (first column) the key is just the column's monotonic remap.
    Otherwise the row is fetched through the current permutation ``indices`` and
    the running ``group_id`` prefix is folded in as ``group_id * key_scale +
    value`` (multiply/add rather than shift/or, matching the rest of the file).

    This fuses what was a ``select -> contiguous -> cast -> add -> gather ->
    mul -> add`` chain of separate ops into one kernel, which is the dominant
    per-pass host/launch cost on backends with a native sort.
    """
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_rows

    if FIRST:
        row = offsets.to(tl.int64)
    else:
        row = tl.load(indices_ptr + offsets, mask=mask, other=0)
    base = row * row_stride + col

    if REMAP_KIND == 0:  # _REMAP_INT
        x = tl.load(flat_ptr + base, mask=mask, other=0).to(tl.int64)
        val = x + KEY_OFFSET
    elif REMAP_KIND == 1:  # _REMAP_FP16
        bits = tl.load(flat_ptr + base, mask=mask, other=0).to(tl.int64) & 0xFFFF
        sign = (bits & 0x8000) != 0
        val = tl.where(sign, bits ^ 0xFFFF, bits ^ 0x8000)
    else:  # _REMAP_FP32
        bits = tl.load(flat_ptr + base, mask=mask, other=0).to(tl.int64) & 0xFFFFFFFF
        sign = (bits & 0x80000000) != 0
        val = tl.where(sign, bits ^ 0xFFFFFFFF, bits ^ 0x80000000)

    if FIRST:
        out = val
    else:
        gid = tl.load(group_id_ptr + offsets, mask=mask, other=0)
        out = gid * KEY_SCALE + val
    tl.store(out_ptr + offsets, out, mask=mask)


def _build_composite_key(
    flat_view: torch.Tensor,
    col: int,
    indices: torch.Tensor | None,
    group_id: torch.Tensor | None,
    num_rows: int,
    row_stride: int,
    key_offset: int,
    key_scale: int,
    remap_kind: int,
) -> torch.Tensor:
    """One-launch composite key for cascade pass ``col``.

    ``indices``/``group_id`` are ``None`` on the first pass; otherwise they are
    the current permutation and running group ids.
    """
    out = torch.empty(num_rows, dtype=torch.int64, device=flat_view.device)
    first = indices is None
    # Triton needs valid tensor handles even for the unused pointers on the
    # first pass; the kernel guards their loads behind ``FIRST``.
    indices_arg = flat_view if first else indices
    group_id_arg = flat_view if first else group_id
    # The float bit-twiddle branches allocate several int64 temporaries per
    # element; at the default tile this overflows the Ascend unified buffer, so
    # floats use a smaller tile. Integer remap is light and keeps the full tile.
    base_block_size = (
        _UNIQUE_DIM_GATHER_BLOCK_SIZE
        if remap_kind == _REMAP_INT
        else _UNIQUE_DIM_BUILD_KEY_FLOAT_BLOCK_SIZE
    )
    block_size = min(base_block_size, triton.next_power_of_2(num_rows))
    grid = (triton.cdiv(num_rows, block_size), 1, 1)
    with torch_device_fn.device(flat_view.device.index):
        _unique_dim_build_key_kernel[grid](
            flat_view,
            indices_arg,
            group_id_arg,
            out,
            num_rows,
            row_stride,
            col,
            KEY_OFFSET=key_offset,
            KEY_SCALE=key_scale,
            REMAP_KIND=remap_kind,
            FIRST=first,
            BLOCK_SIZE=block_size,
            num_warps=4,
        )
    return out


def _can_segmented_refine(dtype: torch.dtype) -> bool:
    return dtype in (torch.int8, torch.uint8, torch.int16, torch.int32)


def _extract_column(flat: torch.Tensor, col: int) -> torch.Tensor:
    num_rows, row_stride = flat.shape
    out = torch.empty(num_rows, dtype=flat.dtype, device=flat.device)
    if num_rows == 0:
        return out
    grid = (triton.cdiv(num_rows, _UNIQUE_DIM_GATHER_BLOCK_SIZE), 1, 1)
    with torch_device_fn.device(flat.device.index):
        _unique_dim_extract_col_kernel[grid](
            flat,
            out,
            num_rows,
            row_stride,
            col,
            BLOCK_SIZE=_UNIQUE_DIM_GATHER_BLOCK_SIZE,
            num_warps=4,
        )
    return out


def _segmented_rank_refine(
    flat_view: torch.Tensor,
    sorted_indices: torch.Tensor,
    group_id: torch.Tensor,
    num_rows: int,
    row_stride: int,
    col: int,
    key_offset: int,
    key_scale: int,
    remap_kind: int,
) -> tuple[torch.Tensor, torch.Tensor, bool] | None:
    refined_indices = torch.empty_like(sorted_indices)
    refined_keys = torch.empty(
        num_rows, dtype=torch.int64, device=sorted_indices.device
    )
    status = torch.zeros((), dtype=torch.int32, device=sorted_indices.device)
    with torch_device_fn.device(sorted_indices.device.index):
        _unique_dim_segmented_rank_refine_kernel[(num_rows, 1, 1)](
            flat_view,
            sorted_indices,
            group_id,
            refined_indices,
            refined_keys,
            status,
            num_rows,
            row_stride,
            col,
            KEY_OFFSET=key_offset,
            KEY_SCALE=key_scale,
            REMAP_KIND=remap_kind,
            BLOCK_GROUP=_UNIQUE_DIM_SEGMENT_RANK_MAX_GROUP,
            num_warps=1,
        )
    status_value = int(status.item())
    if status_value >= 2:
        return None
    return refined_indices, refined_keys, status_value == 0


def _can_small_lex2_fast_path(dtype: torch.dtype) -> bool:
    return dtype in (torch.int8, torch.uint8, torch.int16, torch.int32)


def _unique_dim_small_lex2_fast_path(
    flat: torch.Tensor,
    moved: torch.Tensor,
    dim: int,
    input_shape: torch.Size,
    return_inverse: bool,
    return_counts: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    num_rows, row_len = flat.shape
    if (
        num_rows > _UNIQUE_DIM_SMALL_LEX2_MAX_KEYS
        or row_len == 0
        or not _can_small_lex2_fast_path(flat.dtype)
    ):
        return None

    flat_view, remap_kind, key_offset = _remap_info(flat)
    if remap_kind != _REMAP_INT:
        return None

    sorted_indices = torch.empty(num_rows, dtype=torch.int64, device=flat.device)
    duplicate_flags = torch.empty(num_rows, dtype=torch.bool, device=flat.device)
    block_size = triton.next_power_of_2(num_rows)
    with torch_device_fn.device(flat.device.index):
        _unique_dim_argsort_lex2_rank_kernel[(num_rows, 1, 1)](
            flat_view,
            sorted_indices,
            duplicate_flags,
            num_rows,
            row_len,
            KEY_OFFSET=key_offset,
            BLOCK_SIZE=block_size,
            HAS_COL1=row_len > 1,
            num_warps=_triton_num_warps(block_size),
        )
    if bool(duplicate_flags.cpu().any()):
        return None

    counts = torch.empty(0, dtype=torch.int64, device=flat.device)
    if return_counts:
        counts = torch.ones(num_rows, dtype=torch.int64, device=flat.device)

    if return_inverse:
        output, inverse_indices = _unique_dim_gather_output_and_inverse(
            moved, sorted_indices, dim, input_shape
        )
    else:
        inverse_indices = torch.empty(0, dtype=torch.int64, device=flat.device)
        output = _unique_dim_gather_output(moved, sorted_indices, dim, input_shape)
    return output, inverse_indices, counts


def _unique_dim_small_dim1_2d_fast_path(
    input: torch.Tensor,
    return_inverse: bool,
    return_counts: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if (
        input.ndim != 2
        or not input.is_contiguous()
        or input.shape[1] > _UNIQUE_DIM_SMALL_LEX2_MAX_KEYS
        or input.shape[1] == 0
        or input.shape[0] == 0
        or not _can_small_lex2_fast_path(input.dtype)
    ):
        return None

    num_cols = input.shape[1]
    outer_rows = input.shape[0]
    flat_view, remap_kind, key_offset = _remap_info(input)
    if remap_kind != _REMAP_INT:
        return None

    sorted_indices = torch.empty(num_cols, dtype=torch.int64, device=input.device)
    duplicate_flags = torch.empty(num_cols, dtype=torch.bool, device=input.device)
    block_size = triton.next_power_of_2(num_cols)
    with torch_device_fn.device(input.device.index):
        _unique_dim_argsort_dim1_2d_lex2_rank_kernel[(num_cols, 1, 1)](
            flat_view,
            sorted_indices,
            duplicate_flags,
            num_cols,
            outer_rows,
            KEY_OFFSET=key_offset,
            BLOCK_SIZE=block_size,
            HAS_ROW1=outer_rows > 1,
            num_warps=_triton_num_warps(block_size),
        )
    if bool(duplicate_flags.cpu().any()):
        return None

    output = torch.empty_like(input)
    inverse_indices = (
        torch.empty(num_cols, dtype=torch.int64, device=input.device)
        if return_inverse
        else torch.empty(0, dtype=torch.int64, device=input.device)
    )
    block_size = min(_UNIQUE_DIM_GATHER_BLOCK_SIZE, triton.next_power_of_2(outer_rows))
    grid = (num_cols, triton.cdiv(outer_rows, block_size), 1)
    inverse_arg = inverse_indices if return_inverse else sorted_indices
    with torch_device_fn.device(input.device.index):
        _unique_dim_gather_dim1_2d_inverse_kernel[grid](
            input,
            sorted_indices,
            output,
            inverse_arg,
            num_cols,
            outer_rows,
            num_cols,
            STORE_INVERSE=return_inverse,
            BLOCK_SIZE=block_size,
            num_warps=_triton_num_warps(block_size),
        )

    counts = torch.empty(0, dtype=torch.int64, device=input.device)
    if return_counts:
        counts = torch.ones(num_cols, dtype=torch.int64, device=input.device)
    return output, inverse_indices, counts


def _triton_gather_1d(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    num_elements = indices.numel()
    output = torch.empty(num_elements, dtype=values.dtype, device=values.device)
    if num_elements == 0:
        return output
    block_size = min(
        _UNIQUE_DIM_GATHER_BLOCK_SIZE, triton.next_power_of_2(num_elements)
    )
    grid = (triton.cdiv(num_elements, block_size), 1, 1)
    with torch_device_fn.device(values.device.index):
        _unique_dim_gather_1d_kernel[grid](
            values,
            indices,
            output,
            num_elements,
            BLOCK_SIZE=block_size,
            num_warps=_triton_num_warps(block_size),
        )
    return output


def _argsort_keys(keys: torch.Tensor):
    """Stable ascending argsort of a 1D key tensor.

    Returns ``(perm, sorted_keys)`` where ``sorted_keys = keys[perm]``.

    Small key counts use the single-launch Triton rank-sort kernel. Medium
    key counts use a tiled rank-sort path to avoid backend int64 radix sort,
    whose launch/pass overhead dominates the profiled 4096-key DCU case.
    Larger counts still delegate to ``torch.sort``.
    """
    num_keys = keys.numel()
    if num_keys == 0:
        return torch.empty(0, dtype=torch.int64, device=keys.device), keys
    keys_contig = keys.contiguous()
    if num_keys <= _UNIQUE_DIM_RANK_SORT_MAX_KEYS:
        perm = torch.empty(num_keys, dtype=torch.int64, device=keys.device)
        sorted_keys = torch.empty_like(keys)
        block_size = triton.next_power_of_2(num_keys)
        with torch_device_fn.device(keys.device.index):
            _unique_dim_argsort_rank_kernel[(num_keys, 1, 1)](
                keys_contig,
                perm,
                sorted_keys,
                num_keys,
                BLOCK_SIZE=block_size,
                STORE_SORTED_KEYS=True,
                num_warps=_triton_num_warps(block_size),
            )
        return perm, sorted_keys
    if num_keys <= _UNIQUE_DIM_TILED_RANK_SORT_MAX_KEYS:
        perm = torch.empty(num_keys, dtype=torch.int64, device=keys.device)
        sorted_keys = torch.empty_like(keys)
        block_size = min(
            _UNIQUE_DIM_RANK_SORT_TILE_SIZE,
            triton.next_power_of_2(num_keys),
        )
        num_chunks = triton.cdiv(num_keys, block_size)
        partial_rank = torch.empty(
            (num_keys, num_chunks), dtype=torch.int32, device=keys.device
        )
        with torch_device_fn.device(keys.device.index):
            _unique_dim_argsort_rank_tiled_count_kernel[(num_keys, num_chunks, 1)](
                keys_contig,
                partial_rank,
                num_keys,
                num_chunks,
                BLOCK_SIZE=block_size,
                num_warps=_triton_num_warps(block_size),
            )
            _unique_dim_argsort_rank_tiled_scatter_kernel[(num_keys, 1, 1)](
                keys_contig,
                partial_rank,
                perm,
                sorted_keys,
                num_chunks,
                BLOCK_CHUNKS=triton.next_power_of_2(num_chunks),
                STORE_SORTED_KEYS=True,
                num_warps=_triton_num_warps(triton.next_power_of_2(num_chunks)),
            )
        return perm, sorted_keys
    sorted_keys, perm = torch.sort(keys)
    return perm, sorted_keys


def _group_id_from_sorted(sorted_keys: torch.Tensor):
    """Dense lexicographic group ids for an ascending key tensor.

    Returns ``(group_id, last_group_id)`` where ``group_id[i]`` is the count of
    distinct key values strictly before position ``i`` and ``last_group_id`` is
    the (host-side) value of ``group_id[-1]`` (or ``-1`` when empty).

    Small row counts use the single-launch scan kernel. Larger counts use a
    safe ``int64`` adjacent-difference followed by ``torch.cumsum`` (a FlagGems
    multi-block scan under op interception). The difference is computed as
    ``int64 - int64`` then ``!= 0`` against a scalar; running through the
    registered tensor-vs-tensor comparison op would route int64 through float32
    and lose precision around ``2**24``.
    """
    num_rows = sorted_keys.numel()
    device = sorted_keys.device
    if num_rows == 0:
        return torch.empty(0, dtype=torch.int64, device=device), -1
    if num_rows <= _UNIQUE_DIM_GROUP_SCAN_BLOCK_SIZE:
        group_id = torch.empty(num_rows, dtype=torch.int64, device=device)
        last_group_id = torch.empty((), dtype=torch.int64, device=device)
        block_size = triton.next_power_of_2(num_rows)
        with torch_device_fn.device(device.index):
            _unique_dim_group_id_kernel[(1, 1, 1)](
                sorted_keys,
                group_id,
                last_group_id,
                num_rows,
                BLOCK_SIZE=block_size,
                num_warps=_triton_num_warps(block_size),
            )
        return group_id, int(last_group_id.item())

    diff = ((sorted_keys[1:] - sorted_keys[:-1]) != 0).to(torch.int64)
    group_id = torch.cat(
        [
            torch.zeros(1, dtype=torch.int64, device=device),
            torch.cumsum(diff, dim=0),
        ]
    )
    return group_id, int(group_id[-1].item())


def _lex_argsort_rows_composite(flat: torch.Tensor):
    """Lex-sort rows with global first pass plus segmented prefix refinement.

    The original cascade globally sorts all rows for every new column. That is
    wasteful once most prefix groups are singletons. For integer dtypes, refine
    later columns only inside equal-prefix segments and fall back to the global
    composite-key sort when a segment is too large for the local rank kernel.
    """
    key_bits = _monotonic_key_bits(flat.dtype)
    if key_bits is None:
        return None

    num_rows, num_cols = flat.shape
    device = flat.device
    if num_cols == 0:
        indices = torch.arange(num_rows, dtype=torch.int64, device=device)
        return indices, False
    if num_rows <= 1:
        indices = torch.arange(num_rows, dtype=torch.int64, device=device)
        return indices, True

    key_scale = 1 << key_bits
    flat_view, remap_kind, key_offset = _remap_info(flat)
    indices = None
    group_id = None
    all_unique = False
    start_col = 0
    segment_refine = _can_segmented_refine(flat.dtype)

    if segment_refine and num_rows > _UNIQUE_DIM_TILED_RANK_SORT_MAX_KEYS:
        first_keys = _extract_column(flat, 0)
        sorted_keys, indices = torch.sort(first_keys)
        group_id, last_group_id = _group_id_from_sorted(sorted_keys)
        if last_group_id == num_rows - 1:
            return indices, True
        start_col = 1

    for col in range(start_col, num_cols):
        if col > 0 and segment_refine and group_id is not None:
            refined = _segmented_rank_refine(
                flat_view,
                indices,
                group_id,
                num_rows,
                num_cols,
                col,
                key_offset,
                key_scale,
                remap_kind,
            )
            if refined is not None:
                indices, sorted_keys, prefix_unique = refined
                if prefix_unique:
                    all_unique = True
                    break
                group_id, last_group_id = _group_id_from_sorted(sorted_keys)
                if last_group_id == num_rows - 1:
                    all_unique = True
                    break
                continue

        # One fused launch builds ``group_id * key_scale + monotonic(value)``,
        # gathering through the current permutation when ``col > 0``.
        keys = _build_composite_key(
            flat_view,
            col,
            indices,
            group_id,
            num_rows,
            num_cols,
            key_offset,
            key_scale,
            remap_kind,
        )
        perm, sorted_keys = _argsort_keys(keys)
        indices = perm if indices is None else _triton_gather_1d(indices, perm)
        group_id, last_group_id = _group_id_from_sorted(sorted_keys)
        # Early termination: every row already has a unique lex prefix.
        if last_group_id == num_rows - 1:
            all_unique = True
            break
    return indices, all_unique


def _lex_argsort_rows_cascade(flat: torch.Tensor) -> torch.Tensor:
    """Generic-dtype fallback: cascade of stable argsorts, least to most
    significant column. ``O(M)`` argsorts of length ``D`` with ``O(D)`` memory
    traffic per step. Used for dtypes without a monotonic int64 remap."""
    num_rows, num_cols = flat.shape
    indices = torch.arange(num_rows, dtype=torch.int64, device=flat.device)
    if num_rows <= 1 or num_cols == 0:
        return indices
    flat_t = flat.t().contiguous()
    for col in range(num_cols - 1, -1, -1):
        keys = _triton_gather_1d(flat_t[col], indices)
        # LSD cascade requires a stable sort to preserve previous-column order.
        _, perm = torch.sort(keys, stable=True)
        indices = _triton_gather_1d(indices, perm)
    return indices


def _lex_argsort_rows(flat: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Return indices that sort rows of a 2D tensor lexicographically."""
    composite = _lex_argsort_rows_composite(flat)
    if composite is not None:
        return composite
    return _lex_argsort_rows_cascade(flat), False


def _unique_dim_row_hash(flat: torch.Tensor) -> torch.Tensor:
    num_rows, row_len = flat.shape
    block_size = min(_UNIQUE_DIM_COMPARE_BLOCK_SIZE, triton.next_power_of_2(row_len))
    num_chunks = triton.cdiv(row_len, block_size)
    chunk_hash = torch.empty(
        (num_rows, num_chunks), dtype=torch.int64, device=flat.device
    )
    row_hash = torch.empty(num_rows, dtype=torch.int64, device=flat.device)
    with torch_device_fn.device(flat.device.index):
        _unique_dim_row_hash_chunk_kernel[(num_rows, num_chunks, 1)](
            flat,
            chunk_hash,
            num_rows,
            row_len,
            num_chunks,
            BLOCK_SIZE=block_size,
            num_warps=_triton_num_warps(block_size),
        )
        _unique_dim_row_hash_reduce_kernel[(num_rows, 1, 1)](
            chunk_hash,
            row_hash,
            num_chunks,
            BLOCK_CHUNKS=triton.next_power_of_2(num_chunks),
            num_warps=_triton_num_warps(triton.next_power_of_2(num_chunks)),
        )
    return row_hash


def _unique_dim_first_mask(flat: torch.Tensor, sorted_indices: torch.Tensor):
    """Return a bool mask for first rows in sorted lexicographic groups."""
    num_rows, row_len = flat.shape
    if num_rows == 1 or row_len == 0:
        is_first = torch.zeros(num_rows, dtype=torch.bool, device=flat.device)
        is_first[0] = True
        return is_first

    block_size = min(_UNIQUE_DIM_COMPARE_BLOCK_SIZE, triton.next_power_of_2(row_len))
    num_chunks = triton.cdiv(row_len, block_size)
    is_first = torch.empty(num_rows, dtype=torch.bool, device=flat.device)
    if num_chunks == 1:
        with torch_device_fn.device(flat.device.index):
            _unique_dim_row_single_chunk_first_kernel[(num_rows, 1, 1)](
                flat,
                sorted_indices,
                is_first,
                num_rows,
                row_len,
                BLOCK_SIZE=block_size,
                num_warps=_triton_num_warps(block_size),
            )
        return is_first

    row_chunk_diff = torch.empty(
        (num_rows, num_chunks), dtype=torch.int32, device=flat.device
    )
    grid = (num_rows, num_chunks, 1)
    row_hash = (
        _unique_dim_row_hash(flat) if row_len >= _UNIQUE_DIM_HASH_MIN_ROW_LEN else None
    )
    with torch_device_fn.device(flat.device.index):
        if row_hash is None:
            _unique_dim_row_chunk_diff_kernel[grid](
                flat,
                sorted_indices,
                row_chunk_diff,
                num_rows,
                row_len,
                num_chunks,
                BLOCK_SIZE=block_size,
                num_warps=_triton_num_warps(block_size),
            )
        else:
            _unique_dim_row_chunk_diff_hash_kernel[grid](
                flat,
                sorted_indices,
                row_hash,
                row_chunk_diff,
                num_rows,
                row_len,
                num_chunks,
                BLOCK_SIZE=block_size,
                num_warps=_triton_num_warps(block_size),
            )
        _unique_dim_row_diff_reduce_kernel[(num_rows, 1, 1)](
            row_chunk_diff,
            is_first,
            num_chunks,
            BLOCK_CHUNKS=triton.next_power_of_2(num_chunks),
            num_warps=_triton_num_warps(triton.next_power_of_2(num_chunks)),
        )
    return is_first


def _unique_dim_gather_output(
    moved: torch.Tensor,
    unique_indices: torch.Tensor,
    dim: int,
    input_shape: torch.Size,
) -> torch.Tensor:
    num_unique = unique_indices.numel()
    output_shape = (
        tuple(input_shape[:dim]) + (num_unique,) + tuple(input_shape[dim + 1 :])
    )
    if num_unique == 0:
        return torch.empty(output_shape, dtype=moved.dtype, device=moved.device)

    row_len = moved[0].numel()
    flat = moved.reshape(moved.shape[0], row_len)
    moved_output = torch.empty(
        (num_unique,) + tuple(moved.shape[1:]),
        dtype=moved.dtype,
        device=moved.device,
    )
    block_size = min(_UNIQUE_DIM_GATHER_BLOCK_SIZE, triton.next_power_of_2(row_len))
    num_chunks = triton.cdiv(row_len, block_size)
    grid = (num_unique, num_chunks, 1)
    with torch_device_fn.device(moved.device.index):
        _unique_dim_gather_moved_kernel[grid](
            flat,
            unique_indices,
            moved_output,
            num_unique,
            row_len,
            BLOCK_SIZE=block_size,
            num_warps=_triton_num_warps(block_size),
        )
    return moved_output.movedim(0, dim)


def _unique_dim_gather_output_and_inverse(
    moved: torch.Tensor,
    unique_indices: torch.Tensor,
    dim: int,
    input_shape: torch.Size,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_unique = unique_indices.numel()
    output_shape = (
        tuple(input_shape[:dim]) + (num_unique,) + tuple(input_shape[dim + 1 :])
    )
    inverse_indices = torch.empty(
        moved.shape[0], dtype=torch.int64, device=moved.device
    )
    if num_unique == 0:
        output = torch.empty(output_shape, dtype=moved.dtype, device=moved.device)
        return output, inverse_indices

    row_len = moved[0].numel()
    flat = moved.reshape(moved.shape[0], row_len)
    moved_output = torch.empty(
        (num_unique,) + tuple(moved.shape[1:]),
        dtype=moved.dtype,
        device=moved.device,
    )
    block_size = min(_UNIQUE_DIM_GATHER_BLOCK_SIZE, triton.next_power_of_2(row_len))
    num_chunks = triton.cdiv(row_len, block_size)
    grid = (num_unique, num_chunks, 1)
    with torch_device_fn.device(moved.device.index):
        _unique_dim_gather_moved_inverse_kernel[grid](
            flat,
            unique_indices,
            moved_output,
            inverse_indices,
            num_unique,
            row_len,
            BLOCK_SIZE=block_size,
            num_warps=_triton_num_warps(block_size),
        )
    return moved_output.movedim(0, dim), inverse_indices


def _unique_dim_inverse_from_permutation(sorted_indices: torch.Tensor) -> torch.Tensor:
    """Inverse mapping for the all-unique case: ``inverse[sorted_indices[i]] = i``.

    A plain 1D scatter (no per-element column predicate), which is correct on
    every backend; the fused gather+scatter variant miscompiles its masked
    inverse store on some Ascend/NPU backends.
    """
    num_rows = sorted_indices.numel()
    inverse_indices = torch.empty_like(sorted_indices)
    if num_rows == 0:
        return inverse_indices
    block_size = min(_UNIQUE_DIM_GATHER_BLOCK_SIZE, triton.next_power_of_2(num_rows))
    grid = (triton.cdiv(num_rows, block_size), 1, 1)
    with torch_device_fn.device(sorted_indices.device.index):
        _unique_dim_inverse_permutation_kernel[grid](
            sorted_indices,
            inverse_indices,
            num_rows,
            BLOCK_SIZE=block_size,
            num_warps=_triton_num_warps(block_size),
        )
    return inverse_indices


def _unique_dim_inverse(
    sorted_indices: torch.Tensor,
    is_first: torch.Tensor,
) -> torch.Tensor:
    """Inverse mapping: scatter dense group ids back to original positions."""
    num_rows = sorted_indices.numel()
    inverse_indices = torch.empty(
        num_rows, dtype=torch.int64, device=sorted_indices.device
    )
    if num_rows == 0:
        return inverse_indices

    inverse_in_sorted = torch.cumsum(is_first.to(torch.int64), dim=0) - 1
    block_size = min(_UNIQUE_DIM_GATHER_BLOCK_SIZE, triton.next_power_of_2(num_rows))
    grid = (triton.cdiv(num_rows, block_size), 1, 1)
    with torch_device_fn.device(sorted_indices.device.index):
        _unique_dim_inverse_kernel[grid](
            sorted_indices,
            inverse_in_sorted,
            inverse_indices,
            num_rows,
            BLOCK_SIZE=block_size,
            num_warps=_triton_num_warps(block_size),
        )
    return inverse_indices


def _unique_dim_unique_indices(
    sorted_indices: torch.Tensor,
    is_first: torch.Tensor,
) -> torch.Tensor:
    """Original-space indices of the first row in each sorted group."""
    first_positions = torch.nonzero(is_first, as_tuple=False).flatten()
    return _triton_gather_1d(sorted_indices, first_positions)


def _unique_dim_unique_indices_and_inverse(
    sorted_indices: torch.Tensor,
    is_first: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    unique_indices = _unique_dim_unique_indices(sorted_indices, is_first)
    inverse_indices = _unique_dim_inverse(sorted_indices, is_first)
    return unique_indices, inverse_indices


def _unique_dim_counts(
    is_first: torch.Tensor,
    num_rows: int,
) -> torch.Tensor:
    first_positions = torch.nonzero(is_first, as_tuple=False).flatten()
    num_unique = first_positions.numel()
    counts = torch.empty(num_unique, dtype=torch.int64, device=is_first.device)
    if num_unique == 0:
        return counts

    block_size = min(_UNIQUE_DIM_GATHER_BLOCK_SIZE, triton.next_power_of_2(num_unique))
    grid = (triton.cdiv(num_unique, block_size), 1, 1)
    with torch_device_fn.device(is_first.device.index):
        _unique_dim_counts_kernel[grid](
            first_positions,
            counts,
            num_rows,
            num_unique,
            BLOCK_SIZE=block_size,
            num_warps=_triton_num_warps(block_size),
        )
    return counts


def unique_dim(
    input: torch.Tensor,
    dim: int,
    sorted: bool = True,
    return_inverse: bool = False,
    return_counts: bool = False,
):
    """Dimension-aware ``torch.unique`` (a.k.a. ``aten::unique_dim``).

    Treats each slice along ``dim`` as a single element, returning the unique
    slices, an optional inverse mapping of shape ``(input.size(dim),)`` and an
    optional per-unique count tensor of shape ``(output.size(dim),)``.
    """
    logger.debug("GEMS UNIQUE_DIM")

    ndim = input.ndim if input.ndim > 0 else 1
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= max(input.ndim, 1):
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-input.ndim}, {input.ndim - 1}], but got {dim})"
        )

    device = input.device
    size_dim = input.size(dim) if input.ndim > 0 else input.numel()

    if size_dim == 0:
        output = input.clone()
        inverse_indices = torch.empty(0, dtype=torch.int64, device=device)
        counts = torch.empty(0, dtype=torch.int64, device=device)
        return output, inverse_indices, counts

    if dim == 1:
        small_dim1_fast = _unique_dim_small_dim1_2d_fast_path(
            input,
            return_inverse,
            return_counts,
        )
        if small_dim1_fast is not None:
            return small_dim1_fast

    moved = input.movedim(dim, 0).contiguous()
    flat = moved.reshape(size_dim, -1)

    small_fast = _unique_dim_small_lex2_fast_path(
        flat,
        moved,
        dim,
        input.shape,
        return_inverse,
        return_counts,
    )
    if small_fast is not None:
        return small_fast

    sorted_indices, all_unique = _lex_argsort_rows(flat)

    inverse_indices = torch.empty(0, dtype=torch.int64, device=device)
    counts = torch.empty(0, dtype=torch.int64, device=device)

    if all_unique:
        if return_counts:
            counts = torch.ones(size_dim, dtype=torch.int64, device=device)
        if return_inverse:
            output, inverse_indices = _unique_dim_gather_output_and_inverse(
                moved, sorted_indices, dim, input.shape
            )
        else:
            output = _unique_dim_gather_output(moved, sorted_indices, dim, input.shape)
        return output, inverse_indices, counts

    is_first = _unique_dim_first_mask(flat, sorted_indices)
    if return_inverse:
        unique_in_orig, inverse_indices = _unique_dim_unique_indices_and_inverse(
            sorted_indices,
            is_first,
        )
    else:
        unique_in_orig = _unique_dim_unique_indices(sorted_indices, is_first)

    if return_counts:
        counts = _unique_dim_counts(is_first, size_dim)

    output = _unique_dim_gather_output(moved, unique_in_orig, dim, input.shape)

    return output, inverse_indices, counts
