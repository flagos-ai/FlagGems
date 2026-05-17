import logging
import math
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems.ops.topk import _get_iinfo_val
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

MedianResult = namedtuple("median", ["values", "indices"])

_DIRECT_REDUCTION_LIMIT = 256
_DIRECT_FLAT_LIMIT = 256
_BOOL_FLAT_BLOCK = 1024
_BOOL_COUNT_REDUCE_BLOCK = 1024
_DIRECT_REDUCTION_DTYPES = {
    torch.bool,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
}
_FLAT_SORT_LIMIT = 1024
_LASTDIM_SORT_LIMIT = 1024
_BF16_LASTDIM_SORT_LIMIT = 2048
_LASTDIM_SORT_DTYPES = {torch.float16, torch.bfloat16}
_FLAT_SORT_DTYPES = _LASTDIM_SORT_DTYPES | {torch.float32}
_F16_KEY_SELECT_MIN = 512
_F16_KEY_SELECT_LIMIT = 16384
_F16_KEY_SELECT_DTYPES = {torch.float16, torch.bfloat16}
_FP32_KEY_SELECT_MIN = 257
_FP32_KEY_SELECT_LIMIT = 16384
_FP64_KEY_SELECT_MIN = 257
_FP64_KEY_SELECT_LIMIT = 8192
_INT_LASTDIM_SELECT_LIMIT = 16384
_INT_LASTDIM_SELECT_DTYPES = {
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
}
_STRIDED_SELECT_MIN = _DIRECT_REDUCTION_LIMIT + 1
_STRIDED_SELECT_LIMIT = 4096


@libentry()
@triton.jit
def median_small_dim_kernel(
    inp,
    values,
    indices,
    total_outputs,
    reduction_size,
    inner_size,
    BLOCK_N: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
):
    out_offsets = tl.program_id(0) * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    out_mask = out_offsets < total_outputs
    inner_offsets = out_offsets % inner_size
    outer_offsets = out_offsets // inner_size

    reduction_offsets = tl.arange(0, BLOCK_N)
    sample_mask = (reduction_offsets[None, :] < reduction_size) & out_mask[:, None]
    sample_ptrs = (
        inp
        + outer_offsets[:, None] * reduction_size * inner_size
        + reduction_offsets[None, :] * inner_size
        + inner_offsets[:, None]
    )

    if inp.dtype.element_ty.is_floating():
        high = float("inf")
    else:
        high = _get_iinfo_val(inp.dtype.element_ty, return_max=True)

    samples = tl.load(sample_ptrs, mask=sample_mask, other=high)
    sortable = samples

    if inp.dtype.element_ty.is_floating():
        nan_mask = sample_mask & (samples != samples)
        sortable = tl.where(nan_mask, high, samples)

    ordered = tl.sort(sortable, dim=1)
    rank = (reduction_size - 1) // 2
    rank_mask = reduction_offsets[None, :] == rank
    median_values = tl.sum(tl.where(rank_mask, ordered, tl.zeros_like(ordered)), axis=1)

    first_match = tl.argmax(
        (sample_mask & (samples == median_values[:, None])).to(tl.int32), axis=1
    )

    if inp.dtype.element_ty.is_floating():
        nan_i32 = nan_mask.to(tl.int32)
        has_nan = tl.max(nan_i32, axis=1) != 0
        first_nan = tl.argmax(nan_i32, axis=1)
        nan_values = tl.load(
            inp
            + outer_offsets * reduction_size * inner_size
            + first_nan * inner_size
            + inner_offsets,
            mask=out_mask,
            other=0.0,
        )
        median_values = tl.where(has_nan, nan_values, median_values)
        first_match = tl.where(has_nan, first_nan, first_match)

    tl.store(values + out_offsets, median_values, mask=out_mask)
    tl.store(indices + out_offsets, first_match.to(tl.int64), mask=out_mask)


@libentry()
@triton.jit
def median_small_flat_kernel(
    inp,
    value,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    valid = offsets < WIDTH

    if inp.dtype.element_ty.is_floating():
        high = float("inf")
    elif inp.dtype.element_ty is tl.int1:
        high = True
    else:
        high = _get_iinfo_val(inp.dtype.element_ty, return_max=True)

    data = tl.load(inp + offsets, mask=valid, other=high)
    sortable = data

    if inp.dtype.element_ty.is_floating():
        nan_mask = valid & (data != data)
        sortable = tl.where(nan_mask, high, data)

    ordered = tl.sort(sortable)
    rank = (WIDTH - 1) // 2
    median_value = tl.sum(
        tl.where(offsets == rank, ordered, tl.zeros_like(ordered)), axis=0
    )

    if inp.dtype.element_ty.is_floating():
        nan_i32 = nan_mask.to(tl.int32)
        has_nan = tl.max(nan_i32, axis=0) != 0
        first_nan = tl.argmax(nan_i32, axis=0)
        nan_value = tl.load(inp + first_nan, mask=has_nan, other=0.0)
        median_value = tl.where(has_nan, nan_value, median_value)

    tl.store(value, median_value)


@libentry()
@triton.jit
def median_bool_count_kernel(
    inp,
    counts,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    block_id = tl.program_id(0)
    offsets = block_id * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < WIDTH
    data = tl.load(inp + offsets, mask=valid, other=False)
    true_count = tl.sum((valid & data).to(tl.int64), axis=0)
    tl.store(counts + block_id, true_count)


@libentry()
@triton.jit
def median_bool_from_counts_kernel(
    counts,
    value,
    WIDTH: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    valid = offsets < NUM_BLOCKS
    block_counts = tl.load(counts + offsets, mask=valid, other=0)
    true_count = tl.sum(block_counts, axis=0)
    rank = (WIDTH - 1) // 2
    false_count = WIDTH - true_count
    median_value = rank >= false_count
    tl.store(value, median_value)


@libentry()
@triton.jit
def median_bool_reduce_counts_kernel(
    counts_in,
    counts_out,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    block_id = tl.program_id(0)
    offsets = block_id * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < WIDTH
    block_counts = tl.load(counts_in + offsets, mask=valid, other=0)
    count = tl.sum(block_counts, axis=0)
    tl.store(counts_out + block_id, count)


@libentry()
@triton.jit
def median_bool_dim_count_chunks_kernel(
    inp,
    counts,
    first_false,
    first_true,
    total_outputs,
    reduction_size,
    inner_size,
    chunks_per_output: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    out_offset = pid // chunks_per_output
    chunk_id = pid - out_offset * chunks_per_output
    out_mask = out_offset < total_outputs
    inner_offset = out_offset % inner_size
    outer_offset = out_offset // inner_size

    cols = chunk_id * BLOCK + tl.arange(0, BLOCK)
    valid = (cols < reduction_size) & out_mask
    ptrs = (
        inp
        + outer_offset * reduction_size * inner_size
        + cols * inner_size
        + inner_offset
    )
    data = tl.load(ptrs, mask=valid, other=False)

    true_mask = valid & data
    false_mask = valid & ~data
    true_count = tl.sum(true_mask.to(tl.int64), axis=0)
    first_false_idx = tl.min(tl.where(false_mask, cols, reduction_size), axis=0)
    first_true_idx = tl.min(tl.where(true_mask, cols, reduction_size), axis=0)

    tl.store(counts + pid, true_count)
    tl.store(first_false + pid, first_false_idx.to(tl.int64))
    tl.store(first_true + pid, first_true_idx.to(tl.int64))


@libentry()
@triton.jit
def median_bool_dim_reduce_chunks_kernel(
    counts_in,
    first_false_in,
    first_true_in,
    counts_out,
    first_false_out,
    first_true_out,
    input_chunks: tl.constexpr,
    output_chunks: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    out_chunk = tl.program_id(1)
    chunk_offsets = out_chunk * BLOCK + tl.arange(0, BLOCK)
    valid = chunk_offsets < input_chunks
    in_base = row * input_chunks + chunk_offsets

    counts = tl.load(counts_in + in_base, mask=valid, other=0)
    first_false = tl.load(first_false_in + in_base, mask=valid, other=9223372036854775807)
    first_true = tl.load(first_true_in + in_base, mask=valid, other=9223372036854775807)

    true_count = tl.sum(counts, axis=0)
    first_false_idx = tl.min(first_false, axis=0)
    first_true_idx = tl.min(first_true, axis=0)
    out_base = row * output_chunks + out_chunk
    tl.store(counts_out + out_base, true_count)
    tl.store(first_false_out + out_base, first_false_idx)
    tl.store(first_true_out + out_base, first_true_idx)


@libentry()
@triton.jit
def median_bool_dim_finish_kernel(
    counts,
    first_false,
    first_true,
    values,
    indices,
    reduction_size,
    chunks_per_output: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    chunk_offsets = tl.arange(0, BLOCK)
    valid = chunk_offsets < chunks_per_output
    base = row * chunks_per_output + chunk_offsets

    block_counts = tl.load(counts + base, mask=valid, other=0)
    true_count = tl.sum(block_counts, axis=0)
    false_count = reduction_size - true_count
    rank = (reduction_size - 1) // 2
    median_value = rank >= false_count

    false_indices = tl.load(
        first_false + base, mask=valid, other=9223372036854775807
    )
    true_indices = tl.load(first_true + base, mask=valid, other=9223372036854775807)
    first_false_idx = tl.min(false_indices, axis=0)
    first_true_idx = tl.min(true_indices, axis=0)
    first_match = tl.where(median_value, first_true_idx, first_false_idx)

    tl.store(values + row, median_value)
    tl.store(indices + row, first_match)


@libentry()
@triton.jit
def median_lastdim_sort_kernel(
    row_data,
    values,
    indices,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    valid = cols < WIDTH
    base = row_data + row * WIDTH
    data = tl.load(base + cols, mask=valid, other=float("inf"))

    nan_mask = valid & (data != data)
    sortable = tl.where(nan_mask, float("inf"), data)
    ordered = tl.sort(sortable)
    rank = (WIDTH - 1) // 2
    median_value = tl.sum(
        tl.where(cols == rank, ordered, tl.zeros_like(ordered)), axis=0
    )

    first_match = tl.argmax((valid & (data == median_value)).to(tl.int32), axis=0)
    nan_i32 = nan_mask.to(tl.int32)
    has_nan = tl.max(nan_i32, axis=0) != 0
    first_nan = tl.argmax(nan_i32, axis=0)
    nan_value = tl.load(base + first_nan, mask=has_nan, other=0.0)
    median_value = tl.where(has_nan, nan_value, median_value)
    first_match = tl.where(has_nan, first_nan, first_match)

    tl.store(values + row, median_value)
    tl.store(indices + row, first_match.to(tl.int64))


@libentry()
@triton.jit
def median_int_lastdim_select_kernel(
    row_data,
    values,
    indices,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    valid = cols < WIDTH
    base = row_data + row * WIDTH
    data = tl.load(base + cols, mask=valid, other=0)

    dtype = row_data.dtype.element_ty
    high = _get_iinfo_val(dtype, return_max=True)
    low = _get_iinfo_val(dtype, return_max=False)
    row_min = tl.min(tl.where(valid, data, high), axis=0).to(tl.int64)
    row_max = tl.max(tl.where(valid, data, low), axis=0).to(tl.int64)

    lo = row_min
    hi = row_max
    rank = (WIDTH - 1) // 2
    for _ in tl.static_range(0, SEARCH_STEPS):
        mid = lo + ((hi - lo) // 2)
        le_count = tl.sum((valid & (data <= mid.to(dtype))).to(tl.int32), axis=0)
        take_left = le_count > rank
        hi = tl.where(take_left, mid, hi)
        lo = tl.where(take_left, lo, mid + 1)

    median_value = lo.to(dtype)
    first_match = tl.argmax((valid & (data == median_value)).to(tl.int32), axis=0)
    tl.store(values + row, median_value)
    tl.store(indices + row, first_match.to(tl.int64))


@triton.jit
def _fp32_order_key(x):
    bits = x.to(tl.uint32, bitcast=True)
    signed = x.to(tl.int32, bitcast=True)
    sign = signed >> 31
    sign_mask = tl.full((), 0x80000000, dtype=tl.uint32)
    mask = sign_mask | sign.to(tl.uint32, bitcast=True)
    return bits ^ mask


@triton.jit
def _fp64_order_key(x):
    bits = x.to(tl.uint64, bitcast=True)
    signed = x.to(tl.int64, bitcast=True)
    sign = signed >> 63
    sign_mask = tl.full((), 1, dtype=tl.uint64) << 63
    mask = sign_mask | sign.to(tl.uint64, bitcast=True)
    return bits ^ mask


@triton.jit
def _f16_order_key(x):
    bits = x.to(tl.uint16, bitcast=True)
    signed = x.to(tl.int16, bitcast=True)
    sign = signed >> 15
    sign_mask = tl.full((), 0x8000, dtype=tl.uint16)
    mask = sign_mask | sign.to(tl.uint16, bitcast=True)
    return bits ^ mask


@libentry()
@triton.jit
def median_f16_key_select_kernel(
    row_data,
    values,
    indices,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    valid = cols < WIDTH
    base = row_data + row * WIDTH
    data = tl.load(base + cols, mask=valid, other=0.0)

    nan_mask = valid & (data != data)
    nan_i32 = nan_mask.to(tl.int32)
    has_nan = tl.max(nan_i32, axis=0) != 0
    first_nan = tl.argmax(nan_i32, axis=0)
    nan_value = tl.load(base + first_nan, mask=has_nan, other=0.0)

    finite = valid & ~nan_mask
    neg_inf_mask = finite & (data == -float("inf"))
    pos_inf_mask = finite & (data == float("inf"))
    real_finite = finite & ~(neg_inf_mask | pos_inf_mask)
    neg_inf_count = tl.sum(neg_inf_mask.to(tl.int32), axis=0)
    real_finite_count = tl.sum(real_finite.to(tl.int32), axis=0)

    rank = (WIDTH - 1) // 2
    search_rank = rank - neg_inf_count
    take_neg_inf = rank < neg_inf_count
    take_pos_inf = search_rank >= real_finite_count

    keys = _f16_order_key(data).to(tl.uint32)
    key_min_fill = tl.full((), 0xFFFF, dtype=tl.uint32)
    key_max_fill = tl.full((), 0, dtype=tl.uint32)
    row_min = tl.min(tl.where(real_finite, keys, key_min_fill), axis=0)
    row_max = tl.max(tl.where(real_finite, keys, key_max_fill), axis=0)
    has_real_finite = real_finite_count != 0
    row_min = tl.where(has_real_finite, row_min, 0)
    row_max = tl.where(has_real_finite, row_max, 0)

    lo = row_min
    hi = row_max
    for _ in tl.static_range(0, 16):
        mid = lo + ((hi - lo) >> 1)
        le_count = tl.sum((real_finite & (keys <= mid)).to(tl.int32), axis=0)
        take_left = le_count > search_rank
        hi = tl.where(take_left, mid, hi)
        lo = tl.where(take_left, lo, mid + 1)

    selected_key = lo
    key_match = real_finite & (keys == selected_key)
    selected_key_first = tl.argmax(key_match.to(tl.int32), axis=0)
    selected_value = tl.load(base + selected_key_first)

    first_neg_inf = tl.argmax(neg_inf_mask.to(tl.int32), axis=0)
    neg_inf_value = tl.load(base + first_neg_inf, mask=take_neg_inf, other=0.0)
    first_pos_inf = tl.argmax(pos_inf_mask.to(tl.int32), axis=0)
    pos_inf_value = tl.load(base + first_pos_inf, mask=take_pos_inf, other=0.0)
    selected_value = tl.where(take_neg_inf, neg_inf_value, selected_value)
    selected_value = tl.where(take_pos_inf, pos_inf_value, selected_value)
    selected_key_first = tl.where(take_neg_inf, first_neg_inf, selected_key_first)
    selected_key_first = tl.where(take_pos_inf, first_pos_inf, selected_key_first)

    selected_value = tl.where(has_nan, nan_value, selected_value)
    first_match = tl.where(has_nan, first_nan, selected_key_first)
    tl.store(values + row, selected_value)
    tl.store(indices + row, first_match.to(tl.int64))


@libentry()
@triton.jit
def median_fp32_key_select_kernel(
    row_data,
    values,
    indices,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    valid = cols < WIDTH
    base = row_data + row * WIDTH
    data = tl.load(base + cols, mask=valid, other=0.0)

    nan_mask = valid & (data != data)
    nan_i32 = nan_mask.to(tl.int32)
    has_nan = tl.max(nan_i32, axis=0) != 0
    first_nan = tl.argmax(nan_i32, axis=0)
    nan_value = tl.load(base + first_nan, mask=has_nan, other=0.0)

    keys = _fp32_order_key(data)
    finite = valid & ~nan_mask
    key_min_fill = tl.full((), 0xFFFFFFFF, dtype=tl.uint32)
    key_max_fill = tl.full((), 0, dtype=tl.uint32)
    row_min = tl.min(tl.where(finite, keys, key_min_fill), axis=0)
    row_max = tl.max(tl.where(finite, keys, key_max_fill), axis=0)

    lo = row_min
    hi = row_max
    rank = (WIDTH - 1) // 2
    for _ in tl.static_range(0, 32):
        mid = lo + ((hi - lo) >> 1)
        le_count = tl.sum((finite & (keys <= mid)).to(tl.int32), axis=0)
        take_left = le_count > rank
        hi = tl.where(take_left, mid, hi)
        lo = tl.where(take_left, lo, mid + 1)

    selected_key = lo
    key_match = finite & (keys == selected_key)
    selected_key_first = tl.argmax(key_match.to(tl.int32), axis=0)
    selected_value = tl.load(base + selected_key_first)

    selected_value = tl.where(has_nan, nan_value, selected_value)
    first_match = tl.where(has_nan, first_nan, selected_key_first)
    tl.store(values + row, selected_value)
    tl.store(indices + row, first_match.to(tl.int64))


@libentry()
@triton.jit
def median_fp64_key_select_kernel(
    row_data,
    values,
    indices,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    valid = cols < WIDTH
    base = row_data + row * WIDTH
    data = tl.load(base + cols, mask=valid, other=0.0)

    nan_mask = valid & (data != data)
    nan_i64 = nan_mask.to(tl.int64)
    has_nan = tl.max(nan_i64, axis=0) != 0
    first_nan = tl.argmax(nan_i64, axis=0)
    nan_value = tl.load(base + first_nan, mask=has_nan, other=0.0)

    keys = _fp64_order_key(data)
    finite = valid & ~nan_mask
    key_min_fill = tl.full((), 0xFFFFFFFFFFFFFFFF, dtype=tl.uint64)
    key_max_fill = tl.full((), 0, dtype=tl.uint64)
    row_min = tl.min(tl.where(finite, keys, key_min_fill), axis=0)
    row_max = tl.max(tl.where(finite, keys, key_max_fill), axis=0)

    lo = row_min
    hi = row_max
    rank = (WIDTH - 1) // 2
    for _ in tl.static_range(0, 64):
        mid = lo + ((hi - lo) >> 1)
        le_count = tl.sum((finite & (keys <= mid)).to(tl.int32), axis=0)
        take_left = le_count > rank
        hi = tl.where(take_left, mid, hi)
        lo = tl.where(take_left, lo, mid + 1)

    selected_key = lo
    key_match = finite & (keys == selected_key)
    selected_key_first = tl.argmax(key_match.to(tl.int32), axis=0)
    selected_value = tl.load(base + selected_key_first)

    selected_value = tl.where(has_nan, nan_value, selected_value)
    first_match = tl.where(has_nan, first_nan, selected_key_first)
    tl.store(values + row, selected_value)
    tl.store(indices + row, first_match.to(tl.int64))


@libentry()
@triton.jit
def median_f16_strided_key_select_kernel(
    inp,
    values,
    indices,
    total_outputs,
    reduction_size,
    inner_size,
    BLOCK: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
):
    out_offsets = tl.program_id(0) * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    out_mask = out_offsets < total_outputs
    inner_offsets = out_offsets % inner_size
    outer_offsets = out_offsets // inner_size
    cols = tl.arange(0, BLOCK)
    valid = (cols[None, :] < reduction_size) & out_mask[:, None]
    ptrs = (
        inp
        + outer_offsets[:, None] * reduction_size * inner_size
        + cols[None, :] * inner_size
        + inner_offsets[:, None]
    )
    data = tl.load(ptrs, mask=valid, other=0.0)

    nan_mask = valid & (data != data)
    nan_i32 = nan_mask.to(tl.int32)
    has_nan = tl.max(nan_i32, axis=1) != 0
    first_nan = tl.argmax(nan_i32, axis=1)
    nan_value = tl.load(
        inp
        + outer_offsets * reduction_size * inner_size
        + first_nan * inner_size
        + inner_offsets,
        mask=out_mask,
        other=0.0,
    )

    keys = _f16_order_key(data).to(tl.uint32)
    finite = valid & ~nan_mask
    key_min_fill = tl.full((), 0xFFFF, dtype=tl.uint32)
    key_max_fill = tl.full((), 0, dtype=tl.uint32)
    row_min = tl.min(tl.where(finite, keys, key_min_fill), axis=1)
    row_max = tl.max(tl.where(finite, keys, key_max_fill), axis=1)

    lo = row_min
    hi = row_max
    rank = (reduction_size - 1) // 2
    for _ in tl.static_range(0, 16):
        mid = lo + ((hi - lo) >> 1)
        le_count = tl.sum((finite & (keys <= mid[:, None])).to(tl.int32), axis=1)
        take_left = le_count > rank
        hi = tl.where(take_left, mid, hi)
        lo = tl.where(take_left, lo, mid + 1)

    selected_key = lo
    key_match = finite & (keys == selected_key[:, None])
    selected_key_first = tl.argmax(key_match.to(tl.int32), axis=1)
    selected_value = tl.load(
        inp
        + outer_offsets * reduction_size * inner_size
        + selected_key_first * inner_size
        + inner_offsets,
        mask=out_mask,
        other=0.0,
    )

    selected_value = tl.where(has_nan, nan_value, selected_value)
    first_match = tl.where(has_nan, first_nan, selected_key_first)
    tl.store(values + out_offsets, selected_value, mask=out_mask)
    tl.store(indices + out_offsets, first_match.to(tl.int64), mask=out_mask)


@libentry()
@triton.jit
def median_fp32_strided_key_select_kernel(
    inp,
    values,
    indices,
    total_outputs,
    reduction_size,
    inner_size,
    BLOCK: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
):
    out_offsets = tl.program_id(0) * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    out_mask = out_offsets < total_outputs
    inner_offsets = out_offsets % inner_size
    outer_offsets = out_offsets // inner_size
    cols = tl.arange(0, BLOCK)
    valid = (cols[None, :] < reduction_size) & out_mask[:, None]
    ptrs = (
        inp
        + outer_offsets[:, None] * reduction_size * inner_size
        + cols[None, :] * inner_size
        + inner_offsets[:, None]
    )
    data = tl.load(ptrs, mask=valid, other=0.0)

    nan_mask = valid & (data != data)
    nan_i32 = nan_mask.to(tl.int32)
    has_nan = tl.max(nan_i32, axis=1) != 0
    first_nan = tl.argmax(nan_i32, axis=1)
    nan_value = tl.load(
        inp
        + outer_offsets * reduction_size * inner_size
        + first_nan * inner_size
        + inner_offsets,
        mask=out_mask,
        other=0.0,
    )

    keys = _fp32_order_key(data)
    finite = valid & ~nan_mask
    key_min_fill = tl.full((), 0xFFFFFFFF, dtype=tl.uint32)
    key_max_fill = tl.full((), 0, dtype=tl.uint32)
    row_min = tl.min(tl.where(finite, keys, key_min_fill), axis=1)
    row_max = tl.max(tl.where(finite, keys, key_max_fill), axis=1)

    lo = row_min
    hi = row_max
    rank = (reduction_size - 1) // 2
    for _ in tl.static_range(0, 32):
        mid = lo + ((hi - lo) >> 1)
        le_count = tl.sum((finite & (keys <= mid[:, None])).to(tl.int32), axis=1)
        take_left = le_count > rank
        hi = tl.where(take_left, mid, hi)
        lo = tl.where(take_left, lo, mid + 1)

    selected_key = lo
    key_match = finite & (keys == selected_key[:, None])
    selected_key_first = tl.argmax(key_match.to(tl.int32), axis=1)
    selected_value = tl.load(
        inp
        + outer_offsets * reduction_size * inner_size
        + selected_key_first * inner_size
        + inner_offsets,
        mask=out_mask,
        other=0.0,
    )

    selected_value = tl.where(has_nan, nan_value, selected_value)
    first_match = tl.where(has_nan, first_nan, selected_key_first)
    tl.store(values + out_offsets, selected_value, mask=out_mask)
    tl.store(indices + out_offsets, first_match.to(tl.int64), mask=out_mask)


def _has_names(inp):
    return any(name is not None for name in inp.names)


def _anonymous(inp):
    return inp.rename(None) if _has_names(inp) else inp


def _canonical_dim(ndim, dim):
    lower = -1 if ndim == 0 else -ndim
    upper = 0 if ndim == 0 else ndim - 1
    if dim < lower or dim > upper:
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{lower}, {upper}], but got {dim})"
        )
    return 0 if ndim == 0 else dim % ndim


def _name_to_dim(inp, dim):
    if dim not in inp.names:
        raise RuntimeError(f"Name '{dim}' not found in Tensor{inp.names}.")
    return inp.names.index(dim)


def _kept_names(names, dim, keepdim):
    if names is None:
        return None
    if keepdim:
        return names
    return names[:dim] + names[dim + 1 :]


def _empty_result_value(inp):
    if inp.dtype.is_complex:
        out = torch.empty((), dtype=inp.dtype, device=inp.device)
        out.real.fill_(float("nan"))
        out.imag.zero_()
        return out
    if inp.dtype.is_floating_point:
        return torch.full((), float("nan"), dtype=inp.dtype, device=inp.device)
    if inp.dtype == torch.bool:
        return torch.ones((), dtype=inp.dtype, device=inp.device)
    if inp.dtype in (torch.int32, torch.int64):
        return torch.full(
            (), torch.iinfo(inp.dtype).min, dtype=inp.dtype, device=inp.device
        )
    return torch.zeros((), dtype=inp.dtype, device=inp.device)


def _raise_dim_dtype(dtype):
    dtype_names = {
        torch.bool: "Bool",
        torch.complex64: "ComplexFloat",
        torch.complex128: "ComplexDouble",
    }
    dtype_name = dtype_names.get(dtype, str(dtype).removeprefix("torch."))
    raise NotImplementedError(f'"median_out_impl" not implemented for {dtype_name!r}')


def _int_search_steps(dtype):
    if dtype in (torch.int8, torch.uint8):
        return 8
    if dtype == torch.int16:
        return 16
    if dtype == torch.int32:
        return 32
    if dtype == torch.int64:
        return 64
    raise NotImplementedError(f"median integer selection not implemented for {dtype}")


def _unsupported_width(dtype, width):
    raise NotImplementedError(
        f"median Triton selection not implemented for dtype {dtype} "
        f"with reduction width {width}"
    )


def _median_from_rows(row_data, output_shape):
    width = row_data.shape[-1]
    if _use_f16_key_select(row_data.dtype, width):
        return _median_f16_key_select(row_data, output_shape)
    if _use_lastdim_sort(row_data.dtype, width):
        return _median_lastdim_sort(row_data, output_shape)
    if _use_fp32_key_select(row_data.dtype, width):
        return _median_fp32_key_select(row_data, output_shape)
    if _use_fp64_key_select(row_data.dtype, width):
        return _median_fp64_key_select(row_data, output_shape)
    if (
        width <= _INT_LASTDIM_SELECT_LIMIT
        and row_data.dtype in _INT_LASTDIM_SELECT_DTYPES
    ):
        return _median_int_lastdim_select(row_data, output_shape)
    _unsupported_width(row_data.dtype, width)


def _median_small_flat(inp):
    value = torch.empty((), dtype=inp.dtype, device=inp.device)
    block = triton.next_power_of_2(inp.numel())
    with torch_device_fn.device(inp.device):
        median_small_flat_kernel[(1,)](
            inp.reshape(-1),
            value,
            WIDTH=inp.numel(),
            BLOCK=block,
            num_warps=min(8, max(4, block // 32)),
        )
    return value


def _median_bool_flat(inp):
    width = inp.numel()
    block = _BOOL_FLAT_BLOCK
    num_blocks = triton.cdiv(width, block)
    counts = torch.empty((num_blocks,), dtype=torch.int64, device=inp.device)
    value = torch.empty((), dtype=inp.dtype, device=inp.device)
    with torch_device_fn.device(inp.device):
        median_bool_count_kernel[(num_blocks,)](
            inp.reshape(-1),
            counts,
            WIDTH=width,
            BLOCK=block,
            num_warps=4,
        )
        while counts.numel() > _BOOL_COUNT_REDUCE_BLOCK:
            reduced_blocks = triton.cdiv(counts.numel(), _BOOL_COUNT_REDUCE_BLOCK)
            reduced = torch.empty(
                (reduced_blocks,), dtype=torch.int64, device=inp.device
            )
            median_bool_reduce_counts_kernel[(reduced_blocks,)](
                counts,
                reduced,
                WIDTH=counts.numel(),
                BLOCK=_BOOL_COUNT_REDUCE_BLOCK,
                num_warps=4,
            )
            counts = reduced
        count_block = triton.next_power_of_2(counts.numel())
        median_bool_from_counts_kernel[(1,)](
            counts,
            value,
            WIDTH=width,
            NUM_BLOCKS=counts.numel(),
            BLOCK=count_block,
            num_warps=min(8, max(1, count_block // 32)),
        )
    return value


def _median_bool_dim(inp, dim, output_shape):
    reduction_size = inp.shape[dim]
    inner_size = math.prod(inp.shape[dim + 1 :])
    total_outputs = math.prod(output_shape)
    values = torch.empty(output_shape, dtype=inp.dtype, device=inp.device)
    indices = torch.empty(output_shape, dtype=torch.int64, device=inp.device)
    block = _BOOL_FLAT_BLOCK
    chunks = triton.cdiv(reduction_size, block)
    chunk_shape = (total_outputs, chunks)
    counts = torch.empty(chunk_shape, dtype=torch.int64, device=inp.device)
    first_false = torch.empty(chunk_shape, dtype=torch.int64, device=inp.device)
    first_true = torch.empty(chunk_shape, dtype=torch.int64, device=inp.device)

    with torch_device_fn.device(inp.device):
        median_bool_dim_count_chunks_kernel[(total_outputs * chunks,)](
            inp,
            counts.reshape(-1),
            first_false.reshape(-1),
            first_true.reshape(-1),
            total_outputs,
            reduction_size,
            inner_size,
            chunks_per_output=chunks,
            BLOCK=block,
            num_warps=4,
        )
        while chunks > _BOOL_COUNT_REDUCE_BLOCK:
            reduced_chunks = triton.cdiv(chunks, _BOOL_COUNT_REDUCE_BLOCK)
            reduced_shape = (total_outputs, reduced_chunks)
            reduced_counts = torch.empty(
                reduced_shape, dtype=torch.int64, device=inp.device
            )
            reduced_first_false = torch.empty(
                reduced_shape, dtype=torch.int64, device=inp.device
            )
            reduced_first_true = torch.empty(
                reduced_shape, dtype=torch.int64, device=inp.device
            )
            median_bool_dim_reduce_chunks_kernel[(total_outputs, reduced_chunks)](
                counts.reshape(-1),
                first_false.reshape(-1),
                first_true.reshape(-1),
                reduced_counts.reshape(-1),
                reduced_first_false.reshape(-1),
                reduced_first_true.reshape(-1),
                input_chunks=chunks,
                output_chunks=reduced_chunks,
                BLOCK=_BOOL_COUNT_REDUCE_BLOCK,
                num_warps=4,
            )
            counts = reduced_counts
            first_false = reduced_first_false
            first_true = reduced_first_true
            chunks = reduced_chunks

        finish_block = triton.next_power_of_2(chunks)
        median_bool_dim_finish_kernel[(total_outputs,)](
            counts.reshape(-1),
            first_false.reshape(-1),
            first_true.reshape(-1),
            values.reshape(-1),
            indices.reshape(-1),
            reduction_size,
            chunks_per_output=chunks,
            BLOCK=finish_block,
            num_warps=min(8, max(1, finish_block // 32)),
        )
    return values, indices


def _median_lastdim_sort(row_data, output_shape):
    width = row_data.shape[-1]
    rows = row_data.numel() // width
    values = torch.empty(output_shape, dtype=row_data.dtype, device=row_data.device)
    indices = torch.empty(output_shape, dtype=torch.int64, device=row_data.device)
    block = triton.next_power_of_2(width)
    num_warps = 8 if rows == 1 and block >= 1024 else min(8, max(4, block // 512))
    with torch_device_fn.device(row_data.device):
        median_lastdim_sort_kernel[(rows,)](
            row_data.reshape(rows, width),
            values.reshape(rows),
            indices.reshape(rows),
            WIDTH=width,
            BLOCK=block,
            num_warps=num_warps,
        )
    return values, indices


def _use_lastdim_sort(dtype, width):
    if dtype == torch.bfloat16:
        return width <= _BF16_LASTDIM_SORT_LIMIT
    if dtype == torch.float16:
        return width <= _LASTDIM_SORT_LIMIT
    return False


def _use_f16_key_select(dtype, width):
    return (
        dtype in _F16_KEY_SELECT_DTYPES
        and _F16_KEY_SELECT_MIN <= width <= _F16_KEY_SELECT_LIMIT
    )


def _use_fp32_key_select(dtype, width):
    return (
        dtype == torch.float32
        and _FP32_KEY_SELECT_MIN <= width <= _FP32_KEY_SELECT_LIMIT
    )


def _use_fp64_key_select(dtype, width):
    return (
        dtype == torch.float64
        and _FP64_KEY_SELECT_MIN <= width <= _FP64_KEY_SELECT_LIMIT
    )


def _use_strided_select(dtype, width):
    return (
        _STRIDED_SELECT_MIN <= width <= _STRIDED_SELECT_LIMIT
        and dtype in (_F16_KEY_SELECT_DTYPES | {torch.float32})
    )


def _median_int_lastdim_select(row_data, output_shape):
    width = row_data.shape[-1]
    rows = row_data.numel() // width
    values = torch.empty(output_shape, dtype=row_data.dtype, device=row_data.device)
    indices = torch.empty(output_shape, dtype=torch.int64, device=row_data.device)
    block = triton.next_power_of_2(width)
    search_steps = _int_search_steps(row_data.dtype)
    with torch_device_fn.device(row_data.device):
        median_int_lastdim_select_kernel[(rows,)](
            row_data.reshape(rows, width),
            values.reshape(rows),
            indices.reshape(rows),
            WIDTH=width,
            BLOCK=block,
            SEARCH_STEPS=search_steps,
            num_warps=min(8, max(4, block // 512)),
        )
    return values, indices


def _median_f16_key_select(row_data, output_shape):
    width = row_data.shape[-1]
    rows = row_data.numel() // width
    values = torch.empty(output_shape, dtype=row_data.dtype, device=row_data.device)
    indices = torch.empty(output_shape, dtype=torch.int64, device=row_data.device)
    block = triton.next_power_of_2(width)
    num_warps = 1 if block <= 1024 else 2 if block <= 2048 else 4
    with torch_device_fn.device(row_data.device):
        median_f16_key_select_kernel[(rows,)](
            row_data.reshape(rows, width),
            values.reshape(rows),
            indices.reshape(rows),
            WIDTH=width,
            BLOCK=block,
            num_warps=num_warps,
        )
    return values, indices


def _median_f16_strided_key_select(inp, dim, output_shape):
    reduction_size = inp.shape[dim]
    inner_size = math.prod(inp.shape[dim + 1 :])
    total_outputs = math.prod(output_shape)
    values = torch.empty(output_shape, dtype=inp.dtype, device=inp.device)
    indices = torch.empty(output_shape, dtype=torch.int64, device=inp.device)
    block = triton.next_power_of_2(reduction_size)
    block_out = 2
    num_warps = 1 if block <= 1024 else 2 if block <= 2048 else 4
    with torch_device_fn.device(inp.device):
        median_f16_strided_key_select_kernel[(triton.cdiv(total_outputs, block_out),)](
            inp,
            values.reshape(-1),
            indices.reshape(-1),
            total_outputs,
            reduction_size,
            inner_size,
            BLOCK=block,
            BLOCK_OUT=block_out,
            num_warps=num_warps,
        )
    return values, indices


def _median_fp32_key_select(row_data, output_shape):
    width = row_data.shape[-1]
    rows = row_data.numel() // width
    values = torch.empty(output_shape, dtype=row_data.dtype, device=row_data.device)
    indices = torch.empty(output_shape, dtype=torch.int64, device=row_data.device)
    block = triton.next_power_of_2(width)
    num_warps = 2 if block <= 1024 else 8
    with torch_device_fn.device(row_data.device):
        median_fp32_key_select_kernel[(rows,)](
            row_data.reshape(rows, width),
            values.reshape(rows),
            indices.reshape(rows),
            WIDTH=width,
            BLOCK=block,
            num_warps=num_warps,
        )
    return values, indices


def _median_fp64_key_select(row_data, output_shape):
    width = row_data.shape[-1]
    rows = row_data.numel() // width
    values = torch.empty(output_shape, dtype=row_data.dtype, device=row_data.device)
    indices = torch.empty(output_shape, dtype=torch.int64, device=row_data.device)
    block = triton.next_power_of_2(width)
    num_warps = 2 if block <= 1024 else 8
    with torch_device_fn.device(row_data.device):
        median_fp64_key_select_kernel[(rows,)](
            row_data.reshape(rows, width),
            values.reshape(rows),
            indices.reshape(rows),
            WIDTH=width,
            BLOCK=block,
            num_warps=num_warps,
        )
    return values, indices


def _median_fp32_strided_key_select(inp, dim, output_shape):
    reduction_size = inp.shape[dim]
    inner_size = math.prod(inp.shape[dim + 1 :])
    total_outputs = math.prod(output_shape)
    values = torch.empty(output_shape, dtype=inp.dtype, device=inp.device)
    indices = torch.empty(output_shape, dtype=torch.int64, device=inp.device)
    block = triton.next_power_of_2(reduction_size)
    block_out = 2
    num_warps = 2 if block <= 1024 else 8
    with torch_device_fn.device(inp.device):
        median_fp32_strided_key_select_kernel[(triton.cdiv(total_outputs, block_out),)](
            inp,
            values.reshape(-1),
            indices.reshape(-1),
            total_outputs,
            reduction_size,
            inner_size,
            BLOCK=block,
            BLOCK_OUT=block_out,
            num_warps=num_warps,
        )
    return values, indices


def _median_direct_dim(inp, dim, output_shape):
    reduction_size = inp.shape[dim]
    inner_size = math.prod(inp.shape[dim + 1 :])
    total_outputs = math.prod(output_shape)
    values = torch.empty(output_shape, dtype=inp.dtype, device=inp.device)
    indices = torch.empty(output_shape, dtype=torch.int64, device=inp.device)
    block_n = triton.next_power_of_2(reduction_size)
    block_out = 2 if block_n >= 128 else 16
    if block_n >= 128:
        num_warps = 8 if inp.dtype in (torch.int32, torch.int64) else 4
    else:
        num_warps = 1
    with torch_device_fn.device(inp.device):
        median_small_dim_kernel[(triton.cdiv(total_outputs, block_out),)](
            inp,
            values.reshape(-1),
            indices.reshape(-1),
            total_outputs,
            reduction_size,
            inner_size,
            BLOCK_N=block_n,
            BLOCK_OUT=block_out,
            num_warps=num_warps,
        )
    return values, indices


def _copy_out(src, out, name):
    if out.device != src.device:
        raise RuntimeError(
            f"Expected {name} tensor to have device {src.device}, "
            f"but got {out.device} instead"
        )
    if out.dtype != src.dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {src.dtype}, but got {out.dtype}"
        )
    out.resize_as_(src)
    out.copy_(src)
    return out


def median(inp):
    logger.debug("GEMS MEDIAN")

    inp = _anonymous(inp)
    if inp.numel() == 0:
        return _empty_result_value(inp)
    if inp.dtype.is_complex:
        raise RuntimeError("Sort does not support complex dtypes on CPU")
    if inp.numel() == 1:
        return inp.reshape(()).clone()

    flat = inp.contiguous().reshape(-1)
    if inp.dtype in _DIRECT_REDUCTION_DTYPES and inp.numel() <= _DIRECT_FLAT_LIMIT:
        return _median_small_flat(flat)
    if inp.dtype == torch.bool:
        return _median_bool_flat(flat)

    row_data = flat.reshape(1, inp.numel())
    if _use_f16_key_select(inp.dtype, inp.numel()):
        values, _ = _median_f16_key_select(row_data, ())
    elif inp.dtype in _FLAT_SORT_DTYPES and inp.numel() <= _FLAT_SORT_LIMIT:
        values, _ = _median_lastdim_sort(row_data, ())
    elif _use_fp32_key_select(inp.dtype, inp.numel()):
        values, _ = _median_fp32_key_select(row_data, ())
    elif _use_fp64_key_select(inp.dtype, inp.numel()):
        values, _ = _median_fp64_key_select(row_data, ())
    elif (
        inp.numel() <= _INT_LASTDIM_SELECT_LIMIT
        and inp.dtype in _INT_LASTDIM_SELECT_DTYPES
    ):
        values, _ = _median_int_lastdim_select(row_data, ())
    else:
        values, _ = _median_from_rows(row_data, ())
    return values.reshape(())


def median_out(inp, *, out):
    logger.debug("GEMS MEDIAN.OUT")
    return _copy_out(median(inp), out, "out")


def median_dim(inp, dim=0, keepdim=False):
    logger.debug("GEMS MEDIAN.DIM")

    if isinstance(dim, str):
        dim = _name_to_dim(inp, dim)
    dim = _canonical_dim(inp.ndim, dim)
    names = inp.names if _has_names(inp) else None
    work = _anonymous(inp)

    if work.ndim == 0:
        if work.dtype.is_complex:
            _raise_dim_dtype(work.dtype)
        return MedianResult(
            values=work.clone(),
            indices=torch.zeros((), dtype=torch.int64, device=work.device),
        )

    if work.shape[dim] == 0:
        raise IndexError(
            f"median(): Expected reduction dim {dim} to have non-zero size."
        )

    output_shape = list(work.shape)
    if keepdim:
        output_shape[dim] = 1
    else:
        del output_shape[dim]
    output_names = _kept_names(names, dim, keepdim)

    if work.numel() == 0:
        values = torch.empty(output_shape, dtype=work.dtype, device=work.device)
        indices = torch.empty(output_shape, dtype=torch.int64, device=work.device)
    else:
        if work.dtype.is_complex:
            _raise_dim_dtype(work.dtype)
        if work.dtype == torch.bool:
            values, indices = _median_bool_dim(work.contiguous(), dim, output_shape)
        elif (
            work.shape[dim] <= _DIRECT_REDUCTION_LIMIT
            and work.dtype in _DIRECT_REDUCTION_DTYPES
        ):
            values, indices = _median_direct_dim(work.contiguous(), dim, output_shape)
        elif (
            dim != work.ndim - 1
            and work.is_contiguous()
            and _use_strided_select(work.dtype, work.shape[dim])
        ):
            if work.dtype in _F16_KEY_SELECT_DTYPES:
                values, indices = _median_f16_strided_key_select(
                    work, dim, output_shape
                )
            elif work.dtype == torch.float32:
                values, indices = _median_fp32_strided_key_select(
                    work, dim, output_shape
                )
        elif dim == work.ndim - 1 and _use_f16_key_select(work.dtype, work.shape[dim]):
            values, indices = _median_f16_key_select(work.contiguous(), output_shape)
        elif dim == work.ndim - 1 and _use_lastdim_sort(work.dtype, work.shape[dim]):
            values, indices = _median_lastdim_sort(work.contiguous(), output_shape)
        elif dim == work.ndim - 1 and _use_fp32_key_select(work.dtype, work.shape[dim]):
            values, indices = _median_fp32_key_select(work.contiguous(), output_shape)
        elif dim == work.ndim - 1 and _use_fp64_key_select(work.dtype, work.shape[dim]):
            values, indices = _median_fp64_key_select(work.contiguous(), output_shape)
        elif (
            dim == work.ndim - 1
            and work.shape[dim] <= _INT_LASTDIM_SELECT_LIMIT
            and work.dtype in _INT_LASTDIM_SELECT_DTYPES
        ):
            values, indices = _median_int_lastdim_select(
                work.contiguous(), output_shape
            )
        else:
            rows = torch.movedim(work, dim, -1).contiguous()
            row_output_shape = rows.shape[:-1]
            row_width = rows.shape[-1]
            if _use_f16_key_select(rows.dtype, row_width):
                values, indices = _median_f16_key_select(rows, row_output_shape)
            elif _use_lastdim_sort(rows.dtype, row_width):
                values, indices = _median_lastdim_sort(rows, row_output_shape)
            elif _use_fp32_key_select(rows.dtype, row_width):
                values, indices = _median_fp32_key_select(rows, row_output_shape)
            elif _use_fp64_key_select(rows.dtype, row_width):
                values, indices = _median_fp64_key_select(rows, row_output_shape)
            elif (
                row_width <= _INT_LASTDIM_SELECT_LIMIT
                and rows.dtype in _INT_LASTDIM_SELECT_DTYPES
            ):
                values, indices = _median_int_lastdim_select(rows, row_output_shape)
            else:
                values, indices = _median_from_rows(rows, row_output_shape)
            if keepdim:
                values = torch.movedim(values.unsqueeze(-1), -1, dim)
                indices = torch.movedim(indices.unsqueeze(-1), -1, dim)

    if output_names is not None:
        values = values.refine_names(*output_names)
        indices = indices.refine_names(*output_names)

    return MedianResult(values=values, indices=indices)


def median_dim_values(inp, dim=0, keepdim=False, *, values, indices):
    logger.debug("GEMS MEDIAN.DIM_VALUES")
    result = median_dim(inp, dim=dim, keepdim=keepdim)
    _copy_out(result.values, values, "values")
    _copy_out(result.indices, indices, "indices")
    return MedianResult(values=values, indices=indices)
