import logging
import math
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, tl_extra_shim
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.limits import get_dtype_max, get_dtype_min

from .topk import _get_finfo_val

logger = logging.getLogger(__name__)

NanMedian = namedtuple("nanmedian", ["values", "indices"])
MAX_BLOCK_N = 128
RADIX_BLOCK_N = 1024
RADIX_BITS = 2
MEDIUM_REDUCTION_N = 1024
LARGE_FLOAT_REDUCTION_N = 4096
LONG_RADIX_REDUCTION_N = 131072
ASCEND_FLAT_SORT_MIN_N = 1 << 20
FLAT_RADIX_BLOCK_N = 4096
FLAT_RADIX_BITS = 8
RADIX_SELECT_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
)
ASCEND_HISTOGRAM_SELECT_DTYPES = (
    torch.int8,
    torch.uint8,
)
ASCEND_BYTE_HISTOGRAM_SELECT_DTYPES = (
    torch.int16,
    torch.int32,
)
ASCEND_FLOAT_SELECT_DTYPES = (
    torch.float16,
    torch.float32,
)
ASCEND_HISTOGRAM_BINS = 256
ASCEND_MULTI_HISTOGRAM_MIN_N = 8192


@triton.jit
def _is_not_nan(vals, USE_ISNAN: tl.constexpr):
    vals_fp32 = vals.to(tl.float32)
    if USE_ISNAN:
        return ~tl_extra_shim.isnan(vals_fp32)
    return vals_fp32 == vals_fp32


@triton.jit
def _to_order_key(vals, valid):
    dtype = vals.dtype
    nbits: tl.constexpr = dtype.primitive_bitwidth
    utype = tl.dtype(f"uint{nbits}")
    top_mask: tl.constexpr = 1 << (nbits - 1)
    full_mask: tl.constexpr = (1 << nbits) - 1
    full = tl.full(vals.shape, full_mask, dtype=utype)

    if dtype.is_floating():
        bits = vals.to(utype, bitcast=True)
        sign_mask = tl.where((bits & top_mask) != 0, full_mask, top_mask)
        key = bits ^ sign_mask
    elif dtype.is_int_signed():
        bits = vals.to(utype, bitcast=True)
        key = bits ^ top_mask
    else:
        key = vals.to(utype)
    return tl.where(valid, key, full)


@libentry()
@triton.jit
def count_valid_kernel(
    inp,
    valid_counts,
    M,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_ISNAN: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    count = tl.full((), 0, dtype=tl.int32)
    for start in tl.range(0, N, BLOCK_N):
        cols = start + offsets
        mask = cols < N
        vals = tl.load(inp + pid * N + cols, mask=mask, other=float("nan"))
        valid = mask & _is_not_nan(vals, USE_ISNAN)
        count += tl.sum(valid.to(tl.int32), axis=0)
    tl.store(valid_counts + pid, count)


@libentry()
@triton.jit
def nanmedian_select_kernel(
    inp,
    out_values,
    out_indices,
    M,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_ISNAN: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < N
    dtype = inp.dtype.element_ty
    if dtype.is_floating():
        max_value = _get_finfo_val(dtype, return_max=True)
        fallback_value = _get_finfo_val(dtype, return_max=False)
    else:
        max_value = get_dtype_max(dtype)
        fallback_value = get_dtype_min(dtype)
    vals = tl.load(inp + pid * N + offsets, mask=mask, other=max_value)

    if dtype.is_floating():
        valid = mask & _is_not_nan(vals, USE_ISNAN)
    else:
        valid = mask
    valid_count = tl.sum(valid.to(tl.int32), axis=0)
    median_rank = (valid_count - 1) // 2

    active = valid
    median_val = tl.full((), fallback_value, dtype=vals.dtype)
    median_idx = tl.full((), 0, dtype=tl.int32)
    for select_iter in tl.static_range(0, BLOCK_N):
        select_vals = tl.where(active, vals, max_value)
        cur_val = tl.min(select_vals, axis=0)
        cur_idx = tl.min(tl.where(active & (vals == cur_val), offsets, BLOCK_N), axis=0)
        take = select_iter == median_rank
        median_val = tl.where(take, cur_val, median_val)
        median_idx = tl.where(take, cur_idx, median_idx)
        active = active & (offsets != cur_idx)

    if dtype.is_floating():
        all_nan = valid_count == 0
        median_val = tl.where(all_nan, float("nan"), median_val)
        median_idx = tl.where(all_nan, 0, median_idx)

    tl.store(out_values + pid, median_val)
    tl.store(out_indices + pid, median_idx)


@libentry()
@triton.jit
def nanmedian_float_clean_count_kernel(
    inp,
    cleaned,
    valid_counts,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    dtype = inp.dtype.element_ty
    max_value = _get_finfo_val(dtype, return_max=True)
    count = tl.full((), 0, dtype=tl.int32)

    for start in tl.range(0, N, BLOCK_N):
        cols = start + offsets
        mask = cols < N
        vals = tl.load(inp + pid * N + cols, mask=mask, other=max_value)
        valid = mask & _is_not_nan(vals, False)
        cleaned_vals = tl.where(valid, vals, max_value)
        tl.store(cleaned + pid * N + cols, cleaned_vals, mask=mask)
        count += tl.sum(valid.to(tl.int32), axis=0)

    tl.store(valid_counts + pid, count)


@libentry()
@triton.jit
def nanmedian_float_sorted_gather_kernel(
    sorted_values,
    sorted_indices,
    valid_counts,
    out_values,
    out_indices,
    N: tl.constexpr,
):
    pid = tle.program_id(0)
    count = tl.load(valid_counts + pid)
    rank = tl.where(count > 0, (count - 1) // 2, 0)
    result_val = tl.load(
        sorted_values + pid * N + rank, mask=count > 0, other=float("nan")
    )
    result_idx = tl.load(sorted_indices + pid * N + rank, mask=count > 0, other=0)
    result_val = tl.where(count > 0, result_val, float("nan"))
    result_idx = tl.where(count > 0, result_idx, 0)

    tl.store(out_values + pid, result_val)
    tl.store(out_indices + pid, result_idx)


@libentry()
@triton.jit
def nanmedian_ascend_histogram_select_kernel(
    inp,
    out_values,
    out_indices,
    M,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HISTOGRAM_BINS: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    bins = tl.arange(0, HISTOGRAM_BINS)
    counts = tl.zeros((HISTOGRAM_BINS,), dtype=tl.int32)

    for start in tl.range(0, N, BLOCK_N):
        cols = start + offsets
        mask = cols < N
        vals = tl.load(inp + pid * N + cols, mask=mask, other=0)
        keys = _to_order_key(vals, mask).to(tl.int32)
        keys = tl.where(mask, keys, 0)
        chunk_counts = tl.histogram(keys, HISTOGRAM_BINS).to(tl.int32)
        invalid_count = tl.sum((~mask).to(tl.int32), axis=0)
        counts += chunk_counts - tl.where(bins == 0, invalid_count, 0)

    k_to_find: tl.constexpr = (N + 1) // 2
    cumsum = tl.cumsum(counts, axis=0)
    prev = cumsum - counts
    take = (k_to_find <= cumsum) & (k_to_find > prev)
    selected_key = tl.min(tl.where(take, bins, HISTOGRAM_BINS - 1), axis=0)

    result_idx = tl.full((), N, dtype=tl.int32)
    for start in tl.range(0, N, BLOCK_N):
        cols = start + offsets
        mask = cols < N
        vals = tl.load(inp + pid * N + cols, mask=mask, other=0)
        keys = _to_order_key(vals, mask).to(tl.int32)
        local_idx = tl.min(tl.where(mask & (keys == selected_key), cols, N), axis=0)
        result_idx = tl.where(local_idx < result_idx, local_idx, result_idx)

    result_val = tl.load(inp + pid * N + result_idx)
    tl.store(out_values + pid, result_val)
    tl.store(out_indices + pid, result_idx)


@libentry()
@triton.jit
def nanmedian_ascend_histogram_count_kernel(
    inp,
    partial_counts,
    M,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    HISTOGRAM_BINS: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_chunk = tle.program_id(1)
    offsets = pid_chunk * BLOCK_N + tl.arange(0, BLOCK_N)
    bins = tl.arange(0, HISTOGRAM_BINS)
    mask = offsets < N
    vals = tl.load(inp + pid_m * N + offsets, mask=mask, other=0)
    keys = _to_order_key(vals, mask).to(tl.int32)
    keys = tl.where(mask, keys, 0)
    counts = tl.histogram(keys, HISTOGRAM_BINS).to(tl.int32)
    invalid_count = tl.sum((~mask).to(tl.int32), axis=0)
    counts = counts - tl.where(bins == 0, invalid_count, 0)
    count_offsets = (pid_m * NUM_CHUNKS + pid_chunk) * HISTOGRAM_BINS + bins
    tl.store(partial_counts + count_offsets, counts)


@libentry()
@triton.jit
def nanmedian_ascend_histogram_reduce_kernel(
    inp,
    partial_counts,
    out_values,
    out_indices,
    M,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    HISTOGRAM_BINS: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    bins = tl.arange(0, HISTOGRAM_BINS)
    counts = tl.zeros((HISTOGRAM_BINS,), dtype=tl.int32)

    for chunk in tl.range(0, NUM_CHUNKS):
        count_offsets = (pid * NUM_CHUNKS + chunk) * HISTOGRAM_BINS + bins
        counts += tl.load(partial_counts + count_offsets)

    k_to_find: tl.constexpr = (N + 1) // 2
    cumsum = tl.cumsum(counts, axis=0)
    prev = cumsum - counts
    take = (k_to_find <= cumsum) & (k_to_find > prev)
    selected_key = tl.min(tl.where(take, bins, HISTOGRAM_BINS - 1), axis=0)

    result_idx = tl.full((), N, dtype=tl.int32)
    for start in tl.range(0, N, BLOCK_N):
        cols = start + offsets
        mask = cols < N
        vals = tl.load(inp + pid * N + cols, mask=mask, other=0)
        keys = _to_order_key(vals, mask).to(tl.int32)
        local_idx = tl.min(tl.where(mask & (keys == selected_key), cols, N), axis=0)
        result_idx = tl.where(local_idx < result_idx, local_idx, result_idx)

    result_val = tl.load(inp + pid * N + result_idx)
    tl.store(out_values + pid, result_val)
    tl.store(out_indices + pid, result_idx)


@libentry()
@triton.jit
def nanmedian_ascend_byte_histogram_select_kernel(
    inp,
    out_values,
    out_indices,
    M,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HISTOGRAM_BINS: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    bins = tl.arange(0, HISTOGRAM_BINS)
    dtype = inp.dtype.element_ty
    nbits: tl.constexpr = dtype.primitive_bitwidth
    utype = tl.dtype(f"uint{nbits}")
    byte_mask_val = tl.full((), HISTOGRAM_BINS - 1, dtype=utype)

    k_to_find = tl.full((), (N + 1) // 2, dtype=tl.int32)
    desired = tl.full((), 0, dtype=utype)
    desired_mask = tl.full((), 0, dtype=utype)

    for digit_pos in tl.static_range(nbits - 8, -1, -8):
        counts = tl.zeros((HISTOGRAM_BINS,), dtype=tl.int32)

        for start in tl.range(0, N, BLOCK_N):
            cols = start + offsets
            mask = cols < N
            vals = tl.load(inp + pid * N + cols, mask=mask, other=0)
            keys = _to_order_key(vals, mask)
            active = mask & ((keys & desired_mask) == desired)
            digit = ((keys >> digit_pos) & byte_mask_val).to(tl.int32)
            digit = tl.where(active, digit, 0)
            chunk_counts = tl.histogram(digit, HISTOGRAM_BINS).to(tl.int32)
            inactive_count = tl.sum((~active).to(tl.int32), axis=0)
            counts += chunk_counts - tl.where(bins == 0, inactive_count, 0)

        cumsum = tl.cumsum(counts, axis=0)
        prev = cumsum - counts
        take = (k_to_find <= cumsum) & (k_to_find > prev)
        selected_bin = tl.min(tl.where(take, bins, HISTOGRAM_BINS - 1), axis=0)
        counts_before = tl.max(tl.where(take, prev, 0), axis=0)

        selected_bin = selected_bin.to(utype)
        desired = desired | (selected_bin << digit_pos)
        desired_mask = desired_mask | (byte_mask_val << digit_pos)
        k_to_find = k_to_find - counts_before

    result_idx = tl.full((), N, dtype=tl.int32)
    for start in tl.range(0, N, BLOCK_N):
        cols = start + offsets
        mask = cols < N
        vals = tl.load(inp + pid * N + cols, mask=mask, other=0)
        keys = _to_order_key(vals, mask)
        local_idx = tl.min(tl.where(mask & (keys == desired), cols, N), axis=0)
        result_idx = tl.where(local_idx < result_idx, local_idx, result_idx)

    result_val = tl.load(inp + pid * N + result_idx)
    tl.store(out_values + pid, result_val)
    tl.store(out_indices + pid, result_idx)


@libentry()
@triton.jit
def nanmedian_ascend_byte_histogram_init_kernel(
    state,
    M,
    N: tl.constexpr,
):
    pid = tle.program_id(0)
    base = pid * 3
    tl.store(state + base + 0, 0)
    tl.store(state + base + 1, 0)
    tl.store(state + base + 2, (N + 1) // 2)


@libentry()
@triton.jit
def nanmedian_ascend_byte_histogram_count_kernel(
    inp,
    state,
    partial_counts,
    M,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    HISTOGRAM_BINS: tl.constexpr,
    DIGIT_POS: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_chunk = tle.program_id(1)
    offsets = pid_chunk * BLOCK_N + tl.arange(0, BLOCK_N)
    bins = tl.arange(0, HISTOGRAM_BINS)
    mask = offsets < N

    dtype = inp.dtype.element_ty
    nbits: tl.constexpr = dtype.primitive_bitwidth
    utype = tl.dtype(f"uint{nbits}")
    byte_mask_val = tl.full((), HISTOGRAM_BINS - 1, dtype=utype)
    state_base = pid_m * 3
    desired = tl.load(state + state_base + 0).to(utype)
    desired_mask = tl.load(state + state_base + 1).to(utype)

    vals = tl.load(inp + pid_m * N + offsets, mask=mask, other=0)
    keys = _to_order_key(vals, mask)
    active = mask & ((keys & desired_mask) == desired)
    digit = ((keys >> DIGIT_POS) & byte_mask_val).to(tl.int32)
    digit = tl.where(active, digit, 0)
    counts = tl.histogram(digit, HISTOGRAM_BINS).to(tl.int32)
    inactive_count = tl.sum((~active).to(tl.int32), axis=0)
    counts = counts - tl.where(bins == 0, inactive_count, 0)

    count_offsets = (pid_m * NUM_CHUNKS + pid_chunk) * HISTOGRAM_BINS + bins
    tl.store(partial_counts + count_offsets, counts)


@libentry()
@triton.jit
def nanmedian_ascend_byte_histogram_update_kernel(
    inp,
    partial_counts,
    state,
    M,
    NUM_CHUNKS: tl.constexpr,
    HISTOGRAM_BINS: tl.constexpr,
    DIGIT_POS: tl.constexpr,
):
    pid = tle.program_id(0)
    bins = tl.arange(0, HISTOGRAM_BINS)
    counts = tl.zeros((HISTOGRAM_BINS,), dtype=tl.int32)

    for chunk in tl.range(0, NUM_CHUNKS):
        count_offsets = (pid * NUM_CHUNKS + chunk) * HISTOGRAM_BINS + bins
        counts += tl.load(partial_counts + count_offsets)

    state_base = pid * 3
    k_to_find = tl.load(state + state_base + 2).to(tl.int32)
    cumsum = tl.cumsum(counts, axis=0)
    prev = cumsum - counts
    take = (k_to_find <= cumsum) & (k_to_find > prev)
    selected_bin = tl.min(tl.where(take, bins, HISTOGRAM_BINS - 1), axis=0)
    counts_before = tl.max(tl.where(take, prev, 0), axis=0)

    dtype = inp.dtype.element_ty
    nbits: tl.constexpr = dtype.primitive_bitwidth
    utype = tl.dtype(f"uint{nbits}")
    byte_mask_val = tl.full((), HISTOGRAM_BINS - 1, dtype=utype)
    desired = tl.load(state + state_base + 0).to(utype)
    desired_mask = tl.load(state + state_base + 1).to(utype)
    selected_bin = selected_bin.to(utype)

    desired = desired | (selected_bin << DIGIT_POS)
    desired_mask = desired_mask | (byte_mask_val << DIGIT_POS)
    tl.store(state + state_base + 0, desired)
    tl.store(state + state_base + 1, desired_mask)
    tl.store(state + state_base + 2, k_to_find - counts_before)


@libentry()
@triton.jit
def nanmedian_ascend_byte_histogram_find_index_kernel(
    inp,
    state,
    out_values,
    out_indices,
    M,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    dtype = inp.dtype.element_ty
    nbits: tl.constexpr = dtype.primitive_bitwidth
    utype = tl.dtype(f"uint{nbits}")
    desired = tl.load(state + pid * 3 + 0).to(utype)

    result_idx = tl.full((), N, dtype=tl.int32)
    for start in tl.range(0, N, BLOCK_N):
        cols = start + offsets
        mask = cols < N
        vals = tl.load(inp + pid * N + cols, mask=mask, other=0)
        keys = _to_order_key(vals, mask)
        local_idx = tl.min(tl.where(mask & (keys == desired), cols, N), axis=0)
        result_idx = tl.where(local_idx < result_idx, local_idx, result_idx)

    result_val = tl.load(inp + pid * N + result_idx)
    tl.store(out_values + pid, result_val)
    tl.store(out_indices + pid, result_idx)


@libentry()
@triton.jit
def nanmedian_radix_select_kernel(
    inp,
    out_values,
    out_indices,
    M,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    RADIX_BITS_: tl.constexpr,
    USE_ISNAN: tl.constexpr,
    USE_HISTOGRAM: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    dtype = inp.dtype.element_ty
    nbits: tl.constexpr = dtype.primitive_bitwidth
    utype = tl.dtype(f"uint{nbits}")
    radix_size: tl.constexpr = 1 << RADIX_BITS_
    radix_mask: tl.constexpr = radix_size - 1
    radix_bins = tl.arange(0, radix_size)

    valid_count = tl.full((), 0, dtype=tl.int32)
    for start in tl.range(0, N, BLOCK_N):
        cols = start + offsets
        mask = cols < N
        vals = tl.load(inp + pid * N + cols, mask=mask, other=0.0)
        if dtype.is_floating():
            valid = mask & _is_not_nan(vals, USE_ISNAN)
        else:
            valid = mask
        valid_count += tl.sum(valid.to(tl.int32), axis=0)

    k_to_find = (valid_count + 1) // 2
    desired = tl.full((), 0, dtype=utype)
    desired_mask = tl.full((), 0, dtype=utype)
    radix_mask_val = tl.full((), radix_mask, dtype=utype)

    for digit_pos in tl.static_range(nbits - RADIX_BITS_, -1, -RADIX_BITS_):
        counts = tl.zeros((radix_size,), dtype=tl.int32)
        for start in tl.range(0, N, BLOCK_N):
            cols = start + offsets
            mask = cols < N
            vals = tl.load(inp + pid * N + cols, mask=mask, other=0.0)
            if dtype.is_floating():
                valid = mask & _is_not_nan(vals, USE_ISNAN)
            else:
                valid = mask
            keys = _to_order_key(vals, valid)
            matches = (keys & desired_mask) == desired
            digit = ((keys >> digit_pos) & radix_mask_val).to(tl.int32)
            active = valid & matches
            if USE_HISTOGRAM:
                safe_digit = tl.where(active, digit, 0)
                chunk_counts = tl.histogram(safe_digit, radix_size).to(tl.int32)
                inactive_count = tl.sum((~active).to(tl.int32), axis=0)
                counts += chunk_counts - tl.where(radix_bins == 0, inactive_count, 0)
            else:
                for radix_bin in tl.static_range(0, radix_size):
                    bin_count = tl.sum(
                        (active & (digit == radix_bin)).to(tl.int32), axis=0
                    )
                    counts += tl.where(radix_bins == radix_bin, bin_count, 0)

        cumsum = tl.cumsum(counts, axis=0)
        prev = cumsum - counts
        take = (cumsum >= k_to_find) & (prev < k_to_find)
        selected_bin = tl.min(tl.where(take, radix_bins, radix_size - 1), axis=0)
        counts_before = tl.max(tl.where(take, prev, 0), axis=0)

        selected_bin = selected_bin.to(utype)
        desired = desired | (selected_bin << digit_pos)
        desired_mask = desired_mask | (radix_mask_val << digit_pos)
        k_to_find = k_to_find - counts_before

    result_idx = tl.full((), N, dtype=tl.int32)
    for start in tl.range(0, N, BLOCK_N):
        cols = start + offsets
        mask = cols < N
        vals = tl.load(inp + pid * N + cols, mask=mask, other=0.0)
        if dtype.is_floating():
            valid = mask & _is_not_nan(vals, USE_ISNAN)
        else:
            valid = mask
        keys = _to_order_key(vals, valid)
        local_idx = tl.min(tl.where(valid & (keys == desired), cols, N), axis=0)
        result_idx = tl.where(local_idx < result_idx, local_idx, result_idx)

    if dtype.is_floating():
        fallback_value = _get_finfo_val(dtype, return_max=False)
    else:
        fallback_value = get_dtype_min(dtype)
    result_val = tl.load(
        inp + pid * N + result_idx, mask=valid_count > 0, other=fallback_value
    )

    if dtype.is_floating():
        all_nan = valid_count == 0
        result_val = tl.where(all_nan, float("nan"), result_val)
        result_idx = tl.where(all_nan, 0, result_idx)

    tl.store(out_values + pid, result_val)
    tl.store(out_indices + pid, result_idx)


@libentry()
@triton.jit
def flat_radix_init_kernel(
    valid_count,
    state,
    result_idx,
    N: tl.constexpr,
    IS_FLOAT: tl.constexpr,
):
    tl.store(valid_count, 0 if IS_FLOAT else N)
    tl.store(state + 0, 0)
    tl.store(state + 1, 0)
    tl.store(state + 2, 0)
    tl.store(result_idx, N)


@libentry()
@triton.jit
def flat_radix_count_valid_kernel(
    inp,
    valid_count,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_ISNAN: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < N
    vals = tl.load(inp + offsets, mask=mask, other=0.0)
    valid = mask & _is_not_nan(vals, USE_ISNAN)
    count = tl.sum(valid.to(tl.int64), axis=0)
    tl.atomic_add(valid_count, count, sem="relaxed")


@libentry()
@triton.jit
def flat_radix_init_rank_kernel(valid_count, state):
    count = tl.load(valid_count)
    tl.store(state + 2, (count + 1) // 2)


@libentry()
@triton.jit
def flat_radix_count_kernel(
    inp,
    bin_counts,
    state,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DIGIT_POS: tl.constexpr,
    RADIX_BITS_: tl.constexpr,
    RADIX_SIZE: tl.constexpr,
    USE_ISNAN: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < N
    vals = tl.load(inp + offsets, mask=mask, other=0.0)
    dtype = inp.dtype.element_ty
    nbits: tl.constexpr = dtype.primitive_bitwidth
    utype = tl.dtype(f"uint{nbits}")
    radix_mask: tl.constexpr = (1 << RADIX_BITS_) - 1
    radix_mask_val = tl.full((), radix_mask, dtype=utype)

    if dtype.is_floating():
        valid = mask & _is_not_nan(vals, USE_ISNAN)
    else:
        valid = mask

    desired = tl.load(state + 0).to(utype)
    desired_mask = tl.load(state + 1).to(utype)
    keys = _to_order_key(vals, valid)
    active = valid & ((keys & desired_mask) == desired)
    digit = ((keys >> DIGIT_POS) & radix_mask_val).to(tl.int32)
    bins = tl.arange(0, RADIX_SIZE)
    safe_digit = tl.where(active, digit, 0)
    counts = tl.histogram(safe_digit, RADIX_SIZE).to(tl.int64)
    inactive_count = tl.sum((~active).to(tl.int64), axis=0)
    counts -= tl.where(bins == 0, inactive_count, 0)
    tl.atomic_add(bin_counts + bins, counts, sem="relaxed")


@libentry()
@triton.jit
def flat_radix_update_kernel(
    bin_counts,
    state,
    DIGIT_POS: tl.constexpr,
    RADIX_BITS_: tl.constexpr,
    RADIX_SIZE: tl.constexpr,
):
    bins = tl.arange(0, RADIX_SIZE)
    counts = tl.load(bin_counts + bins)
    k_to_find = tl.load(state + 2)
    cumsum = tl.cumsum(counts, axis=0)
    prev = cumsum - counts
    take = (k_to_find <= cumsum) & (k_to_find > prev)
    selected_bin = tl.min(tl.where(take, bins, RADIX_SIZE - 1), axis=0).to(tl.int64)
    counts_before = tl.max(tl.where(take, prev, 0), axis=0)

    desired = tl.load(state + 0)
    desired_mask = tl.load(state + 1)
    radix_mask: tl.constexpr = (1 << RADIX_BITS_) - 1
    desired = desired | (selected_bin << DIGIT_POS)
    desired_mask = desired_mask | (radix_mask << DIGIT_POS)
    tl.store(state + 0, desired)
    tl.store(state + 1, desired_mask)
    tl.store(state + 2, k_to_find - counts_before)


@libentry()
@triton.jit
def flat_radix_find_index_kernel(
    inp,
    state,
    valid_count,
    result_idx,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_ISNAN: tl.constexpr,
):
    if tl.load(valid_count) > 0:
        pid = tle.program_id(0)
        offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < N
        vals = tl.load(inp + offsets, mask=mask, other=0.0)
        dtype = inp.dtype.element_ty
        nbits: tl.constexpr = dtype.primitive_bitwidth
        utype = tl.dtype(f"uint{nbits}")

        if dtype.is_floating():
            valid = mask & _is_not_nan(vals, USE_ISNAN)
        else:
            valid = mask

        desired = tl.load(state + 0).to(utype)
        keys = _to_order_key(vals, valid)
        local_idx = tl.min(tl.where(valid & (keys == desired), offsets, N), axis=0)
        tl.atomic_min(result_idx, local_idx, sem="relaxed")


@libentry()
@triton.jit
def flat_radix_store_result_kernel(inp, out, valid_count, result_idx):
    dtype = inp.dtype.element_ty
    idx = tl.load(result_idx)
    if dtype.is_floating():
        result = tl.load(inp + idx, mask=tl.load(valid_count) > 0, other=float("nan"))
    else:
        result = tl.load(inp + idx)
    tl.store(out, result)


def _check_supported_dtype(inp):
    if inp.dtype is torch.bool:
        raise NotImplementedError("\"median_out_impl\" not implemented for 'Bool'")


def _normalize_dim(dim, ndim):
    if ndim == 0:
        if dim in (0, -1):
            return 0
    elif -ndim <= dim < ndim:
        return dim % ndim
    raise IndexError(
        f"Dimension out of range (expected to be in range of [{-ndim}, {ndim - 1}], but got {dim})"
    )


def _empty_flat_value(inp):
    out = torch.empty((), dtype=inp.dtype, device=inp.device)
    if torch.is_floating_point(inp):
        out.fill_(float("nan"))
    elif inp.is_cuda:
        out.fill_(torch.iinfo(inp.dtype).min)
    else:
        out.zero_()
    return out


def _use_radix_select(inp, n):
    if inp.is_cuda and inp.dtype in RADIX_SELECT_DTYPES:
        return n <= LONG_RADIX_REDUCTION_N
    return False


def _use_ascend_float_sort_select(inp, n):
    return (
        inp.device.type == "npu"
        and inp.dtype in ASCEND_FLOAT_SELECT_DTYPES
        and MAX_BLOCK_N < n <= LONG_RADIX_REDUCTION_N
    )


def _use_ascend_histogram_select(inp, n):
    return (
        inp.device.type == "npu"
        and inp.dtype in ASCEND_HISTOGRAM_SELECT_DTYPES
        and MAX_BLOCK_N < n <= LONG_RADIX_REDUCTION_N
    )


def _use_ascend_multi_histogram_select(inp, n):
    return _use_ascend_histogram_select(inp, n) and n >= ASCEND_MULTI_HISTOGRAM_MIN_N


def _use_ascend_byte_histogram_select(inp, n):
    return (
        inp.device.type == "npu"
        and inp.dtype in ASCEND_BYTE_HISTOGRAM_SELECT_DTYPES
        and MAX_BLOCK_N < n <= LONG_RADIX_REDUCTION_N
    )


def _use_ascend_multi_byte_histogram_select(inp, n):
    return (
        _use_ascend_byte_histogram_select(inp, n) and n >= ASCEND_MULTI_HISTOGRAM_MIN_N
    )


def _use_cuda_flat_radix_select(inp):
    return (
        inp.is_cuda
        and inp.dtype in RADIX_SELECT_DTYPES
        and LONG_RADIX_REDUCTION_N < inp.numel() <= torch.iinfo(torch.int32).max
    )


def _use_ascend_flat_sort(inp):
    return (
        inp.device.type == "npu"
        and inp.dtype
        in (
            torch.float16,
            torch.float32,
            torch.int8,
            torch.uint8,
            torch.int16,
            torch.int32,
        )
        and inp.numel() >= ASCEND_FLAT_SORT_MIN_N
    )


def _radix_block_n(inp, n):
    block_n = triton.next_power_of_2(n)
    if inp.is_cuda:
        if n > LARGE_FLOAT_REDUCTION_N:
            return min(block_n, 8192)
        if n > MEDIUM_REDUCTION_N:
            return min(block_n, 4096)
        if inp.dtype is torch.uint8:
            return min(block_n, 512)
        return min(block_n, RADIX_BLOCK_N)
    if inp.dtype in (torch.float16, torch.bfloat16):
        if n > LARGE_FLOAT_REDUCTION_N:
            return 2048
        return min(block_n, 2048)
    if inp.dtype is torch.float32 or inp.dtype is torch.int32:
        if n > MEDIUM_REDUCTION_N:
            return 512
        return min(block_n, RADIX_BLOCK_N)
    if inp.dtype in (torch.int8, torch.uint8):
        if n > MEDIUM_REDUCTION_N:
            return RADIX_BLOCK_N
        return min(block_n, 512)
    return min(block_n, RADIX_BLOCK_N)


def _radix_bits(inp, n):
    if inp.is_cuda:
        if n > LARGE_FLOAT_REDUCTION_N:
            return 8
        if n > MEDIUM_REDUCTION_N:
            return 4
    return RADIX_BITS


def _full_nan_result(shape, dtype, device):
    values = torch.full(shape, float("nan"), dtype=dtype, device=device)
    indices = torch.zeros(shape, dtype=torch.long, device=device)
    return NanMedian(values=values, indices=indices)


def _count_block_n(inp, n):
    block_n = triton.next_power_of_2(n)
    if inp.is_cuda and n >= LONG_RADIX_REDUCTION_N:
        return min(block_n, 16384)
    if n >= LONG_RADIX_REDUCTION_N:
        return min(block_n, 4096)
    if n >= LARGE_FLOAT_REDUCTION_N:
        return min(block_n, 2048)
    return min(block_n, RADIX_BLOCK_N)


def _nanmedian_kthvalue_fallback(inp, M, N):
    inp = inp.reshape(M, N)
    if torch.is_floating_point(inp):
        valid_count = torch.empty((M,), dtype=torch.long, device=inp.device)
        block_n = _count_block_n(inp, N)
        with torch_device_fn.device(inp.device):
            count_valid_kernel[(M,)](inp, valid_count, M, N, block_n, inp.is_cuda)
        min_count = int(torch.min(valid_count).item())
        max_count = int(torch.max(valid_count).item())
        if min_count == max_count:
            if max_count == 0:
                return _full_nan_result((M,), inp.dtype, inp.device)
            values, indices = torch.kthvalue(inp, (max_count + 1) // 2, dim=1)
            return NanMedian(values=values, indices=indices)

        if max_count - min_count <= 1:
            min_k = (min_count + 1) // 2 if min_count > 0 else 0
            max_k = (max_count + 1) // 2

            if min_k == max_k:
                values, indices = torch.kthvalue(inp, max_k, dim=1)
                if min_count > 0:
                    return NanMedian(values=values, indices=indices)
                fallback = _full_nan_result((M,), inp.dtype, inp.device)
                positive = valid_count > 0
                return NanMedian(
                    values=torch.where(positive, values, fallback.values),
                    indices=torch.where(positive, indices, fallback.indices),
                )

            result = _full_nan_result((M,), inp.dtype, inp.device)

            if min_count > 0:
                values, indices = torch.kthvalue(inp, min_k, dim=1)
                mask = valid_count == min_count
                result = NanMedian(
                    values=torch.where(mask, values, result.values),
                    indices=torch.where(mask, indices, result.indices),
                )

            values, indices = torch.kthvalue(inp, max_k, dim=1)
            mask = valid_count == max_count
            return NanMedian(
                values=torch.where(mask, values, result.values),
                indices=torch.where(mask, indices, result.indices),
            )

        result = _full_nan_result((M,), inp.dtype, inp.device)
        for count in torch.unique(valid_count).tolist():
            count = int(count)
            if count == 0:
                continue
            row_indices = torch.nonzero(valid_count == count).flatten()
            rows = torch.index_select(inp, 0, row_indices)
            values, indices = torch.kthvalue(rows, (count + 1) // 2, dim=1)
            result.values[row_indices] = values
            result.indices[row_indices] = indices
        return result
    else:
        values, indices = torch.kthvalue(inp, (N + 1) // 2, dim=1)
        return NanMedian(values=values, indices=indices)


def _nanmedian_ascend_float_sort_select(inp, M, N, values, indices):
    inp = inp.reshape(M, N)
    flat_values = values.reshape(M)
    flat_indices = indices.reshape(M)
    if N <= LARGE_FLOAT_REDUCTION_N:
        cleaned = torch.empty_like(inp)
        valid_counts = torch.empty((M,), dtype=torch.int32, device=inp.device)
        block_n = min(triton.next_power_of_2(N), RADIX_BLOCK_N)
        num_warps = 4 if block_n <= 512 else 8
        with torch_device_fn.device(inp.device):
            nanmedian_float_clean_count_kernel[(M,)](
                inp,
                cleaned,
                valid_counts,
                N,
                block_n,
                num_warps=num_warps,
                num_stages=1,
            )
        sorted_values, sorted_indices = torch.sort(cleaned, dim=1)
    else:
        sorted_values, sorted_indices = torch.sort(inp, dim=1)
        valid_counts = torch.sum(
            (sorted_values == sorted_values).to(torch.int32), dim=1
        )

    with torch_device_fn.device(inp.device):
        nanmedian_float_sorted_gather_kernel[(M,)](
            sorted_values,
            sorted_indices,
            valid_counts,
            flat_values,
            flat_indices,
            N,
            num_warps=1,
            num_stages=1,
        )


def _nanmedian_dim_impl(inp, dim, keepdim, out=None, use_ascend_float_select=True):
    dim = _normalize_dim(dim, inp.ndim)

    if inp.ndim == 0:
        if out is None:
            values = inp.clone()
            indices = torch.zeros((), dtype=torch.long, device=inp.device)
        else:
            values, indices = out
            values.copy_(inp)
            indices.zero_()
        return NanMedian(values=values, indices=indices)

    shape = list(inp.shape)
    N = shape[dim]
    out_shape = shape[:dim] + shape[dim + 1 :]
    M = math.prod(out_shape)

    keepdim_shape = shape.copy()
    keepdim_shape[dim] = 1
    output_shape = keepdim_shape if keepdim else out_shape
    compute_shape = output_shape if out is not None else keepdim_shape

    if N == 0:
        if M != 0:
            raise IndexError(
                f"median(): Expected reduction dim {dim} to have non-zero size."
            )
        if out is None:
            values = torch.empty(compute_shape, dtype=inp.dtype, device=inp.device)
            indices = torch.empty(compute_shape, dtype=torch.long, device=inp.device)
            if not keepdim:
                values = torch.squeeze(values, dim)
                indices = torch.squeeze(indices, dim)
        else:
            values, indices = out
        return NanMedian(values=values, indices=indices)

    if out is None:
        values = torch.empty(compute_shape, dtype=inp.dtype, device=inp.device)
        indices = torch.empty(compute_shape, dtype=torch.long, device=inp.device)
    else:
        values, indices = out

    if M == 0:
        if out is None and not keepdim:
            values = torch.squeeze(values, dim)
            indices = torch.squeeze(indices, dim)
        return NanMedian(values=values, indices=indices)

    inp = dim_compress(inp, dim)

    if inp.dtype in RADIX_SELECT_DTYPES and _use_radix_select(inp, N):
        flat_values = values.reshape(M)
        flat_indices = indices.reshape(M)
        block_n = _radix_block_n(inp, N)
        num_warps = 4 if block_n <= 512 else 8
        with torch_device_fn.device(inp.device):
            nanmedian_radix_select_kernel[(M,)](
                inp,
                flat_values,
                flat_indices,
                M,
                N,
                block_n,
                _radix_bits(inp, N),
                inp.is_cuda,
                inp.is_cuda,
                num_warps=num_warps,
                num_stages=1,
            )
    elif use_ascend_float_select and _use_ascend_float_sort_select(inp, N):
        _nanmedian_ascend_float_sort_select(inp, M, N, values, indices)
    elif _use_ascend_multi_histogram_select(inp, N):
        flat_values = values.reshape(M)
        flat_indices = indices.reshape(M)
        block_n = _radix_block_n(inp, N)
        num_chunks = triton.cdiv(N, block_n)
        partial_counts = torch.empty(
            (M, num_chunks, ASCEND_HISTOGRAM_BINS),
            dtype=torch.int32,
            device=inp.device,
        )
        num_warps = 4 if block_n <= 512 else 8
        with torch_device_fn.device(inp.device):
            nanmedian_ascend_histogram_count_kernel[(M, num_chunks)](
                inp,
                partial_counts,
                M,
                N,
                block_n,
                num_chunks,
                ASCEND_HISTOGRAM_BINS,
                num_warps=num_warps,
                num_stages=1,
            )
            nanmedian_ascend_histogram_reduce_kernel[(M,)](
                inp,
                partial_counts,
                flat_values,
                flat_indices,
                M,
                N,
                block_n,
                num_chunks,
                ASCEND_HISTOGRAM_BINS,
                num_warps=num_warps,
                num_stages=1,
            )
    elif _use_ascend_histogram_select(inp, N):
        flat_values = values.reshape(M)
        flat_indices = indices.reshape(M)
        block_n = _radix_block_n(inp, N)
        num_warps = 4 if block_n <= 512 else 8
        with torch_device_fn.device(inp.device):
            nanmedian_ascend_histogram_select_kernel[(M,)](
                inp,
                flat_values,
                flat_indices,
                M,
                N,
                block_n,
                ASCEND_HISTOGRAM_BINS,
                num_warps=num_warps,
                num_stages=1,
            )
    elif _use_ascend_multi_byte_histogram_select(inp, N):
        flat_values = values.reshape(M)
        flat_indices = indices.reshape(M)
        block_n = _radix_block_n(inp, N)
        num_chunks = triton.cdiv(N, block_n)
        partial_counts = torch.empty(
            (M, num_chunks, ASCEND_HISTOGRAM_BINS),
            dtype=torch.int32,
            device=inp.device,
        )
        state = torch.empty((M, 3), dtype=torch.int64, device=inp.device)
        num_warps = 4 if block_n <= 512 else 8
        nbits = inp.element_size() * 8
        with torch_device_fn.device(inp.device):
            nanmedian_ascend_byte_histogram_init_kernel[(M,)](
                state,
                M,
                N,
                num_warps=1,
                num_stages=1,
            )
            for digit_pos in range(nbits - 8, -1, -8):
                nanmedian_ascend_byte_histogram_count_kernel[(M, num_chunks)](
                    inp,
                    state,
                    partial_counts,
                    M,
                    N,
                    block_n,
                    num_chunks,
                    ASCEND_HISTOGRAM_BINS,
                    digit_pos,
                    num_warps=num_warps,
                    num_stages=1,
                )
                nanmedian_ascend_byte_histogram_update_kernel[(M,)](
                    inp,
                    partial_counts,
                    state,
                    M,
                    num_chunks,
                    ASCEND_HISTOGRAM_BINS,
                    digit_pos,
                    num_warps=num_warps,
                    num_stages=1,
                )
            nanmedian_ascend_byte_histogram_find_index_kernel[(M,)](
                inp,
                state,
                flat_values,
                flat_indices,
                M,
                N,
                block_n,
                num_warps=num_warps,
                num_stages=1,
            )
    elif _use_ascend_byte_histogram_select(inp, N):
        flat_values = values.reshape(M)
        flat_indices = indices.reshape(M)
        block_n = _radix_block_n(inp, N)
        num_warps = 4 if block_n <= 512 else 8
        with torch_device_fn.device(inp.device):
            nanmedian_ascend_byte_histogram_select_kernel[(M,)](
                inp,
                flat_values,
                flat_indices,
                M,
                N,
                block_n,
                ASCEND_HISTOGRAM_BINS,
                num_warps=num_warps,
                num_stages=1,
            )
    elif N <= MAX_BLOCK_N and inp.dtype is not torch.float64:
        flat_values = values.reshape(M)
        flat_indices = indices.reshape(M)
        block_n = triton.next_power_of_2(N)
        with torch_device_fn.device(inp.device):
            nanmedian_select_kernel[(M,)](
                inp,
                flat_values,
                flat_indices,
                M,
                N,
                block_n,
                inp.is_cuda,
            )
    else:
        result = _nanmedian_kthvalue_fallback(inp, M, N)
        computed_values = result.values.reshape(compute_shape)
        computed_indices = result.indices.reshape(compute_shape)
        if out is None:
            values = computed_values
            indices = computed_indices
        else:
            values.copy_(computed_values)
            indices.copy_(computed_indices)

    if out is None and not keepdim:
        values = torch.squeeze(values, dim)
        indices = torch.squeeze(indices, dim)

    return NanMedian(values=values, indices=indices)


def _nanmedian_ascend_flat_sort(inp):
    flat = inp.reshape(-1).contiguous()
    sorted_values = torch.sort(flat).values
    if torch.is_floating_point(flat):
        valid_count = (sorted_values == sorted_values).sum()
        rank = (valid_count - 1) // 2
    else:
        rank = (flat.numel() - 1) // 2
    return sorted_values[rank]


def _nanmedian_cuda_flat_radix_select(inp):
    flat = inp.reshape(-1).contiguous()
    n = flat.numel()
    out = torch.empty((), dtype=flat.dtype, device=flat.device)
    valid_count = torch.empty((), dtype=torch.int64, device=flat.device)
    state = torch.empty((3,), dtype=torch.int64, device=flat.device)
    result_idx = torch.empty((), dtype=torch.int64, device=flat.device)
    bin_counts = torch.empty(
        (1 << FLAT_RADIX_BITS,), dtype=torch.int64, device=flat.device
    )
    block_n = min(triton.next_power_of_2(n), FLAT_RADIX_BLOCK_N)
    grid = (triton.cdiv(n, block_n),)
    nbits = flat.element_size() * 8

    with torch_device_fn.device(flat.device):
        flat_radix_init_kernel[(1,)](
            valid_count,
            state,
            result_idx,
            n,
            torch.is_floating_point(flat),
        )
        if torch.is_floating_point(flat):
            flat_radix_count_valid_kernel[grid](
                flat,
                valid_count,
                n,
                block_n,
                True,
                num_warps=8,
                num_stages=1,
            )
        flat_radix_init_rank_kernel[(1,)](valid_count, state)
        for digit_pos in range(nbits - FLAT_RADIX_BITS, -1, -FLAT_RADIX_BITS):
            bin_counts.zero_()
            flat_radix_count_kernel[grid](
                flat,
                bin_counts,
                state,
                n,
                block_n,
                digit_pos,
                FLAT_RADIX_BITS,
                1 << FLAT_RADIX_BITS,
                True,
                num_warps=8,
                num_stages=1,
            )
            flat_radix_update_kernel[(1,)](
                bin_counts,
                state,
                digit_pos,
                FLAT_RADIX_BITS,
                1 << FLAT_RADIX_BITS,
                num_warps=8,
                num_stages=1,
            )
        flat_radix_find_index_kernel[grid](
            flat,
            state,
            valid_count,
            result_idx,
            n,
            block_n,
            True,
            num_warps=8,
            num_stages=1,
        )
        flat_radix_store_result_kernel[(1,)](flat, out, valid_count, result_idx)
    return out


def nanmedian(inp):
    logger.debug("GEMS NANMEDIAN")
    _check_supported_dtype(inp)
    if inp.numel() == 0:
        return _empty_flat_value(inp)
    if _use_cuda_flat_radix_select(inp):
        return _nanmedian_cuda_flat_radix_select(inp)
    if _use_ascend_flat_sort(inp):
        return _nanmedian_ascend_flat_sort(inp)
    return _nanmedian_dim_impl(
        inp.reshape(-1), 0, False, use_ascend_float_select=False
    ).values


def nanmedian_out(inp, *, out):
    logger.debug("GEMS NANMEDIAN OUT")
    result = nanmedian(inp)
    out.copy_(result)
    return out


def nanmedian_dim(inp, dim=-1, keepdim=False):
    logger.debug("GEMS NANMEDIAN DIM")
    _check_supported_dtype(inp)
    return _nanmedian_dim_impl(inp, dim, keepdim)


def nanmedian_dim_values(inp, dim=-1, keepdim=False, *, values, indices):
    logger.debug("GEMS NANMEDIAN DIM VALUES")
    return _nanmedian_dim_impl(inp, dim, keepdim, out=(values, indices))
