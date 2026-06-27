import logging
import math

import torch
import triton
import triton.language as tl
import triton.language.core as core

try:
    # TODO: Triton 2.1 does not implement _log2.
    # Remove the try-catch block once all vendors upgrade to a newer version of Triton.
    from triton.language.standard import _log2, zeros_like
except ImportError:
    pass

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.limits import get_dtype_max, get_dtype_min
from flag_gems.utils.triton_version_utils import HAS_TLE

if HAS_TLE:
    import triton.experimental.tle.language as tle_gpu
else:
    tle_gpu = None

logger = logging.getLogger(__name__)
_MIN_FLOAT32_VAL = tl.constexpr(torch.finfo(torch.float32).min)
_MAX_FLOAT32_VAL = tl.constexpr(torch.finfo(torch.float32).max)
_MIN_FLOAT16_VAL = tl.constexpr(torch.finfo(torch.float16).min)
_MAX_FLOAT16_VAL = tl.constexpr(torch.finfo(torch.float16).max)
_MIN_BFLOAT16_VAL = tl.constexpr(torch.finfo(torch.bfloat16).min)
_MAX_BFLOAT16_VAL = tl.constexpr(torch.finfo(torch.bfloat16).max)
_MIN_INT8_VAL = tl.constexpr(torch.iinfo(torch.int8).min)
_MAX_INT8_VAL = tl.constexpr(torch.iinfo(torch.int8).max)
_MIN_INT16_VAL = tl.constexpr(torch.iinfo(torch.int16).min)
_MAX_INT16_VAL = tl.constexpr(torch.iinfo(torch.int16).max)
_MIN_INT32_VAL = tl.constexpr(torch.iinfo(torch.int32).min)
_MAX_INT32_VAL = tl.constexpr(torch.iinfo(torch.int32).max)
_MIN_INT64_VAL = tl.constexpr(torch.iinfo(torch.int64).min)
_MAX_INT64_VAL = tl.constexpr(torch.iinfo(torch.int64).max)


@triton.jit
def _get_finfo_val(
    dtype,
    return_max,
):
    if dtype is tl.float32:
        if return_max:
            return _MAX_FLOAT32_VAL
        else:
            return _MIN_FLOAT32_VAL
    elif dtype is tl.float16:
        if return_max:
            return _MAX_FLOAT16_VAL
        else:
            return _MIN_FLOAT16_VAL
    elif dtype is tl.bfloat16:
        if return_max:
            return _MAX_BFLOAT16_VAL
        else:
            return _MIN_BFLOAT16_VAL


@triton.jit
def _get_iinfo_val(
    dtype,
    return_max,
):
    if return_max:
        return get_dtype_max(dtype)
    else:
        return get_dtype_min(dtype)


@libentry()
@triton.jit
def topk_single_stage_kernel(
    y_ptr,
    index_ptr,
    x_ptr,
    k: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DESCENDING: tl.constexpr,
):
    cur_batch = tle.program_id(0)
    x_ptr += cur_batch * N
    y_ptr += cur_batch * k
    index_ptr += cur_batch * k

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    pad_val = float("-inf") if DESCENDING else float("inf")
    mask_index_val = _MIN_INT32_VAL if DESCENDING else _MAX_INT32_VAL

    x_val = tl.load(x_ptr + cols, mask=mask, other=pad_val).to(tl.float32)
    idx_val = tl.where(mask, cols, mask_index_val).to(tl.int32)

    sorted_x, sorted_idx = argsort(x_val, idx_val, dim=0, descending=DESCENDING)

    out_mask = cols < k
    out_val = sorted_x.to(tl.load(x_ptr).dtype)

    tl.store(y_ptr + cols, out_val, mask=out_mask)
    tl.store(index_ptr + cols, sorted_idx.to(tl.int64), mask=out_mask)


@libentry()
@triton.jit
def topk_fp8_single_stage_kernel(
    y_ptr,
    index_ptr,
    x_ptr,
    scale_ptr,
    k: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DESCENDING: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
):
    cur_batch = tle.program_id(0)
    x_ptr += cur_batch * N
    scale_ptr += cur_batch * NUM_GROUPS
    y_ptr += cur_batch * k
    index_ptr += cur_batch * k

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    pad_val = float("-inf") if DESCENDING else float("inf")
    mask_index_val = _MIN_INT32_VAL if DESCENDING else _MAX_INT32_VAL

    x_q = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x_scale = tl.load(
        scale_ptr + cols // GROUP_SIZE, mask=mask, other=0.0
    ).to(tl.float32)
    x_val = tl.where(mask, x_q * x_scale, pad_val)
    idx_val = tl.where(mask, cols, mask_index_val).to(tl.int32)

    sorted_x, sorted_idx = argsort(x_val, idx_val, dim=0, descending=DESCENDING)

    out_mask = cols < k
    tl.store(y_ptr + cols, sorted_x, mask=out_mask)
    tl.store(index_ptr + cols, sorted_idx.to(tl.int64), mask=out_mask)


@libentry()
@triton.jit
def topk_stage1_kernel(
    y_ptr,
    index_ptr,
    x_ptr,
    k,
    N: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    DESCENDING: tl.constexpr,
):
    cur_batch = tle.program_id(0)
    cur_chunk_idx = tle.program_id(1)
    chunk_num = tle.num_programs(1)

    y_ptr += cur_batch * chunk_num * k + cur_chunk_idx * k
    index_ptr += cur_batch * chunk_num * k + cur_chunk_idx * k

    chunk_offset = cur_chunk_idx * CHUNK_SIZE
    x_ptr += cur_batch * N + chunk_offset

    cols = tl.arange(0, CHUNK_SIZE)
    mask = (chunk_offset + cols) < N

    mask_val = float("-inf") if DESCENDING else float("inf")
    x_val = tl.load(x_ptr + cols, mask=mask, other=mask_val).to(tl.float32)
    available = mask

    for k_idx in range(k):
        if DESCENDING:
            chunk_select_val = tl.max(x_val)
        else:
            chunk_select_val = tl.min(x_val)
        is_candidate = available & (x_val == chunk_select_val)
        candidate_indices = tl.where(is_candidate, cols, CHUNK_SIZE)
        chunk_select_idx = tl.argmin(candidate_indices, axis=0)

        tl.store(y_ptr + k_idx, chunk_select_val)
        tl.store(index_ptr + k_idx, chunk_select_idx + chunk_offset)
        if DESCENDING:
            x_val = tl.where(cols == chunk_select_idx, float("-inf"), x_val)
        else:
            x_val = tl.where(cols == chunk_select_idx, float("inf"), x_val)
        available = available & (cols != chunk_select_idx)


@libentry()
@triton.jit
def topk_fp8_stage1_kernel(
    y_ptr,
    index_ptr,
    x_ptr,
    scale_ptr,
    k,
    N: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    DESCENDING: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
):
    cur_batch = tle.program_id(0)
    cur_chunk_idx = tle.program_id(1)
    chunk_num = tle.num_programs(1)

    y_ptr += cur_batch * chunk_num * k + cur_chunk_idx * k
    index_ptr += cur_batch * chunk_num * k + cur_chunk_idx * k

    chunk_offset = cur_chunk_idx * CHUNK_SIZE
    x_ptr += cur_batch * N + chunk_offset
    scale_ptr += cur_batch * NUM_GROUPS

    cols = tl.arange(0, CHUNK_SIZE)
    global_cols = chunk_offset + cols
    mask = global_cols < N

    mask_val = float("-inf") if DESCENDING else float("inf")
    x_q = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x_scale = tl.load(
        scale_ptr + global_cols // GROUP_SIZE, mask=mask, other=0.0
    ).to(tl.float32)
    x_val = tl.where(mask, x_q * x_scale, mask_val)
    available = mask

    for k_idx in range(k):
        if DESCENDING:
            chunk_select_val = tl.max(x_val)
        else:
            chunk_select_val = tl.min(x_val)
        is_candidate = available & (x_val == chunk_select_val)
        candidate_indices = tl.where(is_candidate, cols, CHUNK_SIZE)
        chunk_select_idx = tl.argmin(candidate_indices, axis=0)

        tl.store(y_ptr + k_idx, chunk_select_val)
        tl.store(index_ptr + k_idx, chunk_select_idx + chunk_offset)
        if DESCENDING:
            x_val = tl.where(cols == chunk_select_idx, float("-inf"), x_val)
        else:
            x_val = tl.where(cols == chunk_select_idx, float("inf"), x_val)
        available = available & (cols != chunk_select_idx)


"""
Note(Zhengzekang):
Refer from triton2.2 official `sort` implementation:
https://github.com/triton-lang/triton/blob/release/2.2.x/python/triton/language/standard.py#L392-L404
Just add indices to sort with values.
"""


@triton.jit
def _compare_and_swap(x, ids, flip, i: core.constexpr, n_dims: core.constexpr):
    n_outer: core.constexpr = x.numel >> n_dims
    shape: core.constexpr = [n_outer * 2**i, 2, 2 ** (n_dims - i - 1)]

    # tl.device_print("shape is: ", shape)
    y = core.reshape(x, shape)
    y_idx = core.reshape(ids, shape)

    # slice left/right with 'stride' 2**(n_dims - i - 1)
    mask = core.arange(0, 2)[None, :, None]

    y_left_val = core.where(mask == 0, y, 0.0)
    y_right_val = core.where(mask == 1, y, 0.0)
    left = core.broadcast_to(tl.sum(y_left_val, 1)[:, None, :], shape).to(x.dtype)
    right = core.broadcast_to(tl.sum(y_right_val, 1)[:, None, :], shape).to(x.dtype)
    left = core.reshape(left, x.shape)
    right = core.reshape(right, x.shape)

    idx_left_val = core.where(mask == 0, y_idx, 0)
    idx_right_val = core.where(mask == 1, y_idx, 0)
    left_idx = core.broadcast_to(tl.sum(idx_left_val, 1)[:, None, :], shape).to(
        ids.dtype
    )
    right_idx = core.broadcast_to(tl.sum(idx_right_val, 1)[:, None, :], shape).to(
        ids.dtype
    )
    left_idx = core.reshape(left_idx, ids.shape)
    right_idx = core.reshape(right_idx, ids.shape)

    # actual compare-and-swap
    if core.constexpr(x.dtype.primitive_bitwidth) == 8:
        idtype = core.int8
    elif core.constexpr(x.dtype.primitive_bitwidth) == 16:
        idtype = core.int16
    elif core.constexpr(x.dtype.primitive_bitwidth) == 32:
        idtype = core.int32
    elif core.constexpr(x.dtype.primitive_bitwidth) == 64:
        idtype = core.int64
    else:
        raise ValueError("Unsupported dtype")

    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)

    cond = (left > right) ^ flip
    ret = ix ^ core.where(cond, ileft ^ iright, zeros_like(ix))

    if core.constexpr(ids.dtype.primitive_bitwidth) == 8:
        idx_dtype = core.int8
    elif core.constexpr(ids.dtype.primitive_bitwidth) == 16:
        idx_dtype = core.int16
    elif core.constexpr(ids.dtype.primitive_bitwidth) == 32:
        idx_dtype = core.int32
    elif core.constexpr(ids.dtype.primitive_bitwidth) == 64:
        idx_dtype = core.int64
    else:
        raise ValueError("Unsupported dtype")

    ileft_idx = left_idx.to(idx_dtype, bitcast=True)
    iright_idx = right_idx.to(idx_dtype, bitcast=True)
    ix_idx = ids.to(idx_dtype, bitcast=True)
    ret_idx = ix_idx ^ core.where(cond, ileft_idx ^ iright_idx, zeros_like(ix_idx))

    return ret.to(x.dtype, bitcast=True), ret_idx.to(ids.dtype, bitcast=True)


@triton.jit
def _bitonic_merge(
    x, ids, stage: core.constexpr, order: core.constexpr, n_dims: core.constexpr
):
    """
    order_type 0 == ascending
    order_type 1 == descending
    order_type 2 == alternating
    """
    n_outer: core.constexpr = x.numel >> n_dims
    core.static_assert(stage <= n_dims)
    # flip denotes whether to re-arrange sub-sequences of elements in ascending or
    # descending order.
    # if flip = 00000000... then all elements will be re-arranged ascendingly at this stage
    # if flip = 00110011... then all the elements will be re-arranged alternatingly (with
    # a stride of 2) at this stage
    if order == 2:
        shape: core.constexpr = [n_outer * 2 ** (n_dims - 1 - stage), 2, 2**stage]
        flip = core.reshape(
            core.broadcast_to(core.arange(0, 2)[None, :, None], shape), x.shape
        )
    else:
        flip = order
    # perform `stage` rounds of `compare-and-swap`
    for i in core.static_range(stage):
        x, ids = _compare_and_swap(x, ids, flip, i + (n_dims - stage), n_dims)
    return x, ids


@triton.jit
def argsort(x, ids, dim: tl.constexpr, descending: core.constexpr):
    # handle default dimension or check that it is the most minor dim
    _dim: core.constexpr = dim
    n_dims: core.constexpr = _log2(x.shape[_dim])
    for i in core.static_range(1, n_dims + 1):
        x, ids = _bitonic_merge(x, ids, i, 2 if i < n_dims else descending, n_dims)
    return x, ids


@libentry()
@triton.jit
def topk_stage2_kernel(
    y_ptr,
    index_ptr,
    chunk_x,
    chunk_index,
    sort_dim: tl.constexpr,
    k: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DESCENDING: tl.constexpr,
):
    cur_batch = tle.program_id(0)
    chunk_x += cur_batch * N
    chunk_index += cur_batch * N
    y_ptr += cur_batch * k
    index_ptr += cur_batch * k
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    pad_val = float("-inf") if DESCENDING else float("inf")
    mask_index_val = _MIN_INT32_VAL if DESCENDING else _MAX_INT32_VAL

    chunk_x_val = tl.load(chunk_x + cols, mask=mask, other=pad_val).to(tl.float32)
    chunk_index_val = tl.load(chunk_index + cols, mask=mask, other=mask_index_val).to(
        tl.int32
    )

    # Use iterative selection instead of argsort to avoid NaN from -inf * 0
    # in the bitonic sort's compare-and-swap (y * mask where mask has 0).
    available = mask
    for k_idx in range(k):
        if DESCENDING:
            select_val = tl.max(chunk_x_val)
        else:
            select_val = tl.min(chunk_x_val)
        is_candidate = available & (chunk_x_val == select_val)
        candidate_indices = tl.where(is_candidate, cols, BLOCK_SIZE)
        select_idx = tl.argmin(candidate_indices, axis=0)

        tl.store(y_ptr + k_idx, select_val)
        select_global_idx = tl.sum(tl.where(cols == select_idx, chunk_index_val, 0))
        tl.store(index_ptr + k_idx, select_global_idx)
        if DESCENDING:
            chunk_x_val = tl.where(cols == select_idx, float("-inf"), chunk_x_val)
        else:
            chunk_x_val = tl.where(cols == select_idx, float("inf"), chunk_x_val)
        available = available & (cols != select_idx)


@triton.jit
def _fp8_bits_to_ordered_key(bits):
    sign = bits & 0x80
    flip_mask = tl.where(sign != 0, 0xFF, 0x80).to(tl.uint8)
    return bits ^ flip_mask


@libentry()
@triton.jit
def topk_fp8_row_radix_threshold_kernel(
    x_ptr,
    threshold_key_ptr,
    counter_ptr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)
    bins = tl.arange(0, 16)
    desired = tl.full((), 0, dtype=tl.uint8)
    desired_mask = tl.full((), 0, dtype=tl.uint8)
    k_to_find = tl.full((), K, dtype=tl.int32)
    n_tiles = tl.cdiv(N, BLOCK_N)

    for digit_iter in tl.static_range(0, 2):
        shift = 4 - digit_iter * 4
        counts = tl.zeros((16,), dtype=tl.int32)

        for tile in tl.range(0, n_tiles):
            offs = tile * BLOCK_N + tl.arange(0, BLOCK_N)
            mask = offs < N
            q = tl.load(x_ptr + pid * N + offs, mask=mask, other=0.0)
            bits = q.to(tl.uint8, bitcast=True)
            key = _fp8_bits_to_ordered_key(bits)
            matches = (key & desired_mask) == desired
            digit = ((key >> shift) & 0xF).to(tl.int32)
            valid = mask & matches
            counts += tl.sum(
                tl.where(
                    (digit[None, :] == bins[:, None]) & valid[None, :],
                    1,
                    0,
                ),
                axis=1,
            )

        cumsum_desc = tl.cumsum(counts, axis=0, reverse=True)
        selected = tl.full((), 0, dtype=tl.int32)
        counts_gt = tl.full((), 0, dtype=tl.int32)
        found = tl.full((), 0, dtype=tl.int32)

        for rev in tl.static_range(0, 16):
            digit_value = 15 - rev
            cum_d = tl.sum(tl.where(bins == digit_value, cumsum_desc, 0))
            if digit_value + 1 < 16:
                cum_next = tl.sum(tl.where(bins == digit_value + 1, cumsum_desc, 0))
            else:
                cum_next = tl.full((), 0, dtype=tl.int32)
            take = (found == 0) & (cum_d >= k_to_find) & (cum_next < k_to_find)
            selected = tl.where(take, digit_value, selected)
            counts_gt = tl.where(take, cum_next, counts_gt)
            found = tl.where(take, 1, found)

        selected_u8 = selected.to(tl.uint8)
        desired = desired | (selected_u8 << shift)
        desired_mask = desired_mask | (tl.full((), 0xF, dtype=tl.uint8) << shift)
        k_to_find = k_to_find - counts_gt

    tl.store(threshold_key_ptr + pid, desired)
    tl.store(counter_ptr + pid, 0)


@libentry()
@triton.jit
def topk_fp8_row_radix_high_hist_kernel(
    high_hist_ptr,
    x_ptr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    N_TILES: tl.constexpr,
):
    pid = tle.program_id(0)
    tile = tle.program_id(1)
    bins = tl.arange(0, 16)
    offs = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N

    q = tl.load(x_ptr + pid * N + offs, mask=mask, other=0.0)
    bits = q.to(tl.uint8, bitcast=True)
    key = _fp8_bits_to_ordered_key(bits)
    digit = ((key >> 4) & 0xF).to(tl.int32)
    counts = tl.sum(
        tl.where((digit[None, :] == bins[:, None]) & mask[None, :], 1, 0),
        axis=1,
    )
    tl.store(high_hist_ptr + (pid * N_TILES + tile) * 16 + bins, counts)


@libentry()
@triton.jit
def topk_fp8_row_radix_high_reduce_kernel(
    selected_high_ptr,
    k_remaining_ptr,
    high_hist_ptr,
    K: tl.constexpr,
    N_TILES: tl.constexpr,
):
    pid = tle.program_id(0)
    bins = tl.arange(0, 16)
    counts = tl.zeros((16,), dtype=tl.int32)

    for tile in tl.range(0, N_TILES):
        counts += tl.load(high_hist_ptr + (pid * N_TILES + tile) * 16 + bins)

    cumsum_desc = tl.cumsum(counts, axis=0, reverse=True)
    selected = tl.full((), 0, dtype=tl.int32)
    counts_gt = tl.full((), 0, dtype=tl.int32)
    found = tl.full((), 0, dtype=tl.int32)

    for rev in tl.static_range(0, 16):
        digit_value = 15 - rev
        cum_d = tl.sum(tl.where(bins == digit_value, cumsum_desc, 0))
        if digit_value + 1 < 16:
            cum_next = tl.sum(tl.where(bins == digit_value + 1, cumsum_desc, 0))
        else:
            cum_next = tl.full((), 0, dtype=tl.int32)
        take = (found == 0) & (cum_d >= K) & (cum_next < K)
        selected = tl.where(take, digit_value, selected)
        counts_gt = tl.where(take, cum_next, counts_gt)
        found = tl.where(take, 1, found)

    tl.store(selected_high_ptr + pid, selected.to(tl.uint8))
    tl.store(k_remaining_ptr + pid, K - counts_gt)


@libentry()
@triton.jit
def topk_fp8_row_radix_low_hist_kernel(
    low_hist_ptr,
    x_ptr,
    selected_high_ptr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    N_TILES: tl.constexpr,
):
    pid = tle.program_id(0)
    tile = tle.program_id(1)
    bins = tl.arange(0, 16)
    offs = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N

    selected_high = tl.load(selected_high_ptr + pid).to(tl.int32)
    q = tl.load(x_ptr + pid * N + offs, mask=mask, other=0.0)
    bits = q.to(tl.uint8, bitcast=True)
    key = _fp8_bits_to_ordered_key(bits)
    high = ((key >> 4) & 0xF).to(tl.int32)
    low = (key & 0xF).to(tl.int32)
    valid = mask & (high == selected_high)
    counts = tl.sum(
        tl.where((low[None, :] == bins[:, None]) & valid[None, :], 1, 0),
        axis=1,
    )
    tl.store(low_hist_ptr + (pid * N_TILES + tile) * 16 + bins, counts)


@libentry()
@triton.jit
def topk_fp8_row_radix_low_reduce_kernel(
    threshold_key_ptr,
    counter_ptr,
    low_hist_ptr,
    selected_high_ptr,
    k_remaining_ptr,
    N_TILES: tl.constexpr,
):
    pid = tle.program_id(0)
    bins = tl.arange(0, 16)
    counts = tl.zeros((16,), dtype=tl.int32)

    for tile in tl.range(0, N_TILES):
        counts += tl.load(low_hist_ptr + (pid * N_TILES + tile) * 16 + bins)

    k_to_find = tl.load(k_remaining_ptr + pid)
    cumsum_desc = tl.cumsum(counts, axis=0, reverse=True)
    selected_low = tl.full((), 0, dtype=tl.int32)
    found = tl.full((), 0, dtype=tl.int32)

    for rev in tl.static_range(0, 16):
        digit_value = 15 - rev
        cum_d = tl.sum(tl.where(bins == digit_value, cumsum_desc, 0))
        if digit_value + 1 < 16:
            cum_next = tl.sum(tl.where(bins == digit_value + 1, cumsum_desc, 0))
        else:
            cum_next = tl.full((), 0, dtype=tl.int32)
        take = (found == 0) & (cum_d >= k_to_find) & (cum_next < k_to_find)
        selected_low = tl.where(take, digit_value, selected_low)
        found = tl.where(take, 1, found)

    high = tl.load(selected_high_ptr + pid).to(tl.uint8)
    threshold = (high << 4) | selected_low.to(tl.uint8)
    tl.store(threshold_key_ptr + pid, threshold)
    tl.store(counter_ptr + pid, 0)


@libentry()
@triton.jit
def topk_fp8_row_radix_collect_kernel(
    candidate_val_ptr,
    candidate_idx_ptr,
    counter_ptr,
    x_ptr,
    threshold_key_ptr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TAKE_EQUAL: tl.constexpr,
):
    pid = tle.program_id(0)
    tile = tle.program_id(1)
    offs = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N

    q = tl.load(x_ptr + pid * N + offs, mask=mask, other=0.0)
    bits = q.to(tl.uint8, bitcast=True)
    key = _fp8_bits_to_ordered_key(bits)
    threshold = tl.load(threshold_key_ptr + pid)
    if TAKE_EQUAL:
        take = mask & (key == threshold)
    else:
        take = mask & (key > threshold)

    counter_offsets = tl.zeros((BLOCK_N,), dtype=tl.int64)
    old_pos = tl.atomic_add(
        counter_ptr + pid + counter_offsets, 1, sem="relaxed", mask=take
    )
    store_mask = take & (old_pos < K)
    tl.store(
        candidate_val_ptr + pid * K + old_pos,
        q.to(tl.float32),
        mask=store_mask,
    )
    tl.store(
        candidate_idx_ptr + pid * K + old_pos,
        offs.to(tl.int64),
        mask=store_mask,
    )


@libentry()
@triton.jit
def topk_fp8_row_radix_sort_kernel(
    y_ptr,
    index_ptr,
    candidate_val_ptr,
    candidate_idx_ptr,
    scale_ptr,
    K: tl.constexpr,
    K_PAD: tl.constexpr,
):
    pid = tle.program_id(0)
    offs = tl.arange(0, K_PAD)
    mask = offs < K
    vals = tl.load(
        candidate_val_ptr + pid * K + offs,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)
    idx = tl.load(
        candidate_idx_ptr + pid * K + offs,
        mask=mask,
        other=_MIN_INT64_VAL,
    ).to(tl.int64)
    sorted_vals, sorted_idx = argsort(vals, idx, dim=0, descending=True)
    scale = tl.load(scale_ptr + pid).to(tl.float32)
    tl.store(y_ptr + pid * K + offs, sorted_vals * scale, mask=mask)
    tl.store(index_ptr + pid * K + offs, sorted_idx, mask=mask)


if HAS_TLE:

    @triton.jit
    def _get_topmask_and_fullmask(x):
        tl.static_assert(
            x.dtype.is_int_unsigned(),
            "floating-point value must be passed as bits",
        )
        tm: tl.constexpr = 1 << (-1 + x.dtype.primitive_bitwidth)
        fm: tl.constexpr = (1 << x.dtype.primitive_bitwidth) - 1
        tm_arr = tl.full(x.shape, tm, dtype=x.dtype)
        fm_arr = tl.full(x.shape, fm, dtype=x.dtype)
        return tm_arr, fm_arr

    @triton.jit
    def _fpval_to_key_with_nan(x, x_bits):
        tm, fm = _get_topmask_and_fullmask(x_bits)
        mask = tl.where((x_bits & tm) != 0, fm, tm)
        key = x_bits ^ mask
        return tl.where(x == x, key, fm)

    @triton.jit
    def _key_to_fpval(x):
        tm, fm = _get_topmask_and_fullmask(x)
        mask = tl.where((x & tm) != 0, tm, fm)
        return x ^ mask

    @libentry()
    @triton.jit
    def topk_kernel_radix_tle(
        X,
        Yv,
        Yi,
        stride_xm,
        stride_ym,
        n_cols,
        K: tl.constexpr,
        K_PAD: tl.constexpr,
        BLOCK_N: tl.constexpr,
        RADIX_BITS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        x_dtype = X.dtype.element_ty
        x_nbits: tl.constexpr = x_dtype.primitive_bitwidth
        if x_nbits < 16:
            y_nbits: tl.constexpr = 32
        else:
            y_nbits: tl.constexpr = x_nbits * 2
        x_utype = tl.dtype(f"uint{x_nbits}")
        x_ultype = tl.dtype(f"uint{y_nbits}")

        RADIX_SIZE: tl.constexpr = 1 << RADIX_BITS
        RADIX_MASK: tl.constexpr = RADIX_SIZE - 1
        bins = tl.arange(0, RADIX_SIZE)
        one = tl.full([BLOCK_N], 1, tl.int32)

        desired = tl.full((), 0, dtype=x_utype)
        desired_mask = tl.full((), 0, dtype=x_utype)
        k_to_find = tl.full((), K, dtype=tl.int32)
        n_tiles = tl.cdiv(n_cols, BLOCK_N)

        smem_counts = tle_gpu.gpu.alloc(
            [RADIX_SIZE],
            dtype=tl.int32,
            layout=None,
            scope=tle_gpu.gpu.smem,
            nv_mma_shared_layout=False,
        )
        smem_count_ptrs = tle_gpu.gpu.local_ptr(smem_counts, (bins,))

        for digit_pos in tl.static_range(x_nbits - RADIX_BITS, -1, -RADIX_BITS):
            tl.store(smem_count_ptrs, tl.zeros([RADIX_SIZE], dtype=tl.int32))
            for t in tl.range(0, n_tiles):
                offs_n = t * BLOCK_N + tl.arange(0, BLOCK_N)
                mask_n = offs_n < n_cols
                x_ptrs = X + pid * stride_xm + offs_n
                x = tl.load(x_ptrs, mask=mask_n, other=float("-inf"))
                x_bits = x.to(x_utype, bitcast=True)
                x_key = _fpval_to_key_with_nan(x, x_bits)
                matches = (x_key & desired_mask) == desired
                digit = ((x_key >> digit_pos) & RADIX_MASK).to(tl.int32)
                valid = mask_n & matches
                count_addrs = tle_gpu.gpu.local_ptr(smem_counts, (digit,))
                tl.atomic_add(count_addrs, one, mask=valid, sem="relaxed", scope="cta")

            counts = tl.load(smem_count_ptrs)

            cumsum_desc = tl.cumsum(counts, axis=0, reverse=True)
            tl.store(smem_count_ptrs, cumsum_desc)

            selected_scalar = 0
            counts_gt_scalar = 0
            found = 0
            for rev in tl.static_range(RADIX_SIZE):
                d = RADIX_SIZE - 1 - rev
                cum_d = tl.load(tle_gpu.gpu.local_ptr(smem_counts, (d,)))
                if d + 1 < RADIX_SIZE:
                    cum_next = tl.load(tle_gpu.gpu.local_ptr(smem_counts, (d + 1,)))
                else:
                    cum_next = 0
                take = (found == 0) & (cum_d >= k_to_find) & (cum_next < k_to_find)
                selected_scalar = tl.where(take, d, selected_scalar)
                counts_gt_scalar = tl.where(take, cum_next, counts_gt_scalar)
                found = tl.where(take, 1, found)

            selected_u = selected_scalar.to(x_utype)
            desired = desired | (selected_u << digit_pos)
            desired_mask = desired_mask | (
                tl.full((), RADIX_MASK, dtype=x_utype) << digit_pos
            )
            k_to_find = k_to_find - counts_gt_scalar

        thr_key = desired

        min_val = tl.full((), float("-inf"), tl.float32).to(x_dtype)
        min_bits = min_val.to(x_utype, bitcast=True)
        min_key = _fpval_to_key_with_nan(min_val, min_bits)
        min_packed = min_key.to(x_ultype) << 16
        offs_k = tl.arange(0, K_PAD)

        smem_selected = tle_gpu.gpu.alloc(
            [K_PAD],
            dtype=x_ultype,
            layout=None,
            scope=tle_gpu.gpu.smem,
            nv_mma_shared_layout=False,
        )
        smem_selected_ptrs = tle_gpu.gpu.local_ptr(smem_selected, (offs_k,))
        tl.store(smem_selected_ptrs, tl.full([K_PAD], min_packed, dtype=x_ultype))

        smem_write_count = tle_gpu.gpu.alloc(
            [1],
            dtype=tl.int32,
            layout=None,
            scope=tle_gpu.gpu.smem,
            nv_mma_shared_layout=False,
        )
        tl.store(tle_gpu.gpu.local_ptr(smem_write_count, (0,)), 0)
        write_count_ptrs = tle_gpu.gpu.local_ptr(
            smem_write_count, (tl.zeros([BLOCK_N], dtype=tl.int32),)
        )

        for t in tl.range(0, n_tiles):
            offs_n = t * BLOCK_N + tl.arange(0, BLOCK_N)
            mask_n = offs_n < n_cols
            x_ptrs = X + pid * stride_xm + offs_n
            x = tl.load(x_ptrs, mask=mask_n, other=float("-inf"))
            x_bits = x.to(x_utype, bitcast=True)
            x_key = _fpval_to_key_with_nan(x, x_bits)
            idx_key = (n_cols - offs_n).to(x_ultype)
            packed = (x_key.to(x_ultype) << 16) | idx_key
            take_gt = mask_n & (x_key > thr_key)
            pos = tl.atomic_add(
                write_count_ptrs, one, mask=take_gt, sem="relaxed", scope="cta"
            )
            write_mask = take_gt & (pos < K_PAD)
            dst_ptrs = tle_gpu.gpu.local_ptr(smem_selected, (pos.to(tl.int32),))
            tl.store(dst_ptrs, packed, mask=write_mask)

        for t in tl.range(0, n_tiles):
            offs_n = t * BLOCK_N + tl.arange(0, BLOCK_N)
            mask_n = offs_n < n_cols
            x_ptrs = X + pid * stride_xm + offs_n
            x = tl.load(x_ptrs, mask=mask_n, other=float("-inf"))
            x_bits = x.to(x_utype, bitcast=True)
            x_key = _fpval_to_key_with_nan(x, x_bits)
            idx_key = (n_cols - offs_n).to(x_ultype)
            packed = (x_key.to(x_ultype) << 16) | idx_key
            take_eq = mask_n & (x_key == thr_key)
            pos = tl.atomic_add(
                write_count_ptrs, one, mask=take_eq, sem="relaxed", scope="cta"
            )
            write_mask = take_eq & (pos < K_PAD)
            dst_ptrs = tle_gpu.gpu.local_ptr(smem_selected, (pos.to(tl.int32),))
            tl.store(dst_ptrs, packed, mask=write_mask)

        selected_packed = tl.load(smem_selected_ptrs)

        topk = tl.sort(selected_packed, dim=0, descending=True)
        idx_mask = tl.full(topk.shape, (1 << 16) - 1, dtype=topk.dtype)
        idx_raw = (topk & idx_mask).to(tl.uint32)
        y_indices = (n_cols - idx_raw.to(tl.int32)).to(tl.int64)
        y_values_raw = (topk >> 16).to(x_utype)
        y_values = _key_to_fpval(y_values_raw).to(x_dtype, bitcast=True)

        mask_k = offs_k < K
        yv_ptrs = Yv + pid * stride_ym + offs_k
        yi_ptrs = Yi + pid * stride_ym + offs_k
        tl.store(yv_ptrs, y_values, mask=mask_k)
        tl.store(yi_ptrs, y_indices, mask=mask_k)

    @libentry()
    @triton.jit
    def topk_fp8_row_radix_tle_kernel(
        X,
        Scale,
        Yv,
        Yi,
        stride_xm,
        stride_sm,
        stride_ym,
        n_cols,
        K: tl.constexpr,
        K_PAD: tl.constexpr,
        BLOCK_N: tl.constexpr,
        RADIX_BITS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        x_dtype = X.dtype.element_ty
        x_utype = tl.uint8
        x_ultype = tl.uint32

        RADIX_SIZE: tl.constexpr = 1 << RADIX_BITS
        RADIX_MASK: tl.constexpr = RADIX_SIZE - 1
        bins = tl.arange(0, RADIX_SIZE)
        one = tl.full([BLOCK_N], 1, tl.int32)

        desired = tl.full((), 0, dtype=x_utype)
        desired_mask = tl.full((), 0, dtype=x_utype)
        k_to_find = tl.full((), K, dtype=tl.int32)
        n_tiles = tl.cdiv(n_cols, BLOCK_N)

        smem_counts = tle_gpu.gpu.alloc(
            [RADIX_SIZE],
            dtype=tl.int32,
            layout=None,
            scope=tle_gpu.gpu.smem,
            nv_mma_shared_layout=False,
        )
        smem_count_ptrs = tle_gpu.gpu.local_ptr(smem_counts, (bins,))

        for digit_pos in tl.static_range(8 - RADIX_BITS, -1, -RADIX_BITS):
            tl.store(smem_count_ptrs, tl.zeros([RADIX_SIZE], dtype=tl.int32))
            for t in tl.range(0, n_tiles):
                offs_n = t * BLOCK_N + tl.arange(0, BLOCK_N)
                mask_n = offs_n < n_cols
                q = tl.load(X + pid * stride_xm + offs_n, mask=mask_n, other=0.0)
                q_bits = q.to(x_utype, bitcast=True)
                q_key = _fp8_bits_to_ordered_key(q_bits)
                matches = (q_key & desired_mask) == desired
                digit = ((q_key >> digit_pos) & RADIX_MASK).to(tl.int32)
                valid = mask_n & matches
                count_addrs = tle_gpu.gpu.local_ptr(smem_counts, (digit,))
                tl.atomic_add(count_addrs, one, mask=valid, sem="relaxed", scope="cta")

            counts = tl.load(smem_count_ptrs)
            cumsum_desc = tl.cumsum(counts, axis=0, reverse=True)
            tl.store(smem_count_ptrs, cumsum_desc)

            selected_scalar = 0
            counts_gt_scalar = 0
            found = 0
            for rev in tl.static_range(RADIX_SIZE):
                d = RADIX_SIZE - 1 - rev
                cum_d = tl.load(tle_gpu.gpu.local_ptr(smem_counts, (d,)))
                if d + 1 < RADIX_SIZE:
                    cum_next = tl.load(tle_gpu.gpu.local_ptr(smem_counts, (d + 1,)))
                else:
                    cum_next = 0
                take = (found == 0) & (cum_d >= k_to_find) & (cum_next < k_to_find)
                selected_scalar = tl.where(take, d, selected_scalar)
                counts_gt_scalar = tl.where(take, cum_next, counts_gt_scalar)
                found = tl.where(take, 1, found)

            selected_u = selected_scalar.to(x_utype)
            desired = desired | (selected_u << digit_pos)
            desired_mask = desired_mask | (
                tl.full((), RADIX_MASK, dtype=x_utype) << digit_pos
            )
            k_to_find = k_to_find - counts_gt_scalar

        thr_key = desired
        min_key = tl.full((), 0, dtype=x_utype)
        min_packed = min_key.to(x_ultype) << 16
        offs_k = tl.arange(0, K_PAD)

        smem_selected = tle_gpu.gpu.alloc(
            [K_PAD],
            dtype=x_ultype,
            layout=None,
            scope=tle_gpu.gpu.smem,
            nv_mma_shared_layout=False,
        )
        smem_selected_ptrs = tle_gpu.gpu.local_ptr(smem_selected, (offs_k,))
        tl.store(smem_selected_ptrs, tl.full([K_PAD], min_packed, dtype=x_ultype))

        smem_write_count = tle_gpu.gpu.alloc(
            [1],
            dtype=tl.int32,
            layout=None,
            scope=tle_gpu.gpu.smem,
            nv_mma_shared_layout=False,
        )
        tl.store(tle_gpu.gpu.local_ptr(smem_write_count, (0,)), 0)
        write_count_ptrs = tle_gpu.gpu.local_ptr(
            smem_write_count, (tl.zeros([BLOCK_N], dtype=tl.int32),)
        )

        for t in tl.range(0, n_tiles):
            offs_n = t * BLOCK_N + tl.arange(0, BLOCK_N)
            mask_n = offs_n < n_cols
            q = tl.load(X + pid * stride_xm + offs_n, mask=mask_n, other=0.0)
            q_bits = q.to(x_utype, bitcast=True)
            q_key = _fp8_bits_to_ordered_key(q_bits)
            idx_key = (n_cols - offs_n).to(x_ultype)
            packed = (q_key.to(x_ultype) << 16) | idx_key
            take_gt = mask_n & (q_key > thr_key)
            pos = tl.atomic_add(
                write_count_ptrs, one, mask=take_gt, sem="relaxed", scope="cta"
            )
            write_mask = take_gt & (pos < K_PAD)
            dst_ptrs = tle_gpu.gpu.local_ptr(smem_selected, (pos.to(tl.int32),))
            tl.store(dst_ptrs, packed, mask=write_mask)

        for t in tl.range(0, n_tiles):
            offs_n = t * BLOCK_N + tl.arange(0, BLOCK_N)
            mask_n = offs_n < n_cols
            q = tl.load(X + pid * stride_xm + offs_n, mask=mask_n, other=0.0)
            q_bits = q.to(x_utype, bitcast=True)
            q_key = _fp8_bits_to_ordered_key(q_bits)
            idx_key = (n_cols - offs_n).to(x_ultype)
            packed = (q_key.to(x_ultype) << 16) | idx_key
            take_eq = mask_n & (q_key == thr_key)
            pos = tl.atomic_add(
                write_count_ptrs, one, mask=take_eq, sem="relaxed", scope="cta"
            )
            write_mask = take_eq & (pos < K_PAD)
            dst_ptrs = tle_gpu.gpu.local_ptr(smem_selected, (pos.to(tl.int32),))
            tl.store(dst_ptrs, packed, mask=write_mask)

        selected_packed = tl.load(smem_selected_ptrs)
        topk = tl.sort(selected_packed, dim=0, descending=True)
        idx_mask = tl.full(topk.shape, (1 << 16) - 1, dtype=topk.dtype)
        idx_raw = (topk & idx_mask).to(tl.uint32)
        y_indices = (n_cols - idx_raw.to(tl.int32)).to(tl.int64)
        y_keys = (topk >> 16).to(x_utype)
        y_bits = _key_to_fpval(y_keys)
        y_q = y_bits.to(x_dtype, bitcast=True).to(tl.float32)
        scale = tl.load(Scale + pid * stride_sm).to(tl.float32)
        y_values = y_q * scale

        mask_k = offs_k < K
        tl.store(Yv + pid * stride_ym + offs_k, y_values, mask=mask_k)
        tl.store(Yi + pid * stride_ym + offs_k, y_indices, mask=mask_k)


def topk(x, k, dim=-1, largest=True, sorted=True):
    logger.debug("GEMS TOPK")
    # If dim equals to last dim, we set it to -1.
    if dim < 0:
        dim = dim + x.ndim

    assert dim == x.ndim - 1, "Currently only support topk in last dimension"
    # assert sorted, "Currently only support sorted == True"

    # Early return for k=0 to avoid Triton kernel compilation error.
    # Triton's tl.arange(0, BLOCK_SIZE) requires BLOCK_SIZE > 0.
    # When k=0, stage2_elem_cnt becomes 0, leading to BLOCK_SIZE=0.
    if k == 0:
        out_shape = list(x.shape[:-1]) + [0]
        return (
            torch.empty(out_shape, device=x.device, dtype=x.dtype),
            torch.empty(out_shape, device=x.device, dtype=torch.int64),
        )

    descending = True
    if not largest:
        descending = False

    topk_elem_cnt = x.shape[dim]
    batch_size = math.prod(x.shape) // topk_elem_cnt

    if (
        HAS_TLE
        and sorted
        and descending
        and x.is_cuda
        and x.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and k >= 8
        and topk_elem_cnt > 128
        and topk_elem_cnt <= 65535
        and triton.next_power_of_2(k) <= 1024
    ):
        k_pad = triton.next_power_of_2(k)
        out_shape = x.shape[:-1] + (k,)
        y_vals = torch.empty(out_shape, device=x.device, dtype=x.dtype)
        y_idx = torch.empty(out_shape, device=x.device, dtype=torch.int64)
        block_n_radix = max(k_pad, min(512, triton.next_power_of_2(topk_elem_cnt)))
        block_n_radix = min(block_n_radix, 1024)

        x_2d = x.reshape(batch_size, topk_elem_cnt)
        y_vals_2d = y_vals.reshape(batch_size, k)
        y_idx_2d = y_idx.reshape(batch_size, k)
        with torch_device_fn.device(x.device):
            topk_kernel_radix_tle[(batch_size,)](
                x_2d,
                y_vals_2d,
                y_idx_2d,
                x_2d.stride(0),
                y_vals_2d.stride(0),
                topk_elem_cnt,
                K=k,
                K_PAD=k_pad,
                BLOCK_N=block_n_radix,
                RADIX_BITS=4,
                num_warps=4,
                num_stages=1,
            )
        return (y_vals, y_idx)

    if topk_elem_cnt <= 4096:
        out_shape = x.shape[:-1] + (k,)

        y_vals = torch.empty(out_shape, device=x.device, dtype=x.dtype)
        y_idx = torch.empty(out_shape, device=x.device, dtype=torch.int64)

        BLOCK_SIZE = triton.next_power_of_2(topk_elem_cnt)

        with torch_device_fn.device(x.device):
            topk_single_stage_kernel[(batch_size,)](
                y_vals,
                y_idx,
                x,
                k,
                topk_elem_cnt,
                BLOCK_SIZE,
                descending,
            )
        return (y_vals, y_idx)

    # Note(Zhengzekang): Maybe we should add a heuristic search in selecting a proper chunk size.
    if topk_elem_cnt < 1024:
        chunk_size = 256
    else:
        chunk_size = 1024

    # Note(Zhengzekang): We should promise chunk_size is larger than k.
    if chunk_size < k:
        chunk_size = triton.next_power_of_2(k)

    chunk_num = triton.cdiv(topk_elem_cnt, chunk_size)

    stage1_out = torch.empty(batch_size * chunk_num * k, device=x.device, dtype=x.dtype)
    stage1_out_idx = torch.empty(
        batch_size * chunk_num * k, device=x.device, dtype=torch.int64
    )

    out_shape = x.shape[:-1] + (k,)
    stage2_out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
    stage2_out_idx = torch.empty(out_shape, device=x.device, dtype=torch.int64)

    with torch_device_fn.device(x.device):
        topk_stage1_kernel[
            batch_size,
            chunk_num,
        ](
            stage1_out,  # pointer to the output
            stage1_out_idx,  # pointer to the output
            x,  # pointer to the input
            k,
            topk_elem_cnt,
            chunk_size,
            descending,
        )
    stage2_elem_cnt = chunk_num * k
    BLOCK_SIZE = triton.next_power_of_2(stage2_elem_cnt)

    with torch_device_fn.device(x.device):
        topk_stage2_kernel[batch_size,](
            stage2_out,
            stage2_out_idx,
            stage1_out,
            stage1_out_idx,
            dim,
            k,
            stage2_elem_cnt,
            BLOCK_SIZE,
            descending,
        )

    return (stage2_out, stage2_out_idx)


def topk_fp8_w8a16(
    x_fp8,
    x_scale,
    k,
    dim=-1,
    largest=True,
    sorted=True,
    group_size=128,
    out_dtype=torch.bfloat16,
):
    logger.debug("GEMS TOPK FP8 W8A16")
    if dim < 0:
        dim = dim + x_fp8.ndim

    assert dim == x_fp8.ndim - 1, "Currently only support topk in last dimension"
    assert sorted, "Currently only support sorted == True"

    if k == 0:
        out_shape = list(x_fp8.shape[:-1]) + [0]
        return (
            torch.empty(out_shape, device=x_fp8.device, dtype=out_dtype),
            torch.empty(out_shape, device=x_fp8.device, dtype=torch.int64),
        )

    topk_elem_cnt = x_fp8.shape[dim]
    batch_size = math.prod(x_fp8.shape) // topk_elem_cnt
    num_groups = triton.cdiv(topk_elem_cnt, group_size)
    expected_scale_shape = x_fp8.shape[:-1] + (num_groups,)
    assert (
        x_scale.shape == expected_scale_shape
    ), f"x_scale shape should be {expected_scale_shape}, got {x_scale.shape}"

    descending = True
    if not largest:
        descending = False

    out_shape = x_fp8.shape[:-1] + (k,)
    y_vals = torch.empty(out_shape, device=x_fp8.device, dtype=out_dtype)
    y_idx = torch.empty(out_shape, device=x_fp8.device, dtype=torch.int64)

    x_2d = x_fp8.reshape(batch_size, topk_elem_cnt)
    scale_2d = x_scale.reshape(batch_size, num_groups)
    y_vals_2d = y_vals.reshape(batch_size, k)
    y_idx_2d = y_idx.reshape(batch_size, k)

    if (
        HAS_TLE
        and num_groups == 1
        and descending
        and sorted
        and x_fp8.is_cuda
        and k >= 8
        and topk_elem_cnt > 128
        and topk_elem_cnt <= 65535
        and topk_elem_cnt >= 1024
        and triton.next_power_of_2(k) <= 1024
    ):
        k_pad = triton.next_power_of_2(k)
        if topk_elem_cnt >= 8192:
            block_n_radix = 1024
        elif topk_elem_cnt >= 4096:
            block_n_radix = 512
        else:
            block_n_radix = max(k_pad, min(512, triton.next_power_of_2(topk_elem_cnt)))
        with torch_device_fn.device(x_fp8.device):
            topk_fp8_row_radix_tle_kernel[(batch_size,)](
                x_2d,
                scale_2d,
                y_vals_2d,
                y_idx_2d,
                x_2d.stride(0),
                scale_2d.stride(0),
                y_vals_2d.stride(0),
                topk_elem_cnt,
                K=k,
                K_PAD=k_pad,
                BLOCK_N=block_n_radix,
                RADIX_BITS=4,
                num_warps=8,
                num_stages=1,
            )
        return (y_vals, y_idx)

    if num_groups == 1 and descending and topk_elem_cnt >= 4096:
        block_n = 512 if topk_elem_cnt <= 8192 else 1024
        n_tiles = triton.cdiv(topk_elem_cnt, block_n)
        threshold_keys = torch.empty(
            (batch_size,), device=x_fp8.device, dtype=torch.uint8
        )
        candidate_counter = torch.empty(
            (batch_size,), device=x_fp8.device, dtype=torch.int32
        )
        candidate_vals = torch.empty(
            (batch_size, k), device=x_fp8.device, dtype=torch.float32
        )
        candidate_idx = torch.empty(
            (batch_size, k), device=x_fp8.device, dtype=torch.int64
        )

        with torch_device_fn.device(x_fp8.device):
            if topk_elem_cnt >= 16384:
                high_hist = torch.empty(
                    (batch_size, n_tiles, 16),
                    device=x_fp8.device,
                    dtype=torch.int32,
                )
                low_hist = torch.empty(
                    (batch_size, n_tiles, 16),
                    device=x_fp8.device,
                    dtype=torch.int32,
                )
                selected_high = torch.empty(
                    (batch_size,), device=x_fp8.device, dtype=torch.uint8
                )
                k_remaining = torch.empty(
                    (batch_size,), device=x_fp8.device, dtype=torch.int32
                )
                topk_fp8_row_radix_high_hist_kernel[(batch_size, n_tiles)](
                    high_hist,
                    x_2d,
                    N=topk_elem_cnt,
                    BLOCK_N=block_n,
                    N_TILES=n_tiles,
                    num_warps=8,
                )
                topk_fp8_row_radix_high_reduce_kernel[(batch_size,)](
                    selected_high,
                    k_remaining,
                    high_hist,
                    K=k,
                    N_TILES=n_tiles,
                    num_warps=1,
                )
                topk_fp8_row_radix_low_hist_kernel[(batch_size, n_tiles)](
                    low_hist,
                    x_2d,
                    selected_high,
                    N=topk_elem_cnt,
                    BLOCK_N=block_n,
                    N_TILES=n_tiles,
                    num_warps=8,
                )
                topk_fp8_row_radix_low_reduce_kernel[(batch_size,)](
                    threshold_keys,
                    candidate_counter,
                    low_hist,
                    selected_high,
                    k_remaining,
                    N_TILES=n_tiles,
                    num_warps=1,
                )
            else:
                topk_fp8_row_radix_threshold_kernel[(batch_size,)](
                    x_2d,
                    threshold_keys,
                    candidate_counter,
                    K=k,
                    N=topk_elem_cnt,
                    BLOCK_N=block_n,
                    num_warps=8,
                )
            topk_fp8_row_radix_collect_kernel[(batch_size, n_tiles)](
                candidate_vals,
                candidate_idx,
                candidate_counter,
                x_2d,
                threshold_keys,
                K=k,
                N=topk_elem_cnt,
                BLOCK_N=block_n,
                TAKE_EQUAL=False,
                num_warps=8,
            )
            topk_fp8_row_radix_collect_kernel[(batch_size, n_tiles)](
                candidate_vals,
                candidate_idx,
                candidate_counter,
                x_2d,
                threshold_keys,
                K=k,
                N=topk_elem_cnt,
                BLOCK_N=block_n,
                TAKE_EQUAL=True,
                num_warps=8,
            )
            topk_fp8_row_radix_sort_kernel[(batch_size,)](
                y_vals_2d,
                y_idx_2d,
                candidate_vals,
                candidate_idx,
                x_scale.reshape(batch_size),
                K=k,
                K_PAD=triton.next_power_of_2(k),
                num_warps=8,
            )
        return (y_vals, y_idx)

    if topk_elem_cnt < 4096:
        block_size = triton.next_power_of_2(topk_elem_cnt)
        with torch_device_fn.device(x_fp8.device):
            topk_fp8_single_stage_kernel[(batch_size,)](
                y_vals_2d,
                y_idx_2d,
                x_2d,
                scale_2d,
                k,
                topk_elem_cnt,
                block_size,
                descending,
                group_size,
                num_groups,
            )
        return (y_vals, y_idx)

    if topk_elem_cnt < 1024:
        chunk_size = 256
    else:
        chunk_size = 1024

    if chunk_size < k:
        chunk_size = triton.next_power_of_2(k)

    chunk_num = triton.cdiv(topk_elem_cnt, chunk_size)

    stage1_out = torch.empty(
        batch_size * chunk_num * k, device=x_fp8.device, dtype=out_dtype
    )
    stage1_out_idx = torch.empty(
        batch_size * chunk_num * k, device=x_fp8.device, dtype=torch.int64
    )

    with torch_device_fn.device(x_fp8.device):
        topk_fp8_stage1_kernel[batch_size, chunk_num](
            stage1_out,
            stage1_out_idx,
            x_2d,
            scale_2d,
            k,
            topk_elem_cnt,
            chunk_size,
            descending,
            group_size,
            num_groups,
        )

    stage2_elem_cnt = chunk_num * k
    block_size = triton.next_power_of_2(stage2_elem_cnt)

    with torch_device_fn.device(x_fp8.device):
        topk_stage2_kernel[batch_size,](
            y_vals_2d,
            y_idx_2d,
            stage1_out,
            stage1_out_idx,
            dim,
            k,
            stage2_elem_cnt,
            block_size,
            descending,
        )

    return (y_vals, y_idx)
