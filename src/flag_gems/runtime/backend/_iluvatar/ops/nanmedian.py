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
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.nanmedian import (
    INT32_MAX,
    MAX_BLOCK_N,
    RADIX_SELECT_DTYPES,
    NanMedian,
    _check_supported_dtype,
    _count_block_n,
    _full_nan_result,
    _is_not_nan,
)
from flag_gems.ops.nanmedian import _nanmedian_dim_impl as _generic_nanmedian_dim_impl
from flag_gems.ops.nanmedian import _nanmedian_flat_impl as _generic_nanmedian_flat_impl
from flag_gems.ops.nanmedian import _normalize_dim, _to_order_key, count_valid_kernel
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_MAX_GENERIC_TOPK_N = 1 << 20
_NATIVE_KTHVALUE_MAX_N = 1 << 15
_WIDE_RADIX_MAX_N = 1 << 21
_FLAT_RADIX_BLOCK_N = 4096
_FLAT_LOCAL_RADIX_BITS = 4
_FLAT_LOCAL_NUM_WARPS = 16
_FLAT_RADIX_BITS_SMALL_DTYPE = 4
_FLAT_RADIX_BITS_LARGE_DTYPE = 2
_DIM_SMALL_SORT_MAX_N = 128
_DIM_KEY_SELECT_MAX_N = 4096
_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


@libentry()
@triton.jit
def _flat_local_prepare_kernel(
    inp,
    state,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < N
    vals = tl.load(inp + offsets, mask=mask, other=0.0)
    valid = mask & _is_not_nan(vals, True)
    valid_count = tl.sum(valid.to(tl.int32), axis=0).to(tl.int64)
    tl.store(state + 0, 0)
    tl.store(state + 1, 0)
    tl.store(state + 2, (valid_count + 1) // 2)


@libentry()
@triton.jit
def _flat_local_radix_step_kernel(
    inp,
    state,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DIGIT_POS: tl.constexpr,
    RADIX_BITS: tl.constexpr,
    RADIX_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < N
    vals = tl.load(inp + offsets, mask=mask, other=0.0)
    valid = mask & _is_not_nan(vals, True)
    dtype = inp.dtype.element_ty
    nbits: tl.constexpr = dtype.primitive_bitwidth
    utype = tl.dtype(f"uint{nbits}")
    radix_mask: tl.constexpr = (1 << RADIX_BITS) - 1
    radix_mask_val = tl.full((), radix_mask, dtype=utype)

    desired_state = tl.load(state + 0)
    desired_mask_state = tl.load(state + 1)
    desired = desired_state.to(utype)
    desired_mask = desired_mask_state.to(utype)
    keys = _to_order_key(vals, valid)
    active = valid & ((keys & desired_mask) == desired)
    digit = ((keys >> DIGIT_POS) & radix_mask_val).to(tl.int32)
    bins = tl.arange(0, RADIX_SIZE)
    counts = tl.zeros((RADIX_SIZE,), dtype=tl.int32)
    for radix_bin in tl.static_range(0, RADIX_SIZE):
        bin_count = tl.sum((active & (digit == radix_bin)).to(tl.int32), axis=0)
        counts += tl.where(bins == radix_bin, bin_count, 0)

    k_to_find = tl.load(state + 2)
    cumsum = tl.cumsum(counts, axis=0)
    prev = cumsum - counts
    take = (k_to_find <= cumsum) & (k_to_find > prev)
    selected_bin = tl.min(tl.where(take, bins, RADIX_SIZE - 1), axis=0).to(tl.int64)
    counts_before = tl.max(tl.where(take, prev, 0), axis=0).to(tl.int64)
    tl.store(state + 0, desired_state | (selected_bin << DIGIT_POS))
    tl.store(
        state + 1,
        desired_mask_state | (radix_mask << DIGIT_POS),
    )
    tl.store(state + 2, k_to_find - counts_before)


@libentry()
@triton.jit
def _flat_local_store_kernel(
    inp,
    out,
    state,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < N
    vals = tl.load(inp + offsets, mask=mask, other=0.0)
    valid = mask & _is_not_nan(vals, True)
    dtype = inp.dtype.element_ty
    nbits: tl.constexpr = dtype.primitive_bitwidth
    utype = tl.dtype(f"uint{nbits}")
    desired = tl.load(state + 0).to(utype)
    keys = _to_order_key(vals, valid)
    result_idx = tl.min(tl.where(valid & (keys == desired), offsets, N), axis=0)
    has_valid = tl.load(state + 2) > 0
    result = tl.load(
        inp + result_idx,
        mask=has_valid,
        other=float("nan"),
    )
    tl.store(out, result)


@libentry()
@triton.jit
def _flat_radix_init_kernel(
    valid_count,
    state,
    result_idx,
    bin_counts,
    N: tl.constexpr,
    IS_FLOAT: tl.constexpr,
    RADIX_SIZE: tl.constexpr,
):
    tl.store(valid_count, 0 if IS_FLOAT else N)
    tl.store(state + 0, 0)
    tl.store(state + 1, 0)
    tl.store(state + 2, 0)
    tl.store(result_idx, N)
    bins = tl.arange(0, RADIX_SIZE)
    tl.store(bin_counts + bins, 0)


@libentry()
@triton.jit
def _flat_radix_count_valid_kernel(
    inp,
    valid_count,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < N
    vals = tl.load(inp + offsets, mask=mask, other=0.0)
    valid = mask & _is_not_nan(vals, True)
    count = tl.sum(valid.to(tl.int32), axis=0)
    tl.atomic_add(valid_count, count, sem="relaxed")


@libentry()
@triton.jit
def _flat_radix_init_rank_kernel(valid_count, state):
    count = tl.load(valid_count).to(tl.int64)
    tl.store(state + 2, (count + 1) // 2)


@libentry()
@triton.jit
def _flat_radix_count_kernel(
    inp,
    bin_counts,
    state,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DIGIT_POS: tl.constexpr,
    RADIX_BITS: tl.constexpr,
    RADIX_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < N
    vals = tl.load(inp + offsets, mask=mask, other=0.0)
    dtype = inp.dtype.element_ty
    nbits: tl.constexpr = dtype.primitive_bitwidth
    utype = tl.dtype(f"uint{nbits}")
    radix_mask: tl.constexpr = (1 << RADIX_BITS) - 1
    radix_mask_val = tl.full((), radix_mask, dtype=utype)

    if dtype.is_floating():
        valid = mask & _is_not_nan(vals, True)
    else:
        valid = mask

    desired = tl.load(state + 0).to(utype)
    desired_mask = tl.load(state + 1).to(utype)
    keys = _to_order_key(vals, valid)
    active = valid & ((keys & desired_mask) == desired)
    digit = ((keys >> DIGIT_POS) & radix_mask_val).to(tl.int32)
    bins = tl.arange(0, RADIX_SIZE)
    counts = tl.zeros((RADIX_SIZE,), dtype=tl.int32)
    for radix_bin in tl.static_range(0, RADIX_SIZE):
        bin_count = tl.sum((active & (digit == radix_bin)).to(tl.int32), axis=0)
        counts += tl.where(bins == radix_bin, bin_count, 0)
    tl.atomic_add(bin_counts + bins, counts, sem="relaxed")


@libentry()
@triton.jit
def _flat_radix_update_kernel(
    bin_counts,
    state,
    DIGIT_POS: tl.constexpr,
    RADIX_BITS: tl.constexpr,
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
    radix_mask: tl.constexpr = (1 << RADIX_BITS) - 1
    desired = desired | (selected_bin << DIGIT_POS)
    desired_mask = desired_mask | (radix_mask << DIGIT_POS)
    tl.store(state + 0, desired)
    tl.store(state + 1, desired_mask)
    tl.store(state + 2, k_to_find - counts_before)
    tl.store(bin_counts + bins, 0)


@libentry()
@triton.jit
def _flat_radix_find_index_kernel(
    inp,
    state,
    valid_count,
    result_idx,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
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
            valid = mask & _is_not_nan(vals, True)
        else:
            valid = mask

        desired = tl.load(state + 0).to(utype)
        keys = _to_order_key(vals, valid)
        local_idx = tl.min(tl.where(valid & (keys == desired), offsets, N), axis=0)
        tl.atomic_min(result_idx, local_idx, sem="relaxed")


@libentry()
@triton.jit
def _flat_radix_store_result_kernel(inp, out, valid_count, result_idx):
    dtype = inp.dtype.element_ty
    idx = tl.load(result_idx)
    if dtype.is_floating():
        result = tl.load(inp + idx, mask=tl.load(valid_count) > 0, other=float("nan"))
    else:
        result = tl.load(inp + idx)
    tl.store(out, result)


@libentry()
@triton.jit
def _dim_small_sort_kernel(
    inp,
    out_values,
    out_indices,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    KEY_BITS: tl.constexpr,
):
    row = tle.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    values = tl.load(inp + row * N + cols, mask=mask, other=0.0)
    if inp.dtype.element_ty.is_floating():
        valid = mask & _is_not_nan(values, True)
        valid_count = tl.sum(valid.to(tl.int32), axis=0)
    else:
        valid = mask
        valid_count = N

    keys = _to_order_key(values, valid)
    if KEY_BITS <= 16:
        sortable_keys = keys.to(tl.int32)
    else:
        sortable_keys = keys.to(tl.int64)
    ordered_keys = tl.sort(sortable_keys, descending=False)
    rank = tl.where(valid_count > 0, (valid_count - 1) // 2, 0)
    selected_key = tl.sum(tl.where(cols == rank, ordered_keys, 0), axis=0)
    index = tl.min(tl.where(valid & (sortable_keys == selected_key), cols, N), axis=0)
    result = tl.load(
        inp + row * N + index,
        mask=valid_count > 0,
        other=float("nan"),
    )
    index = tl.where(valid_count > 0, index, 0)
    tl.store(out_values + row, result)
    tl.store(out_indices + row, index)


@libentry()
@triton.jit
def _dim_key_select_kernel(
    inp,
    out_values,
    out_indices,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    KEY_BITS: tl.constexpr,
):
    row = tle.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    values = tl.load(inp + row * N + cols, mask=mask, other=0.0)
    if inp.dtype.element_ty.is_floating():
        valid = mask & _is_not_nan(values, True)
    else:
        valid = mask

    valid_count = tl.sum(valid.to(tl.int32), axis=0)
    keys = _to_order_key(values, valid)
    if KEY_BITS <= 16:
        search_keys = keys.to(tl.uint32)
        key_max = tl.full((), (1 << KEY_BITS) - 1, dtype=tl.uint32)
    else:
        search_keys = keys.to(tl.uint64)
        key_max = tl.full((), (1 << KEY_BITS) - 1, dtype=tl.uint64)

    lo = tl.min(tl.where(valid, search_keys, key_max), axis=0)
    hi = tl.max(tl.where(valid, search_keys, 0), axis=0)
    has_valid = valid_count > 0
    lo = tl.where(has_valid, lo, 0)
    hi = tl.where(has_valid, hi, 0)
    rank = tl.where(has_valid, (valid_count - 1) // 2, 0)
    for _ in tl.static_range(0, KEY_BITS):
        mid = lo + ((hi - lo) >> 1)
        le_count = tl.sum((valid & (search_keys <= mid)).to(tl.int32), axis=0)
        take_left = le_count > rank
        hi = tl.where(take_left, mid, hi)
        lo = tl.where(take_left, lo, mid + 1)

    index = tl.min(tl.where(valid & (search_keys == lo), cols, N), axis=0)
    result = tl.load(
        inp + row * N + index,
        mask=has_valid,
        other=float("nan"),
    )
    index = tl.where(has_valid, index, 0)
    tl.store(out_values + row, result)
    tl.store(out_indices + row, index)


@libentry()
@triton.jit
def _dim_radix_init_kernel(
    valid_counts,
    states,
    result_indices,
    bin_counts,
    N: tl.constexpr,
    IS_FLOAT: tl.constexpr,
    RADIX_SIZE: tl.constexpr,
):
    row = tle.program_id(0)
    bins = tl.arange(0, RADIX_SIZE)
    tl.store(valid_counts + row, 0 if IS_FLOAT else N)
    tl.store(states + row * 3, 0)
    tl.store(states + row * 3 + 1, 0)
    tl.store(states + row * 3 + 2, 0)
    tl.store(result_indices + row, N)
    tl.store(bin_counts + row * RADIX_SIZE + bins, 0)


@libentry()
@triton.jit
def _dim_radix_count_valid_kernel(
    inp,
    valid_counts,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tle.program_id(0).to(tl.int64)
    chunk = tle.program_id(1).to(tl.int64)
    cols = chunk * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = cols < N
    values = tl.load(inp + row * N + cols, mask=mask, other=0.0)
    valid = mask & _is_not_nan(values, True)
    count = tl.sum(valid.to(tl.int32), axis=0)
    tl.atomic_add(valid_counts + row, count, sem="relaxed")


@libentry()
@triton.jit
def _dim_radix_init_rank_kernel(valid_counts, states):
    row = tle.program_id(0)
    valid_count = tl.load(valid_counts + row).to(tl.int64)
    tl.store(states + row * 3 + 2, (valid_count + 1) // 2)


@libentry()
@triton.jit
def _dim_radix_count_kernel(
    inp,
    bin_counts,
    states,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DIGIT_POS: tl.constexpr,
    RADIX_BITS: tl.constexpr,
    RADIX_SIZE: tl.constexpr,
):
    row = tle.program_id(0).to(tl.int64)
    chunk = tle.program_id(1).to(tl.int64)
    cols = chunk * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = cols < N
    values = tl.load(inp + row * N + cols, mask=mask, other=0.0)
    dtype = inp.dtype.element_ty
    nbits: tl.constexpr = dtype.primitive_bitwidth
    utype = tl.dtype(f"uint{nbits}")
    radix_mask: tl.constexpr = (1 << RADIX_BITS) - 1
    radix_mask_val = tl.full((), radix_mask, dtype=utype)
    if dtype.is_floating():
        valid = mask & _is_not_nan(values, True)
    else:
        valid = mask

    state = states + row * 3
    desired = tl.load(state).to(utype)
    desired_mask = tl.load(state + 1).to(utype)
    keys = _to_order_key(values, valid)
    active = valid & ((keys & desired_mask) == desired)
    digit = ((keys >> DIGIT_POS) & radix_mask_val).to(tl.int32)
    bins = tl.arange(0, RADIX_SIZE)
    counts = tl.zeros((RADIX_SIZE,), dtype=tl.int32)
    for radix_bin in tl.static_range(0, RADIX_SIZE):
        count = tl.sum((active & (digit == radix_bin)).to(tl.int32), axis=0)
        counts += tl.where(bins == radix_bin, count, 0)
    tl.atomic_add(
        bin_counts + row * RADIX_SIZE + bins,
        counts,
        sem="relaxed",
    )


@libentry()
@triton.jit
def _dim_radix_update_kernel(
    bin_counts,
    states,
    DIGIT_POS: tl.constexpr,
    RADIX_BITS: tl.constexpr,
    RADIX_SIZE: tl.constexpr,
):
    row = tle.program_id(0)
    bins = tl.arange(0, RADIX_SIZE)
    counts_ptr = bin_counts + row * RADIX_SIZE + bins
    counts = tl.load(counts_ptr)
    state = states + row * 3
    k_to_find = tl.load(state + 2)
    cumsum = tl.cumsum(counts, axis=0)
    previous = cumsum - counts
    take = (k_to_find <= cumsum) & (k_to_find > previous)
    selected_bin = tl.min(tl.where(take, bins, RADIX_SIZE - 1), axis=0).to(tl.int64)
    counts_before = tl.max(tl.where(take, previous, 0), axis=0).to(tl.int64)
    desired = tl.load(state)
    desired_mask = tl.load(state + 1)
    radix_mask: tl.constexpr = (1 << RADIX_BITS) - 1
    tl.store(state, desired | (selected_bin << DIGIT_POS))
    tl.store(state + 1, desired_mask | (radix_mask << DIGIT_POS))
    tl.store(state + 2, k_to_find - counts_before)
    tl.store(counts_ptr, 0)


@libentry()
@triton.jit
def _dim_radix_find_index_kernel(
    inp,
    states,
    valid_counts,
    result_indices,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tle.program_id(0).to(tl.int64)
    chunk = tle.program_id(1).to(tl.int64)
    if tl.load(valid_counts + row) > 0:
        cols = chunk * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = cols < N
        values = tl.load(inp + row * N + cols, mask=mask, other=0.0)
        dtype = inp.dtype.element_ty
        nbits: tl.constexpr = dtype.primitive_bitwidth
        utype = tl.dtype(f"uint{nbits}")
        if dtype.is_floating():
            valid = mask & _is_not_nan(values, True)
        else:
            valid = mask
        desired = tl.load(states + row * 3).to(utype)
        keys = _to_order_key(values, valid)
        index = tl.min(tl.where(valid & (keys == desired), cols, N), axis=0)
        tl.atomic_min(result_indices + row, index, sem="relaxed")


@libentry()
@triton.jit
def _dim_radix_store_result_kernel(
    inp,
    out_values,
    out_indices,
    valid_counts,
    result_indices,
    N: tl.constexpr,
):
    row = tle.program_id(0).to(tl.int64)
    valid_count = tl.load(valid_counts + row)
    index = tl.load(result_indices + row)
    if inp.dtype.element_ty.is_floating():
        value = tl.load(
            inp + row * N + index,
            mask=valid_count > 0,
            other=float("nan"),
        )
        index = tl.where(valid_count > 0, index, 0)
    else:
        value = tl.load(inp + row * N + index)
    tl.store(out_values + row, value)
    tl.store(out_indices + row, index)


def _native_kthvalue(inp, k, dim=-1, keepdim=False):
    return torch.ops.aten.kthvalue.default.redispatch(
        _FALLBACK_KEYSET, inp, k, dim, keepdim
    )


def _nanmedian_native_kthvalue_flat(inp, out=None):
    flat = inp.reshape(-1).contiguous()
    n = flat.numel()
    values, _ = _native_kthvalue(flat, (n + 1) // 2)
    if out is None:
        return values
    out.copy_(values)
    return out


def _nanmedian_kthvalue_fallback(inp, M, N):
    inp = inp.reshape(M, N)
    if not torch.is_floating_point(inp):
        values, indices = _native_kthvalue(inp, (N + 1) // 2, dim=1)
        return NanMedian(values=values, indices=indices)

    valid_count = torch.empty((M,), dtype=torch.long, device=inp.device)
    block_n = _count_block_n(inp, N)
    with torch_device_fn.device(inp.device):
        count_valid_kernel[(M,)](inp, valid_count, M, N, block_n, inp.is_cuda)

    kth_inp = torch.where(
        torch.isnan(inp),
        torch.tensor(float("inf"), dtype=inp.dtype, device=inp.device),
        inp,
    )
    min_count = int(torch.min(valid_count).item())
    max_count = int(torch.max(valid_count).item())
    if min_count == max_count:
        if max_count == 0:
            return _full_nan_result((M,), inp.dtype, inp.device)
        values, indices = _native_kthvalue(kth_inp, (max_count + 1) // 2, dim=1)
        return NanMedian(values=values, indices=indices)

    if max_count - min_count <= 1:
        min_k = (min_count + 1) // 2 if min_count > 0 else 0
        max_k = (max_count + 1) // 2
        if min_k == max_k:
            values, indices = _native_kthvalue(kth_inp, max_k, dim=1)
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
            values, indices = _native_kthvalue(kth_inp, min_k, dim=1)
            mask = valid_count == min_count
            result = NanMedian(
                values=torch.where(mask, values, result.values),
                indices=torch.where(mask, indices, result.indices),
            )

        values, indices = _native_kthvalue(kth_inp, max_k, dim=1)
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
        rows = torch.index_select(kth_inp, 0, row_indices)
        values, indices = _native_kthvalue(rows, (count + 1) // 2, dim=1)
        result.values[row_indices] = values
        result.indices[row_indices] = indices
    return result


def _uses_native_kthvalue(inp, dim):
    if inp.ndim == 0:
        return False
    dim = _normalize_dim(dim, inp.ndim)
    shape = list(inp.shape)
    reduction_size = shape[dim]
    output_size = math.prod(shape[:dim] + shape[dim + 1 :])
    return reduction_size > _MAX_GENERIC_TOPK_N and output_size > 0


def _nanmedian_dim_radix(rows, values, indices):
    M, N = rows.shape
    nbits = rows.element_size() * 8
    if N <= _DIM_KEY_SELECT_MAX_N:
        block_n = triton.next_power_of_2(N)
        num_warps = 2 if block_n <= 1024 else 8
        with torch_device_fn.device(rows.device):
            _dim_key_select_kernel[(M,)](
                rows,
                values,
                indices,
                N,
                block_n,
                nbits,
                num_warps=num_warps,
                num_stages=1,
            )
        return

    valid_counts = torch.empty((M,), dtype=torch.int32, device=rows.device)
    states = torch.empty((M, 3), dtype=torch.int64, device=rows.device)
    result_indices = torch.empty((M,), dtype=torch.int32, device=rows.device)
    use_wide_radix = rows.element_size() <= 2 or N <= _WIDE_RADIX_MAX_N
    radix_bits = (
        _FLAT_RADIX_BITS_SMALL_DTYPE if use_wide_radix else _FLAT_RADIX_BITS_LARGE_DTYPE
    )
    radix_size = 1 << radix_bits
    bin_counts = torch.empty((M, radix_size), dtype=torch.int32, device=rows.device)
    block_n = min(triton.next_power_of_2(N), _FLAT_RADIX_BLOCK_N)
    num_chunks = triton.cdiv(N, block_n)
    grid = (M, num_chunks)
    num_warps = 8
    is_float = rows.dtype.is_floating_point

    with torch_device_fn.device(rows.device):
        _dim_radix_init_kernel[(M,)](
            valid_counts,
            states,
            result_indices,
            bin_counts,
            N,
            is_float,
            radix_size,
            num_warps=4,
            num_stages=1,
        )
        if is_float:
            _dim_radix_count_valid_kernel[grid](
                rows,
                valid_counts,
                N,
                block_n,
                num_warps=num_warps,
                num_stages=1,
            )
        _dim_radix_init_rank_kernel[(M,)](
            valid_counts,
            states,
            num_warps=4,
            num_stages=1,
        )
        for digit_pos in range(nbits - radix_bits, -1, -radix_bits):
            _dim_radix_count_kernel[grid](
                rows,
                bin_counts,
                states,
                N,
                block_n,
                digit_pos,
                radix_bits,
                radix_size,
                num_warps=num_warps,
                num_stages=1,
            )
            _dim_radix_update_kernel[(M,)](
                bin_counts,
                states,
                digit_pos,
                radix_bits,
                radix_size,
                num_warps=4,
                num_stages=1,
            )
        _dim_radix_find_index_kernel[grid](
            rows,
            states,
            valid_counts,
            result_indices,
            N,
            block_n,
            num_warps=num_warps,
            num_stages=1,
        )
        _dim_radix_store_result_kernel[(M,)](
            rows,
            values,
            indices,
            valid_counts,
            result_indices,
            N,
            num_warps=4,
            num_stages=1,
        )


def _nanmedian_dim_impl(inp, dim, keepdim, out=None):
    if inp.ndim == 0:
        return _generic_nanmedian_dim_impl(inp, dim, keepdim, out=out)

    dim = _normalize_dim(dim, inp.ndim)
    shape = list(inp.shape)
    N = shape[dim]
    out_shape = shape[:dim] + shape[dim + 1 :]
    M = math.prod(out_shape)
    use_radix = 0 < N <= INT32_MAX and M > 0 and inp.dtype in RADIX_SELECT_DTYPES
    if not use_radix:
        if not _uses_native_kthvalue(inp, dim):
            return _generic_nanmedian_dim_impl(inp, dim, keepdim, out=out)

        keepdim_shape = shape.copy()
        keepdim_shape[dim] = 1
        output_shape = keepdim_shape if keepdim else out_shape
        compute_shape = output_shape if out is not None else keepdim_shape
        result = _nanmedian_kthvalue_fallback(dim_compress(inp, dim), M, N)
        computed_values = result.values.reshape(compute_shape)
        computed_indices = result.indices.reshape(compute_shape)
        if out is None:
            values = computed_values
            indices = computed_indices
        else:
            values, indices = out
            values.copy_(computed_values)
            indices.copy_(computed_indices)
        if out is None and not keepdim:
            values = torch.squeeze(values, dim)
            indices = torch.squeeze(indices, dim)
        return NanMedian(values=values, indices=indices)

    keepdim_shape = shape.copy()
    keepdim_shape[dim] = 1
    output_shape = keepdim_shape if keepdim else out_shape
    compute_shape = output_shape if out is not None else keepdim_shape
    if out is None:
        values = torch.empty(compute_shape, dtype=inp.dtype, device=inp.device)
        indices = torch.empty(compute_shape, dtype=torch.long, device=inp.device)
    else:
        values, indices = out

    rows = dim_compress(inp, dim).reshape(M, N)
    values_contiguous = values.is_contiguous()
    indices_contiguous = indices.is_contiguous()
    flat_values = values.reshape(M) if values_contiguous else values.new_empty((M,))
    flat_indices = indices.reshape(M) if indices_contiguous else indices.new_empty((M,))
    if N <= _DIM_SMALL_SORT_MAX_N:
        block_n = triton.next_power_of_2(N)
        with torch_device_fn.device(inp.device):
            _dim_small_sort_kernel[(M,)](
                rows,
                flat_values,
                flat_indices,
                N,
                block_n,
                inp.element_size() * 8,
                num_warps=4 if block_n <= 64 else 8,
                num_stages=1,
            )
    else:
        _nanmedian_dim_radix(rows, flat_values, flat_indices)

    if not values_contiguous:
        values.copy_(flat_values.reshape(values.shape))
    if not indices_contiguous:
        indices.copy_(flat_indices.reshape(indices.shape))

    if out is None and not keepdim:
        values = torch.squeeze(values, dim)
        indices = torch.squeeze(indices, dim)
    return NanMedian(values=values, indices=indices)


def _nanmedian_flat_impl(inp, out=None):
    n = inp.numel()
    if (
        MAX_BLOCK_N < n <= _FLAT_RADIX_BLOCK_N
        and torch.is_floating_point(inp)
        and inp.dtype in RADIX_SELECT_DTYPES
    ):
        flat = inp.reshape(-1).contiguous()
        if out is None:
            out = torch.empty((), dtype=flat.dtype, device=flat.device)
        state = torch.empty((3,), dtype=torch.int64, device=flat.device)
        block_n = triton.next_power_of_2(n)
        nbits = flat.element_size() * 8
        with torch_device_fn.device(flat.device):
            _flat_local_prepare_kernel[(1,)](
                flat,
                state,
                n,
                block_n,
                num_warps=_FLAT_LOCAL_NUM_WARPS,
                num_stages=1,
            )
            for digit_pos in range(
                nbits - _FLAT_LOCAL_RADIX_BITS,
                -1,
                -_FLAT_LOCAL_RADIX_BITS,
            ):
                _flat_local_radix_step_kernel[(1,)](
                    flat,
                    state,
                    n,
                    block_n,
                    digit_pos,
                    _FLAT_LOCAL_RADIX_BITS,
                    1 << _FLAT_LOCAL_RADIX_BITS,
                    num_warps=_FLAT_LOCAL_NUM_WARPS,
                    num_stages=1,
                )
            _flat_local_store_kernel[(1,)](
                flat,
                out,
                state,
                n,
                block_n,
                num_warps=_FLAT_LOCAL_NUM_WARPS,
                num_stages=1,
            )
        return out

    if (
        MAX_BLOCK_N < n <= _NATIVE_KTHVALUE_MAX_N
        and not torch.is_floating_point(inp)
        and inp.dtype in RADIX_SELECT_DTYPES
    ):
        return _nanmedian_native_kthvalue_flat(inp, out=out)

    if MAX_BLOCK_N < n <= INT32_MAX and inp.dtype in RADIX_SELECT_DTYPES:
        flat = inp.reshape(-1).contiguous()
        if out is None:
            out = torch.empty((), dtype=flat.dtype, device=flat.device)
        valid_count = torch.empty((), dtype=torch.int32, device=flat.device)
        state = torch.empty((3,), dtype=torch.int64, device=flat.device)
        result_idx = torch.empty((), dtype=torch.int32, device=flat.device)
        use_wide_radix = flat.element_size() <= 2 or n <= _WIDE_RADIX_MAX_N
        radix_bits = (
            _FLAT_RADIX_BITS_SMALL_DTYPE
            if use_wide_radix
            else _FLAT_RADIX_BITS_LARGE_DTYPE
        )
        radix_size = 1 << radix_bits
        bin_counts = torch.empty((radix_size,), dtype=torch.int32, device=flat.device)
        block_n = min(triton.next_power_of_2(n), _FLAT_RADIX_BLOCK_N)
        grid = (triton.cdiv(n, block_n),)
        nbits = flat.element_size() * 8

        with torch_device_fn.device(flat.device):
            _flat_radix_init_kernel[(1,)](
                valid_count,
                state,
                result_idx,
                bin_counts,
                n,
                torch.is_floating_point(flat),
                radix_size,
            )
            if torch.is_floating_point(flat):
                _flat_radix_count_valid_kernel[grid](
                    flat,
                    valid_count,
                    n,
                    block_n,
                    num_warps=8,
                    num_stages=1,
                )
            _flat_radix_init_rank_kernel[(1,)](valid_count, state)
            for digit_pos in range(nbits - radix_bits, -1, -radix_bits):
                _flat_radix_count_kernel[grid](
                    flat,
                    bin_counts,
                    state,
                    n,
                    block_n,
                    digit_pos,
                    radix_bits,
                    radix_size,
                    num_warps=8,
                    num_stages=1,
                )
                _flat_radix_update_kernel[(1,)](
                    bin_counts,
                    state,
                    digit_pos,
                    radix_bits,
                    radix_size,
                    num_warps=8,
                    num_stages=1,
                )
            _flat_radix_find_index_kernel[grid](
                flat,
                state,
                valid_count,
                result_idx,
                n,
                block_n,
                num_warps=8,
                num_stages=1,
            )
            _flat_radix_store_result_kernel[(1,)](flat, out, valid_count, result_idx)
        return out

    if n <= _MAX_GENERIC_TOPK_N:
        return _generic_nanmedian_flat_impl(inp, out=out)

    flat = inp.reshape(-1)
    if out is None:
        return _nanmedian_dim_impl(flat, 0, False).values

    indices = torch.empty((), dtype=torch.long, device=inp.device)
    _nanmedian_dim_impl(flat, 0, False, out=(out, indices))
    return out


def nanmedian(inp):
    logger.debug("GEMS_ILUVATAR NANMEDIAN")
    _check_supported_dtype(inp)
    return _nanmedian_flat_impl(inp)


def nanmedian_out(inp, *, out):
    logger.debug("GEMS_ILUVATAR NANMEDIAN OUT")
    _check_supported_dtype(inp)
    return _nanmedian_flat_impl(inp, out=out)


def nanmedian_dim(inp, dim=-1, keepdim=False):
    logger.debug("GEMS_ILUVATAR NANMEDIAN DIM")
    _check_supported_dtype(inp)
    return _nanmedian_dim_impl(inp, dim, keepdim)


def nanmedian_dim_values(inp, dim=-1, keepdim=False, *, values, indices):
    logger.debug("GEMS_ILUVATAR NANMEDIAN DIM VALUES")
    return _nanmedian_dim_impl(
        inp,
        dim,
        keepdim,
        out=(values, indices),
    )
