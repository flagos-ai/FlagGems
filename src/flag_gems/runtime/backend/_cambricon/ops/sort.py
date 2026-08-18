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

from flag_gems.runtime import torch_device_fn

from ..utils import MAX_GRID_SIZE_X

logger = logging.getLogger(__name__)


def unwrap_if_constexpr(o):
    return o.value if isinstance(o, tl.constexpr) else o


@tl.constexpr
def get_int_t(num_bits: tl.constexpr, signed: tl.constexpr) -> tl.dtype:
    num_bits = unwrap_if_constexpr(num_bits)
    signed = unwrap_if_constexpr(signed)
    return tl.core.get_int_dtype(num_bits, signed)


@tl.constexpr
def one_zeros(num_bits: tl.constexpr) -> int:
    num_bits = unwrap_if_constexpr(num_bits)
    return 1 << (num_bits - 1)


@tl.constexpr
def zero_ones(num_bits: tl.constexpr) -> int:
    num_bits = unwrap_if_constexpr(num_bits)
    return (1 << (num_bits - 1)) - 1


@triton.jit
def uint_to_uint(x, descending: tl.constexpr = False):
    out = ~x if descending else x
    return out


@triton.jit
def int_to_uint(x, descending: tl.constexpr = False):
    num_bits: tl.constexpr = x.dtype.primitive_bitwidth
    udtype = get_int_t(num_bits, False)
    ux = tl.cast(x, udtype, bitcast=True)
    if descending:
        # 0111111....1
        bit_mask: tl.constexpr = zero_ones(num_bits)
        bit_mask_tensor = tl.full((), value=bit_mask, dtype=udtype)
        out = ux ^ bit_mask_tensor
    else:
        # 1000000...0
        sign_bit_mask: tl.constexpr = one_zeros(num_bits)
        sign_bit_mask_tensor = tl.full((), value=sign_bit_mask, dtype=udtype)
        out = ux ^ sign_bit_mask_tensor
    return out


@triton.jit
def floating_to_uint(x, descending: tl.constexpr = False):
    num_bits: tl.constexpr = x.dtype.primitive_bitwidth
    sdtype = get_int_t(num_bits, True)
    udtype = get_int_t(num_bits, False)
    sx = x.to(sdtype, bitcast=True)
    ux = x.to(udtype, bitcast=True)

    sign_bit_mask_v: tl.constexpr = one_zeros(num_bits)
    sign_bit_mask = tl.full((), value=sign_bit_mask_v, dtype=udtype)
    # mind the dtype, right_shift for signed is arithmetic right shift
    # Fix for triton 3.1 or else `sx >> rshift_bits` is promoted to int32
    rshift_bits = tl.full((), value=num_bits - 1, dtype=sdtype)
    mask = sign_bit_mask | (sx >> rshift_bits).to(udtype, bitcast=True)
    tl.static_assert(mask.dtype == udtype, "type mismatch")
    # 1000000000...0 for positive
    # 1111111111...1 for negative
    if descending:
        out = ux ^ (~mask)
    else:
        out = ux ^ mask
    return out.to(udtype, bitcast=True)


@triton.jit
def convert_to_uint_preverse_order(x: tl.tensor, descending: tl.constexpr = False):
    if x.dtype.is_floating():
        out = floating_to_uint(x, descending)
    elif x.dtype.is_int_signed():
        out = int_to_uint(x, descending)
    elif x.dtype.is_int_unsigned():
        out = uint_to_uint(x, descending)
    return out


# Two-pass lock-free radix scan (adapted from the kunlunxin backend's
# radix_sort_low_mem). The previous single-pass `sweep` used a decoupled-lookback
# scan whose inter-block spin-wait (`while pack1 == 0`) plus an m-serial
# M_PER_SPLIT loop collapsed parallelism when m >> n (the dim=0 case, where the
# sort dim becomes the tiny batch axis). Here the prefix sum is computed on the
# host with torch.cumsum, so there is no inter-block dependency, and both kernels
# use a grid-stride loop over (row, block) tasks to saturate the device without
# exceeding MAX_GRID_SIZE_X.
@triton.jit
def count_kernel(
    x_ptr,
    counts_ptr,  # (M, grid_n, num_bins): per-(row, block) per-bin local counts
    total_tasks,
    M,
    N,
    bit_offset,
    num_bins: tl.constexpr,
    BLOCK_N: tl.constexpr,
    descending: tl.constexpr,
):
    pid = tl.program_id(0)
    num_jobs = tl.num_programs(0)
    num_blocks_per_row = tl.cdiv(N, BLOCK_N)
    for task_id in range(pid, total_tasks, num_jobs):
        row_idx = task_id // num_blocks_per_row
        block_idx = task_id % num_blocks_per_row

        row_start = row_idx * N
        n_offset = block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = n_offset < N

        val = tl.load(x_ptr + row_start + n_offset, mask=mask, other=0)
        val_u = convert_to_uint_preverse_order(val, descending)
        bfe_mask = num_bins - 1
        key = (val_u >> bit_offset) & bfe_mask

        base = row_idx * num_blocks_per_row * num_bins + block_idx * num_bins
        for i in range(num_bins):
            bin_mask = (key == i) & mask
            count = tl.sum(bin_mask.to(tl.int32))
            tl.store(counts_ptr + base + i, count)


@triton.jit
def scatter_kernel(
    x_ptr,
    x_out_ptr,
    idx_in_ptr,
    idx_out_ptr,
    global_offsets_ptr,  # (M, grid_n, num_bins): global start offset per task per bin
    total_tasks,
    M,
    N,
    bit_offset,
    num_bins: tl.constexpr,
    BLOCK_N: tl.constexpr,
    descending: tl.constexpr,
):
    pid = tl.program_id(0)
    num_jobs = tl.num_programs(0)
    num_blocks_per_row = tl.cdiv(N, BLOCK_N)
    for task_id in range(pid, total_tasks, num_jobs):
        row_idx = task_id // num_blocks_per_row
        block_idx = task_id % num_blocks_per_row

        row_start = row_idx * N
        n_offset = block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = n_offset < N

        val = tl.load(x_ptr + row_start + n_offset, mask=mask, other=0)
        val_u = convert_to_uint_preverse_order(val, descending)
        idx = tl.load(idx_in_ptr + row_start + n_offset, mask=mask, other=0)
        bfe_mask = num_bins - 1
        key = (val_u >> bit_offset) & bfe_mask

        base = row_idx * num_blocks_per_row * num_bins + block_idx * num_bins
        for i in range(num_bins):
            bin_mask = (key == i) & mask
            # stable: within-bin rank preserves the block-local order, and the
            # block's global start offset preserves cross-block order.
            local_rank = tl.cumsum(bin_mask.to(tl.int32), axis=0) - 1
            global_start = tl.load(global_offsets_ptr + base + i)
            dest_idx = row_start + global_start + local_rank
            tl.store(x_out_ptr + dest_idx, val, mask=bin_mask)
            tl.store(idx_out_ptr + dest_idx, idx, mask=bin_mask)


@triton.jit
def init_indices_kernel(indices_ptr, total, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_jobs = tl.num_programs(0)
    total_blocks = tl.cdiv(total, BLOCK)

    for block_id in range(pid, total_blocks, num_jobs):
        offsets = block_id * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < total
        values = offsets % n
        tl.store(indices_ptr + offsets, values, mask=mask)


def radix_sort(arr, k_bits=8, descending=False):
    n = arr.shape[-1]
    m = arr.numel() // n
    assert n < (1 << 30), "we have not implemented 2**30 per launch"
    dtype = arr.dtype
    num_bits = 1 if dtype == torch.bool else (arr.itemsize * 8)

    num_bins = 2**k_bits
    n_passes = triton.cdiv(num_bits, k_bits)

    # Block width along the sort dim. Cap at 1024 so wide rows split into
    # multiple blocks (more parallelism, bounded per-block work); shrink to
    # next_pow2(n) so tiny rows (e.g. n=4 in the dim=0 case) don't waste lanes.
    BLOCK_N = min(1024, triton.next_power_of_2(n))
    grid_n = triton.cdiv(n, BLOCK_N)
    total_tasks = m * grid_n
    # grid-stride: bounded grid, each program strides over many (row, block) tasks
    grid = (min(total_tasks, MAX_GRID_SIZE_X),)
    # host-side cumsum chunking guard for very large m
    max_cumsum_m = 65535

    with torch_device_fn.device(arr.device):
        arr_in = torch.clone(arr)
        indices_in = torch.empty(arr.shape, dtype=torch.int64, device=arr_in.device)
        indices_numel = indices_in.numel()
        indices_block = 1024
        init_indices_kernel[
            (min(triton.cdiv(indices_numel, indices_block), MAX_GRID_SIZE_X),)
        ](indices_in, indices_numel, n, BLOCK=indices_block)
        arr_out = torch.empty_like(arr)
        indices_out = torch.empty_like(indices_in)

        counts = torch.empty((m, grid_n, num_bins), device=arr.device, dtype=torch.int32)

        for i in range(0, n_passes):
            bit_offset = i * k_bits

            # pass 1: per-(row, block) per-bin local counts (no inter-block dep)
            count_kernel[grid](
                arr_in,
                counts,
                total_tasks,
                m,
                n,
                bit_offset,
                num_bins,
                BLOCK_N,
                descending,
            )

            # host prefix sums -> global start offset for each (row, block, bin).
            # total_per_bin: (m, num_bins); bin_starts: exclusive prefix over bins.
            # block_prefix: exclusive prefix over blocks within each bin.
            # global_offsets = bin_starts (broadcast over blocks) + block_prefix.
            def _compute_offsets(cnt):
                total_per_bin = cnt.sum(dim=1)  # (m, num_bins)
                bin_starts = torch.cumsum(total_per_bin, dim=1) - total_per_bin
                block_prefix = torch.cumsum(cnt, dim=1) - cnt  # (m, grid_n, num_bins)
                return bin_starts.unsqueeze(1) + block_prefix

            if m <= max_cumsum_m:
                global_offsets = _compute_offsets(counts)
            else:
                chunks = []
                for s_start in range(0, m, max_cumsum_m):
                    s_end = min(m, s_start + max_cumsum_m)
                    chunks.append(_compute_offsets(counts[s_start:s_end]))
                global_offsets = torch.cat(chunks, dim=0)
            global_offsets = global_offsets.to(torch.int32).contiguous()

            # pass 2: scatter each element to its global position (stable)
            scatter_kernel[grid](
                arr_in,
                arr_out,
                indices_in,
                indices_out,
                global_offsets,
                total_tasks,
                m,
                n,
                bit_offset,
                num_bins,
                BLOCK_N,
                descending,
            )

            arr_in, arr_out = arr_out, arr_in
            indices_in, indices_out = indices_out, indices_in

    return arr_in, indices_in


def sort(inp, dim=-1, descending=False):
    # We only implement stable radix sort here
    logger.debug("GEMS_CAMBRICON SORT")
    return sort_stable(inp, stable=False, dim=dim, descending=descending)


def sort_stable(inp, *, stable, dim=-1, descending=False):
    logger.debug("GEMS_CAMBRICON SORT_STABLE")
    # We only implement stable radix sort here
    _ = stable
    sort_elem_cnt = inp.shape[dim]
    if sort_elem_cnt == 1:
        return inp, torch.zeros_like(inp, dtype=torch.int64)

    if dim < 0:
        dim = dim + inp.ndim
    if dim != inp.ndim - 1:
        inp = torch.movedim(inp, dim, -1).contiguous()
    else:
        inp = inp.contiguous()

    dtype = inp.dtype
    num_bits_per_pass = 1 if dtype == torch.bool else 4
    out, out_index = radix_sort(inp, num_bits_per_pass, descending)

    if dim != inp.ndim - 1:
        out = torch.movedim(out, -1, dim)
        out_index = torch.movedim(out_index, -1, dim)
    return out, out_index
