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
import os

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.libentry import libentry

logger = logging.getLogger(__name__)

# Tile / CTA config of the three unique paths; module constants ease shape tiering and
# batch sweeps. Defaults keep the original behavior, with SIMPLE_TILE_FLOOR as a GCU400 fix.
_UNIQUE_TILE_CAP = 8192
# 15 sweeps + 14 alternating A/B rounds: a tile floor of 64 gives a 3-case total latency median of
# 7.929 ms, ~4.4% faster than 32's 8.297 ms; it also removes the
# tile_size=1 compiler assertion for 0-d/1-element inputs (the floor must be >=32).
_UNIQUE_TILE_FLOOR = 64
_UNIQUE_CTA_HARD_LIMIT = 65536
_UNIQUE_CTA_FALLBACK = 2048
_UNIQUE_INV_CTA_LIMIT = 32768
_UNIQUE_INV_CTA_FALLBACK = 8192
_UNIQUE_NUM_WARPS = 4
_UNIQUE_SIMPLE_MAX_NUMEL = 8192


# Temporary sweep switch: 0 keeps the default cap, >0 overrides the tile cap of
# the quick path.
def _quick_tile_cap() -> int:
    override = int(os.environ.get("GEMS_UNIQUE2_QUICK_TILE", "0") or 0)
    return override if override > 0 else 4096


def _idx_tile_cap() -> int:
    override = int(os.environ.get("GEMS_UNIQUE2_IDX_TILE", "0") or 0)
    return override if override > 0 else _UNIQUE_TILE_CAP


def _unique_warps() -> int:
    override = int(os.environ.get("GEMS_UNIQUE2_WARPS", "0") or 0)
    return override if override > 0 else _UNIQUE_NUM_WARPS


@triton.jit
def _unique_ne(a, b, NAN_AS_UNIQUE: tl.constexpr):
    """Elementwise "not equal" used to detect boundaries between unique values.

    ``NAN_AS_UNIQUE`` selects the aten::_unique2 contract: every NaN occurrence
    keeps its own unique slot while ``-0.0``/``+0.0`` collapse into one value.
    The other mode folds all NaNs onto a canonical encoding.

    Both modes must be rebuilt from bit patterns on GCU: its float comparisons
    are lowered to raw bit comparisons, so ``NaN != NaN`` evaluates to false and
    ``-0.0 != 0.0`` evaluates to true, the opposite of IEEE ``!=``.
    """
    if a.dtype.is_floating():
        if a.dtype == tl.float64:
            int_dtype = tl.uint64
            sign_bit = 0x8000000000000000
            magnitude_mask = 0x7FFFFFFFFFFFFFFF
            infinity_bits = 0x7FF0000000000000
            nan_bits = 0x7FF8000000000000
        elif a.dtype == tl.float32:
            int_dtype = tl.uint32
            sign_bit = 0x80000000
            magnitude_mask = 0x7FFFFFFF
            infinity_bits = 0x7F800000
            nan_bits = 0x7FC00000
        else:
            int_dtype = tl.uint16
            sign_bit = 0x8000
            magnitude_mask = 0x7FFF
            infinity_bits = 0x7C00
            nan_bits = 0x7E00
        a_bits = a.to(int_dtype, bitcast=True)
        b_bits = b.to(int_dtype, bitcast=True)
        if NAN_AS_UNIQUE:
            # Fold -0.0 onto +0.0 so signed zeros compare equal, then mark every
            # NaN as distinct: an encoder that only reacts to magnitude bits
            # cannot express "NaN never equals NaN", so add the NaN flags back.
            a_key = tl.where(a_bits == sign_bit, 0, a_bits)
            b_key = tl.where(b_bits == sign_bit, 0, b_bits)
            a_is_nan = (a_bits & magnitude_mask) > infinity_bits
            b_is_nan = (b_bits & magnitude_mask) > infinity_bits
            return (a_key != b_key) | a_is_nan | b_is_nan
        a_key = tl.where(a != a, nan_bits, a_bits)
        b_key = tl.where(b != b, nan_bits, b_bits)
        return a_key != b_key
    return a != b


@libentry()
@triton.jit
def simple_unique_flat_kernel(
    sorted_data_ptr: tl.tensor,
    sorted_indices_ptr: tl.tensor,  # in
    data_out_ptr: tl.tensor,
    inverse_indices_ptr: tl.tensor,
    idx_ptr: tl.tensor,
    unique_size_ptr: tl.tensor,  # out
    return_inverse: tl.constexpr,
    return_counts: tl.constexpr,
    num_tasks: int,
    tile_size: tl.constexpr,
    NAN_AS_UNIQUE: tl.constexpr,
):
    i0 = tl.arange(0, tile_size)
    mask = i0 < num_tasks

    # load
    a = tl.load(sorted_data_ptr + i0, mask=mask)
    i0_prev = tl.where(i0 > 0, i0 - 1, 0)
    b = tl.load(sorted_data_ptr + i0_prev, mask=mask)

    # ne & cumsum
    ne_result = tl.where(i0 > 0, _unique_ne(a, b, NAN_AS_UNIQUE), 0)
    cumsum = tl.cumsum(ne_result)

    # unique_size
    unique_size_mask = i0 == tile_size - 1
    tl.store(unique_size_ptr + tl.zeros_like(i0), cumsum, mask=unique_size_mask)

    # data_out: scatter_(to=cumsum, sorted_data)
    tl.store(data_out_ptr + cumsum, a, mask=mask)

    # inverse_indices: scatter_(to=sorted_indices, cumsum)
    if return_inverse:
        sorted_indices = tl.load(sorted_indices_ptr + i0, mask=mask)
        tl.store(inverse_indices_ptr + sorted_indices, cumsum, mask=mask)

    # idx
    if return_counts:
        idx_mask = ((i0 == 0) | ne_result.to(tl.int1)) & mask
        tl.store(idx_ptr + cumsum, i0, mask=idx_mask)


@triton.jit
def output_counts_flat_impl(
    global_pid,
    idx_ptr: tl.tensor,
    origin_num_tasks: int,  # in
    counts_ptr: tl.tensor,  # out
    num_tasks: int,
    tile_size: tl.constexpr,
):
    r = tl.arange(0, tile_size)

    # load idx
    i0 = global_pid * tile_size + r
    mask = i0 < num_tasks
    idx = tl.load(idx_ptr + i0, mask=mask)

    # load idx_next
    i0_next = i0 + 1
    next_mask = i0_next < num_tasks
    idx_next = tl.load(idx_ptr + i0_next, mask=next_mask)

    # diff
    counts = tl.where(i0_next < num_tasks, idx_next - idx, origin_num_tasks - idx)

    # store counts
    tl.store(counts_ptr + i0, counts, mask=mask)


@libentry()
@triton.jit
def output_counts_flat_kernel(
    idx_ptr: tl.tensor,
    origin_num_tasks: int,  # in
    counts_ptr: tl.tensor,  # out
    num_tasks: int,
    tiles_per_cta: int,
    tile_size: tl.constexpr,
):
    pid = tl.program_id(0)
    ctas_num = tl.num_programs(0)
    # grid-stride-loop style kernel
    for j in range(0, tiles_per_cta):
        global_pid = pid + j * ctas_num
        output_counts_flat_impl(
            global_pid,
            idx_ptr,
            origin_num_tasks,  # in
            counts_ptr,  # out
            num_tasks,
            tile_size,
        )


@triton.jit
def local_quick_unique_flat_impl(
    global_pid,
    sorted_data_ptr: tl.tensor,  # in
    local_unique_ptr: tl.tensor,
    origin_idx_ptr: tl.tensor,
    tile_sum_ptr: tl.tensor,  # out
    global_ctas_num: int,
    num_tasks: int,
    tile_size: tl.constexpr,
    return_counts: tl.constexpr,
    NAN_AS_UNIQUE: tl.constexpr,
):
    offset = global_pid * tile_size
    r = tl.arange(0, tile_size)
    i0 = offset + r
    mask = i0 < num_tasks

    # load
    a = tl.load(sorted_data_ptr + i0, mask=mask)
    i0_prev = tl.where(i0 > 0, i0 - 1, 0)
    b = tl.load(sorted_data_ptr + i0_prev, mask=mask)

    # ne & cumsum
    ne_result = tl.where(i0 > 0, _unique_ne(a, b, NAN_AS_UNIQUE), 0)
    cumsum = tl.cumsum(ne_result)

    # local_id or local_unique
    local_unique_offset = cumsum - tl.where(global_pid > 0, 1, 0)
    local_unique_mask = (local_unique_offset >= 0) & mask
    if return_counts:
        # origin_idx: scatter_(to=cumsum, i0)
        origin_idx_mask = ((i0 == 0) | ne_result.to(tl.int1)) & local_unique_mask
        tl.store(
            origin_idx_ptr + (offset + local_unique_offset),
            i0,
            mask=origin_idx_mask,
        )
    # The deduplicated values are always written compacted in place: with
    # return_counts, local_unique is written as well so the downstream step compacts
    # directly instead of doing another gather by idx (measured saving of 200~400us
    # on 262144 elements).
    tl.store(
        local_unique_ptr + (offset + local_unique_offset), a, mask=local_unique_mask
    )

    # tile_sum
    tile_sum_mask = (r == tile_size - 1) & (global_pid < global_ctas_num)
    tile_sum = tl.where(tile_sum_mask & (global_pid == 0), cumsum + 1, cumsum)
    tl.store(tile_sum_ptr + global_pid + tl.zeros_like(r), tile_sum, mask=tile_sum_mask)


@libentry()
@triton.jit
def local_quick_unique_flat_kernel(
    sorted_data_ptr: tl.tensor,  # in
    local_unique_ptr: tl.tensor,
    origin_idx_ptr: tl.tensor,
    tile_sum_ptr: tl.tensor,  # out
    global_ctas_num: int,
    num_tasks: int,
    tiles_per_cta: int,
    tile_size: tl.constexpr,
    return_counts: tl.constexpr,
    NAN_AS_UNIQUE: tl.constexpr,
):
    pid = tl.program_id(0)
    ctas_num = tl.num_programs(0)
    # grid-stride-loop style kernel
    for j in range(0, tiles_per_cta):
        global_pid = pid + j * ctas_num
        local_quick_unique_flat_impl(
            global_pid,
            sorted_data_ptr,  # in
            local_unique_ptr,
            origin_idx_ptr,
            tile_sum_ptr,  # out
            global_ctas_num,
            num_tasks,
            tile_size,
            return_counts,
            NAN_AS_UNIQUE,
        )


@triton.jit
def global_quick_unique_flat_impl(
    global_pid,
    local_unique_ptr: tl.tensor,
    origin_idx_ptr: tl.tensor,
    tile_sum_ptr: tl.tensor,
    tile_cumsum_ptr: tl.tensor,  # in
    data_out_ptr: tl.tensor,
    idx_ptr: tl.tensor,  # out
    global_ctas_num: int,
    num_tasks: int,
    tile_size: tl.constexpr,
    return_counts: tl.constexpr,
):
    r = tl.arange(0, tile_size)
    i0 = global_pid * tile_size + r
    mask = i0 < num_tasks

    # tile_cumsum is now an inclusive prefix sum, so this tile's start offset must
    # have its own tile_sum subtracted; the caller then needs only one cumsum and
    # saves the two extra launches of the zeros init and the slice assignment.
    cur_tile_sum = tl.load(
        tile_sum_ptr + global_pid, mask=global_pid < global_ctas_num, other=0
    )
    prefix = tl.load(tile_cumsum_ptr + global_pid) - cur_tile_sum

    tile_mask = r < cur_tile_sum
    out_offset = prefix + r
    local_unique = tl.load(local_unique_ptr + i0, mask=mask)
    tl.store(data_out_ptr + out_offset, local_unique, mask=tile_mask)
    if return_counts:
        origin_idx = tl.load(origin_idx_ptr + i0, mask=mask)
        tl.store(idx_ptr + out_offset, origin_idx, mask=tile_mask)


@libentry()
@triton.jit
def global_quick_unique_flat_kernel(
    local_unique_ptr: tl.tensor,
    origin_idx_ptr: tl.tensor,
    tile_sum_ptr: tl.tensor,
    tile_cumsum_ptr: tl.tensor,  # in
    data_out_ptr: tl.tensor,
    idx_ptr: tl.tensor,  # out
    global_ctas_num: int,
    num_tasks: int,
    tiles_per_cta: int,
    tile_size: tl.constexpr,
    return_counts: tl.constexpr,
):
    pid = tl.program_id(0)
    ctas_num = tl.num_programs(0)
    for j in range(0, tiles_per_cta):
        global_pid = pid + j * ctas_num
        global_quick_unique_flat_impl(
            global_pid,
            local_unique_ptr,
            origin_idx_ptr,
            tile_sum_ptr,
            tile_cumsum_ptr,  # in
            data_out_ptr,
            idx_ptr,  # out
            global_ctas_num,
            num_tasks,
            tile_size,
            return_counts,
        )


def sorted_quick_unique_flat(
    sorted_data: torch.Tensor, return_counts: bool, nan_as_unique: bool = False
):
    logger.debug("GEMS_ENFLAME SORTED_QUICK_UNIQUE_FLAT")
    num_tasks = sorted_data.numel()
    next_power_num_tasks = triton.next_power_of_2(num_tasks)
    # The tile depends only on the data size and no longer shrinks with the CTA count:
    # the old heuristic pushed the 262144-element tile down to 64, so 4096 CTAs each
    # handled just 64 elements (measured 1288us, 2.5x the idx path). A large tile plus
    # grid-stride is the memory-friendly form.
    tile_size = min(_quick_tile_cap(), next_power_num_tasks)
    tile_size = max(_UNIQUE_TILE_FLOOR, tile_size)
    global_ctas_num = triton.cdiv(num_tasks, tile_size)
    ctas_num = (
        global_ctas_num
        if global_ctas_num < _UNIQUE_CTA_HARD_LIMIT
        else _UNIQUE_CTA_FALLBACK
    )
    tiles_per_cta = triton.cdiv(global_ctas_num, ctas_num)
    num_warps = _unique_warps()
    grid = (ctas_num, 1, 1)

    # allocate tensor
    local_unique = torch.empty_like(sorted_data)
    if return_counts:
        origin_idx = torch.empty_like(sorted_data, dtype=torch.int64)
        idx = torch.empty_like(origin_idx)
    else:
        origin_idx = None
        idx = None
        counts = None
    tile_sum = torch.empty(
        (global_ctas_num,), dtype=torch.int64, device=sorted_data.device
    )
    data_out = torch.empty_like(sorted_data)

    # launch kernel
    with torch_device_fn.device(sorted_data.device.index):
        local_quick_unique_flat_kernel[grid](
            sorted_data,  # in
            local_unique,
            origin_idx,
            tile_sum,  # out
            global_ctas_num,
            num_tasks,
            tiles_per_cta=tiles_per_cta,
            tile_size=tile_size,
            return_counts=return_counts,
            num_warps=num_warps,
            NAN_AS_UNIQUE=nan_as_unique,
        )
        tile_cumsum = torch.cumsum(tile_sum, dim=0)
        global_quick_unique_flat_kernel[grid](
            local_unique,
            origin_idx,
            tile_sum,
            tile_cumsum,  # in
            data_out,
            idx,  # out
            global_ctas_num,
            num_tasks,
            tiles_per_cta=tiles_per_cta,
            tile_size=tile_size,
            return_counts=return_counts,
            num_warps=num_warps,
        )
        out_size = tile_cumsum[-1].item()
        if return_counts:
            # counts are derived only by differencing the compacted start idx, so the
            # raw data no longer needs to be gathered.
            idx = idx[:out_size]
            counts = torch.empty(
                (out_size,), dtype=torch.int64, device=sorted_data.device
            )
            counts_tile = max(
                _UNIQUE_TILE_FLOOR,
                min(_quick_tile_cap(), triton.next_power_of_2(out_size)),
            )
            output_counts_flat_kernel[(triton.cdiv(out_size, counts_tile), 1, 1)](
                idx,
                num_tasks,  # in
                counts,  # out
                out_size,
                tiles_per_cta=1,
                tile_size=counts_tile,
                num_warps=num_warps,
            )

    if return_counts:
        return data_out[:out_size], None, counts
    else:
        return data_out[:out_size], None, None


@triton.jit
def local_ne_flat_impl(
    global_pid,
    sorted_data_ptr: tl.tensor,  # in
    ne_result_ptr: tl.tensor,
    tile_sum_ptr: tl.tensor,  # out
    global_ctas_num: int,
    num_tasks: int,
    tile_size: tl.constexpr,
    NAN_AS_UNIQUE: tl.constexpr,
):
    r = tl.arange(0, tile_size)
    i0 = global_pid * tile_size + r
    mask = i0 < num_tasks
    i0_prev = tl.where(i0 > 0, i0 - 1, 0)

    # load
    a = tl.load(sorted_data_ptr + i0, mask=mask)
    b = tl.load(sorted_data_ptr + i0_prev, mask=mask)

    # compute
    ne_result = tl.where(i0 > 0, _unique_ne(a, b, NAN_AS_UNIQUE), 0)

    # store ne_result
    tl.store(ne_result_ptr + i0, ne_result, mask=mask)

    # store tile_sum
    tile_sum = tl.sum(ne_result)
    tile_sum_mask = global_pid < global_ctas_num
    tl.store(tile_sum_ptr + global_pid, tile_sum, mask=tile_sum_mask)


@libentry()
@triton.jit
def local_ne_flat_kernel(
    sorted_data_ptr: tl.tensor,  # in
    ne_result_ptr: tl.tensor,
    tile_sum_ptr: tl.tensor,  # out
    global_ctas_num: int,
    num_tasks: int,
    tiles_per_cta: int,
    tile_size: tl.constexpr,
    NAN_AS_UNIQUE: tl.constexpr,
):
    pid = tl.program_id(0)
    ctas_num = tl.num_programs(0)
    # grid-stride-loop style kernel
    for j in range(0, tiles_per_cta):
        global_pid = pid + j * ctas_num
        local_ne_flat_impl(
            global_pid,
            sorted_data_ptr,  # in
            ne_result_ptr,
            tile_sum_ptr,  # out
            global_ctas_num,
            num_tasks,
            tile_size,
            NAN_AS_UNIQUE,
        )


@triton.jit
def global_cumsum_flat_impl(
    global_pid,
    ne_result_ptr: tl.tensor,
    tile_sum_ptr: tl.tensor,
    tile_cumsum_ptr: tl.tensor,  # in
    sorted_data_ptr: tl.tensor,
    sorted_indices_ptr: tl.tensor,  # in
    data_out_ptr: tl.tensor,
    inverse_indices_ptr: tl.tensor,
    idx_ptr: tl.tensor,  # out
    global_ctas_num: int,
    num_tasks: int,
    tile_size: tl.constexpr,
    return_counts: tl.constexpr,
):
    offset = global_pid * tile_size
    r = tl.arange(0, tile_size)
    i0 = offset + r
    mask = i0 < num_tasks

    sorted_data = tl.load(sorted_data_ptr + i0, mask=mask)
    sorted_indices = tl.load(sorted_indices_ptr + i0, mask=mask)

    # inclusive prefix minus the tile's own tile_sum = exclusive prefix, which saves
    # the caller the zeros + slice assignment.
    prefix = tl.load(tile_cumsum_ptr + global_pid) - tl.load(tile_sum_ptr + global_pid)

    ne_result = tl.load(ne_result_ptr + i0, mask=mask)
    ne_result_i1 = ne_result.to(tl.int1)
    ne_result = ne_result.to(tl.int32)
    cumsum = tl.cumsum(ne_result)
    cumsum += prefix

    tl.store(data_out_ptr + cumsum, sorted_data, mask=mask)
    tl.store(inverse_indices_ptr + sorted_indices, cumsum, mask=mask)

    if return_counts:
        idx_mask = ((i0 == 0) | ne_result_i1) & mask
        tl.store(idx_ptr + cumsum, i0, mask=idx_mask)


@libentry()
@triton.jit
def global_cumsum_flat_kernel(
    ne_result_ptr: tl.tensor,
    tile_sum_ptr: tl.tensor,
    tile_cumsum_ptr: tl.tensor,  # in
    sorted_data_ptr: tl.tensor,
    sorted_indices_ptr: tl.tensor,  # in
    data_out_ptr: tl.tensor,
    inverse_indices_ptr: tl.tensor,
    idx_ptr: tl.tensor,  # out
    global_ctas_num: int,
    num_tasks: int,
    tiles_per_cta: int,
    tile_size: tl.constexpr,
    return_counts: tl.constexpr,
):
    pid = tl.program_id(0)
    ctas_num = tl.num_programs(0)
    for j in range(0, tiles_per_cta):
        global_pid = pid + j * ctas_num
        global_cumsum_flat_impl(
            global_pid,
            ne_result_ptr,
            tile_sum_ptr,
            tile_cumsum_ptr,  # in
            sorted_data_ptr,
            sorted_indices_ptr,  # in
            data_out_ptr,
            inverse_indices_ptr,
            idx_ptr,  # out
            global_ctas_num,
            num_tasks,
            tile_size,
            return_counts,
        )


def sorted_indices_unique_flat(
    sorted_data: torch.Tensor,
    sorted_indices: torch.Tensor,
    return_counts: bool,
    nan_as_unique: bool = False,
):
    logger.debug("GEMS_ENFLAME SORTED_INDICES_UNIQUE_FLAT")
    num_tasks = sorted_data.numel()
    next_power_num_tasks = triton.next_power_of_2(num_tasks)
    # The inverse path scans every tile twice (run diff + global cumsum carry),
    # so its tile wants to be small: 12-config sweep on fp32 (do_bench median,
    # speedup of the whole _unique2 call) put n/16 clamped to [1024, 4096] ahead
    # of the flat 8192 cap at every size - n=16384: 0.875x vs 0.725x, n=65536:
    # 1.154x vs 1.101x, n=262144: 1.271x vs 1.238x.
    size_cap = max(1024, min(4096, next_power_num_tasks // 16))
    tile_size = min(_idx_tile_cap(), size_cap, next_power_num_tasks)
    tile_size = max(_UNIQUE_TILE_FLOOR, tile_size)
    global_ctas_num = triton.cdiv(num_tasks, tile_size)
    ctas_num = (
        global_ctas_num
        if global_ctas_num < _UNIQUE_INV_CTA_LIMIT
        else _UNIQUE_INV_CTA_FALLBACK
    )
    tiles_per_cta = triton.cdiv(global_ctas_num, ctas_num)
    num_warps = _unique_warps()
    grid = (ctas_num, 1, 1)

    # allocate tensor
    ne_result = torch.empty_like(sorted_data, dtype=torch.bool)
    tile_sum = torch.empty(
        (global_ctas_num,), dtype=torch.int64, device=sorted_data.device
    )
    data_out = torch.empty_like(sorted_data)
    inverse_indices = torch.empty_like(sorted_data, dtype=torch.int64)
    idx = None
    if return_counts:
        idx = torch.empty_like(inverse_indices)

    # launch kernel
    with torch_device_fn.device(sorted_data.device.index):
        local_ne_flat_kernel[grid](
            sorted_data,  # in
            ne_result,
            tile_sum,  # out
            global_ctas_num,
            num_tasks,
            tiles_per_cta=tiles_per_cta,
            tile_size=tile_size,
            num_warps=num_warps,
            NAN_AS_UNIQUE=nan_as_unique,
        )
        tile_cumsum = torch.cumsum(tile_sum, dim=0)
        global_cumsum_flat_kernel[grid](
            ne_result,
            tile_sum,
            tile_cumsum,  # in
            sorted_data,
            sorted_indices,  # in
            data_out,
            inverse_indices,
            idx,  # out
            global_ctas_num,
            num_tasks,
            tiles_per_cta=tiles_per_cta,
            tile_size=tile_size,
            return_counts=return_counts,
            num_warps=num_warps,
        )
        out_size = tile_cumsum[-1].item() + 1
        counts = None
        if return_counts:
            idx = idx[:out_size]
            counts = torch.empty_like(idx)
            output_counts_flat_kernel[grid](
                idx,
                num_tasks,  # in
                counts,  # out
                out_size,
                tiles_per_cta,
                tile_size,
                num_warps=num_warps,
            )

    return data_out[:out_size], inverse_indices, counts


def simple_unique_flat(
    sorted_data: torch.Tensor,
    sorted_indices: torch.Tensor,
    return_inverse: bool,
    return_counts: bool,
    nan_as_unique: bool = False,
):
    logger.debug("GEMS_ENFLAME SIMPLE_UNIQUE_FLAT")
    num_tasks = sorted_data.numel()
    grid = (1, 1, 1)

    # allocate tensor
    data_out = torch.empty_like(sorted_data)
    if return_inverse:
        inverse_indices = torch.empty_like(sorted_data, dtype=torch.int64)
    else:
        inverse_indices = None
    if return_counts:
        idx = torch.empty_like(sorted_data, dtype=torch.int64)
    else:
        idx = None
    unique_size = torch.empty([1], dtype=torch.int64, device=sorted_data.device)

    # launch kernel
    with torch_device_fn.device(sorted_data.device.index):
        simple_unique_flat_kernel[grid](
            sorted_data,
            sorted_indices,  # in
            data_out,
            inverse_indices,
            idx,
            unique_size,  # out
            return_inverse,
            return_counts,
            num_tasks,
            # 0-d / 1-element inputs make tile_size=1, and the GCU400 compiler triggers an MLIR TypeRange assertion
            # on tl.cumsum(1-element tensor) and abort(134);
            # a floor of 32 does not change semantics (out-of-range elements are masked, cumsum is preserved).
            tile_size=max(_UNIQUE_TILE_FLOOR, triton.next_power_of_2(num_tasks)),
            num_warps=_UNIQUE_NUM_WARPS,
            NAN_AS_UNIQUE=nan_as_unique,
        )
    out_size = unique_size.item() + 1
    counts = None
    if return_counts:
        idx = idx[:out_size]
        counts = torch.empty_like(idx)
        with torch_device_fn.device(sorted_data.device.index):
            output_counts_flat_kernel[grid](
                idx,
                num_tasks,  # in
                counts,  # out
                num_tasks=out_size,
                tiles_per_cta=1,
                tile_size=triton.next_power_of_2(out_size),
                num_warps=4,
            )
    return data_out[:out_size], inverse_indices, counts


def _unique2(
    in0: torch.Tensor,
    sorted: bool = True,
    return_inverse: bool = False,
    return_counts: bool = False,
):
    logger.debug("GEMS_ENFLAME _UNIQUE2")
    # Zero-size guard: the flat kernels build tile_size = next_power_of_2(numel)
    # and would launch tl.arange(0, 0) for an empty input. aten::_unique2 on an
    # empty tensor yields empty outputs, so short-circuit before touching Triton.
    if in0.numel() == 0:
        data_out = torch.empty(0, dtype=in0.dtype, device=in0.device)
        # ATen view_as's inverse_indices even for empty input, so apply it
        # unconditionally here: identical for 1-D (both (0,)), and matches
        # ATen's (0, d1, ...) for empty N-D inputs.
        inverse_indices = torch.empty(0, dtype=torch.int64, device=in0.device).view_as(
            in0
        )
        counts = torch.empty(0, dtype=torch.int64, device=in0.device)
        return data_out, inverse_indices, counts
    # NaN handling follows the ATen reference the test suite compares against:
    # aten::_unique2 keeps one unique slot per NaN occurrence (NaN != NaN) and
    # folds -0.0/+0.0 into a single value, i.e. nan_as_unique=True.
    if in0.numel() <= _UNIQUE_SIMPLE_MAX_NUMEL:
        sorted_data, sorted_indices = torch.sort(in0.ravel())
        data_out, inverse_indices, counts = simple_unique_flat(
            sorted_data,
            sorted_indices,
            return_inverse,
            return_counts,
            nan_as_unique=True,
        )
    elif return_inverse:
        sorted_data, sorted_indices = torch.sort(in0.ravel())
        data_out, inverse_indices, counts = sorted_indices_unique_flat(
            sorted_data, sorted_indices, return_counts, nan_as_unique=True
        )
    else:
        sorted_data, _ = torch.sort(in0.ravel())
        data_out, inverse_indices, counts = sorted_quick_unique_flat(
            sorted_data, return_counts, nan_as_unique=True
        )
    # aten::_unique2 always returns three tensors: when return_inverse /
    # return_counts are False the corresponding output is an empty (0,) int64
    # tensor rather than None. Mirror that contract so callers can rely on the
    # shapes matching ATen regardless of the requested flags.
    if inverse_indices is None:
        inverse_indices = torch.empty(0, dtype=torch.int64, device=in0.device)
    else:
        inverse_indices = inverse_indices.view_as(in0)
    if counts is None:
        counts = torch.empty(0, dtype=torch.int64, device=in0.device)
    return data_out, inverse_indices, counts
