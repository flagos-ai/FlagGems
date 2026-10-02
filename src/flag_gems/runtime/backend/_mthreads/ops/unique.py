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
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.libentry import libentry

logger = logging.getLogger(__name__)

# Maximum number of tiles the single-CTA exclusive scan can handle.
# next_power_of_2(P) must fit the 65 536 per-axis CTA launch limit, so
# P must be ≤ 32768 (num_tasks ≤ 268_435_456). Above that we fall back
# to torch.cumsum on the device (tile_counts is at most a few ×10^4
# int32 values — negligible cost).
_MAX_SINGLE_CTA_SCAN_TILES = 32768


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
):
    i0 = tl.arange(0, tile_size)
    mask = i0 < num_tasks

    # load
    a = tl.load(sorted_data_ptr + i0, mask=mask)
    i0_prev = tl.where(i0 > 0, i0 - 1, 0)
    b = tl.load(sorted_data_ptr + i0_prev, mask=mask)

    # ne & cumsum
    ne_result = tl.where(i0 > 0, a != b, 0)
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
    pid = ext.program_id(0)
    ctas_num = ext.num_programs(0)
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
def quick_output_flat_impl(
    global_pid,
    sorted_data_ptr: tl.tensor,
    idx_ptr: tl.tensor,
    origin_num_tasks: int,  # in
    data_out_ptr: tl.tensor,
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

    # data_out: gather(sorted_data, from=idx)
    sorted_data = tl.load(sorted_data_ptr + idx, mask=mask)
    tl.store(data_out_ptr + i0, sorted_data, mask=mask)


@libentry()
@triton.jit
def quick_output_flat_kernel(
    sorted_data_ptr: tl.tensor,
    idx_ptr: tl.tensor,
    origin_num_tasks: int,  # in
    data_out_ptr: tl.tensor,
    counts_ptr: tl.tensor,  # out
    num_tasks: int,
    tiles_per_cta: int,
    tile_size: tl.constexpr,
):
    pid = ext.program_id(0)
    ctas_num = ext.num_programs(0)
    # grid-stride-loop style kernel
    for j in range(0, tiles_per_cta):
        global_pid = pid + j * ctas_num
        quick_output_flat_impl(
            global_pid,
            sorted_data_ptr,
            idx_ptr,
            origin_num_tasks,  # in
            data_out_ptr,
            counts_ptr,  # out
            num_tasks,
            tile_size,
        )


@triton.jit
def tile_boundary_counts_impl(
    global_pid,
    sorted_data_ptr: tl.tensor,  # in
    tile_counts_ptr: tl.tensor,  # out
    num_tasks: int,
    tile_size: tl.constexpr,
):
    i0 = global_pid * tile_size + tl.arange(0, tile_size)
    mask = i0 < num_tasks
    a = tl.load(sorted_data_ptr + i0, mask=mask)
    i0_prev = tl.where(i0 > 0, i0 - 1, 0)
    b = tl.load(sorted_data_ptr + i0_prev, mask=mask)
    # a boundary is the first position of each group; position 0 is one
    ne_result = tl.where(i0 > 0, a != b, 0)
    tl.store(tile_counts_ptr + global_pid, tl.sum(ne_result.to(tl.int32), axis=0))


@libentry()
@triton.jit
def tile_boundary_counts_kernel(
    sorted_data_ptr: tl.tensor,  # in
    tile_counts_ptr: tl.tensor,  # out
    num_tasks: int,
    tiles_per_cta: int,
    tile_size: tl.constexpr,
):
    pid = ext.program_id(0)
    ctas_num = ext.num_programs(0)
    # grid-stride-loop style kernel
    for j in range(0, tiles_per_cta):
        global_pid = pid + j * ctas_num
        tile_boundary_counts_impl(
            global_pid,
            sorted_data_ptr,  # in
            tile_counts_ptr,  # out
            num_tasks,
            tile_size,
        )


@libentry()
@triton.jit
def tile_counts_scan_kernel(
    tile_counts_ptr: tl.tensor,  # in
    tile_prefix_ptr: tl.tensor,  # out (exclusive prefix, group id of the tile)
    total_ptr: tl.tensor,  # out (number of groups - 1)
    num_tiles: int,
    next_power_num_tiles: tl.constexpr,
):
    # single-CTA exclusive scan over per-tile group counts.
    # next_power_of_2(P) must fit the 65 536 per-axis CTA limit, so
    # P must be ≤ 32768 (num_tasks ≤ 268_435_456). Above that we fall back
    # to a torch.cumsum on the device (see _unique_post_sort_impl).
    r = tl.arange(0, next_power_num_tiles)
    mask = r < num_tiles
    counts = tl.load(tile_counts_ptr + r, mask=mask, other=0).to(tl.int32)
    cumsum = tl.cumsum(counts, axis=0)
    tl.store(tile_prefix_ptr + r, cumsum - counts, mask=mask)
    # the last group id is total groups - 1; the scan total is that value
    total = tl.sum(tl.where(mask, counts, 0), axis=0)
    tl.store(total_ptr + tl.zeros_like(r), total, mask=(r == num_tiles - 1))


@triton.jit
def unique_scatter_impl(
    global_pid,
    sorted_data_ptr: tl.tensor,  # in
    sorted_indices_ptr: tl.tensor,  # in
    tile_prefix_ptr: tl.tensor,  # in
    data_out_ptr: tl.tensor,  # out
    inverse_indices_ptr: tl.tensor,  # out
    idx_ptr: tl.tensor,  # out
    num_tasks: int,
    tile_size: tl.constexpr,
    write_unique: tl.constexpr,
    write_inverse: tl.constexpr,
    write_idx: tl.constexpr,
):
    i0 = global_pid * tile_size + tl.arange(0, tile_size)
    mask = i0 < num_tasks
    a = tl.load(sorted_data_ptr + i0, mask=mask)
    i0_prev = tl.where(i0 > 0, i0 - 1, 0)
    b = tl.load(sorted_data_ptr + i0_prev, mask=mask)
    # a boundary is the first position of each group; position 0 is one
    ne_result = tl.where(i0 > 0, a != b, 0).to(tl.int32)
    cumsum = tl.cumsum(ne_result, axis=0)
    group_id = tl.load(tile_prefix_ptr + global_pid) + cumsum
    boundary = ((i0 == 0) | (ne_result != 0)) & mask

    if write_unique:
        tl.store(data_out_ptr + group_id, a, mask=boundary)
    if write_inverse:
        sorted_indices = tl.load(sorted_indices_ptr + i0, mask=mask)
        tl.store(inverse_indices_ptr + sorted_indices, group_id, mask=mask)
    if write_idx:
        tl.store(idx_ptr + group_id, i0, mask=boundary)


@libentry()
@triton.jit
def unique_scatter_kernel(
    sorted_data_ptr: tl.tensor,  # in
    sorted_indices_ptr: tl.tensor,  # in
    tile_prefix_ptr: tl.tensor,  # in
    data_out_ptr: tl.tensor,  # out
    inverse_indices_ptr: tl.tensor,  # out
    idx_ptr: tl.tensor,  # out
    num_tasks: int,
    tiles_per_cta: int,
    tile_size: tl.constexpr,
    write_unique: tl.constexpr,
    write_inverse: tl.constexpr,
    write_idx: tl.constexpr,
):
    pid = ext.program_id(0)
    ctas_num = ext.num_programs(0)
    # grid-stride-loop style kernel
    for j in range(0, tiles_per_cta):
        global_pid = pid + j * ctas_num
        unique_scatter_impl(
            global_pid,
            sorted_data_ptr,  # in
            sorted_indices_ptr,  # in
            tile_prefix_ptr,  # in
            data_out_ptr,  # out
            inverse_indices_ptr,  # out
            idx_ptr,  # out
            num_tasks,
            tile_size,
            write_unique,
            write_inverse,
            write_idx,
        )


def _unique_post_sort_impl(
    sorted_data: torch.Tensor,
    sorted_indices: torch.Tensor,
    return_inverse: bool,
    return_counts: bool,
):
    """Dedup + group-id + scatter over already-sorted data.

    Replaces the previous local_ne/global_cumsum pair whose lookback re-read
    O(ctas_num) metadata per tile; here the metadata is one int32 count per
    tile, scanned once, and the scatter pass never materializes the boundary
    or group-id arrays.
    """
    num_tasks = sorted_data.numel()
    tile_size = 8192
    num_tiles = triton.cdiv(num_tasks, tile_size)
    grid = (num_tiles, 1, 1)

    tile_counts = torch.empty(
        (num_tiles,), dtype=torch.int32, device=sorted_data.device
    )
    # tile_prefix / total hold group ids and (num_groups-1), both ≤ tile_counts
    # sum ≤ num_tasks-1, so they overflow int32 only when num_tasks > 2^31-1.
    # The single-CTA path caps num_tasks at 268 435 456 (well within int32);
    # for the device-only fallback we widen to int64 only when that bound is
    # exceeded — this is unreachable in practice (>= 16 GiB just for the
    # sorted index tensor) but keeps the path provably safe.
    meta_dtype = torch.int64 if num_tasks >= (1 << 31) else torch.int32
    tile_prefix = torch.empty((num_tiles,), dtype=meta_dtype, device=sorted_data.device)
    total = torch.empty((1,), dtype=meta_dtype, device=sorted_data.device)

    data_out = torch.empty_like(sorted_data)
    inverse_indices = None
    idx = None
    if return_inverse:
        inverse_indices = torch.empty_like(sorted_data, dtype=torch.int64)
    if return_counts:
        idx = torch.empty_like(sorted_data, dtype=torch.int64)

    with torch_device_fn.device(sorted_data.device.index):
        tile_boundary_counts_kernel[grid](
            sorted_data,  # in
            tile_counts,  # out
            num_tasks,
            tiles_per_cta=1,
            tile_size=tile_size,
            num_warps=8,
        )
        # Exclusive scan over per-tile group counts.
        # Single-CTA scan is bounded by next_power_of_2(P) ≤ 65 536 →
        # P ≤ 32768 (num_tasks ≤ 268_435_456). Above that fall back to
        # torch.cumsum on the device (negligible cost for a few ×10^4 values).
        if num_tiles <= _MAX_SINGLE_CTA_SCAN_TILES:
            tile_counts_scan_kernel[(1,)](
                tile_counts,  # in
                tile_prefix,  # out
                total,  # out
                num_tiles,
                next_power_num_tiles=triton.next_power_of_2(num_tiles),
                num_warps=32,
            )
        else:
            cumsum = tile_counts.cumsum(dim=0).to(meta_dtype)
            tile_prefix.narrow(0, 0, 1).zero_()
            if num_tiles > 1:
                tile_prefix.narrow(0, 1, num_tiles - 1).copy_(
                    cumsum.narrow(0, 0, num_tiles - 1)
                )
            total.copy_(cumsum.narrow(0, num_tiles - 1, 1))
        unique_scatter_kernel[grid](
            sorted_data,  # in
            sorted_indices,  # in
            tile_prefix,  # in
            data_out,  # out
            inverse_indices,  # out
            idx,  # out
            num_tasks,
            tiles_per_cta=1,
            tile_size=tile_size,
            write_unique=True,
            write_inverse=return_inverse,
            write_idx=return_counts,
            num_warps=8,
        )
        out_size = total.item() + 1
        counts = None
        if return_counts:
            idx = idx.narrow(0, 0, out_size)
            counts = torch.empty_like(idx)
            output_counts_flat_kernel[grid](
                idx,
                num_tasks,  # in
                counts,  # out
                out_size,
                tiles_per_cta=1,
                tile_size=tile_size,
                num_warps=8,
            )

    return data_out.narrow(0, 0, out_size), inverse_indices, counts


def simple_unique_flat(
    sorted_data: torch.Tensor,
    sorted_indices: torch.Tensor,
    return_inverse: bool,
    return_counts: bool,
):
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
            tile_size=triton.next_power_of_2(num_tasks),
            num_warps=8,
        )
    out_size = unique_size.item() + 1
    counts = None
    if return_counts:
        idx = idx.narrow(0, 0, out_size)
        counts = torch.empty_like(idx)
        with torch_device_fn.device(sorted_data.device.index):
            output_counts_flat_kernel[grid](
                idx,
                num_tasks,  # in
                counts,  # out
                num_tasks=out_size,
                tiles_per_cta=1,
                tile_size=triton.next_power_of_2(out_size),
                num_warps=8,
            )
    return data_out.narrow(0, 0, out_size), inverse_indices, counts


def _sorted_with_indices(flat: torch.Tensor):
    """Sort via the out= overload.

    The python-registered flag_gems sort override covers torch.sort's
    functional form but not its out= form; the out= form reaches the native
    mudnn radix sort, which is an order of magnitude faster than the python
    radix sort on this stack.
    """
    values = torch.empty_like(flat)
    indices = torch.empty(
        flat.shape,
        dtype=torch.int64,
        device=flat.device,
        memory_format=torch.contiguous_format,
    )
    torch.sort(flat, out=(values, indices))
    return values, indices


def _unique2(
    in0: torch.Tensor,
    sorted: bool = True,
    return_inverse: bool = False,
    return_counts: bool = False,
):
    logger.debug("GEMS_MTHREADS _UNIQUE2")
    if in0.numel() == 0:
        # radix_sort cannot launch on a 0-length dim; produce the empty result
        # directly (matches aten's short-circuit for dim=None unique).
        #
        # NOTE: for a multi-dim empty (e.g. shape (0, 3)) aten returns:
        #   unique : (0,)
        #   inverse: same shape as input  -> use empty_like so .view_as works
        #   counts : (0,)  -> a 1-D empty, NOT empty_like(input) whose
        #                     shape would otherwise be (0, 3)
        return (
            in0.new_empty((0,)),
            torch.empty_like(in0, dtype=torch.int64) if return_inverse else None,
            (
                torch.empty((0,), dtype=torch.int64, device=in0.device)
                if return_counts
                else None
            ),
        )
    flat = in0.ravel()
    if in0.numel() <= 8192:
        sorted_data, sorted_indices = _sorted_with_indices(flat)
        data_out, inverse_indices, counts = simple_unique_flat(
            sorted_data, sorted_indices, return_inverse, return_counts
        )
    else:
        sorted_data, sorted_indices = _sorted_with_indices(flat)
        data_out, inverse_indices, counts = _unique_post_sort_impl(
            sorted_data, sorted_indices, return_inverse, return_counts
        )
    return (
        data_out,
        inverse_indices if inverse_indices is None else inverse_indices.view_as(in0),
        counts,
    )
