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


@libentry()
@triton.jit
def simple_unique_consecutive_flat_kernel(
    data_ptr: tl.tensor,  # in
    data_out_ptr: tl.tensor,
    inverse_indices_ptr: tl.tensor,
    idx_ptr: tl.tensor,
    unique_size_ptr: tl.tensor,  # out
    return_inverse: tl.constexpr,
    return_counts: tl.constexpr,
    num_tasks: int,
    tile_size: tl.constexpr,
):
    """Simple kernel for small inputs that fits in a single tile."""
    i0 = tl.arange(0, tile_size)
    mask = i0 < num_tasks
    # XPU: clamp addresses into range so masked-out padding lanes (tile_size is
    # rounded up to a power of two) cannot cause an OOB GM access -> NOC hang.
    i0_safe = tl.minimum(i0, num_tasks - 1)

    # load current and previous elements
    a = tl.load(data_ptr + i0_safe, mask=mask)
    i0_prev = tl.where(i0 > 0, i0 - 1, 0)
    i0_prev_safe = tl.minimum(i0_prev, num_tasks - 1)
    b = tl.load(data_ptr + i0_prev_safe, mask=mask)

    # Check if element differs from previous (first element always starts a new group)
    ne_result = tl.where(i0 > 0, a != b, 1)
    # padding lanes must not inflate the cumulative count
    ne_result = tl.where(mask, ne_result, 0)
    cumsum = tl.cumsum(ne_result)

    # cumsum gives us 1-indexed positions, we want 0-indexed
    out_idx = cumsum - 1
    out_idx_safe = tl.minimum(tl.maximum(out_idx, 0), num_tasks - 1)

    # unique_size is the total group count = final cumsum value. cumsum is
    # monotonic non-decreasing and padding lanes (ne zeroed) do not increase it,
    # so tl.max over the tile equals the final count. XPU: a masked scalar store
    # to a single address miscompiles (writes the wrong lane when the selected
    # lane is followed by padding), so broadcast the reduced total to every lane
    # and store mask-free -- all lanes write the same value to unique_size[0].
    total = tl.max(cumsum)
    tl.store(unique_size_ptr + tl.zeros_like(i0), tl.zeros_like(i0) + total)

    # data_out: scatter unique values to their output positions. Only the first
    # element of each consecutive group writes; XPU drops the store mask, so
    # redirect every non-start lane (and padding) to a dump slot at
    # data_out[num_tasks-1]. Within a group all lanes carry the same value, and
    # the dump slot lies outside data_out[:out_size] whenever out_size <
    # num_tasks, so the redirect is value-safe.
    write_mask = ne_result.to(tl.int1) & mask
    data_addr = tl.where(write_mask, out_idx_safe, num_tasks - 1)
    tl.store(data_out_ptr + data_addr, a)

    # inverse_indices: each input position maps to its output position
    if return_inverse:
        tl.store(inverse_indices_ptr + i0_safe, out_idx)

    # idx: store the starting position of each unique group (mask-free redirect,
    # same dump-slot scheme as data_out)
    if return_counts:
        idx_addr = tl.where(write_mask, out_idx_safe, num_tasks - 1)
        tl.store(idx_ptr + idx_addr, i0)


@triton.jit
def output_counts_impl(
    global_pid,
    idx_ptr: tl.tensor,
    origin_num_tasks: int,  # in
    counts_ptr: tl.tensor,  # out
    num_tasks: int,
    tile_size: tl.constexpr,
):
    """Compute counts from idx positions."""
    r = tl.arange(0, tile_size)
    i0 = global_pid * tile_size + r
    # XPU: clamp load/store addresses into [0, num_tasks-1] to avoid an OOB GM
    # access (NOC hang) on masked-out tail lanes.
    i0_safe = tl.minimum(i0, num_tasks - 1)

    # XPU: masked loads lower to a contiguous DMA that ignores the mask, and a
    # masked store can drop the last valid lane. Use in-range clamped addresses
    # with mask-free loads/stores. Padding lanes clamp to num_tasks-1 and compute
    # exactly the last group's count (origin - idx[last]), so a mask-free store
    # to the clamped address writes the correct value even for padding lanes.
    idx = tl.load(idx_ptr + i0_safe)

    # load idx_next (clamped, mask-free)
    i0_next = i0 + 1
    i0_next_safe = tl.minimum(i0_next, num_tasks - 1)
    idx_next = tl.load(idx_ptr + i0_next_safe)

    # counts = next_idx - current_idx (or total - current_idx for last element)
    counts = tl.where(i0_next < num_tasks, idx_next - idx, origin_num_tasks - idx)

    # store counts (mask-free; padding lanes redirect to num_tasks-1 with the
    # correct last-group value)
    tl.store(counts_ptr + i0_safe, counts)


@libentry()
@triton.jit
def output_counts_kernel(
    idx_ptr: tl.tensor,
    origin_num_tasks: int,  # in
    counts_ptr: tl.tensor,  # out
    num_tasks: int,
    tiles_per_cta: int,
    tile_size: tl.constexpr,
):
    pid = ext.program_id(0)
    ctas_num = ext.num_programs(0)
    for j in range(0, tiles_per_cta):
        global_pid = pid + j * ctas_num
        output_counts_impl(
            global_pid,
            idx_ptr,
            origin_num_tasks,
            counts_ptr,
            num_tasks,
            tile_size,
        )


@triton.jit
def local_ne_consecutive_impl(
    global_pid,
    data_ptr: tl.tensor,  # in
    ne_result_ptr: tl.tensor,
    tile_sum_ptr: tl.tensor,  # out
    global_ctas_num: int,
    num_tasks: int,
    tile_size: tl.constexpr,
):
    """Compute ne_result (whether each element differs from previous) for a tile."""
    r = tl.arange(0, tile_size)
    i0 = global_pid * tile_size + r
    mask = i0 < num_tasks
    i0_prev = tl.where(i0 > 0, i0 - 1, 0)
    # XPU: clamp addresses into [0, num_tasks-1] so masked-out tail lanes cannot
    # trigger an OOB GM read (NOC hang) if the mask is dropped during lowering.
    i0_safe = tl.minimum(i0, num_tasks - 1)
    i0_prev_safe = tl.minimum(i0_prev, num_tasks - 1)

    # load current and previous
    a = tl.load(data_ptr + i0_safe, mask=mask)
    b = tl.load(data_ptr + i0_prev_safe, mask=mask)

    # compute ne_result
    ne_result = tl.where(i0 > 0, a != b, 1)
    # XPU: the clamped masked loads may still be lowered to a contiguous DMA of
    # tile_size that reads past num_tasks, so padding lanes hold garbage that can
    # compare not-equal. Zero the padding lanes in registers before reducing,
    # otherwise tl.sum inflates the per-tile unique count.
    ne_result = tl.where(mask, ne_result, 0)

    # store ne_result
    tl.store(ne_result_ptr + i0_safe, ne_result, mask=mask)

    # store tile_sum
    tile_sum = tl.sum(ne_result)
    tile_sum_mask = global_pid < global_ctas_num
    global_pid_safe = tl.minimum(global_pid, global_ctas_num - 1)
    tl.store(tile_sum_ptr + global_pid_safe, tile_sum, mask=tile_sum_mask)


@libentry()
@triton.jit
def local_ne_consecutive_kernel(
    data_ptr: tl.tensor,  # in
    ne_result_ptr: tl.tensor,
    tile_sum_ptr: tl.tensor,  # out
    global_ctas_num: int,
    num_tasks: int,
    tiles_per_cta: int,
    tile_size: tl.constexpr,
):
    pid = ext.program_id(0)
    ctas_num = ext.num_programs(0)
    for j in range(0, tiles_per_cta):
        global_pid = pid + j * ctas_num
        local_ne_consecutive_impl(
            global_pid,
            data_ptr,
            ne_result_ptr,
            tile_sum_ptr,
            global_ctas_num,
            num_tasks,
            tile_size,
        )


@triton.jit
def global_cumsum_consecutive_impl(
    global_pid,
    ne_result_ptr: tl.tensor,
    tile_base_ptr: tl.tensor,  # in: exclusive prefix sum of per-tile counts
    data_ptr: tl.tensor,  # in
    data_out_ptr: tl.tensor,
    inverse_indices_ptr: tl.tensor,
    idx_ptr: tl.tensor,  # out
    global_ctas_num: int,
    num_tasks: int,
    tile_size: tl.constexpr,
    return_inverse: tl.constexpr,
    return_counts: tl.constexpr,
):
    """Compute global cumsum and scatter outputs."""
    offset = global_pid * tile_size
    r = tl.arange(0, tile_size)
    i0 = offset + r
    mask = i0 < num_tasks
    # XPU: masked loads may be lowered to a contiguous DMA that ignores the
    # mask, so an offset past the buffer becomes an OOB GM read -> NOC IDLE
    # (soft reset). Clamp the address into range; the mask still selects values.
    i0_safe = tl.minimum(i0, num_tasks - 1)

    # load data
    data = tl.load(data_ptr + i0_safe, mask=mask)

    # This tile's output base offset = number of unique elements in all previous
    # tiles. It is precomputed on the host as an exclusive prefix sum of the
    # per-tile counts. The original in-kernel windowed masked reduction over
    # tile_sum miscompiles on XPU for the last program (the masked load becomes
    # a contiguous DMA that runs off the tile_sum buffer, corrupting `total`),
    # so we load the ready-made base with a single in-range scalar load instead.
    total = tl.load(tile_base_ptr + global_pid)

    ne_result = tl.load(ne_result_ptr + i0_safe, mask=mask)
    # XPU: the load mask may be dropped during lowering, so padding lanes read
    # the clamped last valid element instead of 0. Zero them in registers so the
    # cumsum is not inflated by padding.
    ne_result_i32 = tl.where(mask, ne_result.to(tl.int32), 0)
    ne_result_i1 = ne_result_i32.to(tl.int1)
    cumsum = tl.cumsum(ne_result_i32)
    cumsum += total

    # output index (0-indexed)
    out_idx = cumsum - 1
    # XPU: scatter address must stay in-bounds even for masked-out lanes,
    # otherwise a data-dependent OOB write hangs the NOC. Clamp into [0, N-1];
    # the mask still gates which lanes actually write.
    out_idx_safe = tl.minimum(tl.maximum(out_idx, 0), num_tasks - 1)

    # data_out: scatter unique values (only first element of each consecutive group)
    tl.store(data_out_ptr + out_idx_safe, data, mask=ne_result_i1 & mask)

    # inverse_indices: each input position maps to its output index
    if return_inverse:
        tl.store(inverse_indices_ptr + i0_safe, out_idx, mask=mask)

    # idx: store starting position of each unique group.
    # XPU: the store mask is dropped during lowering, so every lane writes. All
    # lanes of a consecutive group share the start lane's out_idx (mid-group
    # lanes have the same cumsum), so a masked store degenerates to last-writer-
    # wins and records the LAST group position instead of the first. Redirect
    # every non-start lane (and padding) to a dump slot at idx[num_tasks - 1],
    # which lies outside the idx[:out_size] slice whenever out_size < num_tasks;
    # when out_size == num_tasks every lane is a start lane so no redirect occurs.
    # Address arithmetic is respected by the backend, unlike the store mask.
    if return_counts:
        write_start = ne_result_i1 & mask
        idx_store_addr = tl.where(write_start, out_idx_safe, num_tasks - 1)
        tl.store(idx_ptr + idx_store_addr, i0)


@libentry()
@triton.jit
def global_cumsum_consecutive_kernel(
    ne_result_ptr: tl.tensor,
    tile_base_ptr: tl.tensor,  # in: exclusive prefix sum of per-tile counts
    data_ptr: tl.tensor,  # in
    data_out_ptr: tl.tensor,
    inverse_indices_ptr: tl.tensor,
    idx_ptr: tl.tensor,  # out
    global_ctas_num: int,
    num_tasks: int,
    tiles_per_cta: int,
    tile_size: tl.constexpr,
    one_tile_per_cta: tl.constexpr,
    return_inverse: tl.constexpr,
    return_counts: tl.constexpr,
):
    pid = ext.program_id(0)
    ctas_num = ext.num_programs(0)
    if one_tile_per_cta:
        global_cumsum_consecutive_impl(
            pid,
            ne_result_ptr,
            tile_base_ptr,
            data_ptr,
            data_out_ptr,
            inverse_indices_ptr,
            idx_ptr,
            global_ctas_num,
            num_tasks,
            tile_size,
            return_inverse,
            return_counts,
        )
    else:
        for j in range(0, tiles_per_cta):
            global_pid = pid + j * ctas_num
            global_cumsum_consecutive_impl(
                global_pid,
                ne_result_ptr,
                tile_base_ptr,
                data_ptr,
                data_out_ptr,
                inverse_indices_ptr,
                idx_ptr,
                global_ctas_num,
                num_tasks,
                tile_size,
                return_inverse,
                return_counts,
            )


def simple_unique_consecutive_flat(
    data: torch.Tensor,
    return_inverse: bool,
    return_counts: bool,
):
    """Handle small inputs with a single kernel launch."""
    num_tasks = data.numel()
    grid = (1, 1, 1)

    # allocate tensors
    data_out = torch.empty_like(data)
    inverse_indices = (
        torch.empty(num_tasks, dtype=torch.int64, device=data.device)
        if return_inverse
        else None
    )
    idx = (
        torch.empty(num_tasks, dtype=torch.int64, device=data.device)
        if return_counts
        else None
    )
    unique_size = torch.empty([1], dtype=torch.int64, device=data.device)

    # launch kernel
    with torch_device_fn.device(data.device.index):
        simple_unique_consecutive_flat_kernel[grid](
            data,
            data_out,
            inverse_indices,
            idx,
            unique_size,
            return_inverse,
            return_counts,
            num_tasks,
            tile_size=triton.next_power_of_2(num_tasks),
            num_warps=8,
        )

    out_size = unique_size.item()
    counts = None
    if return_counts:
        idx = idx[:out_size]
        counts = torch.empty_like(idx)
        with torch_device_fn.device(data.device.index):
            output_counts_kernel[grid](
                idx,
                num_tasks,
                counts,
                num_tasks=out_size,
                tiles_per_cta=1,
                tile_size=triton.next_power_of_2(out_size),
                num_warps=8,
            )

    return data_out[:out_size], inverse_indices, counts


def large_unique_consecutive_flat(
    data: torch.Tensor,
    return_inverse: bool,
    return_counts: bool,
):
    """Handle larger inputs with multi-kernel approach."""
    num_tasks = data.numel()
    next_power_num_tasks = triton.next_power_of_2(num_tasks)
    tile_size = min(8192, next_power_num_tasks)
    global_ctas_num = triton.cdiv(num_tasks, tile_size)

    if global_ctas_num <= 8192:
        min_tile_size = 512 if global_ctas_num > 32 else 256
        tile_size = max(
            min_tile_size,
            min(triton.next_power_of_2(global_ctas_num), next_power_num_tasks),
        )
        global_ctas_num = triton.cdiv(num_tasks, tile_size)

    next_power_global_ctas_num = triton.next_power_of_2(global_ctas_num)
    ctas_num = global_ctas_num if global_ctas_num < 32768 else 8192
    tiles_per_cta = triton.cdiv(num_tasks, tile_size * ctas_num)
    num_warps = 8 if tiles_per_cta == 1 else 32
    grid = (ctas_num, 1, 1)

    # allocate tensors
    ne_result = torch.empty(num_tasks, dtype=torch.bool, device=data.device)
    tile_sum = torch.empty(global_ctas_num, dtype=torch.int64, device=data.device)
    data_out = torch.empty_like(data)
    inverse_indices = (
        torch.empty(num_tasks, dtype=torch.int64, device=data.device)
        if return_inverse
        else None
    )
    idx = (
        torch.empty(num_tasks, dtype=torch.int64, device=data.device)
        if return_counts
        else None
    )

    # launch kernels
    with torch_device_fn.device(data.device.index):
        local_ne_consecutive_kernel[grid](
            data,
            ne_result,
            tile_sum,
            global_ctas_num,
            num_tasks,
            tiles_per_cta=tiles_per_cta,
            tile_size=tile_size,
            num_warps=num_warps,
        )
        # Exclusive prefix sum of the per-tile unique counts: tile_base[t] is the
        # output offset for tile t (number of uniques in tiles 0..t-1). Only the
        # last (padded) tile's own count can be inflated by OOB padding lanes,
        # and an exclusive prefix never uses the last element as a base, so every
        # base is exact. Computing it on the host avoids the in-kernel windowed
        # masked reduction that miscompiles on XPU.
        tile_base = torch.zeros_like(tile_sum)
        if global_ctas_num > 1:
            tile_base[1:] = torch.cumsum(tile_sum[:-1], dim=0)
        global_cumsum_consecutive_kernel[grid](
            ne_result,
            tile_base,
            data,
            data_out,
            inverse_indices,
            idx,
            global_ctas_num,
            num_tasks,
            tiles_per_cta=tiles_per_cta,
            tile_size=tile_size,
            one_tile_per_cta=tiles_per_cta == 1,
            return_inverse=return_inverse,
            return_counts=return_counts,
            num_warps=num_warps,
        )
        # out_size = total number of unique elements. The kernel's per-tile
        # tl.sum over a partial (padded) tile can fold in OOB padding lanes on
        # XPU (the masked load lowers to a contiguous DMA past num_tasks), so we
        # sum the per-element ne_result (which is written correctly for every
        # in-range element) on the host instead.
        out_size = int(ne_result.sum().item())

        counts = None
        if return_counts:
            idx = idx[:out_size]
            counts = torch.empty_like(idx)
            output_counts_kernel[grid](
                idx,
                num_tasks,
                counts,
                out_size,
                tiles_per_cta,
                tile_size,
                num_warps=num_warps,
            )

    return data_out[:out_size], inverse_indices, counts


def unique_consecutive(
    input: torch.Tensor,
    return_inverse: bool = False,
    return_counts: bool = False,
    dim: int = None,
):
    """
    Eliminates all but the first element from every consecutive group of equivalent elements.

    Args:
        input: the input tensor
        return_inverse: Whether to return inverse indices
        return_counts: Whether to return counts for each unique element
        dim: the dimension to apply unique. If None, the unique of the flattened input is returned.

    Returns:
        (Tensor, Tensor (optional), Tensor (optional)): output, inverse_indices, counts
    """
    logger.debug("GEMS_KUNLUNXIN UNIQUE_CONSECUTIVE")

    if dim is not None:
        raise NotImplementedError(
            "Kunlunxin unique_consecutive currently supports only dim=None"
        )

    # Flatten input for the None dim case
    flat_input = input.ravel()
    num_tasks = flat_input.numel()

    if num_tasks == 0:
        # Handle empty input
        output = torch.empty(0, dtype=input.dtype, device=input.device)
        inverse_indices = (
            torch.empty(0, dtype=torch.int64, device=input.device)
            if return_inverse
            else None
        )
        counts = (
            torch.empty(0, dtype=torch.int64, device=input.device)
            if return_counts
            else None
        )
        return output, inverse_indices, counts

    # Choose algorithm based on input size
    if num_tasks <= 8192:
        output, inverse_indices, counts = simple_unique_consecutive_flat(
            flat_input, return_inverse, return_counts
        )
    else:
        output, inverse_indices, counts = large_unique_consecutive_flat(
            flat_input, return_inverse, return_counts
        )

    # Reshape inverse_indices to match input shape
    if inverse_indices is not None:
        inverse_indices = inverse_indices.view_as(input)

    return output, inverse_indices, counts
