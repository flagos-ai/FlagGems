"""
Unique operator implementation - Sophgo BM1690 TPU backend.

This file implements the TPU version of torch.unique(), returning unique values from the input tensor.

Hardware platform: Sophgo BM1690 (architecture: SG2260)
Implementation: Hybrid TPU/CPU approach.
- TPU Triton kernel: Computes adjacent element comparison results.
- CPU operations: cumsum, sort (workaround, pending TPU implementation).
- TPU scatter: Uses the project's existing TPU scatter operator.

Key limitations:
- TPU does not support int64 data type.
- TPU Top-K operation has size limitations (~32K).
- tl.cumsum has compilation issues (linalg_ext.scan shape validation failure).
"""

import torch
import triton
import triton.language as tl

from flag_gems.ops.scatter import scatter_  # Use the project's existing TPU scatter
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.libentry import libentry


def _sort_workaround(tensor):
    """
    CPU workaround for Sort operation (bypassing TPU Top-K size limitation).

    Problem: TPU's torch.sort() uses Top-K implementation with hardware limitations (axis_size < 32K).
    Solution: Execute sort on CPU, then transfer results back to TPU.

    TODO: Replace this function with TPU implementation when TPU sort operator is ready.
    Replacement: Simply change this function to `return torch.sort(tensor)`.

    Args:
        tensor: Input tensor (on TPU device).

    Returns:
        sorted_data: Sorted data (on TPU device).
        sorted_indices: Sort indices (on TPU device).
    """
    cpu_tensor = tensor.cpu()  # Transfer to CPU
    sorted_data, sorted_indices = torch.sort(cpu_tensor)  # Sort on CPU
    return sorted_data.to(tensor.device), sorted_indices.to(
        tensor.device
    )  # Transfer back to TPU


@libentry()
@triton.jit
def simple_unique_flat_kernel(
    sorted_data_ptr: tl.tensor,
    sorted_indices_ptr: tl.tensor,  # in
    ne_result_ptr: tl.tensor,  # out
    num_tasks: int,
    tile_size: tl.constexpr,
):
    """
    Simplified unique kernel - only computes adjacent element not-equal markers (executed on TPU).

    Function: Compare adjacent elements in the sorted array, marking unequal positions.
    Modification: The original version included tl.cumsum(), but due to the compiler's
        linalg_ext.scan shape validation issue, it now only computes ne_result.
        cumsum is completed on the CPU.

    Algorithm:
        For sorted array [1, 1, 2, 3, 3, 4]:
        - Compare adjacent elements: [-, 0, 1, 1, 0, 1] (0 means equal, 1 means not equal)
        - First element is marked as 0 (no previous element).

    Args:
        sorted_data_ptr: Sorted data pointer.
        sorted_indices_ptr: Sort indices pointer (unused, kept for interface compatibility).
        ne_result_ptr: Output - not-equal marker array.
        num_tasks: Total number of data items.
        tile_size: Tile size (power of 2).
    """
    # Generate index range [0, 1, 2, ..., tile_size-1]
    i0 = tl.arange(0, tile_size)
    mask = i0 < num_tasks  # Handle boundary case

    # Load current element
    a = tl.load(sorted_data_ptr + i0, mask=mask)

    # Calculate previous element index (first element's previous is set to 0)
    i0_prev = tl.where(i0 > 0, i0 - 1, 0)

    # Load previous element
    b = tl.load(sorted_data_ptr + i0_prev, mask=mask)

    # Compute not-equal markers: if i0 > 0 and a != b, mark 1, otherwise 0
    # First element (i0 == 0) is always marked 0
    ne_result = tl.where(i0 > 0, a != b, 0)

    # Store result (cumsum will be done on CPU later)
    tl.store(ne_result_ptr + i0, ne_result, mask=mask)


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
    pid = tle.program_id(0)
    ctas_num = tle.num_programs(0)
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
    pid = tle.program_id(0)
    ctas_num = tle.num_programs(0)
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
    ne_result = tl.where(i0 > 0, a != b, 0)
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
    else:
        # local_unique: scatter_(to=cumsum, sorted_data)
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
):
    pid = tle.program_id(0)
    ctas_num = tle.num_programs(0)
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
        )


@triton.jit
def global_quick_unique_flat_impl(
    global_pid,
    total,
    local_unique_ptr: tl.tensor,
    origin_idx_ptr: tl.tensor,
    tile_sum_ptr: tl.tensor,  # in
    data_out_ptr: tl.tensor,
    idx_ptr: tl.tensor,  # out
    ctas_num: int,
    global_ctas_num: int,
    next_power_global_ctas_num: tl.constexpr,
    num_tasks: int,
    tile_size: tl.constexpr,
    return_counts: tl.constexpr,
):
    r = tl.arange(0, tile_size)
    i0 = global_pid * tile_size + r
    mask = i0 < num_tasks

    # load tile_sum
    p = tl.arange(0, next_power_global_ctas_num)
    pre_tile_sum_mask = (
        (p >= global_pid - ctas_num)
        & (p < global_pid)
        & (p >= 0)
        & (p < global_ctas_num)
    )
    pre_tile_sum = tl.load(tile_sum_ptr + p, mask=pre_tile_sum_mask, other=0)
    cur_tile_sum_mask = global_pid < global_ctas_num
    cur_tile_sum = tl.load(tile_sum_ptr + global_pid, mask=cur_tile_sum_mask)

    # total
    total += tl.sum(pre_tile_sum)
    if global_pid == global_ctas_num - 1:
        last_tile_sum_mask = p == global_pid
        tl.store(tile_sum_ptr + p, total + cur_tile_sum, mask=last_tile_sum_mask)

    # idx or data_out
    tile_mask = r < cur_tile_sum
    out_offset = total + r
    if return_counts:
        # move origin_idx to idx_ptr
        origin_idx = tl.load(origin_idx_ptr + i0, mask=mask)
        tl.store(idx_ptr + out_offset, origin_idx, mask=tile_mask)
    else:
        # move local_unique to data_out_ptr
        local_unique = tl.load(local_unique_ptr + i0, mask=mask)
        tl.store(data_out_ptr + out_offset, local_unique, mask=tile_mask)

    return total


@libentry()
@triton.jit
def global_quick_unique_flat_kernel(
    local_unique_ptr: tl.tensor,
    origin_idx_ptr: tl.tensor,
    tile_sum_ptr: tl.tensor,  # in
    data_out_ptr: tl.tensor,
    idx_ptr: tl.tensor,  # out
    ctas_num: int,
    global_ctas_num: int,
    next_power_global_ctas_num: tl.constexpr,
    num_tasks: int,
    tiles_per_cta: int,
    tile_size: tl.constexpr,
    one_tile_per_cta: tl.constexpr,
    return_counts: tl.constexpr,
):
    pid = tle.program_id(0)
    ctas_num = tle.num_programs(0)
    if one_tile_per_cta:  # monolitic kernel style
        global_quick_unique_flat_impl(
            pid,
            0,
            local_unique_ptr,
            origin_idx_ptr,
            tile_sum_ptr,  # in
            data_out_ptr,
            idx_ptr,  # out
            ctas_num,
            global_ctas_num,
            next_power_global_ctas_num,
            num_tasks,
            tile_size,
            return_counts,
        )
    else:  # grid-stride-loop style kernel
        total = tl.zeros([1], dtype=tl.int32)
        for j in range(0, tiles_per_cta):
            global_pid = pid + j * ctas_num
            total = global_quick_unique_flat_impl(
                global_pid,
                total,
                local_unique_ptr,
                origin_idx_ptr,
                tile_sum_ptr,  # in
                data_out_ptr,
                idx_ptr,  # out
                ctas_num,
                global_ctas_num,
                next_power_global_ctas_num,
                num_tasks,
                tile_size,
                return_counts,
            )


def sorted_quick_unique_flat(sorted_data: torch.Tensor, return_counts: bool):
    num_tasks = sorted_data.numel()
    next_power_num_tasks = triton.next_power_of_2(num_tasks)
    tile_size = min(8192, next_power_num_tasks)
    global_ctas_num = triton.cdiv(num_tasks, tile_size)
    if global_ctas_num <= 8192:
        tile_size = max(
            32, min(triton.next_power_of_2(global_ctas_num), next_power_num_tasks)
        )
        global_ctas_num = triton.cdiv(num_tasks, tile_size)
    next_power_global_ctas_num = triton.next_power_of_2(global_ctas_num)
    ctas_num = global_ctas_num if global_ctas_num < 65536 else 2048
    tiles_per_cta = triton.cdiv(num_tasks, tile_size * ctas_num)
    num_warps = 8 if tiles_per_cta == 1 else 32
    grid = (ctas_num, 1, 1)

    # allocate tensor
    if return_counts:
        local_unique = None
        origin_idx = torch.empty_like(sorted_data, dtype=torch.int32)
        idx = torch.empty_like(origin_idx)
    else:
        local_unique = torch.empty_like(sorted_data)
        origin_idx = None
        idx = None
        counts = None
    tile_sum = torch.empty(
        (global_ctas_num,), dtype=torch.int32, device=sorted_data.device
    )
    data_out = None
    if not return_counts:
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
        )
        global_quick_unique_flat_kernel[grid](
            local_unique,
            origin_idx,
            tile_sum,  # in
            data_out,
            idx,  # out
            ctas_num,
            global_ctas_num,
            next_power_global_ctas_num,
            num_tasks,
            tiles_per_cta=tiles_per_cta,
            tile_size=tile_size,
            one_tile_per_cta=tiles_per_cta == 1,
            return_counts=return_counts,
            num_warps=num_warps,
        )
        out_size = tile_sum[-1].item()
        if return_counts:
            data_out = torch.empty(
                (out_size,), dtype=sorted_data.dtype, device=sorted_data.device
            )
            idx = idx[:out_size]
            counts = origin_idx[:out_size]
            quick_output_flat_kernel[grid](
                sorted_data,
                idx,
                num_tasks,  # in
                data_out,
                counts,  # out
                out_size,
                tiles_per_cta,
                tile_size,
                num_warps=num_warps,
            )

    if return_counts:
        return data_out, None, counts
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
):
    r = tl.arange(0, tile_size)
    i0 = global_pid * tile_size + r
    mask = i0 < num_tasks
    i0_prev = tl.where(i0 > 0, i0 - 1, 0)

    # load
    a = tl.load(sorted_data_ptr + i0, mask=mask)
    b = tl.load(sorted_data_ptr + i0_prev, mask=mask)

    # compute
    ne_result = tl.where(i0 > 0, a != b, 0)

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
):
    pid = tle.program_id(0)
    ctas_num = tle.num_programs(0)
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
        )


@triton.jit
def global_cumsum_flat_impl(
    global_pid,
    total,
    ne_result_ptr: tl.tensor,
    tile_sum_ptr: tl.tensor,  # in
    sorted_data_ptr: tl.tensor,
    sorted_indices_ptr: tl.tensor,  # in
    data_out_ptr: tl.tensor,
    inverse_indices_ptr: tl.tensor,
    idx_ptr: tl.tensor,  # out
    ctas_num: tl.constexpr,
    global_ctas_num: int,
    next_power_global_ctas_num: tl.constexpr,
    num_tasks: int,
    tile_size: tl.constexpr,
    return_counts: tl.constexpr,
):
    offset = global_pid * tile_size
    r = tl.arange(0, tile_size)
    i0 = offset + r
    mask = i0 < num_tasks

    # load sorted_data, sorted_indices
    sorted_data = tl.load(sorted_data_ptr + i0, mask=mask)
    sorted_indices = tl.load(sorted_indices_ptr + i0, mask=mask)

    # load tile_sum
    p = tl.arange(0, next_power_global_ctas_num)
    pre_tile_sum_mask = (
        (p >= global_pid - ctas_num)
        & (p < global_pid)
        & (p >= 0)
        & (p < global_ctas_num)
    )
    pre_tile_sum = tl.load(tile_sum_ptr + p, mask=pre_tile_sum_mask, other=0)

    # cumsum
    total += tl.sum(pre_tile_sum)
    ne_result = tl.load(ne_result_ptr + i0, mask=mask)
    ne_result_i1 = ne_result.to(tl.int1)
    ne_result = ne_result.to(tl.int32)
    cumsum = tl.cumsum(ne_result)

    # tile_sum
    if global_pid == global_ctas_num - 1:
        last_tile_sum_mask = i0 == num_tasks - 1
        tile_sum = tl.where(last_tile_sum_mask, total + cumsum, cumsum)
        tl.store(
            tile_sum_ptr + global_pid + tl.zeros_like(r),
            tile_sum,
            mask=last_tile_sum_mask,
        )
    cumsum += total

    # data_out: scatter_(to=cumsum, sorted_data)
    tl.store(data_out_ptr + cumsum, sorted_data, mask=mask)

    # inverse_indices: scatter_(to=sorted_indices, cumsum)
    tl.store(inverse_indices_ptr + sorted_indices, cumsum, mask=mask)

    # idx
    if return_counts:
        idx_mask = ((i0 == 0) | ne_result_i1) & mask
        tl.store(idx_ptr + cumsum, i0, mask=idx_mask)

    return total


@libentry()
@triton.jit
def global_cumsum_flat_kernel(
    ne_result_ptr: tl.tensor,
    tile_sum_ptr: tl.tensor,  # in
    sorted_data_ptr: tl.tensor,
    sorted_indices_ptr: tl.tensor,  # in
    data_out_ptr: tl.tensor,
    inverse_indices_ptr: tl.tensor,
    idx_ptr: tl.tensor,  # out
    ctas_num: int,
    global_ctas_num: int,
    next_power_global_ctas_num: tl.constexpr,
    num_tasks: int,
    tiles_per_cta: int,
    tile_size: tl.constexpr,
    one_tile_per_cta: tl.constexpr,
    return_counts: tl.constexpr,
):
    pid = tle.program_id(0)
    ctas_num = tle.num_programs(0)
    if one_tile_per_cta:  # monolitic kernel style
        global_cumsum_flat_impl(
            pid,
            0,
            ne_result_ptr,
            tile_sum_ptr,  # in
            sorted_data_ptr,
            sorted_indices_ptr,  # in
            data_out_ptr,
            inverse_indices_ptr,
            idx_ptr,  # out
            ctas_num,
            global_ctas_num,
            next_power_global_ctas_num,
            num_tasks,
            tile_size,
            return_counts,
        )
    else:  # grid-stride-loop style kernel
        total = tl.zeros([1], dtype=tl.int32)
        for j in range(0, tiles_per_cta):
            global_pid = pid + j * ctas_num
            total = global_cumsum_flat_impl(
                global_pid,
                total,
                ne_result_ptr,
                tile_sum_ptr,  # in
                sorted_data_ptr,
                sorted_indices_ptr,  # in
                data_out_ptr,
                inverse_indices_ptr,
                idx_ptr,  # out
                ctas_num,
                global_ctas_num,
                next_power_global_ctas_num,
                num_tasks,
                tile_size,
                return_counts,
            )


def sorted_indices_unique_flat(
    sorted_data: torch.Tensor, sorted_indices: torch.Tensor, return_counts: bool
):
    num_tasks = sorted_data.numel()
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

    # allocate tensor
    ne_result = torch.empty_like(sorted_data, dtype=torch.bool)
    tile_sum = torch.empty(
        (global_ctas_num,), dtype=torch.int32, device=sorted_data.device
    )
    data_out = torch.empty_like(sorted_data)
    inverse_indices = torch.empty_like(sorted_data, dtype=torch.int32)
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
        )
        global_cumsum_flat_kernel[grid](
            ne_result,
            tile_sum,  # in
            sorted_data,
            sorted_indices,  # in
            data_out,
            inverse_indices,
            idx,  # out
            ctas_num,
            global_ctas_num,
            next_power_global_ctas_num,
            num_tasks,
            tiles_per_cta=tiles_per_cta,
            tile_size=tile_size,
            one_tile_per_cta=tiles_per_cta == 1,
            return_counts=return_counts,
            num_warps=num_warps,
        )
        out_size = tile_sum[-1].item() + 1
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
):
    """
    Simplified unique implementation - hybrid TPU/CPU approach (main implementation currently used).

    Algorithm flow:
        1. [TPU] Compute adjacent element not-equal markers (ne_result).
        2. [CPU] Cumulative sum on ne_result (cumsum) - workaround.
        3. [TPU] Reorganize data using scatter operation.
        4. [TPU] Compute inverse indices and counts (if needed).

    Example:
        Input: sorted_data = [1, 1, 2, 3, 3, 4]
        Step 1: ne_result = [0, 0, 1, 1, 0, 1]
        Step 2: cumsum = [0, 0, 1, 2, 2, 3]
        Step 3: data_out = [1, 2, 3, 4] (via scatter operation)

    Performance notes:
        - cumsum on CPU increases data transfer overhead.
        - However, compared to the overall algorithm, cumsum is lightweight, impact is controllable.
        - Full TPU implementation can be restored once the compiler fixes tl.cumsum issues.

    Args:
        sorted_data: Sorted data tensor.
        sorted_indices: Sort indices tensor.
        return_inverse: Whether to return inverse indices.
        return_counts: Whether to return counts.

    Returns:
        data_out: Unique value array.
        inverse_indices: Inverse index array (if return_inverse=True).
        counts: Count array (if return_counts=True).
    """
    num_tasks = sorted_data.numel()
    grid = (1, 1, 1)

    # ========== Step 1: Compute ne_result on TPU ==========
    # Allocate output tensor
    ne_result = torch.empty_like(sorted_data, dtype=torch.int32)

    # Launch TPU kernel to compute adjacent element not-equal markers
    with torch_device_fn.device(sorted_data.device.index):
        simple_unique_flat_kernel[grid](
            sorted_data,
            sorted_indices,
            ne_result,  # out
            num_tasks,
            tile_size=triton.next_power_of_2(num_tasks),
            num_warps=8,
        )

    # ========== Step 2: Compute cumsum on CPU (workaround) ==========
    # Problem: tl.cumsum has compilation error (linalg_ext.scan shape validation failure).
    # Solution: Compute cumsum on CPU, then transfer back to TPU.
    # TODO: Move this part back to TPU kernel when the compiler is fixed.
    ne_result_cpu = ne_result.cpu()  # TPU → CPU
    cumsum_cpu = torch.cumsum(ne_result_cpu, dim=0)  # Cumulative sum on CPU
    cumsum = cumsum_cpu.to(sorted_data.device)  # CPU → TPU

    # ========== Step 3: Compute unique value count ==========
    # The last value of cumsum + 1 is the number of unique values.
    # Example: cumsum = [0, 0, 1, 2, 2, 3] → out_size = 3 + 1 = 4
    out_size = cumsum[-1].item() + 1

    # ========== Step 4: Reorganize data using TPU scatter ==========
    # Use the project's existing TPU scatter operator (supports int32 indices).
    # scatter_(data_out, dim=0, index=cumsum, src=sorted_data)
    # Place elements from sorted_data into data_out based on cumsum indices.
    data_out = torch.empty(
        (out_size,), dtype=sorted_data.dtype, device=sorted_data.device
    )
    scatter_(data_out, 0, cumsum, sorted_data)

    # ========== Step 5: Handle inverse indices (optional) ==========
    # Inverse indices satisfy: data_out[inverse_indices] == sorted_data.
    # Via scatter operation: inverse_indices[sorted_indices] = cumsum.
    inverse_indices = None
    if return_inverse:
        inverse_indices = torch.empty_like(sorted_data, dtype=torch.int32)
        scatter_(inverse_indices, 0, sorted_indices, cumsum)

    # ========== Step 6: Handle counts (optional) ==========
    # Compute the occurrence count for each unique value.
    counts = None
    if return_counts:
        # Find all starting positions of unique values.
        ne_mask = ne_result.bool()
        ne_mask[
            0
        ] = True  # First element is always a starting position of a unique value
        idx = torch.arange(num_tasks, device=sorted_data.device, dtype=torch.int32)[
            ne_mask
        ]

        # Compute difference between adjacent starting positions, which gives the count per unique value.
        counts = torch.empty((out_size,), dtype=torch.int32, device=sorted_data.device)
        idx_next = torch.cat(
            [
                idx[1:],
                torch.tensor([num_tasks], device=sorted_data.device, dtype=torch.int32),
            ]
        )
        counts = idx_next - idx

    return data_out, inverse_indices, counts


def _unique2(
    in0: torch.Tensor,
    sorted: bool = True,
    return_inverse: bool = False,
    return_counts: bool = False,
):
    """
    Main entry function for the Unique operator (corresponds to torch.unique).

    Function: Returns unique values from the input tensor, optionally returning inverse indices and counts.

    Implementation strategy:
        1. Use CPU sort to sort input (workaround).
        2. Call simple_unique_flat to compute unique values.
        3. Reshape inverse indices to original input shape.

    Modification notes:
        - Original implementation selected different algorithm paths based on data size.
        - Currently forces use of simple_unique_flat to avoid tl.cumsum compilation issues in other paths.
        - Multi-path optimization can be restored once the compiler is fixed.

    Args:
        in0: Input tensor (any shape).
        sorted: Whether to sort output (currently always True, since sorting algorithm is used).
        return_inverse: Whether to return inverse indices.
            - Inverse indices satisfy: unique_values[inverse_indices] == in0.ravel()
        return_counts: Whether to return the occurrence count for each unique value.

    Returns:
        tuple: (data_out, inverse_indices, counts)
            - data_out: Unique value tensor (1D).
            - inverse_indices: Inverse index tensor (same shape as input), None if return_inverse=False.
            - counts: Count tensor (1D), None if return_counts=False.

    Example:
        >>> input = torch.tensor([[1, 3, 2], [3, 1, 4]])
        >>> unique, inverse, counts = _unique2(input, return_inverse=True, return_counts=True)
        >>> unique
        tensor([1, 2, 3, 4])
        >>> inverse
        tensor([[0, 2, 1],
                [2, 0, 3]])
        >>> counts
        tensor([2, 1, 2, 1])
    """
    # ========== Step 1: Sort input data ==========
    # Use CPU sort (workaround, bypassing TPU Top-K size limitation).
    # When TPU sort is implemented, replace _sort_workaround with torch.sort.
    sorted_data, sorted_indices = _sort_workaround(in0.ravel())

    # ========== Step 2: Compute unique ==========
    # Force use of simple_unique_flat (avoiding tl.cumsum compilation issues in other functions).
    # Original implementation selected different algorithm paths based on num_tasks:
    #   - num_tasks <= 1024: simple_unique_flat
    #   - num_tasks > 1024: sorted_indices_unique_flat (contains tl.cumsum, causes compilation failure)
    # Current modification: all cases use simple_unique_flat
    data_out, inverse_indices, counts = simple_unique_flat(
        sorted_data, sorted_indices, return_inverse, return_counts
    )

    # ========== Step 3: Reshape inverse indices to original input shape ==========
    # inverse_indices is currently 1D (corresponding to in0.ravel()).
    # Need to reshape to the original input shape.
    return (
        data_out,
        inverse_indices if inverse_indices is None else inverse_indices.view_as(in0),
        counts,
    )
