import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as tle
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

    # load current and previous elements
    a = tl.load(data_ptr + i0, mask=mask)
    i0_prev = tl.where(i0 > 0, i0 - 1, 0)
    b = tl.load(data_ptr + i0_prev, mask=mask)

    # Check if element differs from previous (first element always starts a new group)
    ne_result = tl.where(i0 > 0, a != b, 1)
    cumsum = tl.cumsum(ne_result)

    # cumsum gives us 1-indexed positions, we want 0-indexed
    out_idx = cumsum - 1

    # unique_size is the last cumsum value
    unique_size_mask = i0 == num_tasks - 1
    tl.store(unique_size_ptr + tl.zeros_like(i0), cumsum, mask=unique_size_mask)

    # data_out: scatter unique values to their output positions
    # Only write when this is the first element of a consecutive group
    write_mask = ne_result.to(tl.int1) & mask
    tl.store(data_out_ptr + out_idx, a, mask=write_mask)

    # inverse_indices: each input position maps to its output position
    if return_inverse:
        tl.store(inverse_indices_ptr + i0, out_idx, mask=mask)

    # idx: store the starting position of each unique group
    if return_counts:
        tl.store(idx_ptr + out_idx, i0, mask=write_mask)


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
    mask = i0 < num_tasks

    # load idx
    idx = tl.load(idx_ptr + i0, mask=mask)

    # load idx_next
    i0_next = i0 + 1
    next_mask = i0_next < num_tasks
    idx_next = tl.load(idx_ptr + i0_next, mask=next_mask)

    # counts = next_idx - current_idx (or total - current_idx for last element)
    counts = tl.where(i0_next < num_tasks, idx_next - idx, origin_num_tasks - idx)

    # store counts
    tl.store(counts_ptr + i0, counts, mask=mask)


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
    pid = tle.program_id(0)
    ctas_num = tle.num_programs(0)
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

    # load current and previous
    a = tl.load(data_ptr + i0, mask=mask)
    b = tl.load(data_ptr + i0_prev, mask=mask)

    # compute ne_result
    ne_result = tl.where(i0 > 0, a != b, 1)

    # store ne_result
    tl.store(ne_result_ptr + i0, ne_result, mask=mask)

    # store tile_sum
    tile_sum = tl.sum(ne_result)
    tile_sum_mask = global_pid < global_ctas_num
    tl.store(tile_sum_ptr + global_pid, tile_sum, mask=tile_sum_mask)


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
    pid = tle.program_id(0)
    ctas_num = tle.num_programs(0)
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
    total,
    ne_result_ptr: tl.tensor,
    tile_sum_ptr: tl.tensor,  # in
    data_ptr: tl.tensor,  # in
    data_out_ptr: tl.tensor,
    inverse_indices_ptr: tl.tensor,
    idx_ptr: tl.tensor,  # out
    ctas_num: tl.constexpr,
    global_ctas_num: int,
    next_power_global_ctas_num: tl.constexpr,
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

    # load data
    data = tl.load(data_ptr + i0, mask=mask)

    # load tile_sum for previous tiles
    p = tl.arange(0, next_power_global_ctas_num)
    pre_tile_sum_mask = (
        (p >= global_pid - ctas_num)
        & (p < global_pid)
        & (p >= 0)
        & (p < global_ctas_num)
    )
    pre_tile_sum = tl.load(tile_sum_ptr + p, mask=pre_tile_sum_mask, other=0)

    # cumsum within tile
    total += tl.sum(pre_tile_sum)
    ne_result = tl.load(ne_result_ptr + i0, mask=mask)
    ne_result_i1 = ne_result.to(tl.int1)
    ne_result_i32 = ne_result.to(tl.int32)
    cumsum = tl.cumsum(ne_result_i32)

    # Store final tile sum for the last tile
    if global_pid == global_ctas_num - 1:
        last_tile_sum_mask = i0 == num_tasks - 1
        final_tile_sum = tl.where(last_tile_sum_mask, total + cumsum, cumsum)
        tl.store(
            tile_sum_ptr + global_pid + tl.zeros_like(r),
            final_tile_sum,
            mask=last_tile_sum_mask,
        )
    cumsum += total

    # output index (0-indexed)
    out_idx = cumsum - 1

    # data_out: scatter unique values (only first element of each consecutive group)
    tl.store(data_out_ptr + out_idx, data, mask=ne_result_i1 & mask)

    # inverse_indices: each input position maps to its output index
    if return_inverse:
        tl.store(inverse_indices_ptr + i0, out_idx, mask=mask)

    # idx: store starting position of each unique group
    if return_counts:
        tl.store(idx_ptr + out_idx, i0, mask=ne_result_i1 & mask)

    return total


@libentry()
@triton.jit
def global_cumsum_consecutive_kernel(
    ne_result_ptr: tl.tensor,
    tile_sum_ptr: tl.tensor,  # in
    data_ptr: tl.tensor,  # in
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
    return_inverse: tl.constexpr,
    return_counts: tl.constexpr,
):
    pid = tle.program_id(0)
    ctas_num = tle.num_programs(0)
    if one_tile_per_cta:
        global_cumsum_consecutive_impl(
            pid,
            0,
            ne_result_ptr,
            tile_sum_ptr,
            data_ptr,
            data_out_ptr,
            inverse_indices_ptr,
            idx_ptr,
            ctas_num,
            global_ctas_num,
            next_power_global_ctas_num,
            num_tasks,
            tile_size,
            return_inverse,
            return_counts,
        )
    else:
        total = tl.zeros([1], dtype=tl.int64)
        for j in range(0, tiles_per_cta):
            global_pid = pid + j * ctas_num
            total = global_cumsum_consecutive_impl(
                global_pid,
                total,
                ne_result_ptr,
                tile_sum_ptr,
                data_ptr,
                data_out_ptr,
                inverse_indices_ptr,
                idx_ptr,
                ctas_num,
                global_ctas_num,
                next_power_global_ctas_num,
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
        global_cumsum_consecutive_kernel[grid](
            ne_result,
            tile_sum,
            data,
            data_out,
            inverse_indices,
            idx,
            ctas_num,
            global_ctas_num,
            next_power_global_ctas_num,
            num_tasks,
            tiles_per_cta=tiles_per_cta,
            tile_size=tile_size,
            one_tile_per_cta=tiles_per_cta == 1,
            return_inverse=return_inverse,
            return_counts=return_counts,
            num_warps=num_warps,
        )
        out_size = tile_sum[-1].item()

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
    logger.debug("GEMS UNIQUE_CONSECUTIVE")

    if dim is not None:
        # For dim-wise unique_consecutive, fall back to PyTorch for now
        # This could be implemented with a more complex kernel
        return torch.unique_consecutive(
            input,
            return_inverse=return_inverse,
            return_counts=return_counts,
            dim=dim,
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
