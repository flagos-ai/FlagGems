import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.libentry import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def simple_unique_flat_kernel(
    sorted_data_ptr: tl.tensor,
    sorted_indices_ptr: tl.tensor,
    data_out_ptr: tl.tensor,
    inverse_indices_ptr: tl.tensor,
    idx_ptr: tl.tensor,
    unique_size_ptr: tl.tensor,
    return_inverse: tl.constexpr,
    return_counts: tl.constexpr,
    num_tasks: int,
    tile_size: tl.constexpr,
):
    i0 = tl.arange(0, tile_size)
    mask = i0 < num_tasks

    a = tl.load(sorted_data_ptr + i0, mask=mask)
    i0_prev = tl.where(i0 > 0, i0 - 1, 0)
    b = tl.load(sorted_data_ptr + i0_prev, mask=mask)

    ne_result = tl.where(i0 > 0, a != b, 0)
    cumsum = tl.cumsum(ne_result)

    unique_size_mask = i0 == tile_size - 1
    unique_off = tl.where(unique_size_mask, tl.zeros_like(i0), -1)
    tl.store(unique_size_ptr + unique_off, cumsum, mask=unique_size_mask)

    data_out_off = tl.where(mask, cumsum, -1)
    tl.store(data_out_ptr + data_out_off, a, mask=mask)

    if return_inverse:
        sorted_indices = tl.load(sorted_indices_ptr + i0, mask=mask)
        tl.store(inverse_indices_ptr + sorted_indices, cumsum, mask=mask)

    if return_counts:
        idx_mask = ((i0 == 0) | ne_result.to(tl.int1)) & mask
        tl.store(idx_ptr + cumsum, i0, mask=idx_mask)


@triton.jit
def output_counts_flat_impl(
    global_pid,
    idx_ptr: tl.tensor,
    origin_num_tasks: int,
    counts_ptr: tl.tensor,
    num_tasks: int,
    tile_size: tl.constexpr,
):
    r = tl.arange(0, tile_size)

    i0 = global_pid * tile_size + r
    mask = i0 < num_tasks
    idx = tl.load(idx_ptr + i0, mask=mask)

    i0_next = i0 + 1
    next_mask = i0_next < num_tasks
    idx_next = tl.load(idx_ptr + i0_next, mask=next_mask)

    counts = tl.where(i0_next < num_tasks, idx_next - idx, origin_num_tasks - idx)

    tl.store(counts_ptr + i0, counts, mask=mask)


@libentry()
@triton.jit
def output_counts_flat_kernel(
    idx_ptr: tl.tensor,
    origin_num_tasks: int,
    counts_ptr: tl.tensor,
    num_tasks: int,
    tiles_per_cta: int,
    tile_size: tl.constexpr,
):
    pid = ext.program_id(0)
    ctas_num = ext.num_programs(0)
    for j in range(0, tiles_per_cta):
        global_pid = pid + j * ctas_num
        output_counts_flat_impl(
            global_pid,
            idx_ptr,
            origin_num_tasks,
            counts_ptr,
            num_tasks,
            tile_size,
        )


@triton.jit
def quick_output_flat_impl(
    global_pid,
    sorted_data_ptr: tl.tensor,
    idx_ptr: tl.tensor,
    origin_num_tasks: int,
    data_out_ptr: tl.tensor,
    counts_ptr: tl.tensor,
    num_tasks: int,
    tile_size: tl.constexpr,
):
    r = tl.arange(0, tile_size)

    i0 = global_pid * tile_size + r
    mask = i0 < num_tasks
    idx = tl.load(idx_ptr + i0, mask=mask)

    i0_next = i0 + 1
    next_mask = i0_next < num_tasks
    idx_next = tl.load(idx_ptr + i0_next, mask=next_mask)

    counts = tl.where(i0_next < num_tasks, idx_next - idx, origin_num_tasks - idx)

    tl.store(counts_ptr + i0, counts, mask=mask)

    sorted_data = tl.load(sorted_data_ptr + idx, mask=mask)
    tl.store(data_out_ptr + i0, sorted_data, mask=mask)


@libentry()
@triton.jit
def quick_output_flat_kernel(
    sorted_data_ptr: tl.tensor,
    idx_ptr: tl.tensor,
    origin_num_tasks: int,
    data_out_ptr: tl.tensor,
    counts_ptr: tl.tensor,
    num_tasks: int,
    tiles_per_cta: int,
    tile_size: tl.constexpr,
):
    pid = ext.program_id(0)
    ctas_num = ext.num_programs(0)
    for j in range(0, tiles_per_cta):
        global_pid = pid + j * ctas_num
        quick_output_flat_impl(
            global_pid,
            sorted_data_ptr,
            idx_ptr,
            origin_num_tasks,
            data_out_ptr,
            counts_ptr,
            num_tasks,
            tile_size,
        )


@triton.jit
def local_quick_unique_flat_impl(
    global_pid,
    sorted_data_ptr: tl.tensor,
    local_unique_ptr: tl.tensor,
    origin_idx_ptr: tl.tensor,
    tile_sum_ptr: tl.tensor,
    global_ctas_num: int,
    num_tasks: int,
    tile_size: tl.constexpr,
    return_counts: tl.constexpr,
):
    offset = global_pid * tile_size
    r = tl.arange(0, tile_size)
    i0 = offset + r
    mask = i0 < num_tasks

    a = tl.load(sorted_data_ptr + i0, mask=mask)
    i0_prev = tl.where(i0 > 0, i0 - 1, 0)
    b = tl.load(sorted_data_ptr + i0_prev, mask=mask)

    ne_result = tl.where(i0 > 0, a != b, 0)
    cumsum = tl.cumsum(ne_result)

    local_unique_offset = cumsum - tl.where(global_pid > 0, 1, 0)
    local_unique_mask = (local_unique_offset >= 0) & mask
    if return_counts:
        origin_idx_mask = ((i0 == 0) | ne_result.to(tl.int1)) & local_unique_mask
        lu_store_offset = offset + local_unique_offset
        lu_store_offset = tl.where(origin_idx_mask, lu_store_offset, -1)
        tl.store(
            origin_idx_ptr + lu_store_offset,
            i0,
            mask=origin_idx_mask,
        )
    else:
        lu_store_offset = offset + local_unique_offset
        lu_store_offset = tl.where(local_unique_mask, lu_store_offset, -1)
        tl.store(local_unique_ptr + lu_store_offset, a, mask=local_unique_mask)

    tile_sum_mask = (r == tile_size - 1) & (global_pid < global_ctas_num)
    tile_sum = tl.where(tile_sum_mask & (global_pid == 0), cumsum + 1, cumsum)
    tile_sum_store_offset = global_pid + tl.zeros_like(r)
    tile_sum_store_offset = tl.where(tile_sum_mask, tile_sum_store_offset, -1)
    tl.store(tile_sum_ptr + tile_sum_store_offset, tile_sum, mask=tile_sum_mask)


@libentry()
@triton.jit
def local_quick_unique_flat_kernel(
    sorted_data_ptr: tl.tensor,
    local_unique_ptr: tl.tensor,
    origin_idx_ptr: tl.tensor,
    tile_sum_ptr: tl.tensor,
    global_ctas_num: int,
    num_tasks: int,
    tiles_per_cta: int,
    tile_size: tl.constexpr,
    return_counts: tl.constexpr,
):
    pid = ext.program_id(0)
    ctas_num = ext.num_programs(0)
    for j in range(0, tiles_per_cta):
        global_pid = pid + j * ctas_num
        local_quick_unique_flat_impl(
            global_pid,
            sorted_data_ptr,
            local_unique_ptr,
            origin_idx_ptr,
            tile_sum_ptr,
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
    tile_sum_ptr: tl.tensor,
    data_out_ptr: tl.tensor,
    idx_ptr: tl.tensor,
    ctas_num: tl.constexpr,
    global_ctas_num: tl.constexpr,
    next_power_global_ctas_num: tl.constexpr,
    num_tasks: tl.constexpr,
    tile_size: tl.constexpr,
    return_counts: tl.constexpr,
):
    r = tl.arange(0, tile_size)
    i0 = global_pid * tile_size + r
    mask = i0 < num_tasks

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

    total += tl.sum(pre_tile_sum)
    if global_pid == global_ctas_num - 1:
        last_tile_sum_mask = p == global_pid
        tile_offset = tl.where(last_tile_sum_mask, p, -1)
        tl.store(
            tile_sum_ptr + tile_offset, total + cur_tile_sum, mask=last_tile_sum_mask
        )

    tile_mask = r < cur_tile_sum
    out_offset = total + r
    if return_counts:
        origin_idx = tl.load(origin_idx_ptr + i0, mask=mask)
        idx_offset = tl.where(tile_mask, out_offset, -1)
        tl.store(idx_ptr + idx_offset, origin_idx, mask=tile_mask)
    else:
        local_unique = tl.load(local_unique_ptr + i0, mask=mask)
        data_out_offset = tl.where(tile_mask, out_offset, -1)
        tl.store(data_out_ptr + data_out_offset, local_unique, mask=tile_mask)

    return total


@libentry()
@triton.jit
def global_quick_unique_flat_kernel(
    local_unique_ptr: tl.tensor,
    origin_idx_ptr: tl.tensor,
    tile_sum_ptr: tl.tensor,
    data_out_ptr: tl.tensor,
    idx_ptr: tl.tensor,
    ctas_num: tl.constexpr,
    global_ctas_num: tl.constexpr,
    next_power_global_ctas_num: tl.constexpr,
    num_tasks: tl.constexpr,
    tiles_per_cta: tl.constexpr,
    tile_size: tl.constexpr,
    one_tile_per_cta: tl.constexpr,
    return_counts: tl.constexpr,
):
    pid = ext.program_id(0)
    ctas_num = ext.num_programs(0)
    if one_tile_per_cta:
        global_quick_unique_flat_impl(
            pid,
            0,
            local_unique_ptr,
            origin_idx_ptr,
            tile_sum_ptr,
            data_out_ptr,
            idx_ptr,
            ctas_num,
            global_ctas_num,
            next_power_global_ctas_num,
            num_tasks,
            tile_size,
            return_counts,
        )
    else:
        total = tl.zeros([1], dtype=tl.int64)
        for j in range(0, tiles_per_cta):
            global_pid = pid + j * ctas_num
            total = global_quick_unique_flat_impl(
                global_pid,
                total,
                local_unique_ptr,
                origin_idx_ptr,
                tile_sum_ptr,
                data_out_ptr,
                idx_ptr,
                ctas_num,
                global_ctas_num,
                next_power_global_ctas_num,
                num_tasks,
                tile_size,
                return_counts,
            )


@triton.jit
def global_quick_unique_flat_impl_stage_1(
    global_pid,
    total,
    local_unique_ptr: tl.tensor,
    origin_idx_ptr: tl.tensor,
    tile_sum_ptr: tl.tensor,
    data_out_ptr: tl.tensor,
    idx_ptr: tl.tensor,
    ctas_num: tl.constexpr,
    global_ctas_num: tl.constexpr,
    next_power_global_ctas_num: tl.constexpr,
    num_tasks: tl.constexpr,
    tile_size: tl.constexpr,
    return_counts: tl.constexpr,
):

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

    total += tl.sum(pre_tile_sum)
    if global_pid == global_ctas_num - 1:
        last_tile_sum_mask = p == global_pid
        tile_offset = tl.where(last_tile_sum_mask, p, -1)
        tl.store(
            tile_sum_ptr + tile_offset, total + cur_tile_sum, mask=last_tile_sum_mask
        )

    return total


@libentry()
@triton.jit
def global_quick_unique_flat_kernel_stage_1(
    local_unique_ptr: tl.tensor,
    origin_idx_ptr: tl.tensor,
    tile_sum_ptr: tl.tensor,
    data_out_ptr: tl.tensor,
    idx_ptr: tl.tensor,
    ctas_num: tl.constexpr,
    global_ctas_num: tl.constexpr,
    next_power_global_ctas_num: tl.constexpr,
    num_tasks: tl.constexpr,
    tiles_per_cta: tl.constexpr,
    tile_size: tl.constexpr,
    one_tile_per_cta: tl.constexpr,
    return_counts: tl.constexpr,
):
    pid = ext.program_id(0)
    ctas_num = ext.num_programs(0)
    if one_tile_per_cta:
        global_quick_unique_flat_impl_stage_1(
            pid,
            0,
            local_unique_ptr,
            origin_idx_ptr,
            tile_sum_ptr,
            data_out_ptr,
            idx_ptr,
            ctas_num,
            global_ctas_num,
            next_power_global_ctas_num,
            num_tasks,
            tile_size,
            return_counts,
        )
    else:
        total = tl.zeros([1], dtype=tl.int64)
        for j in range(0, tiles_per_cta):
            global_pid = pid + j * ctas_num
            total = global_quick_unique_flat_impl_stage_1(
                global_pid,
                total,
                local_unique_ptr,
                origin_idx_ptr,
                tile_sum_ptr,
                data_out_ptr,
                idx_ptr,
                ctas_num,
                global_ctas_num,
                next_power_global_ctas_num,
                num_tasks,
                tile_size,
                return_counts,
            )


@triton.jit
def global_quick_unique_flat_impl_stage_2(
    global_pid,
    total,
    local_unique_ptr: tl.tensor,
    origin_idx_ptr: tl.tensor,
    tile_sum_ptr: tl.tensor,
    data_out_ptr: tl.tensor,
    idx_ptr: tl.tensor,
    total_in_ptr,
    ctas_num: tl.constexpr,
    global_ctas_num: tl.constexpr,
    next_power_global_ctas_num: tl.constexpr,
    num_tasks: tl.constexpr,
    tile_size: tl.constexpr,
    return_counts: tl.constexpr,
):
    r = tl.arange(0, tile_size)
    i0 = global_pid * tile_size + r
    mask = i0 < num_tasks

    cur_tile_sum_mask = global_pid < global_ctas_num
    cur_tile_sum = tl.load(tile_sum_ptr + global_pid, mask=cur_tile_sum_mask)

    total_in_mask = global_pid < global_ctas_num
    total = tl.load(total_in_ptr + global_pid, mask=total_in_mask)

    tile_mask = r < cur_tile_sum
    out_offset = total + r
    if return_counts:
        origin_idx = tl.load(origin_idx_ptr + i0, mask=mask)
        idx_offset = tl.where(tile_mask, out_offset, -1)
        tl.store(idx_ptr + idx_offset, origin_idx, mask=tile_mask)
    else:
        local_unique = tl.load(local_unique_ptr + i0, mask=mask)
        data_out_offset = tl.where(tile_mask, out_offset, -1)
        tl.store(data_out_ptr + data_out_offset, local_unique, mask=tile_mask)

    return total


@libentry()
@triton.jit
def global_quick_unique_flat_kernel_stage_2(
    local_unique_ptr: tl.tensor,
    origin_idx_ptr: tl.tensor,
    tile_sum_ptr: tl.tensor,
    data_out_ptr: tl.tensor,
    idx_ptr: tl.tensor,
    total_in_ptr,
    ctas_num: tl.constexpr,
    global_ctas_num: tl.constexpr,
    next_power_global_ctas_num: tl.constexpr,
    num_tasks: tl.constexpr,
    tiles_per_cta: tl.constexpr,
    tile_size: tl.constexpr,
    one_tile_per_cta: tl.constexpr,
    return_counts: tl.constexpr,
):
    pid = ext.program_id(0)
    ctas_num = ext.num_programs(0)
    if one_tile_per_cta:
        global_quick_unique_flat_impl_stage_2(
            pid,
            0,
            local_unique_ptr,
            origin_idx_ptr,
            tile_sum_ptr,
            data_out_ptr,
            idx_ptr,
            total_in_ptr,
            ctas_num,
            global_ctas_num,
            next_power_global_ctas_num,
            num_tasks,
            tile_size,
            return_counts,
        )
    else:
        total = tl.zeros([1], dtype=tl.int64)
        for j in range(0, tiles_per_cta):
            global_pid = pid + j * ctas_num
            total = global_quick_unique_flat_impl_stage_2(
                global_pid,
                total,
                local_unique_ptr,
                origin_idx_ptr,
                tile_sum_ptr,
                data_out_ptr,
                idx_ptr,
                total_in_ptr,
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
    next_power_global_ctas_num = triton.next_power_of_2(global_ctas_num)
    ctas_num = global_ctas_num
    tiles_per_cta = triton.cdiv(num_tasks, tile_size * ctas_num)
    num_warps = 8 if tiles_per_cta == 1 else 32
    grid = (ctas_num, 1, 1)

    if return_counts:
        local_unique = None
        origin_idx = torch.empty_like(sorted_data, dtype=torch.int64)
        idx = torch.empty_like(origin_idx)
    else:
        local_unique = torch.empty_like(sorted_data)
        origin_idx = None
        idx = None
        counts = None
    tile_sum = torch.empty(
        (global_ctas_num,), dtype=torch.int64, device=sorted_data.device
    )
    data_out = None
    if not return_counts:
        data_out = torch.empty_like(sorted_data)
    assert tiles_per_cta == 1
    with torch_device_fn.device(sorted_data.device.index):
        os.environ["TRITONXPU_OTHER_SIM"] = "1"
        os.environ["TRITONXPU_STORE_MASK_SIM"] = "1"
        local_quick_unique_flat_kernel[grid](
            sorted_data,
            local_unique,
            origin_idx,
            tile_sum,
            global_ctas_num,
            num_tasks,
            tiles_per_cta=tiles_per_cta,
            tile_size=tile_size,
            return_counts=return_counts,
            num_warps=num_warps,
        )
        if "TRITONXPU_OTHER_SIM" in os.environ:
            del os.environ["TRITONXPU_OTHER_SIM"]
        if "TRITONXPU_STORE_MASK_SIM" in os.environ:
            del os.environ["TRITONXPU_STORE_MASK_SIM"]

        if num_tasks < 2**26:
            os.environ["TRITONXPU_OTHER_SIM"] = "1"
            os.environ["TRITONXPU_STORE_MASK_SIM"] = "1"
            os.environ["TRITONXPU_INTERLEAVE"] = "0"
            global_quick_unique_flat_kernel[grid](
                local_unique,
                origin_idx,
                tile_sum,
                data_out,
                idx,
                ctas_num,
                global_ctas_num,
                next_power_global_ctas_num,
                num_tasks,
                tiles_per_cta=tiles_per_cta,
                tile_size=tile_size,
                one_tile_per_cta=tiles_per_cta == 1,
                return_counts=return_counts,
                num_warps=num_warps,
                isCloseVectorization=True,
            )
            if "TRITONXPU_OTHER_SIM" in os.environ:
                del os.environ["TRITONXPU_OTHER_SIM"]
            if "TRITONXPU_STORE_MASK_SIM" in os.environ:
                del os.environ["TRITONXPU_STORE_MASK_SIM"]
            if "TRITONXPU_INTERLEAVE" in os.environ:
                del os.environ["TRITONXPU_INTERLEAVE"]
        else:
            total_in = _triton_exclusive_scan(tile_sum)

            os.environ["TRITONXPU_OTHER_SIM"] = "1"
            os.environ["TRITONXPU_STORE_MASK_SIM"] = "1"
            os.environ["TRITONXPU_INTERLEAVE"] = "0"
            global_quick_unique_flat_kernel_stage_1[grid](
                local_unique,
                origin_idx,
                tile_sum,
                data_out,
                idx,
                ctas_num,
                global_ctas_num,
                next_power_global_ctas_num,
                num_tasks,
                tiles_per_cta=tiles_per_cta,
                tile_size=tile_size,
                one_tile_per_cta=tiles_per_cta == 1,
                return_counts=return_counts,
                num_warps=num_warps,
                isCloseVectorization=True,
                buffer_size_limit=128,
            )
            if "TRITONXPU_OTHER_SIM" in os.environ:
                del os.environ["TRITONXPU_OTHER_SIM"]
            if "TRITONXPU_STORE_MASK_SIM" in os.environ:
                del os.environ["TRITONXPU_STORE_MASK_SIM"]
            if "TRITONXPU_INTERLEAVE" in os.environ:
                del os.environ["TRITONXPU_INTERLEAVE"]

            os.environ["TRITONXPU_OTHER_SIM"] = "1"
            os.environ["TRITONXPU_STORE_MASK_SIM"] = "1"
            os.environ["TRITONXPU_INTERLEAVE"] = "0"
            global_quick_unique_flat_kernel_stage_2[grid](
                local_unique,
                origin_idx,
                tile_sum,
                data_out,
                idx,
                total_in,
                ctas_num,
                global_ctas_num,
                next_power_global_ctas_num,
                num_tasks,
                tiles_per_cta=tiles_per_cta,
                tile_size=tile_size,
                one_tile_per_cta=tiles_per_cta == 1,
                return_counts=return_counts,
                num_warps=num_warps,
                isCloseVectorization=True,
            )
            if "TRITONXPU_OTHER_SIM" in os.environ:
                del os.environ["TRITONXPU_OTHER_SIM"]
            if "TRITONXPU_STORE_MASK_SIM" in os.environ:
                del os.environ["TRITONXPU_STORE_MASK_SIM"]
            if "TRITONXPU_INTERLEAVE" in os.environ:
                del os.environ["TRITONXPU_INTERLEAVE"]

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
                num_tasks,
                data_out,
                counts,
                out_size,
                tiles_per_cta,
                tile_size,
                num_warps=num_warps,
                isCloseUnrollControl=(
                    True if sorted_data.dtype == torch.int16 else False
                ),
            )

    if return_counts:
        return data_out, None, counts
    else:
        return data_out[:out_size], None, None


@triton.jit
def local_ne_flat_impl(
    global_pid,
    sorted_data_ptr: tl.tensor,
    ne_result_ptr: tl.tensor,
    tile_sum_ptr: tl.tensor,
    global_ctas_num: int,
    num_tasks: int,
    tile_size: tl.constexpr,
):
    r = tl.arange(0, tile_size)
    i0 = global_pid * tile_size + r
    mask = i0 < num_tasks
    i0_prev = tl.where(i0 > 0, i0 - 1, 0)

    a = tl.load(sorted_data_ptr + i0, mask=mask)
    b = tl.load(sorted_data_ptr + i0_prev, mask=mask)

    ne_result = tl.where(i0 > 0, a != b, 0)

    tl.store(ne_result_ptr + i0, ne_result, mask=mask)

    tile_sum = tl.sum(ne_result)
    tile_sum_mask = global_pid < global_ctas_num
    tl.store(tile_sum_ptr + global_pid, tile_sum, mask=tile_sum_mask)


@libentry()
@triton.jit
def local_ne_flat_kernel(
    sorted_data_ptr: tl.tensor,
    ne_result_ptr: tl.tensor,
    tile_sum_ptr: tl.tensor,
    global_ctas_num: int,
    num_tasks: int,
    tiles_per_cta: int,
    tile_size: tl.constexpr,
):
    pid = ext.program_id(0)
    ctas_num = ext.num_programs(0)
    for j in range(0, tiles_per_cta):
        global_pid = pid + j * ctas_num
        local_ne_flat_impl(
            global_pid,
            sorted_data_ptr,
            ne_result_ptr,
            tile_sum_ptr,
            global_ctas_num,
            num_tasks,
            tile_size,
        )


@triton.jit
def global_cumsum_flat_impl(
    global_pid,
    total,
    ne_result_ptr: tl.tensor,
    tile_sum_ptr: tl.tensor,
    sorted_data_ptr: tl.tensor,
    sorted_indices_ptr: tl.tensor,
    data_out_ptr: tl.tensor,
    inverse_indices_ptr: tl.tensor,
    idx_ptr: tl.tensor,
    cumsum_out,
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

    sorted_data = tl.load(sorted_data_ptr + i0, mask=mask)
    sorted_indices = tl.load(sorted_indices_ptr + i0, mask=mask)

    p = tl.arange(0, next_power_global_ctas_num)
    pre_tile_sum_mask = (
        (p >= global_pid - ctas_num)
        & (p < global_pid)
        & (p >= 0)
        & (p < global_ctas_num)
    )
    pre_tile_sum = tl.load(tile_sum_ptr + p, mask=pre_tile_sum_mask, other=0)

    total += tl.sum(pre_tile_sum)
    ne_result = tl.load(ne_result_ptr + i0, mask=mask)
    ne_result_i1 = ne_result.to(tl.int1)
    ne_result = ne_result.to(tl.int32)
    cumsum = tl.cumsum(ne_result)

    if global_pid == global_ctas_num - 1:
        last_tile_sum_mask = i0 == num_tasks - 1
        tile_sum = tl.where(last_tile_sum_mask, total + cumsum, cumsum)
        tile_offset = tl.where(last_tile_sum_mask, global_pid + tl.zeros_like(r), -1)
        tl.store(
            tile_sum_ptr + tile_offset,
            tile_sum,
            mask=last_tile_sum_mask,
        )
    cumsum += total

    tl.store(data_out_ptr + cumsum, sorted_data, mask=mask)

    tl.store(inverse_indices_ptr + sorted_indices, cumsum, mask=mask)

    if return_counts:
        idx_mask = ((i0 == 0) | ne_result_i1) & mask
        idx_offset = tl.where(idx_mask, cumsum, num_tasks + 1)
        tl.store(idx_ptr + idx_offset, i0, mask=idx_mask)

    return total


@libentry()
@triton.jit
def global_cumsum_flat_kernel(
    ne_result_ptr: tl.tensor,
    tile_sum_ptr: tl.tensor,
    sorted_data_ptr: tl.tensor,
    sorted_indices_ptr: tl.tensor,
    data_out_ptr: tl.tensor,
    inverse_indices_ptr: tl.tensor,
    idx_ptr: tl.tensor,
    cumsum_out,
    ctas_num: int,
    global_ctas_num: int,
    next_power_global_ctas_num: tl.constexpr,
    num_tasks: int,
    tiles_per_cta: int,
    tile_size: tl.constexpr,
    one_tile_per_cta: tl.constexpr,
    return_counts: tl.constexpr,
):
    pid = ext.program_id(0)
    ctas_num = ext.num_programs(0)
    if one_tile_per_cta:
        global_cumsum_flat_impl(
            pid,
            0,
            ne_result_ptr,
            tile_sum_ptr,
            sorted_data_ptr,
            sorted_indices_ptr,
            data_out_ptr,
            inverse_indices_ptr,
            idx_ptr,
            cumsum_out,
            ctas_num,
            global_ctas_num,
            next_power_global_ctas_num,
            num_tasks,
            tile_size,
            return_counts,
        )
    else:
        total = tl.zeros([1], dtype=tl.int64)
        for j in range(0, tiles_per_cta):
            global_pid = pid + j * ctas_num
            total = global_cumsum_flat_impl(
                global_pid,
                total,
                ne_result_ptr,
                tile_sum_ptr,
                sorted_data_ptr,
                sorted_indices_ptr,
                data_out_ptr,
                inverse_indices_ptr,
                idx_ptr,
                cumsum_out,
                ctas_num,
                global_ctas_num,
                next_power_global_ctas_num,
                num_tasks,
                tile_size,
                return_counts,
            )


@triton.jit
def global_cumsum_flat_impl_stage_1(
    global_pid,
    total,
    ne_result_ptr: tl.tensor,
    tile_sum_ptr: tl.tensor,
    sorted_data_ptr: tl.tensor,
    sorted_indices_ptr: tl.tensor,
    data_out_ptr: tl.tensor,
    inverse_indices_ptr: tl.tensor,
    idx_ptr: tl.tensor,
    total_in_ptr,
    cumsum_in_ptr,
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

    total_in_mask = global_pid < global_ctas_num
    total = tl.load(total_in_ptr + global_pid, mask=total_in_mask)

    ne_result = tl.load(ne_result_ptr + i0, mask=mask)
    ne_result = ne_result.to(tl.int32)
    cumsum = tl.load(cumsum_in_ptr + i0)

    if global_pid == global_ctas_num - 1:
        last_tile_sum_mask = i0 == num_tasks - 1
        tile_sum = tl.where(last_tile_sum_mask, total + cumsum, cumsum)
        tile_offset = tl.where(last_tile_sum_mask, global_pid + tl.zeros_like(r), -1)
        tl.store(
            tile_sum_ptr + tile_offset,
            tile_sum,
            mask=last_tile_sum_mask,
        )

    return total


@libentry()
@triton.jit
def global_cumsum_flat_kernel_stage_1(
    ne_result_ptr: tl.tensor,
    tile_sum_ptr: tl.tensor,
    sorted_data_ptr: tl.tensor,
    sorted_indices_ptr: tl.tensor,
    data_out_ptr: tl.tensor,
    inverse_indices_ptr: tl.tensor,
    idx_ptr: tl.tensor,
    total_in_ptr,
    cumsum_in_ptr,
    ctas_num: int,
    global_ctas_num: int,
    next_power_global_ctas_num: tl.constexpr,
    num_tasks: int,
    tiles_per_cta: int,
    tile_size: tl.constexpr,
    one_tile_per_cta: tl.constexpr,
    return_counts: tl.constexpr,
):
    pid = ext.program_id(0)
    ctas_num = ext.num_programs(0)
    if one_tile_per_cta:
        global_cumsum_flat_impl_stage_1(
            pid,
            0,
            ne_result_ptr,
            tile_sum_ptr,
            sorted_data_ptr,
            sorted_indices_ptr,
            data_out_ptr,
            inverse_indices_ptr,
            idx_ptr,
            total_in_ptr,
            cumsum_in_ptr,
            ctas_num,
            global_ctas_num,
            next_power_global_ctas_num,
            num_tasks,
            tile_size,
            return_counts,
        )
    else:
        total = tl.zeros([1], dtype=tl.int64)
        for j in range(0, tiles_per_cta):
            global_pid = pid + j * ctas_num
            total = global_cumsum_flat_impl_stage_1(
                global_pid,
                total,
                ne_result_ptr,
                tile_sum_ptr,
                sorted_data_ptr,
                sorted_indices_ptr,
                data_out_ptr,
                inverse_indices_ptr,
                idx_ptr,
                total_in_ptr,
                cumsum_in_ptr,
                ctas_num,
                global_ctas_num,
                next_power_global_ctas_num,
                num_tasks,
                tile_size,
                return_counts,
            )


@triton.jit
def global_cumsum_flat_impl_stage_2(
    global_pid,
    total,
    ne_result_ptr: tl.tensor,
    tile_sum_ptr: tl.tensor,
    sorted_data_ptr: tl.tensor,
    sorted_indices_ptr: tl.tensor,
    data_out_ptr: tl.tensor,
    inverse_indices_ptr: tl.tensor,
    idx_ptr: tl.tensor,
    total_in_ptr,
    cumsum_in_ptr,
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

    sorted_data = tl.load(sorted_data_ptr + i0, mask=mask)
    sorted_indices = tl.load(sorted_indices_ptr + i0, mask=mask)

    total_in_mask = global_pid < global_ctas_num
    total = tl.load(total_in_ptr + global_pid, mask=total_in_mask)

    ne_result = tl.load(ne_result_ptr + i0, mask=mask)
    ne_result_i1 = ne_result.to(tl.int1)
    ne_result = ne_result.to(tl.int32)
    cumsum = tl.load(cumsum_in_ptr + i0)
    cumsum += total

    tl.store(data_out_ptr + cumsum, sorted_data, mask=mask)

    tl.store(inverse_indices_ptr + sorted_indices, cumsum, mask=mask)

    if return_counts:
        idx_mask = ((i0 == 0) | ne_result_i1) & mask
        idx_offset = tl.where(idx_mask, cumsum, num_tasks + 1)
        tl.store(idx_ptr + idx_offset, i0, mask=idx_mask)

    return total


@libentry()
@triton.jit
def global_cumsum_flat_kernel_stage_2(
    ne_result_ptr: tl.tensor,
    tile_sum_ptr: tl.tensor,
    sorted_data_ptr: tl.tensor,
    sorted_indices_ptr: tl.tensor,
    data_out_ptr: tl.tensor,
    inverse_indices_ptr: tl.tensor,
    idx_ptr: tl.tensor,
    total_in_ptr,
    cumsum_in_ptr,
    ctas_num: int,
    global_ctas_num: int,
    next_power_global_ctas_num: tl.constexpr,
    num_tasks: int,
    tiles_per_cta: int,
    tile_size: tl.constexpr,
    one_tile_per_cta: tl.constexpr,
    return_counts: tl.constexpr,
):
    pid = ext.program_id(0)
    ctas_num = ext.num_programs(0)
    if one_tile_per_cta:
        global_cumsum_flat_impl_stage_2(
            pid,
            0,
            ne_result_ptr,
            tile_sum_ptr,
            sorted_data_ptr,
            sorted_indices_ptr,
            data_out_ptr,
            inverse_indices_ptr,
            idx_ptr,
            total_in_ptr,
            cumsum_in_ptr,
            ctas_num,
            global_ctas_num,
            next_power_global_ctas_num,
            num_tasks,
            tile_size,
            return_counts,
        )
    else:
        total = tl.zeros([1], dtype=tl.int64)
        for j in range(0, tiles_per_cta):
            global_pid = pid + j * ctas_num
            total = global_cumsum_flat_impl_stage_2(
                global_pid,
                total,
                ne_result_ptr,
                tile_sum_ptr,
                sorted_data_ptr,
                sorted_indices_ptr,
                data_out_ptr,
                inverse_indices_ptr,
                idx_ptr,
                total_in_ptr,
                cumsum_in_ptr,
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
    tile_size = min(2048, next_power_num_tasks)
    global_ctas_num = triton.cdiv(num_tasks, tile_size)
    next_power_global_ctas_num = triton.next_power_of_2(global_ctas_num)
    ctas_num = global_ctas_num
    tiles_per_cta = triton.cdiv(num_tasks, tile_size * ctas_num)
    num_warps = 8 if tiles_per_cta == 1 else 32
    grid = (ctas_num, 1, 1)

    ne_result = torch.empty_like(sorted_data, dtype=torch.bool)
    tile_sum = torch.empty(
        (global_ctas_num,), dtype=torch.int64, device=sorted_data.device
    )
    data_out = torch.empty_like(sorted_data)
    inverse_indices = torch.empty_like(sorted_data, dtype=torch.int64)
    idx = None
    if return_counts:
        idx = torch.empty_like(inverse_indices)

    with torch_device_fn.device(sorted_data.device.index):
        os.environ["TRITONXPU_OTHER_SIM"] = "1"
        os.environ["TRITONXPU_STORE_MASK_SIM"] = "1"
        os.environ["TRITONXPU_INTERLEAVE"] = "0"

        local_ne_flat_kernel[grid](
            sorted_data,
            ne_result,
            tile_sum,
            global_ctas_num,
            num_tasks,
            tiles_per_cta=tiles_per_cta,
            tile_size=tile_size,
            num_warps=num_warps,
        )
        if "TRITONXPU_OTHER_SIM" in os.environ:
            del os.environ["TRITONXPU_OTHER_SIM"]
        if "TRITONXPU_STORE_MASK_SIM" in os.environ:
            del os.environ["TRITONXPU_STORE_MASK_SIM"]
        if "TRITONXPU_INTERLEAVE" in os.environ:
            del os.environ["TRITONXPU_INTERLEAVE"]

        if num_tasks < 2**26:
            next_multiple = ((num_tasks // 2048) + 1) * 2048
            cumsum_out = torch.empty(
                next_multiple, dtype=torch.int64, device=sorted_data.device
            )
            os.environ["TRITONXPU_OTHER_SIM"] = "1"
            os.environ["TRITONXPU_STORE_MASK_SIM"] = "1"
            os.environ["TRITONXPU_INTERLEAVE"] = "0"
            global_cumsum_flat_kernel[grid](
                ne_result,
                tile_sum,
                sorted_data,
                sorted_indices,
                data_out,
                inverse_indices,
                idx,
                cumsum_out,
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
            if "TRITONXPU_OTHER_SIM" in os.environ:
                del os.environ["TRITONXPU_OTHER_SIM"]
            if "TRITONXPU_STORE_MASK_SIM" in os.environ:
                del os.environ["TRITONXPU_STORE_MASK_SIM"]
            if "TRITONXPU_INTERLEAVE" in os.environ:
                del os.environ["TRITONXPU_INTERLEAVE"]

        else:
            total_in = _triton_exclusive_scan(tile_sum)

            next_multiple = ((num_tasks // 2048) + 1) * 2048
            num_blocks = next_multiple // 2048
            cumsum_result = torch.empty(
                next_multiple, dtype=torch.int64, device=sorted_data.device
            )
            _scan_rows2d_kernel[(num_blocks,)](
                ne_result, cumsum_result, num_blocks, num_tasks, W=2048
            )

            os.environ["TRITONXPU_OTHER_SIM"] = "1"
            os.environ["TRITONXPU_STORE_MASK_SIM"] = "1"
            os.environ["TRITONXPU_INTERLEAVE"] = "0"
            global_cumsum_flat_kernel_stage_1[grid](
                ne_result,
                tile_sum,
                sorted_data,
                sorted_indices,
                data_out,
                inverse_indices,
                idx,
                total_in,
                cumsum_result,
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
            if "TRITONXPU_OTHER_SIM" in os.environ:
                del os.environ["TRITONXPU_OTHER_SIM"]
            if "TRITONXPU_STORE_MASK_SIM" in os.environ:
                del os.environ["TRITONXPU_STORE_MASK_SIM"]
            if "TRITONXPU_INTERLEAVE" in os.environ:
                del os.environ["TRITONXPU_INTERLEAVE"]

            os.environ["TRITONXPU_OTHER_SIM"] = "1"
            os.environ["TRITONXPU_STORE_MASK_SIM"] = "1"
            os.environ["TRITONXPU_INTERLEAVE"] = "0"
            global_cumsum_flat_kernel_stage_2[grid](
                ne_result,
                tile_sum,
                sorted_data,
                sorted_indices,
                data_out,
                inverse_indices,
                idx,
                total_in,
                cumsum_result,
                ctas_num,
                global_ctas_num,
                next_power_global_ctas_num,
                num_tasks,
                tiles_per_cta=tiles_per_cta,
                tile_size=tile_size,
                one_tile_per_cta=tiles_per_cta == 1,
                return_counts=return_counts,
                num_warps=num_warps,
                isCloseUnrollControl=True,
            )
            if "TRITONXPU_OTHER_SIM" in os.environ:
                del os.environ["TRITONXPU_OTHER_SIM"]
            if "TRITONXPU_STORE_MASK_SIM" in os.environ:
                del os.environ["TRITONXPU_STORE_MASK_SIM"]
            if "TRITONXPU_INTERLEAVE" in os.environ:
                del os.environ["TRITONXPU_INTERLEAVE"]

        out_size = tile_sum[-1].item() + 1
        counts = None
        if return_counts:
            idx = idx[:out_size]
            counts = torch.empty_like(idx)
            os.environ["TRITONXPU_OTHER_SIM"] = "1"
            os.environ["TRITONXPU_STORE_MASK_SIM"] = "1"
            os.environ["TRITONXPU_INTERLEAVE"] = "0"
            output_counts_flat_kernel[grid](
                idx,
                num_tasks,
                counts,
                out_size,
                tiles_per_cta,
                tile_size,
                num_warps=num_warps,
            )
            if "TRITONXPU_OTHER_SIM" in os.environ:
                del os.environ["TRITONXPU_OTHER_SIM"]
            if "TRITONXPU_STORE_MASK_SIM" in os.environ:
                del os.environ["TRITONXPU_STORE_MASK_SIM"]
            if "TRITONXPU_INTERLEAVE" in os.environ:
                del os.environ["TRITONXPU_INTERLEAVE"]

    return data_out[:out_size], inverse_indices, counts


def simple_unique_flat(
    sorted_data: torch.Tensor,
    sorted_indices: torch.Tensor,
    return_inverse: bool,
    return_counts: bool,
):
    num_tasks = sorted_data.numel()
    grid = (1, 1, 1)

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

    with torch_device_fn.device(sorted_data.device.index):
        os.environ["TRITONXPU_OTHER_SIM"] = "1"
        os.environ["TRITONXPU_STORE_MASK_SIM"] = "1"
        os.environ["TRITONXPU_INTERLEAVE"] = "0"
        simple_unique_flat_kernel[grid](
            sorted_data,
            sorted_indices,
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
        if "TRITONXPU_OTHER_SIM" in os.environ:
            del os.environ["TRITONXPU_OTHER_SIM"]
        if "TRITONXPU_STORE_MASK_SIM" in os.environ:
            del os.environ["TRITONXPU_STORE_MASK_SIM"]
        if "TRITONXPU_INTERLEAVE" in os.environ:
            del os.environ["TRITONXPU_INTERLEAVE"]
    out_size = unique_size.item() + 1
    counts = None
    if return_counts:
        idx = idx[:out_size]
        counts = torch.empty_like(idx)
        with torch_device_fn.device(sorted_data.device.index):
            os.environ["TRITONXPU_OTHER_SIM"] = "1"
            os.environ["TRITONXPU_STORE_MASK_SIM"] = "1"
            os.environ["TRITONXPU_INTERLEAVE"] = "0"
            output_counts_flat_kernel[grid](
                idx,
                num_tasks,
                counts,
                num_tasks=out_size,
                tiles_per_cta=1,
                tile_size=triton.next_power_of_2(out_size),
                num_warps=8,
            )
            if "TRITONXPU_OTHER_SIM" in os.environ:
                del os.environ["TRITONXPU_OTHER_SIM"]
            if "TRITONXPU_STORE_MASK_SIM" in os.environ:
                del os.environ["TRITONXPU_STORE_MASK_SIM"]
            if "TRITONXPU_INTERLEAVE" in os.environ:
                del os.environ["TRITONXPU_INTERLEAVE"]
    return data_out[:out_size], inverse_indices, counts


_BOUND_BLOCK = 4096


@libentry()
@triton.jit
def _unique2_boundary_kernel(
    data_ptr,
    ne_ptr,
    cum_ptr,
    N: int,
    BLOCK: tl.constexpr,
):
    """Fused boundary flags for _unique2 (see caller comment).

    For each lane: ne[i] = (i == 0) | (data[i] != data[i-1]);
    cum[0] = 0 and cum[i] = ne[i] (i > 0).  `data` is already sorted, so the
    comparison is done on the native dtype (any int/float width) with no
    cast pass and no `others`/`other=` dependence: the prev load for lane 0
    is clamped to lane 0 (its value is unused because the OR forces True).
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(data_ptr + offs, mask=mask)
    p_offs = tl.where(offs > 0, offs - 1, 0)
    b = tl.load(data_ptr + p_offs, mask=mask)
    is_b = (offs == 0) | ((a != b) & (offs > 0))
    tl.store(ne_ptr + offs, is_b, mask=mask)
    tl.store(cum_ptr + offs, tl.where(offs == 0, 0, is_b.to(tl.int64)), mask=mask)


_SCAN_BLOCK = 2048
_SCAN_CHUNK = 2048


@triton.jit
def _scan_tile_sums_kernel(
    data_ptr,
    sums_ptr,
    N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    v = tl.load(data_ptr + offs, mask=mask, other=0)
    tl.store(sums_ptr + pid, tl.sum(v.to(tl.int64)))


@triton.jit
def _scan_sums_reduce_kernel(
    sums_ptr,
    sums2_ptr,
    G,
    CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    j = pid * CHUNK + tl.arange(0, CHUNK)
    m = j < G
    s = tl.load(sums_ptr + j, mask=m, other=0)
    tl.store(sums2_ptr + pid, tl.sum(s))


@triton.jit
def _scan_sums2_scan_kernel(
    sums2_ptr,
    scanned2_ptr,
    G2,
    CW2: tl.constexpr,
):
    """scanned2[i] = sum(sums2[:i]) (exclusive prefix)."""
    j = tl.arange(0, CW2)
    m = j < G2
    s = tl.load(sums2_ptr + j, mask=m, other=0)
    tl.store(scanned2_ptr + j, tl.cumsum(s) - s, mask=m)


@triton.jit
def _scan_sums_cumsum_kernel(
    sums_ptr,
    tmp_ptr,
    G,
    CHUNK: tl.constexpr,
):
    """Chunk-exclusive prefix of the tile sums: tmp[i] = sum(sums[chunk..i-1])."""
    pid = tl.program_id(0)
    j = pid * CHUNK + tl.arange(0, CHUNK)
    m = j < G
    s = tl.load(sums_ptr + j, mask=m, other=0)
    tl.store(tmp_ptr + j, tl.cumsum(s) - s, mask=m)


@triton.jit
def _scan_sums_apply_kernel(
    tmp_ptr,
    scanned2_ptr,
    sums_out_ptr,
    G,
    CHUNK: tl.constexpr,
):
    """sums_out[i] = tmp[i] + sum(sums2[:chunk]) (exclusive prefix of tile sums).

    Kept separate from _scan_sums_cumsum_kernel: mixing tl.sum with tl.cumsum
    in one kernel miscompiles on this backend (measured garbage), while each op
    alone is exact.
    """
    pid = tl.program_id(0)
    j = pid * CHUNK + tl.arange(0, CHUNK)
    m = j < G
    t = tl.load(tmp_ptr + j, mask=m, other=0)
    carry = tl.load(scanned2_ptr + pid)
    tl.store(sums_out_ptr + j, t + carry, mask=m)


@triton.jit
def _scan_add_kernel(
    data_ptr,
    sums_out_ptr,
    out_ptr,
    N,
    BLOCK: tl.constexpr,
    inclusive: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    v = tl.load(data_ptr + offs, mask=mask, other=0).to(tl.int64)
    carry = tl.load(sums_out_ptr + pid)
    c = carry + tl.cumsum(v)
    if not inclusive:
        c = c - v
    tl.store(out_ptr + offs, c, mask=mask)


def _triton_scan(x: torch.Tensor, inclusive: bool) -> torch.Tensor:
    """out[i] = sum(x[:i+1]) (inclusive) or sum(x[:i]) (exclusive), device-side."""
    N = x.numel()
    if inclusive:
        out = torch.empty_like(x)
    else:
        out = torch.empty(N, dtype=torch.int64, device=x.device)
    if N == 0:
        return out
    G = triton.cdiv(N, _SCAN_BLOCK)
    G2 = triton.cdiv(G, _SCAN_CHUNK)
    sums = torch.empty(G, dtype=torch.int64, device=x.device)
    sums2 = torch.empty(G2, dtype=torch.int64, device=x.device)
    scanned2 = torch.empty_like(sums2)
    tmp = torch.empty_like(sums)
    sums_out = torch.empty_like(sums)
    with torch_device_fn.device(x.device):
        _scan_tile_sums_kernel[(G,)](x, sums, N, BLOCK=_SCAN_BLOCK)
        _scan_sums_reduce_kernel[(G2,)](sums, sums2, G, CHUNK=_SCAN_CHUNK)
        _scan_sums2_scan_kernel[(1,)](
            sums2, scanned2, G2, CW2=triton.next_power_of_2(G2)
        )
        _scan_sums_cumsum_kernel[(G2,)](sums, tmp, G, CHUNK=_SCAN_CHUNK)
        _scan_sums_apply_kernel[(G2,)](tmp, scanned2, sums_out, G, CHUNK=_SCAN_CHUNK)
        _scan_add_kernel[(G,)](
            x, sums_out, out, N, BLOCK=_SCAN_BLOCK, inclusive=inclusive
        )
    return out


def _triton_inclusive_scan(x: torch.Tensor) -> torch.Tensor:
    return _triton_scan(x, True)


def _triton_exclusive_scan(x: torch.Tensor) -> torch.Tensor:
    return _triton_scan(x, False)


@triton.jit
def _scan_rows2d_kernel(
    ne_ptr,
    out_ptr,
    R,
    N,
    W: tl.constexpr,
):
    """Per-row (W-wide, row-major) inclusive cumsum of a bool/int array.

    Row r covers [r*W, (r+1)*W); lanes >= N (the padding) are treated as 0 and
    left uninitialized in `out` (never read by the stage_1/2 kernels, which
    mask to i0 < num_tasks).  Equivalent to
        F.pad(ne, (0, pad), 'constant', 0).view(R, W).cumsum(dim=1).view(-1)
    without the ATen pad/reshape/cumsum round-trip.
    """
    r = tl.program_id(0)
    j = tl.arange(0, W)
    i = r * W + j
    m = i < N
    v = tl.load(ne_ptr + i, mask=m, other=0).to(tl.int64)
    tl.store(out_ptr + i, tl.cumsum(v), mask=m)


@triton.jit
def _run_lengths_kernel(
    start_ptr,
    counts_ptr,
    N,
    n,
    BLOCK: tl.constexpr,
):
    """counts[i] = start[i+1] - start[i] (last: N - start[n-1])."""
    r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = r < n
    s = tl.load(start_ptr + r, mask=m, other=0)
    s_next = tl.load(start_ptr + r + 1, mask=m & (r + 1 < n), other=0)
    s_next = tl.where(r + 1 < n, s_next, N)
    tl.store(counts_ptr + r, s_next - s, mask=m)


def _unique2(
    in0: torch.Tensor,
    sorted: bool = True,
    return_inverse: bool = False,
    return_counts: bool = False,
):
    logger.debug("GEMS_KUNLUNXIN _UNIQUE2")
    _ = sorted
    flat = in0.contiguous().view(-1)
    N = flat.numel()

    if N == 0:
        data_out = flat.clone()
        inverse_indices = (
            torch.empty_like(flat, dtype=torch.int64) if return_inverse else None
        )
        counts = (
            torch.empty(0, dtype=torch.int64, device=flat.device)
            if return_counts
            else None
        )
        return (
            data_out,
            (
                inverse_indices
                if inverse_indices is None
                else inverse_indices.view_as(in0)
            ),
            counts,
        )

    sorted_data, sorted_indices = torch.sort(flat)
    ne = torch.empty(N, dtype=torch.bool, device=flat.device)
    cum_input = torch.empty(N, dtype=torch.int64, device=flat.device)
    with torch_device_fn.device(flat.device):
        _unique2_boundary_kernel[(triton.cdiv(N, _BOUND_BLOCK),)](
            sorted_data, ne, cum_input, N, BLOCK=_BOUND_BLOCK, num_warps=8
        )

    start = torch.nonzero(ne).ravel()
    n_unique = start.numel()
    if n_unique == N:
        data_out = sorted_data
    else:
        data_out = torch.index_select(sorted_data, 0, start)

    inverse_indices = None
    counts = None

    if return_inverse:
        cum = _triton_inclusive_scan(cum_input)
        inverse_indices = torch.empty(N, dtype=torch.int64, device=flat.device)
        inverse_indices.scatter_(0, sorted_indices, cum)

    if return_counts:
        counts = torch.empty(n_unique, dtype=torch.int64, device=flat.device)
        if n_unique > 0:
            _run_lengths_kernel[(triton.cdiv(n_unique, _SCAN_BLOCK),)](
                start, counts, N, n_unique, BLOCK=_SCAN_BLOCK
            )

    return (
        data_out,
        inverse_indices if inverse_indices is None else inverse_indices.view_as(in0),
        counts,
    )
