import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.libentry import libentry

logger = logging.getLogger(f'flag_gems.runtime._ascend.ops.{__name__.split(".")[-1]}')


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
    a = tl.load(sorted_data_ptr + i0, mask=mask, other=0)
    i0_prev = tl.where(i0 > 0, i0 - 1, 0)
    b = tl.load(sorted_data_ptr + i0_prev, mask=mask, other=0)

    # ne & cumsum
    # 对于 i0=0 的位置（第一个元素），ne_result 应该是 1（它是第一个唯一值）
    # 对于其他位置，ne_result = (a != b)
    ne_result = tl.where(i0 > 0, a != b, 1)
    ne_result = tl.where(mask, ne_result, 0)  # 只保留有效位置

    cumsum = tl.cumsum(ne_result)

    # 对于第一个唯一值（i0=0），cumsum=1，所以索引是 0（cumsum-1）
    # 对于其他唯一值，cumsum 递增
    local_unique_offset = cumsum - 1  # cumsum 从 1 开始，所以减 1 得到从 0 开始的索引
    local_unique_mask = mask

    if return_counts:
        # origin_idx: 只在唯一值位置存储
        origin_idx_mask = ne_result.to(tl.int1) & local_unique_mask
        tl.store(
            origin_idx_ptr + (offset + local_unique_offset),
            i0,
            mask=origin_idx_mask,
        )
    else:
        # local_unique: 只在唯一值位置存储
        store_mask = ne_result.to(tl.int1) & local_unique_mask
        tl.store(local_unique_ptr + (offset + local_unique_offset), a, mask=store_mask)

    # tile_sum - 获取最后一个有效位置的 cumsum 值
    valid_cumsum = tl.where(mask, cumsum, 0)
    last_cumsum = tl.max(valid_cumsum)

    # 直接使用 last_cumsum，不需要特殊处理第一个 tile
    if global_pid < global_ctas_num:
        tl.store(tile_sum_ptr + global_pid, last_cumsum)


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
    CHUNK_SIZE: tl.constexpr,  # 每个块的大小
    MAX_CHUNKS: tl.constexpr,  # 最大块数
):
    r = tl.arange(0, tile_size)
    i0 = global_pid * tile_size + r
    mask = i0 < num_tasks

    # load tile_sum - 使用分块处理避免UB overflow
    start_idx = tl.maximum(global_pid - ctas_num, 0)
    end_idx = tl.minimum(global_pid, global_ctas_num)

    # 分块累加 pre_tile_sum
    total_sum = 0
    total_sum = total_sum.to(tl.int64)
    for chunk_id in range(MAX_CHUNKS):
        chunk_start = start_idx + chunk_id * CHUNK_SIZE

        # 只有当这个chunk在有效范围内时才处理
        if chunk_start < end_idx:
            p = tl.arange(0, CHUNK_SIZE)
            p_idx = chunk_start + p

            # 计算mask：需要确保索引在 [start_idx, end_idx) 范围内
            pre_tile_sum_mask = (
                (p_idx < end_idx) & (p_idx >= start_idx) & (p_idx < global_ctas_num)
            )

            pre_tile_sum = tl.load(
                tile_sum_ptr + p_idx, mask=pre_tile_sum_mask, other=0
            )
            total_sum += tl.sum(pre_tile_sum)

    cur_tile_sum_mask = global_pid < global_ctas_num
    cur_tile_sum = tl.load(tile_sum_ptr + global_pid, mask=cur_tile_sum_mask, other=0)

    # total
    total += total_sum

    # tile_sum 存储
    if global_pid == global_ctas_num - 1:
        tl.store(tile_sum_ptr + global_pid, total + cur_tile_sum)

    # idx or data_out
    tile_mask = r < cur_tile_sum
    out_offset = total + r

    if return_counts:
        # move origin_idx to idx_ptr
        origin_idx = tl.load(origin_idx_ptr + i0, mask=mask, other=0)
        tl.store(idx_ptr + out_offset, origin_idx, mask=tile_mask)
    else:
        # move local_unique to data_out_ptr
        local_unique = tl.load(local_unique_ptr + i0, mask=mask, other=0)
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

    # 分块处理参数
    CHUNK_SIZE: tl.constexpr = 2048  # 每块处理2048个元素
    MAX_CHUNKS: tl.constexpr = 32  # 最多32块 (2048 * 32 = 65536)

    if one_tile_per_cta:
        # monolitic kernel style
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
            CHUNK_SIZE,
            MAX_CHUNKS,
        )
    else:
        # grid-stride-loop style kernel
        total = tl.zeros([1], dtype=tl.int64)
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
                CHUNK_SIZE,
                MAX_CHUNKS,
            )


def sorted_quick_unique_flat(sorted_data: torch.Tensor, return_counts: bool):
    num_tasks = sorted_data.numel()
    next_power_num_tasks = triton.next_power_of_2(num_tasks)
    tile_size = min(4096, next_power_num_tasks)
    global_ctas_num = triton.cdiv(num_tasks, tile_size)

    next_power_global_ctas_num = triton.next_power_of_2(global_ctas_num)
    ctas_num = global_ctas_num if global_ctas_num < 65536 else 2048
    tiles_per_cta = triton.cdiv(num_tasks, tile_size * ctas_num)
    num_warps = 8 if tiles_per_cta == 1 else 32
    grid = (ctas_num, 1, 1)

    # allocate tensor
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
    BLOCK_SIZE_SUB: tl.constexpr,  # 新增参数用于分块处理
):
    # 计算当前tile的起始位置
    tile_start = global_pid * tile_size

    # 计算子块数量
    num_sub_blocks = triton.cdiv(tile_size, BLOCK_SIZE_SUB)

    # 初始化tile累加和
    tile_sum_acc = tl.zeros([], dtype=tl.int32)

    # 按子块索引循环处理
    for sub_block_idx in range(num_sub_blocks):
        # 计算当前子块的起始位置
        sub_block_start = tile_start + sub_block_idx * BLOCK_SIZE_SUB

        # 创建子块索引
        r = tl.arange(0, BLOCK_SIZE_SUB)
        i0 = sub_block_start + r

        # 计算mask，确保不越界
        mask = (i0 < num_tasks) & (i0 >= 0)
        i0_prev = tl.where(i0 > 0, i0 - 1, 0)

        # load数据
        a = tl.load(sorted_data_ptr + i0, mask=mask, other=0)
        b = tl.load(sorted_data_ptr + i0_prev, mask=mask, other=0)

        # 计算不等式结果
        # 特殊处理第一个元素（全局索引为0的情况）
        ne_result = tl.where(i0 > 0, a != b, 0)
        ne_result = tl.where(mask, ne_result, 0)

        # 存储ne_result
        tl.store(ne_result_ptr + i0, ne_result, mask=mask)

        # 累加到tile_sum
        sub_block_sum = tl.sum(ne_result)
        tile_sum_acc += sub_block_sum

    # 存储tile累加和
    tile_sum_mask = global_pid < global_ctas_num
    tl.store(tile_sum_ptr + global_pid, tile_sum_acc, mask=tile_sum_mask)


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
            BLOCK_SIZE_SUB=256,
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
    MAX_CTAS_NUM: tl.constexpr,
    CHUNK_SIZE: tl.constexpr = 512,
):
    offset = global_pid * tile_size
    r = tl.arange(0, tile_size)
    i0 = offset + r
    mask = i0 < num_tasks

    # load sorted_data, sorted_indices
    sorted_data = tl.load(sorted_data_ptr + i0, mask=mask)
    sorted_indices = tl.load(sorted_indices_ptr + i0, mask=mask)

    # 计算需要加载的tile_sum范围
    start_idx = tl.maximum(global_pid - ctas_num, 0)
    end_idx = tl.minimum(global_pid, global_ctas_num)
    actual_load_size = end_idx - start_idx
    actual_load_size = actual_load_size.to(tl.int64)

    # 分块累加tile_sum,避免一次性分配过大的张量
    chunk_sum = 0
    chunk_sum = chunk_sum.to(tl.int64)

    for chunk_id in range(tl.cdiv(MAX_CTAS_NUM, CHUNK_SIZE)):
        # 计算当前chunk的范围
        chunk_start = chunk_id * CHUNK_SIZE
        chunk_end = tl.minimum(chunk_start + CHUNK_SIZE, actual_load_size)

        # 只在有效chunk范围内加载
        if chunk_start < actual_load_size:
            p = tl.arange(0, CHUNK_SIZE)
            p_idx = start_idx + chunk_start + p

            # 更精确的mask条件
            pre_tile_sum_mask = (
                (p < (chunk_end - chunk_start))
                & (p_idx >= start_idx)  # 当前chunk内有效
                & (p_idx < end_idx)
                & (p_idx >= 0)
                & (p_idx < global_ctas_num)
            )

            pre_tile_sum = tl.load(
                tile_sum_ptr + p_idx, mask=pre_tile_sum_mask, other=0
            )
            chunk_sum += tl.sum(pre_tile_sum)

    # cumsum
    total += chunk_sum
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
    MAX_CTAS_NUM: tl.constexpr = 65536

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
            MAX_CTAS_NUM,
        )
    else:  # grid-stride-loop style kernel
        total = tl.zeros([1], dtype=tl.int64)
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
                MAX_CTAS_NUM,
            )


def sorted_indices_unique_flat(
    sorted_data: torch.Tensor, sorted_indices: torch.Tensor, return_counts: bool
):
    num_tasks = sorted_data.numel()
    next_power_num_tasks = triton.next_power_of_2(num_tasks)
    if num_tasks >= 167772160:
        tile_size = 4096
    else:
        tile_size = min(2048, next_power_num_tasks)
    global_ctas_num = triton.cdiv(num_tasks, tile_size)
    next_power_global_ctas_num = triton.next_power_of_2(global_ctas_num)
    ctas_num = global_ctas_num if global_ctas_num < 65536 else 8192
    tiles_per_cta = triton.cdiv(num_tasks, tile_size * ctas_num)
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
            )
    return data_out[:out_size], inverse_indices, counts


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
                num_warps=8,
            )
    return data_out[:out_size], inverse_indices, counts


def _unique2(
    in0: torch.Tensor,
    sorted: bool = True,
    return_inverse: bool = False,
    return_counts: bool = False,
):
    logger.debug("GEMS_ASCEND _UNIQUE2")
    if in0.numel() <= 8192:
        sorted_data, sorted_indices = torch.sort(in0.ravel())
        data_out, inverse_indices, counts = simple_unique_flat(
            sorted_data, sorted_indices, return_inverse, return_counts
        )
    elif return_inverse:
        sorted_data, sorted_indices = torch.sort(in0.ravel())
        data_out, inverse_indices, counts = sorted_indices_unique_flat(
            sorted_data, sorted_indices, return_counts
        )
    else:
        sorted_data, _ = torch.sort(in0.ravel())
        data_out, inverse_indices, counts = sorted_quick_unique_flat(
            sorted_data, return_counts
        )
    return (
        data_out,
        inverse_indices if inverse_indices is None else inverse_indices.view_as(in0),
        counts,
    )
