import logging
import operator

import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

DEFAULT_BLOCK_SIZE = 1024
SINGLE_BLOCK_MAX_NUMEL = 16384
SMALL_COUNTS_MAX_BLOCKS = 1024


def _check_int_arg(value, name):
    if isinstance(value, bool):
        raise TypeError(f"nonzero_static(): argument '{name}' must be int, not bool")

    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError(
            f"nonzero_static(): argument '{name}' must be int, "
            f"not {type(value).__name__}"
        ) from exc


@triton.jit
def _load_nonzero_flags(x_ptr, offsets, mask, IS_COMPLEX: tl.constexpr):
    if IS_COMPLEX:
        base_offsets = offsets * 2
        real = tl.load(x_ptr + base_offsets, mask=mask, other=0)
        imag = tl.load(x_ptr + base_offsets + 1, mask=mask, other=0)
        return (real != 0) | (imag != 0)

    vals = tl.load(x_ptr + offsets, mask=mask, other=0)
    return vals != 0


@triton.jit
def _load_sparse_nonzero_flags(
    x_ptr,
    offsets,
    mask,
    IS_COMPLEX: tl.constexpr,
    IS_BFLOAT16: tl.constexpr,
):
    if IS_BFLOAT16:
        bits = tl.load(x_ptr + offsets, mask=mask, other=0)
        magnitude = bits & tl.full(bits.shape, 0x7FFF, tl.uint16)
        return magnitude != 0
    return _load_nonzero_flags(x_ptr, offsets, mask, IS_COMPLEX)


@triton.jit
def _nonzero_static_local_rank(
    flags,
    BLOCK_SIZE: tl.constexpr,
    SCAN_GROUP_SIZE: tl.constexpr,
):
    if SCAN_GROUP_SIZE == BLOCK_SIZE:
        return tl.cumsum(flags.to(tl.int32), 0) - 1

    num_groups: tl.constexpr = BLOCK_SIZE // SCAN_GROUP_SIZE
    grouped = tl.reshape(flags.to(tl.int32), (num_groups, SCAN_GROUP_SIZE))
    transposed = tl.trans(grouped, (1, 0))
    within_group = tl.cumsum(transposed, axis=0) - 1
    within_group = tl.trans(within_group, (1, 0))
    group_counts = tl.sum(grouped, axis=1)
    group_offsets = tl.cumsum(group_counts, axis=0) - group_counts
    ranks = within_group + group_offsets[:, None]
    return tl.reshape(ranks, (BLOCK_SIZE,))


@triton.jit
def _nonzero_static_store_coordinates(
    out_ptr,
    global_rank,
    linear,
    write_mask,
    ndim: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    D3: tl.constexpr,
):
    if ndim == 1:
        tl.store(out_ptr + global_rank * ndim, linear, mask=write_mask)

    if ndim == 2:
        c0 = linear // D1
        c1 = linear % D1
        tl.store(out_ptr + global_rank * ndim, c0, mask=write_mask)
        tl.store(out_ptr + global_rank * ndim + 1, c1, mask=write_mask)

    if ndim == 3:
        s0 = D1 * D2
        c0 = linear // s0
        r0 = linear % s0
        c1 = r0 // D2
        c2 = r0 % D2
        tl.store(out_ptr + global_rank * ndim, c0, mask=write_mask)
        tl.store(out_ptr + global_rank * ndim + 1, c1, mask=write_mask)
        tl.store(out_ptr + global_rank * ndim + 2, c2, mask=write_mask)

    if ndim == 4:
        s0 = D1 * D2 * D3
        s1 = D2 * D3
        c0 = linear // s0
        r0 = linear % s0
        c1 = r0 // s1
        r1 = r0 % s1
        c2 = r1 // D3
        c3 = r1 % D3
        tl.store(out_ptr + global_rank * ndim, c0, mask=write_mask)
        tl.store(out_ptr + global_rank * ndim + 1, c1, mask=write_mask)
        tl.store(out_ptr + global_rank * ndim + 2, c2, mask=write_mask)
        tl.store(out_ptr + global_rank * ndim + 3, c3, mask=write_mask)


@libentry()
@triton.jit
def _nonzero_static_count_kernel(
    x_ptr,
    counts_ptr,
    numel: tl.constexpr,
    num_blocks: tl.constexpr,
    IS_COMPLEX: tl.constexpr,
    IS_BFLOAT16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    n_workers = ext.num_programs(0)
    pid = ext.program_id(0)
    tasks_per_worker = tl.cdiv(num_blocks, n_workers)
    for task_index in range(tasks_per_worker):
        block_id = pid + task_index * n_workers
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = (block_id < num_blocks) & (offsets < numel)

        flags = _load_sparse_nonzero_flags(
            x_ptr, offsets, mask, IS_COMPLEX, IS_BFLOAT16
        )

        cnt = tl.sum(flags.to(tl.int32), axis=0)
        tl.store(counts_ptr + block_id, cnt.to(tl.int64), mask=block_id < num_blocks)


@libentry()
@triton.jit
def _nonzero_static_count_groups_kernel(
    x_ptr,
    workspace_ptr,
    numel: tl.constexpr,
    num_blocks: tl.constexpr,
    num_count_groups: tl.constexpr,
    IS_COMPLEX: tl.constexpr,
    IS_BFLOAT16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    COUNT_GROUP_BLOCKS: tl.constexpr,
):
    n_workers = ext.num_programs(0)
    pid = ext.program_id(0)
    group_tasks = tl.cdiv(num_count_groups, n_workers)
    for group_task in range(group_tasks):
        group_id = pid + group_task * n_workers
        group_count = tl.zeros((), dtype=tl.int64)
        for block_index in range(COUNT_GROUP_BLOCKS):
            block_id = group_id * COUNT_GROUP_BLOCKS + block_index
            offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = (block_id < num_blocks) & (offsets < numel)
            flags = _load_sparse_nonzero_flags(
                x_ptr, offsets, mask, IS_COMPLEX, IS_BFLOAT16
            )
            group_count += tl.sum(flags.to(tl.int32), axis=0)
        tl.store(
            workspace_ptr + group_id,
            group_count,
            mask=group_id < num_count_groups,
        )


@libentry()
@triton.jit
def _nonzero_static_fill_kernel(
    out_ptr,
    total_out: tl.constexpr,
    fill_value: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_out

    vals = tl.full((BLOCK_SIZE,), fill_value, tl.int64)
    tl.store(out_ptr + offsets, vals, mask=mask)


@libentry()
@triton.jit
def _nonzero_static_fill_tail_kernel(
    out_ptr,
    prefix_ptr,
    num_blocks: tl.constexpr,
    size: tl.constexpr,
    ndim: tl.constexpr,
    fill_value: tl.constexpr,
    fill_offset: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    total_nnz = tl.load(prefix_ptr + num_blocks - 1)
    valid_rows = tl.minimum(total_nnz, size)
    tail_start = valid_rows * ndim + fill_offset
    total_out = size * ndim

    pid = ext.program_id(0)
    offsets = tail_start + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_out

    vals = tl.full((BLOCK_SIZE,), fill_value, tl.int64)
    tl.store(out_ptr + offsets, vals, mask=mask)


@libentry()
@triton.jit
def _nonzero_static_single_block_kernel(
    x_ptr,
    out_ptr,
    size: tl.constexpr,
    numel: tl.constexpr,
    ndim: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    D3: tl.constexpr,
    fill_value: tl.constexpr,
    total_out: tl.constexpr,
    IS_COMPLEX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    flags = _load_nonzero_flags(x_ptr, offsets, mask, IS_COMPLEX)

    local_rank = tl.cumsum(flags.to(tl.int32), 0) - 1
    write_mask = mask & flags & (local_rank < size)
    linear = offsets.to(tl.int64)
    global_rank = local_rank.to(tl.int64)

    if ndim == 1:
        c0 = linear
        tl.store(out_ptr + global_rank * ndim, c0, mask=write_mask)

    if ndim == 2:
        c0 = linear // D1
        c1 = linear % D1
        tl.store(out_ptr + global_rank * ndim, c0, mask=write_mask)
        tl.store(out_ptr + global_rank * ndim + 1, c1, mask=write_mask)

    if ndim == 3:
        s0 = D1 * D2
        c0 = linear // s0
        r0 = linear % s0
        c1 = r0 // D2
        c2 = r0 % D2
        tl.store(out_ptr + global_rank * ndim, c0, mask=write_mask)
        tl.store(out_ptr + global_rank * ndim + 1, c1, mask=write_mask)
        tl.store(out_ptr + global_rank * ndim + 2, c2, mask=write_mask)

    if ndim == 4:
        s0 = D1 * D2 * D3
        s1 = D2 * D3
        c0 = linear // s0
        r0 = linear % s0
        c1 = r0 // s1
        r1 = r0 % s1
        c2 = r1 // D3
        c3 = r1 % D3
        tl.store(out_ptr + global_rank * ndim, c0, mask=write_mask)
        tl.store(out_ptr + global_rank * ndim + 1, c1, mask=write_mask)
        tl.store(out_ptr + global_rank * ndim + 2, c2, mask=write_mask)
        tl.store(out_ptr + global_rank * ndim + 3, c3, mask=write_mask)

    total_nnz = tl.sum(flags.to(tl.int32), axis=0)
    valid_rows = tl.minimum(total_nnz, size)
    tail_offsets = valid_rows * ndim + offsets
    tail_mask = tail_offsets < total_out
    tail_vals = tl.full((BLOCK_SIZE,), fill_value, tl.int64)
    tl.store(out_ptr + tail_offsets, tail_vals, mask=tail_mask)


@libentry()
@triton.jit
def _nonzero_static_single_block_generic_kernel(
    x_ptr,
    shape_ptr,
    out_ptr,
    size: tl.constexpr,
    numel: tl.constexpr,
    ndim: tl.constexpr,
    fill_value: tl.constexpr,
    total_out: tl.constexpr,
    IS_COMPLEX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    flags = _load_nonzero_flags(x_ptr, offsets, mask, IS_COMPLEX)

    local_rank = tl.cumsum(flags.to(tl.int32), 0) - 1
    global_rank = local_rank.to(tl.int64)
    write_mask = mask & flags & (local_rank < size)

    idx_flat = offsets.to(tl.int64)
    for dim in range(ndim - 1, -1, -1):
        dim_size = tl.load(shape_ptr + dim)
        coord = idx_flat % dim_size
        idx_flat //= dim_size
        tl.store(out_ptr + global_rank * ndim + dim, coord, mask=write_mask)

    total_nnz = tl.sum(flags.to(tl.int32), axis=0)
    valid_rows = tl.minimum(total_nnz, size)
    tail_offsets = valid_rows * ndim + offsets
    tail_mask = tail_offsets < total_out
    tail_vals = tl.full((BLOCK_SIZE,), fill_value, tl.int64)
    tl.store(out_ptr + tail_offsets, tail_vals, mask=tail_mask)


@libentry()
@triton.jit
def _nonzero_static_write_kernel(
    x_ptr,
    prefix_ptr,
    counts_ptr,
    out_ptr,
    size: tl.constexpr,
    numel: tl.constexpr,
    num_blocks: tl.constexpr,
    ndim: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    D3: tl.constexpr,
    fill_value: tl.constexpr,
    total_out: tl.constexpr,
    IS_COMPLEX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SCAN_GROUP_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    block_nnz = tl.load(counts_ptr + pid)
    prefix = tl.load(prefix_ptr + pid) - block_nnz

    if prefix < size:
        flags = _load_nonzero_flags(x_ptr, offsets, mask, IS_COMPLEX)
        local_rank = _nonzero_static_local_rank(
            flags,
            BLOCK_SIZE,
            SCAN_GROUP_SIZE,
        )
        global_rank = prefix + local_rank.to(tl.int64)
        write_mask = mask & flags & (global_rank < size)
        _nonzero_static_store_coordinates(
            out_ptr,
            global_rank,
            offsets.to(tl.int64),
            write_mask,
            ndim,
            D1,
            D2,
            D3,
        )

    total_nnz = tl.load(prefix_ptr + num_blocks - 1)
    valid_rows = tl.minimum(total_nnz, size)
    tail_offsets = valid_rows * ndim + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tail_mask = tail_offsets < total_out
    tail_vals = tl.full((BLOCK_SIZE,), fill_value, tl.int64)
    tl.store(out_ptr + tail_offsets, tail_vals, mask=tail_mask)


@libentry()
@triton.jit
def _nonzero_static_write_strided_kernel(
    x_ptr,
    prefix_ptr,
    counts_ptr,
    out_ptr,
    size: tl.constexpr,
    numel: tl.constexpr,
    num_blocks: tl.constexpr,
    ndim: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    D3: tl.constexpr,
    fill_value: tl.constexpr,
    total_out: tl.constexpr,
    IS_COMPLEX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SCAN_GROUP_SIZE: tl.constexpr,
):
    n_workers = ext.num_programs(0)
    pid = ext.program_id(0)
    tasks_per_worker = tl.cdiv(num_blocks, n_workers)
    total_nnz = tl.load(prefix_ptr + num_blocks - 1)
    valid_rows = tl.minimum(total_nnz, size)

    for task_index in range(tasks_per_worker):
        block_id = pid + task_index * n_workers
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = (block_id < num_blocks) & (offsets < numel)

        block_nnz = tl.load(counts_ptr + block_id, mask=block_id < num_blocks, other=0)
        prefix = (
            tl.load(prefix_ptr + block_id, mask=block_id < num_blocks, other=0)
            - block_nnz
        )

        if prefix < size:
            flags = _load_nonzero_flags(x_ptr, offsets, mask, IS_COMPLEX)
            local_rank = _nonzero_static_local_rank(
                flags,
                BLOCK_SIZE,
                SCAN_GROUP_SIZE,
            )
            global_rank = prefix + local_rank.to(tl.int64)
            write_mask = mask & flags & (global_rank < size)
            _nonzero_static_store_coordinates(
                out_ptr,
                global_rank,
                offsets.to(tl.int64),
                write_mask,
                ndim,
                D1,
                D2,
                D3,
            )

        tail_offsets = (
            valid_rows * ndim + block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        )
        tail_mask = tail_offsets < total_out
        tail_vals = tl.full((BLOCK_SIZE,), fill_value, tl.int64)
        tl.store(out_ptr + tail_offsets, tail_vals, mask=tail_mask)


@libentry()
@triton.jit
def _nonzero_static_write_sparse_groups_kernel(
    x_ptr,
    workspace_ptr,
    out_ptr,
    size: tl.constexpr,
    numel: tl.constexpr,
    num_blocks: tl.constexpr,
    num_count_groups: tl.constexpr,
    ndim: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    D3: tl.constexpr,
    fill_value: tl.constexpr,
    total_out: tl.constexpr,
    IS_COMPLEX: tl.constexpr,
    IS_BFLOAT16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SCAN_GROUP_SIZE: tl.constexpr,
    SELECT_MAX_NNZ: tl.constexpr,
    PREFIX_BLOCK_SIZE: tl.constexpr,
    COUNT_GROUP_BLOCKS: tl.constexpr,
    QUEUE_CAPACITY_PER_WORKER: tl.constexpr,
    FALLBACK_COUNTS_OFFSET: tl.constexpr,
    FALLBACK_IDS_OFFSET: tl.constexpr,
    FALLBACK_PREFIX_OFFSET: tl.constexpr,
):
    n_workers = ext.num_programs(0)
    pid = ext.program_id(0)
    group_tasks = tl.cdiv(num_count_groups, n_workers)
    count_offsets = tl.arange(0, PREFIX_BLOCK_SIZE)
    count_mask = count_offsets < num_count_groups
    worker_counts = tl.load(workspace_ptr + count_offsets, mask=count_mask, other=0)
    total_nnz = tl.sum(worker_counts, axis=0)
    valid_rows = tl.minimum(total_nnz, size)
    num_groups: tl.constexpr = BLOCK_SIZE // SCAN_GROUP_SIZE
    fallback_count = tl.zeros((), dtype=tl.int32)
    for group_task in range(group_tasks):
        count_group_id = pid + group_task * n_workers
        count_group_mask = count_group_id < num_count_groups
        count_group_prefix = tl.sum(
            tl.where(count_offsets < count_group_id, worker_counts, 0), axis=0
        )
        running_count = tl.zeros((), dtype=tl.int64)
        for block_index in range(COUNT_GROUP_BLOCKS):
            block_id = count_group_id * COUNT_GROUP_BLOCKS + block_index
            block_mask = count_group_mask & (block_id < num_blocks)
            prefix = count_group_prefix + running_count
            if block_mask & (prefix < size):
                local_offsets = tl.arange(0, BLOCK_SIZE)
                offsets = block_id * BLOCK_SIZE + local_offsets
                mask = block_mask & (offsets < numel)
                flags = _load_sparse_nonzero_flags(
                    x_ptr, offsets, mask, IS_COMPLEX, IS_BFLOAT16
                )
                grouped = tl.reshape(flags, (num_groups, SCAN_GROUP_SIZE))
                group_counts = tl.sum(grouped.to(tl.int32), axis=1)
                block_nnz = tl.sum(group_counts, axis=0)
                group_offsets = tl.cumsum(group_counts, axis=0) - group_counts
                needs_fallback = tl.sum(
                    (group_counts > SELECT_MAX_NNZ).to(tl.int32), axis=0
                )
                if needs_fallback > 0:
                    fallback_offset = pid * QUEUE_CAPACITY_PER_WORKER + fallback_count
                    tl.store(
                        workspace_ptr + FALLBACK_IDS_OFFSET + fallback_offset,
                        block_id,
                    )
                    tl.store(
                        workspace_ptr + FALLBACK_PREFIX_OFFSET + fallback_offset,
                        prefix,
                    )
                    fallback_count += 1
                group_ids = tl.arange(0, num_groups)
                positions = tl.arange(0, SCAN_GROUP_SIZE)
                remaining = grouped

                for selected_rank in range(SELECT_MAX_NNZ):
                    selected = tl.min(
                        tl.where(remaining, positions[None, :], SCAN_GROUP_SIZE),
                        axis=1,
                    )
                    linear = (
                        block_id * BLOCK_SIZE + group_ids * SCAN_GROUP_SIZE + selected
                    ).to(tl.int64)
                    global_rank = prefix + group_offsets + selected_rank
                    write_mask = (group_counts > selected_rank) & (global_rank < size)
                    _nonzero_static_store_coordinates(
                        out_ptr,
                        global_rank,
                        linear,
                        write_mask,
                        ndim,
                        D1,
                        D2,
                        D3,
                    )
                    remaining = remaining & (positions[None, :] > selected[:, None])
                running_count += block_nnz

    tail_tasks = tl.cdiv(total_out, n_workers * BLOCK_SIZE)
    for tail_task in range(tail_tasks):
        tail_offsets = valid_rows * ndim + (
            (tail_task * n_workers + pid) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        )
        tail_mask = tail_offsets < total_out
        tail_vals = tl.full((BLOCK_SIZE,), fill_value, tl.int64)
        tl.store(out_ptr + tail_offsets, tail_vals, mask=tail_mask)

    tl.store(workspace_ptr + FALLBACK_COUNTS_OFFSET + pid, fallback_count)


@libentry()
@triton.jit
def _nonzero_static_write_sparse_groups_fallback_kernel(
    x_ptr,
    workspace_ptr,
    out_ptr,
    size: tl.constexpr,
    numel: tl.constexpr,
    num_blocks: tl.constexpr,
    ndim: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    D3: tl.constexpr,
    IS_COMPLEX: tl.constexpr,
    IS_BFLOAT16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SCAN_GROUP_SIZE: tl.constexpr,
    SELECT_MAX_NNZ: tl.constexpr,
    QUEUE_CAPACITY_PER_WORKER: tl.constexpr,
    FALLBACK_COUNTS_OFFSET: tl.constexpr,
    FALLBACK_IDS_OFFSET: tl.constexpr,
    FALLBACK_PREFIX_OFFSET: tl.constexpr,
):
    pid = ext.program_id(0)
    fallback_count = tl.load(workspace_ptr + FALLBACK_COUNTS_OFFSET + pid)
    for task_index in range(QUEUE_CAPACITY_PER_WORKER):
        if task_index < fallback_count:
            fallback_offset = pid * QUEUE_CAPACITY_PER_WORKER + task_index
            block_id = tl.load(workspace_ptr + FALLBACK_IDS_OFFSET + fallback_offset)
            prefix = tl.load(workspace_ptr + FALLBACK_PREFIX_OFFSET + fallback_offset)
            offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < numel
            flags = _load_sparse_nonzero_flags(
                x_ptr, offsets, mask, IS_COMPLEX, IS_BFLOAT16
            )
            num_groups: tl.constexpr = BLOCK_SIZE // SCAN_GROUP_SIZE
            grouped = tl.reshape(flags, (num_groups, SCAN_GROUP_SIZE))
            group_counts = tl.sum(grouped.to(tl.int32), axis=1)
            group_offsets = tl.cumsum(group_counts, axis=0) - group_counts
            group_ids = tl.arange(0, num_groups)
            positions = tl.arange(0, SCAN_GROUP_SIZE)
            first = tl.min(
                tl.where(grouped, positions[None, :], SCAN_GROUP_SIZE),
                axis=1,
            )
            remaining = grouped & (positions[None, :] > first[:, None])
            second = tl.min(
                tl.where(remaining, positions[None, :], SCAN_GROUP_SIZE),
                axis=1,
            )
            linear = (block_id * BLOCK_SIZE + group_ids * SCAN_GROUP_SIZE + second).to(
                tl.int64
            )
            global_rank = prefix + group_offsets + SELECT_MAX_NNZ
            write_mask = (group_counts == SELECT_MAX_NNZ + 1) & (global_rank < size)
            _nonzero_static_store_coordinates(
                out_ptr,
                global_rank,
                linear,
                write_mask,
                ndim,
                D1,
                D2,
                D3,
            )

            needs_exact = tl.sum(
                (group_counts > SELECT_MAX_NNZ + 1).to(tl.int32), axis=0
            )
            if needs_exact > 0:
                local_rank = _nonzero_static_local_rank(
                    flags,
                    BLOCK_SIZE,
                    SCAN_GROUP_SIZE,
                )
                exact_global_rank = prefix + local_rank.to(tl.int64)
                exact_write_mask = mask & flags & (exact_global_rank < size)
                _nonzero_static_store_coordinates(
                    out_ptr,
                    exact_global_rank,
                    offsets.to(tl.int64),
                    exact_write_mask,
                    ndim,
                    D1,
                    D2,
                    D3,
                )


@libentry()
@triton.jit
def _nonzero_static_write_small_counts_kernel(
    x_ptr,
    counts_ptr,
    linear_ptr,
    out_ptr,
    size: tl.constexpr,
    numel: tl.constexpr,
    num_blocks: tl.constexpr,
    ndim: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    D3: tl.constexpr,
    fill_value: tl.constexpr,
    total_out: tl.constexpr,
    IS_COMPLEX: tl.constexpr,
    IS_BFLOAT16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PREFIX_BLOCK_SIZE: tl.constexpr,
    PREFIX_STEP_SIZE: tl.constexpr,
    SCAN_GROUP_SIZE: tl.constexpr,
    STORE_LINEAR: tl.constexpr,
):
    n_workers = ext.num_programs(0)
    pid = ext.program_id(0)
    tasks_per_worker = tl.cdiv(num_blocks, n_workers)
    count_offsets = tl.arange(0, PREFIX_BLOCK_SIZE)
    count_mask = count_offsets < num_blocks
    if not STORE_LINEAR:
        count_vals = tl.load(counts_ptr + count_offsets, mask=count_mask, other=0)
        total_nnz = tl.sum(count_vals, axis=0)
        valid_rows = tl.minimum(total_nnz, size)
    step_offsets = tl.arange(0, PREFIX_STEP_SIZE)
    prefix = tl.sum(
        tl.load(
            counts_ptr + step_offsets,
            mask=step_offsets < pid,
            other=0,
        ),
        axis=0,
    )

    for task_index in range(tasks_per_worker):
        block_id = pid + task_index * n_workers
        block_mask = block_id < num_blocks
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = block_mask & (offsets < numel)

        if block_mask & (prefix < size):
            flags = _load_sparse_nonzero_flags(
                x_ptr, offsets, mask, IS_COMPLEX, IS_BFLOAT16
            )
            local_rank = _nonzero_static_local_rank(
                flags,
                BLOCK_SIZE,
                SCAN_GROUP_SIZE,
            )
            global_rank = prefix + local_rank.to(tl.int64)
            write_mask = mask & flags & (global_rank < size)
            if STORE_LINEAR:
                tl.store(
                    linear_ptr + global_rank,
                    offsets,
                    mask=write_mask,
                )
            else:
                _nonzero_static_store_coordinates(
                    out_ptr,
                    global_rank,
                    offsets.to(tl.int64),
                    write_mask,
                    ndim,
                    D1,
                    D2,
                    D3,
                )

        if not STORE_LINEAR:
            tail_offsets = (
                valid_rows * ndim + block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            )
            tail_mask = block_mask & (tail_offsets < total_out)
            tail_vals = tl.full((BLOCK_SIZE,), fill_value, tl.int64)
            tl.store(out_ptr + tail_offsets, tail_vals, mask=tail_mask)

        if task_index + 1 < tasks_per_worker:
            next_count_offsets = block_id + step_offsets
            next_count_mask = (step_offsets < n_workers) & (
                next_count_offsets < num_blocks
            )
            prefix += tl.sum(
                tl.load(
                    counts_ptr + next_count_offsets,
                    mask=next_count_mask,
                    other=0,
                ),
                axis=0,
            )


@libentry()
@triton.jit
def _nonzero_static_expand_coordinates_kernel(
    linear_ptr,
    counts_ptr,
    out_ptr,
    size: tl.constexpr,
    num_blocks: tl.constexpr,
    ndim: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    D3: tl.constexpr,
    fill_value: tl.constexpr,
    PREFIX_BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_pid = ext.program_id(0)
    axis = ext.program_id(1)
    rows = row_pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    count_offsets = tl.arange(0, PREFIX_BLOCK_SIZE)
    count_mask = count_offsets < num_blocks
    count_vals = tl.load(counts_ptr + count_offsets, mask=count_mask, other=0)
    valid_rows = tl.minimum(tl.sum(count_vals, axis=0), size)
    row_mask = rows < size
    valid_mask = rows < valid_rows
    linear = tl.load(
        linear_ptr + rows,
        mask=valid_mask,
        other=0,
    ).to(tl.int32)

    if ndim == 2:
        c0 = linear // D1
        c1 = linear % D1
        coord = tl.where(axis == 0, c0, c1)

    if ndim == 3:
        s0 = D1 * D2
        c0 = linear // s0
        r0 = linear % s0
        c1 = r0 // D2
        c2 = r0 % D2
        coord = tl.where(axis == 0, c0, tl.where(axis == 1, c1, c2))

    if ndim == 4:
        s0 = D1 * D2 * D3
        s1 = D2 * D3
        c0 = linear // s0
        r0 = linear % s0
        c1 = r0 // s1
        r1 = r0 % s1
        c2 = r1 // D3
        c3 = r1 % D3
        coord = tl.where(
            axis == 0,
            c0,
            tl.where(axis == 1, c1, tl.where(axis == 2, c2, c3)),
        )

    values = tl.where(valid_mask, coord, fill_value)
    tl.store(out_ptr + rows * ndim + axis, values, mask=row_mask)


@libentry()
@triton.jit
def _nonzero_static_write_generic_kernel(
    x_ptr,
    shape_ptr,
    prefix_ptr,
    counts_ptr,
    out_ptr,
    size: tl.constexpr,
    numel: tl.constexpr,
    num_blocks: tl.constexpr,
    ndim: tl.constexpr,
    fill_value: tl.constexpr,
    total_out: tl.constexpr,
    IS_COMPLEX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SCAN_GROUP_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    flags = _load_nonzero_flags(x_ptr, offsets, mask, IS_COMPLEX)

    block_nnz = tl.load(counts_ptr + pid)
    prefix = tl.load(prefix_ptr + pid) - block_nnz

    local_rank = _nonzero_static_local_rank(
        flags,
        BLOCK_SIZE,
        SCAN_GROUP_SIZE,
    )
    global_rank = prefix + local_rank.to(tl.int64)
    write_mask = mask & flags & (global_rank < size)

    idx_flat = offsets.to(tl.int64)
    for dim in range(ndim - 1, -1, -1):
        dim_size = tl.load(shape_ptr + dim)
        coord = idx_flat % dim_size
        idx_flat //= dim_size
        tl.store(out_ptr + global_rank * ndim + dim, coord, mask=write_mask)

    total_nnz = tl.load(prefix_ptr + num_blocks - 1)
    valid_rows = tl.minimum(total_nnz, size)
    tail_offsets = valid_rows * ndim + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tail_mask = tail_offsets < total_out
    tail_vals = tl.full((BLOCK_SIZE,), fill_value, tl.int64)
    tl.store(out_ptr + tail_offsets, tail_vals, mask=tail_mask)


@libentry()
@triton.jit
def _nonzero_static_write_generic_strided_kernel(
    x_ptr,
    shape_ptr,
    prefix_ptr,
    counts_ptr,
    out_ptr,
    size: tl.constexpr,
    numel: tl.constexpr,
    num_blocks: tl.constexpr,
    ndim: tl.constexpr,
    fill_value: tl.constexpr,
    total_out: tl.constexpr,
    IS_COMPLEX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SCAN_GROUP_SIZE: tl.constexpr,
):
    n_workers = ext.num_programs(0)
    pid = ext.program_id(0)
    tasks_per_worker = tl.cdiv(num_blocks, n_workers)
    total_nnz = tl.load(prefix_ptr + num_blocks - 1)
    valid_rows = tl.minimum(total_nnz, size)

    for task_index in range(tasks_per_worker):
        block_id = pid + task_index * n_workers
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = (block_id < num_blocks) & (offsets < numel)

        block_nnz = tl.load(counts_ptr + block_id, mask=block_id < num_blocks, other=0)
        prefix = (
            tl.load(prefix_ptr + block_id, mask=block_id < num_blocks, other=0)
            - block_nnz
        )

        if prefix < size:
            flags = _load_nonzero_flags(x_ptr, offsets, mask, IS_COMPLEX)
            local_rank = _nonzero_static_local_rank(
                flags,
                BLOCK_SIZE,
                SCAN_GROUP_SIZE,
            )
            global_rank = prefix + local_rank.to(tl.int64)
            write_mask = mask & flags & (global_rank < size)

            idx_flat = offsets.to(tl.int64)
            for dim in range(ndim - 1, -1, -1):
                dim_size = tl.load(shape_ptr + dim)
                coord = idx_flat % dim_size
                idx_flat //= dim_size
                tl.store(out_ptr + global_rank * ndim + dim, coord, mask=write_mask)

        tail_offsets = (
            valid_rows * ndim + block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        )
        tail_mask = tail_offsets < total_out
        tail_vals = tl.full((BLOCK_SIZE,), fill_value, tl.int64)
        tl.store(out_ptr + tail_offsets, tail_vals, mask=tail_mask)


def _prepare_nonzero_static_out(input, size, out, transpose):
    expected_shape = (input.dim(), size) if transpose else (size, input.dim())
    if out.dtype != torch.int64:
        raise RuntimeError(
            f"Expected out tensor to have dtype torch.int64, but got {out.dtype} instead"
        )
    if out.device != input.device:
        raise RuntimeError(
            f"Expected out tensor to be on {input.device}, but got {out.device} instead"
        )
    if tuple(out.shape) != expected_shape:
        out.resize_(expected_shape)
    return out


def nonzero_static_ref(
    x: torch.Tensor, *, size: int, fill_value: int = -1, out: torch.Tensor = None
):
    size = _check_int_arg(size, "size")
    fill_value = _check_int_arg(fill_value, "fill_value")

    if size < 0:
        raise RuntimeError("nonzero_static: size must be non-negative")

    ndim = x.dim()
    if out is None:
        work_out = torch.empty((size, ndim), device=x.device, dtype=torch.long)
    else:
        out = _prepare_nonzero_static_out(x, size, out, transpose=False)
        work_out = out

    if size == 0:
        return _finish_nonzero_static_out(out, work_out)

    if ndim == 0:
        return _finish_nonzero_static_out(out, work_out)

    nz = torch.nonzero(x, as_tuple=False)
    copy_len = min(size, nz.shape[0])

    if copy_len > 0:
        work_out[:copy_len].copy_(nz[:copy_len])

    if copy_len < size:
        work_out[copy_len:].fill_(fill_value)

    return _finish_nonzero_static_out(out, work_out)


def _finish_nonzero_static_out(out, work_out, transpose=False):
    if out is None:
        return work_out
    if transpose:
        out.copy_(work_out.transpose(0, 1))
    else:
        out.copy_(work_out)
    return out


def _nonzero_static_impl(
    input: torch.Tensor,
    *,
    size: int,
    fill_value: int = -1,
    out: torch.Tensor = None,
    cumsum_fn=None,
    transpose_out=True,
    block_size=DEFAULT_BLOCK_SIZE,
    single_block_max_numel=SINGLE_BLOCK_MAX_NUMEL,
    small_counts_max_blocks=SMALL_COUNTS_MAX_BLOCKS,
    max_programs=None,
    scan_group_size=None,
    use_sparse_groups=False,
    sparse_group_select_max_nnz=0,
    sparse_scan_group_size=None,
    sparse_count_group_blocks=1,
    small_counts_linear_output=False,
    use_bfloat16_bits=False,
):
    size = _check_int_arg(size, "size")
    fill_value = _check_int_arg(fill_value, "fill_value")

    if size < 0:
        raise RuntimeError("nonzero_static: size must be non-negative")

    ndim = input.dim()

    if input.device.type != flag_gems.device:
        return nonzero_static_ref(input, size=size, fill_value=fill_value, out=out)

    if out is None:
        work_out = torch.empty((size, ndim), device=input.device, dtype=torch.int64)
    else:
        out = _prepare_nonzero_static_out(input, size, out, transpose=transpose_out)
        work_out = torch.empty((size, ndim), device=input.device, dtype=torch.int64)

    if size == 0:
        return _finish_nonzero_static_out(
            out, work_out, transpose=transpose_out and out is not None
        )

    if ndim == 0:
        return _finish_nonzero_static_out(
            out, work_out, transpose=transpose_out and out is not None
        )

    is_complex = input.is_complex()
    source = input.contiguous()
    if is_complex:
        x = torch.view_as_real(source).reshape(-1)
        numel = source.numel()
    else:
        x = source
        numel = x.numel()
    is_bfloat16_bits = use_bfloat16_bits and source.dtype == torch.bfloat16
    count_x = source.view(torch.uint16) if is_bfloat16_bits else x

    total_out = size * ndim
    if scan_group_size is None:
        scan_group_size = block_size
    if sparse_scan_group_size is None:
        sparse_scan_group_size = scan_group_size

    if numel == 0:
        fill_grid = (triton.cdiv(total_out, block_size),)
        with torch_device_fn.device(input.device):
            _nonzero_static_fill_kernel[fill_grid](
                work_out,
                total_out,
                fill_value,
                BLOCK_SIZE=block_size,
            )
        return _finish_nonzero_static_out(
            out, work_out, transpose=transpose_out and out is not None
        )

    num_blocks = triton.cdiv(numel, block_size)
    program_count = (
        num_blocks if max_programs is None else min(num_blocks, max_programs)
    )
    use_generic_ndim = ndim > 4

    if use_generic_ndim:
        shape = torch.tensor(input.shape, dtype=torch.int64, device=input.device)
    else:
        shape = list(input.shape) + [1] * (4 - ndim)

    single_block_elems = max(numel, total_out)
    if single_block_elems <= single_block_max_numel:
        single_block_size = 1 << (single_block_elems - 1).bit_length()
        with torch_device_fn.device(input.device):
            if use_generic_ndim:
                _nonzero_static_single_block_generic_kernel[(1,)](
                    x,
                    shape,
                    work_out,
                    size,
                    numel,
                    ndim,
                    fill_value,
                    total_out,
                    IS_COMPLEX=is_complex,
                    BLOCK_SIZE=single_block_size,
                )
            else:
                _nonzero_static_single_block_kernel[(1,)](
                    x,
                    work_out,
                    size,
                    numel,
                    ndim,
                    shape[0],
                    shape[1],
                    shape[2],
                    shape[3],
                    fill_value,
                    total_out,
                    IS_COMPLEX=is_complex,
                    BLOCK_SIZE=single_block_size,
                )
        return _finish_nonzero_static_out(
            out, work_out, transpose=transpose_out and out is not None
        )

    if use_sparse_groups and not use_generic_ndim and max_programs is not None:
        is_bfloat16 = source.dtype == torch.bfloat16
        sparse_x = source.view(torch.uint16) if is_bfloat16 else x
        num_count_groups = triton.cdiv(num_blocks, sparse_count_group_blocks)
        group_tasks = triton.cdiv(num_count_groups, program_count)
        queue_capacity_per_worker = group_tasks * sparse_count_group_blocks
        fallback_capacity = program_count * queue_capacity_per_worker
        fallback_counts_offset = num_count_groups
        fallback_ids_offset = fallback_counts_offset + program_count
        fallback_prefix_offset = fallback_ids_offset + fallback_capacity
        workspace = torch.empty(
            (fallback_prefix_offset + fallback_capacity,),
            device=input.device,
            dtype=torch.int64,
        )
        prefix_block_size = 1 << (num_count_groups - 1).bit_length()
        with torch_device_fn.device(input.device):
            _nonzero_static_count_groups_kernel[(program_count,)](
                sparse_x,
                workspace,
                numel,
                num_blocks,
                num_count_groups,
                IS_COMPLEX=is_complex,
                IS_BFLOAT16=is_bfloat16,
                BLOCK_SIZE=block_size,
                COUNT_GROUP_BLOCKS=sparse_count_group_blocks,
            )
            _nonzero_static_write_sparse_groups_kernel[(program_count,)](
                sparse_x,
                workspace,
                work_out,
                size,
                numel,
                num_blocks,
                num_count_groups,
                ndim,
                shape[0],
                shape[1],
                shape[2],
                shape[3],
                fill_value,
                total_out,
                IS_COMPLEX=is_complex,
                IS_BFLOAT16=is_bfloat16,
                BLOCK_SIZE=block_size,
                SCAN_GROUP_SIZE=sparse_scan_group_size,
                SELECT_MAX_NNZ=sparse_group_select_max_nnz,
                PREFIX_BLOCK_SIZE=prefix_block_size,
                COUNT_GROUP_BLOCKS=sparse_count_group_blocks,
                QUEUE_CAPACITY_PER_WORKER=queue_capacity_per_worker,
                FALLBACK_COUNTS_OFFSET=fallback_counts_offset,
                FALLBACK_IDS_OFFSET=fallback_ids_offset,
                FALLBACK_PREFIX_OFFSET=fallback_prefix_offset,
            )
            _nonzero_static_write_sparse_groups_fallback_kernel[(program_count,)](
                sparse_x,
                workspace,
                work_out,
                size,
                numel,
                num_blocks,
                ndim,
                shape[0],
                shape[1],
                shape[2],
                shape[3],
                IS_COMPLEX=is_complex,
                IS_BFLOAT16=is_bfloat16,
                BLOCK_SIZE=block_size,
                SCAN_GROUP_SIZE=sparse_scan_group_size,
                SELECT_MAX_NNZ=sparse_group_select_max_nnz,
                QUEUE_CAPACITY_PER_WORKER=queue_capacity_per_worker,
                FALLBACK_COUNTS_OFFSET=fallback_counts_offset,
                FALLBACK_IDS_OFFSET=fallback_ids_offset,
                FALLBACK_PREFIX_OFFSET=fallback_prefix_offset,
            )

        return _finish_nonzero_static_out(
            out, work_out, transpose=transpose_out and out is not None
        )

    counts = torch.empty((num_blocks,), device=input.device, dtype=torch.int64)

    with torch_device_fn.device(input.device):
        _nonzero_static_count_kernel[(program_count,)](
            count_x,
            counts,
            numel,
            num_blocks,
            IS_COMPLEX=is_complex,
            IS_BFLOAT16=is_bfloat16_bits,
            BLOCK_SIZE=block_size,
        )

    if (
        not use_sparse_groups
        and not use_generic_ndim
        and num_blocks <= small_counts_max_blocks
        and total_out <= num_blocks * block_size
    ):
        prefix_block_size = 1 << (num_blocks - 1).bit_length()
        prefix_step_size = 1 << (program_count - 1).bit_length()
        linear_out = (
            torch.empty((size,), device=input.device, dtype=torch.int64)
            if small_counts_linear_output
            else work_out
        )
        with torch_device_fn.device(input.device):
            _nonzero_static_write_small_counts_kernel[(program_count,)](
                count_x,
                counts,
                linear_out,
                work_out,
                size,
                numel,
                num_blocks,
                ndim,
                shape[0],
                shape[1],
                shape[2],
                shape[3],
                fill_value,
                total_out,
                IS_COMPLEX=is_complex,
                IS_BFLOAT16=is_bfloat16_bits,
                BLOCK_SIZE=block_size,
                PREFIX_BLOCK_SIZE=prefix_block_size,
                PREFIX_STEP_SIZE=prefix_step_size,
                SCAN_GROUP_SIZE=scan_group_size,
                STORE_LINEAR=small_counts_linear_output,
            )
            if small_counts_linear_output:
                expand_block_size = min(1024, 1 << (size - 1).bit_length())
                _nonzero_static_expand_coordinates_kernel[
                    (triton.cdiv(size, expand_block_size), ndim)
                ](
                    linear_out,
                    counts,
                    work_out,
                    size,
                    num_blocks,
                    ndim,
                    shape[1],
                    shape[2],
                    shape[3],
                    fill_value,
                    PREFIX_BLOCK_SIZE=prefix_block_size,
                    BLOCK_SIZE=expand_block_size,
                )
        return _finish_nonzero_static_out(
            out, work_out, transpose=transpose_out and out is not None
        )

    if cumsum_fn is None:
        cumsum_fn = flag_gems.cumsum
    prefix = cumsum_fn(counts, dim=0)

    with torch_device_fn.device(input.device):
        if use_generic_ndim:
            write_kernel = (
                _nonzero_static_write_generic_strided_kernel
                if max_programs is not None
                else _nonzero_static_write_generic_kernel
            )
            write_kernel[(program_count,)](
                x,
                shape,
                prefix,
                counts,
                work_out,
                size,
                numel,
                num_blocks,
                ndim,
                fill_value,
                total_out,
                IS_COMPLEX=is_complex,
                BLOCK_SIZE=block_size,
                SCAN_GROUP_SIZE=scan_group_size,
            )
        else:
            write_kernel = (
                _nonzero_static_write_strided_kernel
                if max_programs is not None
                else _nonzero_static_write_kernel
            )
            write_kernel[(program_count,)](
                x,
                prefix,
                counts,
                work_out,
                size,
                numel,
                num_blocks,
                ndim,
                shape[0],
                shape[1],
                shape[2],
                shape[3],
                fill_value,
                total_out,
                IS_COMPLEX=is_complex,
                BLOCK_SIZE=block_size,
                SCAN_GROUP_SIZE=scan_group_size,
            )

    if total_out > num_blocks * block_size:
        filled_tail_elems = num_blocks * block_size
        fill_grid = (triton.cdiv(total_out - filled_tail_elems, block_size),)
        with torch_device_fn.device(input.device):
            _nonzero_static_fill_tail_kernel[fill_grid](
                work_out,
                prefix,
                num_blocks,
                size,
                ndim,
                fill_value,
                filled_tail_elems,
                BLOCK_SIZE=block_size,
            )

    return _finish_nonzero_static_out(
        out, work_out, transpose=transpose_out and out is not None
    )


def nonzero_static(input: torch.Tensor, *, size: int, fill_value: int = -1):
    logger.debug("GEMS NONZERO_STATIC")
    return _nonzero_static_impl(input, size=size, fill_value=fill_value)


def nonzero_static_out(
    input: torch.Tensor, *, size: int, fill_value: int = -1, out: torch.Tensor
):
    logger.debug("GEMS NONZERO_STATIC_OUT")
    return _nonzero_static_impl(input, size=size, fill_value=fill_value, out=out)
