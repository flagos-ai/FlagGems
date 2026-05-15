import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import broadcastable, libentry
from flag_gems.utils.shape_utils import bracket_next_power_of_2

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def masked_scatter_single_pass_kernel(
    inp_ptr, mask_ptr, src_ptr, N, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    block_mask = offsets < N

    mask_val = tl.load(mask_ptr + offsets, mask=block_mask, other=0).to(tl.int1)

    mask_ints = mask_val.to(tl.int32)
    src_indices = tl.cumsum(mask_ints, axis=0) - 1

    active = block_mask & mask_val
    src_val = tl.load(src_ptr + src_indices, mask=active)
    tl.store(inp_ptr + offsets, src_val, mask=active)


@libentry()
@triton.jit(do_not_specialize=["N", "num_blocks", "num_blocks_per_row"])
def mask_part_sum_kernel(
    mask_ptr,
    part_sums_ptr,
    counter_ptr,
    N,
    num_blocks,
    num_blocks_per_row,
    NP_BLOCK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    start_block = row_id * num_blocks_per_row
    offset = start_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros((BLOCK_SIZE,), dtype=part_sums_ptr.dtype.element_ty)

    last_block_id = min(num_blocks - 1, start_block + num_blocks_per_row - 1)

    for block_id in range(start_block, last_block_id):
        select = tl.load(mask_ptr + offset)
        select_ints = select.to(part_sums_ptr.dtype.element_ty)
        acc += select_ints
        offset += BLOCK_SIZE

    select = tl.load(mask_ptr + offset, mask=offset < N, other=0)
    select_ints = select.to(part_sums_ptr.dtype.element_ty)
    acc += select_ints

    part_sum = tl.sum(acc, axis=0)
    tl.store(part_sums_ptr + row_id, part_sum)

    count = tl.atomic_add(counter_ptr, 1, sem="acq_rel")
    np = tl.num_programs(0)

    if count == np - 1:
        mask = tl.arange(0, NP_BLOCK) < np
        part_sums = tl.load(part_sums_ptr + tl.arange(0, NP_BLOCK), mask=mask)
        final_sum = tl.sum(part_sums, axis=0)
        pre_sums = tl.cumsum(part_sums, axis=0)
        tl.store(
            part_sums_ptr + tl.arange(0, NP_BLOCK), pre_sums - part_sums, mask=mask
        )
        tl.store(part_sums_ptr + np, final_sum)


@libentry()
@triton.jit(do_not_specialize=["N", "num_blocks", "num_blocks_per_row"])
def masked_scatter_kernel(
    inp_ptr,
    mask_ptr,
    src_ptr,
    part_sums_ptr,
    N,
    num_blocks,
    num_blocks_per_row,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)

    start_block = row_id * num_blocks_per_row
    offset = start_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    advance = tl.load(part_sums_ptr + row_id)

    last_block_id = min(num_blocks - 1, start_block + num_blocks_per_row - 1)

    for block_id in range(start_block, last_block_id):
        select_mask = tl.load(mask_ptr + offset).to(tl.int1)
        select_ints = select_mask.to(tl.int32)

        block_cumsum = tl.cumsum(select_ints, axis=0) - 1
        global_src_idx = advance + block_cumsum

        advance += tl.sum(select_ints, axis=0)

        src_val = tl.load(src_ptr + global_src_idx, mask=select_mask)
        tl.store(inp_ptr + offset, src_val, mask=select_mask)

        offset += BLOCK_SIZE

    block_mask = offset < N
    select_mask = tl.load(mask_ptr + offset, mask=block_mask, other=0).to(tl.int1)

    select_ints = select_mask.to(tl.int32)
    block_cumsum = tl.cumsum(select_ints, axis=0) - 1
    global_src_idx = advance + block_cumsum

    active = block_mask & select_mask
    src_val = tl.load(src_ptr + global_src_idx, mask=active)
    tl.store(inp_ptr + offset, src_val, mask=active)


def masked_scatter_impl(inp, mask, source, N):
    if N <= 4096:
        BLOCK_SIZE = triton.next_power_of_2(N)
        num_warps = 4
        if BLOCK_SIZE >= 2048:
            num_warps = 8
        if BLOCK_SIZE >= 4096:
            num_warps = 16

        masked_scatter_single_pass_kernel[(1,)](
            inp, mask, source, N, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps
        )
        return inp

    BLOCK_SIZE = bracket_next_power_of_2(N, 128, 4096)
    num_warps = min(16, BLOCK_SIZE // 32)

    np = torch_device_fn.get_device_properties(mask.device).multi_processor_count
    n_blocks = triton.cdiv(N, BLOCK_SIZE)
    np = min(n_blocks, np)
    n_blocks_per_row = triton.cdiv(n_blocks, np)
    np = triton.cdiv(n_blocks, n_blocks_per_row)
    NP_BLOCK = triton.next_power_of_2(np)

    with torch_device_fn.device(inp.device):
        dtype = torch.int32 if N < 2**31 else torch.int64
        part_sums = torch.empty(np + 1, dtype=dtype, device=mask.device)
        barrier = torch.zeros([], dtype=torch.int, device=mask.device)

        mask_part_sum_kernel[(np,)](
            mask,
            part_sums,
            barrier,
            N,
            n_blocks,
            n_blocks_per_row,
            NP_BLOCK=NP_BLOCK,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

        masked_scatter_kernel[(np,)](
            inp,
            mask,
            source,
            part_sums,
            N,
            n_blocks,
            n_blocks_per_row,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

    return inp


def masked_scatter(inp, mask, source):
    logger.debug("GEMS MASKED SCATTER")

    assert broadcastable(
        inp.shape, mask.shape
    ), "The shapes of the `mask` and the `input` tensor must be broadcastable"

    _, mask = torch.broadcast_tensors(inp, mask)

    out = inp.clone()
    if not out.is_contiguous():
        out = out.contiguous()
    if not mask.is_contiguous():
        mask = mask.contiguous()
    if not source.is_contiguous():
        source = source.contiguous()

    N = out.numel()

    masked_scatter_impl(out, mask, source, N)

    return out


def masked_scatter_(inp, mask, source):
    logger.debug("GEMS MASKED SCATTER_")

    assert broadcastable(inp.shape, mask.shape)
    _, mask = torch.broadcast_tensors(inp, mask)

    if not inp.is_contiguous():
        raise RuntimeError(
            "in-place operation currently requires contiguous input tensor. "
        )

    mask = mask if mask.is_contiguous() else mask.contiguous()
    source = source if source.is_contiguous() else source.contiguous()

    N = inp.numel()
    masked_scatter_impl(inp, mask, source, N)

    return inp
