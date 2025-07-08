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
def masked_select_single_pass_kernel(
    inp_ptr, mask_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp = tl.load(inp_ptr + offsets, mask=offsets < N)
    mask = tl.load(mask_ptr + offsets, mask=offsets < N).to(tl.int1)
    mask_ints = mask.to(tl.int32)
    out_offsets = tl.cumsum(mask_ints, axis=0) - 1

    tl.store(out_ptr + out_offsets, inp, mask=offsets < N and mask)


def masked_select_single_pass(inp, mask, out, N):
    BLOCK_SIZE = triton.next_power_of_2(N)
    if BLOCK_SIZE <= 512:
        num_warps = 4
    elif BLOCK_SIZE <= 2048:
        num_warps = 8
    else:
        num_warps = 16
    masked_select_single_pass_kernel[(1,)](
        inp, mask, out, N, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps
    )
    return out


@libentry()
@triton.jit(do_not_specialize=["N", "nr", "row_stride"])
def mask_part_sum_kernel(
    inp_ptr,
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
    # Peeled last block
    select = tl.load(mask_ptr + offset, mask=offset < N, other=0)
    select_ints = select.to(part_sums_ptr.dtype.element_ty)
    acc += select_ints

    part_sum = tl.sum(acc, axis=0)
    tl.store(part_sums_ptr + row_id, part_sum)
    # cumsum the part_sums
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
@triton.jit(do_not_specialize=["N", "nr", "row_stride"])
def write_back_kernel(
    inp_ptr,
    mask_ptr,
    part_sums_ptr,
    out_ptr,
    N,
    num_blocks,
    num_blocks_per_row,
    NP_BLOCK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)

    start_block = row_id * num_blocks_per_row
    offset = start_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    advance = tl.load(part_sums_ptr + row_id)

    last_block_id = min(num_blocks - 1, start_block + num_blocks_per_row - 1)

    for block_id in range(start_block, last_block_id):
        inp = tl.load(inp_ptr + offset)
        select_mask = tl.load(mask_ptr + offset).to(tl.int1)
        select_ints = select_mask.to(tl.constexpr(part_sums_ptr.dtype.element_ty))
        out_ptr += advance
        advance = tl.sum(select_ints, axis=0)
        pre_sums = tl.cumsum(select_ints, axis=0) - 1
        tl.store(out_ptr + pre_sums, inp, mask=select_mask)
        offset += BLOCK_SIZE
    # Peeled last block
    inp = tl.load(inp_ptr + offset, mask=offset < N)
    select_mask = tl.load(mask_ptr + offset, mask=offset < N, other=0).to(tl.int1)
    select_ints = select_mask.to(tl.constexpr(part_sums_ptr.dtype.element_ty))
    out_ptr += advance
    pre_sums = tl.cumsum(select_ints, axis=0) - 1
    tl.store(out_ptr + pre_sums, inp, mask=offset < N and select_mask)


def masked_select(inp, mask):
    logger.debug("GEMS MASKED SELECT")

    inp_shape = tuple(inp.shape)
    mask_shape = tuple(mask.shape)

    assert broadcastable(
        inp_shape, mask_shape
    ), "The shapes of the `mask` and the `input` tensor must be broadcastable"
    inp, mask = torch.broadcast_tensors(inp, mask)

    inp = inp.contiguous()
    mask = mask.contiguous()

    N = inp.numel()
    if N <= 4096:
        out = torch.empty(mask.sum(), dtype=inp.dtype, device=inp.device)
        return masked_select_single_pass(inp, mask, out, N)

    # return mask_select(inp, mask)

    BLOCK_SIZE = bracket_next_power_of_2(N, 128, 4096)
    num_warps = min(16, BLOCK_SIZE // 32)

    # max degree of parallelism
    np = torch_device_fn.get_device_properties(mask.device).multi_processor_count

    # arranged as np rows of blocks
    n_blocks = triton.cdiv(N, BLOCK_SIZE)
    np = min(n_blocks, np)
    n_blocks_per_row = triton.cdiv(n_blocks, np)
    np = triton.cdiv(n_blocks, n_blocks_per_row)
    NP_BLOCK = triton.next_power_of_2(np)

    with torch_device_fn.device(inp.device):
        # Compute per cta sums and cumulative sums across ctas
        dtype = torch.int32 if N < 2**31 else torch.int64
        part_sums = torch.empty(np + 1, dtype=dtype, device=mask.device)
        barrier = torch.zeros([], dtype=torch.int, device=mask.device)
        mask_part_sum_kernel[(np,)](
            inp,
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

        # Write back selected data
        out = torch.empty(part_sums[-1], dtype=inp.dtype, device=mask.device)
        # write_offsets = pre_sums - part_sums
        write_back_kernel[(np,)](
            inp,
            mask,
            part_sums,
            out,
            N,
            n_blocks,
            n_blocks_per_row,
            NP_BLOCK=triton.next_power_of_2(np),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

    return out
