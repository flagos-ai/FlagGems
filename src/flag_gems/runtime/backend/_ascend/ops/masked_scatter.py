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
@triton.jit
def count_mask_per_block_kernel(mask_ptr, counts_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    block_mask = offset < N
    mask_val = tl.load(mask_ptr + offset, mask=block_mask, other=0).to(tl.int32)
    count = tl.sum(mask_val)
    tl.store(counts_ptr + pid, count)


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
    true_count = mask.sum().item()
    if true_count == 0:
        return inp

    if N <= 4096:
        BLOCK_SIZE = triton.next_power_of_2(N)
        masked_scatter_single_pass_kernel[(1,)](
            inp, mask, source, N, BLOCK_SIZE=BLOCK_SIZE
        )
        return inp

    BLOCK_SIZE = bracket_next_power_of_2(N, 128, 4096)
    n_blocks = triton.cdiv(N, BLOCK_SIZE)

    with torch_device_fn.device(inp.device):
        block_counts = torch.empty(n_blocks, dtype=torch.int64, device=mask.device)
        count_mask_per_block_kernel[(n_blocks,)](
            mask, block_counts, N, BLOCK_SIZE=BLOCK_SIZE
        )

        counts_cpu = block_counts.cpu().to(torch.int64)
        prefix_sum = torch.zeros(n_blocks, dtype=torch.int64)
        torch.cumsum(counts_cpu[:-1], dim=0, out=prefix_sum[1:])
        part_sums = prefix_sum.to(mask.device)

        masked_scatter_kernel[(n_blocks,)](
            inp,
            mask,
            source,
            part_sums,
            N,
            n_blocks,
            1,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return inp


def masked_scatter(inp, mask, source):
    logger.debug("GEMS_ASCEND MASKED SCATTER")

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
    logger.debug("GEMS_ASCEND MASKED SCATTER_")

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
