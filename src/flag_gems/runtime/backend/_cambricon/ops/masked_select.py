import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import broadcastable

from ..utils import TOTAL_CORE_NUM

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@triton.autotune(configs=runtime.get_tuned_config("masked_select"), key=["n_elements"])
@triton.jit
def masked_select_kernel(
    inp_ptr,
    select_mask_ptr,
    select_val_ptr,
    select_num_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_p = tl.num_programs(axis=0)
    split_n = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    step = BLOCK_SIZE * num_p
    offset_start = pid * BLOCK_SIZE
    loop = 0
    for offset in tl.range(offset_start, n_elements, step):
        offsets = offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        inp = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
        select_mask = tl.load(select_mask_ptr + offsets, mask=mask, other=0.0).to(
            tl.int1
        )
        select_val, select_num = tl.masked_select(inp, select_mask)
        tl.store(select_val_ptr + offsets, select_val, mask=mask)
        num_select_offset = loop * num_p + pid + tl.arange(0, 1)
        loop += 1
        num_select_mask = num_select_offset < split_n
        tl.store(select_num_ptr + num_select_offset, select_num, mask=num_select_mask)


@triton.jit
def get_out_kernel(
    select_val_ptr,
    select_num_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_p = tl.num_programs(axis=0)
    step = BLOCK_SIZE * num_p
    split_n: tl.constexpr = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    all_select_num_offset = tl.arange(0, split_n)
    all_select_num_mask = all_select_num_offset < split_n
    all_select_num = tl.load(
        select_num_ptr + all_select_num_offset, mask=all_select_num_mask, other=0.0
    )
    prefix_select_num = tl.cumsum(all_select_num, 0)

    offset_start = pid * BLOCK_SIZE
    loop = 0
    for offset in tl.range(offset_start, n_elements, step):
        offsets = offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        select_val = tl.load(select_val_ptr + offsets, mask=mask, other=0.0)
        select_num_offset = loop * num_p + pid + tl.arange(0, 1)
        select_num_mask = select_num_offset < split_n
        select_num = tl.load(
            select_num_ptr + select_num_offset, mask=select_num_mask, other=0.0
        )
        if loop == 0 and pid == 0:
            output_offset = tl.arange(0, BLOCK_SIZE)
        else:
            output_offset = prefix_select_num[loop * num_p + pid - 1] + tl.arange(
                0, BLOCK_SIZE
            )
        loop += 1
        output_mask = tl.arange(0, BLOCK_SIZE) < select_num
        tl.store(output_ptr + output_offset, select_val, mask=output_mask)


def masked_select(inp, mask):
    logger.debug("GEMS_CAMBRICON MASKED SELECT")

    inp_shape = tuple(inp.shape)
    mask_shape = tuple(mask.shape)

    assert broadcastable(
        inp_shape, mask_shape
    ), "The shapes of the `mask` and the `input` tensor must be broadcastable"
    inp, mask = torch.broadcast_tensors(inp, mask)

    inp = inp.contiguous()
    mask = mask.contiguous()

    n_elements = inp.numel()
    grid = lambda meta: (
        min(triton.cdiv(n_elements, meta["BLOCK_SIZE"]), TOTAL_CORE_NUM),
    )
    with torch_device_fn.device(inp.device):
        select_val = torch.empty(n_elements, dtype=inp.dtype, device=inp.device)
        select_num = torch.empty(n_elements, dtype=torch.int32, device=inp.device)
        masked_select_kernel[grid](inp, mask, select_val, select_num, n_elements)

        cur_block_size = masked_select_kernel.best_config.kwargs["BLOCK_SIZE"]
        num_select = mask.sum().item()
        output = torch.empty(num_select, dtype=inp.dtype, device=inp.device)
        get_out_kernel[grid](select_val, select_num, output, n_elements, cur_block_size)

    return output
