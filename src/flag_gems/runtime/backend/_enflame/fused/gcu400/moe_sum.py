import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit(do_not_specialize=["num_tokens", "topk", "hidden_size",
                                "input_stride_token", "input_stride_topk",
                                "output_stride_token"])
def moe_sum_kernel(
    input_ptr,
    output_ptr,
    num_tokens,
    topk,
    hidden_size,
    input_stride_token,
    input_stride_topk,
    output_stride_token,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    num_h_blocks = tl.cdiv(hidden_size, BLOCK_SIZE)
    total_blocks = num_tokens * num_h_blocks

    block_id = pid
    while block_id < total_blocks:
        token_idx = block_id // num_h_blocks
        h_block_idx = block_id % num_h_blocks
        h_start = h_block_idx * BLOCK_SIZE

        acc = tl.zeros((BLOCK_TOPK, BLOCK_SIZE), dtype=tl.float32)
        for topk_start in tl.range(0, topk, BLOCK_TOPK):
            in_block = tl.make_block_ptr(
                base=input_ptr + token_idx * input_stride_token,
                shape=(topk, hidden_size),
                strides=(input_stride_topk, 1),
                offsets=(topk_start, h_start),
                block_shape=(BLOCK_TOPK, BLOCK_SIZE),
                order=(1, 0),
            )
            tile = tl.load(in_block, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
            acc += tile

        result = tl.sum(acc, axis=0)

        out_block = tl.make_block_ptr(
            base=output_ptr + token_idx * output_stride_token,
            shape=(hidden_size,),
            strides=(1,),
            offsets=(h_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )
        tl.store(out_block, result.to(output_ptr.dtype.element_ty), boundary_check=(0,))

        block_id += num_programs


def moe_sum(
    input: torch.Tensor,
    output: torch.Tensor,
):
    logger.debug("GEMS MOE SUM (GCU400)")
    num_tokens, topk, hidden_size = input.shape
    input_strides = input.stride()
    output_strides = output.stride()

    if topk == 1:
        output.copy_(input.squeeze(1))
        return

    BLOCK_SIZE = min(triton.next_power_of_2(hidden_size), 2048)
    BLOCK_TOPK = min(triton.next_power_of_2(topk), 64)
    # DSM constraint: BLOCK_TOPK * BLOCK_SIZE <= 32768
    while BLOCK_TOPK * BLOCK_SIZE > 32768 and BLOCK_TOPK > 1:
        BLOCK_TOPK //= 2

    total_blocks = num_tokens * triton.cdiv(hidden_size, BLOCK_SIZE)
    num_programs = min(total_blocks, 48)

    moe_sum_kernel[(num_programs,)](
        input,
        output,
        num_tokens,
        topk,
        hidden_size,
        input_strides[0],
        input_strides[1],
        output_strides[0],
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_TOPK=BLOCK_TOPK,
        num_warps=1,
    )
