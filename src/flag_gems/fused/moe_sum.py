import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


def _get_moe_sum_config(
    num_tokens: int,
    hidden_size: int,
    topk: int,
):
    if hidden_size <= 64:
        block_size = 64
        num_warps = 1
    elif hidden_size <= 128:
        block_size = 128
        num_warps = 2
    elif hidden_size <= 256:
        block_size = 256
        num_warps = 2 if num_tokens < 8 and topk <= 2 else 4
    elif hidden_size <= 512:
        block_size = 512
        num_warps = 4
    else:
        block_size = 1024
        if num_tokens < 4 and topk <= 2:
            num_warps = 4
        else:
            num_warps = 8

    return {
        "BLOCK_SIZE": block_size,
        "num_warps": num_warps,
        "num_stages": 1,
    }


@triton.jit
def moe_sum_kernel(
    input_ptr,
    output_ptr,
    num_tokens,
    topk: tl.constexpr,
    hidden_size,
    input_stride_token,
    input_stride_topk,
    input_stride_hidden,
    output_stride_token,
    output_stride_hidden,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    hidden_start = block_idx * BLOCK_SIZE
    hidden_offsets = hidden_start + tl.arange(0, BLOCK_SIZE)
    hidden_mask = hidden_offsets < hidden_size
    if token_idx >= num_tokens:
        return
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    input_base = input_ptr + token_idx * input_stride_token

    for expert_idx in tl.static_range(topk):
        expert_ptr = input_base + expert_idx * input_stride_topk
        expert_data = tl.load(expert_ptr + hidden_offsets, mask=hidden_mask, other=0.0)
        acc += expert_data
    output_ptr_pos = output_ptr + token_idx * output_stride_token + hidden_offsets

    tl.store(
        output_ptr_pos,
        acc.to(tl.float16) if input_ptr.dtype.element_ty == tl.float16 else acc,
        mask=hidden_mask,
    )


def moe_sum(
    input: torch.Tensor,
    output: torch.Tensor,
):
    logger.debug("GEMS MOE SUM")
    num_tokens, topk, hidden_size = input.shape
    input_strides = input.stride()
    output_strides = output.stride()
    config = _get_moe_sum_config(num_tokens, hidden_size, topk)
    grid = lambda meta: (num_tokens, triton.cdiv(hidden_size, meta["BLOCK_SIZE"]))
    moe_sum_kernel[grid](
        input,
        output,
        num_tokens,
        topk,
        hidden_size,
        input_strides[0],
        input_strides[1],
        input_strides[2],
        output_strides[0],
        output_strides[1],
        BLOCK_SIZE=config["BLOCK_SIZE"],
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
