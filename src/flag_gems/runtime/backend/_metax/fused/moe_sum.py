import torch
import triton
import triton.language as tl


@triton.jit
def _metax_moe_sum_kernel(
    input_ptr,
    output_ptr,
    num_tokens,
    hidden_size,
    input_stride_token,
    input_stride_topk,
    input_stride_hidden,
    output_stride_token,
    output_stride_hidden,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    hidden_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    hidden_mask = hidden_offsets < hidden_size

    if token_idx >= num_tokens:
        return

    input_base = input_ptr + token_idx * input_stride_token
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for expert_idx in tl.static_range(0, TOPK):
        expert_ptr = input_base + expert_idx * input_stride_topk
        expert_data = tl.load(
            expert_ptr + hidden_offsets * input_stride_hidden,
            mask=hidden_mask,
            other=0.0,
        )
        acc += expert_data

    output_ptrs = (
        output_ptr
        + token_idx * output_stride_token
        + hidden_offsets * output_stride_hidden
    )
    tl.store(output_ptrs, acc, mask=hidden_mask)


def _metax_moe_sum_block_size(hidden_size: int) -> int:
    if hidden_size <= 256:
        return 256
    if hidden_size <= 512:
        return 512
    return 1024


def moe_sum(input: torch.Tensor, output: torch.Tensor):
    num_tokens, topk, hidden_size = input.shape
    input_strides = input.stride()
    output_strides = output.stride()
    block_size = _metax_moe_sum_block_size(hidden_size)
    grid = (num_tokens, triton.cdiv(hidden_size, block_size))
    _metax_moe_sum_kernel[grid](
        input,
        output,
        num_tokens,
        hidden_size,
        input_strides[0],
        input_strides[1],
        input_strides[2],
        output_strides[0],
        output_strides[1],
        TOPK=topk,
        BLOCK_SIZE=block_size,
        num_warps=8 if block_size >= 1024 else 4,
    )
