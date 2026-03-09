import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

autotune_configs = [
    triton.Config({"BLOCK_Q": 1, "BLOCK_K": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_Q": 1, "BLOCK_K": 256}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_Q": 1, "BLOCK_K": 512}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_Q": 1, "BLOCK_K": 1024}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_Q": 1, "BLOCK_K": 2048}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_Q": 2, "BLOCK_K": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_Q": 4, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_Q": 4, "BLOCK_K": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_Q": 4, "BLOCK_K": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_Q": 8, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_Q": 8, "BLOCK_K": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_Q": 16, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_Q": 32, "BLOCK_K": 128}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_Q": 32, "BLOCK_K": 256}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_Q": 32, "BLOCK_K": 512}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_Q": 64, "BLOCK_K": 128}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_Q": 64, "BLOCK_K": 256}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_Q": 64, "BLOCK_K": 512}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_Q": 64, "BLOCK_K": 1024}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_Q": 128, "BLOCK_K": 512}, num_warps=8, num_stages=4),
]


@libentry()
@triton.autotune(configs=autotune_configs, key=["query_seq_len", "key_seq_len"])
@triton.jit
def scaled_softmax_forward_kernel(
    output_ptr,
    input_ptr,
    scale_factor,
    query_seq_len,
    key_seq_len,
    stride_b,
    stride_h,
    stride_q,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    query_seq_tile_idx = tl.program_id(0)
    attn_head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)

    start_query_idx = query_seq_tile_idx * BLOCK_Q
    query_offsets = start_query_idx + tl.arange(0, BLOCK_Q)

    query_mask = query_offsets < query_seq_len

    row_start_ptr = (
        input_ptr
        + batch_idx * stride_b
        + attn_head_idx * stride_h
        + query_offsets * stride_q
    )

    m = tl.full([BLOCK_Q], -float("inf"), dtype=tl.float32)
    exp_sum = tl.zeros([BLOCK_Q], dtype=tl.float32)

    for k_block_idx in range(0, tl.cdiv(key_seq_len, BLOCK_K)):
        k_offsets = k_block_idx * BLOCK_K + tl.arange(0, BLOCK_K)
        block_ptr = row_start_ptr[:, None] + k_offsets[None, :]

        row_mask = query_mask[:, None]
        col_mask = k_offsets[None, :] < key_seq_len
        mask = row_mask & col_mask

        s_block = tl.load(
            block_ptr, mask=mask, other=-float("inf"), cache_modifier=".ca"
        )
        s_block = s_block * scale_factor

        m_new = tl.max(s_block, axis=1)
        m_old = m
        m = tl.maximum(m_old, m_new)

        s_prev = tl.exp(m_old - m)
        exp_sum = exp_sum * s_prev

        s_curr = tl.exp(s_block - m[:, None])
        l_new = tl.sum(tl.where(mask, s_curr, 0.0), axis=1)
        exp_sum = exp_sum + l_new

    exp_sum_inv = 1.0 / exp_sum

    out_row_start_ptr = (
        output_ptr
        + batch_idx * stride_b
        + attn_head_idx * stride_h
        + query_offsets * stride_q
    )

    for k_block_idx in range(0, tl.cdiv(key_seq_len, BLOCK_K)):
        k_offsets = k_block_idx * BLOCK_K + tl.arange(0, BLOCK_K)

        block_ptr_in = row_start_ptr[:, None] + k_offsets[None, :]
        block_ptr_out = out_row_start_ptr[:, None] + k_offsets[None, :]

        row_mask = query_mask[:, None]
        col_mask = k_offsets[None, :] < key_seq_len
        mask = row_mask & col_mask

        s_block = tl.load(
            block_ptr_in, mask=mask, other=-float("inf"), eviction_policy="evict_first"
        )

        s_block = s_block * scale_factor
        s_block = s_block - m[:, None]
        p_block = tl.exp(s_block)
        p_block = p_block * exp_sum_inv[:, None]

        tl.store(block_ptr_out, p_block, mask=mask, cache_modifier=".cs")


def scaled_softmax_forward(input_t: torch.Tensor, scale_factor: float):
    assert input_t.dim() == 4, "expected 4D tensor"
    batch_size, attn_heads, query_seq_len, key_seq_len = input_t.shape
    assert input_t.dtype in [
        torch.float16,
        torch.bfloat16,
    ], "Only fp16 and bf16 are supported"
    assert key_seq_len <= 16384, "Key sequence length must be 16384 or less"
    assert key_seq_len % 8 == 0, "Key sequence length must be divisible by 8"
    assert query_seq_len > 1, "Query sequence length must be greater than 1"

    def grid(meta):
        BLOCK_Q = meta["BLOCK_Q"]
        query_seq_tile_len = triton.cdiv(query_seq_len, BLOCK_Q)
        return (query_seq_tile_len, attn_heads, batch_size)

    output_t = torch.empty_like(input_t)

    stride_b = input_t.stride(0)
    stride_h = input_t.stride(1)
    stride_q = input_t.stride(2)

    scaled_softmax_forward_kernel[grid](
        output_t,
        input_t,
        scale_factor,
        query_seq_len,
        key_seq_len,
        stride_b,
        stride_h,
        stride_q,
    )
    return output_t


@libentry()
@triton.autotune(configs=autotune_configs, key=["query_seq_len", "key_seq_len"])
@triton.jit
def scaled_softmax_backward_kernel(
    grad_input_ptr,
    grad_output_ptr,
    output_ptr,
    scale_factor,
    query_seq_len,
    key_seq_len,
    stride_b,
    stride_h,
    stride_q,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    query_seq_tile_idx = tl.program_id(0)
    attn_head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)

    start_query_idx = query_seq_tile_idx * BLOCK_Q
    query_offsets = start_query_idx + tl.arange(0, BLOCK_Q)

    query_mask = query_offsets < query_seq_len

    output_row_ptr = (
        output_ptr
        + batch_idx * stride_b
        + attn_head_idx * stride_h
        + query_offsets * stride_q
    )

    grad_output_row_ptr = (
        grad_output_ptr
        + batch_idx * stride_b
        + attn_head_idx * stride_h
        + query_offsets * stride_q
    )

    grad_input_row_ptr = (
        grad_input_ptr
        + batch_idx * stride_b
        + attn_head_idx * stride_h
        + query_offsets * stride_q
    )

    D = tl.zeros([BLOCK_Q], dtype=tl.float32)

    for k_block_idx in range(0, tl.cdiv(key_seq_len, BLOCK_K)):
        k_offsets = k_block_idx * BLOCK_K + tl.arange(0, BLOCK_K)
        row_mask = query_mask[:, None]
        col_mask = k_offsets[None, :] < key_seq_len
        mask = row_mask & col_mask

        ptr_P = output_row_ptr[:, None] + k_offsets[None, :]
        ptr_dP = grad_output_row_ptr[:, None] + k_offsets[None, :]

        P_block = tl.load(ptr_P, mask=mask, other=0.0, cache_modifier=".ca")
        dP_block = tl.load(ptr_dP, mask=mask, other=0.0, cache_modifier=".ca")

        dot_block = P_block * dP_block
        D += tl.sum(tl.where(mask, dot_block, 0.0), axis=1)

    for k_block_idx in range(0, tl.cdiv(key_seq_len, BLOCK_K)):
        k_offsets = k_block_idx * BLOCK_K + tl.arange(0, BLOCK_K)
        row_mask = query_mask[:, None]
        col_mask = k_offsets[None, :] < key_seq_len
        mask = row_mask & col_mask

        ptr_P = output_row_ptr[:, None] + k_offsets[None, :]
        ptr_dP = grad_output_row_ptr[:, None] + k_offsets[None, :]
        ptr_dS = grad_input_row_ptr[:, None] + k_offsets[None, :]

        P_block = tl.load(ptr_P, mask=mask, other=0.0, eviction_policy="evict_first")
        dP_block = tl.load(ptr_dP, mask=mask, other=0.0, eviction_policy="evict_first")

        dZ_block = P_block * (dP_block - D[:, None])
        dS_block = scale_factor * dZ_block

        tl.store(ptr_dS, dS_block, mask=mask, cache_modifier=".cs")


def scaled_softmax_backward(
    grad_output: torch.Tensor, softmax_results: torch.Tensor, scale_factor: float
):
    assert grad_output.dim() == 4, "expected 4D tensor"
    assert softmax_results.dim() == 4, "expected 4D tensor"
    assert grad_output.dtype in [
        torch.float16,
        torch.bfloat16,
    ], "Only fp16 and bf16 are supported"
    assert softmax_results.dtype in [
        torch.float16,
        torch.bfloat16,
    ], "Only fp16 and bf16 are supported"

    grad_output = grad_output.contiguous()
    softmax_results = softmax_results.contiguous()

    batch_size, attn_heads, query_seq_len, key_seq_len = softmax_results.shape

    def grid(meta):
        BLOCK_Q = meta["BLOCK_Q"]
        query_seq_tile_len = triton.cdiv(query_seq_len, BLOCK_Q)
        return (query_seq_tile_len, attn_heads, batch_size)

    grad_input = torch.empty_like(grad_output)

    stride_b = softmax_results.stride(0)
    stride_h = softmax_results.stride(1)
    stride_q = softmax_results.stride(2)

    scaled_softmax_backward_kernel[grid](
        grad_input,
        grad_output,
        softmax_results,
        scale_factor,
        query_seq_len,
        key_seq_len,
        stride_b,
        stride_h,
        stride_q,
    )

    return grad_input
