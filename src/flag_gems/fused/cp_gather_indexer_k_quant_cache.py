# Adapted from vLLM v0.20.2:
# csrc/cache_kernels.cu::cp_gather_indexer_k_quant_cache_kernel

import torch
import triton
import triton.language as tl


@triton.jit
def _cp_gather_indexer_quant_cache_kernel(
    kv_cache_ptr,
    kv_cache_scale_ptr,
    k_fp8_ptr,
    k_scale_ptr,
    block_table_ptr,
    cu_seqlen_ptr,
    block_size,
    block_table_stride,
    kv_cache_stride,
    kv_cache_scale_stride,
    k_fp8_stride,
    num_quant_blocks,
    batch_size: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    BATCH_BLOCK: tl.constexpr,
):
    tid = tl.program_id(0)
    quant_block_id = tl.program_id(1)
    batch_offsets = tl.arange(0, BATCH_BLOCK)
    batch_mask = batch_offsets < batch_size
    seq_starts = tl.load(cu_seqlen_ptr + batch_offsets, mask=batch_mask, other=0)
    seq_ends = tl.load(cu_seqlen_ptr + batch_offsets + 1, mask=batch_mask, other=0)
    in_batch = (tid >= seq_starts) & (tid < seq_ends) & batch_mask
    batch_id = tl.max(tl.where(in_batch, batch_offsets, -1), axis=0)
    if batch_id < 0:
        return

    batch_start = tl.load(cu_seqlen_ptr + batch_id)
    batch_offset = tid - batch_start

    block_table_id = batch_offset // block_size
    block_offset = batch_offset % block_size
    block_table_offset = batch_id * block_table_stride + block_table_id
    block_id = tl.load(block_table_ptr + block_table_offset)

    offsets = quant_block_id * QUANT_BLOCK_SIZE + tl.arange(0, QUANT_BLOCK_SIZE)
    mask = offsets < HEAD_DIM
    src_cache_offset = block_id * kv_cache_stride + block_offset * HEAD_DIM
    src_scale_offset = (
        block_id * kv_cache_scale_stride
        + block_offset * num_quant_blocks
        + quant_block_id
    )
    dst_offset = tid * k_fp8_stride

    src_scale_ptr = kv_cache_scale_ptr + src_scale_offset
    src_cache_ptr = kv_cache_ptr + src_cache_offset
    dst_k_ptr = k_fp8_ptr + dst_offset

    scale_val = tl.load(src_scale_ptr)
    tl.store(k_scale_ptr + tid * num_quant_blocks + quant_block_id, scale_val)
    val = tl.load(src_cache_ptr + offsets, mask=mask)
    tl.store(dst_k_ptr + offsets, val, mask=mask)


def cp_gather_indexer_k_quant_cache(
    k_cache: torch.Tensor,
    k_fp8: torch.Tensor,
    k_fp8_scale: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlen: torch.Tensor,
):
    num_tokens = k_fp8.size(0)
    block_size = k_cache.size(1)
    block_table_stride = block_table.stride(0)
    head_dim = k_fp8.shape[-1]
    num_blocks = k_cache.shape[0]
    quant_block_size = head_dim * 4 // k_fp8_scale.size(1)
    if head_dim % quant_block_size != 0:
        raise ValueError("head_dim must be divisible by quant_block_size")
    num_quant_blocks = head_dim // quant_block_size

    k_cache_flat = k_cache.view(num_blocks, -1)
    k_cache_value = k_cache_flat[:, : block_size * head_dim]
    k_cache_scale = k_cache_flat[:, block_size * head_dim :].view(torch.float32)
    k_fp8 = k_fp8.view(torch.uint8)
    k_fp8_scale = k_fp8_scale.view(torch.float32)
    batch_size = block_table.shape[0]
    batch_block = triton.next_power_of_2(batch_size)

    _cp_gather_indexer_quant_cache_kernel[(num_tokens, num_quant_blocks)](
        k_cache_value,
        k_cache_scale,
        k_fp8,
        k_fp8_scale,
        block_table,
        cu_seqlen,
        block_size,
        block_table_stride,
        k_cache_value.stride(0),
        k_cache_scale.stride(0),
        k_fp8.stride(0),
        num_quant_blocks,
        batch_size,
        head_dim,
        quant_block_size,
        batch_block,
    )
