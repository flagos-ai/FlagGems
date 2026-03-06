import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _cp_gather_indexer_k_quant_cache_kernel(
    kv_cache_value_ptr,
    kv_cache_scale_ptr,
    dst_k_ptr,
    dst_scale_ptr,
    block_table_ptr,
    cu_seq_lens_ptr,
    token_to_seq_ptr,
    # strides
    kv_cache_value_stride,
    kv_cache_scale_stride,
    block_table_stride,
    # dims
    cache_block_size,
    # constexpr
    HEAD_DIM: tl.constexpr,
):
    """
    Gather quantized K cache data from paged KV cache.

    Args:
        kv_cache_value_ptr: [num_blocks, block_size * head_dim] - FP8 K values
        kv_cache_scale_ptr: [num_blocks, block_size] - float32 scales (one per token)
        dst_k_ptr: [num_tokens, head_dim] - destination for FP8 K data
        dst_scale_ptr: [num_tokens] - destination for scales
        block_table_ptr: [batch_size, num_blocks_per_seq] - block indices
        cu_seq_lens_ptr: [batch_size + 1] - cumulative sequence lengths
        token_to_seq_ptr: [num_tokens] - precomputed token to batch mapping
    """
    token_idx = tl.program_id(0)

    # 1. Look up batch index from precomputed mapping
    batch_idx = tl.load(token_to_seq_ptr + token_idx)

    # 2. Calculate cache position
    seq_start = tl.load(cu_seq_lens_ptr + batch_idx)
    inbatch_seq_idx = token_idx - seq_start
    block_table_idx = inbatch_seq_idx // cache_block_size
    block_offset = inbatch_seq_idx % cache_block_size

    # Load block index from block table
    block_idx = tl.load(
        block_table_ptr + batch_idx * block_table_stride + block_table_idx
    )

    # 3. Copy K data (vectorized)
    # Layout: [num_blocks, block_size * head_dim]
    src_cache_offset = block_idx * kv_cache_value_stride + block_offset * HEAD_DIM
    dst_offset = token_idx * HEAD_DIM

    offset = tl.arange(0, HEAD_DIM)
    val = tl.load(kv_cache_value_ptr + src_cache_offset + offset)
    tl.store(dst_k_ptr + dst_offset + offset, val)

    # 4. Copy scale data
    # Layout: [num_blocks, block_size]
    src_scale_offset = block_idx * kv_cache_scale_stride + block_offset
    scale_val = tl.load(kv_cache_scale_ptr + src_scale_offset)
    tl.store(dst_scale_ptr + token_idx, scale_val)


class CpGatherIndexerKQuantCache(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        kv_cache: torch.Tensor,
        dst_k: torch.Tensor,
        dst_scale: torch.Tensor,
        block_table: torch.Tensor,
        cu_seq_lens: torch.Tensor,
    ):
        """
        Gather quantized K cache data from paged KV cache.

        Args:
            kv_cache: [num_blocks, block_size, cache_stride] - source cache
                      cache_stride = head_dim + 4 (4 bytes for scale per token)
            dst_k: [num_tokens, head_dim] - output FP8 K data
            dst_scale: [num_tokens, ...] - output scales (viewed as float32)
            block_table: [batch_size, num_blocks_per_seq] - block indices
            cu_seq_lens: [batch_size + 1] - cumulative sequence lengths
        """
        num_tokens = dst_k.size(0)
        head_dim = dst_k.size(1)
        batch_size = block_table.size(0)
        num_blocks = kv_cache.size(0)
        cache_block_size = kv_cache.size(1)

        device = dst_k.device

        # Precompute token_to_seq mapping
        # This maps each token index to its batch index
        token_to_seq = torch.empty(num_tokens, dtype=torch.int32, device=device)
        cu_seq_lens_cpu = cu_seq_lens.cpu()
        for i in range(batch_size):
            start = int(cu_seq_lens_cpu[i].item())
            end = int(cu_seq_lens_cpu[i + 1].item())
            token_to_seq[start:end] = i

        # Split kv_cache into value and scale portions
        # kv_cache: [num_blocks, block_size, head_dim + 4]
        # Flatten to [num_blocks, block_size * (head_dim + 4)]
        kv_cache_flat = kv_cache.view(num_blocks, -1)

        # Value portion: first block_size * head_dim elements per block
        kv_cache_value = kv_cache_flat[:, : cache_block_size * head_dim]

        # Scale portion: remaining elements, viewed as float32
        # [num_blocks, block_size * 4] bytes -> [num_blocks, block_size] floats
        kv_cache_scale = kv_cache_flat[:, cache_block_size * head_dim :].view(
            torch.float32
        )

        # Get strides
        kv_cache_value_stride = kv_cache_value.stride(0)
        kv_cache_scale_stride = kv_cache_scale.stride(0)
        block_table_stride = block_table.stride(0)

        grid = (num_tokens,)

        # View dst_scale as float32
        dst_scale_f32 = dst_scale.view(-1).view(torch.float32)

        with torch_device_fn.device(device):
            _cp_gather_indexer_k_quant_cache_kernel[grid](
                kv_cache_value,
                kv_cache_scale,
                dst_k.view(-1),
                dst_scale_f32,
                block_table,
                cu_seq_lens,
                token_to_seq,
                # strides
                kv_cache_value_stride,
                kv_cache_scale_stride,
                block_table_stride,
                # dims
                cache_block_size,
                # constexpr
                HEAD_DIM=head_dim,
            )

        return None


def cp_gather_indexer_k_quant_cache(
    kv_cache: torch.Tensor,
    dst_k: torch.Tensor,
    dst_scale: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
) -> None:
    """
    Gather quantized K cache data from paged KV cache.

    Args:
        kv_cache: [num_blocks, block_size, cache_stride] - source cache
        dst_k: [num_tokens, head_dim] - output FP8 K data
        dst_scale: [num_tokens, ...] - output scales
        block_table: [batch_size, num_blocks_per_seq] - block indices
        cu_seq_lens: [batch_size + 1] - cumulative sequence lengths
    """
    logger.debug("GEMS CP_GATHER_INDEXER_K_QUANT_CACHE")
    return CpGatherIndexerKQuantCache.apply(
        kv_cache, dst_k, dst_scale, block_table, cu_seq_lens
    )
