import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@triton.jit
def rwkv_mm_sparsity_kernel(
    k_ptr,
    v_ptr,
    output_ptr,
    v_cols: tl.constexpr,
    k_size: tl.constexpr,
    block_k: tl.constexpr,
    block_n: tl.constexpr,
):
    pid = tl.program_id(0)
    col_idx = pid * block_n + tl.arange(0, block_n)
    col_mask = col_idx < v_cols

    acc = tl.zeros((block_n,), dtype=tl.float32)

    for i in range(0, tl.cdiv(k_size, block_k)):
        k_offset = i * block_k + tl.arange(0, block_k)
        k_mask = k_offset < k_size
        k = tl.load(k_ptr + k_offset, mask=k_mask, other=0.0)
        k_nonzero_mask = k != 0

        v_ptr_block = v_ptr + k_offset[:, None] * v_cols + col_idx[None, :]
        v = tl.load(
            v_ptr_block,
            mask=k_mask[:, None] & col_mask[None, :] & k_nonzero_mask[:, None],
            other=0.0,
        )
        acc += tl.sum(k[:, None].to(tl.float32) * v.to(tl.float32), axis=0)

    tl.store(output_ptr + col_idx, acc, mask=col_mask)


def rwkv_mm_sparsity(k: torch.Tensor, v: torch.Tensor):
    logger.debug("GEMS_CAMBRICON RWKV MM SPARSITY")
    assert k.dim() == 1 and v.dim() == 2
    assert k.size(0) == v.size(0)

    v_cols = v.size(1)
    output = torch.empty(v_cols, device=k.device, dtype=k.dtype)

    block_k = 256
    block_n = triton.next_power_of_2(16)
    k_size = k.size(0)
    grid = (triton.cdiv(v_cols, block_n),)

    rwkv_mm_sparsity_kernel[grid](
        k,
        v,
        output,
        v_cols,
        k_size,
        block_k,
        block_n,
    )
    return output
