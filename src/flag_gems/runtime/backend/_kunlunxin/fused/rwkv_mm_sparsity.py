import torch
import triton
import triton.language as tl


@triton.jit
def rwkv_mm_sparsity_kernel(
    k_ptr,
    v_ptr,
    output_ptr,
    v_cols: tl.constexpr,
    k_size: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    完全使用 2D 张量实现矩阵-向量乘法: output[1, N] = k[1, K] @ V[K, N]
    所有中间变量保持 2D 形状，无需 tl.sum 挤压维度。
    """
    pid_n = tl.program_id(axis=0)

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offs_n < v_cols
    accumulator = tl.zeros((1, BLOCK_SIZE_N), dtype=tl.float32)

    for k_block_idx in range(0, tl.cdiv(k_size, BLOCK_SIZE_K)):
        offs_k = k_block_idx * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offs_k < k_size
        k_ptrs = k_ptr + offs_k
        k_block_1d = tl.load(k_ptrs, mask=mask_k, other=0.0)
        k_block = k_block_1d[None, :]
        v_ptrs = v_ptr + (offs_k[:, None] * v_cols) + offs_n[None, :]
        v_mask = mask_k[:, None] & mask_n[None, :]
        v_block = tl.load(v_ptrs, mask=v_mask, other=0.0)

        # k_block: (1, BLOCK_SIZE_K), v_block: (BLOCK_SIZE_K, BLOCK_SIZE_N) -> accumulator: (1, BLOCK_SIZE_N)
        accumulator += tl.dot(k_block, v_block, allow_tf32=False)

    output_ptrs = output_ptr + offs_n
    output_1d = tl.view(accumulator, (BLOCK_SIZE_N,))
    tl.store(output_ptrs, output_1d, mask=mask_n)


def rwkv_mm_sparsity(k: torch.Tensor, v: torch.Tensor):
    assert k.dim() == 1 and v.dim() == 2
    assert k.size(0) == v.size(0)

    v_cols = v.size(1)
    output = torch.empty(v_cols, device=k.device, dtype=k.dtype)

    blk_size = triton.next_power_of_2(256)
    k_size = triton.next_power_of_2(k.size(0))
    block_size = triton.next_power_of_2(128) if 128 < k_size else k_size
    grid = (triton.cdiv(v_cols, block_size),)

    rwkv_mm_sparsity_kernel[grid](
        k,
        v,
        output,
        v_cols,
        k_size,
        blk_size,
        block_size,
    )
    return output
