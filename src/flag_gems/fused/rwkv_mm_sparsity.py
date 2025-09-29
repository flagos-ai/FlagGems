import torch
import triton
import triton.language as tl


@triton.jit
def rwkv_mm_sparsity_kernel(
    k_ptr,
    v_ptr,
    output_ptr,
    v_cols: tl.constexpr,
    blk_size: tl.constexpr,
    k_size: tl.constexpr,
    block_size: tl.constexpr,
):
    pid = tl.program_id(0)
    col_idx = pid * block_size + tl.arange(0, block_size)
    col_mask = col_idx < v_cols

    acc = tl.zeros((block_size,), dtype=tl.float32)

    for i in range(0, tl.cdiv(k_size, blk_size)):
        k_offset = i * blk_size + tl.arange(0, blk_size)
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

    out_ptr = output_ptr + col_idx
    tl.store(out_ptr, acc, mask=col_mask)


def rwkv_mm_sparsity(k: torch.Tensor, v: torch.Tensor):
    assert k.dim() == 1 and v.dim() == 2
    assert k.size(0) == v.size(0)

    v_cols = v.size(1)
    output = torch.empty(v_cols, device=k.device, dtype=k.dtype)

    blk_size = triton.next_power_of_2(512)
    block_size = triton.next_power_of_2(16)
    k_size = triton.next_power_of_2(k.size(0))
    grid = (triton.cdiv(v_cols, block_size),)

    rwkv_mm_sparsity_kernel[grid](
        k,
        v,
        output,
        v_cols,
        blk_size,
        k_size,
        block_size,
    )
    return output
