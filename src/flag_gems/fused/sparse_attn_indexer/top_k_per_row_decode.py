import torch
import triton
import triton.language as tl


@triton.jit
def _mask_invalid_decode_kernel(
    logits_ptr,
    row_starts_ptr,
    row_ends_ptr,
    stride0,
    stride1,
    BLOCK_SIZE: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)

    start = tl.load(row_starts_ptr)
    end = tl.load(row_ends_ptr)

    offs = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (offs < VOCAB_SIZE) & ((offs < start) | (offs >= end))

    tl.store(logits_ptr + offs * stride1, float("-inf"), mask=mask)


@triton.jit
def _postprocess_decode_kernel(
    src_ptr,
    dst_ptr,
    row_starts_ptr,
    top_k: tl.constexpr,
    src_stride0: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_start = tl.load(row_starts_ptr)

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < top_k

    src_vals = tl.load(src_ptr + offs, mask=mask, other=0)
    dst_vals = (src_vals - row_start).to(tl.int32)

    tl.store(dst_ptr + offs, dst_vals, mask=mask)


def top_k_per_row_decode(
    logits,
    row_starts,
    row_ends,
    indices,
    num_rows,
    stride0,
    stride1,
    top_k,
):
    num_rows = int(num_rows)
    top_k = int(top_k)

    if num_rows != 1:
        raise ValueError("top_k_per_row_decode only supports num_rows == 1")

    if logits.dim() != 2:
        raise ValueError("logits must be a 2D tensor with shape [1, vocab_size]")

    if logits.shape[0] != 1:
        raise ValueError("logits.shape[0] must be 1 for decode")

    vocab_size = logits.shape[1]

    if top_k <= 0:
        return

    if top_k > vocab_size:
        raise ValueError("top_k must be <= vocab_size")

    mask_block_size = 8192
    num_mask_blocks = triton.cdiv(vocab_size, mask_block_size)

    _mask_invalid_decode_kernel[(num_mask_blocks,)](
        logits,
        row_starts,
        row_ends,
        stride0,
        stride1,
        BLOCK_SIZE=mask_block_size,
        VOCAB_SIZE=vocab_size,
        num_warps=2,
    )

    postprocess_block_size = triton.next_power_of_2(top_k)

    sorted_idx = torch.argsort(logits, dim=1, descending=True, stable=False)

    _postprocess_decode_kernel[(1,)](
        sorted_idx,
        indices,
        row_starts,
        top_k=top_k,
        src_stride0=vocab_size,
        BLOCK_SIZE=postprocess_block_size,
        num_warps=4,
    )
