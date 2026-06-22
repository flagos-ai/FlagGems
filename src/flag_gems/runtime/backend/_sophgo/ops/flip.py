import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import tl_extra_shim

logger = logging.getLogger(__name__)


@triton.jit
def flip_kernel(
    in_ptr,
    out_ptr,
    flip_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Flip contiguous elements along the innermost dimension (block-wise).

    One program handles a single block of one row. The source block is read
    from the mirrored position so that the in-block flip matches the whole-row
    flip semantics: out[dst_start + r] = in[flip_dim - 1 - dst_start - r].
    """
    pid = tl.program_id(0)
    num_blocks: tl.constexpr = tl.cdiv(flip_dim, BLOCK_SIZE)
    row = pid // num_blocks
    blk = pid % num_blocks

    row_base = row * flip_dim
    dst_start = blk * BLOCK_SIZE

    r = tl.arange(0, BLOCK_SIZE)
    dst_offs = dst_start + r  # ascending write
    src_offs = (flip_dim - dst_start - BLOCK_SIZE) + r  # mirrored source block

    dst_mask = dst_offs < flip_dim
    src_mask = (src_offs >= 0) & (src_offs < flip_dim)

    x = tl.load(in_ptr + row_base + src_offs, mask=src_mask, other=0.0)
    x_rev = tl_extra_shim.flip(x, 0)
    tl.store(out_ptr + row_base + dst_offs, x_rev, mask=dst_mask)


@triton.jit
def flip_copy_kernel(
    in_ptr,
    out_ptr,
    flip_size: tl.constexpr,
    post_size: tl.constexpr,
    block_size: tl.constexpr,
):
    """Flip (flip_size, post_size) rows in reverse order.
    Grid = pre_size. Each program loops over flip_size rows and post_size blocks,
    loading block_size elements from row i, storing to row flip_size-1-i."""
    pid = tl.program_id(0)
    base = pid * flip_size * post_size
    r = tl.arange(0, block_size)

    for block_start in range(0, post_size, block_size):
        block_offs = block_start + r
        block_mask = block_offs < post_size
        for i in range(flip_size):
            src_offs = base + i * post_size + block_offs
            x = tl.load(in_ptr + src_offs, mask=block_mask, other=0.0)
            dst_offs = base + (flip_size - 1 - i) * post_size + block_offs
            tl.store(out_ptr + dst_offs, x, mask=block_mask)


def flip(A: torch.Tensor, dims) -> torch.Tensor:
    ndim = A.ndim
    norm_dims = [d if d >= 0 else ndim + d for d in dims]

    if A.numel() <= 1 or len(norm_dims) == 0:
        return A.clone()

    A_work = A
    for d in sorted(norm_dims):
        shape = A_work.shape
        flip_size = shape[d]
        if flip_size <= 1:
            continue
        cur_ndim = len(shape)

        if d == cur_ndim - 1:
            # Last dim: direct 2D flip_kernel
            pre_size = A_work.numel() // flip_size
            A_2d = A_work.reshape(pre_size, flip_size).contiguous()
            out_2d = torch.empty_like(A_2d)
            BLOCK_SIZE = 1024
            num_blocks = triton.cdiv(flip_size, BLOCK_SIZE)
            flip_kernel[(pre_size * num_blocks,)](
                A_2d,
                out_2d,
                flip_dim=flip_size,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            A_work = out_2d.reshape(shape)
        else:
            # Non-last dim: reshape to 3D, use flip_copy_kernel
            pre_size = 1
            for i in range(d):
                pre_size *= shape[i]
            post_size = 1
            for i in range(d + 1, cur_ndim):
                post_size *= shape[i]

            A_3d = A_work.reshape(pre_size, flip_size, post_size).contiguous()
            out_3d = torch.empty_like(A_3d)
            flip_copy_kernel[(pre_size,)](
                A_3d,
                out_3d,
                flip_size=flip_size,
                post_size=post_size,
                block_size=512,
            )
            A_work = out_3d.reshape(shape)

    return A_work.reshape(A.shape)
