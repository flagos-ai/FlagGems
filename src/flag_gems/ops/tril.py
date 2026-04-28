import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _tril_kernel(
    in_ptr,
    out_ptr,
    rows,
    cols,
    batch_stride,
    row_stride,
    col_stride,
    out_batch_stride,
    out_row_stride,
    out_col_stride,
    diagonal: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Kernel for tril: zero out elements above the diagonal."""
    pid_batch = tl.program_id(2)
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    row_offsets = pid_row * BLOCK_R + tl.arange(0, BLOCK_R)
    col_offsets = pid_col * BLOCK_C + tl.arange(0, BLOCK_C)

    row_mask = row_offsets < rows
    col_mask = col_offsets < cols

    # 2D mask
    r = row_offsets[:, None]
    c = col_offsets[None, :]
    mask_2d = row_mask[:, None] & col_mask[None, :]

    # tril condition: keep element if col <= row + diagonal
    tril_mask = c <= (r + diagonal)

    in_offsets = (
        pid_batch * batch_stride
        + r * row_stride
        + c * col_stride
    )
    out_offsets = (
        pid_batch * out_batch_stride
        + r * out_row_stride
        + c * out_col_stride
    )

    vals = tl.load(in_ptr + in_offsets, mask=mask_2d, other=0.0)
    vals = tl.where(tril_mask, vals, tl.zeros_like(vals))
    tl.store(out_ptr + out_offsets, vals, mask=mask_2d)


def tril(input: torch.Tensor, diagonal: int = 0) -> torch.Tensor:
    """Return lower triangular part of a matrix (or batch of matrices).

    Args:
        input: Input tensor of shape (..., M, N).
        diagonal: Diagonal offset. 0 = main diagonal, positive = above, negative = below.

    Returns:
        Tensor with same shape as input, upper triangle zeroed out.
    """
    logger.debug("GEMS TRIL")
    if input.ndim < 2:
        raise ValueError(f"tril requires at least 2D input, got {input.ndim}D")

    rows, cols = input.shape[-2], input.shape[-1]
    batch_size = input.numel() // (rows * cols)

    # Flatten batch dims
    x = input.contiguous().view(batch_size, rows, cols)
    out = torch.empty_like(x)

    BLOCK_R = 32
    BLOCK_C = 32
    grid = (
        triton.cdiv(rows, BLOCK_R),
        triton.cdiv(cols, BLOCK_C),
        batch_size,
    )

    with torch_device_fn.device(input.device):
        _tril_kernel[grid](
            x,
            out,
            rows,
            cols,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            diagonal=diagonal,
            BLOCK_R=BLOCK_R,
            BLOCK_C=BLOCK_C,
        )

    return out.view(input.shape)


def tril_(input: torch.Tensor, diagonal: int = 0) -> torch.Tensor:
    """In-place version of tril."""
    logger.debug("GEMS TRIL_")
    result = tril(input, diagonal)
    input.copy_(result)
    return input
