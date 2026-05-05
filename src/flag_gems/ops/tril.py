import logging

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    logging.warning("Triton is not installed. Fallback to PyTorch implementation.")


def tril(input: torch.Tensor, diagonal: int = 0) -> torch.Tensor:
    """
    Returns the lower triangular part of a matrix (2-D tensor) or batch of matrices.
    
    Args:
        input (torch.Tensor): Input tensor, must be at least 2-dimensional.
        diagonal (int, optional): Diagonal to keep. Defaults to 0, the main diagonal.
            - diagonal = 0: Keep the main diagonal and all elements below it
            - diagonal > 0: Additionally include diagonals above the main diagonal
            - diagonal < 0: Exclude the main diagonal and diagonals below it
    
    Returns:
        torch.Tensor: Resultant tensor with lower triangular part preserved, other elements are 0
    
    Raises:
        ValueError: If input tensor has fewer than 2 dimensions
        TypeError: If input is not a tensor
    
    Example:
        >>> a = torch.randn(3, 3)
        >>> torch.tril(a)  # diagonal=0, keep main diagonal and below
        >>> torch.tril(a, diagonal=1)  # additionally include one diagonal above main
        >>> torch.tril(a, diagonal=-1)  # exclude main diagonal itself
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"input must be a torch.Tensor, got {type(input).__name__}")
    
    if input.ndim < 2:
        raise ValueError(f"tril expects a tensor with at least 2 dimensions, got {input.ndim}")

    if not HAS_TRITON:
        return tril_pytorch_fallback(input, diagonal)

    output = torch.empty_like(input)
    n_rows, n_cols = input.shape[-2], input.shape[-1]
    total_elements = input.numel()

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(total_elements, meta["BLOCK_SIZE"]),)

    _tril_kernel[grid](
        input,
        output,
        n_rows,
        n_cols,
        diagonal,
        total_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


if HAS_TRITON:

    @triton.jit
    def _tril_kernel(
        input_ptr,
        output_ptr,
        n_rows,
        n_cols,
        diagonal,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        row_idx = offsets // n_cols
        col_idx = offsets % n_cols

        keep = col_idx <= row_idx + diagonal

        val = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        result = tl.where(keep, val, 0.0)

        tl.store(output_ptr + offsets, result, mask=mask)


def tril_pytorch_fallback(input: torch.Tensor, diagonal: int = 0) -> torch.Tensor:
    """
    PyTorch fallback implementation of tril, used when Triton is not available.
    
    Args:
        input (torch.Tensor): Input tensor
        diagonal (int): Diagonal parameter
    
    Returns:
        torch.Tensor: Lower triangular part of input tensor
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"input must be a torch.Tensor, got {type(input).__name__}")
    
    if input.ndim < 2:
        raise ValueError(f"tril expects a tensor with at least 2 dimensions, got {input.ndim}")

    n_rows, n_cols = input.shape[-2], input.shape[-1]
    batch_shape = input.shape[:-2]

    row_idx = torch.arange(n_rows, device=input.device).unsqueeze(1)
    col_idx = torch.arange(n_cols, device=input.device).unsqueeze(0)

    if len(batch_shape) > 0:
        row_idx = row_idx.unsqueeze(0).expand(*batch_shape, n_rows, n_cols)
        col_idx = col_idx.unsqueeze(0).expand(*batch_shape, n_rows, n_cols)

    keep_mask = col_idx <= row_idx + diagonal

    output = torch.where(keep_mask, input, torch.zeros_like(input))
    return output