import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import get_tuned_config
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.autotune(configs=get_tuned_config("index_fill"), key=["M", "N", "dim_stride"])
@triton.jit
def index_fill_kernel(
    output_ptr,
    index_ptr,
    value,
    M,
    N,
    dim_size,
    dim_stride,
    output_numel,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Index fill kernel supporting arbitrary rank tensors.

    Args:
        output_ptr: Output tensor pointer
        index_ptr: Index tensor pointer (1D)
        value: Scalar value to fill
        M: Number of indices
        N: Number of elements per index
        dim_size: Size of the dimension being indexed
        dim_stride: Stride of the dimension being indexed
        output_numel: Total number of elements in output
        BLOCK_M: Block size for M dimension
        BLOCK_N: Block size for N dimension
    """
    # 2D grid: M for indices, N for elements per index
    pid_m = tle.program_id(axis=0)
    pid_n = tle.program_id(axis=1)

    # Index dimension offsets
    offset_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offset_m < M

    # Load indices and handle negative indices
    idx = tl.load(index_ptr + offset_m, mask=mask_m, other=0)
    idx = tl.where(idx < 0, idx + dim_size, idx)

    # Elements dimension offsets
    offset_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offset_n < N

    # Broadcast to 2D
    idx_2d = idx[:, None]
    offset_n_2d = offset_n[None, :]

    # Calculate position: (outer * size + idx) * stride + inner
    outer_offset = offset_n_2d // dim_stride
    inner_offset = offset_n_2d % dim_stride
    positions = (outer_offset * dim_size + idx_2d) * dim_stride + inner_offset

    # Combined mask
    mask_2d = mask_m[:, None] & mask_n[None, :]
    mask_2d = mask_2d & (idx_2d >= 0) & (idx_2d < dim_size)
    mask_2d = mask_2d & (positions < output_numel)

    # Store the value
    tl.store(output_ptr + positions, value, mask=mask_2d)


def index_fill(input, dim, index, value):
    """
    Fill elements of input tensor along a dimension with a scalar value.

    This implementation supports arbitrary rank tensors (1D, 2D, 3D, ..., any dimension).

    Note: The implementation clones the input with contiguous memory layout for optimal
    performance, following the same pattern as the index_select operator.

    Args:
        input: Input tensor of any rank
        dim: Dimension along which to fill
        index: 1-D tensor containing indices to fill
        value: Scalar value (Python number or 0-dim tensor) to fill with

    Returns:
        New tensor with filled values

    Raises:
        ValueError: If value is not a scalar
        RuntimeError: If index is not 1-D
        IndexError: If dim is out of range

    Examples:
        >>> x = torch.randn(3, 5)
        >>> index_fill(x, 1, torch.tensor([1, 3]), 99.0)
    """
    logger.debug("GEMS INDEX FILL")

    # Validate inputs
    if index.ndim != 1:
        raise RuntimeError("index_fill(): Expected a 1-D tensor for index")

    # Convert negative dim to positive
    ndim = input.ndim
    original_dim = dim
    if dim < 0:
        dim = dim + ndim
    if dim < 0 or dim >= ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{-ndim}, {ndim-1}], but got {original_dim})"
        )

    # Early return for empty index
    if index.numel() == 0:
        return input.clone()

    # Ensure index is int64
    if index.dtype != torch.int64:
        index = index.to(torch.int64)

    # Convert value to scalar if it's a tensor
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            value = value.item()
        else:
            raise ValueError("Value tensor must be a scalar (0-dim tensor or number)")

    # Clone and ensure contiguous layout for optimal kernel performance
    # This follows the same pattern as index_select operator
    output = input.clone()
    if not output.is_contiguous():
        output = output.contiguous()

    # Pre-compute dim-related information
    dim_stride = output.stride()[dim]
    dim_size = output.shape[dim]
    output_numel = output.numel()

    # Calculate grid dimensions for 2D parallelization
    M = index.numel()  # Number of indices to process
    N = output.numel() // dim_size  # Elements per index

    # Define grid
    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

    # Launch kernel
    index_fill_kernel[grid](
        output,
        index,
        value,
        M,
        N,
        dim_size,
        dim_stride,
        output_numel,
    )

    return output
