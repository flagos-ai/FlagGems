import logging

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    logging.warning("Triton is not installed. Fallback to PyTorch implementation.")


def median(input: torch.Tensor, dim: int = None, keepdim: bool = False):
    """
    Returns the median of values in the input tensor.

    Args:
        input (torch.Tensor): The input tensor.
        dim (int, optional): The dimension to reduce. If None, all dimensions are reduced.
            Default: None.
        keepdim (bool, optional): Whether the output tensor has dim retained or not.
            Default: False.

    Returns:
        torch.Tensor: If dim is None, returns the median of all values.
            Otherwise, returns a namedtuple (values, indices) where values contains the median
            and indices contains the index of the median values found in the dimension dim.

    Raises:
        TypeError: If input is not a torch.Tensor.
        ValueError: If dim is out of range or input has no elements.

    Example:
        >>> a = torch.randn(4, 4)
        >>> torch.median(a)
        tensor(0.1234)
        >>> torch.median(a, dim=1)
        torch.return_types.median(values=tensor([...]), indices=tensor([...]))
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"input must be a torch.Tensor, got {type(input).__name__}")

    if input.numel() == 0:
        raise ValueError("median cannot operate on an empty tensor")

    if dim is not None:
        if not isinstance(dim, int):
            raise TypeError(f"dim must be an int, got {type(dim).__name__}")
        if dim < -input.ndim or dim >= input.ndim:
            raise ValueError(f"dim out of range, got dim={dim} for tensor with {input.ndim} dimensions")

    if not HAS_TRITON:
        return median_pytorch_fallback(input, dim, keepdim)

    if dim is None:
        return median_global_triton_kernel(input)

    output_values, output_indices = median_dim_triton_kernel(input, dim)
    if keepdim:
        output_values = output_values.unsqueeze(dim)
        output_indices = output_indices.unsqueeze(dim)

    return torch.return_types.median((output_values, output_indices))


if HAS_TRITON:

    @triton.jit
    def _median_dim_kernel(
        input_ptr,
        output_values_ptr,
        output_indices_ptr,
        n_elements,
        stride,
        n_others,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_others

        n_check = n_elements // 2 if (n_elements % 2 == 1) else n_elements // 2 - 1

        local_min_val = 1e38
        local_min_idx = 0

        for i in range(n_elements):
            val = tl.load(input_ptr + offsets * stride + i, mask=mask, other=0.0)
            less_than = (val < local_min_val).to(tl.int32)
            local_min_val = tl.where(less_than, val, local_min_val)
            local_min_idx = tl.where(less_than, i, local_min_idx)

        median_val = local_min_val
        count_lt = 0
        for i in range(n_elements):
            val = tl.load(input_ptr + offsets * stride + i, mask=mask, other=0.0)
            count_lt += (val < median_val).to(tl.int32)

        count_eq = 0
        for i in range(n_elements):
            val = tl.load(input_ptr + offsets * stride + i, mask=mask, other=0.0)
            count_eq += (val == median_val).to(tl.int32)

        target_rank = n_check
        found_idx = local_min_idx
        current_rank = count_lt

        for i in range(n_elements):
            val = tl.load(input_ptr + offsets * stride + i, mask=mask, other=0.0)
            is_eq = val == median_val
            is_target = is_eq & (current_rank == target_rank)
            found_idx = tl.where(is_target, i, found_idx)
            current_rank = tl.where(is_eq, current_rank + 1, current_rank)

        tl.store(output_values_ptr + offsets, median_val, mask=mask)
        tl.store(output_indices_ptr + offsets, found_idx, mask=mask)


def median_dim_triton_kernel(input: torch.Tensor, dim: int):
    n_elements = input.shape[dim]
    n_others = input.numel() // n_elements

    output_values = torch.empty(input.shape[:dim] + input.shape[dim+1:], dtype=input.dtype, device=input.device)
    output_indices = torch.empty(input.shape[:dim] + input.shape[dim+1:], dtype=torch.long, device=input.device)

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_others, meta["BLOCK_SIZE"]),)

    _median_dim_kernel[grid](
        input,
        output_values,
        output_indices,
        n_elements,
        input.stride(dim),
        n_others,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output_values, output_indices


def median_global_triton_kernel(input: torch.Tensor) -> torch.Tensor:
    n_elements = input.numel()
    output = torch.empty((), dtype=input.dtype, device=input.device)

    BLOCK_SIZE = 1024
    grid = lambda meta: (1,)

    _median_global_kernel[grid](
        input,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


if HAS_TRITON:

    @triton.jit
    def _median_global_kernel(
        input_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        local_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)

        sorted_vals = tl.sort(local_vals)
        median_idx = n_elements // 2 if n_elements % 2 == 1 else n_elements // 2 - 1
        median_val = sorted_vals[median_idx]

        tl.store(output_ptr, median_val, mask=mask)


def median_pytorch_fallback(input: torch.Tensor, dim: int = None, keepdim: bool = False):
    """
    PyTorch fallback implementation of median, used when Triton is not available.
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"input must be a torch.Tensor, got {type(input).__name__}")

    if input.numel() == 0:
        raise ValueError("median cannot operate on an empty tensor")

    if dim is None:
        sorted_input = torch.sort(input.flatten())
        n = sorted_input.values.numel()
        median_idx = n // 2 if n % 2 == 1 else n // 2 - 1
        median_val = sorted_input.values[median_idx]
        return median_val

    values, indices = input.median(dim=dim, keepdim=keepdim)
    return torch.return_types.median((values, indices))