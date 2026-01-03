import torch
import triton
import triton.language as tl


@triton.jit
def abs_(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.abs(x)
    tl.store(x_ptr + offsets, y, mask=mask)


# Alias the kernel before defining the Python wrapper with the same name
abs__kernel = abs_


def abs_(*args, **kwargs):
    # Extract input tensor
    x = args[0] if len(args) > 0 else kwargs.get('input', None)
    if x is None:
        raise ValueError("abs_ expects a tensor as the first positional argument or 'input' keyword argument.")
    if not isinstance(x, torch.Tensor):
        raise TypeError("abs_ expects a torch.Tensor as input.")

    # Handle trivial/unsupported cases
    if x.numel() == 0:
        return x
    if x.dtype == torch.bool:
        # abs on boolean is identity; nothing to do
        return x
    if x.is_complex():
        raise TypeError("abs_ does not support complex tensors in-place.")

    # Ensure tensor is on CUDA and contiguous
    assert x.is_cuda, "abs_ expects a CUDA tensor."
    assert x.is_contiguous(), "abs_ expects a contiguous tensor."

    # Launch kernel
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    abs__kernel[grid](x, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return x