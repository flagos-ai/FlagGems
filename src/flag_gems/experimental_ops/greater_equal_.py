import torch
import triton
import triton.language as tl


@triton.jit
def ge_inplace_kernel(
    x_ptr,  # *Pointer* to input/output tensor (in-place).
    y_ptr,  # *Pointer* to comparison tensor (same shape as x).
    n_elements,  # Number of elements.
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    out = x >= y
    tl.store(x_ptr + offsets, out, mask=mask)


def greater_equal__Scalar(*args, **kwargs):
    # Parse inputs
    if len(args) >= 2:
        x = args[0]
        other = args[1]
    else:
        x = kwargs.get("self", kwargs.get("input"))
        other = kwargs.get("other")
    assert isinstance(x, torch.Tensor), "self must be a torch.Tensor"
    assert x.is_cuda, "Input tensor must be on CUDA device"
    # Create a tensor filled with the scalar 'other' matching x's shape and dtype
    y = torch.full_like(x, other)
    # Ensure contiguity
    if not x.is_contiguous():
        raise RuntimeError("greater_equal__Scalar: input tensor must be contiguous")
    if not y.is_contiguous():
        y = y.contiguous()
    # Launch kernel
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    ge_inplace_kernel[grid](x, y, n_elements, BLOCK_SIZE=1024)
    return x


def greater_equal__Tensor(*args, **kwargs):
    # Parse inputs
    if len(args) >= 2:
        x = args[0]
        other = args[1]
    else:
        x = kwargs.get("self", kwargs.get("input"))
        other = kwargs.get("other")
    assert isinstance(x, torch.Tensor) and isinstance(
        other, torch.Tensor
    ), "Inputs must be torch.Tensors"
    assert x.is_cuda and other.is_cuda, "Both tensors must be on CUDA device"
    # Cast other to x's dtype to ensure the comparison works in kernel
    if other.dtype != x.dtype:
        other = other.to(dtype=x.dtype)
    # Broadcast other to x's shape and make contiguous for simple indexing
    if other.shape != x.shape:
        other = other.expand_as(x).contiguous()
    else:
        if not other.is_contiguous():
            other = other.contiguous()
    # Ensure x (in-place) is contiguous
    if not x.is_contiguous():
        raise RuntimeError("greater_equal__Tensor: input tensor must be contiguous")
    # Launch kernel
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    ge_inplace_kernel[grid](x, other, n_elements, BLOCK_SIZE=1024)
    return x
