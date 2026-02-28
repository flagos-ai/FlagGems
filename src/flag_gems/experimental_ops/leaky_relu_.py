import torch
import triton
import triton.language as tl


@triton.jit
def leaky_relu_(
    x_ptr,  # *Pointer* to input tensor data (modified in-place).
    n_elements,  # Number of elements to process.
    negative_slope,  # Scalar negative slope.
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.where(x >= 0, x, x * negative_slope)
    tl.store(x_ptr + offsets, y, mask=mask)


_leaky_relu_kernel = leaky_relu_


def leaky_relu_(*args, **kwargs):
    # Parse arguments: expect (input, negative_slope=0.01)
    if len(args) >= 1:
        x = args[0]
    else:
        x = kwargs.get("self", kwargs.get("input", None))
    if x is None:
        raise TypeError("leaky_relu_ expected a tensor as the first argument")

    negative_slope = 0.01
    if len(args) >= 2:
        negative_slope = args[1]
    else:
        negative_slope = kwargs.get("negative_slope", negative_slope)

    if isinstance(negative_slope, torch.Tensor):
        negative_slope = negative_slope.item()
    negative_slope = float(negative_slope)

    # Fallbacks for unsupported environments/dtypes
    if not isinstance(x, torch.Tensor):
        raise TypeError("leaky_relu_ expected a torch.Tensor")
    if not x.is_cuda or x.numel() == 0:
        return torch.ops.aten.leaky_relu_(x, negative_slope)

    # For dtypes not well supported by Triton math, fallback to PyTorch
    supported_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    if x.dtype not in supported_dtypes:
        return torch.ops.aten.leaky_relu_(x, negative_slope)

    # Ensure contiguous memory for in-place kernel; otherwise operate on a temp and copy back.
    if not x.is_contiguous():
        tmp = x.contiguous()
        n_elements = tmp.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _leaky_relu_kernel[grid](tmp, n_elements, negative_slope, BLOCK_SIZE=1024)
        x.copy_(tmp)
        return x

    # Launch Triton kernel in-place
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _leaky_relu_kernel[grid](x, n_elements, negative_slope, BLOCK_SIZE=1024)
    return x
