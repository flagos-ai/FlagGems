import torch
import triton
import triton.language as tl


@triton.jit
def exp_(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x_fp32 = x.to(tl.float32)
    y = tl.exp(x_fp32)
    y = y.to(x.dtype)
    tl.store(x_ptr + offsets, y, mask=mask)


# Preserve reference to the Triton kernel before defining the Python wrapper
# with the same name.
exp__kernel = exp_


def exp_(*args, **kwargs):
    # Extract the input tensor
    x = None
    if len(args) >= 1:
        x = args[0]
    elif "input" in kwargs:
        x = kwargs["input"]
    elif "self" in kwargs:
        x = kwargs["self"]
    else:
        raise ValueError(
            "exp_ expects a tensor as the first positional argument "
            "or 'input'/'self' keyword."
        )

    # Handle empty tensors quickly
    if x.numel() == 0:
        return x

    # Fallbacks for unsupported cases
    # - Non-CUDA tensors
    # - Non-floating or complex dtypes
    # - float64 (fp64) dtype
    # - Non-contiguous tensors
    if (
        (not x.is_cuda)
        or x.is_complex()
        or (not x.is_floating_point())
        or (x.dtype == torch.float64)
        or (not x.is_contiguous())
    ):
        # Use PyTorch's in-place operation as a safe fallback
        return torch.ops.aten.exp_(x)

    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # noqa: E731
    exp__kernel[grid](x, n_elements, BLOCK_SIZE=1024)
    return x
