import torch
import triton
import triton.language as tl


@triton.jit
def sin_(
    x_ptr,  # Pointer to input/output tensor (in-place).
    n_elements,  # Number of elements.
    BLOCK_SIZE: tl.constexpr,  # Elements processed per program.
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    x_fp32 = x.to(tl.float32)
    y_fp32 = tl.sin(x_fp32)
    y = y_fp32.to(x.dtype)
    tl.store(x_ptr + offsets, y, mask=mask)


# Keep a reference to the Triton kernel before defining the Python wrapper with the same name.
sin__kernel = sin_


def sin_(*args, **kwargs):
    # Extract the tensor argument similar to aten.sin_
    x = None
    if len(args) > 0:
        x = args[0]
    else:
        x = kwargs.get("input", kwargs.get("self", None))
    if x is None:
        raise ValueError("sin_ expects a tensor as the first argument")

    if not x.is_cuda:
        raise ValueError("Input tensor must be on CUDA device")
    if not x.is_contiguous():
        raise ValueError(
            "Input tensor must be contiguous for this Triton implementation"
        )

    # Fallback for unsupported dtypes
    if not x.is_floating_point() or x.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        # Use PyTorch fallback for unsupported dtypes (e.g., float64, complex)
        torch.sin_(x)
        return x

    n_elements = x.numel()
    if n_elements == 0:
        return x

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    sin__kernel[grid](x, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return x
