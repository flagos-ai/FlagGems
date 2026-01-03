import torch
import triton
import triton.language as tl


@triton.jit
def celu_(
    x_ptr,  # Pointer to input tensor (will be modified in-place)
    n_elements,  # Number of elements in the tensor
    alpha,  # CELU alpha parameter (scalar)
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    x32 = tl.cast(x, tl.float32)
    alpha32 = tl.cast(alpha, tl.float32)

    neg_part = alpha32 * (tl.exp(x32 / alpha32) - 1.0)
    y32 = tl.where(x32 > 0, x32, neg_part)
    y = tl.cast(y32, x.dtype)

    tl.store(x_ptr + offsets, y, mask=mask)


# Preserve reference to the Triton kernel before defining the Python wrapper with the same name
celu_kernel = celu_


def celu_(x: torch.Tensor, alpha: float = 1.0):
    assert x.is_cuda, "Input tensor must be on CUDA device."
    assert x.is_floating_point(), "CELU requires a floating point tensor."
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    celu_kernel[grid](x, n_elements, alpha, BLOCK_SIZE=1024)
    return x
