import torch
import triton
import triton.language as tl


@triton.jit
def absolute_(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x_abs = tl.abs(x)
    tl.store(x_ptr + offsets, x_abs, mask=mask)


# Keep a reference to the Triton kernel before redefining the wrapper with the same name
absolute__kernel = absolute_


def absolute_(*args, **kwargs):
    x = None
    if len(args) >= 1:
        x = args[0]
    else:
        x = kwargs.get("self", None)
        if x is None:
            x = kwargs.get("input", None)
    if x is None or not isinstance(x, torch.Tensor):
        raise TypeError("absolute_ expects a torch.Tensor as the first argument")

    # If tensor has no elements, nothing to do
    if x.numel() == 0:
        return x

    # Dtypes supported by this Triton kernel
    supported_dtypes = {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }

    use_triton = x.is_cuda and x.is_contiguous() and x.dtype in supported_dtypes

    if not use_triton:
        # Fallback to PyTorch implementation for unsupported cases (e.g., CPU, non-contiguous, unsupported dtype)
        torch.ops.aten.absolute_(x)
        return x

    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    absolute__kernel[grid](x, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return x
