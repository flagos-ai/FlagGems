import torch
import triton
import triton.language as tl


@triton.jit
def zero_(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load to obtain dtype, then compute zeros of the same dtype
    vals = tl.load(x_ptr + offsets, mask=mask, other=0)
    zeros = vals - vals
    tl.store(x_ptr + offsets, zeros, mask=mask)


_zero_kernel = zero_


def zero_(*args, **kwargs):
    # Extract the tensor argument
    x = None
    if len(args) > 0:
        x = args[0]
    else:
        x = kwargs.get("self", kwargs.get("input", None))
    if x is None:
        raise ValueError("zero_ expects a Tensor as the first argument.")
    if not isinstance(x, torch.Tensor):
        raise TypeError("zero_ expects a torch.Tensor.")

    # Handle empty tensor quickly
    if x.numel() == 0:
        return x

    # Fallback for non-CUDA or non-contiguous tensors
    if (not x.is_cuda) or (not x.is_contiguous()):
        x.zero_()
        return x

    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _zero_kernel[grid](x, n_elements, BLOCK_SIZE=1024)
    return x
