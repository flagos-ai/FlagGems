import torch
import triton
import triton.language as tl


@triton.jit
def logical_not__kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    y = x == 0
    tl.store(x_ptr + offsets, y, mask=mask)


def logical_not_(*args, **kwargs):
    # Extract the input tensor following aten.logical_not_ schema: (Tensor self) -> Tensor
    x = None
    if len(args) >= 1:
        x = args[0]
    elif "self" in kwargs:
        x = kwargs["self"]
    elif "input" in kwargs:
        x = kwargs["input"]
    if x is None:
        raise ValueError("logical_not_ expects a tensor as the first argument.")

    if x.dtype is not torch.bool:
        raise TypeError("logical_not_ only supports bool tensors (in-place operation cannot change dtype).")

    # If not CUDA tensor, fall back to PyTorch implementation for correctness
    if not x.is_cuda:
        return torch.ops.aten.logical_not_(x)

    n_elements = x.numel()
    if n_elements == 0:
        return x

    # Work on a contiguous buffer; copy back if original is not contiguous
    needs_copy_back = not x.is_contiguous()
    buf = x if not needs_copy_back else x.contiguous()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    logical_not__kernel[grid](buf, n_elements, BLOCK_SIZE=1024)

    if needs_copy_back:
        x.copy_(buf)

    return x