import torch
import triton
import triton.language as tl


@triton.jit
def hardtanh_(
    x_ptr,  # *Pointer* to input/output tensor (in-place).
    n_elements,  # Number of elements.
    min_val,  # Minimum clamp value (scalar).
    max_val,  # Maximum clamp value (scalar).
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0)

    # Cast min/max to tensor dtype
    min_v = tl.full([1], min_val, dtype=x.dtype)
    max_v = tl.full([1], max_val, dtype=x.dtype)

    x = tl.minimum(x, max_v)
    x = tl.maximum(x, min_v)

    tl.store(x_ptr + offsets, x, mask=mask)


# Keep a reference to the Triton kernel before defining the Python wrapper of the same name
hardtanh___kernel = hardtanh_


def hardtanh_(*args, **kwargs):
    # Parse arguments: expected signature hardtanh_(x, min_val=-1.0, max_val=1.0)
    if len(args) == 0 and "input" not in kwargs and "self" not in kwargs:
        raise TypeError("hardtanh_ expected at least 1 argument: a CUDA tensor")

    # Accept common naming: positional 0, or keyword 'input'/'self'
    x = None
    if len(args) >= 1:
        x = args[0]
    elif "input" in kwargs:
        x = kwargs["input"]
    elif "self" in kwargs:
        x = kwargs["self"]

    # Defaults
    min_val = -1.0
    max_val = 1.0

    # Override from positional args if provided
    if len(args) >= 2:
        min_val = args[1]
    if len(args) >= 3:
        max_val = args[2]

    # Override from kwargs if provided
    if "min_val" in kwargs and kwargs["min_val"] is not None:
        min_val = kwargs["min_val"]
    if "max_val" in kwargs and kwargs["max_val"] is not None:
        max_val = kwargs["max_val"]

    if not isinstance(x, torch.Tensor):
        raise TypeError("hardtanh_ expects a torch.Tensor as the first argument")

    # Fallback for unsupported device/dtypes
    if not x.is_cuda:
        # CPU fallback using PyTorch
        return torch.clamp_(x, min=min_val, max=max_val)

    if not x.is_floating_point():
        # For non-floating types, use PyTorch fallback to preserve semantics
        return torch.clamp_(x, min=min_val, max=max_val)

    # Require contiguous memory for in-place update
    if not x.is_contiguous():
        # To preserve in-place semantics on non-contiguous tensors, use PyTorch
        return torch.clamp_(x, min=min_val, max=max_val)

    n_elements = x.numel()
    if n_elements == 0:
        return x

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # Launch Triton kernel
    hardtanh___kernel[grid](
        x, n_elements, float(min_val), float(max_val), BLOCK_SIZE=BLOCK_SIZE  # in-place
    )
    return x
