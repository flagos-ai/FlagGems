import torch
import triton
import triton.language as tl


@triton.jit
def ge_inplace_scalar_kernel(x_ptr, n_elements, other, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    cmp = x >= other
    one = tl.full([BLOCK_SIZE], 1, x.dtype)
    zero = tl.full([BLOCK_SIZE], 0, x.dtype)
    out = tl.where(cmp, one, zero)
    tl.store(x_ptr + offsets, out, mask=mask)


@triton.jit
def ge_inplace_tensor_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    cmp = x >= y
    one = tl.full([BLOCK_SIZE], 1, x.dtype)
    zero = tl.full([BLOCK_SIZE], 0, x.dtype)
    out = tl.where(cmp, one, zero)
    tl.store(x_ptr + offsets, out, mask=mask)


def ge__Scalar(*args, **kwargs):
    x = args[0]
    other = args[1] if len(args) > 1 else kwargs.get("other", None)

    # If other is a tensor with a single element, treat it as a scalar
    if isinstance(other, torch.Tensor):
        if other.numel() == 1:
            other_val = other.to(dtype=x.dtype, device=x.device).item()
        else:
            return ge__Tensor(x, other)
    else:
        other_val = other

    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    ge_inplace_scalar_kernel[grid](x, n_elements, other_val, BLOCK_SIZE=1024)
    return x


def ge__Tensor(*args, **kwargs):
    x = args[0]
    y = args[1]

    # Handle scalar-like tensor by delegating to scalar implementation
    if isinstance(y, torch.Tensor) and y.numel() == 1:
        return ge__Scalar(x, y)

    # Match device and dtype, then expand and ensure contiguity for y
    y_prepped = y.to(device=x.device, dtype=x.dtype)
    if y_prepped.shape != x.shape:
        y_prepped = y_prepped.expand_as(x)
    y_prepped = y_prepped.contiguous()

    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    ge_inplace_tensor_kernel[grid](x, y_prepped, n_elements, BLOCK_SIZE=1024)
    return x
