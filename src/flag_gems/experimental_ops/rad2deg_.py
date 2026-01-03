import torch
import triton
import triton.language as tl


@triton.jit
def rad2deg__kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # Convert radians to degrees: deg = rad * (180/pi)
    out = x * (180.0 / 3.141592653589793)
    tl.store(x_ptr + offsets, out, mask=mask)


def rad2deg_(*args, **kwargs):
    # Accept first positional argument or common keyword names
    x = args[0] if len(args) > 0 else kwargs.get('input', kwargs.get('self', None))
    if x is None:
        raise ValueError("rad2deg_ expects a tensor as its first argument")
    if not isinstance(x, torch.Tensor):
        raise TypeError("rad2deg_ expects a torch.Tensor")
    if not x.is_floating_point():
        raise TypeError("rad2deg_ only supports floating point tensors for in-place operation")
    if not x.is_cuda:
        raise AssertionError("Input tensor must be on CUDA device")

    # If non-contiguous, operate on a contiguous copy and copy back in place
    original = x
    needs_copy_back = False
    if not x.is_contiguous():
        x = x.contiguous()
        needs_copy_back = True

    n_elements = x.numel()
    if n_elements == 0:
        return original

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    rad2deg__kernel[grid](x, n_elements, BLOCK_SIZE=1024)

    if needs_copy_back:
        original.copy_(x)
        return original
    return x