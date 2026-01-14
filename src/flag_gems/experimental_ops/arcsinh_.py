import torch
import triton
import triton.language as tl


@triton.jit
def arcsinh_(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    x32 = x.to(tl.float32)
    x2 = x32 * x32
    tmp = tl.sqrt(x2 + 1.0)
    y32 = tl.log(x32 + tmp)
    y = y32.to(x.dtype)

    tl.store(x_ptr + offsets, y, mask=mask)


# Preserve reference to the kernel before defining the wrapper with the same name
arcsinh__kernel = arcsinh_


def arcsinh_(*args, **kwargs):
    if len(args) == 0:
        raise TypeError("arcsinh_ expected at least 1 argument (a Tensor)")
    x = args[0]
    if not isinstance(x, torch.Tensor):
        raise TypeError("arcsinh_ expected a torch.Tensor as the first argument")

    # Fallback for unsupported cases
    if (not x.is_cuda) or (not x.is_contiguous()) or (not x.dtype.is_floating_point):
        torch.ops.aten.arcsinh_(x)
        return x

    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    arcsinh__kernel[grid](x, n_elements, BLOCK_SIZE=1024)
    return x
