import torch
import triton
import triton.language as tl


@triton.jit
def hardswish_(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    three = 3.0
    six = 6.0
    zero = 0.0

    tmp = x + three
    tmp = tl.maximum(tmp, zero)
    tmp = tl.minimum(tmp, six)
    y = x * (tmp / six)

    tl.store(x_ptr + offsets, y, mask=mask)


# Preserve a reference to the Triton kernel before defining the Python wrapper with the same name.
hardswish__kernel = hardswish_


def hardswish_(*args, **kwargs):
    # Resolve input tensor from positional or keyword arguments
    if len(args) >= 1:
        x = args[0]
    else:
        x = kwargs.get("input", kwargs.get("self", None))

    if x is None:
        raise ValueError("hardswish_: expected a Tensor as the first argument")
    if not isinstance(x, torch.Tensor):
        raise TypeError("hardswish_: expected a Tensor")
    if not x.is_cuda:
        raise ValueError("hardswish_: expected a CUDA tensor")
    if not x.is_floating_point():
        raise TypeError("hardswish_: expected a floating point tensor")

    orig = x
    x_work = x if x.is_contiguous() else x.contiguous()

    n_elements = x_work.numel()
    if n_elements == 0:
        return orig

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    hardswish__kernel[grid](x_work, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    if x_work.data_ptr() != orig.data_ptr():
        orig.copy_(x_work)

    return orig
