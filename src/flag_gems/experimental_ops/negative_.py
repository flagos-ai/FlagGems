import torch
import triton
import triton.language as tl


@triton.jit
def negative_(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x = -x
    tl.store(x_ptr + offsets, x, mask=mask)


_negative__kernel = negative_


def negative_(*args, **kwargs):
    x = args[0] if len(args) > 0 else kwargs.get('input', kwargs.get('self', None))
    if x is None:
        raise ValueError("negative_ expects a tensor as the first argument")
    assert x.is_cuda, "Input tensor must be on CUDA device"
    assert x.is_contiguous(), "Input tensor must be contiguous"
    n_elements = x.numel()
    if n_elements == 0:
        return x
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _negative__kernel[grid](x, n_elements, BLOCK_SIZE=1024)
    return x