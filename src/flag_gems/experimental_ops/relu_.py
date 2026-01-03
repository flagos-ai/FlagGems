import torch
import triton
import triton.language as tl


@triton.jit
def relu_(x_ptr,  # *Pointer* to input/output tensor (in-place).
          n_elements,  # Number of elements.
          BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    zero = x * 0
    y = tl.where(x > 0, x, zero)
    tl.store(x_ptr + offsets, y, mask=mask)


# Keep a reference to the Triton kernel before defining the Python wrapper with the same name.
relu__kernel = relu_


def relu_(*args, **kwargs):
    # Expect the first positional argument to be the tensor.
    x = args[0] if len(args) > 0 else kwargs.get('input', kwargs.get('x', None))
    if x is None:
        raise ValueError("relu_ expects a tensor as the first positional argument.")
    if not x.is_cuda:
        raise ValueError("relu_ Triton implementation requires a CUDA tensor.")
    if not x.is_contiguous():
        raise ValueError("relu_ Triton implementation requires a contiguous tensor.")

    n_elements = x.numel()
    if n_elements == 0:
        return x

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    relu__kernel[grid](x, n_elements, BLOCK_SIZE=1024)
    return x