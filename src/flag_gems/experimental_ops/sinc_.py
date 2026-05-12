import torch
import triton
import triton.language as tl


@triton.jit
def sinc_(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    x_f32 = x.to(tl.float32)

    pi = 3.141592653589793
    z = x_f32 * pi
    is_zero = x_f32 == 0.0
    denom = tl.where(is_zero, 1.0, z)
    s = tl.sin(z)
    y_f32 = s / denom
    y_f32 = tl.where(is_zero, 1.0, y_f32)

    y = y_f32.to(x.dtype)
    tl.store(x_ptr + offsets, y, mask=mask)


_sinc_kernel = sinc_


def sinc_(x: torch.Tensor):
    assert x.is_cuda, "Input tensor must be on CUDA device."
    assert x.is_contiguous(), "Input tensor must be contiguous."
    assert x.is_floating_point(), "sinc_ expects a floating point tensor."

    n_elements = x.numel()
    if n_elements == 0:
        return x

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _sinc_kernel[grid](x, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return x
